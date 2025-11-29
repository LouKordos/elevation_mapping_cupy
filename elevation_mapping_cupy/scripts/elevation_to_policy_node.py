#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from grid_map_msgs.msg import GridMap
import numpy as np
import zmq
import tf2_ros
from tf2_ros import TransformException
from scipy.spatial.transform import Rotation as ScipyRotation
from scipy.interpolate import griddata
import os       
import struct  

class ElevationToPolicyNode(Node):
    """
    Transforms GridMap data into a body-centric, policy-aligned grid
    and publishes it over ZMQ and logs it to a robust binary file.

    This node replicates the *exact* logic from the training environment:
    1.  It interpolates the 'elevation' layer, which (due to the elevation_mapping
        code) stores z_relative = z_map - z_base.
    2.  It queries this data at 143 points defined in the robot's 'base' frame.
    3.  To find where these query points are in the 'map' frame, it uses a
        2D (yaw-only) transform, matching the 'attach_yaw_only=True'
        setting in the training's ray-caster.
    """
    def __init__(self):
        super().__init__('elevation_to_policy_node')
        self.declare_parameter("store_absolute_z", False) # Coming directly from elevation map layer
        self.store_absolute_z = self.get_parameter("store_absolute_z").get_parameter_value().bool_value
        self.get_logger().info("Initializing ZMQ publishers...")
        self.zmq_context = zmq.Context()        
        self.zmq_pub_raw = self.zmq_context.socket(zmq.PUB)
        self.zmq_pub_raw.bind("tcp://*:6970")
        self.get_logger().info("ZMQ PUB socket bound to tcp://*:6970 for RAW policy heights")
        self.zmq_pub_filtered = self.zmq_context.socket(zmq.PUB)
        self.zmq_pub_filtered.bind("tcp://*:6971")
        self.get_logger().info("ZMQ PUB socket bound to tcp://*:6971 for FILTERED policy heights")

        self.map_frame = "odom"
        self.robot_base_frame = "base"
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        np.set_printoptions(floatmode="fixed")

        # 13x11 grid with 8cm resolution (spans 0.96m x 0.8m)
        x_points = 13
        y_points = 11
        x_span = (x_points - 1) * 0.08  # 0.96m
        y_span = (y_points - 1) * 0.08  # 0.80m
        self.x_coords_policy = np.linspace(-x_span / 2.0, x_span / 2.0, x_points) # -0.48 to +0.48
        self.y_coords_policy = np.linspace(-y_span / 2.0, y_span / 2.0, y_points) # -0.40 to +0.40
        
        grid_x_policy, grid_y_policy = np.meshgrid(self.x_coords_policy, self.y_coords_policy)
        
        # Flatten for griddata query (shape: 143, 2)
        # training code uses "xy" ordering, which is row-major.
        self.query_points_body = np.vstack([grid_x_policy.ravel(), grid_y_policy.ravel()]).T
        
        self.create_subscription(GridMap, "/elevation_mapping_node/elevation_map_raw", self.raw_map_callback, 10)
        self.create_subscription(GridMap, "/elevation_mapping_node/elevation_map_filter", self.filtered_map_callback, 10)
        
        self.map_data_cache = {}
        self.interpolation_method = "nearest"
        self.fill_value = -0.33 # Default height for unseen points

        self.log_filename = "policy_data.bin"
        try:
            self.log_file = open(self.log_filename, 'ab') # Append-binary mode
            self.log_file_fd = self.log_file.fileno()
            # 'd' = float64 (timestamp), 'B' = uint8 (type), '143f' = 143x float32 (data)
            self.record_format = struct.Struct('d B 143f')
            self.get_logger().info(f"Logging policy data to {self.log_filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to open log file {self.log_filename}: {e}")
            self.log_file = None

        self.get_logger().info("Elevation to Policy Node initialized.")

    def raw_map_callback(self, msg: GridMap):
        # Use the 'elevation' layer for raw data
        self.process_and_publish(msg, 'elevation', self.zmq_pub_raw)

    def filtered_map_callback(self, msg: GridMap):
        # Use the 'min_filter' layer for filtered data
        self.process_and_publish(msg, 'min_filter', self.zmq_pub_filtered)

    def process_and_publish(self, msg: GridMap, layer_name: str, zmq_publisher: zmq.Socket):
        stamp = msg.header.stamp
        
        # Get all valid map points transformed into the body frame
        result = self.get_policy_heights(msg, layer_name, stamp)
        if result is None:
            self.get_logger().warn(f"Could not get points for layer '{layer_name}'", throttle_duration_sec=0.5)
            return

        heights_policy_grid = result
        print(heights_policy_grid)
        
        payload = heights_policy_grid.tobytes()
        if len(payload) != 143 * 4: # 143 floats * 4 bytes/float
            self.get_logger().error(f"Payload size is {len(payload)} bytes, expected {143*4}!")
            return
        zmq_publisher.send(payload)

        if self.log_file:
            try:
                # Convert timestamp to float
                timestamp = stamp.sec + stamp.nanosec * 1e-9
                # 0 for raw, 1 for filtered
                data_type = 0 if zmq_publisher == self.zmq_pub_raw else 1
                packed_data = self.record_format.pack(timestamp, data_type, *heights_policy_grid)
                self.log_file.write(packed_data)
                self.log_file.flush()
                os.fsync(self.log_file_fd)
            except Exception as e:
                self.get_logger().error(f"Failed to write to log file: {e}")

    def get_policy_heights(self, msg: GridMap, layer_name: str, stamp):
        """
        Interpolates the elevation map to get the 143 policy-specific
        height points, matching the training logic.
        """
        try:
            # Get the robot's 3D pose in the map frame (T_odom_base)
            tf_base_to_map = self.tf_buffer.lookup_transform(
                target_frame=self.map_frame, 
                source_frame=self.robot_base_frame, 
                time=stamp, 
                timeout=Duration(seconds=0.1)
            )
            trans_base_to_map = tf_base_to_map.transform.translation
            rot_base_to_map = tf_base_to_map.transform.rotation
            
            t_base_to_map_2D = np.array([trans_base_to_map.x, trans_base_to_map.y])
            R_base_to_map_3D = ScipyRotation.from_quat([rot_base_to_map.x, rot_base_to_map.y, rot_base_to_map.z, rot_base_to_map.w]).as_matrix()
            yaw = np.arctan2(R_base_to_map_3D[1, 0], R_base_to_map_3D[0, 0]) # More robust than euler angles
            R_yaw_2D = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw),  np.cos(yaw)]
            ])

        except TransformException as ex:
            self.get_logger().warn(f'Could not look up transform: {ex}', throttle_duration_sec=2)
            return None

        # --- Get Grid Data to Interpolate From ---
        msg_id = (msg.info.pose.position.x, msg.info.length_x, msg.info.resolution, msg.info.length_y)
        if msg_id not in self.map_data_cache:
            rows = int(round(msg.info.length_x / msg.info.resolution))
            cols = int(round(msg.info.length_y / msg.info.resolution))
            map_pos_x, map_pos_y = msg.info.pose.position.x, msg.info.pose.position.y
            map_len_x, map_len_y = msg.info.length_x, msg.info.length_y
            
            x_coords = np.linspace(map_pos_x - map_len_x / 2.0, map_pos_x + map_len_x / 2.0, rows)
            y_coords = np.linspace(map_pos_y - map_len_y / 2.0, map_pos_y + map_len_y / 2.0, cols)
            Y_map, X_map = np.meshgrid(y_coords, x_coords)
            self.map_data_cache[msg_id] = (X_map, Y_map, rows, cols)
        
        X_map, Y_map, rows, cols = self.map_data_cache[msg_id]

        if layer_name not in msg.layers:
            self.get_logger().warn(f"Layer '{layer_name}' not in map. Available: {msg.layers}", throttle_duration_sec=1)
            return None

        # This is the "value" field for interpolation. It already contains z_relative = z_map - z_base (e.g., -0.27m)
        Z_map_relative = np.array(msg.data[msg.layers.index(layer_name)].data, dtype=np.float32).reshape(cols, rows).T
        points_map_frame = np.stack([X_map.ravel(), Y_map.ravel()], axis=1) # "coordinate" field for interpolation.
        values_relative_z = Z_map_relative.ravel()

        # Filter out NaNs from the interpolation data
        finite_mask = np.isfinite(values_relative_z)
        if not finite_mask.any():
            self.get_logger().warn("No finite map data to interpolate.", throttle_duration_sec=1)
            return np.full(self.query_points_body.shape[0], self.fill_value, dtype=np.float32)

        points_map_frame = points_map_frame[finite_mask]
        values_relative_z = values_relative_z[finite_mask]

        # `self.query_points_body` is (143, 2) in 'base' frame
        # Transform them to 'odom' (yaw-only)
        # p_odom = R_yaw * p_base + t_odom
        p_base_2D_T = self.query_points_body.T # Shape (2, 143)
        p_odom_2D_T = (R_yaw_2D @ p_base_2D_T)
        p_odom_2D_query = p_odom_2D_T.T + t_base_to_map_2D # Shape (143, 2)
        
        heights_policy_grid = griddata(
            points_map_frame,       # (N, 2) odom coordinates
            values_relative_z,      # (N,)   relative_z values
            p_odom_2D_query,        # (143, 2) odom query coordinates
            method=self.interpolation_method,
            fill_value=self.fill_value
        ).astype(np.float32)

        return heights_policy_grid

    def destroy_node(self):
        if self.log_file:
            self.get_logger().info("Closing log file.")
            self.log_file.close()
        
        self.get_logger().info("Shutting down ZMQ sockets.")
        self.zmq_pub_raw.close()
        self.zmq_pub_filtered.close()
        self.zmq_context.term()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ElevationToPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()