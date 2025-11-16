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
    """
    def __init__(self):
        super().__init__('elevation_to_policy_node')
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
        np.set_printoptions(floatmode="fixed")
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
        result = self.get_points_in_body_frame(msg, layer_name, stamp)
        if result is None:
            self.get_logger().warn(f"Could not get points for layer '{layer_name}'", throttle_duration_sec=0.5)
            return
        xy_body, z_body = result
        
        if xy_body.shape[0] == 0:
            heights_policy_grid = np.full(self.query_points_body.shape[0], self.fill_value, dtype=np.float32)
        else:
            # Interpolate policy grid from body-frame point cloud
            heights_policy_grid = griddata(
                xy_body,                # (N, 2) array of (x,y) points in base frame
                z_body,                 # (N,) array of (z) values in base frame
                self.query_points_body, # (143, 2) array of policy (x,y) query points
                method=self.interpolation_method,
                fill_value=self.fill_value
            ).astype(np.float32)
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

    def get_points_in_body_frame(self, msg: GridMap, layer_name: str, stamp):
        """
        Transforms all valid points from a GridMap layer into the robot's body frame.
        """
        try:
            # 1. Get the transform to move points from ODOM -> BASE
            # This is T_base_odom. Its translation will be ~-0.27m
            tf_map_to_base = self.tf_buffer.lookup_transform(
                target_frame=self.robot_base_frame, 
                source_frame=self.map_frame, 
                time=stamp, 
                timeout=Duration(seconds=0.1)
            )
            trans_map_to_base = tf_map_to_base.transform.translation
            t_map_to_base = np.array([trans_map_to_base.x, trans_map_to_base.y, trans_map_to_base.z])
            rot_map_to_base = tf_map_to_base.transform.rotation
            R_map_to_base = ScipyRotation.from_quat([rot_map_to_base.x, rot_map_to_base.y, rot_map_to_base.z, rot_map_to_base.w]).as_matrix()

            # This is T_odom_base. Its translation will be ~+0.27m
            tf_base_to_map = self.tf_buffer.lookup_transform(
                target_frame=self.map_frame, 
                source_frame=self.robot_base_frame, 
                time=stamp, 
                timeout=Duration(seconds=0.1)
            )
            center_z = tf_base_to_map.transform.translation.z # This is the +0.27m value

        except TransformException as ex:
            self.get_logger().warn(f'Could not look up transforms: {ex}', throttle_duration_sec=2)
            return None

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

        Z_map_relative = np.array(msg.data[msg.layers.index(layer_name)].data, dtype=np.float32).reshape(cols, rows).T
        
        # z_map = z_rel + center_z
        Z_map_absolute = Z_map_relative + center_z
        finite_mask_grid = np.isfinite(Z_map_absolute)
        points_map_frame = np.stack([X_map.ravel(), Y_map.ravel(), Z_map_absolute.ravel()], axis=1)
        
        valid_indices = finite_mask_grid.ravel()
        points_map_frame_valid = points_map_frame[valid_indices]
        if points_map_frame_valid.shape[0] == 0:
            return (np.array([]), np.array([]))

        # p_base = R_base_odom * p_odom + t_base_odom
        points_body_frame_valid = (R_map_to_base @ points_map_frame_valid.T).T + t_map_to_base
        finite_mask_after_tf = np.isfinite(points_body_frame_valid).all(axis=1)
        points_body_frame_finite = points_body_frame_valid[finite_mask_after_tf]
        
        if points_body_frame_finite.shape[0] == 0:
             self.get_logger().warn("All valid points became non-finite after transform.", throttle_duration_sec=1)
             return (np.array([]), np.array([]))

        xy_body = points_body_frame_finite[:, :2] # (N, 2)
        z_body = points_body_frame_finite[:, 2] # (N,)
        # print(t_map_to_base[2])
        # print(np.mean(z_body))
        return (xy_body, z_body)

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