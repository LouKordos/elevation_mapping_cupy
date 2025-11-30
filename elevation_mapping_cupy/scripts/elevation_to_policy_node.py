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
from scipy.interpolate import RegularGridInterpolator
import os       
import struct  

class ElevationToPolicyNode(Node):
    """
    Transforms GridMap data into a body-centric, policy-aligned grid.
    
    This node mimics the RL training observation logic:
    - Generates a grid around the robot base.
    - Interpolates the global elevation map at these grid points.
    - Computes relative height (Map Z - Base Z) to match the policy input.
    """
    def __init__(self):
        super().__init__("elevation_to_policy_node")
        
        self.declare_parameter("store_absolute_z", False)
        self.store_absolute_z = self.get_parameter("store_absolute_z").get_parameter_value().bool_value
        
        self.zmq_context = zmq.Context()        
        self.zmq_pub_raw = self.zmq_context.socket(zmq.PUB)
        self.zmq_pub_raw.bind("tcp://*:6970")
        self.zmq_pub_filtered = self.zmq_context.socket(zmq.PUB)
        self.zmq_pub_filtered.bind("tcp://*:6971")

        self.map_frame = "odom"
        self.robot_base_frame = "base"
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Restore print options for troubleshooting
        np.set_printoptions(floatmode="fixed", precision=4, linewidth=1000,suppress=True) # For consistent printouts

        # Grid definitions matching GridPatternCfg
        # Size: [1.0, 0.8], Res: 0.08 -> 13x11 points
        x_points = 13
        y_points = 11
        x_span = (x_points - 1) * 0.08  # 0.96m 
        y_span = (y_points - 1) * 0.08  # 0.80m
        
        # Grid centered at (0,0) in body frame
        x_coords = np.linspace(-x_span / 2.0, x_span / 2.0, x_points) 
        y_coords = np.linspace(-y_span / 2.0, y_span / 2.0, y_points)
        
        # indexing='xy' creates grid_x with shape (Rows=11, Cols=13)
        sensor_offset_x = 0.2 # From training config, sensor is 20cm to the front
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        grid_x += sensor_offset_x
        
        # Flatten for vector operations (Shape: 143, 2)
        self.query_points_body_frame = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
        
        self.create_subscription(GridMap, "/elevation_mapping_node/elevation_map_raw", self.raw_map_callback, 10)
        self.create_subscription(GridMap, "/elevation_mapping_node/elevation_map_filter", self.filtered_map_callback, 10)
        
        # Sentinel values
        # NOTE: 0.0 is risky for absolute fill value if the ground is actually at Z=0.0. 
        # Consider using -100.0 or np.nan if possible.
        self.fill_value_body_frame = -0.27
        self.fill_value_absolute = 0.0

        self.log_filename = "policy_data.bin"
        self.log_file = None
        try:
            self.log_file = open(self.log_filename, "ab")
            self.log_file_fd = self.log_file.fileno()
            # 'd' = float64 (timestamp), 'B' = uint8 (type), '143f' = 143x float32 (data)
            self.record_format = struct.Struct("d B 143f")
        except Exception as e:
            self.get_logger().error(f"Failed to open log file {self.log_filename}: {e}")

        mode_str = "ABSOLUTE" if self.store_absolute_z else "RELATIVE"
        self.get_logger().info(f"Node Initialized. Mode: {mode_str}")

    def raw_map_callback(self, msg: GridMap):
        self.process_and_publish(msg, "elevation", self.zmq_pub_raw)

    def filtered_map_callback(self, msg: GridMap):
        self.process_and_publish(msg, "min_filter", self.zmq_pub_filtered)

    def process_and_publish(self, msg: GridMap, layer_name: str, zmq_publisher: zmq.Socket):
        stamp = msg.header.stamp
        
        result = self.get_policy_heights(msg, layer_name, stamp)
        
        if result is None:
            self.get_logger().warn(f"Failed to compute policy heights for layer: {layer_name}", throttle_duration_sec=1.0)
            return

        heights_policy_grid = result.astype(np.float32)
        
        # Reshape to (11, 13) for visualization where Rows=Y (Left/Right) and Cols=X (Back/Front)
        print(heights_policy_grid.reshape(11, 13))
        
        payload = heights_policy_grid.tobytes()
        if len(payload) != 143 * 4: 
            self.get_logger().error(f"Payload size mismatch! Got {len(payload)}, expected {143*4}")
            return
        zmq_publisher.send(payload)

        if self.log_file:
            try:
                timestamp = stamp.sec + stamp.nanosec * 1e-9
                data_type = 0 if zmq_publisher == self.zmq_pub_raw else 1
                packed_data = self.record_format.pack(timestamp, data_type, *heights_policy_grid)
                self.log_file.write(packed_data)
            except Exception as e:
                self.get_logger().error(f"Failed to write to log file: {e}")

    def get_policy_heights(self, msg: GridMap, layer_name: str, stamp):
        try:
            # Robot Pose Lookup (Base -> Odom)
            # Training uses 'attach_yaw_only=True', so we must manually construct a Yaw-only rotation matrix.
            tf_base_to_map = self.tf_buffer.lookup_transform(
                target_frame=self.map_frame, 
                source_frame=self.robot_base_frame, 
                time=rclpy.time.Time(), # Use latest for control responsiveness
                timeout=Duration(seconds=0.1)
            )
            trans = tf_base_to_map.transform.translation
            rot = tf_base_to_map.transform.rotation
            
            t_base_map_2D = np.array([trans.x, trans.y])
            robot_z = trans.z
            
            # Extract Yaw
            R_quat = ScipyRotation.from_quat([rot.x, rot.y, rot.z, rot.w])
            yaw = R_quat.as_euler("zxy")[0] 
            
            # 2D Rotation (Yaw only)
            c, s = np.cos(yaw), np.sin(yaw)
            R_yaw_2D = np.array([[c, -s], [s, c]])

        except TransformException as ex:
            self.get_logger().warn(f"TF Lookup Failed between {self.map_frame} and {self.robot_base_frame}: {ex}", throttle_duration_sec=1.0)
            return None

        if layer_name not in msg.layers:
            self.get_logger().warn(f"Layer '{layer_name}' missing from GridMap. Available: {msg.layers}", throttle_duration_sec=2.0)
            return None

        idx = msg.layers.index(layer_name)
        
        # GridMap Geometry extraction
        res = msg.info.resolution
        len_x = msg.info.length_x
        len_y = msg.info.length_y
        pos_x = msg.info.pose.position.x
        pos_y = msg.info.pose.position.y
        
        n_cells_x = int(round(len_x / res))
        n_cells_y = int(round(len_y / res))
        
        # Upstream elevation_mapping_node publishes .T.flatten()
        # We must reshape (H, W) then Transpose to get correct (x, y) alignment
        data_flat = np.array(msg.data[idx].data, dtype=np.float32)
        try:
            map_data = data_flat.reshape(n_cells_y, n_cells_x).T
        except ValueError as e:
            self.get_logger().error(f"GridMap Reshape Error. Expected ({n_cells_x}x{n_cells_y}), got flat size {data_flat.size}. Error: {e}")
            return None

        # Check for empty or invalid map data (Safety)
        if not np.any(np.isfinite(map_data)):
             self.get_logger().warn(f"Map layer '{layer_name}' contains no finite values!", throttle_duration_sec=2.0)

        # Coordinate Setup for Interpolation
        x_vec = np.linspace(pos_x + len_x/2.0 - res/2.0, pos_x - len_x/2.0 + res/2.0, n_cells_x)
        y_vec = np.linspace(pos_y + len_y/2.0 - res/2.0, pos_y - len_y/2.0 + res/2.0, n_cells_y)
        
        # RegularGridInterpolator requires strictly increasing coordinates
        if x_vec[0] > x_vec[-1]:
            x_vec = np.flip(x_vec)
            map_data = np.flip(map_data, axis=0)
            
        if y_vec[0] > y_vec[-1]:
            y_vec = np.flip(y_vec)
            map_data = np.flip(map_data, axis=1)

        # Transform Query Points
        # P_map = R_yaw * P_body_relative + T_base
        p_rotated = self.query_points_body_frame @ R_yaw_2D.T 
        query_points_odom = p_rotated + t_base_map_2D

        # Select Fill Value based on mode
        fill_value = self.fill_value_absolute if self.store_absolute_z else self.fill_value_body_frame

        # Interpolate
        # bounds_error=False allows extrapolation or use of fill_value for out-of-bounds points
        interpolator = RegularGridInterpolator((x_vec, y_vec), map_data, bounds_error=False, fill_value=fill_value)
        interpolated_z_abs = interpolator(query_points_odom)

        # Safety: Check for NaNs in output which can crash policy/controller
        if np.isnan(interpolated_z_abs).any():
            self.get_logger().error(f"NaNs detected in interpolated output! Replacing with fill_value={fill_value} to ensure safety.")
            interpolated_z_abs = np.nan_to_num(interpolated_z_abs, nan=fill_value)

        # Format Output
        if self.store_absolute_z:
            return interpolated_z_abs
        else:
            # Training Logic: Height = Hit_Z_World - Base_Z_World
            # Mask out invalid (filled) points so we don't subtract robot_z from them
            # We use isclose because float equality checks can be flaky
            valid_mask = ~np.isclose(interpolated_z_abs, fill_value)
            
            result = np.full_like(interpolated_z_abs, fill_value)
            # Subtract robot Z from valid map points
            result[valid_mask] = interpolated_z_abs[valid_mask] - robot_z
            return result

    def destroy_node(self):
        if self.log_file:
            self.log_file.close()
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

if __name__ == "__main__":
    main()