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
from datetime import datetime

class ElevationToPolicyNode(Node):
    """
    Transforms GridMap data into a body-centric, policy-aligned grid.
    Writes continuous data streams to timestamped binary files (one per layer).
    
    BINARY FORMAT SPECIFICATION (Little Endian '<'):
    
    [FILE HEADER] - Written ONCE at start of file (Size: 80 Bytes)
    | Offset | Type     | Name           | Description                          |
    |--------|----------|----------------|--------------------------------------|
    | 0      | uint8    | version        | Current: 1                           |
    | 1      | float32  | resolution     | Grid resolution (e.g., 0.08)         |
    | 5      | float32  | sensor_off_x   | Sensor offset (e.g., 0.2)            |
    | 9      | uint16   | num_x          | Grid Width                           |
    | 11     | uint16   | num_y          | Grid Height                          |
    | 13     | 67 bytes | reserved_file  | Padding for future static configs    |
    
    [FRAME RECORD] - Written REPEATEDLY for every timestep (Size: Variable)
    | Offset | Type     | Name           | Description                          |
    |--------|----------|----------------|--------------------------------------|
    | 0      | double   | timestamp      | Unix time (sec.nanosec)              |
    | 8      | uint8    | layer_id       | 0=elev, 1=min, 2=smooth              |
    | 9      | float32  | valid_ratio    | 0.0-1.0 (integrity check)            |
    | 13     | 7x float | robot_pose     | x, y, z, qx, qy, qz, qw              |
    | 41     | 32 bytes | reserved_frame | Padding for future frame data        |
    | 73     | N x flt  | grid_data      | Flattened grid (N = num_x * num_y)   |
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

        np.set_printoptions(floatmode="fixed", precision=4, linewidth=1000, suppress=True)

        # --- Grid Config ---
        self.x_points = 13
        self.y_points = 11
        self.resolution = 0.08
        self.sensor_offset_x = 0.2
        
        x_span = (self.x_points - 1) * self.resolution  
        y_span = (self.y_points - 1) * self.resolution
        x_coords = np.linspace(-x_span / 2.0, x_span / 2.0, self.x_points) 
        y_coords = np.linspace(-y_span / 2.0, y_span / 2.0, self.y_points)
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        grid_x += self.sensor_offset_x
        self.query_points_body_frame = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
        
        # --- File I/O Config ---
        self.target_layers = ["elevation", "min_filter", "smooth"]
        self.layer_ids = {"elevation": 0, "min_filter": 1, "smooth": 2}
        
        cwd = os.getcwd()
        self.log_dir = os.path.join(cwd, "elevation_to_policy_conversion_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.get_logger().info(f"Logging data to: {self.log_dir}")
        
        # File Handles (Keys: layer_name, Values: file_object)
        self.files = {}
        self.data_version = 1
        
        # Structs
        # File Header: Version(B), Res(f), OffX(f), nx(H), ny(H), Reserved(67x)
        self.file_header_fmt = struct.Struct("< B f f H H 67x")
        
        # Frame Record: Time(d), ID(B), Valid(f), Pose(7f), Reserved(32x)
        # Note: Grid data is appended raw after this struct
        self.frame_header_fmt = struct.Struct("< d B f 7f 32x")

        self.fill_value_body_frame = -0.27
        self.fill_value_absolute = 0.0

        self.create_subscription(GridMap, "/elevation_mapping_node/elevation_map_raw", self.raw_map_callback, 10)
        self.create_subscription(GridMap, "/elevation_mapping_node/elevation_map_filter", self.filtered_map_callback, 10)
        
        mode_str = "ABSOLUTE" if self.store_absolute_z else "RELATIVE"
        self.get_logger().info(f"Node Initialized. Mode: {mode_str}. Version: {self.data_version}")

    def init_log_files(self):
        """Creates one file per layer using the start timestamp."""
        timestamp_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f") # To avoid special characters in paths
        
        for layer in self.target_layers:
            filename = f"{timestamp_str}_{layer}.bin"
            filepath = os.path.join(self.log_dir, filename)
            try:
                f = open(filepath, "wb")
                
                # Write Static File Header ONCE
                header_bytes = self.file_header_fmt.pack(
                    self.data_version,
                    self.resolution,
                    self.sensor_offset_x,
                    self.x_points,
                    self.y_points
                )
                f.write(header_bytes)
                f.flush()
                
                self.files[layer] = f
                self.get_logger().info(f"Created log file: {filename}")
            except Exception as e:
                self.get_logger().error(f"Failed to create log file {filename}: {e}")

    def raw_map_callback(self, msg: GridMap):
        self.process_and_publish_zmq(msg, "elevation", self.zmq_pub_raw)

    def filtered_map_callback(self, msg: GridMap):
        stamp = msg.header.stamp
        
        # Initialize files on first callback
        if not self.files:
            self.init_log_files()

        timestamp_scalar = stamp.sec + stamp.nanosec * 1e-9
        
        # 1. Compute Geometry (Shared)
        geometry_data = self.compute_geometry(msg.info.pose)
        if geometry_data is None:
            return

        query_points_odom, robot_z, robot_pose = geometry_data

        # 2. Process Layers
        for layer_name in self.target_layers:
            # Skip if file handle creation failed earlier
            if layer_name not in self.files: continue

            result = self.interpolate_layer(msg, layer_name, query_points_odom, robot_z)
            
            if result is not None:
                heights, valid_ratio = result
                layer_id = self.layer_ids.get(layer_name, 255)
                f = self.files[layer_name]

                try:
                    # Write Frame Header
                    frame_header = self.frame_header_fmt.pack(
                        timestamp_scalar,
                        layer_id,
                        valid_ratio,
                        *robot_pose
                    )
                    f.write(frame_header)
                    
                    # Write Grid Data
                    f.write(heights.tobytes())
                    f.flush()
                except Exception as e:
                    self.get_logger().error(f"Write error for {layer_name}: {e}")
                
                if layer_name == "min_filter":
                    self.publish_zmq(heights, self.zmq_pub_filtered)

    def compute_geometry(self, map_pose):
        try:
            tf_base_to_map = self.tf_buffer.lookup_transform(
                target_frame=self.map_frame, 
                source_frame=self.robot_base_frame, 
                time=rclpy.time.Time(), 
                timeout=Duration(seconds=0.1)
            )
            trans = tf_base_to_map.transform.translation
            rot = tf_base_to_map.transform.rotation
            
            robot_pose = (trans.x, trans.y, trans.z, rot.x, rot.y, rot.z, rot.w)

            R_quat = ScipyRotation.from_quat([rot.x, rot.y, rot.z, rot.w])
            yaw = R_quat.as_euler("zxy")[0] 
            c, s = np.cos(yaw), np.sin(yaw)
            R_yaw_2D = np.array([[c, -s], [s, c]])
            
            t_base_map_2D = np.array([trans.x, trans.y])
            p_rotated = self.query_points_body_frame @ R_yaw_2D.T 
            query_points_odom = p_rotated + t_base_map_2D
            
            return query_points_odom, trans.z, robot_pose

        except TransformException as ex:
            self.get_logger().warn(f"TF Lookup Failed: {ex}", throttle_duration_sec=1.0)
            return None

    def interpolate_layer(self, msg: GridMap, layer_name: str, query_points_odom, robot_z):
        if layer_name not in msg.layers:
            if layer_name != "elevation": 
                self.get_logger().warn(f"Layer '{layer_name}' missing.", throttle_duration_sec=2.0)
            return None

        idx = msg.layers.index(layer_name)
        
        # GridMap Setup
        res = msg.info.resolution
        len_x = msg.info.length_x
        len_y = msg.info.length_y
        pos_x = msg.info.pose.position.x
        pos_y = msg.info.pose.position.y
        n_cells_x = int(round(len_x / res))
        n_cells_y = int(round(len_y / res))
        
        data_flat = np.array(msg.data[idx].data, dtype=np.float32)
        try:
            map_data = data_flat.reshape(n_cells_y, n_cells_x).T
        except ValueError:
            return None

        if not np.any(np.isfinite(map_data)):
             return None

        x_vec = np.linspace(pos_x + len_x/2.0 - res/2.0, pos_x - len_x/2.0 + res/2.0, n_cells_x)
        y_vec = np.linspace(pos_y + len_y/2.0 - res/2.0, pos_y - len_y/2.0 + res/2.0, n_cells_y)
        
        if x_vec[0] > x_vec[-1]:
            x_vec = np.flip(x_vec)
            map_data = np.flip(map_data, axis=0)
        if y_vec[0] > y_vec[-1]:
            y_vec = np.flip(y_vec)
            map_data = np.flip(map_data, axis=1)

        fill_value = self.fill_value_absolute if self.store_absolute_z else self.fill_value_body_frame

        interpolator = RegularGridInterpolator((x_vec, y_vec), map_data, bounds_error=False, fill_value=fill_value)
        interpolated_z_abs = interpolator(query_points_odom)

        if np.isnan(interpolated_z_abs).any():
            interpolated_z_abs = np.nan_to_num(interpolated_z_abs, nan=fill_value)

        valid_mask = ~np.isclose(interpolated_z_abs, fill_value, atol=1e-5)
        valid_ratio = float(np.sum(valid_mask)) / float(len(interpolated_z_abs))

        if self.store_absolute_z:
            final_data = interpolated_z_abs.astype(np.float32)
        else:
            result = np.full_like(interpolated_z_abs, fill_value)
            result[valid_mask] = interpolated_z_abs[valid_mask] - robot_z
            final_data = result.astype(np.float32)
            
        return final_data, valid_ratio

    def process_and_publish_zmq(self, msg: GridMap, layer_name: str, zmq_publisher: zmq.Socket):
        geometry_data = self.compute_geometry(msg.info.pose)
        if geometry_data is None: return
        query_points_odom, robot_z, _ = geometry_data
        
        result = self.interpolate_layer(msg, layer_name, query_points_odom, robot_z)
        if result is not None:
            heights, _ = result
            self.publish_zmq(heights, zmq_publisher)

    def publish_zmq(self, heights, zmq_sock):
        payload = heights.tobytes()
        if len(payload) == self.x_points * self.y_points * 4:
            zmq_sock.send(payload)

    def destroy_node(self):
        # Close all open files
        for name, f in self.files.items():
            try:
                f.close()
                self.get_logger().info(f"Closed log file for {name}")
            except:
                pass
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
