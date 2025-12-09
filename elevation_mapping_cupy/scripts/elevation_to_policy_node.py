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
from datetime import timezone

class ElevationToPolicyNode(Node):
    """
    Transforms GridMap data into a body-centric, policy-aligned grid.
    Writes continuous data streams to timestamped binary files (two per layer: abs and rel).
    
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
        
        self.zmq_context = zmq.Context()        
        
        # ZMQ Publishers (6 Total: 3 Layers x [Abs, Rel])
        # Format: self.zmq_sockets[layer_name][type]
        self.zmq_sockets = {}
        
        # Port map configuration
        # Base ports: 6970 (Elev), 6972 (Min), 6974 (Smooth)
        base_ports = {
            "elevation": 6970,
            "min_filter": 6972,
            "smooth": 6974
        }
        
        for layer, port in base_ports.items():
            self.zmq_sockets[layer] = {}
            
            # Absolute socket (Even port)
            sock_abs = self.zmq_context.socket(zmq.PUB)
            sock_abs.bind(f"tcp://*:{port}")
            self.zmq_sockets[layer]["abs"] = sock_abs
            
            # Relative socket (Odd port)
            sock_rel = self.zmq_context.socket(zmq.PUB)
            sock_rel.bind(f"tcp://*:{port + 1}")
            self.zmq_sockets[layer]["rel"] = sock_rel

            print(f"Started abs socket on port={port} and rel socket on port={port+1} for layer {layer}")

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
        
        # File Handles (Keys: "layer_type", Values: file_object)
        self.files = {}
        self.data_version = 1
        
        # Structs
        # File Header: Version(B), Res(f), OffX(f), nx(H), ny(H), Reserved(67x)
        self.file_header_fmt = struct.Struct("< B f f H H 67x")
        
        # Frame Record: Time(d), ID(B), Valid(f), Pose(7f), Reserved(32x)
        self.frame_header_fmt = struct.Struct("< d B f 7f 32x")

        self.fill_value_body_frame = -0.27
        self.fill_value_absolute = 0.0

        self.create_subscription(GridMap, "/elevation_mapping_node/elevation_map_filter", self.filtered_map_callback, 10)
        
        self.get_logger().info(f"Node Initialized. Source: elevation_map_filter. Mode: Dual (Abs/Rel).")

    def init_log_files(self):
        """Creates two files per layer (absolute and relative) using the start timestamp."""
        timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%f")
        sub_types = ["abs", "rel"]
        
        for layer in self.target_layers:
            for st in sub_types:
                filename = f"{timestamp_str}_{layer}_{st}.bin"
                filepath = os.path.join(self.log_dir, filename)
                key = f"{layer}_{st}"
                
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
                    
                    self.files[key] = f
                    self.get_logger().info(f"Created log file: {filename}")
                except Exception as e:
                    self.get_logger().error(f"Failed to create log file {filename}: {e}")

    def filtered_map_callback(self, msg: GridMap):
        stamp = msg.header.stamp
        
        if not self.files:
            self.init_log_files()

        timestamp_scalar = stamp.sec + stamp.nanosec * 1e-9
        
        geometry_data = self.compute_geometry(msg.info.pose, stamp)
        if geometry_data is None:
            return

        query_points_odom, robot_z, robot_pose = geometry_data

        for layer_name in self.target_layers:
            # Skip if layer not in message (warn handled in interpolate_layer_dual)
            result = self.interpolate_layer_dual(msg, layer_name, query_points_odom, robot_z)
            if result is not None:
                h_abs, h_rel, valid_ratio = result
                layer_id = self.layer_ids.get(layer_name, 255)
                self.write_frame(layer_name, "abs", layer_id, timestamp_scalar, valid_ratio, robot_pose, h_abs)
                self.publish_zmq(h_abs, self.zmq_sockets[layer_name]["abs"], timestamp_scalar)
                self.write_frame(layer_name, "rel", layer_id, timestamp_scalar, valid_ratio, robot_pose, h_rel)
                self.publish_zmq(h_rel, self.zmq_sockets[layer_name]["rel"], timestamp_scalar)
            else:
                self.get_logger().warn(f"Interpolation result for layer={layer_name} is None! Not publishing on ZMQ NOR storing in binary file")

    def write_frame(self, layer_base, sub_type, layer_id, timestamp, valid_ratio, pose, data):
        key = f"{layer_base}_{sub_type}"
        if key not in self.files: return
        f = self.files[key]
        try:
            frame_header = self.frame_header_fmt.pack(timestamp, layer_id, valid_ratio, *pose)
            f.write(frame_header)
            f.write(data.tobytes())
            f.flush()
        except Exception as e:
            self.get_logger().error(f"Write error for {key}: {e}")

    def compute_geometry(self, map_pose, timestamp_ros):
        try:
            lookup_time = rclpy.time.Time.from_msg(timestamp_ros)
            lookup_time = rclpy.time.Time()
            tf_base_to_map = self.tf_buffer.lookup_transform(
                target_frame=self.map_frame, 
                source_frame=self.robot_base_frame, 
                time=lookup_time, # Use latest time to avoid waiting for data!
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

    def interpolate_layer_dual(self, msg: GridMap, layer_name: str, query_points_odom, robot_z):
        """Computes BOTH absolute and relative grids."""
        if layer_name not in msg.layers:
            # Silence warning for Elevation if not explicitly needed, though generally Elevation, Min, Smooth should all be in Filtered map.
            self.get_logger().warn(f"Layer '{layer_name}' missing.", throttle_duration_sec=2.0)
            return None

        idx = msg.layers.index(layer_name)
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

        interpolator = RegularGridInterpolator((x_vec, y_vec), map_data, bounds_error=False, fill_value=np.nan)
        interpolated_raw = interpolator(query_points_odom)

        valid_mask = ~np.isnan(interpolated_raw)
        valid_ratio = float(np.sum(valid_mask)) / float(len(interpolated_raw))
        data_abs = np.full_like(interpolated_raw, self.fill_value_absolute)
        data_abs[valid_mask] = interpolated_raw[valid_mask]
        data_rel = np.full_like(interpolated_raw, self.fill_value_body_frame)
        data_rel[valid_mask] = interpolated_raw[valid_mask] - robot_z
        return data_abs.astype(np.float32), data_rel.astype(np.float32), valid_ratio

    def publish_zmq(self, heights, zmq_sock, timestamp_scalar):
        header = struct.pack("<d", timestamp_scalar) # Used in C++ when writing to file so that timestamps are synchronized
        payload = header + heights.tobytes()
        # if len(payload) == self.x_points * self.y_points * 4:
        zmq_sock.send(payload)

    def destroy_node(self):
        for name, f in self.files.items():
            try:
                f.close()
            except:
                pass
        
        for layer in self.zmq_sockets:
            for st in self.zmq_sockets[layer]:
                try:
                    self.zmq_sockets[layer][st].close()
                except:
                    pass
                    
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