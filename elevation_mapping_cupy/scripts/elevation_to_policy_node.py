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
import json
import threading
import queue
from datetime import datetime, timezone

class NumpyEncoder(json.JSONEncoder):
    """
    Ensures numpy types are converted to native python types for JSON serialization.
    Preserves precision by using Python's default float repr.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)

class ElevationToPolicyNode(Node):
    def __init__(self):
        super().__init__("elevation_to_policy_node")
        
        self.map_frame = "odom"
        self.robot_base_frame = "base"
        self.foot_frames = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        
        self.grid_points_x = 13
        self.grid_points_y = 11
        self.grid_resolution = 0.08
        self.sensor_offset_x = 0.2
        self.fill_value_body_frame = -0.27
        self.fill_value_absolute = 0.0
        
        self.zmq_context = zmq.Context()        
        self.zmq_sockets = {}
        
        base_ports = {
            "elevation": 6970,
            "min_filter": 6972,
            "smooth": 6974
        }
        
        for layer_name, port_number in base_ports.items():
            self.zmq_sockets[layer_name] = {}
            
            socket_absolute = self.zmq_context.socket(zmq.PUB)
            socket_absolute.bind(f"tcp://*:{port_number}")
            self.zmq_sockets[layer_name]["abs"] = socket_absolute
            
            socket_relative = self.zmq_context.socket(zmq.PUB)
            socket_relative.bind(f"tcp://*:{port_number + 1}")
            self.zmq_sockets[layer_name]["rel"] = socket_relative

            print(f"ZMQ Init: {layer_name} (Abs: {port_number}, Rel: {port_number+1})")

        np.set_printoptions(suppress=True)
        x_span = (self.grid_points_x - 1) * self.grid_resolution  
        y_span = (self.grid_points_y - 1) * self.grid_resolution
        x_coords = np.linspace(-x_span / 2.0, x_span / 2.0, self.grid_points_x) 
        y_coords = np.linspace(-y_span / 2.0, y_span / 2.0, self.grid_points_y)
        grid_mesh_x, grid_mesh_y = np.meshgrid(x_coords, y_coords)
        grid_mesh_x += self.sensor_offset_x
        self.query_points_body_frame = np.vstack([grid_mesh_x.ravel(), grid_mesh_y.ravel()]).T
        
        self.target_layers = ["elevation", "min_filter", "smooth"]
        
        current_working_directory = os.getcwd()
        self.log_directory = os.path.join(current_working_directory, "elevation_to_policy_logs_json")
        os.makedirs(self.log_directory, exist_ok=True)
        
        self.active_file_handles = {}
        
        self.write_queue = queue.Queue() 
        self.is_running = True
        self.io_thread = threading.Thread(target=self._io_worker, daemon=True)
        self.io_thread.start()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_subscription(GridMap, "/elevation_mapping_node/elevation_map_filter", self.filtered_map_callback, 10)
        
        self.get_logger().info(f"Initialized. Logging NDJSON to: {self.log_directory}")

    def init_log_files(self):
        """Creates file handles and writes the Metadata Header line."""
        timestamp_string = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%f")
        layer_types = ["abs", "rel"]
        
        for layer_name in self.target_layers:
            for layer_type in layer_types:
                filename = f"{timestamp_string}_{layer_name}_{layer_type}.jsonl"
                filepath = os.path.join(self.log_directory, filename)
                file_key = f"{layer_name}_{layer_type}"
                
                try:
                    log_file = open(filepath, "w", encoding='utf-8')
                    fill_value = float(self.fill_value_absolute if layer_type == "abs" else self.fill_value_body_frame)

                    metadata = {
                        "type": "metadata",
                        "version": 3,
                        "config": {
                            "resolution": self.grid_resolution,
                            "sensor_offset_x": self.sensor_offset_x,
                            "num_x": int(self.grid_points_x),
                            "num_y": int(self.grid_points_y),
                            "fill_value": fill_value
                        }
                    }
                    
                    log_file.write(json.dumps(metadata) + "\n")
                    log_file.flush()
                    self.active_file_handles[file_key] = log_file
                    self.get_logger().info(f"Created log file: {filename}")
                except Exception as e:
                    self.get_logger().error(f"Failed to create log file {filename}: {e}")

    def _io_worker(self):
        while self.is_running:
            try:
                queue_item = self.write_queue.get(timeout=1.0) 
                file_key, json_string_data = queue_item
                
                if file_key in self.active_file_handles:
                    log_file = self.active_file_handles[file_key]
                    log_file.write(json_string_data + "\n")
                    log_file.flush()
                
                self.write_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Async Write Error: {e}")

    def filtered_map_callback(self, grid_map_message: GridMap):
        if not self.active_file_handles:
            self.init_log_files()
            
        stamp = grid_map_message.header.stamp
        timestamp_scalar = stamp.sec + stamp.nanosec * 1e-9
        geometry_data = self.compute_geometry(grid_map_message.info.pose, stamp)
        if geometry_data is None:
            self.get_logger().warn("Failed to get transformed points, returning in subscriber callback!")
            return

        query_points_odom, robot_height_z, robot_pose_tuple = geometry_data
        pose_dictionary = {
            "x": robot_pose_tuple[0], "y": robot_pose_tuple[1], "z": robot_pose_tuple[2],
            "qx": robot_pose_tuple[3], "qy": robot_pose_tuple[4], "qz": robot_pose_tuple[5], "qw": robot_pose_tuple[6]
        }
        feet_positions_list = self.get_foot_positions()

        for layer_name in self.target_layers:
            interpolation_result = self.interpolate_layer_dual(grid_map_message, layer_name, query_points_odom, robot_height_z)
            
            if interpolation_result is not None:
                grid_absolute, grid_relative, valid_ratio = interpolation_result
                
                packet_absolute = {
                    "ts": timestamp_scalar,
                    "layer": layer_name,
                    "type": "abs",
                    "valid": valid_ratio,
                    "pose": pose_dictionary,
                    "feet": feet_positions_list,
                    "grid": grid_absolute
                }
                self.dispatch_data(layer_name, "abs", packet_absolute)
                
                packet_relative = {
                    "ts": timestamp_scalar,
                    "layer": layer_name,
                    "type": "rel",
                    "valid": valid_ratio,
                    "pose": pose_dictionary,
                    "feet": feet_positions_list,
                    "grid": grid_relative
                }
                self.dispatch_data(layer_name, "rel", packet_relative)

    def dispatch_data(self, layer_name, layer_type, data_dictionary):
        try:
            json_string_payload = json.dumps(data_dictionary, cls=NumpyEncoder, separators=(',', ':'))
            
            self.zmq_sockets[layer_name][layer_type].send(json_string_payload.encode('utf-8'))
            
            file_key = f"{layer_name}_{layer_type}"
            if file_key in self.active_file_handles:
                self.write_queue.put((file_key, json_string_payload))
                
        except Exception as e:
            self.get_logger().error(f"Dispatch Error: {e}")

    def get_foot_positions(self):
        feet_coordinates = []
        lookup_time = rclpy.time.Time()
        
        for foot_frame in self.foot_frames:
            try:
                transform_stamped = self.tf_buffer.lookup_transform(
                    target_frame=self.robot_base_frame,
                    source_frame=foot_frame,
                    time=lookup_time,
                    timeout=Duration(seconds=0.005)
                )
                feet_coordinates.extend([
                    transform_stamped.transform.translation.x,
                    transform_stamped.transform.translation.y,
                    transform_stamped.transform.translation.z
                ])
            except TransformException:
                self.get_logger().warn("Failed to get foot positions!")
                feet_coordinates.extend([None, None, None])
        return feet_coordinates

    def compute_geometry(self, map_pose, timestamp_ros):
        try:
            lookup_time = rclpy.time.Time()
            tf_base_to_map = self.tf_buffer.lookup_transform(
                target_frame=self.map_frame, 
                source_frame=self.robot_base_frame, 
                time=lookup_time,
                timeout=Duration(seconds=0.005)
            )
            translation = tf_base_to_map.transform.translation
            rotation = tf_base_to_map.transform.rotation
            robot_pose_tuple = (translation.x, translation.y, translation.z, rotation.x, rotation.y, rotation.z, rotation.w)

            rotation_object = ScipyRotation.from_quat([rotation.x, rotation.y, rotation.z, rotation.w])
            yaw_angle = rotation_object.as_euler("zxy")[0] 
            cosine_yaw, sine_yaw = np.cos(yaw_angle), np.sin(yaw_angle)
            rotation_matrix_2d = np.array([[cosine_yaw, -sine_yaw], [sine_yaw, cosine_yaw]])
            
            translation_base_map_2d = np.array([translation.x, translation.y])
            points_rotated = self.query_points_body_frame @ rotation_matrix_2d.T 
            query_points_odom = points_rotated + translation_base_map_2d
            
            return query_points_odom, translation.z, robot_pose_tuple
        except TransformException as ex:
            self.get_logger().error(f"TF Lookup Failed (Base->Map): {ex}")
            return None

    def interpolate_layer_dual(self, grid_map_message: GridMap, layer_name: str, query_points_odom, robot_height_z):
        if layer_name not in grid_map_message.layers:
            return None

        layer_index = grid_map_message.layers.index(layer_name)
        resolution = grid_map_message.info.resolution
        length_x = grid_map_message.info.length_x
        length_y = grid_map_message.info.length_y
        position_x = grid_map_message.info.pose.position.x
        position_y = grid_map_message.info.pose.position.y
        num_cells_x = int(round(length_x / resolution))
        num_cells_y = int(round(length_y / resolution))
        
        data_flat = np.array(grid_map_message.data[layer_index].data, dtype=np.float32)
        try:
            map_data_grid = data_flat.reshape(num_cells_y, num_cells_x).T
        except ValueError:
            return None

        if not np.any(np.isfinite(map_data_grid)):
             return None

        x_vector = np.linspace(position_x + length_x/2.0 - resolution/2.0, position_x - length_x/2.0 + resolution/2.0, num_cells_x)
        y_vector = np.linspace(position_y + length_y/2.0 - resolution/2.0, position_y - length_y/2.0 + resolution/2.0, num_cells_y)
        
        if x_vector[0] > x_vector[-1]:
            x_vector = np.flip(x_vector)
            map_data_grid = np.flip(map_data_grid, axis=0)
        if y_vector[0] > y_vector[-1]:
            y_vector = np.flip(y_vector)
            map_data_grid = np.flip(map_data_grid, axis=1)

        interpolator = RegularGridInterpolator((x_vector, y_vector), map_data_grid, bounds_error=False, fill_value=np.nan)
        interpolated_raw_values = interpolator(query_points_odom)

        valid_mask = ~np.isnan(interpolated_raw_values)
        valid_ratio = float(np.sum(valid_mask)) / float(len(interpolated_raw_values))
        
        data_absolute = np.full_like(interpolated_raw_values, self.fill_value_absolute)
        data_absolute[valid_mask] = interpolated_raw_values[valid_mask]
        
        data_relative = np.full_like(interpolated_raw_values, self.fill_value_body_frame)
        data_relative[valid_mask] = interpolated_raw_values[valid_mask] - robot_height_z
        
        return data_absolute, data_relative, valid_ratio

    def destroy_node(self):
        self.is_running = False
        if self.io_thread.is_alive():
            self.io_thread.join(timeout=2.0)
            
        for key, file_handle in self.active_file_handles.items():
            try:
                file_handle.close()
            except:
                pass
        
        for layer in self.zmq_sockets:
            for layer_type in self.zmq_sockets[layer]:
                try:
                    self.zmq_sockets[layer][layer_type].close()
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