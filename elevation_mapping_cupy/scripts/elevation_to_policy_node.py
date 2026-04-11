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
import time
import json
import threading
import queue
import gc
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
        
        # Coordinate Frames
        self.odometry_frame_id = "vicon/world"
        self.robot_base_frame_id = "vicon/Go2_Loukas/Go2_Loukas"
        self.foot_frame_ids = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        
        # Grid Configuration
        self.num_grid_points_x = 13
        self.num_grid_points_y = 11
        self.grid_resolution_meters = 0.08
        self.sensor_offset_x_meters = 0.2
        self.fill_value_body_frame = -0.27
        self.fill_value_absolute = 0.0
        self.last_publish_time = time.perf_counter()
        
        # ZMQ Configuration
        self.zmq_context = zmq.Context()        
        self.zmq_sockets = {}
        
        base_ports = {
            "elevation": 6970,
            "min_filter": 6972,
            "smooth": 6974
        }
        
        # Was having issues with occasional latency spikes that also occured even with ICMP (ping -i 0.02 -s 2000)
        # so these params are optimized for low latency and discarding older messages
        for layer_name, port_number in base_ports.items():
            self.zmq_sockets[layer_name] = {}
            socket_absolute = self.zmq_context.socket(zmq.PUB)
            
            socket_absolute.setsockopt(zmq.CONFLATE, 1) # Keep only the absolute newest message, drop all others
            socket_absolute.setsockopt(zmq.SNDHWM, 2) # Limit High Water Mark (redundant with conflate, but good practice)
            socket_absolute.setsockopt(zmq.LINGER, 0) # Prevent socket from hanging on shutdown trying to send old data
            
            socket_absolute.bind(f"tcp://*:{port_number}")
            self.zmq_sockets[layer_name]["abs"] = socket_absolute
            
            socket_relative = self.zmq_context.socket(zmq.PUB)
            
            socket_relative.setsockopt(zmq.CONFLATE, 1)
            socket_relative.setsockopt(zmq.SNDHWM, 2)
            socket_relative.setsockopt(zmq.LINGER, 0)
            
            socket_relative.bind(f"tcp://*:{port_number + 1}")
            self.zmq_sockets[layer_name]["rel"] = socket_relative


            self.get_logger().info(f"ZMQ Init: {layer_name} (Abs: {port_number}, Rel: {port_number+1})")

        # Pre-compute local grid points in the robot's body frame
        # This creates a meshgrid centered around (0,0) locally, then shifts it by the sensor offset.
        np.set_printoptions(suppress=True)
        span_x_meters = (self.num_grid_points_x - 1) * self.grid_resolution_meters  
        span_y_meters = (self.num_grid_points_y - 1) * self.grid_resolution_meters
        
        local_x_coordinates = np.linspace(-span_x_meters / 2.0, span_x_meters / 2.0, self.num_grid_points_x) 
        local_y_coordinates = np.linspace(-span_y_meters / 2.0, span_y_meters / 2.0, self.num_grid_points_y)
        
        local_grid_mesh_x, local_grid_mesh_y = np.meshgrid(local_x_coordinates, local_y_coordinates)
        local_grid_mesh_x += self.sensor_offset_x_meters
        
        # Flatten and stack to create an (N, 2) array of local points [x, y]
        self.query_points_body_frame = np.vstack([local_grid_mesh_x.ravel(), local_grid_mesh_y.ravel()]).T
        self.target_layer_names = ["elevation", "min_filter", "smooth"]
        
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
        # Shared memory for the latest map interpolator
        self.latest_map_context = None
        self.map_lock = threading.Lock()
        self.create_timer(1.0 / 50.0, self.publish_timer_callback)
        self.get_logger().info(f"Initialized. Logging NDJSON to: {self.log_directory}")

    def init_log_files(self):
        """
        Initializes file handles for NDJSON logging.
        Writes the metadata header to each new file.
        """
        timestamp_string = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%f")
        layer_types = ["abs", "rel"]
        
        for layer_name in self.target_layer_names:
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
                            "resolution": self.grid_resolution_meters,
                            "sensor_offset_x": self.sensor_offset_x_meters,
                            "num_x": int(self.num_grid_points_x),
                            "num_y": int(self.num_grid_points_y),
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
        """
        Daemon thread worker that pulls JSON strings from a queue and writes them to disk.
        Prevents file I/O from blocking the main ROS execution loop.
        """
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
                print(f"Async Write Error: {e}") # Using print here as logging might be risky inside a daemon thread if ROS is shutting down

    def filtered_map_callback(self, grid_map_message: GridMap):
        """
        Callback for the Elevation Map. 
        Converts raw grid data into scipy RegularGridInterpolators.
        These interpolators are cached for high-frequency querying in the timer loop.
        """
        if not self.active_file_handles:
            self.init_log_files()

        stamp = grid_map_message.header.stamp
        map_ts_scalar = stamp.sec + stamp.nanosec * 1e-9

        new_interpolators = {}

        for layer_name in self.target_layer_names:
            interpolator = self.create_layer_interpolator(grid_map_message, layer_name)
            if interpolator is not None:
                new_interpolators[layer_name] = interpolator
            else:
                self.get_logger().warn(f"Could not create interpolator for layer: {layer_name}", throttle_duration_sec=0.2)
        
        if new_interpolators:
            with self.map_lock:
                self.latest_map_context = {
                    "ts": map_ts_scalar,
                    "interpolators": new_interpolators
                }
        else:
            self.get_logger().warn("No valid interpolators created from map update.", throttle_duration_sec=0.2)

    def publish_timer_callback(self):
        """
        Main Control Loop.
        1. Gets the latest Robot Pose (TF).
        2. Retrieves the latest cached Map Interpolator.
        3. Queries the map at the robot's current location.
        4. Publishes ZMQ messages.
        """
        gc.disable()
        # Retrieve the latest map context safely
        current_map_context = None
        with self.map_lock:
            if self.latest_map_context:
                current_map_context = self.latest_map_context
        
        if current_map_context is None:
            self.get_logger().warn("Waiting for first map...", throttle_duration_sec=0.2)
            return
        
        delta_t_ms_last_iteration = (time.perf_counter() - self.last_publish_time) / 1e+3
        if delta_t_ms_last_iteration > 20.0:
            print("!!!!!!!!!!!!!!!!!!! Iteration took longer than 20ms!!!!!!!!!!!!!!!!")
            print(f"actual delta_t={delta_t_ms_last_iteration}")
        self.last_publish_time = time.perf_counter()

        # Get the current time for the robot pose
        current_ros_time = self.get_clock().now()
        current_time_scalar = current_ros_time.nanoseconds * 1e-9
        
        # Compute where the local grid points fall in the global frame based on current robot pose
        geometry_data = self.compute_geometry(current_ros_time)
        
        if geometry_data is None:
            self.get_logger().warn("Skipping cycle: Failed to compute geometry (TF missing?).", throttle_duration_sec=0.2)
            return 

        query_points_global_frame, robot_height_z, robot_pose_tuple = geometry_data
        pose_dictionary = {
            "x": robot_pose_tuple[0], "y": robot_pose_tuple[1], "z": robot_pose_tuple[2],
            "qx": robot_pose_tuple[3], "qy": robot_pose_tuple[4], "qz": robot_pose_tuple[5], "qw": robot_pose_tuple[6]
        }
        feet_positions_list = self.get_foot_positions()
        map_ts = current_map_context["ts"]
        interpolators = current_map_context["interpolators"]

        for layer_name in self.target_layer_names:
            if layer_name not in interpolators:
                continue

            # Query the cached interpolator. This performs bilinear interpolation at the specific global coordinates
            layer_interpolator = interpolators[layer_name]
            interpolated_values = layer_interpolator(query_points_global_frame)

            # Determine which points landed on valid map data
            valid_cells_mask = ~np.isnan(interpolated_values)
            valid_ratio = float(np.sum(valid_cells_mask)) / float(len(interpolated_values))
            
            # Prepare Absolute Data (World Frame Heights)
            grid_absolute = np.full_like(interpolated_values, self.fill_value_absolute)
            grid_absolute[valid_cells_mask] = interpolated_values[valid_cells_mask]
            
            # Prepare Relative Data (Body Frame Heights)
            # Z_rel = Z_world_point - Z_robot_base
            grid_relative = np.full_like(interpolated_values, self.fill_value_body_frame)
            grid_relative[valid_cells_mask] = interpolated_values[valid_cells_mask] - robot_height_z

            # Construct Packets
            packet_absolute = {
                "ts": current_time_scalar,
                "map_ts": map_ts,
                "layer": layer_name,
                "type": "abs",
                "valid": valid_ratio,
                "pose": pose_dictionary,
                "feet": feet_positions_list,
                "grid": grid_absolute
            }
            self.dispatch_data(layer_name, "abs", packet_absolute)
            
            packet_relative = {
                "ts": current_time_scalar,
                "map_ts": map_ts,
                "layer": layer_name,
                "type": "rel",
                "valid": valid_ratio,
                "pose": pose_dictionary,
                "feet": feet_positions_list,
                "grid": grid_relative
            }
            self.dispatch_data(layer_name, "rel", packet_relative)
            gc.enable()

    def dispatch_data(self, layer_name, layer_type, data_dictionary):
        """
        Serializes data to JSON and sends via ZMQ.
        Also queues the data for logging to disk.
        """
        try:
            json_string_payload = json.dumps(data_dictionary, cls=NumpyEncoder, separators=(',', ':'))
            
            self.zmq_sockets[layer_name][layer_type].send(json_string_payload.encode('utf-8'))
            
            file_key = f"{layer_name}_{layer_type}"
            if file_key in self.active_file_handles:
                self.write_queue.put((file_key, json_string_payload))
                
        except Exception as e:
            self.get_logger().error(f"Dispatch Error in {layer_name}-{layer_type}: {e}", throttle_duration_sec=1.0)

    def get_foot_positions(self):
        """
        Lookups the position of all foot frames relative to the robot base.
        Returns a flat list [x1, y1, z1, x2, y2, z2...].
        """
        feet_coordinates_flat = []
        lookup_time = rclpy.time.Time() # Time(0) gives the latest available transform
        
        for foot_frame in self.foot_frame_ids:
            try:
                transform_stamped = self.tf_buffer.lookup_transform(
                    target_frame=self.robot_base_frame_id,
                    source_frame=foot_frame,
                    time=lookup_time,
                    timeout=Duration(seconds=0.005)
                )
                feet_coordinates_flat.extend([
                    transform_stamped.transform.translation.x,
                    transform_stamped.transform.translation.y,
                    transform_stamped.transform.translation.z
                ])
            except TransformException as ex:
                self.get_logger().warn(f"Failed to lookup foot {foot_frame}: {ex}", throttle_duration_sec=0.2)
                feet_coordinates_flat.extend([None, None, None])
        return feet_coordinates_flat

    def compute_geometry(self, timestamp_ros):
        """
        Calculates the global coordinates of the grid points surrounding the robot.
        Math: P_global = R_yaw * P_local + T_robot
        """
        try:
            # Use Time(0) to get the very latest transform available
            lookup_time = rclpy.time.Time()
            tf_base_to_map = self.tf_buffer.lookup_transform(
                target_frame=self.odometry_frame_id, 
                source_frame=self.robot_base_frame_id, 
                time=lookup_time,
                timeout=Duration(seconds=0.005)
            )
            
            translation = tf_base_to_map.transform.translation
            rotation = tf_base_to_map.transform.rotation
            robot_pose_tuple = (translation.x, translation.y, translation.z, rotation.x, rotation.y, rotation.z, rotation.w)

            # Extract Yaw (rotation around Z) to align the grid with the robot's heading
            rotation_object = ScipyRotation.from_quat([rotation.x, rotation.y, rotation.z, rotation.w])
            yaw_angle = rotation_object.as_euler("zxy")[0] 
            
            # Construct 2D Rotation Matrix
            cosine_yaw, sine_yaw = np.cos(yaw_angle), np.sin(yaw_angle)
            rotation_matrix_2d = np.array([[cosine_yaw, -sine_yaw], [sine_yaw, cosine_yaw]])
            
            # Apply Rotation and Translation. P_local is (N, 2), Rotation is (2, 2). Result is (N, 2)
            points_rotated = self.query_points_body_frame @ rotation_matrix_2d.T 
            
            translation_base_map_2d = np.array([translation.x, translation.y])
            query_points_global_frame = points_rotated + translation_base_map_2d
            
            return query_points_global_frame, translation.z, robot_pose_tuple
            
        except TransformException as ex:
            self.get_logger().error(f"TF Lookup Failed (Base->Map): {ex}", throttle_duration_sec=1.0)
            return None

    def create_layer_interpolator(self, grid_map_message: GridMap, layer_name: str):
        """
        Extracts specific layer data from GridMap and builds a RegularGridInterpolator.
        
        Inference:
        - position_x/y in GridMap info represents the geometric center of the map.
        - bounds are [center - length/2, center + length/2].
        """
        if layer_name not in grid_map_message.layers:
            self.get_logger().warn(f"Layer {layer_name} not found in map.", throttle_duration_sec=5.0)
            return None

        layer_index = grid_map_message.layers.index(layer_name)
        resolution = grid_map_message.info.resolution
        length_x = grid_map_message.info.length_x
        length_y = grid_map_message.info.length_y
        
        # Center of the map in the map frame
        map_center_x = grid_map_message.info.pose.position.x
        map_center_y = grid_map_message.info.pose.position.y
        num_cells_x = int(round(length_x / resolution))
        num_cells_y = int(round(length_y / resolution))
        data_flat = np.array(grid_map_message.data[layer_index].data, dtype=np.float32)
        
        try:
            # Reshape raw data 1D -> 2D (Cols, Rows) then Transpose to match (X, Y) indexing
            map_data_grid = data_flat.reshape(num_cells_y, num_cells_x).T
        except ValueError as e:
            self.get_logger().error(f"Failed to reshape map data: {e}", throttle_duration_sec=2.0)
            return None

        # Check for completely empty/invalid map
        if not np.any(np.isfinite(map_data_grid)):
             return None

        # Create coordinate vectors corresponding to the grid cell centers
        x_vector = np.linspace(map_center_x + length_x/2.0 - resolution/2.0, 
                               map_center_x - length_x/2.0 + resolution/2.0, 
                               num_cells_x)
        y_vector = np.linspace(map_center_y + length_y/2.0 - resolution/2.0, 
                               map_center_y - length_y/2.0 + resolution/2.0, 
                               num_cells_y)
        
        # Ensure vectors are strictly ascending for the Interpolator
        if x_vector[0] > x_vector[-1]:
            x_vector = np.flip(x_vector)
            map_data_grid = np.flip(map_data_grid, axis=0)
        if y_vector[0] > y_vector[-1]:
            y_vector = np.flip(y_vector)
            map_data_grid = np.flip(map_data_grid, axis=1)

        PRINT_RAW_VALUES = False
        if PRINT_RAW_VALUES and layer_name == "min_filter":
            # Slice Front Half (Mid -> End) and Downsample (Step 2)
            # map_data_grid is (X, Y). x-axis is forward.
            mid_idx = map_data_grid.shape[0] // 2
            front_raw_downsampled = (map_data_grid[mid_idx::2, ::2] + 0.0) * 100.0
            # Configure Numpy to print EVERYTHING (no truncation)
            # linewidth=400 attempts to keep rows on one line in wide terminals
            np.set_printoptions(threshold=20000, linewidth=400, precision=1, suppress=True)
            print(front_raw_downsampled)

        try:
            interpolator = RegularGridInterpolator((x_vector, y_vector), map_data_grid, bounds_error=False, fill_value=np.nan)
            return interpolator
        except Exception as e:
            self.get_logger().error(f"Failed to create RegularGridInterpolator: {e}", throttle_duration_sec=2.0)
            return None

    def destroy_node(self):
        self.is_running = False
        if self.io_thread.is_alive():
            self.io_thread.join(timeout=2.0)
            
        for key, file_handle in self.active_file_handles.items():
            try:
                file_handle.close()
            except Exception as e:
                self.get_logger().warn(f"Error closing file {key}: {e}")
        
        for layer in self.zmq_sockets:
            for layer_type in self.zmq_sockets[layer]:
                try:
                    self.zmq_sockets[layer][layer_type].close()
                except Exception:
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