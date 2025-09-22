import rclpy
from rclpy.node import Node
from grid_map_msgs.msg import GridMap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import tf2_ros
from tf2_ros import TransformException
from scipy.spatial.transform import Rotation as ScipyRotation
import os


class HeightGridVisualizer(Node):
    """
    Subscribes to a GridMap, transforms it to the robot's body frame,
    and visualizes it as both a 3D plot and a formatted terminal printout.
    """
    def __init__(self):
        super().__init__('height_grid_visualizer')
        
        # --- Parameters ---
        self.map_frame = 'odom'
        self.robot_base_frame = 'base'
        self.map_topic = '/elevation_mapping_node/elevation_map_filter'
        #self.input_layer = 'min_filter'
        self.input_layer = 'heights_body_frame'


        # --- TF Listener ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # --- Subscription ---
        self.subscription = self.create_subscription(
            GridMap, self.map_topic, self.map_callback, 10)
        
        # --- Matplotlib Plot Initialization ---
        plt.ion()
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.map_data_initialized = False
        
        self.get_logger().info(f"Visualizer ready. Trying to transform from '{self.map_frame}' to '{self.robot_base_frame}'.")


    def map_callback(self, msg: GridMap):
        """Processes incoming GridMap and TF data to update the 3D plot."""
        
        try:
            # --- 1. Get Robot Pose from TF ---
            tf_map_to_base = self.tf_buffer.lookup_transform(
                self.map_frame, self.robot_base_frame, rclpy.time.Time())


            tf_base_to_map = self.tf_buffer.lookup_transform(
                self.robot_base_frame, self.map_frame, rclpy.time.Time())


            trans = tf_base_to_map.transform.translation
            t_map_to_base = np.array([trans.x, trans.y, trans.z])
            rot = tf_base_to_map.transform.rotation
            R_map_to_base = ScipyRotation.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()


            # --- 2. Extract Absolute Height Grid ---
            if self.input_layer not in msg.layers:
                self.get_logger().warn(f"Layer '{self.input_layer}' not in map. Available: {msg.layers}", throttle_duration_sec=5)
                return


            rows = int(msg.info.length_x / msg.info.resolution)
            cols = int(msg.info.length_y / msg.info.resolution)


            grid_map_frame = np.array(msg.data[msg.layers.index(self.input_layer)].data, dtype=np.float32).reshape(cols, rows).T
            
            # --- 3. Create 3D Point Cloud in Map Frame ---
            if not self.map_data_initialized:
                self.rows, self.cols = grid_map_frame.shape
                map_pos_x, map_pos_y = msg.info.pose.position.x, msg.info.pose.position.y
                map_len_x, map_len_y = msg.info.length_x, msg.info.length_y
                
                x_coords = np.linspace(map_pos_x - map_len_x / 2.0, map_pos_x + map_len_x / 2.0, self.rows)
                y_coords = np.linspace(map_pos_y - map_len_y / 2.0, map_pos_y + map_len_y / 2.0, self.cols)
                self.Y_map, self.X_map = np.meshgrid(y_coords, x_coords)
                self.map_data_initialized = True


            points_map = np.stack([self.X_map.flatten(), self.Y_map.flatten(), grid_map_frame.flatten()], axis=1)
            
            # --- 4. Transform Points to Body Frame ---
            points_body = (R_map_to_base @ points_map.T).T + t_map_to_base


            # --- 5. Reconstruct Body-Frame Grids ---
            X_body_grid = points_body[:, 0].reshape(self.rows, self.cols)
            Y_body_grid = points_body[:, 1].reshape(self.rows, self.cols)
            Z_body_grid = points_body[:, 2].reshape(self.rows, self.cols)
            
            # --- 6. Update Plot ---
            self.ax.clear()
            plot_Z = np.nan_to_num(Z_body_grid, nan=0.0)
            self.ax.plot_surface(Y_body_grid, X_body_grid, plot_Z, cmap=cm.terrain, rstride=5, cstride=5, vmin=-0.5, vmax=0.5)


            self.ax.set_title(f"Robot-Centric Height Map (from '{self.input_layer}' layer)")
            self.ax.set_xlabel("Y-axis (left) [m]")
            self.ax.set_ylabel("X-axis (forward) [m]")
            self.ax.set_zlabel("Height relative to base [m]")
            self.ax.set_zlim(-0.5, 0.5)
            self.ax.view_init(elev=40., azim=-120)
            
            robot_height = tf_map_to_base.transform.translation.z
            min_h, max_h = np.nanmin(Z_body_grid), np.nanmax(Z_body_grid)
            stats_text = f"Body Height: {robot_height:.3f} m\nMin Rel. H: {min_h:.3f} m\nMax Rel. H: {max_h:.3f} m"
            self.ax.text2D(0.02, 0.98, stats_text, transform=self.ax.transAxes,
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))


            self.fig.canvas.draw()
            self.fig.canvas.flush_events()


            # --- 7. Print Raw Values to Terminal ---
            self.print_grid_to_terminal(X_body_grid, Y_body_grid, Z_body_grid, robot_height)


        except TransformException as ex:
            self.get_logger().warn(f'Could not transform: {ex}', throttle_duration_sec=2)
            return


    def print_grid_to_terminal(self, x_grid, y_grid, z_grid, robot_height):
        """Clears the terminal and prints a formatted, centered sub-grid of height values."""
        # Clear terminal screen (ANSI escape code)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("--- Real-Time Robot-Centric Height Grid (m) ---")
        print(f"Robot Height (odom): {robot_height:.3f} m")
        print("Coordinate System): +X (Forward), +Y (Left)\n")


        # Define the size of the grid to print (e.g., 15x15)
        print_size = 15
        center_row, center_col = self.rows // 2, self.cols // 2
        
        start_row = center_row - (print_size + 60) // 2
        end_row = min(self.rows, start_row + print_size + 10)
        start_col = max(0, center_col - print_size // 2)
        end_col = min(self.cols, start_col + print_size)


        # Extract the central subgrids
        z_subgrid = z_grid[start_row:end_row, start_col:end_col]
        x_coords = x_grid[start_row:end_row, center_col]
        y_coords = y_grid[center_row, start_col:end_col]
        
        # Print header (Y-axis coordinates)
        header = "  X\\Y  |"
        for y in y_coords:
            header += f" {y:^6.2f} |"
        print(header)
        print("-" * len(header))


        # Print each row with its X-axis coordinate
        for i, x in enumerate(x_coords):
            row_str = f" {x:^6.2f} |"
            for val in z_subgrid[i, :]:
                if np.isnan(val):
                    row_str += "  NAN   |"
                else:
                    # Add color for negative/positive values
                    if val < -0.02: # Negative (ground)
                        row_str += f" \033[94m{val:^6.3f}\033[0m |" # Blue
                    elif val > 0.02: # Positive (obstacle)
                        row_str += f" \033[91m{val:^6.3f}\033[0m |" # Red
                    else:
                        row_str += f" {val:^6.3f} |"
            print(row_str)
        print("-" * len(header))




    def destroy_node(self):
        plt.close(self.fig)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    visualizer = HeightGridVisualizer()
    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        pass
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
