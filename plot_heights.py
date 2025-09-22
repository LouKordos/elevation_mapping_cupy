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
    and visualizes it as both a 3D plot and a 2D annotated heatmap.
    """
    def __init__(self):
        super().__init__('height_grid_visualizer')
        
        # --- Parameters ---
        self.map_frame = 'odom'
        self.robot_base_frame = 'base'
        self.map_topic = '/elevation_mapping_node/elevation_map_filter'
        self.input_layer = 'min_filter' # Changed back to min_filter for map-frame data


        # --- Plotting Ranges ---
        # Set your desired plotting ranges here (in meters, in the robot's body frame)
        self.set_plot_ranges(x_range=[-1.5, 0], y_range=[-0.5, 0.5])


        # --- TF Listener ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # --- Subscription ---
        self.subscription = self.create_subscription(
            GridMap, self.map_topic, self.map_callback, 10)
        
        # --- Matplotlib Plot Initialization ---
        plt.ion()
        self.fig = plt.figure(figsize=(28, 12))
        self.ax_3d = self.fig.add_subplot(121, projection='3d')
        self.ax_2d = self.fig.add_subplot(122)
        
        self.map_data_initialized = False
        self.cbar = None
        
        self.get_logger().info(f"Visualizer ready. Trying to transform from '{self.map_frame}' to '{self.robot_base_frame}'.")


    def set_plot_ranges(self, x_range, y_range):
        """Sets the desired X and Y ranges for plotting."""
        # Ensure x_range is [min, max]
        self.plot_x_min, self.plot_x_max = min(x_range), max(x_range)
        # Ensure y_range is [min, max]
        self.plot_y_min, self.plot_y_max = min(y_range), max(y_range)
        self.get_logger().info(f"Plotting range set to X: [{self.plot_x_min}, {self.plot_x_max}], Y: [{self.plot_y_min}, {self.plot_y_max}]")


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
            
            # --- 6. Crop Grids to Desired Plotting Range ---
            crop_mask = (X_body_grid >= self.plot_x_min) & (X_body_grid <= self.plot_x_max) & \
                        (Y_body_grid >= self.plot_y_min) & (Y_body_grid <= self.plot_y_max)
            
            if not np.any(crop_mask):
                self.get_logger().warn("Desired plot range is outside the current map view.", throttle_duration_sec=5)
                return
            
            # Apply mask to get indices for subgrids
            row_indices, col_indices = np.where(crop_mask)
            min_row, max_row = np.min(row_indices), np.max(row_indices)
            min_col, max_col = np.min(col_indices), np.max(col_indices)


            # Create subgrids for plotting
            x_subgrid = X_body_grid[min_row:max_row+1, min_col:max_col+1]
            y_subgrid = Y_body_grid[min_row:max_row+1, min_col:max_col+1]
            z_subgrid = Z_body_grid[min_row:max_row+1, min_col:max_col+1]


            # --- 7. Update Plots ---
            self.ax_3d.clear()
            self.ax_2d.clear()
            
            cmap = cm.terrain
            vmin, vmax = -0.5, 0.5


            # --- 3D Surface Plot (Left) ---
            plot_Z_3d = np.nan_to_num(z_subgrid, nan=0.0)
            self.ax_3d.plot_surface(y_subgrid, x_subgrid, plot_Z_3d, cmap=cmap, rstride=1, cstride=1, vmin=vmin, vmax=vmax)
            self.ax_3d.set_title("3D Surface View (Cropped)")
            self.ax_3d.set_xlabel("Y-axis (left) [m]")
            self.ax_3d.set_ylabel("X-axis (forward) [m]")
            self.ax_3d.set_zlabel("Height relative to base [m]")
            self.ax_3d.set_zlim(vmin, vmax)
            self.ax_3d.view_init(elev=40., azim=-120)


            # --- 2D Heatmap Plot (Right) ---
            im = self.ax_2d.imshow(np.flipud(z_subgrid), cmap=cmap, vmin=vmin, vmax=vmax, 
                                   extent=[y_subgrid.min(), y_subgrid.max(), x_subgrid.min(), x_subgrid.max()],
                                   aspect='equal')


            for r in range(z_subgrid.shape[0]):
                for c in range(z_subgrid.shape[1]):
                    val = z_subgrid[r, c]
                    if not np.isnan(val):
                        # imshow Y-axis is flipped, so we adjust indexing
                        self.ax_2d.text(y_subgrid[0, c], x_subgrid[r, 0], f"{val:.2f}",
                                        ha="center", va="center", color="black", fontsize=8)


            self.ax_2d.set_title("2D Top-Down Heatmap (Cropped)")
            self.ax_2d.set_xlabel("Y-axis (left) [m]")
            self.ax_2d.set_ylabel("X-axis (forward) [m]")
            self.ax_2d.grid(True, which='both', linestyle='--', linewidth=0.5)


            # --- Update Colorbar, Stats and Redraw ---
            if self.cbar:
                self.cbar.update_normal(im)
            else:
                self.cbar = self.fig.colorbar(im, ax=[self.ax_2d, self.ax_3d], shrink=0.6, label='Height [m]')


            robot_height = tf_map_to_base.transform.translation.z
            min_h, max_h = np.nanmin(Z_body_grid), np.nanmax(Z_body_grid)
            stats_text = f"Body Height: {robot_height:.3f} m\nMin Rel. H (Full): {min_h:.3f} m\nMax Rel. H (Full): {max_h:.3f} m"
            self.fig.suptitle(stats_text, fontsize=14)
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except TransformException as ex:
            self.get_logger().warn(f'Could not transform: {ex}', throttle_duration_sec=2)
            return


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
