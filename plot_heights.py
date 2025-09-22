import rclpy
from rclpy.node import Node
from grid_map_msgs.msg import GridMap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
import tf2_ros
from tf2_ros import TransformException
from scipy.spatial.transform import Rotation as ScipyRotation


class HeightGridVisualizer(Node):
    """
    Subscribes to a GridMap message and visualizes the height data.
    It can optionally transform the map into the robot's base frame and display
    it as interactive 3D surface and 2D heatmap plots.
    """
    def __init__(self):
        super().__init__('height_grid_visualizer')
        
        # --- Configuration Parameters ---
        self.map_frame = 'odom'
        self.robot_base_frame = 'base'
        self.map_topic = '/elevation_mapping_node/elevation_map_filter'
        #self.input_layer = 'heights_body_frame'
        self.input_layer = 'min_filter'
        
        # --- Feature Flags ---
        # Requirement 0: Add a boolean feature flag for TF transformation.
        self.use_tf_transformation = True


        # --- Plotting Ranges & Appearance ---
        # Requirement 2: Allow setting xy ranges for 3d and 2d separately.
        # Define plotting ranges in meters. If use_tf_transformation is True,
        # these ranges are relative to the robot's base_frame.
        self.plot_x_range_3d = [-1.5, -0.2]
        self.plot_y_range_3d = [-0.5, 0.5]
        self.plot_x_range_2d = [-1.5, -0.2]
        self.plot_y_range_2d = [-0.5, 0.5]
        
        # --- TF Listener ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # --- Subscription ---
        self.subscription = self.create_subscription(
            GridMap, self.map_topic, self.map_callback, 10)
        
        # --- Matplotlib Plot Initialization ---
        plt.ion()
        # Requirement 3: Reduce figure height by 1 unit.
        self.fig = plt.figure(figsize=(28, 11))
        
        # Requirement 1: Make the 2D plot wider and adjust padding.
        # Use GridSpec for a flexible layout, allocating more space to the 2D plot.
        gs = self.fig.add_gridspec(1, 2, width_ratios=[2, 3])
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.15)
        
        self.ax_3d = self.fig.add_subplot(gs[0, 0], projection='3d')
        self.ax_2d = self.fig.add_subplot(gs[0, 1])
        
        self.map_data_initialized = False
        self.cbar = None
        
        self.get_logger().info(f"Visualizer ready. TF Transformation is {'ENABLED' if self.use_tf_transformation else 'DISABLED'}.")


    def _crop_grids(self, X_grid, Y_grid, Z_grid, x_range, y_range):
        """Helper function to crop input grids based on specified x and y ranges."""
        # Ensure ranges are ordered [min, max]
        x_min, x_max = min(x_range), max(x_range)
        y_min, y_max = min(y_range), max(y_range)


        # Create a boolean mask for points within the desired ranges
        crop_mask = (X_grid >= x_min) & (X_grid <= x_max) & \
                    (Y_grid >= y_min) & (Y_grid <= y_max)


        if not np.any(crop_mask):
            return None, None, None


        # Find the bounding box of the valid indices to create contiguous subgrids
        row_indices, col_indices = np.where(crop_mask)
        min_row, max_row = np.min(row_indices), np.max(row_indices)
        min_col, max_col = np.min(col_indices), np.max(col_indices)


        x_subgrid = X_grid[min_row:max_row+1, min_col:max_col+1]
        y_subgrid = Y_grid[min_row:max_row+1, min_col:max_col+1]
        z_subgrid = Z_grid[min_row:max_row+1, min_col:max_col+1]
        
        return x_subgrid, y_subgrid, z_subgrid


    def map_callback(self, msg: GridMap):
        """Processes incoming GridMap data and updates the plots."""
        # Find and reshape the specified height layer from the message
        if self.input_layer not in msg.layers:
            self.get_logger().warn(f"Layer '{self.input_layer}' not in map. Available: {msg.layers}", throttle_duration_sec=5)
            return


        rows = int(msg.info.length_x / msg.info.resolution)
        cols = int(msg.info.length_y / msg.info.resolution)
        height_grid_map_frame = np.array(msg.data[msg.layers.index(self.input_layer)].data, dtype=np.float32).reshape(cols, rows).T
        
        # One-time initialization of map coordinate grids
        if not self.map_data_initialized:
            self.rows, self.cols = height_grid_map_frame.shape
            map_pos_x, map_pos_y = msg.info.pose.position.x, msg.info.pose.position.y
            map_len_x, map_len_y = msg.info.length_x, msg.info.length_y
            
            x_coords = np.linspace(map_pos_x - map_len_x / 2.0, map_pos_x + map_len_x / 2.0, self.rows)
            y_coords = np.linspace(map_pos_y - map_len_y / 2.0, map_pos_y + map_len_y / 2.0, self.cols)
            self.Y_map, self.X_map = np.meshgrid(y_coords, x_coords)
            self.map_data_initialized = True


        # By default, the grids to plot are in the map frame
        X_plot, Y_plot, Z_plot = self.X_map, self.Y_map, height_grid_map_frame
        robot_height_in_map = 0.0


        if self.use_tf_transformation:
            try:
                tf_base_to_map = self.tf_buffer.lookup_transform(
                    self.robot_base_frame, self.map_frame, rclpy.time.Time())


                trans = tf_base_to_map.transform.translation
                t_map_to_base = np.array([trans.x, trans.y, trans.z])
                rot = tf_base_to_map.transform.rotation
                R_map_to_base = ScipyRotation.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()


                points_map = np.stack([self.X_map.flatten(), self.Y_map.flatten(), Z_plot.flatten()], axis=1)
                points_body = (R_map_to_base @ points_map.T).T + t_map_to_base
                
                X_plot = points_body[:, 0].reshape(self.rows, self.cols)
                Y_plot = points_body[:, 1].reshape(self.rows, self.cols)
                Z_plot = points_body[:, 2].reshape(self.rows, self.cols)


                tf_map_to_base = self.tf_buffer.lookup_transform(self.map_frame, self.robot_base_frame, rclpy.time.Time())
                robot_height_in_map = tf_map_to_base.transform.translation.z


            except TransformException as ex:
                self.get_logger().warn(f'Could not perform TF transformation: {ex}', throttle_duration_sec=2)
                return


        # Crop the final grids for each subplot
        x_subgrid_3d, y_subgrid_3d, z_subgrid_3d = self._crop_grids(
            X_plot, Y_plot, Z_plot, self.plot_x_range_3d, self.plot_y_range_3d)
        
        x_subgrid_2d, y_subgrid_2d, z_subgrid_2d = self._crop_grids(
            X_plot, Y_plot, Z_plot, self.plot_x_range_2d, self.plot_y_range_2d)


        if x_subgrid_3d is None or x_subgrid_2d is None:
            self.get_logger().warn("Desired plot range is outside the current map view.", throttle_duration_sec=5)
            return
            
        # Update Plots
        self.ax_3d.clear()
        self.ax_2d.clear()
        
        cmap = cm.terrain
        vmin, vmax = -0.5, 0.5


        # 3D Surface Plot
        plot_Z_3d = np.nan_to_num(z_subgrid_3d, nan=0.0)
        self.ax_3d.plot_surface(y_subgrid_3d, x_subgrid_3d, plot_Z_3d, cmap=cmap, rstride=1, cstride=1, vmin=vmin, vmax=vmax)
        self.ax_3d.set_zlim(vmin, vmax)
        self.ax_3d.view_init(elev=40., azim=-120)
        self.ax_3d.xaxis.set_major_locator(MaxNLocator(10))
        self.ax_3d.yaxis.set_major_locator(MaxNLocator(10))


        # 2D Heatmap Plot
        x_min, x_max = x_subgrid_2d.min(), x_subgrid_2d.max()
        y_min, y_max = y_subgrid_2d.min(), y_subgrid_2d.max()
        
        # === FIX STARTS HERE ===
        # Use origin='lower' to make coordinate calculation intuitive (no np.flipud needed).
        im = self.ax_2d.imshow(z_subgrid_2d, cmap=cmap, vmin=vmin, vmax=vmax, 
                                extent=[y_min, y_max, x_min, x_max],
                                aspect='equal', interpolation='nearest', origin='lower')
        
        rows_2d, cols_2d = z_subgrid_2d.shape
        cell_height = (x_max - x_min) / rows_2d
        cell_width = (y_max - y_min) / cols_2d


        # Calculate text positions based on the straightened grid, not the rotated coordinate data.
        for r in range(rows_2d):
            for c in range(cols_2d):
                val = z_subgrid_2d[r, c]
                if not np.isnan(val):
                    # Calculate the center of the cell in the plot's coordinate system
                    x_pos = x_min + (r + 0.5) * cell_height
                    y_pos = y_min + (c + 0.5) * cell_width
                    self.ax_2d.text(y_pos, x_pos, f"{val:.2f}",
                                    ha="center", va="center", color="black", fontsize=8)


        # Set ticks and grid to align with cell boundaries
        x_ticks = np.linspace(x_min, x_max, rows_2d + 1)
        y_ticks = np.linspace(y_min, y_max, cols_2d + 1)
        self.ax_2d.set_xticks(y_ticks)
        self.ax_2d.set_yticks(x_ticks)
        self.ax_2d.grid(True, which='major', linestyle='--', linewidth=0.5, color='black')
        plt.setp(self.ax_2d.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        # === FIX ENDS HERE ===


        # Set plot titles and labels
        if self.use_tf_transformation:
            self.ax_3d.set_title("3D View (Robot Body Frame)")
            self.ax_3d.set_xlabel("Y-axis (lateral) [m]")
            self.ax_3d.set_ylabel("X-axis (forward) [m]")
            self.ax_3d.set_zlabel("Height relative to base [m]")
            self.ax_2d.set_title("2D Top-Down Heatmap (Robot Body Frame)")
            self.ax_2d.set_xlabel("Y-axis (lateral) [m]")
            self.ax_2d.set_ylabel("X-axis (forward) [m]")
        else:
            self.ax_3d.set_title(f"3D View ({self.map_frame} Frame)")
            self.ax_3d.set_xlabel("Y-axis (map) [m]")
            self.ax_3d.set_ylabel("X-axis (map) [m]")
            self.ax_3d.set_zlabel("Absolute Height [m]")
            self.ax_2d.set_title(f"2D Top-Down Heatmap ({self.map_frame} Frame)")
            self.ax_2d.set_xlabel("Y-axis (map) [m]")
            self.ax_2d.set_ylabel("X-axis (map) [m]")


        # Update Colorbar
        if self.cbar:
            self.cbar.update_normal(im)
        else:
            self.cbar = self.fig.colorbar(im, ax=[self.ax_2d, self.ax_3d], shrink=0.6, label='Height [m]')


        # Update statistics title
        min_h, max_h = np.nanmin(Z_plot), np.nanmax(Z_plot)
        if self.use_tf_transformation:
            stats_text = (f"Body Height (map frame): {robot_height_in_map:.3f} m\n"
                          f"Min/Max Relative Height (Full View): {min_h:.3f} m / {max_h:.3f} m")
        else:
            stats_text = f"Displaying in '{self.map_frame}' | Min/Max Absolute Height: {min_h:.3f} m / {max_h:.3f} m"
        self.fig.suptitle(stats_text, fontsize=14)
        
        # Redraw canvas
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()   


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
