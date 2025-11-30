import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import struct
import argparse
import sys

class GridConfig:
    def __init__(self, width_points, height_points, resolution):
        self.nx = width_points
        self.ny = height_points
        self.res = resolution
        self.total_points = self.nx * self.ny
        
        # Calculate physical dimensions based on points and resolution
        # The span is the distance between the first and last point center
        self.span_x = (self.nx - 1) * self.res
        self.span_y = (self.ny - 1) * self.res
        
        # Center coordinates (where the data lives)
        self.x_centers = np.linspace(-self.span_x / 2.0, self.span_x / 2.0, self.nx)
        self.y_centers = np.linspace(-self.span_y / 2.0, self.span_y / 2.0, self.ny)
        
        # Extents for imshow (The physical edges of the grid)
        # We extend half a resolution outward from the centers
        half_res = self.res / 2.0
        self.extent = [
            self.x_centers[0] - half_res,  # Left
            self.x_centers[-1] + half_res, # Right
            self.y_centers[0] - half_res,  # Bottom
            self.y_centers[-1] + half_res  # Top
        ]

def load_data(filename, source_type, config: GridConfig):
    """
    Loads data dynamically based on grid configuration.
    """
    policy_shape = (config.ny, config.nx) # Rows=Y, Cols=X
    data_type_to_load = 1 if source_type == 'filtered' else 0

    # Dynamic struct format: 
    # 'd' (double timestamp), 'B' (byte type), '{N}f' (N floats)
    fmt_str = f'd B {config.total_points}f'
    try:
        record_format = struct.Struct(fmt_str)
    except Exception as e:
        print(f"Error creating struct format: {e}")
        return None, None
        
    record_size = record_format.size
    timestamps = []
    frames = []

    print(f"Loading '{source_type}' data from {filename}...")
    print(f"Expect: {config.nx}x{config.ny} grid ({config.total_points} floats per frame)")

    try:
        with open(filename, 'rb') as f:
            while True:
                buffer = f.read(record_size)
                if len(buffer) < record_size:
                    break 
                
                unpacked = record_format.unpack(buffer)
                timestamp = unpacked[0]
                data_type = unpacked[1]
                
                if data_type == data_type_to_load:
                    timestamps.append(timestamp)
                    # Slice from index 2 to end
                    flat_data = np.array(unpacked[2:], dtype=np.float32)
                    try:
                        frames.append(flat_data.reshape(policy_shape))
                    except ValueError:
                        print(f"Error: Data mismatch. Binary has {len(flat_data)} floats, expected {config.total_points}.")
                        return None, None

    except FileNotFoundError:
        print(f"Error: File not found at {filename}")
        return None, None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None
    
    if not frames:
        print(f"No data of type '{source_type}' found.")
        return None, None
    
    print(f"Loaded {len(frames)} frames.")
    return timestamps, frames

def main():
    parser = argparse.ArgumentParser(description="Visualize policy height map data.")
    parser.add_argument('filename', type=str, default='policy_data.bin', nargs='?', help='Path to binary file')
    parser.add_argument('--source', type=str, default='filtered', choices=['raw', 'filtered'], help="Data source")
    parser.add_argument('--width', type=int, default=13, help='Number of points in X (default: 13)')
    parser.add_argument('--height', type=int, default=11, help='Number of points in Y (default: 11)')
    parser.add_argument('--res', type=float, default=0.08, help='Grid resolution in meters (default: 0.08)')
    args = parser.parse_args()

    np.set_printoptions(floatmode="fixed", precision=4, linewidth=1000,suppress=True) # For consistent printouts

    cfg = GridConfig(args.width, args.height, args.res)
    timestamps, frames = load_data(args.filename, args.source, cfg)
    if timestamps is None:
        sys.exit(1)

    # We use constrained_layout to handle labels better than tight_layout
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=False)
    plt.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9)

    all_data = np.stack(frames)
    vmin, vmax = np.min(all_data), np.max(all_data)
    v_mid = (vmin + vmax) / 2.0
    
    # origin='lower' is critical to match coordinate systems (Y+ is Up)
    im = ax.imshow(
        frames[0],
        vmin=vmin, 
        vmax=vmax,
        extent=cfg.extent,
        origin='lower',
        interpolation='nearest',
        cmap='viridis'
    )
    
    # We want borders at the EDGES of the pixels, not the centers.
    # The edges are exactly half_res offset from the centers.
    x_edges = np.linspace(cfg.extent[0], cfg.extent[1], cfg.nx + 1)
    y_edges = np.linspace(cfg.extent[2], cfg.extent[3], cfg.ny + 1)
    
    # Set ticks: Major for coordinates (centers), Minor for grid lines (edges)
    ax.set_xticks(cfg.x_centers)
    ax.set_yticks(cfg.y_centers)
    
    ax.set_xticks(x_edges, minor=True)
    ax.set_yticks(y_edges, minor=True)
    
    # Style: Hide major tick marks (just keep labels), show minor grid lines
    ax.tick_params(which='major', length=0) 
    ax.tick_params(which='minor', length=0)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    
    # Simplify labels (optional: reduce clutter if grid is huge)
    if cfg.nx > 15:
        ax.set_xticks(cfg.x_centers[::2]) # Show every 2nd label
    if cfg.ny > 15:
        ax.set_yticks(cfg.y_centers[::2])

    ax.set_xlabel("X (m, body frame)")
    ax.set_ylabel("Y (m, body frame)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Height (m)')
    
    start_time = timestamps[0]
    title = ax.set_title(f"Source: {args.source.title()} | Frame 0 | T: 0.00s")

    text_annotations = []
    for y_idx in range(cfg.ny):
        row_texts = []
        for x_idx in range(cfg.nx):
            val = frames[0][y_idx, x_idx]
            color = "white" if val < v_mid else "black"
            
            t = ax.text(
                cfg.x_centers[x_idx],
                cfg.y_centers[y_idx],
                f"{val:.3f}",
                ha="center", 
                va="center",
                color=color,
                fontsize=7,
                fontweight='bold'
            )
            row_texts.append(t)
        text_annotations.append(row_texts)

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(
        ax=ax_slider,
        label='Frame',
        valmin=0,
        valmax=len(frames) - 1,
        valinit=0,
        valstep=1
    )

    def update(val):
        idx = int(slider.val)
        data = frames[idx]
        im.set_data(data)
        for y_i in range(cfg.ny):
            for x_i in range(cfg.nx):
                val = data[y_i, x_i]
                print(data)
                txt_obj = text_annotations[y_i][x_i]
                txt_obj.set_text(f"{val:.3f}")
                txt_obj.set_color("white" if val < v_mid else "black")
        
        t_curr = timestamps[idx] - start_time
        title.set_text(f"Source: {args.source.title()} | Frame {idx} | T: {t_curr:.2f}s")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    print("Visualization started. Close window to exit.")
    plt.show()

if __name__ == '__main__':
    main()