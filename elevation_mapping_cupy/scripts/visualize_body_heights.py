import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import struct
import argparse
import sys

class GridConfig:
    def __init__(self, width_points, height_points, resolution):
        self.nx = width_points
        self.ny = height_points
        self.res = resolution
        self.total_points = self.nx * self.ny
        
        self.span_x = (self.nx - 1) * self.res
        self.span_y = (self.ny - 1) * self.res
        
        self.x_centers = np.linspace(-self.span_x / 2.0, self.span_x / 2.0, self.nx)
        self.y_centers = np.linspace(-self.span_y / 2.0, self.span_y / 2.0, self.ny)
        
        half_res = self.res / 2.0
        self.extent = [
            self.x_centers[0] - half_res,
            self.x_centers[-1] + half_res,
            self.y_centers[0] - half_res,
            self.y_centers[-1] + half_res
        ]

def load_data(filename, source_type, config: GridConfig):
    policy_shape = (config.ny, config.nx)
    data_type_to_load = 1 if source_type == 'filtered' else 0

    try:
        record_format = struct.Struct(f'd B {config.total_points}f')
    except Exception as e:
        print(f"Error creating struct format: {e}")
        return None, None
        
    timestamps = []
    frames = []

    print(f"Loading '{source_type}' data from {filename}...")

    try:
        with open(filename, 'rb') as f:
            while True:
                buffer = f.read(record_format.size)
                if len(buffer) < record_format.size: break 
                
                unpacked = record_format.unpack(buffer)
                if unpacked[1] == data_type_to_load:
                    timestamps.append(unpacked[0])
                    flat_data = np.array(unpacked[2:], dtype=np.float32)
                    frames.append(flat_data.reshape(policy_shape))

    except FileNotFoundError:
        print(f"Error: File not found at {filename}")
        return None, None
    
    if not frames:
        print("No data found.")
        return None, None
    
    print(f"Loaded {len(frames)} frames.")
    return timestamps, frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, default='policy_data.bin', nargs='?')
    parser.add_argument('--source', type=str, default='filtered', choices=['raw', 'filtered'])
    parser.add_argument('--width', type=int, default=13)
    parser.add_argument('--height', type=int, default=11)
    parser.add_argument('--res', type=float, default=0.08)
    args = parser.parse_args()

    np.set_printoptions(floatmode="fixed", precision=4, linewidth=1000, suppress=True)

    cfg = GridConfig(args.width, args.height, args.res)
    timestamps, frames = load_data(args.filename, args.source, cfg)
    if timestamps is None: sys.exit(1)

    state = {'swapped': False, 'text_objs': [], 'im': None, 'cbar': None}
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9)

    all_data = np.stack(frames)
    vmin, vmax = np.min(all_data), np.max(all_data)
    v_mid = (vmin + vmax) / 2.0

    def setup_plot():
        ax.clear()
        
        if state['swapped']:
            nx, ny = cfg.ny, cfg.nx
            x_ticks, y_ticks = cfg.y_centers, cfg.x_centers
            extent = [cfg.extent[2], cfg.extent[3], cfg.extent[0], cfg.extent[1]]
            lbl_x, lbl_y = "Y (m, body frame)", "X (m, body frame)"
            frame_data = frames[int(slider.val)].T
        else:
            nx, ny = cfg.nx, cfg.ny
            x_ticks, y_ticks = cfg.x_centers, cfg.y_centers
            extent = cfg.extent
            lbl_x, lbl_y = "X (m, body frame)", "Y (m, body frame)"
            frame_data = frames[int(slider.val)]

        state['im'] = ax.imshow(frame_data, vmin=vmin, vmax=vmax, extent=extent,
                                origin='lower', interpolation='nearest', cmap='viridis')

        # Set ticks (centers) and grid (edges)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticks(np.linspace(extent[0], extent[1], nx + 1), minor=True)
        ax.set_yticks(np.linspace(extent[2], extent[3], ny + 1), minor=True)
        
        ax.tick_params(which='major', length=0)
        ax.tick_params(which='minor', length=0)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

        if nx > 15: ax.set_xticks(x_ticks[::2])
        if ny > 15: ax.set_yticks(y_ticks[::2])

        ax.set_xlabel(lbl_x)
        ax.set_ylabel(lbl_y)

        state['text_objs'] = []
        for y_idx in range(ny):
            row = []
            for x_idx in range(nx):
                t = ax.text(x_ticks[x_idx], y_ticks[y_idx], "", ha="center", va="center", 
                            fontsize=7, fontweight='bold')
                row.append(t)
            state['text_objs'].append(row)

        if state['cbar']: state['cbar'].remove()
        state['cbar'] = fig.colorbar(state['im'], ax=ax, fraction=0.046, pad=0.04)
        state['cbar'].set_label('Height (m)')
        
        update(slider.val)

    def update(val):
        idx = int(val)
        data = frames[idx].T if state['swapped'] else frames[idx]
        
        state['im'].set_data(data)
        
        ny, nx = data.shape
        for y in range(ny):
            for x in range(nx):
                val = data[y, x]
                txt = state['text_objs'][y][x]
                txt.set_text(f"{val:.3f}")
                txt.set_color("white" if val < v_mid else "black")
        
        t_curr = timestamps[idx] - timestamps[0]
        ax.set_title(f"Source: {args.source.title()} | Frame {idx} | T: {t_curr:.2f}s")
        fig.canvas.draw_idle()

    ax_slider = plt.axes([0.2, 0.05, 0.5, 0.03])
    slider = Slider(ax=ax_slider, label='Frame', valmin=0, valmax=len(frames) - 1, valinit=0, valstep=1)
    slider.on_changed(update)

    ax_btn = plt.axes([0.8, 0.05, 0.1, 0.04])
    btn = Button(ax_btn, 'Swap X/Y')
    
    def toggle_swap(event):
        state['swapped'] = not state['swapped']
        setup_plot()
    
    btn.on_clicked(toggle_swap)

    setup_plot()
    print("Visualization started.")
    plt.show()

if __name__ == '__main__':
    main()