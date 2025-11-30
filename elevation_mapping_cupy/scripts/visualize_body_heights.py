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
        
        # Physical coordinates
        self.x_centers = np.linspace(-self.span_x / 2.0, self.span_x / 2.0, self.nx)
        self.y_centers = np.linspace(-self.span_y / 2.0, self.span_y / 2.0, self.ny)
        
        # Extents for default imshow [left, right, bottom, top]
        # Default: X is horizontal, Y is vertical
        half_res = self.res / 2.0
        self.extent_default = [
            self.x_centers[0] - half_res,  # Left (X min)
            self.x_centers[-1] + half_res, # Right (X max)
            self.y_centers[0] - half_res,  # Bottom (Y min)
            self.y_centers[-1] + half_res  # Top (Y max)
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

    state = {
        'swapped': False, 
        'playing': False,
        'text_objs': [], 
        'im': None, 
        'cbar': None,
        'timer': None
    }
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9)

    all_data = np.stack(frames)
    vmin, vmax = np.min(all_data), np.max(all_data)
    v_mid = (vmin + vmax) / 2.0

    # --- Core Logic for Data/Grid orientation ---
    def get_plot_config(frame_idx):
        raw_data = frames[frame_idx]
        
        if state['swapped']:
            # EGO VIEW:
            # Vertical Axis = Robot X (Up is +)
            # Horizontal Axis = Robot Y (Left is +, Right is -)
            
            # 1. Transpose: X becomes rows (vertical), Y becomes cols (horizontal)
            # 2. Flip Cols: Visual Left becomes +Y, Visual Right becomes -Y
            data = np.flip(raw_data.T, axis=1)
            
            # Ticks/Coords for Screen Horizontal Axis (Y coordinates)
            # Since we flipped the data columns, we flip the coordinate array too
            # Screen Left = Y Max, Screen Right = Y Min
            screen_x_ticks = np.flip(cfg.y_centers)
            screen_y_ticks = cfg.x_centers
            
            # Extent [left, right, bottom, top]
            # Left = +Y limit, Right = -Y limit
            extent = [
                cfg.extent_default[3], # Y Top (+Y) -> Screen Left
                cfg.extent_default[2], # Y Bottom (-Y) -> Screen Right
                cfg.extent_default[0], # X Left (-X) -> Screen Bottom
                cfg.extent_default[1]  # X Right (+X) -> Screen Top
            ]
            
            lbl_x = "Y (m, Left +, Right -)"
            lbl_y = "X (m, Forward +)"
            nx_screen, ny_screen = cfg.ny, cfg.nx
            btn_text = "Switch to Grid View"
            
        else:
            # GRID VIEW (Default):
            # Vertical Axis = Y
            # Horizontal Axis = X
            data = raw_data
            
            screen_x_ticks = cfg.x_centers
            screen_y_ticks = cfg.y_centers
            extent = cfg.extent_default
            
            lbl_x = "X (m, body frame)"
            lbl_y = "Y (m, body frame)"
            nx_screen, ny_screen = cfg.nx, cfg.ny
            btn_text = "Switch to Ego View"
            
        return data, extent, screen_x_ticks, screen_y_ticks, lbl_x, lbl_y, nx_screen, ny_screen, btn_text

    def setup_plot():
        # Clean up previous elements
        if state['cbar']: 
            try: state['cbar'].remove()
            except: pass
            state['cbar'] = None
        ax.clear()
        
        # Get configuration for current state
        data, extent, x_ticks, y_ticks, lbl_x, lbl_y, nx_s, ny_s, btn_txt = get_plot_config(int(slider.val))
        
        btn_swap.label.set_text(btn_txt)

        # Plot Image
        state['im'] = ax.imshow(data, vmin=vmin, vmax=vmax, extent=extent,
                                origin='lower', interpolation='nearest', cmap='viridis')

        # Configure Ticks
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        
        # Configure Grid (Minor ticks at edges)
        ax.set_xticks(np.linspace(extent[0], extent[1], nx_s + 1), minor=True)
        ax.set_yticks(np.linspace(extent[2], extent[3], ny_s + 1), minor=True)
        
        ax.tick_params(which='major', length=0)
        ax.tick_params(which='minor', length=0)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

        # Decimate ticks if dense
        if nx_s > 15: ax.set_xticks(x_ticks[::2])
        if ny_s > 15: ax.set_yticks(y_ticks[::2])

        ax.set_xlabel(lbl_x)
        ax.set_ylabel(lbl_y)

        # Setup Text Annotations
        # We iterate over SCREEN coordinates (ny_s rows, nx_s cols)
        state['text_objs'] = []
        for r in range(ny_s):
            row_objs = []
            for c in range(nx_s):
                # Place text at the physical coordinate of this pixel center
                t = ax.text(x_ticks[c], y_ticks[r], "", 
                            ha="center", va="center", fontsize=7, fontweight='bold')
                row_objs.append(t)
            state['text_objs'].append(row_objs)

        state['cbar'] = fig.colorbar(state['im'], ax=ax, fraction=0.046, pad=0.04)
        state['cbar'].set_label('Height (m)')
        
        update(slider.val) 

    def update(val):
        idx = int(val)
        
        # Retrieve strictly formatted data for display
        data, _, _, _, _, _, _, _, _ = get_plot_config(idx)
        state['im'].set_data(data)
        
        rows, cols = data.shape
        
        # Update text values using the EXACT data array being displayed
        for r in range(rows):
            for c in range(cols):
                val = data[r, c]
                txt = state['text_objs'][r][c]
                txt.set_text(f"{val:.3f}")
                txt.set_color("white" if val < v_mid else "black")
        
        t_curr = timestamps[idx] - timestamps[0]
        ax.set_title(f"Source: {args.source.title()} | Frame {idx} | T: {t_curr:.2f}s")
        fig.canvas.draw_idle()

    # --- UI Components ---
    ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
    slider = Slider(ax=ax_slider, label='Frame', valmin=0, valmax=len(frames) - 1, valinit=0, valstep=1)
    
    ax_prev = plt.axes([0.15, 0.04, 0.05, 0.04])
    btn_prev = Button(ax_prev, '<')

    ax_play = plt.axes([0.21, 0.04, 0.06, 0.04])
    btn_play = Button(ax_play, 'Play')

    ax_next = plt.axes([0.28, 0.04, 0.05, 0.04])
    btn_next = Button(ax_next, '>')

    ax_swap = plt.axes([0.65, 0.04, 0.2, 0.04])
    btn_swap = Button(ax_swap, 'Switch to Ego View')

    def toggle_swap(event):
        state['swapped'] = not state['swapped']
        setup_plot()
    
    def prev_frame(event):
        slider.set_val(max(0, slider.val - 1))

    def next_frame(event):
        slider.set_val(min(len(frames) - 1, slider.val + 1))

    def toggle_play(event):
        state['playing'] = not state['playing']
        if state['playing']:
            btn_play.label.set_text('Pause')
            state['timer'].start()
        else:
            btn_play.label.set_text('Play')
            state['timer'].stop()

    def on_timer():
        if state['playing']:
            new_val = slider.val + 1
            if new_val >= len(frames): new_val = 0
            slider.set_val(new_val)

    btn_swap.on_clicked(toggle_swap)
    btn_prev.on_clicked(prev_frame)
    btn_next.on_clicked(next_frame)
    btn_play.on_clicked(toggle_play)
    slider.on_changed(update)

    state['timer'] = fig.canvas.new_timer(interval=100)
    state['timer'].add_callback(on_timer)

    setup_plot()
    print("Visualization started.")
    plt.show()

if __name__ == '__main__':
    main()