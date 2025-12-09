import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import struct
import argparse
import sys
import os
from datetime import datetime
from datetime import timezone
class GridConfig:
    def __init__(self, width_points, height_points, resolution, sensor_off_x):
        self.nx = width_points
        self.ny = height_points
        self.res = resolution
        self.total_points = self.nx * self.ny
        self.sensor_off_x = sensor_off_x
        
        self.span_x = (self.nx - 1) * self.res
        self.span_y = (self.ny - 1) * self.res
        
        # Physical coordinates (Meters)
        self.x_centers = np.linspace(-self.span_x / 2.0, self.span_x / 2.0, self.nx)
        self.y_centers = np.linspace(-self.span_y / 2.0, self.span_y / 2.0, self.ny)
        
        # Extents for default imshow [left, right, bottom, top] (Meters)
        half_res = self.res / 2.0
        self.extent_default = [
            self.x_centers[0] - half_res,
            self.x_centers[-1] + half_res,
            self.y_centers[0] - half_res,
            self.y_centers[-1] + half_res
        ]

def load_data(filename):
    frames_data = []
    
    # File Header: Ver(B), Res(f), OffX(f), nx(H), ny(H), 67x(Pad) -> 80 Bytes
    file_header_struct = struct.Struct('< B f f H H 67x')
    
    # Frame Header: Time(d), ID(B), Valid(f), Pose(7f), 32x(Pad) -> 73 Bytes
    frame_header_struct = struct.Struct('< d B f 7f 32x')

    print(f"Loading data from {filename}...")

    try:
        with open(filename, 'rb') as f:
            file_header_bytes = f.read(file_header_struct.size)
            if len(file_header_bytes) < file_header_struct.size:
                print("Error: File too short for header.")
                return None, None

            (ver, res, off_x, nx, ny) = file_header_struct.unpack(file_header_bytes)
            config = GridConfig(nx, ny, res, off_x)
            # Calculate dynamic data size
            num_points = nx * ny
            grid_body_size = num_points * 4 # 4 bytes per float

            print(f"File Header: Ver={ver}, Grid={nx}x{ny}, Res={res:.3f}m")

            while True:
                frame_header_bytes = f.read(frame_header_struct.size)
                if len(frame_header_bytes) < frame_header_struct.size:
                    break # EOF
                
                (ts, lid, valid, 
                 rx, ry, rz, rqx, rqy, rqz, rqw) = frame_header_struct.unpack(frame_header_bytes)

                grid_bytes = f.read(grid_body_size)
                if len(grid_bytes) < grid_body_size:
                    print("Warning: Incomplete frame body at EOF.")
                    break

                flat_data = np.frombuffer(grid_bytes, dtype=np.float32)
                
                frames_data.append({
                    'timestamp': ts,
                    'layer_id': lid,
                    'validity': valid,
                    'pose': (rx, ry, rz),
                    'quat': (rqx, rqy, rqz, rqw),
                    'grid': flat_data.reshape(ny, nx)
                })

        print(f"First timestamp in unix seconds.nanoseconds={frames_data[0]['timestamp']}")
        print(f"Last timestamp in unix seconds.nanoseconds={frames_data[-1]['timestamp']}")

    except FileNotFoundError:
        print(f"Error: File not found at {filename}")
        return None, None
    
    if not frames_data:
        print("No data found.")
        return None, None
    
    print(f"Loaded {len(frames_data)} frames.")
    return frames_data, config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, default='policy_data.bin', nargs='?')
    args = parser.parse_args()

    np.set_printoptions(floatmode="fixed", precision=4, linewidth=1000, suppress=True)

    # Load data and config
    frames, cfg = load_data(args.filename)
    if frames is None: sys.exit(1)

    # State container
    state = {
        'swapped': False, 
        'unit': 'm', # 'm' or 'cm'
        'playing': False,
        'text_objs': [], 
        'im': None, 
        'cbar': None,
        'timer': None
    }
    
    fig, ax = plt.subplots(figsize=(10, 9))
    plt.subplots_adjust(bottom=0.15, top=0.88, left=0.1, right=0.9)

    # Calculate global min/max
    all_grids = np.stack([f['grid'] for f in frames])
    global_vmin_m = np.min(all_grids)
    global_vmax_m = np.max(all_grids)
    
    layer_names = {0: "Elevation", 1: "Min Filter", 2: "Smooth"}

    def get_plot_config(frame_idx):
        frame = frames[frame_idx]
        raw_data = frame['grid']
        
        if state['unit'] == 'cm':
            scale = 100.0
            unit_label = "cm"
            fmt_str = "{:.1f}" 
        else:
            scale = 1.0
            unit_label = "m"
            fmt_str = "{:.3f}" 

        curr_vmin = global_vmin_m * scale
        curr_vmax = global_vmax_m * scale
        
        if state['swapped']:
            data = np.flip(raw_data.T, axis=1) * scale
            screen_x_ticks = np.flip(cfg.y_centers) * scale
            screen_y_ticks = cfg.x_centers * scale
            extent = [
                cfg.extent_default[3] * scale, 
                cfg.extent_default[2] * scale,
                cfg.extent_default[0] * scale, 
                cfg.extent_default[1] * scale
            ]
            lbl_x = f"Y ({unit_label}, Left +, Right -)"
            lbl_y = f"X ({unit_label}, Forward +)"
            nx_s, ny_s = cfg.ny, cfg.nx
            btn_swap_txt = "View: Grid"
        else:
            data = raw_data * scale
            screen_x_ticks = cfg.x_centers * scale
            screen_y_ticks = cfg.y_centers * scale
            extent = [x * scale for x in cfg.extent_default]
            lbl_x = f"X ({unit_label}, body frame)"
            lbl_y = f"Y ({unit_label}, body frame)"
            nx_s, ny_s = cfg.nx, cfg.ny
            btn_swap_txt = "View: Ego"
            
        return {
            'data': data,
            'extent': extent,
            'x_ticks': screen_x_ticks,
            'y_ticks': screen_y_ticks,
            'lbl_x': lbl_x,
            'lbl_y': lbl_y,
            'nx': nx_s,
            'ny': ny_s,
            'vmin': curr_vmin,
            'vmax': curr_vmax,
            'unit': unit_label,
            'fmt': fmt_str,
            'swap_txt': btn_swap_txt,
            'meta': frame 
        }

    def setup_plot():
        if state['cbar']: 
            try: state['cbar'].remove()
            except: pass
            state['cbar'] = None

        ax.clear()
        
        cfg_plot = get_plot_config(int(slider.val))
        
        btn_swap.label.set_text(cfg_plot['swap_txt'])
        btn_unit.label.set_text(f"Unit: {cfg_plot['unit']}")

        state['im'] = ax.imshow(
            cfg_plot['data'], 
            vmin=cfg_plot['vmin'], 
            vmax=cfg_plot['vmax'], 
            extent=cfg_plot['extent'],
            origin='lower', interpolation='nearest', cmap='viridis'
        )

        ax.set_xticks(cfg_plot['x_ticks'])
        ax.set_yticks(cfg_plot['y_ticks'])
        
        # Minor grid lines
        ax.set_xticks(np.linspace(cfg_plot['extent'][0], cfg_plot['extent'][1], cfg_plot['nx'] + 1), minor=True)
        ax.set_yticks(np.linspace(cfg_plot['extent'][2], cfg_plot['extent'][3], cfg_plot['ny'] + 1), minor=True)
        ax.tick_params(which='major', length=0)
        ax.tick_params(which='minor', length=0)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

        if cfg_plot['nx'] > 15: ax.set_xticks(cfg_plot['x_ticks'][::2])
        if cfg_plot['ny'] > 15: ax.set_yticks(cfg_plot['y_ticks'][::2])

        ax.set_xlabel(cfg_plot['lbl_x'])
        ax.set_ylabel(cfg_plot['lbl_y'])

        state['text_objs'] = []
        for r in range(cfg_plot['ny']):
            row_objs = []
            for c in range(cfg_plot['nx']):
                t = ax.text(cfg_plot['x_ticks'][c], cfg_plot['y_ticks'][r], "", 
                            ha="center", va="center", fontsize=7, fontweight='bold')
                row_objs.append(t)
            state['text_objs'].append(row_objs)

        state['cbar'] = fig.colorbar(state['im'], ax=ax, fraction=0.046, pad=0.04)
        state['cbar'].set_label(f'Height ({cfg_plot["unit"]})')
        
        update(slider.val) 

    def update(val):
        idx = int(val)
        cfg_plot = get_plot_config(idx)
        
        state['im'].set_data(cfg_plot['data'])
        v_mid = (cfg_plot['vmin'] + cfg_plot['vmax']) / 2.0
        data = cfg_plot['data']
        rows, cols = data.shape
        
        for r in range(rows):
            for c in range(cols):
                val = data[r, c]
                txt = state['text_objs'][r][c]
                txt.set_text(cfg_plot['fmt'].format(val))
                txt.set_color("white" if val < v_mid else "black")
        
        meta = cfg_plot['meta']
        t_curr = meta['timestamp'] - frames[0]['timestamp']
        layer_name = layer_names.get(meta['layer_id'], "Unknown")
        rx, ry, rz = meta['pose']
        
        title_str = (
            f"Frame {idx} | T: {t_curr:.2f}s | Layer: {layer_name}\n"
            f"{datetime.strftime(datetime.fromtimestamp(meta['timestamp'],tz=timezone.utc), '%Y-%m-%dT%H-%M-%S.%f')} | Valid: {meta['validity']*100:.1f}% | Res: {cfg.res:.2f}m\n"
            f"Robot Pose: X={rx:.2f}, Y={ry:.2f}, Z={rz:.2f}"
        )
        ax.set_title(title_str, fontsize=10)
        fig.canvas.draw_idle()

    ax_slider = plt.axes([0.15, 0.08, 0.7, 0.03])
    slider = Slider(ax=ax_slider, label='Frame', valmin=0, valmax=len(frames) - 1, valinit=0, valstep=1)
    
    ax_prev = plt.axes([0.15, 0.02, 0.05, 0.04])
    btn_prev = Button(ax_prev, '<')
    ax_play = plt.axes([0.21, 0.02, 0.06, 0.04])
    btn_play = Button(ax_play, 'Play')
    ax_next = plt.axes([0.28, 0.02, 0.05, 0.04])
    btn_next = Button(ax_next, '>')
    ax_swap = plt.axes([0.65, 0.02, 0.12, 0.04])
    btn_swap = Button(ax_swap, 'View: Grid')
    ax_unit = plt.axes([0.78, 0.02, 0.12, 0.04])
    btn_unit = Button(ax_unit, 'Unit: m')

    def toggle_swap(event):
        state['swapped'] = not state['swapped']
        setup_plot()
    def toggle_unit(event):
        state['unit'] = 'cm' if state['unit'] == 'm' else 'm'
        setup_plot()
    def prev_frame(event): slider.set_val(max(0, slider.val - 1))
    def next_frame(event): slider.set_val(min(len(frames) - 1, slider.val + 1))
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
    btn_unit.on_clicked(toggle_unit)
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