import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import argparse
import sys
import os
import copy
import json
from datetime import datetime
from datetime import timezone

class GridConfig:
    def __init__(self, width_points, height_points, resolution, sensor_off_x, fill_value):
        self.nx = width_points
        self.ny = height_points
        self.res = resolution
        self.total_points = self.nx * self.ny
        self.sensor_off_x = sensor_off_x
        self.fill_value = fill_value
        
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
    
    # Map string layer names back to IDs for the visualizer colors/legend
    layer_map = {
        "elevation": 0,
        "min_filter": 1,
        "min_filter_rel": 1, # C++ uses this name
        "smooth": 2,
        "smooth_rel": 2
    }

    print(f"Loading data from {filename}...")

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            try:
                header_line = f.readline()
                if not header_line:
                    print("Error: File is empty.")
                    return None, None
                
                meta = json.loads(header_line)
                if meta.get("type") != "metadata":
                    print("Error: First line is not a metadata object.")
                    return None, None
                
                cfg_data = meta["config"]
                # Parse config
                res = cfg_data["resolution"]
                off_x = cfg_data["sensor_offset_x"]
                nx = int(cfg_data["num_x"])
                ny = int(cfg_data["num_y"])
                fill_val = cfg_data["fill_value"]
                ver = meta.get("version", 0)
                
                config = GridConfig(nx, ny, res, off_x, fill_val)
                print(f"File Header: Ver={ver}, Grid={nx}x{ny}, Res={res:.3f}m, Fill={fill_val:.3f}")
                
            except json.JSONDecodeError as e:
                print(f"Error decoding metadata header: {e}")
                return None, None

            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line: continue
                
                try:
                    frame = json.loads(line)
                    
                    # Basic Fields
                    ts = frame.get("ts", 0.0)
                    layer_str = frame.get("layer", "unknown")
                    lid = layer_map.get(layer_str, 255) # Default to 255 if unknown
                    valid = frame.get("valid", 0.0)
                    
                    # Pose (Object to Tuple)
                    p = frame.get("pose", {})
                    rx, ry, rz = p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)
                    rqx, rqy, rqz, rqw = p.get("qx", 0.0), p.get("qy", 0.0), p.get("qz", 0.0), p.get("qw", 1.0)
                    
                    # Feet (List to List of Tuples)
                    feet_raw = frame.get("feet")
                    feet_coords = []
                    
                    if feet_raw and len(feet_raw) >= 12:
                        for i in range(0, 12, 3):
                            fx = feet_raw[i]
                            fy = feet_raw[i+1]
                            # Handle JSON 'null' which becomes None in Python
                            if fx is None: fx = np.nan
                            if fy is None: fy = np.nan
                            feet_coords.append((fx, fy))
                    else:
                        # Fallback for empty/null feet
                        feet_coords = [(np.nan, np.nan)] * 4

                    # Grid (List to Numpy)
                    grid_list = frame.get("grid", [])
                    if len(grid_list) != nx * ny:
                        print(f"Warning: Frame at line {line_idx+2} has wrong grid size ({len(grid_list)}). Skipping.")
                        continue
                    
                    flat_data = np.array(grid_list, dtype=np.float32)

                    frames_data.append({
                        'timestamp': ts,
                        'layer_id': lid,
                        'validity': valid,
                        'pose': (rx, ry, rz),
                        'quat': (rqx, rqy, rqz, rqw),
                        'feet': feet_coords, 
                        'grid': flat_data.reshape(ny, nx)
                    })
                except json.JSONDecodeError:
                    print(f"Warning: Malformed JSON at line {line_idx+2}")
                    continue

    except FileNotFoundError:
        print(f"Error: File not found at {filename}")
        return None, None
    
    if not frames_data:
        print("No data found.")
        return None, None
    
    print(f"First timestamp in unix seconds.nanoseconds={frames_data[0]['timestamp']}")
    print(f"Last timestamp in unix seconds.nanoseconds={frames_data[-1]['timestamp']}")
    print(f"Loaded {len(frames_data)} frames.")
    return frames_data, config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, default='policy_data.jsonl', nargs='?')
    args = parser.parse_args()

    np.set_printoptions(floatmode="fixed", precision=4, linewidth=1000, suppress=True)

    frames, cfg = load_data(args.filename)
    if frames is None: sys.exit(1)

    state = {
        'swapped': False, 
        'unit': 'm', # 'm' or 'cm'
        'playing': False,
        'text_objs': [], 
        'im': None, 
        'cbar': None,
        'timer': None,
        'feet_scatter': None 
    }
    
    fig, ax = plt.subplots(figsize=(10, 9))
    # Adjusted right margin to make room for legend outside
    plt.subplots_adjust(bottom=0.15, top=0.88, left=0.1, right=0.80)

    all_grids = np.stack([f['grid'] for f in frames])
    masked_all = np.ma.masked_values(all_grids, cfg.fill_value, rtol=1e-5)
    
    if masked_all.count() > 0:
        global_vmin_m = np.min(masked_all)
        global_vmax_m = np.max(masked_all)
    else:
        global_vmin_m, global_vmax_m = -1.0, 1.0 
    
    layer_names = {0: "Elevation", 1: "Min Filter", 2: "Smooth", 255: "Unknown"}

    cmap = copy.copy(plt.cm.viridis)
    cmap.set_bad(color='#FF1493') # Deep Pink for fill value

    def get_plot_config(frame_idx):
        frame = frames[frame_idx]
        raw_data = frame['grid']
        raw_feet = frame['feet'] 
        
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
        
        is_fill = np.isclose(raw_data, cfg.fill_value, atol=1e-5)
        masked_data = np.ma.masked_where(is_fill, raw_data)

        feet_plot_x = []
        feet_plot_y = []

        if state['swapped']:
            data = np.flip(masked_data.T, axis=1) * scale
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

            # Plot x-axis is World Y. Plot y-axis is World X.
            for (fx, fy) in raw_feet:
                feet_plot_x.append(fy * scale) 
                feet_plot_y.append(fx * scale) 
        else:
            data = masked_data * scale
            screen_x_ticks = cfg.x_centers * scale
            screen_y_ticks = cfg.y_centers * scale
            extent = [x * scale for x in cfg.extent_default]
            lbl_x = f"X ({unit_label}, body frame)"
            lbl_y = f"Y ({unit_label}, body frame)"
            nx_s, ny_s = cfg.nx, cfg.ny
            btn_swap_txt = "View: Ego"

            for (fx, fy) in raw_feet:
                feet_plot_x.append(fx * scale)
                feet_plot_y.append(fy * scale)
            
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
            'feet_x': feet_plot_x,
            'feet_y': feet_plot_y,
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
            origin='lower', interpolation='nearest', cmap=cmap
        )

        state['feet_scatter'] = ax.scatter(
            cfg_plot['feet_x'], cfg_plot['feet_y'], 
            c='white', edgecolors='black', s=100, label='Feet', zorder=10
        )
        
        # Legend moved outside the plot
        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0.)

        ax.set_xticks(cfg_plot['x_ticks'])
        ax.set_yticks(cfg_plot['y_ticks'])
        
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
        
        state['feet_scatter'].set_offsets(np.c_[cfg_plot['feet_x'], cfg_plot['feet_y']])

        v_mid = (cfg_plot['vmin'] + cfg_plot['vmax']) / 2.0
        data = cfg_plot['data'] 
        rows, cols = data.shape
        
        for r in range(rows):
            for c in range(cols):
                val_masked = data[r, c]
                txt = state['text_objs'][r][c]
                
                if np.ma.is_masked(val_masked):
                    # txt.set_text("N/A")
                    txt.set_text(cfg_plot['fmt'].format(val_masked))
                    txt.set_color("black")
                else:
                    txt.set_text(cfg_plot['fmt'].format(val_masked))
                    txt.set_color("white" if val_masked < v_mid else "black")
        
        meta = cfg_plot['meta']
        t_curr = meta['timestamp'] - frames[0]['timestamp']
        layer_name = layer_names.get(meta['layer_id'], "Unknown")
        rx, ry, rz = meta['pose']
        
        title_str = (
            f"Frame {idx} | T: {t_curr:.2f}s | Layer: {layer_name}\n"
            f"{datetime.strftime(datetime.fromtimestamp(meta['timestamp'],tz=timezone.utc), '%Y-%m-%dT%H-%M-%S.%f')} | Valid: {meta['validity']*100:.1f}% | Res: {cfg.res:.2f}m\n"
            f"Robot Pose: X={rx:.2f}, Y={ry:.2f}, Z={rz:.2f} | Fill Val (Pink): {cfg.fill_value:.2f}"
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