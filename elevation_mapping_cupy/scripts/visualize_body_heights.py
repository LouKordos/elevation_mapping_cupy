import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import struct
import argparse

def load_data(filename, source_type):
    """
    Loads data from the binary log file.
    """
    # 13x11 grid -> 143 points
    policy_shape = (11, 13) # 11 rows (y), 13 columns (x)
    data_type_to_load = 1 if source_type == 'filtered' else 0

    # 'd' = float64 (timestamp), 'B' = uint8 (type), '143f' = 143x float32 (data)
    record_format = struct.Struct('d B 143f')
    record_size = record_format.size

    timestamps = []
    frames = []

    print(f"Loading '{source_type}' (type {data_type_to_load}) data from {filename}...")
    try:
        with open(filename, 'rb') as f:
            while True:
                buffer = f.read(record_size)
                if len(buffer) < record_size:
                    break # End of file or incomplete record
                
                # Unpack the data
                unpacked = record_format.unpack(buffer)
                timestamp = unpacked[0]
                data_type = unpacked[1]
                
                if data_type == data_type_to_load:
                    timestamps.append(timestamp)
                    # The rest of the tuple (unpacked[2:]) are the 143 floats
                    flat_data = np.array(unpacked[2:], dtype=np.float32)
                    frames.append(flat_data.reshape(policy_shape))

    except FileNotFoundError:
        print(f"Error: File not found at {filename}")
        return None, None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None
    
    if not frames:
        print(f"No data of type '{source_type}' found in file.")
        return None, None
    
    print(f"Loaded {len(frames)} frames.")
    return timestamps, frames

def main():
    parser = argparse.ArgumentParser(description="Visualize policy height map data from a binary log.")
    parser.add_argument(
        'filename',
        type=str,
        default='policy_data.bin',
        nargs='?',
        help='Path to the policy_data.bin file (default: policy_data.bin)'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='filtered',
        choices=['raw', 'filtered'],
        help="Data source to display: 'raw' (type 0) or 'filtered' (type 1) (default: filtered)"
    )
    args = parser.parse_args()

    timestamps, frames = load_data(args.filename, args.source)
    if timestamps is None:
        return

    fig, ax = plt.subplots(figsize=(12.8, 9.6))
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.25) # Make room for the slider

    all_data = np.stack(frames)
    vmin = np.min(all_data)
    vmax = np.max(all_data)
    v_mid = (vmin + vmax) / 2.0 # For text color logic
    
    # Grid dimensions from original script:
    # x: -0.48 to +0.48 (13 points)
    # y: -0.40 to +0.40 (11 points)
    # imshow extent is (left, right, bottom, top)
    extent = [-0.48, 0.48, -0.40, 0.40]

    # Display the first frame
    im = ax.imshow(
        frames[0],
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        origin='lower', # Puts -0.40 at the bottom
        interpolation='nearest',
        cmap='viridis' # Use a common colormap
    )
    
    ax.set_xlabel("X (m, body frame)")
    ax.set_ylabel("Y (m, body frame)")
    fig.colorbar(im, ax=ax, label='Height (m)')
    
    # Add a dynamic title
    start_time = timestamps[0]
    title = ax.set_title(f"Source: {args.source.title()} | Frame 0 | Time: {timestamps[0] - start_time:.2f} s")

    # Define the grid cell center coordinates
    x_points = 13
    y_points = 11
    x_coords = np.linspace(-0.48, 0.48, x_points)
    y_coords = np.linspace(-0.40, 0.40, y_points)
    
    # Create a list of lists to hold the text objects
    text_annotations = []
    for y_idx in range(y_points):
        row_texts = []
        for x_idx in range(x_points):
            value = frames[0][y_idx, x_idx]
            # Set text color based on value for readability
            color = "white" if value < v_mid else "black"
            text = ax.text(
                x_coords[x_idx],
                y_coords[y_idx],
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8
            )
            row_texts.append(text)
        text_annotations.append(row_texts)

    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03]) # [left, bottom, width, height]
    slider = Slider(
        ax=ax_slider,
        label='Frame',
        valmin=0,
        valmax=len(frames) - 1,
        valinit=0,
        valstep=1 # Integer steps
    )

    def update(val):
        frame_index = int(slider.val)
        current_frame = frames[frame_index]
        im.set_data(current_frame)
        for y_idx in range(y_points):
            for x_idx in range(x_points):
                value = current_frame[y_idx, x_idx]
                text_annotations[y_idx][x_idx].set_text(f"{value:.2f}")
                color = "white" if value < v_mid else "black"
                text_annotations[y_idx][x_idx].set_color(color)
        
        timestamp = timestamps[frame_index]
        title.set_text(f"Source: {args.source.title()} | Frame {frame_index} | Time: {timestamp - start_time:.2f} s")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

if __name__ == '__main__':
    main()