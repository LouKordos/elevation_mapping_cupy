#!/usr/bin/env python3
import struct
import os
import argparse
import numpy as np
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

# File Header: Version(B), Res(f), OffX(f), nx(H), ny(H), Reserved(67x) -> 80 Bytes
FILE_HEADER_FMT = struct.Struct("< B f f H H 67x")
# Frame: Time(d), ID(B), Valid(f), Pose(7f), Reserved(32x) -> 73 Bytes
FRAME_META_FMT = struct.Struct("< d B f 7f 32x")

@dataclass
class GridMapConfiguration:
    version: int
    resolution: float
    sensor_offset_x: float
    num_x: int
    num_y: int

@dataclass
class FramePayload:
    timestamp: float
    layer_id: int
    validity_ratio: float
    robot_pose: Tuple[float, ...]
    grid_data: np.ndarray

class ElevationLogReader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.config: Optional[GridMapConfiguration] = None
        self.frames: Dict[float, FramePayload] = {} 

    def load(self, t_start: Optional[float] = None, t_end: Optional[float] = None):
        if not os.path.exists(self.filepath):
            print(f"[Error] File not found: {self.filepath}")
            sys.exit(1)

        file_size = os.path.getsize(self.filepath)
        print(f"[*] Loading: {self.filename} ({file_size / 1024:.2f} KB)...")

        with open(self.filepath, "rb") as f:
            header_bytes = f.read(FILE_HEADER_FMT.size)
            if len(header_bytes) < FILE_HEADER_FMT.size:
                print(f"[Error] File {self.filename} too short for header.")
                sys.exit(1)

            (ver, res, off_x, nx, ny) = FILE_HEADER_FMT.unpack(header_bytes)
            self.config = GridMapConfiguration(ver, res, off_x, nx, ny)
            
            grid_byte_size = nx * ny * 4
            
            while True:
                meta_bytes = f.read(FRAME_META_FMT.size)
                if len(meta_bytes) < FRAME_META_FMT.size:
                    break 

                unpacked = FRAME_META_FMT.unpack(meta_bytes)
                timestamp = unpacked[0]
                
                # We must read the grid bytes to advance the file pointer, 
                # even if we discard the frame based on the time window.
                grid_bytes = f.read(grid_byte_size)
                if len(grid_bytes) < grid_byte_size:
                    break

                if t_start is not None and timestamp < t_start:
                    continue
                if t_end is not None and timestamp > t_end:
                    continue

                grid_array = np.frombuffer(grid_bytes, dtype=np.float32)
                
                # Reshaping matches the generating script's column-major construction 
                # (vstack of ravels). If visual output is wrong, this reshape order is the suspect.
                try:
                    grid_array = grid_array.reshape((ny, nx))
                except ValueError:
                    sys.exit(f"[!] Shape mismatch: Header {ny}x{nx} vs Data {grid_array.size}")

                self.frames[timestamp] = FramePayload(
                    timestamp, unpacked[1], unpacked[2], unpacked[3:10], grid_array
                )

            print(f"    -> Loaded {len(self.frames)} frames.")

def visualize_divergence(diff_grid: np.ndarray, threshold: float, resolution: float):
    # Transformation to match Body Frame: +X Up, +Y Left
    # Input is (Rows=Y, Cols=X). 
    # Transpose -> (Rows=X, Cols=Y). 
    # FlipUD -> +X moves to top row. 
    # FlipLR -> +Y moves to left column.
    rotated_grid = np.flipud(np.fliplr(diff_grid.T))
    
    rows, cols = rotated_grid.shape
    
    print(f"    {'='*80}")
    print(f"    VISUALIZATION (+X Up, +Y Left) | Res: {resolution}m/cell")
    print(f"    {'='*80}")

    for r in range(rows):
        line_buffer = [f"    x={rows-1-r:02d} |"]
        for c in range(cols):
            val = rotated_grid[r, c]
            if val <= threshold:
                char = "   "
            elif val <= threshold * 10:
                char = " . " 
            elif val <= threshold * 100:
                char = " X "
            else:
                char = " # "
            line_buffer.append(char)
        
        line_buffer.append("|")
        print("".join(line_buffer))
    
    axis_line = ["    " + " " * 6]
    for c in range(cols):
        axis_line.append("---")
    print("".join(axis_line))
    print(f"           <-- +Y Direction (cols 0 to {cols-1})")
    print("")

def compute_statistics(errors: List[float]) -> Dict[str, float]:
    if not errors:
        return {}
    arr = np.array(errors)
    return {
        "mean": np.mean(arr),
        "std_dev": np.std(arr),
        "min": np.min(arr),
        "max": np.max(arr),
        "p50": np.percentile(arr, 50),
        "p95": np.percentile(arr, 95),
        "p99": np.percentile(arr, 99)
    }

def main():
    parser = argparse.ArgumentParser(description="Comparator for elevation map binary dumps.")
    parser.add_argument("ref_file", help="Path to reference binary file")
    parser.add_argument("cmp_file", help="Path to comparison binary file")
    parser.add_argument("--tol", type=float, default=1e-4, help="Float tolerance (default: 0.0001)")
    parser.add_argument("--start", type=float, default=None, help="Start timestamp")
    parser.add_argument("--end", type=float, default=None, help="End timestamp")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-frame output")
    
    args = parser.parse_args()
    verbose = not args.quiet

    ref_log = ElevationLogReader(args.ref_file)
    cmp_log = ElevationLogReader(args.cmp_file)
    
    # Load with time slicing to save RAM and processing time
    ref_log.load(args.start, args.end)
    cmp_log.load(args.start, args.end)

    if (ref_log.config.num_x != cmp_log.config.num_x) or (ref_log.config.num_y != cmp_log.config.num_y):
        sys.exit("[!] FATAL: Grid dimension mismatch between files.")

    # Intersection of keys ensures we only compare frames present in both files.
    # We rely on exact timestamp matching as the data source is identical.
    common_timestamps = sorted(list(set(ref_log.frames.keys()) & set(cmp_log.frames.keys())))
    
    missing_in_cmp = len(ref_log.frames) - len(common_timestamps)
    missing_in_ref = len(cmp_log.frames) - len(common_timestamps)

    print("\n" + "="*40 + " COMPARISON START " + "="*40)
    print(f"Common Frames: {len(common_timestamps)} | Unique Ref: {missing_in_cmp} | Unique Cmp: {missing_in_ref}")
    print("-" * 100)

    stats_accumulator = [] 
    all_pixel_errors = []
    divergent_frames_count = 0
    
    for ts in common_timestamps:
        ref_frame = ref_log.frames[ts]
        cmp_frame = cmp_log.frames[ts]

        # Handle NaNs: treat as -9999 so they don't propagate NaN into diff_grid.
        # This allows detecting if one is NaN and the other is valid (result > tol).
        r_data = np.nan_to_num(ref_frame.grid_data, nan=-9999.0)
        c_data = np.nan_to_num(cmp_frame.grid_data, nan=-9999.0)
        
        diff_grid = np.abs(r_data - c_data)
        
        mask_fail = diff_grid > args.tol
        num_failures = np.sum(mask_fail)
        
        if num_failures > 0:
            divergent_frames_count += 1
            max_diff = np.max(diff_grid)
            stats_accumulator.append(max_diff)
            all_pixel_errors.extend(diff_grid[mask_fail].tolist())

            if verbose:
                print(f"[!] DIVERGENCE @ TS={ts:.6f}")
                print(f"    Max Error: {max_diff:.6f} | Failed Cells: {num_failures}/{diff_grid.size}")
                visualize_divergence(diff_grid, args.tol, ref_log.config.resolution)
                print("-" * 100)

    print("\n" + "="*40 + " FINAL REPORT " + "="*40)
    
    if divergent_frames_count == 0:
        print("[SUCCESS] All matched frames are identical within tolerance.")
    else:
        print(f"[FAILURE] Found {divergent_frames_count} divergent frames.")
        
        frame_stats = compute_statistics(stats_accumulator)
        print("\n--- Frame Max-Error Statistics ---")
        for k, v in frame_stats.items():
            print(f"  {k:<10}: {v:.6f}")
        
        pixel_stats = compute_statistics(all_pixel_errors)
        print("\n--- Global Pixel Error Statistics (> tol) ---")
        for k, v in pixel_stats.items():
            print(f"  {k:<10}: {v:.6f}")

if __name__ == "__main__":
    main()