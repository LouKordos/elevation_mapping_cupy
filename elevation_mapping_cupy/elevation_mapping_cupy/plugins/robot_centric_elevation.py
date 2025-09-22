#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import string
from typing import List
from .plugin_manager import PluginBase


class RobotCentricElevation(PluginBase):
    """Generates an elevation map with respect to the robot frame."""

    def __init__(
        self,
        cell_n: int = 100,
        resolution: float = 0.05,
        threshold: float = 0.4,
        use_threshold: bool = False,
        input_layer_name: str = "elevation",
        **kwargs,
    ):
        super().__init__()
        self.input_layer_name = input_layer_name
        self.width = cell_n
        self.height = cell_n
        self.min_filtered = cp.zeros((self.width, self.height), dtype=cp.float32)

        self.base_elevation_kernel = cp.ElementwiseKernel(
            in_params="raw U map, raw U mask, raw U R",
            out_params="raw U newmap",
            preamble=string.Template(
                """
                __device__ int get_map_idx(int idx, int layer_n) {
                    const int layer = ${width} * ${height};
                    return layer * layer_n + idx;
                }

                __device__ bool is_inside(int idx) {
                    int idx_x = idx / ${width};
                    int idx_y = idx % ${width};
                    if (idx_x <= 0 || idx_x >= ${width} - 1) {
                        return false;
                    }
                    if (idx_y <= 0 || idx_y >= ${height} - 1) {
                        return false;
                    }
                    return true;
                }
                __device__ float get_map_x(int idx){
                    float center_offset = ${width} * ${resolution} / 2.0;
                    float idx_x = idx / ${width} * ${resolution} - center_offset;
                    return idx_x;
                }
                __device__ float get_map_y(int idx){
                    float center_offset = ${height} * ${resolution} / 2.0;
                    float idx_y = idx % ${height} * ${resolution} - center_offset;
                    return idx_y;
                }
                __device__ float transform_p(float x, float y, float z,
                                        float r0, float r1, float r2) {
                    return r0 * x + r1 * y + r2 * z ;
                }
                """
            ).substitute(width=self.width, height=self.height, resolution=resolution),
            operation=string.Template(
                """
                U rz = map[get_map_idx(i, 0)];
                U valid = mask[get_map_idx(i, 0)];
                if (valid > 0.5) {
                    U rx = get_map_x(get_map_idx(i, 0));
                    U ry = get_map_y(get_map_idx(i, 0));
                    U x_b = transform_p(rx, ry, rz, R[0], R[3], R[6]);
                    U y_b = transform_p(rx, ry, rz, R[1], R[4], R[7]);
                    U z_b = transform_p(rx, ry, rz, R[2], R[5], R[8]);
                    if (${use_threshold} && z_b>= ${threshold} ) {
                        newmap[get_map_idx(i, 0)] = 1.0;
                    }
                    else if (${use_threshold} && z_b< ${threshold} ){
                        newmap[get_map_idx(i, 0)] = 0.0;
                    }
                    else{
                        newmap[get_map_idx(i, 0)] = z_b;
                    }
                }
                """
            ).substitute(threshold=threshold, use_threshold=int(use_threshold)),
            name="base_elevation_kernel",
        )

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        semantic_map: cp.ndarray,
        semantic_layer_names: List[str],
        rotation,
        *args,
    ) -> cp.ndarray:
        # Get the specified input layer data
        input_heights = self.get_layer_data(
            elevation_map,
            layer_names,
            plugin_layers,
            plugin_layer_names,
            semantic_map,
            semantic_layer_names,
            self.input_layer_name,
        )
        if input_heights is None:
            # Fallback to raw elevation if the desired layer is not found
            print(f"Warning: layer '{self.input_layer_name}' not found for RobotCentricElevation. Falling back to 'elevation'.")
            input_heights = elevation_map[0]

        # Use the valid mask from the raw elevation map (layer 2)
        valid_mask = elevation_map[2]

        # Process maps here. The rotation is from base to map, so we need the transpose
        # to transform points from map to base.
        self.min_filtered = input_heights.copy()
        self.base_elevation_kernel(
            input_heights, valid_mask, rotation.T, self.min_filtered, size=(self.width * self.height),
        )
        return self.min_filtered