# l3_pipeline.py

import xarray as xr
import numpy as np
from typing import List, Optional, Tuple, Dict


class L3DatasetLoader:
    """
    Single Responsibility: Loads and prepares Level-3/4 datasets.
    Assumes variables are in the root group and lat/lon are coordinates.
    Can perform subsetting using a bounding box.
    """

    def __init__(self, variables: List[str], 
                 bbox: Optional[Tuple[float, float, float, float]] = None):
        if not isinstance(variables, list):
            raise TypeError("variables must be a list of strings.")
        self.variables = variables
        self.bbox = bbox # Expected format: (lon_min, lon_max, lat_min, lat_max)

    def load_dataset(self, file_path: str) -> Optional[xr.Dataset]:
        try:
            ds = xr.open_dataset(file_path)

            if self.bbox and 'lat' in ds.coords and 'lon' in ds.coords:
                lon_min_bbox, lon_max_bbox, lat_min_bbox, lat_max_bbox = self.bbox
                
                lat_is_descending = False
                if ds['lat'].size > 1 and ds['lat'][0].item() > ds['lat'][1].item():
                    lat_is_descending = True
                
                lat_slice = slice(lat_max_bbox, lat_min_bbox) if lat_is_descending else slice(lat_min_bbox, lat_max_bbox)
                lon_slice = slice(lon_min_bbox, lon_max_bbox)
                
                try:
                    ds_subset = ds.sel(lat=lat_slice, lon=lon_slice)
                    if ds_subset.sizes['lat'] == 0 or ds_subset.sizes['lon'] == 0:
                        print(f"Warning: Subsetting {file_path} with bbox {self.bbox} resulted in an empty dataset. Original data might not cover the bbox. Skipping file.")
                        return None
                    ds = ds_subset
                except Exception as e:
                    print(f"Warning: Failed to subset {file_path} with bbox {self.bbox}. Error: {e}. Using full dataset. This might lead to memory issues.")

            final_vars_to_select = [v for v in self.variables if v in ds]
            missing_vars = [v for v in self.variables if v not in final_vars_to_select]

            if missing_vars:
                print(f"Warning: Skipping {file_path}. Missing data variables after potential subsetting: {', '.join(missing_vars)} in dataset.")
                return None
            
            return ds[final_vars_to_select]

        except Exception as e:
            print(f"Skipping {file_path} due to error: {e}")
            return None

    def load_multiple(self, file_paths: List[str]) -> List[xr.Dataset]:
        return [ds for ds in (self.load_dataset(path) for path in file_paths) if ds is not None]

class GPUDataExtractor:
    """
    Extracts data from datasets using GPU acceleration for multiple variables.
    """
    def __init__(self, variables: List[str]):
        if not isinstance(variables, list):
            raise TypeError("variables must be a list of strings.")
        self.variables = variables

    def extract(self, datasets: List[xr.Dataset]) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        import cupy as cp
        import numpy as np

        def _extract(ds_item):
            lat_values = ds_item['lat'].values
            lon_values = ds_item['lon'].values

            if lat_values.ndim == 1 and lon_values.ndim == 1:
                lon_cp_2d, lat_cp_2d = cp.meshgrid(cp.asarray(lon_values), cp.asarray(lat_values))
            elif lat_values.ndim == 2 and lon_values.ndim == 2:
                lat_cp_2d = cp.asarray(lat_values)
                lon_cp_2d = cp.asarray(lon_values)
            else:
                raise ValueError(
                    f"Latitude and longitude arrays have unexpected dimensions: "
                    f"lat_ndim={lat_values.ndim}, lon_ndim={lon_values.ndim}. "
                    f"Expected 1D for L3 or 2D for L2."
                )

            base_mask = ~cp.isnan(lat_cp_2d) & ~cp.isnan(lon_cp_2d)
            vars_data_cp = {}
            for var_name in self.variables:
                var_data_np = ds_item[var_name].values
                current_var_cp_array = cp.asarray(var_data_np)

                if current_var_cp_array.ndim == 3 and current_var_cp_array.shape[0] == 1 and lat_cp_2d.ndim == 2:
                    if current_var_cp_array.shape[1:] == lat_cp_2d.shape:
                        current_var_cp_array = current_var_cp_array.squeeze(axis=0)
                    else:
                        raise ValueError(
                            f"Variable '{var_name}' (original shape {var_data_np.shape}) has a leading singleton dimension, "
                            f"but its spatial dimensions {current_var_cp_array.shape[1:]} do not match "
                            f"the lat/lon grid shape {lat_cp_2d.shape} after potential squeezing."
                        )
                
                if current_var_cp_array.shape == lat_cp_2d.shape:
                    vars_data_cp[var_name] = current_var_cp_array
                else:
                    try:
                        vars_data_cp[var_name] = cp.broadcast_to(current_var_cp_array, lat_cp_2d.shape)
                    except ValueError as e:
                        raise ValueError(
                            f"Variable '{var_name}' (original shape {var_data_np.shape}, processed CuPy shape {current_var_cp_array.shape}) "
                            f"is not compatible with and could not be broadcast to the 2D lat/lon grid shape {lat_cp_2d.shape}. Error: {e}"
                        )

            combined_mask = base_mask
            for var_name in self.variables:
                if vars_data_cp[var_name].shape != lat_cp_2d.shape:
                    raise ValueError(
                        f"Shape mismatch for variable '{var_name}' ({vars_data_cp[var_name].shape}) "
                        f"and lat/lon grid ({lat_cp_2d.shape}) before masking."
                    )
                combined_mask &= ~cp.isnan(vars_data_cp[var_name])
            
            final_lon_np = cp.asnumpy(lon_cp_2d[combined_mask].flatten())
            final_lat_np = cp.asnumpy(lat_cp_2d[combined_mask].flatten())
            final_vars_np = {
                var_name: cp.asnumpy(vars_data_cp[var_name][combined_mask].flatten())
                for var_name in self.variables
            }

            del lat_cp_2d, lon_cp_2d, vars_data_cp, base_mask, combined_mask

            return final_lon_np, final_lat_np, final_vars_np

        results = []
        if datasets:
            for ds_item_loop in datasets:
                if ds_item_loop is not None:
                    try:
                        result_tuple = _extract(ds_item_loop)
                        results.append(result_tuple)
                    except Exception as e:
                        print(f"Error processing a dataset: {e}. Skipping this dataset.")
                    finally:
                        cp.get_default_memory_pool().free_all_blocks()
                        cp.get_default_pinned_memory_pool().free_all_blocks()
                else:
                    print("Encountered a None dataset, skipping.")
        
        if not results:
            empty_vars_dict = {var: np.array([]) for var in self.variables}
            return np.array([]), np.array([]), empty_vars_dict

        all_lon = np.concatenate([r[0] for r in results if r is not None])
        all_lat = np.concatenate([r[1] for r in results if r is not None])

        all_vars_dict = {}
        for var in self.variables:
            concatenated_var_data = [r[2][var] for r in results if r is not None and var in r[2] and r[2][var].size > 0]
            if concatenated_var_data:
                all_vars_dict[var] = np.concatenate(concatenated_var_data)
            else:
                all_vars_dict[var] = np.array([])

        return all_lon, all_lat, all_vars_dict

class SelectiveInterpolator:
    """
    Encapsulates selective interpolation of small NaN regions in 2D gridded datasets.
    """
    @staticmethod
    def interpolate(binned_data, lat_edges, lon_edges, threshold=32):
        """
        Interpolate small NaN regions in the provided 2D data.
        
        Parameters:
            binned_data : 2D numpy array
                Data that contains NaN values.
            lat_edges : 1D numpy array
                Latitude bin edges.
            lon_edges : 1D numpy array
                Longitude bin edges.
            threshold : int, optional
                Only regions with fewer than this number of connected bins are interpolated.
        
        Returns:
            2D numpy array with small NaN regions filled.
        """
        from scipy import ndimage
        from scipy.interpolate import griddata
        import numpy as np
        
        nan_mask = np.isnan(binned_data)
        s = ndimage.generate_binary_structure(2, 1)
        labeled_array, num_features = ndimage.label(nan_mask, structure=s)
        region_sizes = np.bincount(labeled_array.flatten())[1:]
        small_regions_mask = np.zeros_like(labeled_array, dtype=bool)
        for i, size in enumerate(region_sizes, start=1):
            if size < threshold:
                small_regions_mask |= (labeled_array == i)
        interpolated_data = binned_data.copy()
        known_y, known_x = np.where(~nan_mask)
        known_values = binned_data[known_y, known_x]
        interp_y, interp_x = np.where(small_regions_mask)
        if len(interp_y) > 0:
            interp_values = griddata(
                (known_y, known_x),
                known_values,
                (interp_y, interp_x),
                method='linear'
            )
            interpolated_data[interp_y, interp_x] = interp_values
        return interpolated_data
