# l2_pipeline.py

import xarray as xr
import numpy as np
from scipy.interpolate import interp1d
from typing import List, Optional, Tuple, Dict


class LatLonAttacher:
    """
    Attaches lat/lon coordinates to a dataset. Supports both full arrays and interpolated control points.
    """
    @staticmethod
    def attach(geo_ds: xr.Dataset, nav_ds: xr.Dataset) -> xr.Dataset:
        if {'latitude', 'longitude'} <= set(nav_ds.variables):
            lat = nav_ds['latitude'].data
            lon = nav_ds['longitude'].data

            geo_ds = geo_ds.assign_coords(
                lat=(('number_of_lines', 'pixels_per_line'), lat),
                lon=(('number_of_lines', 'pixels_per_line'), lon)
            )
            return geo_ds
        elif 'cntl_pt_cols' in nav_ds.variables:
            lat_ctl = nav_ds['latitude'].data
            lon_ctl = nav_ds['longitude'].data
            cols = nav_ds['cntl_pt_cols'].data

            n_lines = geo_ds.sizes['number_of_lines']
            n_pixels = geo_ds.sizes['pixels_per_line']

            lat_full = np.empty((n_lines, n_pixels), dtype=np.float32)
            lon_full = np.empty((n_lines, n_pixels), dtype=np.float32)

            for i in range(n_lines):
                f_lat = interp1d(cols, lat_ctl[i, :], bounds_error=False, fill_value="extrapolate")
                f_lon = interp1d(cols, lon_ctl[i, :], bounds_error=False, fill_value="extrapolate")
                lat_full[i, :] = f_lat(np.arange(n_pixels))
                lon_full[i, :] = f_lon(np.arange(n_pixels))

            geo_ds = geo_ds.assign_coords(
                lat=(('number_of_lines', 'pixels_per_line'), lat_full),
                lon=(('number_of_lines', 'pixels_per_line'), lon_full)
            )
            return geo_ds
        else:
            raise ValueError("No usable latitude/longitude information found in navigation_data.")

class L2DatasetLoader:
    """
    Single Responsibility: Loads and prepares Level-2 datasets.
    """

    def __init__(self, variables: List[str]):
        # Store the list of variables to load
        if not isinstance(variables, list):
            raise TypeError("variables must be a list of strings.")
        self.variables = variables

    def load_dataset(self, file_path: str) -> Optional[xr.Dataset]:
        try:
            geo_ds = xr.open_dataset(file_path, group='geophysical_data')
            nav_ds = xr.open_dataset(file_path, group='navigation_data')

            geo_ds = LatLonAttacher.attach(geo_ds, nav_ds)
            # Select latitude, longitude, and all specified variables
            required_vars = ['lat', 'lon'] + self.variables
            # Check if all required variables exist
            missing_vars = [v for v in self.variables if v not in geo_ds]
            if missing_vars:
                print(f"Warning: Skipping {file_path}. Missing variables: {', '.join(missing_vars)}")
                return None

            return geo_ds[required_vars]

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
        from concurrent.futures import ThreadPoolExecutor

        def _extract(ds):
            # Convert core coords to CuPy arrays
            lat_cp = cp.asarray(ds['lat'].values)
            lon_cp = cp.asarray(ds['lon'].values)

            # Base mask for NaNs in lat/lon
            base_mask = ~cp.isnan(lat_cp) & ~cp.isnan(lon_cp)

            # Load all variable data
            vars_data_cp = {var: cp.asarray(ds[var].values) for var in self.variables}

            # Combine masks: NaN in lat OR lon OR any variable
            combined_mask = base_mask
            for var in self.variables:
                combined_mask &= ~cp.isnan(vars_data_cp[var])

            # Flatten and apply the combined mask
            final_lon = cp.asnumpy(lon_cp[combined_mask].flatten())
            final_lat = cp.asnumpy(lat_cp[combined_mask].flatten())
            final_vars = {
                var: cp.asnumpy(vars_data_cp[var][combined_mask].flatten())
                for var in self.variables
            }

            return final_lon, final_lat, final_vars

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(_extract, datasets))

        # Concatenate results from all datasets
        all_lon = np.concatenate([r[0] for r in results])
        all_lat = np.concatenate([r[1] for r in results])

        # Concatenate variable data
        all_vars_dict = {}
        for var in self.variables:
            all_vars_dict[var] = np.concatenate([r[2][var] for r in results])

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
        
        # Create a mask for NaN values
        nan_mask = np.isnan(binned_data)
        s = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
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