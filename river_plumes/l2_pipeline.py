# l2_pipeline.py

import xarray as xr
import numpy as np
from scipy.interpolate import interp1d
from typing import List, Optional, Tuple


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

    def __init__(self, variable: str = 'chlor_a'):
        self.variable = variable

    def load_dataset(self, file_path: str) -> Optional[xr.Dataset]:
        try:
            geo_ds = xr.open_dataset(file_path, group='geophysical_data')
            nav_ds = xr.open_dataset(file_path, group='navigation_data')

            geo_ds = LatLonAttacher.attach(geo_ds, nav_ds)
            return geo_ds[[self.variable, 'lat', 'lon']]

        except Exception as e:
            print(f"Skipping {file_path} due to error: {e}")
            return None

    def load_multiple(self, file_paths: List[str]) -> List[xr.Dataset]:
        return [ds for ds in (self.load_dataset(path) for path in file_paths) if ds is not None]

class GPUDataExtractor:
    def extract(self, datasets: List[xr.Dataset]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        import cupy as cp
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor

        def _extract(ds):
            # Convert variables to CuPy arrays and flatten
            var = cp.asarray(ds['chlor_a'].values).flatten()
            lat = cp.asarray(ds['lat'].values).flatten()
            lon = cp.asarray(ds['lon'].values).flatten()
            # Mask NaN values on the GPU
            mask = ~cp.isnan(var) & ~cp.isnan(lat) & ~cp.isnan(lon)
            return cp.asnumpy(lon[mask]), cp.asnumpy(lat[mask]), cp.asnumpy(var[mask])

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(_extract, datasets))

        all_lon = np.concatenate([r[0] for r in results])
        all_lat = np.concatenate([r[1] for r in results])
        all_var = np.concatenate([r[2] for r in results])

        return all_lon, all_lat, all_var

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