# l2_pipeline.py

import xarray as xr
import numpy as np
from scipy.interpolate import interp1d
from typing import List, Optional


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
