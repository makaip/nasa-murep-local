"""Dataset-specific processing functions for the multi-variable satellite pickler."""

import glob
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

from pickler_config import (
    BATHYMETRY_FILE,
    BATHYMETRY_VAR,
    CHLOR_VAR,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    MAX_CDOM,
    MAX_CHLOR,
    MAX_SST_CELSIUS,
    MAX_SSL,
    MIN_CDOM,
    MIN_CHLOR,
    MIN_SST_CELSIUS,
    MIN_SSL,
    MODIS_OC_PATTERN,
    MUR_SST_PATTERN,
    RRS_412_VAR,
    RRS_443_VAR,
    RRS_547_VAR,
    RRS_555_VAR,
    SSL_NATIVE_RES_DEG,
    SSL_PATTERN,
    SSL_VAR,
    SST_VAR,
)
from pickler_utils import (
    bin_data_to_grid,
    calculate_cdom,
    calculate_cdom_slopes,
    extract_date_from_filepath,
    gaussian_fill_nans,
)

# Add parent directory to path (cdom_sst folder containing pipelines)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
sys.dont_write_bytecode = True

from pipelines.l2_pipeline import GPUDataExtractor, L2DatasetLoader
from pipelines.l3_pipeline import L3DatasetLoader


def process_bathymetry_data(lat_edges: np.ndarray, lon_edges: np.ndarray) -> Optional[np.ndarray]:
    """Load and interpolate static bathymetry data to the target grid."""
    print(f"\n{'=' * 60}")
    print("Processing bathymetry data...")
    print(f"{'=' * 60}")

    if not os.path.exists(BATHYMETRY_FILE):
        print(f"  Warning: Bathymetry file not found at {BATHYMETRY_FILE}")
        return None

    try:
        ds = xr.open_dataset(BATHYMETRY_FILE)

        rename_dict = {}
        if "latitude" in ds.coords:
            rename_dict["latitude"] = "lat"
        if "longitude" in ds.coords:
            rename_dict["longitude"] = "lon"
        if rename_dict:
            ds = ds.rename(rename_dict)

        if "lat" not in ds.coords or "lon" not in ds.coords:
            print("  Warning: Bathymetry dataset missing lat/lon coordinates")
            ds.close()
            return None

        if BATHYMETRY_VAR not in ds:
            print(f"  Warning: Bathymetry variable '{BATHYMETRY_VAR}' not found")
            ds.close()
            return None

        subset = ds.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MIN, LAT_MAX))
        bathy_data = subset[BATHYMETRY_VAR].values
        lat_native = subset["lat"].values.copy()
        lon_native = subset["lon"].values.copy()

        if bathy_data.ndim != 2 or len(lat_native) < 2 or len(lon_native) < 2:
            print("  Warning: Bathymetry data shape is not compatible for interpolation")
            ds.close()
            return None

        lat_flip = lat_native[0] > lat_native[-1]
        lon_flip = lon_native[0] > lon_native[-1]
        if lat_flip:
            lat_native = lat_native[::-1]
            bathy_data = bathy_data[::-1, :]
        if lon_flip:
            lon_native = lon_native[::-1]
            bathy_data = bathy_data[:, ::-1]

        lat_centers_tgt = lat_edges[:-1] + np.diff(lat_edges) / 2
        lon_centers_tgt = lon_edges[:-1] + np.diff(lon_edges) / 2
        lon_grid_tgt, lat_grid_tgt = np.meshgrid(lon_centers_tgt, lat_centers_tgt)
        query_points = np.column_stack([lat_grid_tgt.ravel(), lon_grid_tgt.ravel()])

        interp = RegularGridInterpolator(
            (lat_native, lon_native),
            bathy_data,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        bathy_interp = interp(query_points).reshape(len(lat_centers_tgt), len(lon_centers_tgt))
        ds.close()
        print(f"  Bathymetry gridded to shape: {bathy_interp.shape}")
        return bathy_interp

    except Exception as e:
        print(f"  Warning: Failed to process bathymetry data: {e}")
        return None


def process_sst_data(year: int, lat_edges: np.ndarray, lon_edges: np.ndarray) -> List[Dict]:
    """Process SST data for a given year."""
    print(f"\n{'=' * 60}")
    print(f"Processing SST data for {year}...")
    print(f"{'=' * 60}")

    data_dir = MUR_SST_PATTERN.format(year=year)
    nc_files = sorted(glob.glob(os.path.join(data_dir, "*.nc")))

    if not nc_files:
        print(f"  Warning: No SST files found in {data_dir}")
        return []

    print(f"  Found {len(nc_files)} SST files")

    loader = L3DatasetLoader(
        variables=[SST_VAR],
        bbox=(LON_MIN, LON_MAX, LAT_MIN, LAT_MAX),
    )

    results = []

    for file_path in tqdm(nc_files, desc=f"  SST {year}", unit="file"):
        try:
            file_date = extract_date_from_filepath(file_path)
            if not file_date:
                tqdm.write(f"  Warning: Could not extract date from {os.path.basename(file_path)}")
                continue

            ds = loader.load_dataset(file_path)
            if ds is None:
                continue

            sst_kelvin = ds[SST_VAR].values
            sst_celsius = sst_kelvin - 273.15
            sst_celsius = np.where(
                (sst_celsius >= MIN_SST_CELSIUS) & (sst_celsius <= MAX_SST_CELSIUS),
                sst_celsius,
                np.nan,
            )

            lat_data = ds["lat"].values
            lon_data = ds["lon"].values

            if lat_data.ndim == 1 and lon_data.ndim == 1:
                lon_2d, lat_2d = np.meshgrid(lon_data, lat_data)
            else:
                lat_2d, lon_2d = lat_data, lon_data

            lat_flat = lat_2d.flatten()
            lon_flat = lon_2d.flatten()
            sst_flat = sst_celsius.flatten()

            valid_mask = ~np.isnan(sst_flat) & ~np.isnan(lat_flat) & ~np.isnan(lon_flat)
            lat_valid = lat_flat[valid_mask]
            lon_valid = lon_flat[valid_mask]
            sst_valid = sst_flat[valid_mask]

            if len(sst_valid) == 0:
                continue

            binned_sst = bin_data_to_grid(lon_valid, lat_valid, sst_valid, lat_edges, lon_edges)
            results.append({"date": file_date, "sst": binned_sst, "n_points": len(sst_valid)})

        except Exception as e:
            tqdm.write(f"  Error processing {os.path.basename(file_path)}: {e}")
            continue

    print(f"  Successfully processed {len(results)} SST files for {year}")
    return results


def process_chlorophyll_data(year: int, lat_edges: np.ndarray, lon_edges: np.ndarray) -> List[Dict]:
    """Process chlorophyll data for a given year."""
    print(f"\n{'=' * 60}")
    print(f"Processing Chlorophyll data for {year}...")
    print(f"{'=' * 60}")

    data_dir = MODIS_OC_PATTERN.format(year=year)
    nc_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.nc"), recursive=True))

    if not nc_files:
        print(f"  Warning: No chlorophyll files found in {data_dir}")
        return []

    print(f"  Found {len(nc_files)} MODIS OC files")

    loader = L2DatasetLoader(variables=[CHLOR_VAR], group="geophysical_data")
    extractor = GPUDataExtractor(variables=[CHLOR_VAR])

    daily_data = {}

    for file_path in tqdm(nc_files, desc=f"  Chlor {year}", unit="file"):
        try:
            file_date = extract_date_from_filepath(file_path)
            if not file_date:
                continue

            datasets = loader.load_multiple([file_path])
            if not datasets:
                continue

            lon_flat, lat_flat, vars_dict = extractor.extract(datasets)
            if len(lon_flat) == 0 or CHLOR_VAR not in vars_dict:
                continue

            chlor_values = vars_dict[CHLOR_VAR]
            chlor_values = np.where(
                (chlor_values >= MIN_CHLOR) & (chlor_values <= MAX_CHLOR),
                chlor_values,
                np.nan,
            )

            valid_mask = ~np.isnan(chlor_values) & ~np.isnan(lat_flat) & ~np.isnan(lon_flat)
            lat_valid = lat_flat[valid_mask]
            lon_valid = lon_flat[valid_mask]
            chlor_valid = chlor_values[valid_mask]

            if len(chlor_valid) == 0:
                continue

            if file_date not in daily_data:
                daily_data[file_date] = {"lat": [], "lon": [], "chlor": []}

            daily_data[file_date]["lat"].extend(lat_valid.tolist())
            daily_data[file_date]["lon"].extend(lon_valid.tolist())
            daily_data[file_date]["chlor"].extend(chlor_valid.tolist())

        except Exception as e:
            tqdm.write(f"  Error processing {file_path}: {e}")
            continue

    results = []
    for date, data in daily_data.items():
        if len(data["chlor"]) > 0:
            lat_arr = np.array(data["lat"])
            lon_arr = np.array(data["lon"])
            chlor_arr = np.array(data["chlor"])

            binned_chlor = bin_data_to_grid(lon_arr, lat_arr, chlor_arr, lat_edges, lon_edges)
            results.append({"date": date, "chlorophyll": binned_chlor, "n_points": len(chlor_arr)})

    print(f"  Successfully processed {len(results)} chlorophyll records for {year}")
    return results


def process_cdom_data(year: int, lat_edges: np.ndarray, lon_edges: np.ndarray) -> List[Dict]:
    """Process CDOM 412 and both CDOM spectral slopes for a given year."""
    print(f"\n{'=' * 60}")
    print(f"Processing CDOM data for {year}...")
    print(f"{'=' * 60}")

    data_dir = MODIS_OC_PATTERN.format(year=year)
    nc_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.nc"), recursive=True))

    if not nc_files:
        print(f"  Warning: No CDOM files found in {data_dir}")
        return []

    print(f"  Found {len(nc_files)} MODIS OC files")

    loader = L2DatasetLoader(
        variables=[RRS_412_VAR, RRS_555_VAR, RRS_443_VAR, RRS_547_VAR],
        group="geophysical_data",
    )
    extractor = GPUDataExtractor(variables=[RRS_412_VAR, RRS_555_VAR, RRS_443_VAR, RRS_547_VAR])

    daily_data = {}

    for file_path in tqdm(nc_files, desc=f"  CDOM {year}", unit="file"):
        try:
            file_date = extract_date_from_filepath(file_path)
            if not file_date:
                continue

            datasets = loader.load_multiple([file_path])
            if not datasets:
                continue

            lon_flat, lat_flat, vars_dict = extractor.extract(datasets)
            required_vars = [RRS_412_VAR, RRS_555_VAR, RRS_443_VAR, RRS_547_VAR]
            if len(lon_flat) == 0 or any(var not in vars_dict for var in required_vars):
                continue

            rrs_412 = vars_dict[RRS_412_VAR]
            rrs_555 = vars_dict[RRS_555_VAR]
            rrs_443 = vars_dict[RRS_443_VAR]
            rrs_547 = vars_dict[RRS_547_VAR]

            cdom_412_values = calculate_cdom(rrs_412, rrs_555)
            slope_275_295_values, slope_300_600_values = calculate_cdom_slopes(rrs_443, rrs_547)

            cdom_412_values = np.where(
                (cdom_412_values >= MIN_CDOM) & (cdom_412_values <= MAX_CDOM),
                cdom_412_values,
                np.nan,
            )

            valid_geo = ~np.isnan(lat_flat) & ~np.isnan(lon_flat)
            cdom_mask = valid_geo & ~np.isnan(cdom_412_values)
            s275_mask = valid_geo & ~np.isnan(slope_275_295_values)
            s300_mask = valid_geo & ~np.isnan(slope_300_600_values)

            if not np.any(cdom_mask) and not np.any(s275_mask) and not np.any(s300_mask):
                continue

            if file_date not in daily_data:
                daily_data[file_date] = {
                    "cdom_412_lat": [],
                    "cdom_412_lon": [],
                    "cdom_412": [],
                    "s275_295_lat": [],
                    "s275_295_lon": [],
                    "s275_295": [],
                    "s300_600_lat": [],
                    "s300_600_lon": [],
                    "s300_600": [],
                }

            if np.any(cdom_mask):
                daily_data[file_date]["cdom_412_lat"].extend(lat_flat[cdom_mask].tolist())
                daily_data[file_date]["cdom_412_lon"].extend(lon_flat[cdom_mask].tolist())
                daily_data[file_date]["cdom_412"].extend(cdom_412_values[cdom_mask].tolist())

            if np.any(s275_mask):
                daily_data[file_date]["s275_295_lat"].extend(lat_flat[s275_mask].tolist())
                daily_data[file_date]["s275_295_lon"].extend(lon_flat[s275_mask].tolist())
                daily_data[file_date]["s275_295"].extend(slope_275_295_values[s275_mask].tolist())

            if np.any(s300_mask):
                daily_data[file_date]["s300_600_lat"].extend(lat_flat[s300_mask].tolist())
                daily_data[file_date]["s300_600_lon"].extend(lon_flat[s300_mask].tolist())
                daily_data[file_date]["s300_600"].extend(slope_300_600_values[s300_mask].tolist())

        except Exception as e:
            tqdm.write(f"  Error processing {file_path}: {e}")
            continue

    results = []
    for date, data in daily_data.items():
        record = {"date": date}

        if len(data["cdom_412"]) > 0:
            cdom_lat = np.array(data["cdom_412_lat"])
            cdom_lon = np.array(data["cdom_412_lon"])
            cdom_arr = np.array(data["cdom_412"])
            record["cdom_412"] = bin_data_to_grid(cdom_lon, cdom_lat, cdom_arr, lat_edges, lon_edges)
            record["cdom_412_n_points"] = len(cdom_arr)

        if len(data["s275_295"]) > 0:
            s275_lat = np.array(data["s275_295_lat"])
            s275_lon = np.array(data["s275_295_lon"])
            s275_arr = np.array(data["s275_295"])
            record["cdom_slope_275_295"] = bin_data_to_grid(s275_lon, s275_lat, s275_arr, lat_edges, lon_edges)
            record["cdom_slope_275_295_n_points"] = len(s275_arr)

        if len(data["s300_600"]) > 0:
            s300_lat = np.array(data["s300_600_lat"])
            s300_lon = np.array(data["s300_600_lon"])
            s300_arr = np.array(data["s300_600"])
            record["cdom_slope_300_600"] = bin_data_to_grid(s300_lon, s300_lat, s300_arr, lat_edges, lon_edges)
            record["cdom_slope_300_600_n_points"] = len(s300_arr)

        if len(record) > 1:
            results.append(record)

    print(f"  Successfully processed {len(results)} CDOM records for {year}")
    return results


def process_ssl_data(year: int, lat_edges: np.ndarray, lon_edges: np.ndarray) -> List[Dict]:
    """Process SSL (Sea Surface Level/Height) data for a given year."""
    print(f"\n{'=' * 60}")
    print(f"Processing SSL data for {year}...")
    print(f"{'=' * 60}")

    file_path = os.path.join(SSL_PATTERN, f"{year}.nc")

    if not os.path.exists(file_path):
        print(f"  Warning: No SSL file found at {file_path}")
        return []

    print(f"  Found SSL file: {os.path.basename(file_path)}")
    results = []

    try:
        ds = xr.open_dataset(file_path)
        print(f"  Dataset opened. Coords: {list(ds.coords)}, Vars: {list(ds.data_vars)}")

        rename_dict = {}
        if "latitude" in ds.coords:
            rename_dict["latitude"] = "lat"
        if "longitude" in ds.coords:
            rename_dict["longitude"] = "lon"
        if rename_dict:
            ds = ds.rename(rename_dict)

        if SSL_VAR not in ds:
            print(f"  Warning: Variable '{SSL_VAR}' not found in {file_path}")
            return []

        if "time" not in ds.dims:
            print(f"  Warning: No time dimension found in {file_path}")
            return []

        n_times = len(ds["time"])
        print(f"  Processing {n_times} time steps...")

        lat_centers_tgt = lat_edges[:-1] + np.diff(lat_edges) / 2
        lon_centers_tgt = lon_edges[:-1] + np.diff(lon_edges) / 2
        lat_bin_size = np.mean(np.diff(lat_edges))
        lon_bin_size = np.mean(np.diff(lon_edges))
        native_patch_sigma = 0.5

        print(
            f"  SSL → target scale: "
            f"{SSL_NATIVE_RES_DEG / lat_bin_size:.1f}× (lat), "
            f"{SSL_NATIVE_RES_DEG / lon_bin_size:.1f}× (lon); "
            f"using bicubic interpolation"
        )

        lon_grid_tgt, lat_grid_tgt = np.meshgrid(lon_centers_tgt, lat_centers_tgt)
        query_points = np.column_stack([lat_grid_tgt.ravel(), lon_grid_tgt.ravel()])

        lat_native = ds["lat"].values.copy()
        lon_native = ds["lon"].values.copy()

        lat_flip = lat_native[0] > lat_native[-1]
        lon_flip = lon_native[0] > lon_native[-1]
        if lat_flip:
            lat_native = lat_native[::-1]
        if lon_flip:
            lon_native = lon_native[::-1]

        for t_idx in tqdm(range(n_times), desc=f"  SSL {year}", unit="time"):
            try:
                time_val = ds["time"].isel(time=t_idx).values
                file_date = pd.Timestamp(time_val).to_pydatetime()

                ssl_data = ds[SSL_VAR].isel(time=t_idx).values

                if lat_flip:
                    ssl_data = ssl_data[::-1, :]
                if lon_flip:
                    ssl_data = ssl_data[:, ::-1]

                ssl_data = np.where(
                    (ssl_data >= MIN_SSL) & (ssl_data <= MAX_SSL),
                    ssl_data,
                    np.nan,
                )

                n_valid = int(np.sum(~np.isnan(ssl_data)))
                if n_valid == 0:
                    continue

                ssl_data = gaussian_fill_nans(ssl_data, sigma=native_patch_sigma)
                valid_mask = ~np.isnan(ssl_data)

                if not np.all(valid_mask):
                    from scipy.ndimage import distance_transform_edt

                    indices = distance_transform_edt(
                        np.isnan(ssl_data),
                        return_distances=False,
                        return_indices=True,
                    )
                    ssl_data_for_spline = ssl_data[tuple(indices)]
                else:
                    ssl_data_for_spline = ssl_data

                interp = RegularGridInterpolator(
                    (lat_native, lon_native),
                    ssl_data_for_spline,
                    method="cubic",
                    bounds_error=False,
                    fill_value=np.nan,
                )
                ssl_interp = interp(query_points).reshape(len(lat_centers_tgt), len(lon_centers_tgt))

                results.append({"date": file_date, "ssl": ssl_interp, "n_points": n_valid})

            except Exception as e:
                tqdm.write(f"  Error processing time step {t_idx}: {e}")
                continue

        ds.close()

    except Exception as e:
        import traceback

        print(f"  Error processing {file_path}: {e}")
        traceback.print_exc()
        return []

    print(f"  Successfully processed {len(results)} SSL records for {year}")
    return results
