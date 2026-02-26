"""Pipeline orchestration for multi-variable satellite data pickling."""

import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd

from pickler_config import (
    BATHYMETRY_FILE,
    BATHYMETRY_VAR,
    END_YEAR,
    LAT_BINS,
    LAT_MAX,
    LAT_MIN,
    LON_BINS,
    LON_MAX,
    LON_MIN,
    OUTPUT_DIR,
    OUTPUT_FILENAME,
    START_YEAR,
)
from pickler_processors import (
    process_bathymetry_data,
    process_cdom_data,
    process_chlorophyll_data,
    process_sst_data,
    process_ssl_data,
)
from pickler_utils import create_spatial_grid


def _merge_year_data(
    sst_results: List[Dict],
    chlor_results: List[Dict],
    cdom_results: List[Dict],
    ssl_results: List[Dict],
) -> List[Dict]:
    """Merge per-product records into a single list of per-date records for one year."""
    year_data = {}

    for record in sst_results:
        date = record["date"]
        year_data.setdefault(date, {})
        year_data[date]["sst"] = record["sst"]
        year_data[date]["sst_n_points"] = record["n_points"]

    for record in chlor_results:
        date = record["date"]
        year_data.setdefault(date, {})
        year_data[date]["chlorophyll"] = record["chlorophyll"]
        year_data[date]["chlor_n_points"] = record["n_points"]

    for record in cdom_results:
        date = record["date"]
        year_data.setdefault(date, {})

        if "cdom_412" in record:
            year_data[date]["cdom_412"] = record["cdom_412"]
            year_data[date]["cdom"] = record["cdom_412"]  # Backward-compatible alias
            year_data[date]["cdom_412_n_points"] = record.get("cdom_412_n_points", np.nan)

        if "cdom_slope_275_295" in record:
            year_data[date]["cdom_slope_275_295"] = record["cdom_slope_275_295"]
            year_data[date]["cdom_slope_275_295_n_points"] = record.get(
                "cdom_slope_275_295_n_points", np.nan
            )

        if "cdom_slope_300_600" in record:
            year_data[date]["cdom_slope_300_600"] = record["cdom_slope_300_600"]
            year_data[date]["cdom_slope_300_600_n_points"] = record.get(
                "cdom_slope_300_600_n_points", np.nan
            )

    for record in ssl_results:
        date = record["date"]
        year_data.setdefault(date, {})
        year_data[date]["ssl"] = record["ssl"]
        year_data[date]["ssl_n_points"] = record["n_points"]

    merged_records = []
    for date, data in year_data.items():
        merged = {"date": date}
        merged.update(data)
        merged_records.append(merged)
    return merged_records


def run_pipeline() -> None:
    """Main processing pipeline."""
    print("\n" + "=" * 70)
    print("Multi-Variable Satellite Data Pickler")
    print("=" * 70)
    print(f"Time range: {START_YEAR}-{END_YEAR}")
    print(f"Region: [{LON_MIN}, {LAT_MIN}] to [{LON_MAX}, {LAT_MAX}]")
    print(f"Grid resolution: {LAT_BINS} x {LON_BINS}")
    print(f"Output: {os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)}")
    print("=" * 70)

    lat_edges, lon_edges, lat_centers, lon_centers = create_spatial_grid()
    print("\nSpatial grid created:")
    print(f"  Latitude bins: {len(lat_centers)} ({LAT_MIN}° to {LAT_MAX}°)")
    print(f"  Longitude bins: {len(lon_centers)} ({LON_MIN}° to {LON_MAX}°)")

    bathymetry_grid = process_bathymetry_data(lat_edges, lon_edges)

    all_data = []

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\n{'#' * 70}")
        print(f"# PROCESSING YEAR {year}")
        print(f"{'#' * 70}")

        sst_results = process_sst_data(year, lat_edges, lon_edges)
        chlor_results = process_chlorophyll_data(year, lat_edges, lon_edges)
        cdom_results = process_cdom_data(year, lat_edges, lon_edges)
        ssl_results = process_ssl_data(year, lat_edges, lon_edges)

        year_records = _merge_year_data(sst_results, chlor_results, cdom_results, ssl_results)
        all_data.extend(year_records)

        print(f"\nYear {year} summary: {len(year_records)} unique dates")

    print(f"\n{'=' * 70}")
    print("Creating unified DataFrame...")
    print(f"{'=' * 70}")

    if not all_data:
        print("ERROR: No data collected! Check file paths and data availability.")
        return

    all_data.sort(key=lambda x: x["date"])

    df = pd.DataFrame(all_data)
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    print("\nDataFrame created:")
    print(f"  Total records: {len(df)}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Columns: {list(df.columns)}")
    print("\nData availability:")

    for col, label in [
        ("sst", "SST"),
        ("chlorophyll", "Chlorophyll"),
        ("cdom_412", "CDOM 412"),
        ("cdom_slope_275_295", "CDOM slope S275:295"),
        ("cdom_slope_300_600", "CDOM slope S300:600"),
        ("ssl", "SSL"),
    ]:
        if col in df.columns:
            print(f"  {label}: {df[col].notna().sum()} days")
        else:
            print(f"  {label}: 0 days (no data collected)")

    print("\nStatic layer availability:")
    if bathymetry_grid is not None:
        print(f"  Bathymetry: available ({bathymetry_grid.shape[0]} x {bathymetry_grid.shape[1]})")
    else:
        print("  Bathymetry: unavailable")

    metadata = {
        "lat_centers": lat_centers,
        "lon_centers": lon_centers,
        "lat_edges": lat_edges,
        "lon_edges": lon_edges,
        "bbox": (LON_MIN, LON_MAX, LAT_MIN, LAT_MAX),
        "grid_shape": (LAT_BINS, LON_BINS),
        "bathymetry": bathymetry_grid,
        "bathymetry_file": BATHYMETRY_FILE,
        "bathymetry_var": BATHYMETRY_VAR,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    print(f"\n{'=' * 70}")
    print(f"Saving pickle file to: {output_path}")
    print(f"{'=' * 70}")

    with open(output_path, "wb") as f:
        pickle.dump({"data": df, "metadata": metadata}, f)

    print("\n✓ SUCCESS! Pickle file saved.")
    print(f"  File size: {os.path.getsize(output_path) / (1024 ** 2):.2f} MB")
    print("\nTo load the data:")
    print("  import pickle")
    print(f"  with open('{output_path}', 'rb') as f:")
    print("      data = pickle.load(f)")
    print("  df = data['data']")
    print("  metadata = data['metadata']")
    print("\n" + "=" * 70)
