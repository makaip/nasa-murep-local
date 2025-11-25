"""
Multi-Variable Satellite Data Pickler
======================================
This script processes and combines SST (MUR L4), Chlorophyll (MODIS L2), 
and CDOM (MODIS L2) data from 2005-2011 into a single pickle file.

Data is binned to a common spatial grid, temporally aligned, and saved
as a pandas DataFrame for easy manipulation and plotting.

Date: November 2025
"""

import glob
import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import binned_statistic_2d
from datetime import datetime, timedelta
import pickle
import sys
from typing import List, Tuple, Dict, Optional
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Add parent directory to path (cdom_sst folder containing pipelines)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)  # This should be cdom_sst/
sys.path.insert(0, parent_dir)
sys.dont_write_bytecode = True

# Import custom pipeline components
from pipelines.l2_pipeline import L2DatasetLoader, GPUDataExtractor
from pipelines.l3_pipeline import L3DatasetLoader, GPUDataExtractor as L3GPUDataExtractor

# =====================================================================
# CONFIGURATION
# =====================================================================

# Geographical Bounding Box - Texas Louisiana Shelf
LON_MIN, LON_MAX = -94.0, -88.0
LAT_MIN, LAT_MAX = 27.5, 30.5

# Binning parameters (consistent spatial grid for all variables)
LAT_BINS = 200
LON_BINS = 300

# Time range
START_YEAR = 2005
END_YEAR = 2011

# Data directories (patterns with {year} placeholder)
MUR_SST_PATTERN = r"E:\satdata\MUR-JPL-L4-GLOB-v4.1_Texas Louisiana Shelf_{year}-01-01_{year}-12-31"
MODIS_OC_PATTERN = r"E:\satdata\Texas Louisiana Shelf_{year}-01-01_{year}-12-31"

# Output directory
OUTPUT_DIR = r"E:\satdata\Custom"
OUTPUT_FILENAME = "combined_sst_chlor_cdom_2005-2011.pkl"

# Variable names
SST_VAR = 'analysed_sst'
CHLOR_VAR = 'chlor_a'
RRS_412_VAR = 'Rrs_412'
RRS_555_VAR = 'Rrs_555'

# CDOM calculation constants
CDOM_B0 = 0.2487
CDOM_B1 = 14.028
CDOM_B2 = 4.085

# Quality control thresholds
MIN_SST_CELSIUS = -2.0
MAX_SST_CELSIUS = 40.0
MIN_CHLOR = 0.01
MAX_CHLOR = 100.0
MIN_CDOM = 0.0
MAX_CDOM = 1.0

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def create_spatial_grid() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create consistent spatial grid for binning."""
    lat_edges = np.linspace(LAT_MIN, LAT_MAX, LAT_BINS + 1)
    lon_edges = np.linspace(LON_MIN, LON_MAX, LON_BINS + 1)
    
    # Create bin centers
    lat_centers = lat_edges[:-1] + np.diff(lat_edges) / 2
    lon_centers = lon_edges[:-1] + np.diff(lon_edges) / 2
    
    return lat_edges, lon_edges, lat_centers, lon_centers


def extract_date_from_filepath(filepath: str) -> Optional[datetime]:
    """Extract date from satellite filename."""
    import re
    
    basename = os.path.basename(filepath)
    
    # Common patterns in satellite filenames
    patterns = [
        r'(\d{4})(\d{2})(\d{2})',  # YYYYMMDD
        r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD
        r'(\d{4})(\d{3})',  # YYYYDDD (year + day of year)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, basename)
        if match:
            try:
                if len(match.groups()) == 3:  # YYYY MM DD format
                    year, month, day = map(int, match.groups())
                    return datetime(year, month, day)
                elif len(match.groups()) == 2:  # YYYY DDD format
                    year, day_of_year = map(int, match.groups())
                    return datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
            except ValueError:
                continue
    
    return None


def bin_data_to_grid(lon: np.ndarray, lat: np.ndarray, values: np.ndarray,
                     lat_edges: np.ndarray, lon_edges: np.ndarray) -> np.ndarray:
    """Bin scattered data to 2D grid using mean statistic."""
    binned_data, _, _, _ = binned_statistic_2d(
        lat, lon, values,
        statistic='mean',
        bins=[lat_edges, lon_edges],
        range=[[LAT_MIN, LAT_MAX], [LON_MIN, LON_MAX]]
    )
    return binned_data


def calculate_cdom(rrs_412: np.ndarray, rrs_555: np.ndarray) -> np.ndarray:
    """Calculate CDOM from Rrs_412 and Rrs_555."""
    cdom_values = np.full_like(rrs_412, np.nan)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        term_ratio = rrs_412 / rrs_555
        term_numerator = term_ratio - CDOM_B0
        term_division = term_numerator / CDOM_B2
        valid_log_mask = term_division > 0
        
        if np.any(valid_log_mask):
            cdom_values[valid_log_mask] = (np.log(term_division[valid_log_mask])) / (-CDOM_B1)
    
    return cdom_values


# =====================================================================
# DATA PROCESSING FUNCTIONS
# =====================================================================

def process_sst_data(year: int, lat_edges: np.ndarray, lon_edges: np.ndarray) -> List[Dict]:
    """Process SST data for a given year."""
    print(f"\n{'='*60}")
    print(f"Processing SST data for {year}...")
    print(f"{'='*60}")
    
    data_dir = MUR_SST_PATTERN.format(year=year)
    nc_files = sorted(glob.glob(os.path.join(data_dir, '*.nc')))
    
    if not nc_files:
        print(f"  Warning: No SST files found in {data_dir}")
        return []
    
    print(f"  Found {len(nc_files)} SST files")
    
    # Initialize L3 loader with bounding box
    loader = L3DatasetLoader(
        variables=[SST_VAR],
        bbox=(LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
    )
    
    results = []
    
    for i, file_path in enumerate(tqdm(nc_files, desc=f"  SST {year}", unit="file")):
        try:
            # Extract date from filename
            file_date = extract_date_from_filepath(file_path)
            if not file_date:
                tqdm.write(f"  Warning: Could not extract date from {os.path.basename(file_path)}")
                continue
            
            # Load dataset
            ds = loader.load_dataset(file_path)
            if ds is None:
                continue
            
            # Extract SST data
            sst_kelvin = ds[SST_VAR].values
            
            # Convert to Celsius
            sst_celsius = sst_kelvin - 273.15
            
            # Apply quality control
            sst_celsius = np.where(
                (sst_celsius >= MIN_SST_CELSIUS) & (sst_celsius <= MAX_SST_CELSIUS),
                sst_celsius,
                np.nan
            )
            
            # Get lat/lon
            lat_data = ds['lat'].values
            lon_data = ds['lon'].values
            
            # Handle 1D coordinates (typical for L3/L4)
            if lat_data.ndim == 1 and lon_data.ndim == 1:
                lon_2d, lat_2d = np.meshgrid(lon_data, lat_data)
            else:
                lat_2d, lon_2d = lat_data, lon_data
            
            # Flatten and filter valid data
            lat_flat = lat_2d.flatten()
            lon_flat = lon_2d.flatten()
            sst_flat = sst_celsius.flatten()
            
            valid_mask = ~np.isnan(sst_flat) & ~np.isnan(lat_flat) & ~np.isnan(lon_flat)
            lat_valid = lat_flat[valid_mask]
            lon_valid = lon_flat[valid_mask]
            sst_valid = sst_flat[valid_mask]
            
            if len(sst_valid) == 0:
                continue
            
            # Bin to grid
            binned_sst = bin_data_to_grid(lon_valid, lat_valid, sst_valid, lat_edges, lon_edges)
            
            results.append({
                'date': file_date,
                'sst': binned_sst,
                'n_points': len(sst_valid)
            })
                
        except Exception as e:
            tqdm.write(f"  Error processing {os.path.basename(file_path)}: {e}")
            continue
    
    print(f"  Successfully processed {len(results)} SST files for {year}")
    return results


def process_chlorophyll_data(year: int, lat_edges: np.ndarray, lon_edges: np.ndarray) -> List[Dict]:
    """Process chlorophyll data for a given year."""
    print(f"\n{'='*60}")
    print(f"Processing Chlorophyll data for {year}...")
    print(f"{'='*60}")
    
    data_dir = MODIS_OC_PATTERN.format(year=year)
    nc_files = sorted(glob.glob(os.path.join(data_dir, '**', '*.nc'), recursive=True))
    
    if not nc_files:
        print(f"  Warning: No chlorophyll files found in {data_dir}")
        return []
    
    print(f"  Found {len(nc_files)} MODIS OC files")
    
    # Initialize L2 loader
    loader = L2DatasetLoader(
        variables=[CHLOR_VAR],
        group='geophysical_data'
    )
    
    # Initialize GPU extractor
    extractor = GPUDataExtractor(variables=[CHLOR_VAR])
    
    results = []
    
    # Process in batches to manage memory
    batch_size = 50
    total_batches = (len(nc_files) + batch_size - 1) // batch_size
    for batch_start in tqdm(range(0, len(nc_files), batch_size), desc=f"  Chlor {year}", unit="batch", total=total_batches):
        batch_files = nc_files[batch_start:batch_start + batch_size]
        
        try:
            # Load batch
            datasets = loader.load_multiple(batch_files)
            
            if not datasets:
                continue
            
            # Extract data
            lon_flat, lat_flat, vars_dict = extractor.extract(datasets)
            
            if len(lon_flat) == 0 or CHLOR_VAR not in vars_dict:
                continue
            
            chlor_values = vars_dict[CHLOR_VAR]
            
            # Apply quality control
            chlor_values = np.where(
                (chlor_values >= MIN_CHLOR) & (chlor_values <= MAX_CHLOR),
                chlor_values,
                np.nan
            )
            
            # Filter valid data
            valid_mask = ~np.isnan(chlor_values) & ~np.isnan(lat_flat) & ~np.isnan(lon_flat)
            lat_valid = lat_flat[valid_mask]
            lon_valid = lon_flat[valid_mask]
            chlor_valid = chlor_values[valid_mask]
            
            if len(chlor_valid) == 0:
                continue
            
            # For L2 data, we need to aggregate by day since multiple files per day
            # Extract dates from all files in batch
            file_dates = []
            for f in batch_files:
                date = extract_date_from_filepath(f)
                if date:
                    file_dates.append(date)
            
            # Use the most common date in batch (or first date)
            if file_dates:
                batch_date = max(set(file_dates), key=file_dates.count)
            else:
                continue
            
            # Bin to grid
            binned_chlor = bin_data_to_grid(lon_valid, lat_valid, chlor_valid, lat_edges, lon_edges)
            
            results.append({
                'date': batch_date,
                'chlorophyll': binned_chlor,
                'n_points': len(chlor_valid)
            })
            
        except Exception as e:
            tqdm.write(f"  Error processing batch starting at index {batch_start}: {e}")
            continue
    
    print(f"  Successfully processed {len(results)} chlorophyll records for {year}")
    return results


def process_cdom_data(year: int, lat_edges: np.ndarray, lon_edges: np.ndarray) -> List[Dict]:
    """Process CDOM data for a given year."""
    print(f"\n{'='*60}")
    print(f"Processing CDOM data for {year}...")
    print(f"{'='*60}")
    
    data_dir = MODIS_OC_PATTERN.format(year=year)
    nc_files = sorted(glob.glob(os.path.join(data_dir, '**', '*.nc'), recursive=True))
    
    if not nc_files:
        print(f"  Warning: No CDOM files found in {data_dir}")
        return []
    
    print(f"  Found {len(nc_files)} MODIS OC files")
    
    # Initialize L2 loader for RRS variables
    loader = L2DatasetLoader(
        variables=[RRS_412_VAR, RRS_555_VAR],
        group='geophysical_data'
    )
    
    # Initialize GPU extractor
    extractor = GPUDataExtractor(variables=[RRS_412_VAR, RRS_555_VAR])
    
    results = []
    
    # Process in batches
    batch_size = 50
    total_batches = (len(nc_files) + batch_size - 1) // batch_size
    for batch_start in tqdm(range(0, len(nc_files), batch_size), desc=f"  CDOM {year}", unit="batch", total=total_batches):
        batch_files = nc_files[batch_start:batch_start + batch_size]
        
        try:
            # Load batch
            datasets = loader.load_multiple(batch_files)
            
            if not datasets:
                continue
            
            # Extract data
            lon_flat, lat_flat, vars_dict = extractor.extract(datasets)
            
            if len(lon_flat) == 0 or RRS_412_VAR not in vars_dict or RRS_555_VAR not in vars_dict:
                continue
            
            rrs_412 = vars_dict[RRS_412_VAR]
            rrs_555 = vars_dict[RRS_555_VAR]
            
            # Calculate CDOM
            cdom_values = calculate_cdom(rrs_412, rrs_555)
            
            # Apply quality control
            cdom_values = np.where(
                (cdom_values >= MIN_CDOM) & (cdom_values <= MAX_CDOM),
                cdom_values,
                np.nan
            )
            
            # Filter valid data
            valid_mask = ~np.isnan(cdom_values) & ~np.isnan(lat_flat) & ~np.isnan(lon_flat)
            lat_valid = lat_flat[valid_mask]
            lon_valid = lon_flat[valid_mask]
            cdom_valid = cdom_values[valid_mask]
            
            if len(cdom_valid) == 0:
                continue
            
            # Extract dates from batch
            file_dates = []
            for f in batch_files:
                date = extract_date_from_filepath(f)
                if date:
                    file_dates.append(date)
            
            if file_dates:
                batch_date = max(set(file_dates), key=file_dates.count)
            else:
                continue
            
            # Bin to grid
            binned_cdom = bin_data_to_grid(lon_valid, lat_valid, cdom_valid, lat_edges, lon_edges)
            
            results.append({
                'date': batch_date,
                'cdom': binned_cdom,
                'n_points': len(cdom_valid)
            })
            
        except Exception as e:
            tqdm.write(f"  Error processing batch starting at index {batch_start}: {e}")
            continue
    
    print(f"  Successfully processed {len(results)} CDOM records for {year}")
    return results


# =====================================================================
# MAIN PROCESSING PIPELINE
# =====================================================================

def main():
    """Main processing pipeline."""
    print("\n" + "="*70)
    print("Multi-Variable Satellite Data Pickler")
    print("="*70)
    print(f"Time range: {START_YEAR}-{END_YEAR}")
    print(f"Region: [{LON_MIN}, {LAT_MIN}] to [{LON_MAX}, {LAT_MAX}]")
    print(f"Grid resolution: {LAT_BINS} x {LON_BINS}")
    print(f"Output: {os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)}")
    print("="*70)
    
    # Create spatial grid
    lat_edges, lon_edges, lat_centers, lon_centers = create_spatial_grid()
    print(f"\nSpatial grid created:")
    print(f"  Latitude bins: {len(lat_centers)} ({LAT_MIN}° to {LAT_MAX}°)")
    print(f"  Longitude bins: {len(lon_centers)} ({LON_MIN}° to {LON_MAX}°)")
    
    # Collect all data
    all_data = []
    
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\n{'#'*70}")
        print(f"# PROCESSING YEAR {year}")
        print(f"{'#'*70}")
        
        # Process SST
        sst_results = process_sst_data(year, lat_edges, lon_edges)
        
        # Process Chlorophyll
        chlor_results = process_chlorophyll_data(year, lat_edges, lon_edges)
        
        # Process CDOM
        cdom_results = process_cdom_data(year, lat_edges, lon_edges)
        
        # Combine results for this year
        # Create a dictionary indexed by date
        year_data = {}
        
        # Add SST data
        for record in sst_results:
            date = record['date']
            if date not in year_data:
                year_data[date] = {}
            year_data[date]['sst'] = record['sst']
            year_data[date]['sst_n_points'] = record['n_points']
        
        # Add chlorophyll data
        for record in chlor_results:
            date = record['date']
            if date not in year_data:
                year_data[date] = {}
            year_data[date]['chlorophyll'] = record['chlorophyll']
            year_data[date]['chlor_n_points'] = record['n_points']
        
        # Add CDOM data
        for record in cdom_results:
            date = record['date']
            if date not in year_data:
                year_data[date] = {}
            year_data[date]['cdom'] = record['cdom']
            year_data[date]['cdom_n_points'] = record['n_points']
        
        # Convert to list of records
        for date, data in year_data.items():
            record = {'date': date}
            record.update(data)
            all_data.append(record)
        
        print(f"\nYear {year} summary: {len(year_data)} unique dates")
    
    # Create DataFrame
    print(f"\n{'='*70}")
    print("Creating unified DataFrame...")
    print(f"{'='*70}")
    
    if not all_data:
        print("ERROR: No data collected! Check file paths and data availability.")
        return
    
    # Sort by date
    all_data.sort(key=lambda x: x['date'])
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    
    print(f"\nDataFrame created:")
    print(f"  Total records: {len(df)}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\nData availability:")
    print(f"  SST: {df['sst'].notna().sum()} days")
    print(f"  Chlorophyll: {df['chlorophyll'].notna().sum()} days")
    print(f"  CDOM: {df['cdom'].notna().sum()} days")
    
    # Add grid coordinates to the pickle
    metadata = {
        'lat_centers': lat_centers,
        'lon_centers': lon_centers,
        'lat_edges': lat_edges,
        'lon_edges': lon_edges,
        'bbox': (LON_MIN, LON_MAX, LAT_MIN, LAT_MAX),
        'grid_shape': (LAT_BINS, LON_BINS)
    }
    
    # Save to pickle
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    
    print(f"\n{'='*70}")
    print(f"Saving pickle file to: {output_path}")
    print(f"{'='*70}")
    
    with open(output_path, 'wb') as f:
        pickle.dump({
            'data': df,
            'metadata': metadata
        }, f)
    
    print(f"\n✓ SUCCESS! Pickle file saved.")
    print(f"  File size: {os.path.getsize(output_path) / (1024**2):.2f} MB")
    print(f"\nTo load the data:")
    print(f"  import pickle")
    print(f"  with open('{output_path}', 'rb') as f:")
    print(f"      data = pickle.load(f)")
    print(f"  df = data['data']")
    print(f"  metadata = data['metadata']")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
