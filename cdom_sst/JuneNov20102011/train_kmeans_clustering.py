r"""
K-Means Clustering Training Script (2010-2011) with Weekly Normalization
=========================================================================
Trains K-means clustering model on MUR L4 SST and MODIS L2 chlorophyll data
from 2010 to 2011 (June-November periods) for the Texas-Louisiana Shelf.

Saves trained model and grid information to E:\satdata\Custom
"""

import glob
import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import binned_statistic_2d
from scipy.interpolate import RegularGridInterpolator
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pickle
import sys
import time
from tqdm import tqdm
from datetime import datetime, timedelta

# Add parent directory to path for pipeline imports
sys.dont_write_bytecode = True
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, "..")
sys.path.insert(0, parent_dir)

from pipelines.l3_pipeline import L3DatasetLoader
from pipelines.l2_pipeline import L2DatasetLoader, GPUDataExtractor

import warnings
warnings.filterwarnings('ignore')


def extract_date_from_filename(filename):
    """Extract date from MUR SST filename."""
    basename = os.path.basename(filename)
    # Filename format: YYYYMMDDHHMMSS-...
    try:
        date_str = basename.split('-')[0][:8]
        return datetime.strptime(date_str, '%Y%m%d')
    except:
        return None


def group_files_by_week(files):
    """Group files by week, returning dict of week_number -> file_list."""
    weeks = {}
    
    for f in files:
        date = extract_date_from_filename(f)
        if date is None:
            continue
        
        # Get week number (ISO week)
        week_key = date.strftime('%Y-W%W')
        
        if week_key not in weeks:
            weeks[week_key] = []
        weeks[week_key].append(f)
    
    return weeks


def normalize_sst_weekly(sst_datasets, sst_var):
    """
    Normalize SST data week-by-week using each week's high/low temperatures.
    Returns list of normalized SST arrays and overall stats.
    """
    print("  Normalizing SST data weekly...")
    
    # Group datasets by week
    weeks = {}
    failed_count = 0
    for ds in sst_datasets:
        # Extract date from time coordinate
        try:
            time_val = ds['time'].values
            if hasattr(time_val, '__iter__'):
                time_val = time_val[0]
            date = pd.Timestamp(time_val).to_pydatetime()
            week_key = date.strftime('%Y-W%W')
            
            if week_key not in weeks:
                weeks[week_key] = []
            weeks[week_key].append(ds)
        except Exception as e:
            failed_count += 1
            if failed_count <= 3:  # Print first few errors
                print(f"    WARNING: Failed to parse date from dataset: {e}")
            continue
    
    print(f"    Found {len(weeks)} weeks of data")
    
    # Process each week
    normalized_sst_list = []
    week_stats = []
    
    for week_key in sorted(weeks.keys()):
        week_datasets = weeks[week_key]
        
        # Collect all SST values for this week
        week_sst_values = []
        for ds in week_datasets:
            sst_kelvin = ds[sst_var].values.squeeze()
            sst_celsius = sst_kelvin - 273.15
            # Basic QC
            sst_celsius[(sst_celsius < -2) | (sst_celsius > 35)] = np.nan
            week_sst_values.append(sst_celsius)
        
        # Calculate week's min and max
        week_sst_concat = np.concatenate([s.ravel() for s in week_sst_values])
        week_min = np.nanmin(week_sst_concat)
        week_max = np.nanmax(week_sst_concat)
        week_range = week_max - week_min
        
        if week_range == 0:
            week_range = 1.0  # Avoid division by zero
        
        week_stats.append({
            'week': week_key,
            'min': week_min,
            'max': week_max,
            'range': week_range,
            'n_files': len(week_datasets)
        })
        
        # Normalize this week's data
        for sst_celsius in week_sst_values:
            sst_normalized = (sst_celsius - week_min) / week_range
            normalized_sst_list.append(sst_normalized)
    
    # Print statistics
    print(f"\n    Weekly SST normalization statistics:")
    for stats in week_stats[:5]:  # Show first 5 weeks
        print(f"      {stats['week']}: [{stats['min']:.2f}, {stats['max']:.2f}]°C "
              f"(range: {stats['range']:.2f}°C, {stats['n_files']} files)")
    if len(week_stats) > 5:
        print(f"      ... and {len(week_stats) - 5} more weeks")
    
    if not normalized_sst_list:
        raise ValueError(f"No SST data was successfully processed. Failed to parse dates from {failed_count} datasets.")
    
    return normalized_sst_list, week_stats


def main():
    script_start = time.time()
    
    print("=" * 80)
    print("K-MEANS CLUSTERING TRAINING (2010-2011) - WEEKLY NORMALIZATION")
    print("=" * 80)
    
    # Configuration
    lon_min, lon_max = -94.0, -88.0
    lat_min, lat_max = 27.5, 30.5
    lat_bins = 200
    lon_bins = 300
    
    sst_base_pattern = r"E:\satdata\MUR-JPL-L4-GLOB-v4.1_Texas Louisiana Shelf_{year}-06-01_{year}-11-30"
    chlor_base_pattern = r"E:\satdata\Texas Louisiana Shelf_{year}-06-01_{year}-11-30"
    output_dir = r"E:\satdata\Custom"
    
    sst_var = 'analysed_sst'
    chlor_var = 'chlor_a'
    chlor_group = 'geophysical_data'
    
    n_clusters = 3  # K-means with exactly 3 clusters
    random_state = 42
    max_iter = 300
    n_init = 10
    years = list(range(2005, 2012))  # 2005-2011, not inclusive ending

    print(f"\nConfiguration:")
    print(f"  Region: [{lon_min}, {lat_min}] to [{lon_max}, {lat_max}]")
    print(f"  Grid: {lat_bins} x {lon_bins}")
    print(f"  Years: {years[0]}-{years[-1]}")
    print(f"  K-means clusters: {n_clusters}")
    print(f"  Normalization: Weekly (by week's high/low)")
    
    # Find and load SST files
    print("\n" + "=" * 80)
    print("LOADING SST DATA")
    print("=" * 80)
    
    t_sst_start = time.time()
    
    sst_files = []
    for year in years:
        sst_dir = sst_base_pattern.format(year=year)
        year_files = glob.glob(os.path.join(sst_dir, '*.nc'))
        sst_files.extend(year_files)
        print(f"  Year {year}: {len(year_files)} files")
    
    sst_files = sorted(sst_files)
    print(f"  Total SST files: {len(sst_files)}")
    
    if not sst_files:
        raise FileNotFoundError("No SST files found")
    
    # Load SST datasets
    sst_loader = L3DatasetLoader(variables=[sst_var], bbox=(lon_min, lon_max, lat_min, lat_max))
    sst_datasets = sst_loader.load_multiple(sst_files)
    
    if not sst_datasets:
        raise ValueError("Failed to load SST datasets")
    
    print(f"  Successfully loaded {len(sst_datasets)} SST datasets")
    
    # Get grid from first dataset
    lat_sst = sst_datasets[0]['lat'].values
    lon_sst = sst_datasets[0]['lon'].values
    
    # Process SST with weekly normalization
    print("  Processing SST data with weekly normalization...")
    sst_list, week_stats = normalize_sst_weekly(sst_datasets, sst_var)
    
    if not sst_list:
        raise ValueError(f"No SST data was successfully processed from {len(sst_datasets)} loaded datasets")
    
    print(f"  Successfully processed {len(sst_list)} SST arrays")
    sst_mean = np.nanmean(np.stack(sst_list, axis=0), axis=0)
    print(f"\n  SST range (after normalization): {np.nanmin(sst_mean):.4f} to {np.nanmax(sst_mean):.4f}")
    print(f"  Valid SST pixels: {np.sum(~np.isnan(sst_mean))}")
    
    t_sst_elapsed = time.time() - t_sst_start
    print(f"  ⏱ SST processing completed in {t_sst_elapsed:.2f}s")
    
    # Find and load chlorophyll files
    print("\n" + "=" * 80)
    print("LOADING CHLOROPHYLL DATA")
    print("=" * 80)
    
    t_chlor_start = time.time()
    
    chlor_files = []
    for year in years:
        chlor_dir = chlor_base_pattern.format(year=year)
        year_files = glob.glob(os.path.join(chlor_dir, '**', '*.nc'), recursive=True)
        chlor_files.extend(year_files)
        print(f"  Year {year}: {len(year_files)} files")
    
    print(f"  Total chlorophyll files: {len(chlor_files)}")
    
    # Load chlorophyll datasets
    chlor_loader = L2DatasetLoader(variables=[chlor_var], group=chlor_group)
    chlor_datasets = chlor_loader.load_multiple(chlor_files)
    print(f"  Loaded {len(chlor_datasets)} chlorophyll datasets")
    
    # Extract chlorophyll data
    print("  Extracting chlorophyll data...")
    try:
        extractor = GPUDataExtractor(variables=[chlor_var])
        lon_chlor_flat, lat_chlor_flat, chlor_data = extractor.extract(chlor_datasets)
        chlor_flat = chlor_data[chlor_var]
        print(f"  GPU extraction: {len(chlor_flat)} points")
    except Exception as e:
        print(f"  GPU failed ({e}), using CPU...")
        all_lons, all_lats, all_chlor = [], [], []
        for ds in tqdm(chlor_datasets, desc="  Extracting chlorophyll", unit="file"):
            lat = ds['lat'].values.flatten()
            lon = ds['lon'].values.flatten()
            chlor = ds[chlor_var].values.flatten()
            valid = (~np.isnan(lat) & ~np.isnan(lon) & ~np.isnan(chlor) & (chlor > 0))
            all_lats.extend(lat[valid])
            all_lons.extend(lon[valid])
            all_chlor.extend(chlor[valid])
        lat_chlor_flat = np.array(all_lats)
        lon_chlor_flat = np.array(all_lons)
        chlor_flat = np.array(all_chlor)
        print(f"  CPU extraction: {len(chlor_flat)} points")
    
    t_chlor_elapsed = time.time() - t_chlor_start
    print(f"  ⏱ Chlorophyll processing completed in {t_chlor_elapsed:.2f}s")
    
    # Grid chlorophyll data
    print("\n" + "=" * 80)
    print("GRIDDING DATA")
    print("=" * 80)
    
    t_grid_start = time.time()
    
    clip_mask = ((lon_chlor_flat >= lon_min) & (lon_chlor_flat <= lon_max) &
                 (lat_chlor_flat >= lat_min) & (lat_chlor_flat <= lat_max))
    
    lon_clipped = lon_chlor_flat[clip_mask]
    lat_clipped = lat_chlor_flat[clip_mask]
    chlor_clipped = chlor_flat[clip_mask]
    
    print(f"  Chlorophyll points in region: {len(chlor_clipped)}")
    
    chlor_binned, lat_edges, lon_edges, _ = binned_statistic_2d(
        lat_clipped, lon_clipped, chlor_clipped,
        statistic='mean',
        bins=[lat_bins, lon_bins],
        range=[[lat_min, lat_max], [lon_min, lon_max]]
    )
    
    print(f"  Valid chlorophyll pixels: {np.sum(~np.isnan(chlor_binned))}")
    
    # Resample SST to chlorophyll grid
    print("  Resampling SST to common grid...")
    sst_interpolator = RegularGridInterpolator(
        (lat_sst, lon_sst), 
        sst_mean,
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )
    
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)
    
    points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    sst_regridded = sst_interpolator(points).reshape(chlor_binned.shape)
    
    print(f"  Valid SST pixels after regridding: {np.sum(~np.isnan(sst_regridded))}")
    
    t_grid_elapsed = time.time() - t_grid_start
    print(f"  ⏱ Gridding completed in {t_grid_elapsed:.2f}s")
    
    # Prepare features for clustering
    print("\n" + "=" * 80)
    print("PREPARING FEATURES")
    print("=" * 80)
    
    valid_mask = ~np.isnan(sst_regridded) & ~np.isnan(chlor_binned)
    n_valid = np.sum(valid_mask)
    print(f"  Pixels with both SST and chlorophyll: {n_valid}")
    
    sst_valid = sst_regridded[valid_mask]
    chlor_valid = chlor_binned[valid_mask]
    
    # Use normalized SST and chlorophyll (no log transformation)
    features = np.column_stack([sst_valid, chlor_valid])
    
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    print(f"  Feature matrix shape: {features_scaled.shape}")
    print(f"  Feature 1 (SST MinMax): min={features_scaled[:, 0].min():.4f}, max={features_scaled[:, 0].max():.4f}, mean={features_scaled[:, 0].mean():.4f}")
    print(f"  Feature 2 (Chlor MinMax): min={features_scaled[:, 1].min():.4f}, max={features_scaled[:, 1].max():.4f}, mean={features_scaled[:, 1].mean():.4f}")
    
    # Train K-Means
    print("\n" + "=" * 80)
    print("TRAINING K-MEANS")
    print("=" * 80)
    
    t_train_start = time.time()
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        max_iter=max_iter,
        n_init=n_init,
        algorithm='lloyd',
        verbose=0
    )
    
    labels = kmeans.fit_predict(features_scaled)
    
    t_train_elapsed = time.time() - t_train_start
    
    print(f"  Clusters: {n_clusters} (guaranteed)")
    print(f"  Iterations: {kmeans.n_iter_}")
    print(f"  Inertia: {kmeans.inertia_:.2f}")
    print(f"  ⏱ Training completed in {t_train_elapsed:.2f}s")
    
    # Print cluster statistics
    print("\n  Cluster characteristics:")
    for label in range(n_clusters):
        mask = labels == label
        n_pixels = mask.sum()
        sst_cluster = sst_valid[mask]
        chlor_cluster = chlor_valid[mask]
        print(f"    Cluster {label}: {n_pixels} pixels ({100*n_pixels/len(labels):.1f}%) | "
              f"SST(norm)={sst_cluster.mean():.4f}±{sst_cluster.std():.4f} | "
              f"Chlor={chlor_cluster.mean():.4f}±{chlor_cluster.std():.4f} mg/m³")
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    t_save_start = time.time()
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'kmeans_model': kmeans,
        'scaler': scaler,
        'lat_centers': lat_centers,
        'lon_centers': lon_centers,
        'lat_edges': lat_edges,
        'lon_edges': lon_edges,
        'bbox': (lon_min, lon_max, lat_min, lat_max),
        'grid_shape': (lat_bins, lon_bins),
        'n_clusters': n_clusters,
        'training_years': years,
        'sst_interpolator': sst_interpolator,
        'training_labels': labels,
        'training_sst': sst_valid,
        'training_chlor': chlor_valid,
        'cluster_centers': kmeans.cluster_centers_,
        'inertia': kmeans.inertia_,
        'weekly_stats': week_stats,
        'normalization': 'weekly'
    }
    
    output_file = os.path.join(output_dir, 'kmeans_model_2010_2011_weekly.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    t_save_elapsed = time.time() - t_save_start
    
    print(f"  Saved to: {output_file}")
    print(f"  File size: {os.path.getsize(output_file) / 1e6:.2f} MB")
    print(f"  ⏱ Saving completed in {t_save_elapsed:.2f}s")
    
    # Final summary
    script_elapsed = time.time() - script_start
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"  Total runtime: {script_elapsed:.2f}s ({script_elapsed/60:.2f} min)")
    print("\n  Breakdown:")
    print(f"    SST loading:     {t_sst_elapsed:8.2f}s ({100*t_sst_elapsed/script_elapsed:5.1f}%)")
    print(f"    Chlor loading:   {t_chlor_elapsed:8.2f}s ({100*t_chlor_elapsed/script_elapsed:5.1f}%)")
    print(f"    Gridding:        {t_grid_elapsed:8.2f}s ({100*t_grid_elapsed/script_elapsed:5.1f}%)")
    print(f"    Training:        {t_train_elapsed:8.2f}s ({100*t_train_elapsed/script_elapsed:5.1f}%)")
    print(f"    Saving:          {t_save_elapsed:8.2f}s ({100*t_save_elapsed/script_elapsed:5.1f}%)")
    print("\n  Model Summary:")
    print(f"    Algorithm: K-means")
    print(f"    Clusters: {n_clusters}")
    print(f"    Normalization: Weekly (by week's high/low SST)")
    print(f"    Features: Normalized SST + log10(Chlorophyll)")
    print("=" * 80)


if __name__ == '__main__':
    main()
