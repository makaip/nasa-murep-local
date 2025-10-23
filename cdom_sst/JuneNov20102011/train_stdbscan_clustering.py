r"""
ST-DBSCAN Clustering Training Script (2005-2011)
=================================================
Trains ST-DBSCAN (Spatio-Temporal DBSCAN) clustering model on MUR L4 SST 
and MODIS L2 chlorophyll data from 2005 to 2011 (June-November periods) 
for the Texas-Louisiana Shelf.

ST-DBSCAN extends DBSCAN by incorporating temporal information, allowing
detection of clusters that are close both spatially (in feature space) 
and temporally (across time).

Saves trained model and grid information to E:\satdata\Custom
"""

import glob
import os
import numpy as np
import xarray as xr
from scipy.stats import binned_statistic_2d
from scipy.interpolate import RegularGridInterpolator
from st_dbscan import ST_DBSCAN
from sklearn.preprocessing import MinMaxScaler
import pickle
import sys
import time
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path for pipeline imports
sys.dont_write_bytecode = True
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, "..")
sys.path.insert(0, parent_dir)

from pipelines.l3_pipeline import L3DatasetLoader
from pipelines.l2_pipeline import L2DatasetLoader, GPUDataExtractor

import warnings
warnings.filterwarnings('ignore')


def main():
    script_start = time.time()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("=" * 80)
    print("ST-DBSCAN CLUSTERING TRAINING (2005-2011)")
    print("=" * 80)
    
    # Configuration
    lon_min, lon_max = -94.0, -88.0
    lat_min, lat_max = 27.5, 30.5
    lat_bins = 100  # from 200 to manage memory
    lon_bins = 150  # from 300 to manage memory
    subsample_fraction = 0.5
    
    sst_base_pattern = r"E:\satdata\MUR-JPL-L4-GLOB-v4.1_Texas Louisiana Shelf_{year}-06-01_{year}-11-30"
    chlor_base_pattern = r"E:\satdata\Texas Louisiana Shelf_{year}-06-01_{year}-11-30"
    output_dir = r"E:\satdata\Custom"
    
    sst_var = 'analysed_sst'
    chlor_var = 'chlor_a'
    chlor_group = 'geophysical_data'
    
    # ST-DBSCAN parameters
    eps1 = 0.01  # Spatial threshold (in normalized feature space)
    eps2 = 4   # Temporal threshold (in days)
    min_samples = 10
    years = [2011]  # 2011-2011, not inclusive

    print(f"\nConfiguration:")
    print(f"  Region: [{lon_min}, {lat_min}] to [{lon_max}, {lat_max}]")
    print(f"  Grid: {lat_bins} x {lon_bins}")
    print(f"  Subsample: {subsample_fraction*100:.0f}% of valid points per timestep")
    print(f"  Years: {years[0]}-{years[-1]}")
    print(f"  eps1 (spatial): {eps1}")
    print(f"  eps2 (temporal): {eps2} days")
    print(f"  min_samples: {min_samples}")
    
    # Find and load SST files
    print("\n" + "=" * 80)
    print("LOADING SST DATA")
    print("=" * 80)
    
    t_sst_start = time.time()
    
    sst_files = []
    for year in years:
        sst_dir = sst_base_pattern.format(year=year)
        year_files = sorted(glob.glob(os.path.join(sst_dir, '*.nc')))
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
    
    # Get grid from first dataset
    lat_sst = sst_datasets[0]['lat'].values
    lon_sst = sst_datasets[0]['lon'].values
    
    # Process SST data per time step
    print("  Processing SST data...")
    sst_list = []
    time_indices = []
    
    for idx, ds in enumerate(tqdm(sst_datasets, desc="  Processing SST files", unit="file")):
        sst_kelvin = ds[sst_var].values.squeeze()
        sst_celsius = sst_kelvin - 273.15
        sst_celsius[(sst_celsius < -2) | (sst_celsius > 35)] = np.nan
        sst_list.append(sst_celsius)
        
        # Extract time index (days since start)
        if 'time' in ds.coords:
            time_val = ds['time'].values
            if idx == 0:
                start_time = time_val
            # Convert to days since start
            time_diff = (time_val - start_time) / np.timedelta64(1, 'D')
            time_indices.append(float(time_diff))
        else:
            # If no time coordinate, use index as proxy
            time_indices.append(float(idx))
    
    print(f"  Time range: {time_indices[0]:.1f} to {time_indices[-1]:.1f} days")
    print(f"  Valid SST datasets: {len(sst_list)}")
    
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
        year_files = sorted(glob.glob(os.path.join(chlor_dir, '**', '*.nc'), recursive=True))
        chlor_files.extend(year_files)
        print(f"  Year {year}: {len(year_files)} files")
    
    print(f"  Total chlorophyll files: {len(chlor_files)}")
    
    # Group chlorophyll files by date (for temporal matching)
    chlor_by_date = {}
    for fpath in chlor_files:
        fname = os.path.basename(fpath)
        # Extract date from filename (adjust pattern as needed)
        # Typical pattern: AQUA_MODIS.20110601T*.nc
        try:
            date_part = fname.split('.')[1][:8]  # YYYYMMDD
            if date_part not in chlor_by_date:
                chlor_by_date[date_part] = []
            chlor_by_date[date_part].append(fpath)
        except:
            pass
    
    print(f"  Unique dates: {len(chlor_by_date)}")
    
    t_chlor_elapsed = time.time() - t_chlor_start
    print(f"  ⏱ Chlorophyll file grouping completed in {t_chlor_elapsed:.2f}s")
    
    # Grid data and prepare temporal features
    print("\n" + "=" * 80)
    print("GRIDDING AND TEMPORAL PROCESSING")
    print("=" * 80)
    
    t_grid_start = time.time()
    
    # Create spatial grid
    lat_edges = np.linspace(lat_min, lat_max, lat_bins + 1)
    lon_edges = np.linspace(lon_min, lon_max, lon_bins + 1)
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)
    
    # Prepare data structure for ST-DBSCAN
    # Format: [time, sst, chlor] for each valid spatial pixel at each time step
    temporal_data = []
    
    # Process each SST time step
    for time_idx, (sst_data, time_val) in enumerate(tqdm(
        zip(sst_list, time_indices), 
        desc="  Processing time steps", 
        total=len(sst_list),
        unit="step"
    )):
        # Resample SST to grid
        sst_interpolator = RegularGridInterpolator(
            (lat_sst, lon_sst), 
            sst_data,
            method='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        
        points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        sst_regridded = sst_interpolator(points).reshape(lat_bins, lon_bins)
        
        # Process chlorophyll for this date (approximate matching)
        # Get date string from time index
        date_key = None
        for key in sorted(chlor_by_date.keys()):
            # Simple matching - can be improved
            if time_idx < len(chlor_by_date):
                date_key = sorted(chlor_by_date.keys())[time_idx]
                break
        
        if date_key and date_key in chlor_by_date:
            # Load and grid chlorophyll for this date
            chlor_loader = L2DatasetLoader(variables=[chlor_var], group=chlor_group)
            date_files = chlor_by_date[date_key]
            
            try:
                chlor_datasets = chlor_loader.load_multiple(date_files[:10])  # Limit files per date
                
                # Extract chlorophyll data
                try:
                    extractor = GPUDataExtractor(variables=[chlor_var])
                    lon_chlor_flat, lat_chlor_flat, chlor_data = extractor.extract(chlor_datasets)
                    chlor_flat = chlor_data[chlor_var]
                except:
                    # CPU fallback
                    all_lons, all_lats, all_chlor = [], [], []
                    for ds in chlor_datasets:
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
                
                # Clip to region
                clip_mask = ((lon_chlor_flat >= lon_min) & (lon_chlor_flat <= lon_max) &
                            (lat_chlor_flat >= lat_min) & (lat_chlor_flat <= lat_max))
                
                lon_clipped = lon_chlor_flat[clip_mask]
                lat_clipped = lat_chlor_flat[clip_mask]
                chlor_clipped = chlor_flat[clip_mask]
                
                # Grid chlorophyll
                chlor_binned, _, _, _ = binned_statistic_2d(
                    lat_clipped, lon_clipped, chlor_clipped,
                    statistic='mean',
                    bins=[lat_bins, lon_bins],
                    range=[[lat_min, lat_max], [lon_min, lon_max]]
                )
            except Exception as e:
                print(f"  Warning: Failed to process chlorophyll for {date_key}: {e}")
                chlor_binned = np.full((lat_bins, lon_bins), np.nan)
        else:
            chlor_binned = np.full((lat_bins, lon_bins), np.nan)
        
        # Extract valid pixels and create temporal feature vectors
        valid_mask = ~np.isnan(sst_regridded) & ~np.isnan(chlor_binned)
        n_valid = np.sum(valid_mask)
        
        if n_valid > 0:
            sst_valid = sst_regridded[valid_mask]
            chlor_valid = chlor_binned[valid_mask]
            
            # MEMORY OPTIMIZATION: Subsample to reduce total points
            # Use configured subsample_fraction from top of script
            n_subsample = max(1, int(n_valid * subsample_fraction))
            
            if n_subsample < n_valid:
                # Random sampling without replacement
                sample_indices = np.random.choice(n_valid, size=n_subsample, replace=False)
                sst_valid = sst_valid[sample_indices]
                chlor_valid = chlor_valid[sample_indices]
                n_valid = n_subsample
            
            # Create feature vectors: [time, sst, chlor]
            time_vec = np.full(n_valid, time_val)
            features = np.column_stack([time_vec, sst_valid, chlor_valid])
            temporal_data.append(features)
    
    # Combine all temporal data
    data = np.vstack(temporal_data)
    
    print(f"  Total data points: {len(data)}")
    print(f"  Time range: {data[:, 0].min():.1f} to {data[:, 0].max():.1f}")
    print(f"  SST range: {np.nanmin(data[:, 1]):.2f} to {np.nanmax(data[:, 1]):.2f} °C")
    print(f"  Chlor range: {np.nanmin(data[:, 2]):.4f} to {np.nanmax(data[:, 2]):.4f} mg/m³")
    
    t_grid_elapsed = time.time() - t_grid_start
    print(f"  ⏱ Gridding completed in {t_grid_elapsed:.2f}s")
    
    # Normalize features (excluding time column)
    print("\n" + "=" * 80)
    print("PREPARING FEATURES")
    print("=" * 80)
    
    scaler = MinMaxScaler()
    data_normalized = data.copy()
    data_normalized[:, 1:] = scaler.fit_transform(data[:, 1:])
    
    print(f"  Feature matrix shape: {data_normalized.shape}")
    print(f"  Time (not normalized): {data_normalized[:, 0].min():.1f} to {data_normalized[:, 0].max():.1f}")
    print(f"  SST (normalized): {data_normalized[:, 1].min():.3f} to {data_normalized[:, 1].max():.3f}")
    print(f"  Chlor (normalized): {data_normalized[:, 2].min():.3f} to {data_normalized[:, 2].max():.3f}")
    
    # Train ST-DBSCAN
    print("\n" + "=" * 80)
    print("TRAINING ST-DBSCAN")
    print("=" * 80)
    
    t_train_start = time.time()
    
    # Frame-based processing parameters for memory efficiency
    frame_size = int(eps2 * 3)  # Process 3x temporal window at a time
    frame_overlap = eps2  # Overlap by temporal threshold
    
    print(f"  Parameters:")
    print(f"    eps1 (spatial): {eps1}")
    print(f"    eps2 (temporal): {eps2} days")
    print(f"    min_samples: {min_samples}")
    print(f"    frame_size: {frame_size} days (for memory efficiency)")
    print(f"    frame_overlap: {frame_overlap} days")
    print()
    print(f"  Using fit_frame_split() for large dataset memory management...")
    
    st_dbscan = ST_DBSCAN(eps1=eps1, eps2=eps2, min_samples=min_samples, n_jobs=-1)
    st_dbscan.fit_frame_split(data_normalized, frame_size=frame_size, frame_overlap=frame_overlap)
    
    labels = st_dbscan.labels
    
    t_train_elapsed = time.time() - t_train_start
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    
    print(f"  Clusters: {n_clusters}")
    print(f"  Noise points: {n_noise} ({100*n_noise/len(labels):.1f}%)")
    print(f"  ⏱ Training completed in {t_train_elapsed:.2f}s")
    
    # Print cluster statistics
    print("\n  Cluster characteristics:")
    for label in sorted(set(labels)):
        if label == -1:
            continue
        mask = labels == label
        time_cluster = data[mask, 0]
        sst_cluster = data[mask, 1]
        chlor_cluster = data[mask, 2]
        print(f"    Cluster {label}: {mask.sum()} points | "
              f"Time={time_cluster.min():.0f}-{time_cluster.max():.0f} days | "
              f"SST={sst_cluster.mean():.2f}°C | "
              f"Chlor={chlor_cluster.mean():.4f} mg/m³")
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    t_save_start = time.time()
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'st_dbscan_model': st_dbscan,
        'scaler': scaler,
        'lat_centers': lat_centers,
        'lon_centers': lon_centers,
        'lat_edges': lat_edges,
        'lon_edges': lon_edges,
        'bbox': (lon_min, lon_max, lat_min, lat_max),
        'grid_shape': (lat_bins, lon_bins),
        'eps1': eps1,
        'eps2': eps2,
        'min_samples': min_samples,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'training_years': years,
        'training_labels': labels,
        'training_data': data,
        'training_data_normalized': data_normalized,
        'time_indices': time_indices
    }
    
    output_file = os.path.join(output_dir, 'stdbscan_model_2005_2011.pkl')
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
    print("=" * 80)


if __name__ == '__main__':
    main()
