r"""
DBSCAN Clustering Training Script (2005-2011)
==============================================
Trains DBSCAN clustering model on MUR L4 SST and MODIS L2 chlorophyll data
from 2005 to 2011 (June-November periods) for the Texas-Louisiana Shelf.

Saves trained model and grid information to E:\satdata\Custom
"""

import glob
import os
import numpy as np
import xarray as xr
from scipy.stats import binned_statistic_2d
from scipy.interpolate import RegularGridInterpolator
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import pickle
import sys
import time
from tqdm import tqdm

# Add parent directory to path for pipeline imports
sys.dont_write_bytecode = True
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, "..")
sys.path.insert(0, parent_dir)

from pipelines.l3_pipeline import L3DatasetLoader
from pipelines.l2_pipeline import L2DatasetLoader, GPUDataExtractor

import warnings
warnings.filterwarnings('ignore')


def find_optimal_eps(features_scaled, min_samples, target_clusters=3):
    """Find optimal epsilon for DBSCAN to produce target number of clusters."""
    print(f"Finding epsilon for {target_clusters} clusters...")
    
    t0 = time.time()
    
    # Compute k-distance
    print("  Computing k-nearest neighbors...")
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(features_scaled)
    distances, _ = nbrs.kneighbors(features_scaled)
    k_dist = np.sort(distances[:, -1])
    
    # Parameter sweep with finer resolution
    eps_min = np.percentile(k_dist, 50)
    eps_max = np.percentile(k_dist, 99)
    eps_values = np.linspace(eps_min, eps_max, 50)  # More granular search
    
    print(f"  Searching eps range: [{eps_min:.4f}, {eps_max:.4f}]")
    
    best_eps = None
    best_diff = np.inf
    best_n_clusters = None
    
    # Track all results
    results = []
    
    for e in tqdm(eps_values, desc="  Testing eps values", unit="eps"):
        db = DBSCAN(eps=e, min_samples=min_samples, n_jobs=-1).fit(features_scaled)
        labels_temp = db.labels_
        n_clusters_temp = len(set(labels_temp)) - (1 if -1 in labels_temp else 0)
        n_noise_temp = np.sum(labels_temp == -1)
        
        results.append({
            'eps': e,
            'n_clusters': n_clusters_temp,
            'n_noise': n_noise_temp,
            'noise_pct': 100 * n_noise_temp / len(labels_temp)
        })
        
        # Find eps that produces exactly target_clusters
        diff = abs(n_clusters_temp - target_clusters)
        if diff < best_diff:
            best_diff = diff
            best_eps = e
            best_n_clusters = n_clusters_temp
            
        # If we found exact match, we can stop early
        if n_clusters_temp == target_clusters:
            print(f"\n  ✓ Found eps={e:.4f} with exactly {target_clusters} clusters")
            elapsed = time.time() - t0
            print(f"  Completed in {elapsed:.2f}s")
            return e
    
    # If exact match not found, show what we got
    elapsed = time.time() - t0
    print(f"\n  ⚠ Exact match not found. Using eps={best_eps:.4f}")
    print(f"     This produces {best_n_clusters} clusters (target was {target_clusters})")
    print(f"  Completed in {elapsed:.2f}s")
    
    # Show distribution of results
    print("\n  Cluster count distribution:")
    unique_counts = sorted(set(r['n_clusters'] for r in results))
    for nc in unique_counts[:10]:  # Show first 10
        count = sum(1 for r in results if r['n_clusters'] == nc)
        print(f"    {nc} clusters: {count} eps values")
    if len(unique_counts) > 10:
        print(f"    ... and {len(unique_counts) - 10} more")
    
    return best_eps


def main():
    script_start = time.time()
    
    print("=" * 80)
    print("DBSCAN CLUSTERING TRAINING (2005-2011)")
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
    
    min_samples = 15  # Increased to merge small clusters
    target_clusters = 11  # Explicitly set target
    manual_eps = None  # Set to a float to override automatic search
    years = list(range(2011, 2012))  # 2011-2011, not inclusive

    print(f"\nConfiguration:")
    print(f"  Region: [{lon_min}, {lat_min}] to [{lon_max}, {lat_max}]")
    print(f"  Grid: {lat_bins} x {lon_bins}")
    print(f"  Years: {years[0]}-{years[-1]}")
    print(f"  Min samples: {min_samples}")
    print(f"  Target clusters: {target_clusters}")
    print(f"  Manual eps: {manual_eps if manual_eps else 'Auto'}")
    
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
    
    # Get grid from first dataset
    lat_sst = sst_datasets[0]['lat'].values
    lon_sst = sst_datasets[0]['lon'].values
    
    # Process and average SST
    print("  Processing SST data...")
    sst_list = []
    for ds in tqdm(sst_datasets, desc="  Processing SST files", unit="file"):
        sst_kelvin = ds[sst_var].values.squeeze()
        sst_celsius = sst_kelvin - 273.15
        sst_celsius[(sst_celsius < -2) | (sst_celsius > 35)] = np.nan
        sst_list.append(sst_celsius)
    
    sst_mean = np.nanmean(np.stack(sst_list, axis=0), axis=0)
    print(f"  SST range: {np.nanmin(sst_mean):.2f} to {np.nanmax(sst_mean):.2f} °C")
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
    
    features = np.column_stack([sst_valid, chlor_valid])
    
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    print(f"  Feature matrix shape: {features_scaled.shape}")
    
    # Find optimal epsilon
    print("\n" + "=" * 80)
    print("OPTIMIZING PARAMETERS")
    print("=" * 80)
    
    if manual_eps is not None:
        eps = manual_eps
        print(f"  Using manual eps: {eps:.4f}")
    else:
        eps = find_optimal_eps(features_scaled, min_samples, target_clusters=target_clusters)
        print(f"  Selected eps: {eps:.4f}")
    
    # Train DBSCAN
    print("\n" + "=" * 80)
    print("TRAINING DBSCAN")
    print("=" * 80)
    
    t_train_start = time.time()
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
    labels = dbscan.fit_predict(features_scaled)
    
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
        sst_cluster = sst_valid[mask]
        chlor_cluster = chlor_valid[mask]
        print(f"    Cluster {label}: {mask.sum()} pixels | "
              f"SST={sst_cluster.mean():.2f}°C | "
              f"Chlor={chlor_cluster.mean():.4f} mg/m³")
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    t_save_start = time.time()
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'dbscan_model': dbscan,
        'scaler': scaler,
        'lat_centers': lat_centers,
        'lon_centers': lon_centers,
        'lat_edges': lat_edges,
        'lon_edges': lon_edges,
        'bbox': (lon_min, lon_max, lat_min, lat_max),
        'grid_shape': (lat_bins, lon_bins),
        'eps': eps,
        'min_samples': min_samples,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'training_years': years,
        'sst_interpolator': sst_interpolator,  # Save for applying to new data
        'training_labels': labels,
        'training_sst': sst_valid,
        'training_chlor': chlor_valid
    }
    
    output_file = os.path.join(output_dir, 'dbscan_model_2005_2011.pkl')
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
