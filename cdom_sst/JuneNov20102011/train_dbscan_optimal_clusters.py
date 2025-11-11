r"""
DBSCAN Clustering Training Script with Optimal Cluster Detection (2010-2011)
============================================================================
Trains DBSCAN clustering model on MUR L4 SST and MODIS L2 chlorophyll data
from 2010 to 2011 (June-November periods) for the Texas-Louisiana Shelf.

Uses the same methodology as clustering_sst_chlorophyll.ipynb:
- Log-transforms chlorophyll before scaling
- Uses StandardScaler for feature normalization
- Finds optimal epsilon via parameter sweep with stability analysis
- Selects eps based on: cluster stability + low noise + reasonable count

Saves trained model and grid information to E:\satdata\Custom
"""

import glob
import os
import numpy as np
import xarray as xr
from scipy.stats import binned_statistic_2d
from scipy.interpolate import RegularGridInterpolator
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
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


def find_optimal_epsilon(features_scaled, min_samples):
    """
    Find optimal epsilon for DBSCAN using parameter sweep with stability analysis.
    
    This matches the notebook's approach: test multiple eps values and select based on:
    1. Cluster count stability (derivative < 1)
    2. Low noise fraction (< 15%)
    3. Reasonable cluster count (2-10)
    """
    print(f"Finding optimal epsilon using parameter sweep (k={min_samples})...")
    
    t0 = time.time()
    
    # Compute k-distance graph
    print("  Computing k-nearest neighbors...")
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(features_scaled)
    distances, _ = nbrs.kneighbors(features_scaled)
    k_dist = np.sort(distances[:, -1])  # Distance to k-th nearest neighbor
    
    # Print k-distance percentiles
    print(f"\n  k-distance percentiles:")
    for p in [50, 70, 80, 90, 95, 99]:
        print(f"    {p}th percentile: {np.percentile(k_dist, p):.4f}")
    
    # Parameter sweep to find optimal eps
    print(f"\n  Running parameter sweep...")
    eps_min = np.percentile(k_dist, 50)
    eps_max = np.percentile(k_dist, 99)
    eps_values = np.linspace(eps_min, eps_max, 20)
    
    sweep_results = []
    for e in tqdm(eps_values, desc="  Testing eps values", unit="eps"):
        db = DBSCAN(eps=e, min_samples=min_samples, n_jobs=-1).fit(features_scaled)
        labels_temp = db.labels_
        n_clusters_temp = len(set(labels_temp)) - (1 if -1 in labels_temp else 0)
        n_noise_temp = np.sum(labels_temp == -1)
        noise_frac = n_noise_temp / len(labels_temp)
        sweep_results.append((e, n_clusters_temp, n_noise_temp, noise_frac))
    
    # Analyze sweep results
    eps_arr = np.array([r[0] for r in sweep_results])
    n_clusters_arr = np.array([r[1] for r in sweep_results])
    noise_frac_arr = np.array([r[3] for r in sweep_results])
    
    # Print sweep summary
    print(f"\n  Parameter sweep results:")
    print(f"  {'eps':>8s} {'clusters':>10s} {'noise':>10s} {'noise_frac':>12s}")
    print("  " + "-" * 44)
    for e, nc, nn, nf in sweep_results[::2]:  # print every other result
        print(f"  {e:8.4f} {nc:10d} {nn:10d} {nf:12.1%}")
    
    # Select optimal eps: balance between cluster count stability and low noise
    # Strategy: find eps where clusters stabilize (derivative is small) and noise < 15%
    cluster_diff = np.abs(np.diff(n_clusters_arr))
    stable_mask = (cluster_diff < 1) if len(cluster_diff) > 0 else np.ones(len(n_clusters_arr) - 1, dtype=bool)
    stable_mask = np.append(stable_mask, True)  # include last point
    low_noise_mask = noise_frac_arr < 0.15
    
    # Find candidates: stable clusters + low noise + reasonable cluster count
    candidate_mask = stable_mask & low_noise_mask & (n_clusters_arr >= 2) & (n_clusters_arr <= 10)
    
    if np.any(candidate_mask):
        # Pick the one with lowest noise among candidates
        candidate_idx = np.where(candidate_mask)[0]
        best_idx = candidate_idx[np.argmin(noise_frac_arr[candidate_idx])]
        print(f"\n  ✓ Found optimal eps using stability + low noise criteria")
    else:
        # Fallback: pick eps at 85th percentile of k-distance
        best_idx = np.argmin(np.abs(eps_arr - np.percentile(k_dist, 85)))
        print(f"\n  ⚠ No candidates met criteria, using 85th percentile fallback")
    
    optimal_eps = sweep_results[best_idx][0]
    optimal_n_clusters = sweep_results[best_idx][1]
    optimal_noise = sweep_results[best_idx][2]
    optimal_noise_frac = sweep_results[best_idx][3]
    
    elapsed = time.time() - t0
    
    print(f"\n  Optimal parameters selected:")
    print(f"    eps: {optimal_eps:.4f}")
    print(f"    Expected clusters: {optimal_n_clusters}")
    print(f"    Expected noise: {optimal_noise} ({optimal_noise_frac:.1%})")
    print(f"  ⏱ Parameter sweep completed in {elapsed:.2f}s")
    
    return optimal_eps, k_dist, sweep_results


def main():
    script_start = time.time()
    
    print("=" * 80)
    print("DBSCAN CLUSTERING TRAINING WITH OPTIMAL CLUSTER DETECTION (2005-2011)")
    print("=" * 80)
    
    # Configuration
    lon_min, lon_max = -94.0, -88.0
    lat_min, lat_max = 27.5, 30.5
    lat_bins = 200
    lon_bins = 300
    
    sst_base_pattern = r"E:\satdata\MUR-JPL-L4-GLOB-v4.1_Texas Louisiana Shelf_{year}-01-01_{year}-12-31"
    chlor_base_pattern = r"E:\satdata\Texas Louisiana Shelf_{year}-01-01_{year}-12-31"
    output_dir = r"E:\satdata\Custom"
    
    sst_var = 'analysed_sst'
    chlor_var = 'chlor_a'
    chlor_group = 'geophysical_data'
    
    # DBSCAN parameters (matches notebook)
    min_samples = 4
    years = [2005, 2011]  # June-November 2010 and 2011 (matches notebook)

    print(f"\nConfiguration:")
    print(f"  Region: [{lon_min}, {lat_min}] to [{lon_max}, {lat_max}]")
    print(f"  Grid: {lat_bins} x {lon_bins}")
    print(f"  Years: {years}")
    print(f"  Min samples (k): {min_samples}")
    print(f"  Method: Parameter sweep with stability analysis (matches notebook)")
    
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
    
    # Create feature matrix (N x 2) with log-transformed chlorophyll (matches notebook)
    features = np.column_stack([sst_valid, np.log10(chlor_valid)])
    
    print(f"  Feature matrix shape: {features.shape}")
    print(f"  SST mean: {sst_valid.mean():.2f} °C, std: {sst_valid.std():.2f}")
    print(f"  Chlorophyll mean: {chlor_valid.mean():.6f} mg/m³, std: {chlor_valid.std():.6f}")
    
    # Standardize features for DBSCAN (matches notebook)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    print(f"\n  Scaled features shape: {features_scaled.shape}")
    print(f"  Feature 0 (SST) mean: {features_scaled[:, 0].mean():.4f}, std: {features_scaled[:, 0].std():.4f}")
    print(f"  Feature 1 (log10 Chlor) mean: {features_scaled[:, 1].mean():.4f}, std: {features_scaled[:, 1].std():.4f}")
    
    # Find optimal epsilon using parameter sweep
    print("\n" + "=" * 80)
    print("FINDING OPTIMAL EPSILON")
    print("=" * 80)
    
    t_optimize_start = time.time()
    eps, k_distances, sweep_results = find_optimal_epsilon(features_scaled, min_samples)
    t_optimize_elapsed = time.time() - t_optimize_start
    
    # Train final DBSCAN with optimal parameters
    print("\n" + "=" * 80)
    print("TRAINING FINAL MODEL")
    print("=" * 80)
    
    t_train_start = time.time()
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
    labels = dbscan.fit_predict(features_scaled)
    
    t_train_elapsed = time.time() - t_train_start
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    
    print(f"  Clusters found: {n_clusters}")
    print(f"  Noise points: {n_noise} ({100*n_noise/len(labels):.1f}%)")
    print(f"  ⏱ Training completed in {t_train_elapsed:.2f}s")
    
    # Calculate clustering quality metrics
    if n_clusters >= 2:
        mask_no_noise = labels != -1
        if np.sum(mask_no_noise) > 0:
            silhouette = silhouette_score(
                features_scaled[mask_no_noise], 
                labels[mask_no_noise],
                sample_size=min(10000, np.sum(mask_no_noise))
            )
            davies_bouldin = davies_bouldin_score(
                features_scaled[mask_no_noise],
                labels[mask_no_noise]
            )
            calinski = calinski_harabasz_score(
                features_scaled[mask_no_noise],
                labels[mask_no_noise]
            )
            print(f"\n  Clustering quality metrics:")
            print(f"    Silhouette score:       {silhouette:.4f}")
            print(f"    Davies-Bouldin index:   {davies_bouldin:.4f}")
            print(f"    Calinski-Harabasz:      {calinski:.2f}")
        else:
            silhouette = davies_bouldin = calinski = None
    else:
        silhouette = davies_bouldin = calinski = None
        print(f"\n  Warning: Only {n_clusters} cluster(s) found, quality metrics not computed")
    
    # Print cluster statistics
    print("\n  Cluster characteristics:")
    print(f"  {'Cluster':<10} {'Size':<10} {'SST (°C)':<15} {'Chlor (mg/m³)':<15}")
    print("  " + "-" * 50)
    for label in sorted(set(labels)):
        if label == -1:
            continue
        mask = labels == label
        sst_cluster = sst_valid[mask]
        chlor_cluster = chlor_valid[mask]
        print(f"  {label:<10} {mask.sum():<10} "
              f"{sst_cluster.mean():<7.2f} ± {sst_cluster.std():<5.2f}  "
              f"{chlor_cluster.mean():<7.4f} ± {chlor_cluster.std():<7.4f}")
    
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
        'sst_interpolator': sst_interpolator,
        'training_labels': labels,
        'training_sst': sst_valid,
        'training_chlor': chlor_valid,
        'k_distances': k_distances,  # For k-distance plot
        'sweep_results': sweep_results,  # For parameter sweep analysis
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'calinski_harabasz_score': calinski
    }
    
    output_file = os.path.join(output_dir, 'dbscan_model_optimal_2005_2011_full.pkl')
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
    print(f"    SST loading:      {t_sst_elapsed:8.2f}s ({100*t_sst_elapsed/script_elapsed:5.1f}%)")
    print(f"    Chlor loading:    {t_chlor_elapsed:8.2f}s ({100*t_chlor_elapsed/script_elapsed:5.1f}%)")
    print(f"    Gridding:         {t_grid_elapsed:8.2f}s ({100*t_grid_elapsed/script_elapsed:5.1f}%)")
    print(f"    Param sweep:      {t_optimize_elapsed:8.2f}s ({100*t_optimize_elapsed/script_elapsed:5.1f}%)")
    print(f"    Training:         {t_train_elapsed:8.2f}s ({100*t_train_elapsed/script_elapsed:5.1f}%)")
    print(f"    Saving:           {t_save_elapsed:8.2f}s ({100*t_save_elapsed/script_elapsed:5.1f}%)")
    print("=" * 80)


if __name__ == '__main__':
    main()
