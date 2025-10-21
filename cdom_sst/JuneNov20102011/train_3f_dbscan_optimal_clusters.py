r"""
DBSCAN Clustering Training Script with 3 Features: SST, Chlorophyll, and CDOM Slope
===================================================================================
Trains DBSCAN clustering model on:
- MUR L4 SST data
- MODIS L2 chlorophyll data  
- MODIS L2 RRS data (for CDOM spectral slope calculation)

Features:
1. SST (°C)
2. Log10(Chlorophyll) (mg/m³)
3. CDOM Spectral Slope (nm⁻¹) - Configurable S_275:295 or S_300:600

Training period: 2010-2011 (June-November) for the Texas-Louisiana Shelf

Uses Mannino et al. Multiple Linear Regression (MLR) algorithm for CDOM spectral slopes:
- S_275:295: Ln[S] = -3.258 + 0.336 × Ln[Rrs(443)] - 0.279 × Ln[Rrs(547)]
- S_300:600: Ln[S] = -3.640 + 0.186 × Ln[Rrs(443)] - 0.146 × Ln[Rrs(547)]

Saves trained model to E:\satdata\Custom
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
from enum import Enum

# Add parent directory to path for pipeline imports
sys.dont_write_bytecode = True
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, "..")
sys.path.insert(0, parent_dir)

from pipelines.l3_pipeline import L3DatasetLoader
from pipelines.l2_pipeline import L2DatasetLoader, GPUDataExtractor

import warnings
warnings.filterwarnings('ignore')


class CDOMSlopeType(Enum):
    """Enum for CDOM spectral slope types"""
    S275_295 = "S275_295"
    S300_600 = "S300_600"


# CDOM Spectral Slope Calculation Constants using Mannino et al. Algorithm
CDOM_COEFFICIENTS = {
    CDOMSlopeType.S275_295: {
        'B0': -3.258,
        'B1': 0.336,
        'B2': -0.279,
        'wavelengths': [443, 547],
        'description': 'S_275:295 nm⁻¹'
    },
    CDOMSlopeType.S300_600: {
        'B0': -3.640,
        'B1': 0.186,
        'B2': -0.146,
        'wavelengths': [443, 547],
        'description': 'S_300:600 nm⁻¹'
    }
}


def calculate_cdom_slope(rrs_443, rrs_547, slope_type: CDOMSlopeType):
    """
    Calculate CDOM spectral slope using Mannino et al. MLR algorithm.
    
    Parameters:
    -----------
    rrs_443 : np.ndarray
        Remote sensing reflectance at 443 nm
    rrs_547 : np.ndarray
        Remote sensing reflectance at 547 nm
    slope_type : CDOMSlopeType
        Type of spectral slope to calculate (S275_295 or S300_600)
    
    Returns:
    --------
    np.ndarray
        CDOM spectral slope values in nm⁻¹
    """
    coeffs = CDOM_COEFFICIENTS[slope_type]
    
    # Initialize with NaN
    cdom_slope = np.full_like(rrs_443, np.nan, dtype=float)
    
    # Mask for valid data (positive values, no NaNs)
    valid_mask = (~np.isnan(rrs_443) & ~np.isnan(rrs_547) & 
                  (rrs_443 > 0) & (rrs_547 > 0))
    
    if np.any(valid_mask):
        # Apply Mannino et al. MLR algorithm
        # Ln[S] = B0 + B1 × Ln[Rrs(443)] + B2 × Ln[Rrs(547)]
        ln_rrs_443 = np.log(rrs_443[valid_mask])
        ln_rrs_547 = np.log(rrs_547[valid_mask])
        
        ln_slope = (coeffs['B0'] + 
                   coeffs['B1'] * ln_rrs_443 + 
                   coeffs['B2'] * ln_rrs_547)
        
        # Convert back to linear space: S = exp(Ln[S])
        cdom_slope[valid_mask] = np.exp(ln_slope)
    
    return cdom_slope


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
    print("DBSCAN CLUSTERING TRAINING WITH 3 FEATURES (2010-2011)")
    print("Features: SST + Log10(Chlorophyll) + CDOM Spectral Slope")
    print("=" * 80)
    
    # Configuration
    lon_min, lon_max = -94.0, -88.0
    lat_min, lat_max = 27.5, 30.5
    lat_bins = 200
    lon_bins = 300
    
    # CDOM slope configuration - CHANGE THIS TO SWITCH BETWEEN SLOPE TYPES
    SLOPE_TYPE = CDOMSlopeType.S300_600  # or CDOMSlopeType.S275_295
    slope_desc = CDOM_COEFFICIENTS[SLOPE_TYPE]['description']
    
    sst_base_pattern = r"E:\satdata\MUR-JPL-L4-GLOB-v4.1_Texas Louisiana Shelf_{year}-06-01_{year}-11-30"
    chlor_base_pattern = r"E:\satdata\Texas Louisiana Shelf_{year}-06-01_{year}-11-30"
    rrs_base_pattern = r"E:\satdata\Texas Louisiana Shelf_{year}-06-01_{year}-11-30"  # Same as chlor
    output_dir = r"E:\satdata\Custom"
    
    sst_var = 'analysed_sst'
    chlor_var = 'chlor_a'
    chlor_group = 'geophysical_data'
    rrs_vars = ['Rrs_443', 'Rrs_547']
    rrs_group = 'geophysical_data'
    
    # DBSCAN parameters
    min_samples = 4
    years = [2010, 2011]  # June-November 2010 and 2011

    print(f"\nConfiguration:")
    print(f"  Region: [{lon_min}, {lat_min}] to [{lon_max}, {lat_max}]")
    print(f"  Grid: {lat_bins} x {lon_bins}")
    print(f"  Years: {years}")
    print(f"  CDOM Slope Type: {slope_desc}")
    print(f"  Min samples (k): {min_samples}")
    print(f"  Method: Parameter sweep with stability analysis")
    
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
    
    # Find and load RRS files for CDOM slope
    print("\n" + "=" * 80)
    print("LOADING RRS DATA FOR CDOM SLOPE")
    print("=" * 80)
    
    t_rrs_start = time.time()
    
    rrs_files = []
    for year in years:
        rrs_dir = rrs_base_pattern.format(year=year)
        year_files = glob.glob(os.path.join(rrs_dir, '**', '*.nc'), recursive=True)
        rrs_files.extend(year_files)
        print(f"  Year {year}: {len(year_files)} files")
    
    print(f"  Total RRS files: {len(rrs_files)}")
    
    # Load RRS datasets
    rrs_loader = L2DatasetLoader(variables=rrs_vars, group=rrs_group)
    rrs_datasets = rrs_loader.load_multiple(rrs_files)
    print(f"  Loaded {len(rrs_datasets)} RRS datasets")
    
    # Extract RRS data
    print("  Extracting RRS data...")
    try:
        rrs_extractor = GPUDataExtractor(variables=rrs_vars)
        lon_rrs_flat, lat_rrs_flat, rrs_data = rrs_extractor.extract(rrs_datasets)
        rrs_443_flat = rrs_data['Rrs_443']
        rrs_547_flat = rrs_data['Rrs_547']
        print(f"  GPU extraction: {len(rrs_443_flat)} points")
    except Exception as e:
        print(f"  GPU failed ({e}), using CPU...")
        all_lons_rrs, all_lats_rrs, all_rrs_443, all_rrs_547 = [], [], [], []
        for ds in tqdm(rrs_datasets, desc="  Extracting RRS", unit="file"):
            lat = ds['lat'].values.flatten()
            lon = ds['lon'].values.flatten()
            rrs_443 = ds['Rrs_443'].values.flatten()
            rrs_547 = ds['Rrs_547'].values.flatten()
            valid = (~np.isnan(lat) & ~np.isnan(lon) & 
                    ~np.isnan(rrs_443) & ~np.isnan(rrs_547) & 
                    (rrs_443 > 0) & (rrs_547 > 0))
            all_lats_rrs.extend(lat[valid])
            all_lons_rrs.extend(lon[valid])
            all_rrs_443.extend(rrs_443[valid])
            all_rrs_547.extend(rrs_547[valid])
        lat_rrs_flat = np.array(all_lats_rrs)
        lon_rrs_flat = np.array(all_lons_rrs)
        rrs_443_flat = np.array(all_rrs_443)
        rrs_547_flat = np.array(all_rrs_547)
        print(f"  CPU extraction: {len(rrs_443_flat)} points")
    
    # Calculate CDOM slope
    print(f"  Calculating CDOM spectral slope ({slope_desc})...")
    cdom_slope_flat = calculate_cdom_slope(rrs_443_flat, rrs_547_flat, SLOPE_TYPE)
    valid_slope_count = np.sum(~np.isnan(cdom_slope_flat))
    print(f"  Valid CDOM slope values: {valid_slope_count}")
    if valid_slope_count > 0:
        print(f"  CDOM slope range: {np.nanmin(cdom_slope_flat):.6f} to {np.nanmax(cdom_slope_flat):.6f} nm⁻¹")
    
    t_rrs_elapsed = time.time() - t_rrs_start
    print(f"  ⏱ RRS/CDOM processing completed in {t_rrs_elapsed:.2f}s")
    
    # Grid chlorophyll and CDOM slope data
    print("\n" + "=" * 80)
    print("GRIDDING DATA")
    print("=" * 80)
    
    t_grid_start = time.time()
    
    # Clip chlorophyll to region
    clip_mask_chlor = ((lon_chlor_flat >= lon_min) & (lon_chlor_flat <= lon_max) &
                       (lat_chlor_flat >= lat_min) & (lat_chlor_flat <= lat_max))
    
    lon_chlor_clipped = lon_chlor_flat[clip_mask_chlor]
    lat_chlor_clipped = lat_chlor_flat[clip_mask_chlor]
    chlor_clipped = chlor_flat[clip_mask_chlor]
    
    print(f"  Chlorophyll points in region: {len(chlor_clipped)}")
    
    # Bin chlorophyll
    chlor_binned, lat_edges, lon_edges, _ = binned_statistic_2d(
        lat_chlor_clipped, lon_chlor_clipped, chlor_clipped,
        statistic='mean',
        bins=[lat_bins, lon_bins],
        range=[[lat_min, lat_max], [lon_min, lon_max]]
    )
    
    print(f"  Valid chlorophyll pixels: {np.sum(~np.isnan(chlor_binned))}")
    
    # Clip CDOM slope to region
    clip_mask_rrs = ((lon_rrs_flat >= lon_min) & (lon_rrs_flat <= lon_max) &
                     (lat_rrs_flat >= lat_min) & (lat_rrs_flat <= lat_max))
    
    lon_rrs_clipped = lon_rrs_flat[clip_mask_rrs]
    lat_rrs_clipped = lat_rrs_flat[clip_mask_rrs]
    cdom_slope_clipped = cdom_slope_flat[clip_mask_rrs]
    
    print(f"  CDOM slope points in region: {len(cdom_slope_clipped)}")
    
    # Bin CDOM slope using same grid
    cdom_slope_binned, _, _, _ = binned_statistic_2d(
        lat_rrs_clipped, lon_rrs_clipped, cdom_slope_clipped,
        statistic='mean',
        bins=[lat_bins, lon_bins],
        range=[[lat_min, lat_max], [lon_min, lon_max]]
    )
    
    print(f"  Valid CDOM slope pixels: {np.sum(~np.isnan(cdom_slope_binned))}")
    
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
    print("PREPARING FEATURES (3D)")
    print("=" * 80)
    
    valid_mask = (~np.isnan(sst_regridded) & ~np.isnan(chlor_binned) & ~np.isnan(cdom_slope_binned))
    n_valid = np.sum(valid_mask)
    print(f"  Pixels with SST, chlorophyll, and CDOM slope: {n_valid}")
    
    sst_valid = sst_regridded[valid_mask]
    chlor_valid = chlor_binned[valid_mask]
    cdom_slope_valid = cdom_slope_binned[valid_mask]
    
    # Create feature matrix (N x 3) with log-transformed chlorophyll
    features = np.column_stack([sst_valid, np.log10(chlor_valid), cdom_slope_valid])
    
    print(f"  Feature matrix shape: {features.shape}")
    print(f"  SST mean: {sst_valid.mean():.2f} °C, std: {sst_valid.std():.2f}")
    print(f"  Chlorophyll mean: {chlor_valid.mean():.6f} mg/m³, std: {chlor_valid.std():.6f}")
    print(f"  CDOM slope mean: {cdom_slope_valid.mean():.6f} nm⁻¹, std: {cdom_slope_valid.std():.6f}")
    
    # Standardize features for DBSCAN
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    print(f"\n  Scaled features shape: {features_scaled.shape}")
    print(f"  Feature 0 (SST) mean: {features_scaled[:, 0].mean():.4f}, std: {features_scaled[:, 0].std():.4f}")
    print(f"  Feature 1 (log10 Chlor) mean: {features_scaled[:, 1].mean():.4f}, std: {features_scaled[:, 1].std():.4f}")
    print(f"  Feature 2 (CDOM slope) mean: {features_scaled[:, 2].mean():.4f}, std: {features_scaled[:, 2].std():.4f}")
    
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
    print(f"  {'Cluster':<10} {'Size':<10} {'SST (°C)':<18} {'Chlor (mg/m³)':<18} {'Slope (nm⁻¹)':<18}")
    print("  " + "-" * 76)
    for label in sorted(set(labels)):
        if label == -1:
            continue
        mask = labels == label
        sst_cluster = sst_valid[mask]
        chlor_cluster = chlor_valid[mask]
        slope_cluster = cdom_slope_valid[mask]
        print(f"  {label:<10} {mask.sum():<10} "
              f"{sst_cluster.mean():<7.2f} ± {sst_cluster.std():<8.2f} "
              f"{chlor_cluster.mean():<7.4f} ± {chlor_cluster.std():<8.4f} "
              f"{slope_cluster.mean():<7.5f} ± {slope_cluster.std():<8.5f}")
    
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
        'training_cdom_slope': cdom_slope_valid,
        'k_distances': k_distances,
        'sweep_results': sweep_results,
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'calinski_harabasz_score': calinski,
        'slope_type': SLOPE_TYPE.value,
        'slope_description': slope_desc,
        'cdom_coefficients': CDOM_COEFFICIENTS[SLOPE_TYPE],
        'n_features': 3
    }
    
    slope_type_str = SLOPE_TYPE.value
    output_file = os.path.join(output_dir, f'dbscan_model_3f_{slope_type_str}_2010_2011.pkl')
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
    print(f"    RRS/CDOM loading: {t_rrs_elapsed:8.2f}s ({100*t_rrs_elapsed/script_elapsed:5.1f}%)")
    print(f"    Gridding:         {t_grid_elapsed:8.2f}s ({100*t_grid_elapsed/script_elapsed:5.1f}%)")
    print(f"    Param sweep:      {t_optimize_elapsed:8.2f}s ({100*t_optimize_elapsed/script_elapsed:5.1f}%)")
    print(f"    Training:         {t_train_elapsed:8.2f}s ({100*t_train_elapsed/script_elapsed:5.1f}%)")
    print(f"    Saving:           {t_save_elapsed:8.2f}s ({100*t_save_elapsed/script_elapsed:5.1f}%)")
    print("=" * 80)


if __name__ == '__main__':
    main()
