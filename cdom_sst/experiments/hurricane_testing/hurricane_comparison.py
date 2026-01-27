"""
Hurricane Impact Comparison Visualizer
========================================
For each hurricane in dates.tsv, creates before/after comparison plots
showing SST, Chlorophyll-a, and CDOM one week before and one week after.

Output: 6-panel figures (3 variables x 2 time periods) saved to output/
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
import cmocean
from datetime import datetime, timedelta
import os

# =====================================================================
# CONFIGURATION
# =====================================================================

PICKLE_FILE = r'E:\satdata\Custom\combined_sst_chlor_cdom_2005-2011.pkl'
DATES_FILE = r'f:\Programming\GitHub\nasa-murep-local\cdom_sst\preprocessing\dates.tsv'
OUTPUT_DIR = r'f:\Programming\GitHub\nasa-murep-local\cdom_sst\preprocessing\output'

# SST color scale
SST_VMIN, SST_VMAX = 20, 32

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def load_pickle_data(pickle_path):
    """Load the combined satellite data pickle file."""
    print(f"Loading pickle data from: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data['data'], data['metadata']


def load_hurricane_dates(dates_path):
    """Load hurricane dates from TSV file."""
    print(f"Loading hurricane dates from: {dates_path}")
    df = pd.read_csv(dates_path, sep='\t')
    df['Official start (UTC)'] = pd.to_datetime(df['Official start (UTC)'])
    df['Official end (UTC)'] = pd.to_datetime(df['Official end (UTC)'])
    return df


def extract_temporal_mean(df, column_name, start_date, end_date):
    """Extract 2D arrays from column and compute temporal mean for date range."""
    arrays = []
    
    # Filter dataframe to date range
    mask = (df.index >= start_date) & (df.index <= end_date)
    filtered_df = df[mask]
    
    for date_idx, row in filtered_df.iterrows():
        data = row[column_name]
        if data is not None and not (isinstance(data, float) and np.isnan(data)):
            data_array = np.array(data)
            arrays.append(data_array)
    
    if len(arrays) > 0:
        stacked = np.stack(arrays, axis=0)
        temporal_mean = np.nanmean(stacked, axis=0)
        return temporal_mean, len(arrays)
    else:
        return None, 0


def plot_hurricane_comparison(hurricane_name, before_data, after_data, 
                              metadata, output_path):
    """Create 6-panel comparison plot (3 variables x before/after)."""
    
    # Unpack metadata
    lat_edges = metadata['lat_edges']
    lon_edges = metadata['lon_edges']
    lon_min, lon_max, lat_min, lat_max = metadata['bbox']
    
    # Unpack data grids and counts
    sst_before, sst_before_count = before_data['sst']
    chlor_before, chlor_before_count = before_data['chlor']
    cdom_before, cdom_before_count = before_data['cdom']
    
    sst_after, sst_after_count = after_data['sst']
    chlor_after, chlor_after_count = after_data['chlor']
    cdom_after, cdom_after_count = after_data['cdom']
    
    # Create figure with 3 rows x 2 columns
    fig = plt.figure(figsize=(16, 18))
    
    # ===== ROW 1: SST =====
    
    # SST Before
    ax1 = plt.subplot(3, 2, 1, projection=ccrs.PlateCarree())
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    if sst_before is not None:
        mesh1 = ax1.pcolormesh(
            lon_edges, lat_edges, sst_before,
            cmap=cmocean.cm.thermal, shading='auto',
            vmin=SST_VMIN, vmax=SST_VMAX,
            transform=ccrs.PlateCarree()
        )
        plt.colorbar(mesh1, ax=ax1, label='SST (°C)', shrink=0.7)
    
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='dimGray')
    ax1.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
    ax1.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='dimGray')
    ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    ax1.set_title(f'SST - Week Before\n({sst_before_count} days averaged)', fontsize=11, fontweight='bold')
    
    # SST After
    ax2 = plt.subplot(3, 2, 2, projection=ccrs.PlateCarree())
    ax2.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    if sst_after is not None:
        mesh2 = ax2.pcolormesh(
            lon_edges, lat_edges, sst_after,
            cmap=cmocean.cm.thermal, shading='auto',
            vmin=SST_VMIN, vmax=SST_VMAX,
            transform=ccrs.PlateCarree()
        )
        plt.colorbar(mesh2, ax=ax2, label='SST (°C)', shrink=0.7)
    
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='dimGray')
    ax2.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
    ax2.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='dimGray')
    ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    ax2.set_title(f'SST - Week After\n({sst_after_count} days averaged)', fontsize=11, fontweight='bold')
    
    # ===== ROW 2: Chlorophyll =====
    
    # Chlorophyll Before
    ax3 = plt.subplot(3, 2, 3, projection=ccrs.PlateCarree())
    ax3.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    if chlor_before is not None:
        valid_chlor = chlor_before[~np.isnan(chlor_before)]
        if len(valid_chlor) > 0:
            vmin = np.percentile(valid_chlor, 5)
            vmax = np.percentile(valid_chlor, 95)
            if vmin <= 0:
                vmin = 0.01
            if vmax <= vmin:
                vmax = vmin * 10
        else:
            vmin, vmax = 0.01, 10
            
        mesh3 = ax3.pcolormesh(
            lon_edges, lat_edges, chlor_before,
            cmap='viridis', shading='auto',
            norm=colors.LogNorm(vmin=vmin, vmax=vmax),
            transform=ccrs.PlateCarree()
        )
        plt.colorbar(mesh3, ax=ax3, label='Chlorophyll-a (mg/m³)', shrink=0.7)
    
    ax3.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
    ax3.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
    ax3.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
    ax3.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    ax3.set_title(f'Chlorophyll-a - Week Before\n({chlor_before_count} days averaged)', fontsize=11, fontweight='bold')
    
    # Chlorophyll After
    ax4 = plt.subplot(3, 2, 4, projection=ccrs.PlateCarree())
    ax4.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    if chlor_after is not None:
        valid_chlor = chlor_after[~np.isnan(chlor_after)]
        if len(valid_chlor) > 0:
            vmin = np.percentile(valid_chlor, 5)
            vmax = np.percentile(valid_chlor, 95)
            if vmin <= 0:
                vmin = 0.01
            if vmax <= vmin:
                vmax = vmin * 10
        else:
            vmin, vmax = 0.01, 10
            
        mesh4 = ax4.pcolormesh(
            lon_edges, lat_edges, chlor_after,
            cmap='viridis', shading='auto',
            norm=colors.LogNorm(vmin=vmin, vmax=vmax),
            transform=ccrs.PlateCarree()
        )
        plt.colorbar(mesh4, ax=ax4, label='Chlorophyll-a (mg/m³)', shrink=0.7)
    
    ax4.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
    ax4.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
    ax4.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
    ax4.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    ax4.set_title(f'Chlorophyll-a - Week After\n({chlor_after_count} days averaged)', fontsize=11, fontweight='bold')
    
    # ===== ROW 3: CDOM =====
    
    # CDOM Before
    ax5 = plt.subplot(3, 2, 5, projection=ccrs.PlateCarree())
    ax5.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    if cdom_before is not None:
        valid_cdom = cdom_before[~np.isnan(cdom_before)]
        if len(valid_cdom) > 0:
            vmin = np.percentile(valid_cdom, 5)
            vmax = np.percentile(valid_cdom, 95)
            if vmin >= vmax:
                vmin = np.nanmin(valid_cdom)
                vmax = np.nanmax(valid_cdom)
            if vmin == vmax:
                vmin -= 0.01
                vmax += 0.01
        else:
            vmin, vmax = 0, 0.1
            
        mesh5 = ax5.pcolormesh(
            lon_edges, lat_edges, cdom_before,
            cmap='viridis', shading='auto',
            vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree()
        )
        plt.colorbar(mesh5, ax=ax5, label='CDOM Index', shrink=0.7)
    
    ax5.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='dimGray')
    ax5.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
    ax5.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='dimGray')
    ax5.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    ax5.set_title(f'CDOM - Week Before\n({cdom_before_count} days averaged)', fontsize=11, fontweight='bold')
    
    # CDOM After
    ax6 = plt.subplot(3, 2, 6, projection=ccrs.PlateCarree())
    ax6.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    if cdom_after is not None:
        valid_cdom = cdom_after[~np.isnan(cdom_after)]
        if len(valid_cdom) > 0:
            vmin = np.percentile(valid_cdom, 5)
            vmax = np.percentile(valid_cdom, 95)
            if vmin >= vmax:
                vmin = np.nanmin(valid_cdom)
                vmax = np.nanmax(valid_cdom)
            if vmin == vmax:
                vmin -= 0.01
                vmax += 0.01
        else:
            vmin, vmax = 0, 0.1
            
        mesh6 = ax6.pcolormesh(
            lon_edges, lat_edges, cdom_after,
            cmap='viridis', shading='auto',
            vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree()
        )
        plt.colorbar(mesh6, ax=ax6, label='CDOM Index', shrink=0.7)
    
    ax6.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='dimGray')
    ax6.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
    ax6.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='dimGray')
    ax6.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    ax6.set_title(f'CDOM - Week After\n({cdom_after_count} days averaged)', fontsize=11, fontweight='bold')
    
    # Overall title
    fig.suptitle(f'{hurricane_name}\nBefore vs After Comparison', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


# =====================================================================
# MAIN PROCESSING
# =====================================================================

def main():
    """Main processing pipeline."""
    print("\n" + "="*70)
    print("Hurricane Impact Comparison Visualizer")
    print("="*70)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load data
    df, metadata = load_pickle_data(PICKLE_FILE)
    hurricane_df = load_hurricane_dates(DATES_FILE)
    
    print(f"\nLoaded {len(df)} days of satellite data")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"\nFound {len(hurricane_df)} hurricanes/storms to process\n")
    
    # Process each hurricane
    skipped_count = 0
    processed_count = 0
    
    for idx, row in hurricane_df.iterrows():
        storm_name = row['Storm (season)']
        start_date = row['Official start (UTC)']
        end_date = row['Official end (UTC)']
        
        print(f"\n{'='*70}")
        print(f"Processing: {storm_name}")
        print(f"  Official dates: {start_date.date()} to {end_date.date()}")
        
        # Calculate before/after date ranges (1 week = 7 days)
        before_start = start_date - timedelta(days=7)
        before_end = start_date - timedelta(days=1)
        
        after_start = end_date + timedelta(days=1)
        after_end = end_date + timedelta(days=7)
        
        print(f"  Week before: {before_start.date()} to {before_end.date()}")
        print(f"  Week after: {after_start.date()} to {after_end.date()}")
        
        # Extract data for before period
        sst_before, sst_before_count = extract_temporal_mean(df, 'sst', before_start, before_end)
        chlor_before, chlor_before_count = extract_temporal_mean(df, 'chlorophyll', before_start, before_end)
        cdom_before, cdom_before_count = extract_temporal_mean(df, 'cdom', before_start, before_end)
        
        before_data = {
            'sst': (sst_before, sst_before_count),
            'chlor': (chlor_before, chlor_before_count),
            'cdom': (cdom_before, cdom_before_count)
        }
        
        # Extract data for after period
        sst_after, sst_after_count = extract_temporal_mean(df, 'sst', after_start, after_end)
        chlor_after, chlor_after_count = extract_temporal_mean(df, 'chlorophyll', after_start, after_end)
        cdom_after, cdom_after_count = extract_temporal_mean(df, 'cdom', after_start, after_end)
        
        after_data = {
            'sst': (sst_after, sst_after_count),
            'chlor': (chlor_after, chlor_after_count),
            'cdom': (cdom_after, cdom_after_count)
        }
        
        print(f"  Data availability:")
        print(f"    Before - SST: {sst_before_count} days, Chlor: {chlor_before_count} days, CDOM: {cdom_before_count} days")
        print(f"    After  - SST: {sst_after_count} days, Chlor: {chlor_after_count} days, CDOM: {cdom_after_count} days")
        
        # Check if we have enough data to make meaningful plots
        # Require at least 1 day of SST data in either before or after period
        total_sst_days = sst_before_count + sst_after_count
        total_all_days = (sst_before_count + sst_after_count + 
                         chlor_before_count + chlor_after_count + 
                         cdom_before_count + cdom_after_count)
        
        if total_all_days == 0:
            print(f"  ⚠ SKIPPED: No data available for any variable in either time period")
            print(f"  → This likely means the pickle file is missing data for this date range")
            skipped_count += 1
            continue
        elif total_sst_days == 0:
            print(f"  ⚠ WARNING: No SST data available (only ocean color products)")
        
        # Create sanitized filename
        safe_name = storm_name.replace(' ', '_').replace('(', '').replace(')', '')
        output_filename = f"hurricane_comparison_{safe_name}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Generate plot
        plot_hurricane_comparison(storm_name, before_data, after_data, metadata, output_path)
        processed_count += 1
    
    print(f"\n{'='*70}")
    print("✓ Hurricane comparison processing complete!")
    print(f"  Processed: {processed_count} storms")
    print(f"  Skipped: {skipped_count} storms (no data available)")
    if skipped_count > 0:
        print(f"\n  ⚠ WARNING: Some storms were skipped due to missing data in pickle file")
        print(f"  → Check if source data exists and re-run pickler for those years")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
