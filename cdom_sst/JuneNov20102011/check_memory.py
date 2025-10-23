import numpy as np

print("=" * 80)
print("ST-DBSCAN MEMORY CALCULATION - WITH fit_frame_split()")
print("=" * 80)
print("Based on actual st_dbscan implementation:")
print("https://github.com/eren-ck/st_dbscan/blob/main/src/st_dbscan/st_dbscan.py")
print()

lat_bins = 200
lon_bins = 300
n_days = 183  # June-Nov
subsample = 1
eps1 = 0.1  # Spatial threshold (in normalized feature space)
eps2 = 10   # Temporal threshold (in days)

# Frame-based processing (as used in updated training script)
frame_size = int(eps2 * 3)  # Process 3x temporal window at a time
frame_overlap = eps2

points_per_day = lat_bins * lon_bins * 0.5 * subsample
total_points = points_per_day * n_days

print(f'Configuration:')
print(f'  Grid: {lat_bins}x{lon_bins}')
print(f'  Days: {n_days}')
print(f'  Subsample: {subsample*100:.0f}%')
print(f'  eps1 (spatial): {eps1}')
print(f'  eps2 (temporal): {eps2} days')
print(f'  Frame size: {frame_size} days')
print(f'  Frame overlap: {frame_overlap} days')
print()
print(f'  Points per day: ~{points_per_day:,.0f}')
print(f'  Total points: ~{total_points:,.0f}')
print()

# Calculate points per frame (NOT total points!)
points_per_frame = points_per_day * frame_size
num_frames = int(np.ceil((n_days - frame_overlap) / (frame_size - frame_overlap)))

print(f'Frame-based processing:')
print(f'  Points per frame: ~{points_per_frame:,.0f}')
print(f'  Number of frames: {num_frames} (processed sequentially)')
print()

# Memory calculation per frame (not total!)
avg_neighbors_spatial_only = min(points_per_frame, int(np.pi * eps1**2 * points_per_frame * 0.5))
memory_spatial_graph = (points_per_frame * avg_neighbors_spatial_only * 16) / 1e9

temporal_fraction = min(1.0, 2 * eps2 / frame_size)
avg_neighbors_temporal_only = int(points_per_frame * temporal_fraction * 0.5)
memory_temporal_graph = (points_per_frame * avg_neighbors_temporal_only * 16) / 1e9

avg_neighbors_combined = int(avg_neighbors_spatial_only * temporal_fraction)
memory_combined = (points_per_frame * avg_neighbors_combined * 16) / 1e9

print("=" * 80)
print("MEMORY REQUIREMENTS PER FRAME")
print("=" * 80)
print(f"Peak memory during computation (PER FRAME, not total):")
print(f"  Spatial graph:  {memory_spatial_graph:8.1f} GB  <- BOTTLENECK!")
print(f"  Temporal graph: {memory_temporal_graph:8.1f} GB")
print(f"  Combined graph: {memory_combined:8.1f} GB")
print()
print(f"  Spatial neighbors/point: ~{avg_neighbors_spatial_only:,}")
print(f"  Temporal window: ±{eps2} days = {temporal_fraction*100:.1f}% of frame")
print(f"  Combined neighbors/point: ~{avg_neighbors_combined:,}")
print()
print(f"Peak RAM needed per frame: ~{max(memory_spatial_graph, memory_temporal_graph):.1f} GB")
print(f"Available RAM: 337 GB")
print()

if max(memory_spatial_graph, memory_temporal_graph) > 337:
    print(f"❌ WILL FAIL - Not enough memory!")
    needed_subsample = subsample * 337 / max(memory_spatial_graph, memory_temporal_graph)
    print(f"   Reduce subsample to {needed_subsample:.3f} ({needed_subsample*100:.1f}%)")
elif max(memory_spatial_graph, memory_temporal_graph) > 200:
    print(f"⚠️  RISKY - Very high memory usage ({max(memory_spatial_graph, memory_temporal_graph)/337*100:.1f}% of RAM)")
elif max(memory_spatial_graph, memory_temporal_graph) > 100:
    print(f"⚠️  CAUTION - Moderate memory usage ({max(memory_spatial_graph, memory_temporal_graph)/337*100:.1f}% of RAM)")
else:
    print(f"✅ SAFE - Should work fine ({max(memory_spatial_graph, memory_temporal_graph)/337*100:.1f}% of RAM)")
    print(f"   Safety margin: {337/max(memory_spatial_graph, memory_temporal_graph):.1f}x")
    
print("=" * 80)
