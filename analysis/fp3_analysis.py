#!/usr/bin/env python3
"""
FP.3 Keypoint Match Filtering Analysis Script

Analyzes the effect of Euclidean distance-based outlier removal on keypoint matches.
Compares match counts before and after filtering, and examines distance distributions.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def get_tracked_vehicle_track_id(track_id_file=None):
    """
    Read the tracked preceding vehicle track_id from file.
    
    Args:
        track_id_file: Path to the file containing the track_id (default: output/tracked_preceding_vehicle_track_id.txt)
    
    Returns:
        The track_id as an integer, or None if not found
    """
    if track_id_file is None:
        track_id_file = "output/tracked_preceding_vehicle_track_id.txt"
    
    if os.path.exists(track_id_file):
        try:
            with open(track_id_file, 'r') as f:
                track_id = int(f.read().strip())
                return track_id
        except (ValueError, IOError) as e:
            print(f"Warning: Could not read track_id from {track_id_file}: {e}")
            return None
    else:
        print(f"Warning: track_id file not found at {track_id_file}")
        return None


def load_data(csv_path):
    """Load keypoint match filtering data from CSV."""
    df = pd.read_csv(csv_path)
    return df


def plot_comparison(df, output_prefix="fp3_kpt"):
    """Generate comparison plots for keypoint match filtering."""
    
    # Create output directory if needed
    os.makedirs("output", exist_ok=True)
    
    # Filter for preceding vehicle using track_id from file
    tracked_track_id = get_tracked_vehicle_track_id()
    
    if 'track_id' in df.columns and tracked_track_id is not None:
        # Use the track_id from the file
        track_data = df[df['track_id'] == tracked_track_id]
        if len(track_data) > 0:
            df_preceding = track_data.copy()
            print(f"  Using track_id={tracked_track_id} (from tracked_preceding_vehicle_track_id.txt) for analysis")
        else:
            print(f"  Warning: track_id={tracked_track_id} not found in data, using all data")
            df_preceding = df.copy()
    elif 'track_id' in df.columns:
        # track_id file not found, fallback to most common track_id
        track_ids = df['track_id'].mode()
        if len(track_ids) > 0:
            tracked_track_id = track_ids[0]
            df_preceding = df[df['track_id'] == tracked_track_id].copy()
            print(f"  Warning: track_id file not found, using most common track_id={tracked_track_id}")
        else:
            df_preceding = df.copy()
            print("  Warning: No track_id data found, using all data")
    else:
        # Fallback: try boxID 1 then 0 (legacy behavior)
        print("  Warning: No track_id column found, falling back to box_id filtering")
        df_preceding = df[df['box_id'] == 1].copy()
        if len(df_preceding) == 0:
            df_preceding = df[df['box_id'] == 0].copy()
        if len(df_preceding) == 0:
            df_preceding = df.copy()
            print("  Warning: No data for specific box ID, using all data")
    
    # Create figure with 3 subplots (removed subplot 3, split subplot 4 into two)
    plt.figure(figsize=(16, 12))
    plt.suptitle('FP.3: Keypoint Match Displacement Filtering Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Match counts before vs after filtering
    plt.subplot(2, 2, 1)
    frames = df_preceding['frame_index']
    before = df_preceding['matches_before']
    after = df_preceding['matches_after']
    
    plt.plot(frames, before, 'o-', label='Before Filtering', color='red', linewidth=2, markersize=4)
    plt.plot(frames, after, 'o-', label='After Filtering', color='green', linewidth=2, markersize=4)
    plt.title('Keypoint Match Counts: Before vs After Displacement Filtering')
    plt.xlabel('Frame Index')
    plt.ylabel('Number of Matches')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Plot 2: Outlier removal percentage
    plt.subplot(2, 2, 2)
    plt.plot(frames, df_preceding['outliers_removed_pct'], 'o-', color='blue', linewidth=2, markersize=4)
    plt.axhline(y=5.0, color='red', linestyle='--', alpha=0.5, label='5%')
    plt.title('Matches Removed by Displacement Threshold')
    plt.xlabel('Frame Index')
    plt.ylabel('Percentage Removed (%)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Compute global y-axis limits for consistent scaling between subplots
    global_y_min = 0
    global_y_max = 100  # Default range
    
    if 'mean_distance_unfiltered' in df_preceding.columns and 'min_distance_unfiltered' in df_preceding.columns:
        mean_unfiltered = df_preceding['mean_distance_unfiltered']
        median_unfiltered = df_preceding['median_distance_unfiltered']
        min_unfiltered = df_preceding['min_distance_unfiltered']
        max_unfiltered = df_preceding['max_distance_unfiltered']
        
        # Compute combined range
        all_unfiltered = np.concatenate([mean_unfiltered.values, median_unfiltered.values, 
                                         min_unfiltered.values, max_unfiltered.values])
        global_y_min = min(global_y_min, np.min(all_unfiltered))
        global_y_max = max(global_y_max, np.max(all_unfiltered))
    
    if 'mean_distance' in df_preceding.columns and 'min_distance' in df_preceding.columns:
        mean_filtered = df_preceding['mean_distance']
        median_filtered = df_preceding['median_distance']
        min_filtered = df_preceding['min_distance']
        max_filtered = df_preceding['max_distance']
        
        # Compute combined range
        all_filtered = np.concatenate([mean_filtered.values, median_filtered.values, 
                                       min_filtered.values, max_filtered.values])
        global_y_min = min(global_y_min, np.min(all_filtered))
        global_y_max = max(global_y_max, np.max(all_filtered))
    
    # Add 10% padding for better visualization
    y_range = global_y_max - global_y_min
    global_y_min -= y_range * 0.1
    global_y_max += y_range * 0.1
    
    # Plot 3: Unfiltered distance statistics (FP.1) - without standard deviation
    plt.subplot(2, 2, 3)
    if 'mean_distance_unfiltered' in df_preceding.columns and 'min_distance_unfiltered' in df_preceding.columns:
        mean_unfiltered = df_preceding['mean_distance_unfiltered']
        median_unfiltered = df_preceding['median_distance_unfiltered']
        min_unfiltered = df_preceding['min_distance_unfiltered']
        max_unfiltered = df_preceding['max_distance_unfiltered']
        
        plt.plot(frames, mean_unfiltered, 'o-', label='Mean', color='blue', linewidth=2, markersize=4)
        plt.plot(frames, median_unfiltered, 's-', label='Median', color='green', linewidth=2, markersize=4)
        plt.plot(frames, min_unfiltered, ':', color='purple', linewidth=2, label='Min')
        plt.plot(frames, max_unfiltered, ':', color='purple', linewidth=2, label='Max')
        
        plt.title('FP.1: Unfiltered Distance Statistics (Mean, Median, Min, Max)')
        plt.xlabel('Frame Index')
        plt.ylabel('Distance (pixels)')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_ylim(global_y_min, global_y_max)
    else:
        plt.title('FP.1: Unfiltered data not available')
        plt.xlabel('Frame Index')
        plt.ylabel('Distance (pixels)')
        plt.grid(True, alpha=0.3)
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_ylim(global_y_min, global_y_max)
    
    # Plot 4: Filtered distance statistics (FP.3) - without standard deviation
    plt.subplot(2, 2, 4)
    if 'mean_distance' in df_preceding.columns and 'min_distance' in df_preceding.columns:
        mean_filtered = df_preceding['mean_distance']
        median_filtered = df_preceding['median_distance']
        min_filtered = df_preceding['min_distance']
        max_filtered = df_preceding['max_distance']
        
        plt.plot(frames, mean_filtered, 'o-', label='Mean', color='blue', linewidth=2, markersize=4)
        plt.plot(frames, median_filtered, 's-', label='Median', color='green', linewidth=2, markersize=4)
        plt.plot(frames, min_filtered, ':', color='purple', linewidth=2, label='Min')
        plt.plot(frames, max_filtered, ':', color='purple', linewidth=2, label='Max')
        
        plt.title('FP.3: Filtered Distance Statistics (Mean, Median, Min, Max)')
        plt.xlabel('Frame Index')
        plt.ylabel('Distance (pixels)')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_ylim(global_y_min, global_y_max)
    else:
        plt.title('FP.3: Filtered data not available')
        plt.xlabel('Frame Index')
        plt.ylabel('Distance (pixels)')
        plt.grid(True, alpha=0.3)
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_ylim(global_y_min, global_y_max)
    
    plt.tight_layout()
    
    # Save plots
    output_path = f"../analysis/output/{output_prefix}_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {output_path}")
    
    return output_path


def print_statistics(df):
    """Print summary statistics for keypoint match displacement filtering."""
    
    print("\n" + "="*80)
    print("FP.3: KEYPOINT MATCH DISPLACEMENT FILTERING STATISTICS")
    print("="*80)
    
    # Overall statistics
    total_before = df['matches_before'].sum()
    total_after = df['matches_after'].sum()
    total_removed = total_before - total_after
    avg_removal_pct = df['outliers_removed_pct'].mean()
    
    print(f"\nOverall Statistics:")
    print(f"  Total matches before filtering: {total_before}")
    print(f"  Total matches after filtering: {total_after}")
    print(f"  Total matches removed: {total_removed} ({100*total_removed/total_before:.1f}%)")
    print(f"  Average removal rate per frame: {avg_removal_pct:.1f}%")
    
    # Per-frame statistics
    print(f"\nPer-Frame Statistics:")
    print(f"  Mean matches before: {df['matches_before'].mean():.1f}")
    print(f"  Mean matches after: {df['matches_after'].mean():.1f}")
    print(f"  Median matches before: {df['matches_before'].median():.1f}")
    print(f"  Median matches after: {df['matches_after'].median():.1f}")
    
    # Distance statistics
    print(f"\nDisplacement Distance Statistics:")
    print(f"  Mean distance: {df['mean_distance'].mean():.2f} pixels")
    print(f"  Median distance: {df['median_distance'].median():.2f} pixels")
    if 'stddev_distance' in df.columns:
        print(f"  Mean std dev: {df['stddev_distance'].mean():.2f} pixels")
    
    # Unfiltered distance statistics (if available)
    if 'mean_distance_unfiltered' in df.columns:
        print(f"\nUnfiltered Distance Statistics:")
        print(f"  Mean distance: {df['mean_distance_unfiltered'].mean():.2f} pixels")
        print(f"  Median distance: {df['median_distance_unfiltered'].median():.2f} pixels")
        if 'min_distance_unfiltered' in df.columns and 'max_distance_unfiltered' in df.columns:
            print(f"  Min distance: {df['min_distance_unfiltered'].min():.2f} pixels")
            print(f"  Max distance: {df['max_distance_unfiltered'].max():.2f} pixels")
    
    # Box ID statistics
    print(f"\nBox ID Distribution:")
    print(f"  Unique box IDs: {df['box_id'].nunique()}")
    print(f"  Box ID counts:\n{df['box_id'].value_counts().head(10)}")


def main():
    parser = argparse.ArgumentParser(
        description='FP.3 Keypoint Match Displacement Filtering Analysis'
    )
    parser.add_argument('--csv', type=str, default='output/kpt_matches_filtering.csv',
                       help='Path to keypoint match displacement filtering CSV file (default: output/kpt_matches_filtering.csv)')
    parser.add_argument('--output', type=str, default='fp3_kpt',
                       help='Output prefix for plots (default: fp3_kpt, saves to output/fp3_kpt_displacement_comparison.png)')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively')
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.csv)
    
    # Print info
    print(f"Loaded {len(df)} keypoint match displacement filtering records from {args.csv}")
    print(f"Frames: {df['frame_index'].min()} to {df['frame_index'].max()}")
    print(f"Columns: {list(df.columns)}")
    
    # Print statistics
    print_statistics(df)
    
    # Generate plots
    plot_path = plot_comparison(df, args.output)
    
    if args.show:
        plt.show()
    
    print(f"\nAnalysis complete!")
    print(f"Plots saved to: {plot_path}")


if __name__ == '__main__':
    main()
