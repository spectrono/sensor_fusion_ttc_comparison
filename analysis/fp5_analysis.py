#!/usr/bin/env python3
"""
FP.5 Performance Assessment 1 Analysis Script

Identifies and analyzes frames where LIDAR TTC estimation appears implausible.
Compares camera and LIDAR TTC measurements to find discrepancies.

Usage:
    python fp5_analysis.py [--camera-csv output/ttc_camera.csv] [--lidar-csv output/ttc_lidar_comparison.csv]
                         [--output output] [--show] [--threshold 0.2]
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def get_tracked_vehicle_track_id(track_id_file=None):
    """
    Read the tracked preceding vehicle track_id from file.
    
    Args:
        track_id_file: Path to the file containing the track_id
    
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


def load_data(camera_csv, lidar_csv):
    """Load camera and LIDAR TTC data."""
    df_camera = pd.read_csv(camera_csv)
    df_lidar = pd.read_csv(lidar_csv)
    
    return df_camera, df_lidar


def merge_ttc_data(df_camera, df_lidar, lidar_method='unfiltered'):
    """
    Merge camera and LIDAR TTC data on frame_index.
    """
    # Filter by track_id to ensure we're using the tracked preceding vehicle
    tracked_track_id = get_tracked_vehicle_track_id()
    
    if 'track_id' in df_camera.columns and 'track_id' in df_lidar.columns and tracked_track_id is not None:
        camera_track_data = df_camera[df_camera['track_id'] == tracked_track_id]
        lidar_track_data = df_lidar[df_lidar['track_id'] == tracked_track_id]
        
        if len(camera_track_data) > 0 and len(lidar_track_data) > 0:
            df_camera_filtered = camera_track_data.copy()
            df_lidar_filtered = lidar_track_data.copy()
            print(f"  Using track_id={tracked_track_id} (from tracked_preceding_vehicle_track_id.txt)")
        else:
            print(f"  Warning: track_id={tracked_track_id} not found in data, using all data")
            df_camera_filtered = df_camera.copy()
            df_lidar_filtered = df_lidar.copy()
    elif 'track_id' in df_camera.columns and 'track_id' in df_lidar.columns:
        camera_track_ids = df_camera['track_id'].mode()
        lidar_track_ids = df_lidar['track_id'].mode()
        
        if len(camera_track_ids) > 0 and len(lidar_track_ids) > 0:
            track_id = camera_track_ids[0]
            df_camera_filtered = df_camera[df_camera['track_id'] == track_id].copy()
            df_lidar_filtered = df_lidar[df_lidar['track_id'] == track_id].copy()
            print(f"  Warning: track_id file not found, using most common track_id={track_id}")
        else:
            df_camera_filtered = df_camera.copy()
            df_lidar_filtered = df_lidar.copy()
    else:
        df_camera_filtered = df_camera.copy()
        df_lidar_filtered = df_lidar.copy()
    
    # Extract the specified LIDAR method
    if lidar_method in df_lidar_filtered.columns:
        df_lidar_method = df_lidar_filtered[['frame_index', lidar_method]].copy()
        df_lidar_method.columns = ['frame_index', 'ttc_lidar']
    else:
        ttc_cols = [col for col in df_lidar_filtered.columns if 'ttc' in col.lower() or col in ['unfiltered', 'percentile_mean', 'percentile_median']]
        if ttc_cols:
            df_lidar_method = df_lidar_filtered[['frame_index', ttc_cols[0]]].copy()
            df_lidar_method.columns = ['frame_index', 'ttc_lidar']
            print(f"Warning: Using '{ttc_cols[0]}' as LIDAR TTC column")
        else:
            print("Error: No valid LIDAR TTC column found!")
            return None
    
    # Merge on frame_index
    df_merged = pd.merge(
        df_camera_filtered,
        df_lidar_method,
        on='frame_index',
        how='inner'
    )
    
    return df_merged, df_camera_filtered


def pearson_correlation(x, y):
    """Calculate Pearson correlation coefficient using numpy."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)
    
    if n < 2:
        return 0.0
    
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    std_x = np.std(x, ddof=1)
    std_y = np.std(y, ddof=1)
    
    if std_x == 0 or std_y == 0:
        return 0.0
    
    cov = np.sum((x - mean_x) * (y - mean_y)) / (n - 1)
    r = cov / (std_x * std_y)
    
    return r


def spearman_correlation(x, y):
    """Calculate Spearman rank correlation using numpy."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    # Get ranks, handling ties with average rank
    ranks_x = np.argsort(np.argsort(x))
    ranks_y = np.argsort(np.argsort(y))
    
    # Calculate Pearson correlation on ranks (this is Spearman's rho)
    return pearson_correlation(ranks_x, ranks_y)


def detect_discrepancies(df_merged, top_n=3):
    """
    Detect top N frames where LIDAR TTC differs most significantly from Camera TTC.
    
    Instead of using a threshold, this identifies the frames with the largest 
    absolute differences, which represent the most significant outliers.
    
    Args:
        df_merged: DataFrame with frame_index, ttc_camera, ttc_lidar columns
        top_n: Number of top outliers to identify (default: 3)
    
    Returns:
        DataFrame with discrepancy analysis
    """
    df_clean = df_merged.dropna(subset=['ttc_camera', 'ttc_lidar'])
    
    # Calculate differences
    df_clean['abs_diff'] = np.abs(df_clean['ttc_camera'] - df_clean['ttc_lidar'])
    df_clean['rel_diff'] = np.abs((df_clean['ttc_camera'] - df_clean['ttc_lidar']) / df_clean['ttc_camera'])
    
    # Sort by absolute difference and identify top N outliers
    df_clean = df_clean.sort_values('abs_diff', ascending=False)
    
    # Mark top N as outliers, others as normal
    df_clean['assessment'] = 'Good agreement'
    df_clean.loc[df_clean.head(top_n).index, 'assessment'] = 'Top Outlier'
    
    # Sort back by frame_index for consistent display
    df_clean = df_clean.sort_values('frame_index')
    
    return df_clean


def plot_discrepancy_analysis(df_discrepancy, output_dir=".", output_prefix="fp5"):
    """
    Generate comprehensive discrepancy analysis plots.
    
    Creates a 2x2 figure with:
    - Top-left: TTC comparison with discrepancy highlighting
    - Top-right: Relative difference per frame
    - Bottom-left: Absolute difference per frame
    - Bottom-right: Scatter plot with threshold
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(16, 12))
    sns.set_style("whitegrid")
    
    # Plot 1: TTC comparison with discrepancy highlighting
    plt.subplot(2, 2, 1)
    colors = {'Top Outlier': 'red', 'Good agreement': 'green'}
    for assessment, color in colors.items():
        subset = df_discrepancy[df_discrepancy['assessment'] == assessment]
        if len(subset) > 0:
            plt.scatter(subset['frame_index'], subset['ttc_camera'], 
                       color=color, label=f'Camera - {assessment}', s=100, edgecolor='black')
            plt.scatter(subset['frame_index'], subset['ttc_lidar'], 
                       color=color, marker='s', s=100, edgecolor='black')
    
    # Connect camera and lidar points for each frame
    for frame in df_discrepancy['frame_index']:
        frame_data = df_discrepancy[df_discrepancy['frame_index'] == frame]
        if len(frame_data) > 0:
            assessment = frame_data.iloc[0]['assessment']
            color = colors.get(assessment, 'gray')
            plt.plot([frame, frame], 
                    [frame_data.iloc[0]['ttc_camera'], frame_data.iloc[0]['ttc_lidar']],
                    color=color, alpha=0.3, linestyle='--')
    
    plt.title('FP.5: Camera vs LIDAR TTC with Discrepancy Detection', fontsize=14)
    plt.xlabel('Frame Index', fontsize=12)
    plt.ylabel('TTC (seconds)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Plot 2: Relative difference per frame
    plt.subplot(2, 2, 2)
    bars = plt.bar(df_discrepancy['frame_index'], df_discrepancy['rel_diff'] * 100, 
                  color=[colors.get(a, 'gray') for a in df_discrepancy['assessment']], 
                  alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bar, rel_diff, assessment in zip(bars, df_discrepancy['rel_diff'] * 100, df_discrepancy['assessment']):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{rel_diff:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.title('Relative Difference per Frame', fontsize=14)
    plt.xlabel('Frame Index', fontsize=12)
    plt.ylabel('Relative Difference (%)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Plot 3: Absolute difference per frame
    plt.subplot(2, 2, 3)
    bars = plt.bar(df_discrepancy['frame_index'], df_discrepancy['abs_diff'], 
                  color=[colors.get(a, 'gray') for a in df_discrepancy['assessment']], 
                  alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bar, abs_diff in zip(bars, df_discrepancy['abs_diff']):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{abs_diff:.2f}s', ha='center', va='bottom', fontsize=8)
    
    plt.title('Absolute Difference per Frame', fontsize=14)
    plt.xlabel('Frame Index', fontsize=12)
    plt.ylabel('Absolute Difference (seconds)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Plot 4: Scatter plot highlighting top outliers
    plt.subplot(2, 2, 4)
    
    # Plot all points
    plt.scatter(df_discrepancy['ttc_lidar'], df_discrepancy['ttc_camera'],
               c=[colors.get(a, 'gray') for a in df_discrepancy['assessment']],
               s=100, alpha=0.7, edgecolor='black')
    
    # Add identity line
    max_val = max(df_discrepancy['ttc_lidar'].max(), df_discrepancy['ttc_camera'].max()) * 1.1
    min_val = min(df_discrepancy['ttc_lidar'].min(), df_discrepancy['ttc_camera'].min()) * 0.9
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.5, label='y = x')
    
    # Add legend for assessment
    legend_elements = [
        Patch(facecolor='red', edgecolor='black', label='Top Outlier'),
        Patch(facecolor='green', edgecolor='black', label='Good agreement')
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize=10)
    
    plt.title('Camera vs LIDAR TTC Scatter - Top 3 Outliers', fontsize=14)
    plt.xlabel('LIDAR TTC (seconds)', fontsize=12)
    plt.ylabel('Camera TTC (seconds)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('FP.5: Top 3 Outlier Analysis - Camera vs LIDAR TTC', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f"{output_prefix}_discrepancy_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved discrepancy analysis plot to: {output_path}")
    return output_path


def plot_top_outliers(df_discrepancy, output_dir=".", output_prefix="fp5"):
    """
    Generate focused plot on top outlier frames only.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get top outlier frames
    outliers = df_discrepancy[df_discrepancy['assessment'] == 'Top Outlier']
    
    if len(outliers) == 0:
        print("No top outlier frames to plot")
        return None
    
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Create a wider x-axis range for better visualization
    x_positions = np.arange(len(outliers))
    
    # Plot camera and LIDAR TTC for outlier frames
    width = 0.35
    plt.bar(x_positions - width/2, outliers['ttc_camera'], width, 
            color='blue', alpha=0.7, label='Camera TTC', edgecolor='black')
    plt.bar(x_positions + width/2, outliers['ttc_lidar'], width, 
            color='orange', alpha=0.7, label='LIDAR TTC (UNFILTERED)', edgecolor='black')
    
    # Add frame labels and difference annotations
    for i, (frame, row) in enumerate(outliers.iterrows()):
        frame_idx = row['frame_index']
        camera_ttc = row['ttc_camera']
        lidar_ttc = row['ttc_lidar']
        abs_diff = row['abs_diff']
        rel_diff = row['rel_diff'] * 100
        
        # Add frame number above the bars
        plt.text(i, max(camera_ttc, lidar_ttc) + 0.5, f'Frame {frame_idx}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add difference annotation
        mid_x = i
        mid_y = (camera_ttc + lidar_ttc) / 2
        plt.text(mid_x, mid_y, f'{abs_diff:.2f}s ({rel_diff:.1f}%)',
                ha='center', va='center', fontsize=9, color='red', fontweight='bold')
    
    plt.title('FP.5: Top 3 Outlier Frames - Camera vs LIDAR TTC', fontsize=16, pad=20)
    plt.xlabel('Frame', fontsize=14)
    plt.ylabel('TTC (seconds)', fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(x_positions, [f"{frame}" for frame in outliers['frame_index']])
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f"{output_prefix}_top_outliers.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved top outliers plot to: {output_path}")
    return output_path


def print_discrepancy_report(df_discrepancy, top_n):
    """Print detailed outlier analysis report."""
    
    print("\n" + "="*80)
    print("FP.5: PERFORMANCE ASSESSMENT 1 - OUTLIER ANALYSIS REPORT")
    print("="*80)
    
    print(f"\nTop {top_n} Outliers Selected by Absolute Difference")
    print(f"Total frames analyzed: {len(df_discrepancy)}")
    
    # Summary statistics
    print("\n" + "-"*80)
    print("SUMMARY STATISTICS")
    print("-"*80)
    
    # Calculate Pearson correlation manually to avoid scipy dependency
    x = df_discrepancy['ttc_lidar'].values
    y = df_discrepancy['ttc_camera'].values
    
    # Pearson correlation
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    std_x = np.std(x, ddof=1)
    std_y = np.std(y, ddof=1)
    cov = np.sum((x - mean_x) * (y - mean_y)) / (len(x) - 1)
    pearson_r = cov / (std_x * std_y) if (std_x > 0 and std_y > 0) else 0.0
    
    # Spearman correlation using our existing function
    spearman_r = spearman_correlation(x, y)
    
    print(f"Pearson correlation coefficient: {pearson_r:.4f}")
    print(f"Spearman rank correlation: {spearman_r:.4f}")
    print(f"Mean absolute difference: {df_discrepancy['abs_diff'].mean():.2f} s")
    print(f"Mean relative difference: {df_discrepancy['rel_diff'].mean()*100:.1f}%")
    print(f"RMSE: {np.sqrt(np.mean(df_discrepancy['abs_diff']**2)):.2f} s")
    
    # Assessment breakdown
    print("\n" + "-"*80)
    print("ASSESSMENT BREAKDOWN")
    print("-"*80)
    assessment_counts = df_discrepancy['assessment'].value_counts()
    for assessment, count in assessment_counts.items():
        print(f"  {assessment}: {count} frames")
    
    # Top outliers
    outliers = df_discrepancy[df_discrepancy['assessment'] == 'Top Outlier']
    if len(outliers) > 0:
        print("\n" + "-"*80)
        print("TOP OUTLIER FRAMES (Largest Absolute Differences)")
        print("-"*80)
        print("Frame | Camera TTC | LIDAR TTC | Abs Diff | Rel Diff")
        print("-" + "-"*65)
        # Sort by absolute difference descending
        outliers_sorted = outliers.sort_values('abs_diff', ascending=False)
        for _, row in outliers_sorted.iterrows():
            print(f"  {int(row['frame_index']):2d}  | {row['ttc_camera']:8.2f}s | {row['ttc_lidar']:11.2f}s | {row['abs_diff']:7.2f}s | {row['rel_diff']*100:6.1f}%")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='FP.5 Performance Assessment 1 - Identify implausible LIDAR TTC estimates'
    )
    parser.add_argument('--camera-csv', type=str, 
                       default='output/ttc_camera.csv',
                       help='Path to camera TTC CSV file (default: output/ttc_camera.csv)')
    parser.add_argument('--lidar-csv', type=str,
                       default='output/ttc_lidar_comparison.csv',
                       help='Path to LIDAR TTC comparison CSV file (default: output/ttc_lidar_comparison.csv)')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory for plots (default: output)')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively')
    parser.add_argument('--lidar-method', type=str, default='unfiltered',
                       help='LIDAR method to use for comparison (default: unfiltered)')
    parser.add_argument('--top-n', type=int, default=3,
                       help='Number of top outliers to identify (default: 3)')
    
    args = parser.parse_args()
    
    # Print header
    print("="*80)
    print("FP.5 PERFORMANCE ASSESSMENT 1 - DISCREPANCY DETECTION")
    print("="*80)
    
    # Load data
    print(f"\nLoading data from:")
    print(f"  Camera TTC: {args.camera_csv}")
    print(f"  LIDAR TTC: {args.lidar_csv}")
    
    try:
        df_camera, df_lidar = load_data(args.camera_csv, args.lidar_csv)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure all CSV files exist and run the 3D_object_tracking executable first.")
        return
    
    print(f"\nLoaded {len(df_camera)} camera TTC records")
    print(f"Loaded {len(df_lidar)} LIDAR TTC records")
    
    # Merge camera and LIDAR data
    print("\nMerging camera and LIDAR TTC data...")
    df_merged, df_camera_filtered = merge_ttc_data(
        df_camera, df_lidar, args.lidar_method
    )
    
    if df_merged is None or len(df_merged) == 0:
        print("Error: No valid merged data for comparison!")
        return
    
    print(f"Merged data: {len(df_merged)} matching frame pairs")
    
    # Detect discrepancies
    print(f"\nDetecting top {args.top_n} outlier frames by absolute difference...")
    df_discrepancy = detect_discrepancies(df_merged, args.top_n)
    
    # Print report
    print_discrepancy_report(df_discrepancy, args.top_n)
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Plot 1: Comprehensive outlier analysis
    plot_discrepancy_analysis(df_discrepancy, args.output, "fp5")
    
    # Plot 2: Focus on top outliers
    plot_top_outliers(df_discrepancy, args.output, "fp5")
    
    # Show plots if requested
    if args.show:
        print("\nDisplaying plots interactively...")
        
        # Re-generate and show discrepancy analysis
        plot_discrepancy_analysis(df_discrepancy, args.output, "fp5")
        
        # Re-generate and show implausible frames
        plot_implausible_frames(df_discrepancy, args.output, "fp5")
        
        # Display
        plt.figure(figsize=(16, 12))
        # ... (plots would be regenerated for display)
        plt.show()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nOutput files saved to:")
    print(f"  {os.path.abspath(args.output)}/fp5_discrepancy_analysis.png")
    print(f"  {os.path.abspath(args.output)}/fp5_top_outliers.png")
    print(f"\nUse these plots to identify and analyze the top {args.top_n} outlier frames.")
    print("Outliers are selected by largest absolute difference in TTC values.")


if __name__ == '__main__':
    main()
