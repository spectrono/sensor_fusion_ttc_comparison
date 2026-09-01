#!/usr/bin/env python3
"""
FP.4 Camera TTC Analysis Script

Analyzes camera-based TTC estimation performance and the impact of background cluster filtering.
Compares camera TTC with LIDAR TTC and visualizes scale ratio distributions.

Usage:
    python fp4_analysis.py [--camera-csv output/ttc_camera.csv] [--lidar-csv output/ttc_lidar_comparison.csv] 
                         [--scale-csv output/ttc_camera_scale_stats.csv] [--output output] [--show]
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


def load_data(camera_csv, lidar_csv, scale_csv):
    """Load all three CSV files and return DataFrames."""
    df_camera = pd.read_csv(camera_csv)
    df_lidar = pd.read_csv(lidar_csv)
    df_scale = pd.read_csv(scale_csv)
    
    return df_camera, df_lidar, df_scale


def merge_ttc_data(df_camera, df_lidar, lidar_method='unfiltered'):
    """
    Merge camera and LIDAR TTC data on frame_index.
    Extracts the specified LIDAR method for comparison.
    
    Note: We filter by track_id to ensure we're comparing the same physical vehicle
    across frames, as box_id can change due to YOLO detection inconsistencies.
    """
    # Filter by track_id to ensure we're using the tracked preceding vehicle
    # The track_id is read from the file written by the C++ program
    tracked_track_id = get_tracked_vehicle_track_id()
    
    if 'track_id' in df_camera.columns and 'track_id' in df_lidar.columns and tracked_track_id is not None:
        # Use the track_id from the file
        camera_track_data = df_camera[df_camera['track_id'] == tracked_track_id]
        lidar_track_data = df_lidar[df_lidar['track_id'] == tracked_track_id]
        
        if len(camera_track_data) > 0 and len(lidar_track_data) > 0:
            df_camera_filtered = camera_track_data.copy()
            df_lidar_filtered = lidar_track_data.copy()
            print(f"  Using track_id={tracked_track_id} (from tracked_preceding_vehicle_track_id.txt) for filtered comparison")
        else:
            print(f"  Warning: track_id={tracked_track_id} not found in data, using all data")
            df_camera_filtered = df_camera.copy()
            df_lidar_filtered = df_lidar.copy()
    elif 'track_id' in df_camera.columns and 'track_id' in df_lidar.columns:
        # track_id file not found, fallback to most common track_id
        camera_track_ids = df_camera['track_id'].mode()
        lidar_track_ids = df_lidar['track_id'].mode()
        
        # Use the same track_id for both (they should match)
        if len(camera_track_ids) > 0 and len(lidar_track_ids) > 0:
            track_id = camera_track_ids[0]  # Use camera track_id
            df_camera_filtered = df_camera[df_camera['track_id'] == track_id].copy()
            df_lidar_filtered = df_lidar[df_lidar['track_id'] == track_id].copy()
            print(f"  Warning: track_id file not found, using most common track_id={track_id} for filtered comparison")
        else:
            df_camera_filtered = df_camera.copy()
            df_lidar_filtered = df_lidar.copy()
    else:
        # Fallback: use all data if track_id not available
        df_camera_filtered = df_camera.copy()
        df_lidar_filtered = df_lidar.copy()
    
    # Check if lidar_method exists in the dataframe
    if lidar_method in df_lidar_filtered.columns:
        df_lidar_method = df_lidar_filtered[['frame_index', lidar_method]].copy()
        df_lidar_method.columns = ['frame_index', 'ttc_lidar']
    else:
        # Try to find any TTC column
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


def plot_ttc_comparison(df, output_dir=".", output_prefix="fp4_ttc"):
    """
    Generate plot comparing camera and LIDAR TTC over frames.
    
    Args:
        df: DataFrame with frame_index, ttc_camera, ttc_lidar columns
        output_dir: Directory to save plots
        output_prefix: Prefix for output filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(14, 7))
    sns.set_style("whitegrid")
    
    # Plot camera TTC
    sns.lineplot(
        data=df,
        x='frame_index',
        y='ttc_camera',
        label='Camera TTC',
        marker='o',
        markersize=8,
        linewidth=2.5,
        color='#1f77b4'
    )
    
    # Plot LIDAR TTC
    sns.lineplot(
        data=df,
        x='frame_index',
        y='ttc_lidar',
        label='LIDAR TTC',
        marker='s',
        markersize=8,
        linewidth=2.5,
        color='#ff7f0e'
    )
    
    # Customize plot
    plt.title('FP.4: Camera vs LIDAR TTC Comparison', fontsize=16, pad=20)
    plt.xlabel('Frame Index', fontsize=14)
    plt.ylabel('TTC (seconds)', fontsize=14)
    plt.legend(title='Method', loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Set integer ticks on x-axis
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Save plot
    output_path = os.path.join(output_dir, f"{output_prefix}_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved TTC comparison plot to: {output_path}")
    return output_path


def pearson_correlation(x, y):
    """Calculate Pearson correlation coefficient using numpy."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)
    
    if n < 2:
        return 0.0
    
    # Calculate Pearson r
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


def plot_correlation_scatter(df, output_dir=".", output_prefix="fp4_correlation"):
    """
    Generate scatter plot with correlation coefficient between camera and LIDAR TTC.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Drop NaN values for correlation calculation
    df_clean = df.dropna(subset=['ttc_camera', 'ttc_lidar'])
    
    # Calculate correlation coefficients
    pearson_r = pearson_correlation(df_clean['ttc_lidar'], df_clean['ttc_camera'])
    spearman_r = spearman_correlation(df_clean['ttc_lidar'], df_clean['ttc_camera'])
    
    # Calculate statistics
    mean_abs_diff = np.mean(np.abs(df_clean['ttc_camera'] - df_clean['ttc_lidar']))
    mean_rel_diff = np.mean(np.abs((df_clean['ttc_camera'] - df_clean['ttc_lidar']) / df_clean['ttc_lidar'])) * 100
    rmse = np.sqrt(np.mean((df_clean['ttc_camera'] - df_clean['ttc_lidar']) ** 2))
    
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Scatter plot
    scatter = sns.scatterplot(
        data=df_clean,
        x='ttc_lidar',
        y='ttc_camera',
        s=100,
        color='#1f77b4',
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Add regression line
    sns.regplot(
        data=df_clean,
        x='ttc_lidar',
        y='ttc_camera',
        scatter=False,
        color='red',
        line_kws={'linewidth': 2, 'linestyle': '--'}
    )
    
    # Add identity line (y = x)
    max_val = max(df_clean['ttc_lidar'].max(), df_clean['ttc_camera'].max()) * 1.1
    min_val = min(df_clean['ttc_lidar'].min(), df_clean['ttc_camera'].min()) * 0.9
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.5, label='y = x')
    
    # Annotate correlation coefficients
    correlation_text = (f'Pearson r = {pearson_r:.3f}\n'
                       f'Spearman r = {spearman_r:.3f}\n'
                       f'Mean Abs Diff = {mean_abs_diff:.2f} s\n'
                       f'Mean Rel Diff = {mean_rel_diff:.1f}%\n'
                       f'RMSE = {rmse:.2f} s')
    
    plt.text(
        0.02, 0.98,
        correlation_text,
        transform=plt.gca().transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
        fontsize=11
    )
    
    # Customize plot
    plt.title('FP.4: Camera vs LIDAR TTC Correlation', fontsize=16, pad=20)
    plt.xlabel('LIDAR TTC (seconds)', fontsize=14)
    plt.ylabel('Camera TTC (seconds)', fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_path = os.path.join(output_dir, f"{output_prefix}_scatter.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved correlation scatter plot to: {output_path}")
    
    return {
        'pearson_r': pearson_r,
        'spearman_r': spearman_r,
        'mean_abs_diff': mean_abs_diff,
        'mean_rel_diff': mean_rel_diff,
        'rmse': rmse
    }


def plot_scale_distributions(df_scale, output_dir=".", output_prefix="fp4_scale"):
    """
    Plot scale ratio distributions over frames.
    
    Generates a 2x1 figure with two subplots:
    - Top: Distance ratios per frame with min/max range, std dev, median, and mean
    - Bottom: Histogram of all distance ratios across all frames
    
    Args:
        df_scale: DataFrame with scale ratio statistics
        output_dir: Directory to save plots
        output_prefix: Prefix for output filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter by track_id to ensure we're analyzing the tracked preceding vehicle
    # The track_id is read from the file written by the C++ program
    tracked_track_id = get_tracked_vehicle_track_id()
    
    if 'track_id' in df_scale.columns and tracked_track_id is not None:
        # Use the track_id from the file
        track_data = df_scale[df_scale['track_id'] == tracked_track_id]
        if len(track_data) > 0:
            df_scale = track_data.copy()
            print(f"  Using track_id={tracked_track_id} (from tracked_preceding_vehicle_track_id.txt) for scale analysis")
        else:
            print(f"  Warning: track_id={tracked_track_id} not found in data, using all data")
    elif 'track_id' in df_scale.columns:
        # track_id file not found, fallback to most common track_id
        track_ids = df_scale['track_id'].mode()
        if len(track_ids) > 0:
            track_id = track_ids[0]
            df_scale = df_scale[df_scale['track_id'] == track_id].copy()
            print(f"  Warning: track_id file not found, using most common track_id={track_id} for scale analysis")
    
    plt.figure(figsize=(14, 10))
    sns.set_style("whitegrid")
    
    # Plot 1 (top): Median ratios with fill between min/max
    plt.subplot(2, 1, 1)
    frames = df_scale['frame_index']
    
    plt.fill_between(frames, df_scale['min_ratio'], df_scale['max_ratio'], 
                     alpha=0.2, color='blue', label='Range')
    plt.fill_between(frames, df_scale['mean_ratio'] - df_scale['stddev_ratio'],
                         df_scale['mean_ratio'] + df_scale['stddev_ratio'],
                     alpha=0.3, color='blue', label='Std Dev')
    
    sns.lineplot(data=df_scale, x='frame_index', y='median_ratio', 
                 color='blue', linewidth=2.5, marker='o', markersize=6, label='Median')
    sns.lineplot(data=df_scale, x='frame_index', y='mean_ratio', 
                 color='cyan', linewidth=2, marker='o', markersize=6, label='Mean')
    
    plt.title('Distance Ratios per Frame', fontsize=14)
    plt.xlabel('Frame Index', fontsize=12)
    plt.ylabel('Distance Ratio', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Plot 2 (bottom): Ratio distribution histogram (all frames combined)
    plt.subplot(2, 1, 2)
    
    # Show the overall distribution of medians
    all_medians = df_scale['median_ratio'].dropna()
    sns.kdeplot(all_medians, color='blue', linewidth=2, label='Distance Ratios', fill=True, alpha=0.3)
    plt.axvline(x=1.0, color='black', linestyle='--', alpha=0.7, label='Ratio = 1.0')
    
    plt.title('Distance Ratio Distribution', fontsize=14)
    plt.xlabel('Distance Ratio', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('FP.4: Scale Ratio Distributions', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f"{output_prefix}_distributions.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved scale distribution plot to: {output_path}")
    return output_path


def plot_filtering_impact(df_scale, output_dir=".", output_prefix="fp4_filtering"):
    """
    Visualize the impact of background filtering on scale ratios.
    
    Args:
        df_scale: DataFrame with scale ratio statistics including num_ratios and num_filtered
        output_dir: Directory to save plots
        output_prefix: Prefix for output filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter by track_id to ensure we're analyzing the tracked preceding vehicle
    # The track_id is read from the file written by the C++ program
    tracked_track_id = get_tracked_vehicle_track_id()
    
    if 'track_id' in df_scale.columns and tracked_track_id is not None:
        # Use the track_id from the file
        track_data = df_scale[df_scale['track_id'] == tracked_track_id]
        if len(track_data) > 0:
            df_scale = track_data.copy()
        else:
            print(f"  Warning: track_id={tracked_track_id} not found in data, using all data")
    elif 'track_id' in df_scale.columns:
        # track_id file not found, fallback to most common track_id
        track_ids = df_scale['track_id'].mode()
        if len(track_ids) > 0:
            track_id = track_ids[0]
            df_scale = df_scale[df_scale['track_id'] == track_id].copy()
    
    plt.figure(figsize=(16, 10))
    sns.set_style("whitegrid")
    
    frames = df_scale['frame_index']
    
    # Plot 1: Number of ratios before and after filtering
    plt.subplot(2, 2, 1)
    
    if 'num_ratios' in df_scale.columns and 'num_filtered' in df_scale.columns:
        plt.bar(frames - 0.2, df_scale['num_ratios'], width=0.4, 
                color='blue', alpha=0.7, label='Before Filtering')
        plt.bar(frames + 0.2, df_scale['num_filtered'], width=0.4,
                color='red', alpha=0.7, label='After Filtering')
        plt.title('Number of Distance Ratios: Before vs After Filtering', fontsize=14)
        plt.xlabel('Frame Index', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    else:
        plt.text(0.5, 0.5, 'Column data not available', 
                 transform=plt.gca().transAxes, ha='center', va='center', fontsize=12)
        plt.title('Number of Distance Ratios', fontsize=14)
    
    # Plot 2: Percentage removed per frame
    plt.subplot(2, 2, 2)
    
    if 'num_ratios' in df_scale.columns and 'num_filtered' in df_scale.columns:
        pct_removed = ((df_scale['num_ratios'] - df_scale['num_filtered']) / df_scale['num_ratios']) * 100
        
        bars = plt.bar(frames, pct_removed, color='purple', alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.title('Percentage of Ratios Removed by Filtering', fontsize=14)
        plt.xlabel('Frame Index', fontsize=12)
        plt.ylabel('Percentage Removed (%)', fontsize=12)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        plt.grid(True, alpha=0.3, axis='y')
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    else:
        plt.text(0.5, 0.5, 'Column data not available', 
                 transform=plt.gca().transAxes, ha='center', va='center', fontsize=12)
        plt.title('Percentage Removed', fontsize=14)
    
    # Plot 3: Median ratio shift due to filtering
    plt.subplot(2, 2, 3)
    
    if 'median_ratio' in df_scale.columns and 'filtered_median' in df_scale.columns:
        median_shift = df_scale['filtered_median'] - df_scale['median_ratio']
        
        plt.bar(frames, median_shift, color='orange', alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        plt.title('Median Ratio Shift Due to Filtering', fontsize=14)
        plt.xlabel('Frame Index', fontsize=12)
        plt.ylabel('Shift (Filtered - Unfiltered)', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    else:
        plt.text(0.5, 0.5, 'Filtered median data not available', 
                 transform=plt.gca().transAxes, ha='center', va='center', fontsize=12)
        plt.title('Median Ratio Shift', fontsize=14)
    
    # Plot 4: Cumulative statistics
    plt.subplot(2, 2, 4)
    
    if 'num_ratios' in df_scale.columns and 'num_filtered' in df_scale.columns:
        total_before = df_scale['num_ratios'].sum()
        total_after = df_scale['num_filtered'].sum()
        total_removed = total_before - total_after
        pct_total_removed = (total_removed / total_before) * 100 if total_before > 0 else 0
        
        sizes = [total_before - total_removed, total_removed]
        labels = [f'Kept\n{total_after} ({100-pct_total_removed:.1f}%)', 
                  f'Removed\n{total_removed} ({pct_total_removed:.1f}%)']
        colors = ['green', 'red']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, explode=(0, 0.1))
        plt.title(f'Total Filtering Impact\n{total_removed}/{total_before} ratios removed', 
                  fontsize=14)
    else:
        plt.text(0.5, 0.5, 'Column data not available', 
                 transform=plt.gca().transAxes, ha='center', va='center', fontsize=12)
        plt.title('Total Filtering Impact', fontsize=14)
    
    plt.suptitle('FP.4: Background Filtering Impact Analysis', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f"{output_prefix}_impact.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved filtering impact plot to: {output_path}")
    return output_path


def print_statistics(df_camera, df_lidar, df_scale):
    """Print comprehensive statistics for FP.4 analysis."""
    
    print("\n" + "="*80)
    print("FP.4: CAMERA TTC ANALYSIS STATISTICS")
    print("="*80)
    
    # Filter all dataframes by track_id to ensure we're analyzing the tracked preceding vehicle
    # The track_id is read from the file written by the C++ program
    tracked_track_id = get_tracked_vehicle_track_id()
    
    if ('track_id' in df_camera.columns and 'track_id' in df_lidar.columns and 
        'track_id' in df_scale.columns and tracked_track_id is not None):
        # Use the track_id from the file
        camera_track_data = df_camera[df_camera['track_id'] == tracked_track_id]
        lidar_track_data = df_lidar[df_lidar['track_id'] == tracked_track_id]
        scale_track_data = df_scale[df_scale['track_id'] == tracked_track_id]
        
        if (len(camera_track_data) > 0 and len(lidar_track_data) > 0 and len(scale_track_data) > 0):
            print(f"\nFiltering statistics for track_id={tracked_track_id} (from tracked_preceding_vehicle_track_id.txt)")
            df_camera = camera_track_data
            df_lidar = lidar_track_data
            df_scale = scale_track_data
        else:
            print(f"\nWarning: track_id={tracked_track_id} not found in all dataframes, using all data")
    elif 'track_id' in df_camera.columns and 'track_id' in df_lidar.columns and 'track_id' in df_scale.columns:
        # track_id file not found, fallback to most common track_id
        camera_track_ids = df_camera['track_id'].mode()
        lidar_track_ids = df_lidar['track_id'].mode()
        scale_track_ids = df_scale['track_id'].mode()
        
        if (len(camera_track_ids) > 0 and len(lidar_track_ids) > 0 and len(scale_track_ids) > 0):
            track_id = camera_track_ids[0]
            print(f"\nWarning: track_id file not found, using most common track_id={track_id}")
            df_camera = df_camera[df_camera['track_id'] == track_id]
            df_lidar = df_lidar[df_lidar['track_id'] == track_id]
            df_scale = df_scale[df_scale['track_id'] == track_id]
    
    # Camera TTC statistics
    print("\n" + "-"*80)
    print("CAMERA TTC STATISTICS")
    print("-"*80)
    camera_ttc = df_camera['ttc_camera'].dropna()
    print(f"Total camera TTC samples: {len(camera_ttc)}")
    print(f"Mean TTC: {camera_ttc.mean():.2f} s")
    print(f"Median TTC: {camera_ttc.median():.2f} s")
    print(f"Std Dev: {camera_ttc.std():.2f} s")
    print(f"Min TTC: {camera_ttc.min():.2f} s")
    print(f"Max TTC: {camera_ttc.max():.2f} s")
    print(f"25th percentile: {camera_ttc.quantile(0.25):.2f} s")
    print(f"75th percentile: {camera_ttc.quantile(0.75):.2f} s")
    print(f"NaN count: {df_camera['ttc_camera'].isna().sum()}")
    
    # LIDAR TTC statistics (using unfiltered)
    print("\n" + "-"*80)
    print("LIDAR TTC STATISTICS (unfiltered method)")
    print("-"*80)
    lidar_ttc = df_lidar['unfiltered'].dropna()
    if len(lidar_ttc) > 0:
        print(f"Total LIDAR TTC samples: {len(lidar_ttc)}")
        print(f"Mean TTC: {lidar_ttc.mean():.2f} s")
        print(f"Median TTC: {lidar_ttc.median():.2f} s")
        print(f"Std Dev: {lidar_ttc.std():.2f} s")
        print(f"Min TTC: {lidar_ttc.min():.2f} s")
        print(f"Max TTC: {lidar_ttc.max():.2f} s")
        print(f"25th percentile: {lidar_ttc.quantile(0.25):.2f} s")
        print(f"75th percentile: {lidar_ttc.quantile(0.75):.2f} s")
    else:
        print("No LIDAR TTC data found")
    
    # Scale ratio statistics
    print("\n" + "-"*80)
    print("SCALE RATIO STATISTICS")
    print("-"*80)
    
    if 'num_ratios' in df_scale.columns:
        total_ratios = df_scale['num_ratios'].sum()
        total_filtered = df_scale['num_filtered'].sum() if 'num_filtered' in df_scale.columns else total_ratios
        total_removed = total_ratios - total_filtered
        avg_removal_pct = ((total_ratios - total_filtered) / total_ratios * 100) if total_ratios > 0 else 0
        
        print(f"Total distance ratios (before filtering): {total_ratios}")
        print(f"Total distance ratios (after filtering): {total_filtered}")
        print(f"Total ratios removed: {total_removed} ({avg_removal_pct:.1f}%)")
    
    if 'median_ratio' in df_scale.columns:
        print(f"\nUnfiltered median ratio (overall): {df_scale['median_ratio'].median():.6f}")
        print(f"Mean unfiltered median ratio: {df_scale['median_ratio'].mean():.6f}")
        print(f"Std dev of unfiltered median ratio: {df_scale['median_ratio'].std():.6f}")
    
    if 'filtered_median' in df_scale.columns:
        print(f"\nFiltered median ratio (overall): {df_scale['filtered_median'].median():.6f}")
        print(f"Mean filtered median ratio: {df_scale['filtered_median'].mean():.6f}")
        print(f"Std dev of filtered median ratio: {df_scale['filtered_median'].std():.6f}")
        
        # Compare medians
        median_shift = (df_scale['filtered_median'] - df_scale['median_ratio']).mean()
        print(f"\nAverage median shift due to filtering: {median_shift:.6f}")
    
    # Per-frame statistics
    print("\n" + "-"*80)
    print("PER-FRAME STATISTICS")
    print("-"*80)
    print(f"Frames with filtering applied (num_filtered < num_ratios): ")
    if 'num_ratios' in df_scale.columns and 'num_filtered' in df_scale.columns:
        filtering_frames = df_scale[df_scale['num_filtered'] < df_scale['num_ratios']]
        print(f"  {len(filtering_frames)} frames")
        if len(filtering_frames) > 0:
            print(f"  Frame indices: {filtering_frames['frame_index'].tolist()}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='FP.4 Camera TTC Analysis - Compare camera TTC with LIDAR and analyze background filtering impact'
    )
    parser.add_argument('--camera-csv', type=str, 
                       default='output/ttc_camera.csv',
                       help='Path to camera TTC CSV file (default: output/ttc_camera.csv)')
    parser.add_argument('--lidar-csv', type=str,
                       default='output/ttc_lidar_comparison.csv',
                       help='Path to LIDAR TTC comparison CSV file (default: output/ttc_lidar_comparison.csv)')
    parser.add_argument('--scale-csv', type=str,
                       default='output/ttc_camera_scale_stats.csv',
                       help='Path to scale stats CSV file (default: output/ttc_camera_scale_stats.csv)')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory for plots (default: output)')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively')
    parser.add_argument('--lidar-method', type=str, default='unfiltered',
                       help='LIDAR method to use for comparison (default: unfiltered)')
    
    args = parser.parse_args()
    
    # Print header
    print("="*80)
    print("FP.4 CAMERA TTC ANALYSIS")
    print("="*80)
    
    # Load data
    print(f"\nLoading data from:")
    print(f"  Camera TTC: {args.camera_csv}")
    print(f"  LIDAR TTC: {args.lidar_csv}")
    print(f"  Scale Stats: {args.scale_csv}")
    
    try:
        df_camera, df_lidar, df_scale = load_data(args.camera_csv, args.lidar_csv, args.scale_csv)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure all CSV files exist and run the 3D_object_tracking executable first.")
        return
    
    print(f"\nLoaded {len(df_camera)} camera TTC records")
    print(f"Loaded {len(df_lidar)} LIDAR TTC records")
    print(f"Loaded {len(df_scale)} scale statistics records")
    
    # Merge camera and LIDAR data
    print("\nMerging camera and LIDAR TTC data...")
    df_merged, df_camera_filtered = merge_ttc_data(
        df_camera, df_lidar, args.lidar_method
    )
    
    if df_merged is None or len(df_merged) == 0:
        print("Error: No valid merged data for comparison!")
        return
    
    print(f"Merged data: {len(df_merged)} matching frame pairs")
    
    # Print statistics
    print("\nComputing statistics...")
    print_statistics(df_camera_filtered, df_lidar, df_scale)
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Plot 1: TTC comparison
    plot_ttc_comparison(df_merged, args.output, "fp4_ttc")
    
    # Plot 2: Correlation scatter
    correlation_stats = plot_correlation_scatter(df_merged, args.output, "fp4_correlation")
    
    # Print correlation statistics
    print("\n" + "-"*80)
    print("CORRELATION STATISTICS")
    print("-"*80)
    print(f"Pearson correlation coefficient: {correlation_stats['pearson_r']:.4f}")
    print(f"Spearman rank correlation: {correlation_stats['spearman_r']:.4f}")
    print(f"Mean absolute difference: {correlation_stats['mean_abs_diff']:.2f} s")
    print(f"Mean relative difference: {correlation_stats['mean_rel_diff']:.1f}%")
    print(f"RMSE: {correlation_stats['rmse']:.2f} s")
    
    # Plot 3: Scale distributions
    plot_scale_distributions(df_scale, args.output, "fp4_scale")
    
    # Plot 4: Filtering impact
    plot_filtering_impact(df_scale, args.output, "fp4_filtering")
    
    # Show plots if requested
    if args.show:
        print("\nDisplaying plots interactively...")
        
        # Re-generate and show plots
        plot_ttc_comparison(df_merged, args.output, "fp4_ttc")
        df_merged_for_display = df_merged.copy()
        
        # For display, create the plot but don't close it
        plt.figure(figsize=(14, 7))
        sns.set_style("whitegrid")
        sns.lineplot(data=df_merged_for_display, x='frame_index', y='ttc_camera', 
                     label='Camera TTC', marker='o', markersize=8, linewidth=2.5, color='#1f77b4')
        sns.lineplot(data=df_merged_for_display, x='frame_index', y='ttc_lidar', 
                     label='LIDAR TTC', marker='s', markersize=8, linewidth=2.5, color='#ff7f0e')
        plt.title('FP.4: Camera vs LIDAR TTC Comparison', fontsize=16, pad=20)
        plt.xlabel('Frame Index', fontsize=14)
        plt.ylabel('TTC (seconds)', fontsize=14)
        plt.legend(title='Method', loc='best', fontsize=12)
        plt.grid(True, alpha=0.3)
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.tight_layout()
        plt.show()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nOutput files saved to:")
    print(f"  {os.path.abspath(args.output)}/fp4_ttc_comparison.png")
    print(f"  {os.path.abspath(args.output)}/fp4_correlation_scatter.png")
    print(f"  {os.path.abspath(args.output)}/fp4_scale_distributions.png")
    print(f"  {os.path.abspath(args.output)}/fp4_filtering_impact.png")
    print("\nUse these plots to evaluate the impact of background filtering on TTC estimation.")


if __name__ == '__main__':
    main()
