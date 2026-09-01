#!/usr/bin/env python3
"""
FP.2 Lidar TTC Analysis Script

Compares different TTC calculation methods (unfiltered, percentile-filtered, 
percentile+mean, percentile+median) and generates visualizations.
Designed to be extended later for comparing with camera-based TTC (FP.4).
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
    """Load TTC comparison data from CSV."""
    df = pd.read_csv(csv_path)
    return df


def plot_ttc_comparison(df, output_prefix="fp2_ttc"):
    """Generate comparison plots for different TTC methods."""
    
    # Define method colors and order
    method_order = ["unfiltered", "percentile_mean", "percentile_median"]
    method_colors = {
        "unfiltered": "red",
        "percentile_mean": "blue",
        "percentile_median": "green"
    }
    
    # Create output directory if needed
    os.makedirs("output", exist_ok=True)
    
    # Filter for preceding vehicle using track_id from file
    # The CSV contains: frame_index, track_id, prev_box_id, curr_box_id, ...
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
            print("  Warning: No track_id data found, using all data")
            df_preceding = df.copy()
    else:
        # Fallback: try curr_box_id == 1 then 0 (legacy behavior)
        print("  Warning: No track_id column found, falling back to curr_box_id filtering")
        df_preceding = df[df['curr_box_id'] == 1].copy()
        if len(df_preceding) == 0:
            df_preceding = df[df['curr_box_id'] == 0].copy()
        if len(df_preceding) == 0:
            print("  Warning: No data for preceding vehicle, using all data")
            df_preceding = df.copy()
    
    # Replace 'nan' strings with actual NaN
    for method in method_order:
        df_preceding[method] = pd.to_numeric(df_preceding[method], errors='coerce')
    
    # Create figure
    plt.figure(figsize=(16, 12))
    plt.suptitle('FP.2: Lidar TTC Method Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: All methods over time (line plot)
    plt.subplot(2, 2, 1)
    for method in method_order:
        method_data = df_preceding.dropna(subset=[method])
        plt.plot(method_data['frame_index'], method_data[method], 
                'o-', label=method, color=method_colors.get(method, 'gray'), 
                linewidth=2, markersize=4, alpha=0.8)
    plt.title('TTC Over Frames (Preceding Vehicle)')
    plt.xlabel('Frame Index')
    plt.ylabel('TTC (seconds)')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Method', loc='best')
    # Set integer ticks on x-axis
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Plot 2: Boxplot comparison
    plt.subplot(2, 2, 2)
    df_box = df_preceding.melt(
        id_vars=['frame_index', 'prev_box_id', 'curr_box_id'],
        value_vars=method_order,
        var_name='method',
        value_name='TTC'
    ).dropna(subset=['TTC'])
    sns.boxplot(data=df_box, x='method', y='TTC', order=method_order, 
                palette=method_colors)
    plt.title('TTC Distribution by Method')
    plt.xlabel('Method')
    plt.ylabel('TTC (seconds)')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Violin plot
    plt.subplot(2, 2, 3)
    sns.violinplot(data=df_box, x='method', y='TTC', order=method_order,
                   palette=method_colors, inner='quartile', cut=0)
    plt.title('TTC Density by Method')
    plt.xlabel('Method')
    plt.ylabel('TTC (seconds)')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Method differences relative to percentile_median
    plt.subplot(2, 2, 4)
    baseline = 'percentile_median'
    df_diff = df_preceding[['frame_index']].copy()
    for method in method_order:
        if method != baseline:
            df_diff[f'{method}_diff'] = df_preceding[method] - df_preceding[baseline]
    
    # Plot differences
    diff_columns = [col for col in df_diff.columns if col != 'frame_index']
    if len(diff_columns) > 0:
        df_diff_melted = df_diff.melt(
            id_vars=['frame_index'],
            value_vars=diff_columns,
            var_name='method',
            value_name='difference'
        )
        # Extract method name from column (remove _diff suffix)
        df_diff_melted['method'] = df_diff_melted['method'].str.replace('_diff', '')
        sns.lineplot(data=df_diff_melted, x='frame_index', y='difference', 
                    hue='method', style='method', markers=True,
                    palette=method_colors, dashes=False)
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        plt.title(f'TTC Difference from {baseline}')
        plt.xlabel('Frame Index')
        plt.ylabel('TTC Difference (seconds)')
        plt.grid(True, alpha=0.3)
        # Set integer ticks on x-axis
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    
    # Save plots
    output_path = f"output/{output_prefix}_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {output_path}")
    
    return output_path


def print_statistics(df, method_order):
    """Print summary statistics for each method."""
    
    print("\n" + "="*80)
    print("FP.2: TTC METHOD COMPARISON STATISTICS")
    print("="*80)
    
    for method in method_order:
        if method not in df.columns:
            continue
            
        valid_data = df[method].dropna()
        total_count = len(df)
        valid_count = len(valid_data)
        nan_count = total_count - valid_count
        
        print(f"\n{method}:")
        print(f"  Valid samples: {valid_count} / {total_count} ({100*valid_count/total_count:.1f}%)")
        print(f"  NaN count: {nan_count} ({100*nan_count/total_count:.1f}%)")
        
        if len(valid_data) > 0:
            print(f"  Mean TTC: {valid_data.mean():.2f} s")
            print(f"  Median TTC: {valid_data.median():.2f} s")
            print(f"  Std Dev: {valid_data.std():.2f} s")
            print(f"  Min TTC: {valid_data.min():.2f} s")
            print(f"  Max TTC: {valid_data.max():.2f} s")
            print(f"  25th percentile: {valid_data.quantile(0.25):.2f} s")
            print(f"  75th percentile: {valid_data.quantile(0.75):.2f} s")


def analyze_stability(df, method_order):
    """Analyze TTC stability across frames."""
    
    print("\n" + "="*80)
    print("FP.2: TTC STABILITY ANALYSIS")
    print("="*80)
    
    # Filter for preceding vehicle using track_id from file
    tracked_track_id = get_tracked_vehicle_track_id()
    
    if 'track_id' in df.columns and tracked_track_id is not None:
        # Use the track_id from the file
        track_data = df[df['track_id'] == tracked_track_id]
        if len(track_data) > 0:
            df_preceding = track_data.copy()
        else:
            print(f"  Warning: track_id={tracked_track_id} not found in data, using all data")
            df_preceding = df.copy()
    elif 'track_id' in df.columns:
        # track_id file not found, fallback to most common track_id
        track_ids = df['track_id'].mode()
        if len(track_ids) > 0:
            tracked_track_id = track_ids[0]
            df_preceding = df[df['track_id'] == tracked_track_id].copy()
        else:
            df_preceding = df.copy()
    else:
        # Fallback: try curr_box_id == 1 then 0 (legacy behavior)
        df_preceding = df[df['curr_box_id'] == 1].copy()
        if len(df_preceding) == 0:
            df_preceding = df[df['curr_box_id'] == 0].copy()
        if len(df_preceding) == 0:
            df_preceding = df.copy()
    
    for method in method_order:
        if method not in df_preceding.columns:
            continue
            
        valid_data = df_preceding[method].dropna()
        
        if len(valid_data) < 2:
            print(f"\n{method}: Insufficient data for stability analysis")
            continue
        
        # Calculate frame-to-frame differences
        diffs = valid_data.diff().abs()
        
        print(f"\n{method}:")
        print(f"  Mean frame-to-frame change: {diffs.mean():.2f} s")
        print(f"  Median frame-to-frame change: {diffs.median():.2f} s")
        print(f"  Max frame-to-frame change: {diffs.max():.2f} s")
        print(f"  Std Dev of changes: {diffs.std():.2f} s")
        
        # Count large jumps (> 2 seconds difference)
        large_jumps = (diffs > 2.0).sum()
        print(f"  Large jumps (>2s): {large_jumps} ({100*large_jumps/len(diffs):.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='FP.2 Lidar TTC Analysis - Compare different filtering methods'
    )
    parser.add_argument('--csv', type=str, default='output/ttc_lidar_comparison.csv',
                       help='Path to TTC comparison CSV file (default: output/ttc_lidar_comparison.csv)')
    parser.add_argument('--output', type=str, default='fp2_ttc',
                       help='Output prefix for plots (default: fp2_ttc, saves to output/fp2_ttc_comparison.png)')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively')
    args = parser.parse_args()
    
    # Define method order
    method_order = ["unfiltered", "percentile_filtered", "percentile_mean", "percentile_median"]
    
    # Load data
    df = load_data(args.csv)
    
    # Print info
    print(f"Loaded {len(df)} TTC records from {args.csv}")
    print(f"Frames: {df['frame_index'].min()} to {df['frame_index'].max()}")
    print(f"Columns: {list(df.columns)}")
    
    # Print statistics
    print_statistics(df, method_order)
    
    # Stability analysis
    analyze_stability(df, method_order)
    
    # Generate plots
    plot_path = plot_ttc_comparison(df, args.output)
    
    if args.show:
        # Display plots
        plt.figure(figsize=(16, 12))
        plt.suptitle('FP.2: Lidar TTC Method Comparison', fontsize=16, fontweight='bold')
        
        # Filter for preceding vehicle using track_id from file
        tracked_track_id = get_tracked_vehicle_track_id()
        
        if 'track_id' in df.columns and tracked_track_id is not None:
            # Use the track_id from the file
            track_data = df[df['track_id'] == tracked_track_id]
            if len(track_data) > 0:
                df_preceding = track_data.copy()
            else:
                print(f"  Warning: track_id={tracked_track_id} not found in data, using all data")
                df_preceding = df.copy()
        elif 'track_id' in df.columns:
            # track_id file not found, fallback to most common track_id
            track_ids = df['track_id'].mode()
            if len(track_ids) > 0:
                df_preceding = df[df['track_id'] == track_ids[0]].copy()
            else:
                df_preceding = df.copy()
        else:
            df_preceding = df[df['curr_box_id'] == 0].copy()
            if len(df_preceding) == 0:
                df_preceding = df.copy()
        
        plt.subplot(2, 2, 1)
        for method in method_order:
            if method in df_preceding.columns:
                method_data = df_preceding.dropna(subset=[method])
                plt.plot(method_data['frame_index'], method_data[method], 
                        'o-', label=method, linewidth=2, markersize=4)
        plt.title('TTC Over Frames')
        plt.xlabel('Frame Index')
        plt.ylabel('TTC (seconds)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    print(f"\nAnalysis complete!")
    print(f"Plots saved to: {plot_path}")


if __name__ == '__main__':
    main()
