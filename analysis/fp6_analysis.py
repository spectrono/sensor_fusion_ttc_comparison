#!/usr/bin/env python3
"""
FP.6 Performance Assessment 2 - Detector/Descriptor Combination Analysis

Analyzes camera-based TTC estimation across all detector/descriptor combinations.
The main drivers for evaluation are:
  1. Smoothness of TTC plots (frame-to-frame consistency)
  2. Largest outliers (combinations producing inconsistent TTC values)

The script generates:
  - fp6_ttc_overview.png: All combinations plotted together with lidar reference
  - fp6_smoothness_ranking.png: Smoothness metrics bar chart
  - fp6_outlier_analysis.png: Outlier frames per combination
  - fp6_top_combinations.png: Best/worst combinations side by side
  - fp6_statistics_table.png: Summary statistics table

Usage:
    python fp6_analysis.py
    python fp6_analysis.py --csv output/ttc_camera_combinations.csv --lidar-csv output/ttc_lidar_comparison.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(combo_csv, lidar_csv=None):
    """Load combination TTC data and optional lidar reference."""
    df = pd.read_csv(combo_csv)
    
    df_lidar = None
    if lidar_csv and os.path.exists(lidar_csv):
        df_lidar = pd.read_csv(lidar_csv)
    
    return df, df_lidar


def get_tracked_vehicle_track_id(track_id_file=None):
    """Read the tracked preceding vehicle track_id from file."""
    if track_id_file is None:
        track_id_file = "output/tracked_preceding_vehicle_track_id.txt"
    
    if os.path.exists(track_id_file):
        try:
            with open(track_id_file, 'r') as f:
                return int(f.read().strip())
        except (ValueError, IOError):
            return None
    return None


def filter_by_track_id(df, df_lidar=None):
    """Filter data by tracked vehicle track_id."""
    track_id = get_tracked_vehicle_track_id()
    
    if track_id is not None and 'track_id' in df.columns:
        df = df[df['track_id'] == track_id].copy()
        if df_lidar is not None and 'track_id' in df_lidar.columns:
            df_lidar = df_lidar[df_lidar['track_id'] == track_id].copy()
    
    return df, df_lidar


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_smoothness_metrics(df):
    """
    Compute smoothness metrics for each detector/descriptor combination.
    
    Returns a DataFrame with one row per combination containing:
    - mean_ttc, std_ttc, min_ttc, max_ttc
    - mean_frame_change: mean of absolute frame-to-frame differences
    - max_frame_change: largest single frame-to-frame jump
    - num_valid: number of valid (non-NaN) TTC values
    - num_outliers: number of frames where |TTC - median| > 2 * std
    - smoothness_score: 1 / (1 + mean_frame_change) — higher is smoother
    """
    results = []
    
    for (det, desc), group in df.groupby(['detector', 'descriptor']):
        group = group.sort_values('frame_index')
        ttcs = group['ttc_camera'].values
        valid_mask = ~np.isnan(ttcs)
        valid_ttcs = ttcs[valid_mask]
        
        if len(valid_ttcs) < 2:
            results.append({
                'detector': det, 'descriptor': desc,
                'mean_ttc': np.nan, 'std_ttc': np.nan,
                'min_ttc': np.nan, 'max_ttc': np.nan,
                'mean_frame_change': np.nan, 'max_frame_change': np.nan,
                'num_valid': len(valid_ttcs), 'num_total': len(ttcs),
                'num_outliers': 0, 'outlier_pct': 0,
                'smoothness_score': 0
            })
            continue
        
        # Frame-to-frame differences
        diffs = np.abs(np.diff(valid_ttcs))
        mean_change = np.mean(diffs)
        max_change = np.max(diffs)
        
        # Outlier detection: frames where TTC deviates > 2 std from median
        median_ttc = np.median(valid_ttcs)
        std_ttc = np.std(valid_ttcs)
        if std_ttc > 0:
            outlier_threshold = 2 * std_ttc
            num_outliers = np.sum(np.abs(valid_ttcs - median_ttc) > outlier_threshold)
        else:
            num_outliers = 0
        
        results.append({
            'detector': det, 'descriptor': desc,
            'mean_ttc': np.mean(valid_ttcs),
            'std_ttc': std_ttc,
            'min_ttc': np.min(valid_ttcs),
            'max_ttc': np.max(valid_ttcs),
            'mean_frame_change': mean_change,
            'max_frame_change': max_change,
            'num_valid': len(valid_ttcs),
            'num_total': len(ttcs),
            'num_outliers': num_outliers,
            'outlier_pct': 100.0 * num_outliers / len(valid_ttcs),
            'smoothness_score': 1.0 / (1.0 + mean_change)
        })
    
    return pd.DataFrame(results)


def identify_largest_outliers(df, top_n=5):
    """
    Identify the largest outlier frames across all combinations.
    
    For each frame, computes the spread (max - min) of TTC across all
    combinations, and identifies frames with the largest spread.
    """
    results = []
    
    for frame in sorted(df['frame_index'].unique()):
        frame_data = df[df['frame_index'] == frame]
        valid_ttcs = frame_data['ttc_camera'].dropna()
        
        if len(valid_ttcs) < 2:
            continue
        
        results.append({
            'frame_index': frame,
            'ttc_mean': valid_ttcs.mean(),
            'ttc_median': valid_ttcs.median(),
            'ttc_min': valid_ttcs.min(),
            'ttc_max': valid_ttcs.max(),
            'ttc_spread': valid_ttcs.max() - valid_ttcs.min(),
            'ttc_std': valid_ttcs.std(),
            'num_valid': len(valid_ttcs)
        })
    
    df_frames = pd.DataFrame(results)
    df_frames = df_frames.sort_values('ttc_spread', ascending=False)
    return df_frames


def identify_combination_outliers(df, smoothness_df):
    """
    Identify combinations with the most extreme outliers.
    
    For each combination, finds the frames where TTC deviates most
    from the cross-combination median for that frame.
    """
    outlier_records = []
    
    # Compute per-frame median across all combinations
    frame_medians = df.groupby('frame_index')['ttc_camera'].median()
    
    for (det, desc), group in df.groupby(['detector', 'descriptor']):
        group = group.sort_values('frame_index')
        for _, row in group.iterrows():
            if np.isnan(row['ttc_camera']):
                continue
            frame = row['frame_index']
            frame_median = frame_medians.get(frame, np.nan)
            if np.isnan(frame_median):
                continue
            
            deviation = abs(row['ttc_camera'] - frame_median)
            relative_dev = deviation / frame_median if frame_median > 0 else 0
            
            outlier_records.append({
                'detector': det,
                'descriptor': desc,
                'frame_index': frame,
                'ttc_camera': row['ttc_camera'],
                'frame_median': frame_median,
                'absolute_deviation': deviation,
                'relative_deviation': relative_dev
            })
    
    return pd.DataFrame(outlier_records).sort_values('absolute_deviation', ascending=False)


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_ttc_overview(df, df_lidar, output_dir):
    """
    Plot all combinations' TTC over frames with lidar reference.
    """
    fig, axes = plt.subplots(2, 1, figsize=(18, 14))
    
    # Top plot: All combinations
    ax1 = axes[0]
    for (det, desc), group in df.groupby(['detector', 'descriptor']):
        group = group.sort_values('frame_index')
        ax1.plot(group['frame_index'], group['ttc_camera'], 
                alpha=0.4, linewidth=0.8, label=f"{det}/{desc}")
    
    if df_lidar is not None:
        ax1.plot(df_lidar['frame_index'], df_lidar['unfiltered'], 
                'k-o', linewidth=3, markersize=6, label='Lidar UNFILTERED', zorder=10)
    
    ax1.set_xlabel('Frame Index', fontsize=12)
    ax1.set_ylabel('TTC (seconds)', fontsize=12)
    ax1.set_title('FP.6: Camera TTC for All Detector/Descriptor Combinations', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=6, ncol=3)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 60)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Bottom plot: Per-detector subplots
    ax2 = axes[1]
    detectors = sorted(df['detector'].unique())
    n_det = len(detectors)
    
    # Create inset axes for each detector
    gs = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=ax2.get_subplotspec() 
                                          if hasattr(ax2, 'get_subplotspec') else None)
    ax2.remove()
    
    for i, det in enumerate(detectors):
        row, col = divmod(i, 4)
        ax = fig.add_subplot(gs[row, col])
        det_data = df[df['detector'] == det]
        
        for desc, group in det_data.groupby('descriptor'):
            group = group.sort_values('frame_index')
            ax.plot(group['frame_index'], group['ttc_camera'], 
                   alpha=0.7, linewidth=1.5, label=desc)
        
        if df_lidar is not None:
            ax.plot(df_lidar['frame_index'], df_lidar['unfiltered'], 
                   'k--o', linewidth=1.5, markersize=3, alpha=0.5)
        
        ax.set_title(f'{det}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Frame', fontsize=8)
        ax.set_ylabel('TTC (s)', fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 60)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.suptitle('FP.6: TTC by Detector (per descriptor, dashed = Lidar reference)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = os.path.join(output_dir, 'fp6_ttc_overview.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved TTC overview plot: {output_path}")
    return output_path


def plot_smoothness_ranking(smoothness_df, output_dir):
    """
    Plot smoothness metrics ranking for all combinations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Sort by smoothness score (descending = best first)
    df_sorted = smoothness_df.sort_values('smoothness_score', ascending=False)
    labels = [f"{d}/{s}" for d, s in zip(df_sorted['detector'], df_sorted['descriptor'])]
    
    # Plot 1: Smoothness score
    ax1 = axes[0, 0]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(df_sorted)))
    bars = ax1.barh(range(len(df_sorted)), df_sorted['smoothness_score'], color=colors)
    ax1.set_yticks(range(len(df_sorted)))
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel('Smoothness Score (1/(1+mean_change))', fontsize=11)
    ax1.set_title('Smoothness Ranking (higher = smoother)', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Mean frame-to-frame change
    ax2 = axes[0, 1]
    df_sorted2 = smoothness_df.sort_values('mean_frame_change')
    labels2 = [f"{d}/{s}" for d, s in zip(df_sorted2['detector'], df_sorted2['descriptor'])]
    colors2 = plt.cm.RdYlGn(np.linspace(0.9, 0.2, len(df_sorted2)))
    ax2.barh(range(len(df_sorted2)), df_sorted2['mean_frame_change'], color=colors2)
    ax2.set_yticks(range(len(df_sorted2)))
    ax2.set_yticklabels(labels2, fontsize=8)
    ax2.set_xlabel('Mean Frame-to-Frame Change (s)', fontsize=11)
    ax2.set_title('Mean TTC Change (lower = smoother)', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Plot 3: Std dev
    ax3 = axes[1, 0]
    df_sorted3 = smoothness_df.sort_values('std_ttc')
    labels3 = [f"{d}/{s}" for d, s in zip(df_sorted3['detector'], df_sorted3['descriptor'])]
    colors3 = plt.cm.RdYlGn(np.linspace(0.9, 0.2, len(df_sorted3)))
    ax3.barh(range(len(df_sorted3)), df_sorted3['std_ttc'], color=colors3)
    ax3.set_yticks(range(len(df_sorted3)))
    ax3.set_yticklabels(labels3, fontsize=8)
    ax3.set_xlabel('TTC Std Dev (s)', fontsize=11)
    ax3.set_title('TTC Std Dev (lower = more consistent)', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Plot 4: Number of valid frames
    ax4 = axes[1, 1]
    df_sorted4 = smoothness_df.sort_values('num_valid', ascending=False)
    labels4 = [f"{d}/{s}" for d, s in zip(df_sorted4['detector'], df_sorted4['descriptor'])]
    colors4 = plt.cm.RdYlGn(np.linspace(0.9, 0.2, len(df_sorted4)))
    ax4.barh(range(len(df_sorted4)), df_sorted4['num_valid'], color=colors4)
    ax4.set_yticks(range(len(df_sorted4)))
    ax4.set_yticklabels(labels4, fontsize=8)
    ax4.set_xlabel('Number of Valid TTC Values', fontsize=11)
    ax4.set_title('Valid Frames (higher = more robust)', fontsize=12, fontweight='bold')
    ax4.invert_yaxis()
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.set_xlim(0, 19)
    
    plt.suptitle('FP.6: Smoothness and Consistency Metrics', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'fp6_smoothness_ranking.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved smoothness ranking plot: {output_path}")
    return output_path


def plot_outlier_analysis(df, frame_outliers, combo_outliers, output_dir):
    """
    Plot outlier analysis: frame spreads and combination deviations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Plot 1: TTC spread per frame (max - min across combinations)
    ax1 = axes[0, 0]
    df_sorted = frame_outliers.sort_values('frame_index')
    colors = ['red' if s > df_sorted['ttc_spread'].median() else 'steelblue' 
              for s in df_sorted['ttc_spread']]
    ax1.bar(df_sorted['frame_index'], df_sorted['ttc_spread'], color=colors, alpha=0.8)
    ax1.set_xlabel('Frame Index', fontsize=11)
    ax1.set_ylabel('TTC Spread (max - min) [s]', fontsize=11)
    ax1.set_title('TTC Spread Across Combinations per Frame', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Annotate top 3 outlier frames
    top3 = frame_outliers.head(3)
    for _, row in top3.iterrows():
        ax1.annotate(f"Frame {int(row['frame_index'])}\n({row['ttc_spread']:.1f}s)",
                    (row['frame_index'], row['ttc_spread']),
                    textcoords="offset points", xytext=(0, 10), ha='center',
                    fontsize=8, fontweight='bold', color='red')
    
    # Plot 2: Top 10 combination-frame outliers (largest deviations from frame median)
    ax2 = axes[0, 1]
    top_outliers = combo_outliers.head(15)
    labels = [f"{d}/{s}\nF{f}" for d, s, f in zip(
        top_outliers['detector'], top_outliers['descriptor'], top_outliers['frame_index'])]
    ax2.barh(range(len(top_outliers)), top_outliers['absolute_deviation'], color='crimson', alpha=0.8)
    ax2.set_yticks(range(len(top_outliers)))
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel('Absolute Deviation from Frame Median (s)', fontsize=11)
    ax2.set_title('Top 15 Combination-Frame Outliers', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Plot 3: Outlier count per combination
    ax3 = axes[1, 0]
    outlier_counts = combo_outliers.groupby(['detector', 'descriptor']).size().reset_index(name='count')
    outlier_counts = outlier_counts.sort_values('count', ascending=False)
    combo_labels = [f"{d}/{s}" for d, s in zip(outlier_counts['detector'], outlier_counts['descriptor'])]
    ax3.barh(range(len(outlier_counts)), outlier_counts['count'], color='orange', alpha=0.8)
    ax3.set_yticks(range(len(outlier_counts)))
    ax3.set_yticklabels(combo_labels, fontsize=8)
    ax3.set_xlabel('Number of Outlier Frames', fontsize=11)
    ax3.set_title('Outlier Count per Combination', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Plot 4: TTC range (min-max) per combination
    ax4 = axes[1, 1]
    range_data = df.groupby(['detector', 'descriptor']).agg(
        ttc_min=('ttc_camera', 'min'),
        ttc_max=('ttc_camera', 'max')
    ).reset_index()
    range_data['ttc_range'] = range_data['ttc_max'] - range_data['ttc_min']
    range_data = range_data.sort_values('ttc_range')
    range_labels = [f"{d}/{s}" for d, s in zip(range_data['detector'], range_data['descriptor'])]
    ax4.barh(range(len(range_data)), range_data['ttc_range'], color='purple', alpha=0.8)
    ax4.set_yticks(range(len(range_data)))
    ax4.set_yticklabels(range_labels, fontsize=8)
    ax4.set_xlabel('TTC Range (max - min) [s]', fontsize=11)
    ax4.set_title('TTC Range per Combination', fontsize=12, fontweight='bold')
    ax4.invert_yaxis()
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('FP.6: Outlier Analysis', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'fp6_outlier_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved outlier analysis plot: {output_path}")
    return output_path


def plot_top_combinations(df, df_lidar, smoothness_df, output_dir):
    """
    Plot top 3 best and top 3 worst combinations side by side.
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Best: highest smoothness score
    best = smoothness_df.nlargest(3, 'smoothness_score')
    # Worst: lowest smoothness score
    worst = smoothness_df.nsmallest(3, 'smoothness_score')
    
    # Plot best combinations
    for i, (_, row) in enumerate(best.iterrows()):
        ax = axes[0, i]
        combo_data = df[(df['detector'] == row['detector']) & 
                        (df['descriptor'] == row['descriptor'])].sort_values('frame_index')
        
        ax.plot(combo_data['frame_index'], combo_data['ttc_camera'], 
               'b-o', linewidth=2, markersize=5, label='Camera TTC')
        
        if df_lidar is not None:
            ax.plot(df_lidar['frame_index'], df_lidar['unfiltered'], 
                   'r--s', linewidth=2, markersize=4, alpha=0.7, label='Lidar UNFILTERED')
        
        ax.set_title(f"BEST #{i+1}: {row['detector']}/{row['descriptor']}\n"
                    f"smoothness={row['smoothness_score']:.3f}, mean_change={row['mean_frame_change']:.2f}s",
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Frame Index', fontsize=10)
        ax.set_ylabel('TTC (s)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 40)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Plot worst combinations
    for i, (_, row) in enumerate(worst.iterrows()):
        ax = axes[1, i]
        combo_data = df[(df['detector'] == row['detector']) & 
                        (df['descriptor'] == row['descriptor'])].sort_values('frame_index')
        
        ax.plot(combo_data['frame_index'], combo_data['ttc_camera'], 
               'b-o', linewidth=2, markersize=5, label='Camera TTC')
        
        if df_lidar is not None:
            ax.plot(df_lidar['frame_index'], df_lidar['unfiltered'], 
                   'r--s', linewidth=2, markersize=4, alpha=0.7, label='Lidar UNFILTERED')
        
        ax.set_title(f"WORST #{i+1}: {row['detector']}/{row['descriptor']}\n"
                    f"smoothness={row['smoothness_score']:.3f}, mean_change={row['mean_frame_change']:.2f}s",
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Frame Index', fontsize=10)
        ax.set_ylabel('TTC (s)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 60)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.suptitle('FP.6: Best vs Worst Combinations (by Smoothness)', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'fp6_top_combinations.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved top combinations plot: {output_path}")
    return output_path


def plot_statistics_table(smoothness_df, output_dir):
    """
    Generate a summary statistics table as a figure.
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('off')
    
    # Sort by smoothness score
    df = smoothness_df.sort_values('smoothness_score', ascending=False).reset_index(drop=True)
    
    # Prepare table data
    table_data = []
    headers = ['Rank', 'Detector', 'Descriptor', 'Mean TTC', 'Std Dev', 
               'Mean Change', 'Max Change', 'Valid', 'Outliers', 'Smoothness']
    
    for i, row in df.iterrows():
        table_data.append([
            i + 1,
            row['detector'],
            row['descriptor'],
            f"{row['mean_ttc']:.2f}s" if not np.isnan(row['mean_ttc']) else "N/A",
            f"{row['std_ttc']:.2f}s" if not np.isnan(row['std_ttc']) else "N/A",
            f"{row['mean_frame_change']:.2f}s" if not np.isnan(row['mean_frame_change']) else "N/A",
            f"{row['max_frame_change']:.2f}s" if not np.isnan(row['max_frame_change']) else "N/A",
            f"{row['num_valid']}/{row['num_total']}",
            f"{row['num_outliers']} ({row['outlier_pct']:.0f}%)",
            f"{row['smoothness_score']:.3f}" if not np.isnan(row['smoothness_score']) else "0"
        ])
    
    table = ax.table(cellText=table_data, colLabels=headers, loc='center',
                    cellLoc='center', colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    
    # Color the best (green) and worst (red) rows
    for i in range(len(table_data)):
        if i < 3:
            table[i + 1, 0].set_facecolor('#90EE90')
        elif i >= len(table_data) - 3:
            table[i + 1, 0].set_facecolor('#FFB6C1')
    
    ax.set_title('FP.6: Detector/Descriptor Combination Statistics (sorted by smoothness)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'fp6_statistics_table.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved statistics table: {output_path}")
    return output_path


def plot_outlier_frames_detail(df, df_lidar, frame_outliers, output_dir):
    """
    Plot detailed view of the top 3 outlier frames showing all combinations.
    """
    top3_frames = frame_outliers.head(3)['frame_index'].values
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for i, frame in enumerate(top3_frames):
        ax = axes[i]
        frame_data = df[df['frame_index'] == frame].sort_values('ttc_camera')
        
        # Create bar chart of TTC values for this frame
        labels = [f"{d}/{s}" for d, s in zip(frame_data['detector'], frame_data['descriptor'])]
        ttcs = frame_data['ttc_camera'].values
        valid_mask = ~np.isnan(ttcs)
        
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(frame_data)))
        bars = ax.barh(range(len(frame_data)), 
                       [t if not np.isnan(t) else 0 for t in ttcs], 
                       color=colors, alpha=0.8)
        
        # Mark NaN values
        for j, t in enumerate(ttcs):
            if np.isnan(t):
                ax.text(0.5, j, 'NaN', va='center', fontsize=7, color='red')
        
        # Add frame median line
        if df_lidar is not None:
            lidar_val = df_lidar[df_lidar['frame_index'] == frame]['unfiltered'].values
            if len(lidar_val) > 0:
                ax.axvline(lidar_val[0], color='red', linestyle='--', linewidth=2, label=f'Lidar={lidar_val[0]:.1f}s')
        
        frame_median = np.nanmedian(ttcs)
        ax.axvline(frame_median, color='green', linestyle=':', linewidth=2, label=f'Median={frame_median:.1f}s')
        
        ax.set_yticks(range(len(frame_data)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel('TTC (s)', fontsize=10)
        ax.set_title(f'Frame {frame} (spread={frame_outliers.iloc[i]["ttc_spread"]:.1f}s)', 
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('FP.6: Top 3 Outlier Frames - TTC per Combination', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'fp6_outlier_frames_detail.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved outlier frames detail plot: {output_path}")
    return output_path


# =============================================================================
# REPORT
# =============================================================================

def print_report(smoothness_df, frame_outliers, combo_outliers):
    """Print analysis summary to console."""
    print("\n" + "=" * 80)
    print("FP.6: PERFORMANCE ASSESSMENT 2 - COMBINATION ANALYSIS REPORT")
    print("=" * 80)
    
    print(f"\nTotal combinations: {len(smoothness_df)}")
    print(f"Total frames analyzed: {smoothness_df['num_total'].max()}")
    
    print("\n" + "-" * 80)
    print("SMOOTHNESS RANKING (Top 5 best / Top 5 worst)")
    print("-" * 80)
    
    best5 = smoothness_df.nlargest(5, 'smoothness_score')
    worst5 = smoothness_df.nsmallest(5, 'smoothness_score')
    
    print("\n  BEST (smoothest TTC curves):")
    for _, row in best5.iterrows():
        print(f"    {row['detector']:10s}/{row['descriptor']:6s}  "
              f"smoothness={row['smoothness_score']:.3f}  "
              f"mean_change={row['mean_frame_change']:.2f}s  "
              f"std={row['std_ttc']:.2f}s  "
              f"valid={row['num_valid']}/{row['num_total']}")
    
    print("\n  WORST (most inconsistent TTC curves):")
    for _, row in worst5.iterrows():
        print(f"    {row['detector']:10s}/{row['descriptor']:6s}  "
              f"smoothness={row['smoothness_score']:.3f}  "
              f"mean_change={row['mean_frame_change']:.2f}s  "
              f"std={row['std_ttc']:.2f}s  "
              f"valid={row['num_valid']}/{row['num_total']}")
    
    print("\n" + "-" * 80)
    print("LARGEST OUTLIER FRAMES (highest TTC spread across combinations)")
    print("-" * 80)
    for _, row in frame_outliers.head(5).iterrows():
        print(f"  Frame {int(row['frame_index']):2d}: "
              f"spread={row['ttc_spread']:.1f}s  "
              f"min={row['ttc_min']:.1f}s  max={row['ttc_max']:.1f}s  "
              f"median={row['ttc_median']:.1f}s  "
              f"std={row['ttc_std']:.1f}s")
    
    print("\n" + "-" * 80)
    print("TOP 10 COMBINATION-FRAME OUTLIERS (largest deviation from frame median)")
    print("-" * 80)
    for _, row in combo_outliers.head(10).iterrows():
        print(f"  {row['detector']:10s}/{row['descriptor']:6s}  "
              f"Frame {int(row['frame_index']):2d}: "
              f"TTC={row['ttc_camera']:.1f}s  "
              f"median={row['frame_median']:.1f}s  "
              f"dev={row['absolute_deviation']:.1f}s ({row['relative_deviation']*100:.0f}%)")
    
    print("\n" + "=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='FP.6 Performance Assessment 2 - Detector/Descriptor Combination Analysis'
    )
    parser.add_argument('--csv', type=str,
                       default='output/ttc_camera_combinations.csv',
                       help='Path to combination TTC CSV file')
    parser.add_argument('--lidar-csv', type=str,
                       default='output/ttc_lidar_comparison.csv',
                       help='Path to LIDAR TTC comparison CSV file (for reference)')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("FP.6 PERFORMANCE ASSESSMENT 2 - COMBINATION ANALYSIS")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading data from: {args.csv}")
    try:
        df, df_lidar = load_data(args.csv, args.lidar_csv)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the 3D_object_tracking executable with bTestAllCombinationsTTC = true first.")
        return
    
    # Filter by track_id
    df, df_lidar = filter_by_track_id(df, df_lidar)
    
    print(f"Loaded {len(df)} combination records")
    if df_lidar is not None:
        print(f"Loaded {len(df_lidar)} lidar reference records")
    
    # Analyze
    print("\nComputing smoothness metrics...")
    smoothness_df = compute_smoothness_metrics(df)
    
    print("Identifying largest outlier frames...")
    frame_outliers = identify_largest_outliers(df)
    
    print("Identifying combination-frame outliers...")
    combo_outliers = identify_combination_outliers(df, smoothness_df)
    
    # Print report
    print_report(smoothness_df, frame_outliers, combo_outliers)
    
    # Generate plots
    print("\nGenerating plots...")
    os.makedirs(args.output, exist_ok=True)
    
    plot_ttc_overview(df, df_lidar, args.output)
    plot_smoothness_ranking(smoothness_df, args.output)
    plot_outlier_analysis(df, frame_outliers, combo_outliers, args.output)
    plot_top_combinations(df, df_lidar, smoothness_df, args.output)
    plot_statistics_table(smoothness_df, args.output)
    plot_outlier_frames_detail(df, df_lidar, frame_outliers, args.output)
    
    # Save smoothness metrics to CSV
    csv_path = os.path.join(args.output, 'fp6_smoothness_metrics.csv')
    smoothness_df.to_csv(csv_path, index=False)
    print(f"\nSaved smoothness metrics CSV: {csv_path}")
    
    print("\n" + "=" * 80)
    print("FP.6 ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput files saved to: {os.path.abspath(args.output)}")
    print("  - fp6_ttc_overview.png")
    print("  - fp6_smoothness_ranking.png")
    print("  - fp6_outlier_analysis.png")
    print("  - fp6_top_combinations.png")
    print("  - fp6_statistics_table.png")
    print("  - fp6_outlier_frames_detail.png")
    print("  - fp6_smoothness_metrics.csv")


if __name__ == '__main__':
    main()
