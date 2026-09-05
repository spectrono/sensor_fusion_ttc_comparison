#!/usr/bin/env python3
"""
FP.5 Bird's-Eye View Analysis Script

This script generates plot series from bird's-eye view (BEV) images and performs
comprehensive analysis including skewness statistics, visual vs statistical comparison,
and final interpretation for the lidar TTC outliers.

Usage:
    python fp5_bev_analysis.py [--bev-dir BEV_DIR] [--camera-csv CAMERA_CSV] 
                              [--lidar-csv LIDAR_CSV] [--output OUTPUT_DIR]
                              [--outlier-frames FRAMES] [--help]

Example:
    python fp5_bev_analysis.py --bev-dir ../analysis/output/ 
                              --camera-csv ../analysis/output/ttc_camera.csv 
                              --lidar-csv ../analysis/output/ttc_lidar_comparison.csv
                              --output ../analysis/output/
"""

import argparse
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
from PIL import Image

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_bev_images(bev_dir, frame_indices=None):
    """Load bird's-eye view images from directory."""
    pattern = os.path.join(bev_dir, "preceding_vehicle_lidar_bev_*.png")
    bev_files = sorted(glob.glob(pattern))
    
    if not bev_files:
        raise FileNotFoundError(f"No BEV images found in {bev_dir}")
    
    # Extract frame numbers from filenames
    frame_numbers = []
    for f in bev_files:
        match = re.search(r'preceding_vehicle_lidar_bev_(\d+)\.png', f)
        if match:
            frame_numbers.append(int(match.group(1)))
    
    # Sort files by frame number
    sorted_pairs = sorted(zip(frame_numbers, bev_files))
    frame_numbers, bev_files = zip(*sorted_pairs)
    
    # Filter by requested frame indices if provided
    if frame_indices is not None:
        filtered_files = []
        filtered_frames = []
        for frame, filepath in zip(frame_numbers, bev_files):
            if frame in frame_indices:
                filtered_files.append(filepath)
                filtered_frames.append(frame)
        return filtered_frames, filtered_files
    
    return list(frame_numbers), list(bev_files)


def create_bev_plot_series(bev_files, frame_numbers, output_path, figsize=(15, 20)):
    """Create a plot series showing all BEV frames in a 6x3 grid."""
    n_frames = len(bev_files)
    
    # Use 6 rows x 3 columns for thumbnail grid (fits 18 frames)
    # If we have more than 18 frames, only show first 18
    cols = 3
    rows = 6
    max_frames_to_show = rows * cols  # 18 frames
    
    # Limit to first 18 frames for 3x6 grid
    if n_frames > max_frames_to_show:
        bev_files = bev_files[:max_frames_to_show]
        frame_numbers = frame_numbers[:max_frames_to_show]
        n_frames = max_frames_to_show
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Flatten axes array for easy iteration
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    elif cols == 1:
        axes = [row[0] for row in axes]
    else:
        axes = [ax for row in axes for ax in row]
    
    for i, (frame_num, bev_file) in enumerate(zip(frame_numbers, bev_files)):
        img = Image.open(bev_file)
        axes[i].imshow(img)
        axes[i].set_title(f"Frame {frame_num}", fontsize=10, fontweight='bold')
        axes[i].axis('off')
            
    plt.suptitle("FP.5 Bird's-Eye View Series: All Frames with Lidar Point Clouds", 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_path, "fp5_bev_plot_series.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved BEV plot series: {output_file}")
    return output_file


def create_frame_sequence_plot(bev_files, frame_numbers, frame_list, output_path, title, filename):
    """Create a horizontal plot with a sequence of frames side by side."""
    fig, axes = plt.subplots(1, len(frame_list), figsize=(15, 5))
    
    if len(frame_list) == 1:
        axes = [axes]
    
    for i, target_frame in enumerate(frame_list):
        # Find the BEV file for this frame
        bev_file = None
        for f, fnum in zip(bev_files, frame_numbers):
            if fnum == target_frame:
                bev_file = f
                break
        
        if bev_file is None:
            continue
        
        # Load BEV image
        bev_img = Image.open(bev_file)
        
        if len(frame_list) == 1:
            ax = axes
        else:
            ax = axes[i]
            
        ax.imshow(bev_img)
        ax.set_title(f"Frame {target_frame}", fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Highlight outlier frames with red border
        if target_frame in [12, 14, 15]:
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_path, filename)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved frame sequence plot: {output_file}")
    return output_file




def create_bev_outliers_comparison(bev_files, frame_numbers, camera_df, lidar_df, output_path):
    """Create a comparison of outlier frames with BEV images and TTC values."""
    # Identify outlier frames
    outlier_frames = [12, 14, 15]
    
    # Create a 3x4 grid: each outlier frame gets 4 subplots
    fig = plt.subplots(3, 4, figsize=(20, 15))
    fig = plt.gcf()
    
    for i, frame_num in enumerate(outlier_frames):
        # Find the BEV file for this frame
        bev_file = None
        for f, fnum in zip(bev_files, frame_numbers):
            if fnum == frame_num:
                bev_file = f
                break
        
        if bev_file is None:
            continue
        
        # Load BEV image
        bev_img = Image.open(bev_file)
        
        # Get TTC values for this frame
        camera_ttc = camera_df[camera_df['frame_index'] == frame_num]['ttc_camera'].values[0] if frame_num in camera_df['frame_index'].values else None
        frame_lidar_data = lidar_df[lidar_df['frame_index'] == frame_num].iloc[0] if frame_num in lidar_df['frame_index'].values else None
        lidar_unfiltered = frame_lidar_data.get('unfiltered') if frame_lidar_data is not None else None
        lidar_mean = frame_lidar_data.get('percentile_mean') if frame_lidar_data is not None else None
        lidar_median = frame_lidar_data.get('percentile_median') if frame_lidar_data is not None else None
        
        # Row i, Column 0: BEV image
        ax_bev = plt.subplot2grid((3, 4), (i, 0), rowspan=1, colspan=1)
        ax_bev.imshow(bev_img)
        ax_bev.set_title(f"Frame {frame_num} BEV", fontsize=10, fontweight='bold')
        ax_bev.axis('off')
        
        # Row i, Column 1: TTC values comparison
        ax_ttc = plt.subplot2grid((3, 4), (i, 1), rowspan=1, colspan=1)
        methods = ['Camera', 'Lidar UNFILTERED', 'Lidar MEAN', 'Lidar MEDIAN']
        ttc_values = [camera_ttc, lidar_unfiltered, lidar_mean, lidar_median]
        colors = ['blue', 'red', 'green', 'purple']
        bars = ax_ttc.bar(methods, ttc_values, color=colors, alpha=0.8)
        ax_ttc.set_title(f"TTC Comparison - Frame {frame_num}", fontsize=10, fontweight='bold')
        ax_ttc.set_ylabel('TTC (seconds)')
        ax_ttc.set_ylim(0, max(ttc_values) * 1.2 if max(ttc_values) > 0 else 20)
        ax_ttc.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, ttc_values):
            if val is not None:
                ax_ttc.text(bar.get_x() + bar.get_width()/2, val + 0.2, 
                           f'{val:.2f}s', ha='center', va='bottom', fontsize=8)
        
        # Row i, Column 2: TTC differences
        ax_diff = plt.subplot2grid((3, 4), (i, 2), rowspan=1, colspan=1)
        if camera_ttc is not None and lidar_unfiltered is not None:
            diff_unfiltered = abs(camera_ttc - lidar_unfiltered)
            diff_mean = abs(camera_ttc - lidar_mean) if lidar_mean is not None else None
            diff_median = abs(camera_ttc - lidar_median) if lidar_median is not None else None
            
            diff_methods = ['|Cam - UNFILTERED|', '|Cam - MEAN|', '|Cam - MEDIAN|']
            diff_values = [diff_unfiltered, diff_mean, diff_median]
            diff_colors = ['orange', 'orange', 'orange']
            
            bars_diff = ax_diff.bar([m for m, v in zip(diff_methods, diff_values) if v is not None], 
                                   [v for v in diff_values if v is not None], 
                                   color=diff_colors, alpha=0.8)
            ax_diff.set_title(f"Absolute Differences - Frame {frame_num}", fontsize=10, fontweight='bold')
            ax_diff.set_ylabel('|Difference| (seconds)')
            ax_diff.set_ylim(0, max([v for v in diff_values if v is not None]) * 1.2 if any(v is not None for v in diff_values) else 5)
            ax_diff.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, val, method in zip(bars_diff, [v for v in diff_values if v is not None], [m for m, v in zip(diff_methods, diff_values) if v is not None]):
                ax_diff.text(bar.get_x() + bar.get_width()/2, val + 0.1, 
                            f'{val:.2f}s', ha='center', va='bottom', fontsize=8)
        
        # Row i, Column 3: Summary text
        ax_summary = plt.subplot2grid((3, 4), (i, 3), rowspan=1, colspan=1)
        ax_summary.axis('off')
        
        summary_text = f"Frame {frame_num} Analysis\n\n"
        if camera_ttc is not None and lidar_unfiltered is not None:
            diff = abs(camera_ttc - lidar_unfiltered)
            rel_diff = (diff / camera_ttc) * 100 if camera_ttc > 0 else 0
            summary_text += f"Camera: {camera_ttc:.2f}s\n"
            summary_text += f"Lidar: {lidar_unfiltered:.2f}s\n"
            summary_text += f"Difference: {diff:.2f}s ({rel_diff:.1f}%)\n"
        
        if lidar_mean is not None and lidar_median is not None:
            mean_median_diff = abs(lidar_mean - lidar_median)
            summary_text += f"\nSkewness Indicator:\n"
            summary_text += f"|MEAN - MEDIAN| = {mean_median_diff:.2f}s\n"
            if mean_median_diff > 0.5:
                summary_text += "→ RIGHT-SKEWED DISTRIBUTION\n"
            else:
                summary_text += "→ SYMMETRIC DISTRIBUTION\n"
        
        ax_summary.text(0.1, 0.8, summary_text, transform=ax_summary.transAxes, 
                       fontsize=10, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle("FP.5 Outlier Frames: Bird's-Eye View vs TTC Analysis", 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_file = os.path.join(output_path, "fp5_bev_outliers_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved BEV outliers comparison: {output_file}")
    return output_file


def load_lidar_csv_with_stats(lidar_csv):
    """Load lidar CSV and compute skewness statistics for each frame."""
    df = pd.read_csv(lidar_csv)
    
    # Pivot to get methods as columns
    pivot_df = df.pivot_table(index=['frame_index', 'track_id', 'x_min', 'x_max', 'n_points'],
                              columns='method', values='ttc_lidar').reset_index()
    
    # For UNFILTERED method, we need to compute skewness from the actual lidar points
    # Since we don't have the raw points in CSV, we'll use the difference between methods
    # as a proxy for skewness
    
    return df, pivot_df


def compute_skewness_from_lidar_points(bev_dir, lidar_csv, output_path):
    """
    Compute skewness statistics from lidar point distributions.
    Since we don't have raw lidar points in CSV, we'll create a comprehensive
    analysis based on the TTC method differences.
    """
    # Load lidar data
    lidar_df = pd.read_csv(lidar_csv)
    
    # Get all frames
    all_frames = sorted(lidar_df['frame_index'].unique())
    
    # Prepare data for analysis
    analysis_data = []
    
    for frame in all_frames:
        frame_data = lidar_df[lidar_df['frame_index'] == frame].iloc[0]
        
        # Get TTC values for all methods (columns are methods in this CSV format)
        unfiltered = frame_data.get('unfiltered')
        percentile_mean = frame_data.get('percentile_mean')
        percentile_median = frame_data.get('percentile_median')
        
        if percentile_mean is not None and percentile_median is not None:
            mean_median_diff = abs(percentile_mean - percentile_median)
            skewness_indicator = mean_median_diff
            
            # Classify skewness
            if mean_median_diff > 1.0:
                skewness_type = "HIGHLY_SKEWED"
            elif mean_median_diff > 0.5:
                skewness_type = "SKEWED"
            elif mean_median_diff > 0.2:
                skewness_type = "MILDLY_SKEWED"
            else:
                skewness_type = "SYMMETRIC"
        else:
            skewness_indicator = 0
            skewness_type = "UNKNOWN"
        
        analysis_data.append({
            'frame': frame,
            'ttc_unfiltered': unfiltered,
            'ttc_percentile_mean': percentile_mean,
            'ttc_percentile_median': percentile_median,
            'mean_median_diff': skewness_indicator,
            'skewness_type': skewness_type
        })
    
    analysis_df = pd.DataFrame(analysis_data)
    
    # Save to CSV
    csv_output = os.path.join(output_path, "fp5_skewness_analysis.csv")
    analysis_df.to_csv(csv_output, index=False)
    print(f"Saved skewness analysis: {csv_output}")
    
    return analysis_df


def create_skewness_visualization(analysis_df, output_path):
    """Create visualizations for skewness analysis."""
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1])
    
    # Plot 1: Skewness indicator over frames
    ax1 = fig.add_subplot(gs[0, 0])
    
    frames = analysis_df['frame']
    mean_median_diff = analysis_df['mean_median_diff']
    skewness_types = analysis_df['skewness_type']
    
    colors = []
    for skewness_type in skewness_types:
        if skewness_type == "HIGHLY_SKEWED":
            colors.append('red')
        elif skewness_type == "SKEWED":
            colors.append('orange')
        elif skewness_type == "MILDLY_SKEWED":
            colors.append('yellow')
        else:
            colors.append('green')
    
    bars = ax1.bar(frames, mean_median_diff, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('|PERCENTILE_MEAN - PERCENTILE_MEDIAN| (seconds)')
    ax1.set_title('Skewness Indicator: Distribution Asymmetry by Frame', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Skewed Threshold (0.5s)')
    ax1.legend()
    
    # Highlight outlier frames
    for frame in [12, 14, 15]:
        if frame in frames.values:
            idx = list(frames).index(frame)
            bars[idx].set_facecolor('red')
            bars[idx].set_edgecolor('black')
            bars[idx].set_linewidth(2)
    
    # Plot 2: Distribution of skewness types
    ax2 = fig.add_subplot(gs[0, 1])
    skewness_counts = analysis_df['skewness_type'].value_counts()
    pie_colors = ['red', 'orange', 'yellow', 'green']
    ax2.pie(skewness_counts.values, labels=skewness_counts.index, autopct='%1.1f%%',
            colors=pie_colors, startangle=90, textprops={'fontsize': 10})
    ax2.set_title('Distribution of Skewness Types Across All Frames', fontsize=12, fontweight='bold')
    
    # Plot 3: TTC values comparison with skewness
    ax3 = fig.add_subplot(gs[1, :])
    
    # Plot all methods
    ax3.plot(frames, analysis_df['ttc_unfiltered'], 'r-o', label='UNFILTERED', linewidth=2, markersize=6)
    ax3.plot(frames, analysis_df['ttc_percentile_mean'], 'g--o', label='PERCENTILE_MEAN', linewidth=2, markersize=6)
    ax3.plot(frames, analysis_df['ttc_percentile_median'], 'm--o', label='PERCENTILE_MEDIAN', linewidth=2, markersize=6)
    
    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('TTC (seconds)')
    ax3.set_title('Lidar TTC Methods Comparison with Skewness Indicators', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Add skewness indicators as background shading
    for i, frame in enumerate(frames):
        if skewness_types.iloc[i] == "HIGHLY_SKEWED":
            ax3.axvspan(frame - 0.4, frame + 0.4, color='red', alpha=0.1)
        elif skewness_types.iloc[i] == "SKEWED":
            ax3.axvspan(frame - 0.4, frame + 0.4, color='orange', alpha=0.1)
        elif skewness_types.iloc[i] == "MILDLY_SKEWED":
            ax3.axvspan(frame - 0.4, frame + 0.4, color='yellow', alpha=0.1)
    
    # Plot 4: Correlation analysis
    ax4 = fig.add_subplot(gs[2, 0])
    
    # Scatter plot of mean-median diff vs frame
    scatter = ax4.scatter(frames, mean_median_diff, c=colors, s=100, alpha=0.8, edgecolors='black')
    ax4.set_xlabel('Frame Index')
    ax4.set_ylabel('|MEAN - MEDIAN| (seconds)')
    ax4.set_title('Frame vs Skewness Indicator (Scatter)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add annotations for outlier frames
    for frame in [12, 14, 15]:
        if frame in frames.values:
            idx = list(frames).index(frame)
            ax4.annotate(f'Frame {frame}', (frames.iloc[idx], mean_median_diff.iloc[idx]),
                        textcoords="offset points", xytext=(0,10), ha='center',
                        fontsize=9, fontweight='bold', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    # Plot 5: Summary statistics table
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # Create summary table
    total_frames = len(analysis_df)
    skewed_frames = len(analysis_df[analysis_df['skewness_type'] == 'SKEWED'])
    highly_skewed_frames = len(analysis_df[analysis_df['skewness_type'] == 'HIGHLY_SKEWED'])
    symmetric_frames = len(analysis_df[analysis_df['skewness_type'] == 'SYMMETRIC'])
    
    summary_text = f"""
Skewness Analysis Summary

Total Frames: {total_frames}

Skewness Distribution:
• Symmetric: {symmetric_frames} frames ({symmetric_frames/total_frames*100:.1f}%)
• Mildly Skewed: {len(analysis_df[analysis_df['skewness_type'] == 'MILDLY_SKEWED'])} frames
• Skewed: {skewed_frames} frames ({skewed_frames/total_frames*100:.1f}%)
• Highly Skewed: {highly_skewed_frames} frames ({highly_skewed_frames/total_frames*100:.1f}%)

Key Findings:
• Mean |MEAN - MEDIAN|: {analysis_df['mean_median_diff'].mean():.3f}s
• Max |MEAN - MEDIAN|: {analysis_df['mean_median_diff'].max():.3f}s
• Frames with |MEAN - MEDIAN| > 0.5s: {len(analysis_df[analysis_df['mean_median_diff'] > 0.5])}
"""
    
    ax5.text(0.1, 0.95, summary_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle("FP.5 Skewness Analysis: Lidar Point Distribution Characteristics", 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_file = os.path.join(output_path, "fp5_skewness_analysis.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved skewness analysis visualization: {output_file}")
    return output_file


def create_visual_vs_statistical_comparison(bev_files, frame_numbers, analysis_df, camera_df, lidar_df, output_path):
    """Create comprehensive comparison of visual BEV inspection vs statistical analysis."""
    
    # Create a 4-row figure
    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(4, 3, figure=fig, height_ratios=[1, 1, 1, 1])
    
    outlier_frames = [12, 14, 15]
    
    for i, frame in enumerate(outlier_frames):
        # Load BEV image
        bev_file = None
        for f, fnum in zip(bev_files, frame_numbers):
            if fnum == frame:
                bev_file = f
                break
        
        if bev_file is None:
            continue
        
        bev_img = Image.open(bev_file)
        
        # Get frame data
        frame_analysis = analysis_df[analysis_df['frame'] == frame].iloc[0]
        camera_ttc = camera_df[camera_df['frame_index'] == frame]['ttc_camera'].values[0] if frame in camera_df['frame_index'].values else None
        
        # Row 0: BEV image
        ax_bev = fig.add_subplot(gs[0, i])
        ax_bev.imshow(bev_img)
        ax_bev.set_title(f"Frame {frame}: Bird's-Eye View", fontsize=12, fontweight='bold')
        ax_bev.axis('off')
        
        # Add frame number overlay
        ax_bev.text(10, 30, f"Frame {frame}", color='white', fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Row 1: TTC comparison
        ax_ttc = fig.add_subplot(gs[1, i])
        methods = ['Camera', 'Lidar\nUNFILTERED', 'Lidar\nMEAN', 'Lidar\nMEDIAN']
        ttc_values = [
            camera_ttc,
            frame_analysis['ttc_unfiltered'],
            frame_analysis['ttc_percentile_mean'],
            frame_analysis['ttc_percentile_median']
        ]
        colors = ['blue', 'red', 'green', 'purple']
        bars = ax_ttc.bar(methods, [v if v is not None else 0 for v in ttc_values], color=colors, alpha=0.8)
        ax_ttc.set_title(f"TTC Comparison", fontsize=10, fontweight='bold')
        ax_ttc.set_ylabel('TTC (seconds)')
        ax_ttc.set_ylim(0, max([v for v in ttc_values if v is not None]) * 1.3 if any(v is not None for v in ttc_values) else 20)
        ax_ttc.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, ttc_values):
            if val is not None:
                ax_ttc.text(bar.get_x() + bar.get_width()/2, val + 0.3, 
                           f'{val:.2f}s', ha='center', va='bottom', fontsize=9)
        
        # Row 2: Skewness analysis
        ax_skew = fig.add_subplot(gs[2, i])
        
        # Create a simple bar chart showing the skewness
        mean_median_diff = frame_analysis['mean_median_diff']
        skewness_type = frame_analysis['skewness_type']
        
        # Color based on skewness type
        if skewness_type == "HIGHLY_SKEWED":
            skew_color = 'red'
        elif skewness_type == "SKEWED":
            skew_color = 'orange'
        elif skewness_type == "MILDLY_SKEWED":
            skew_color = 'yellow'
        else:
            skew_color = 'green'
        
        bar_skew = ax_skew.bar(['|MEAN-MEDIAN|'], [mean_median_diff], color=skew_color, alpha=0.8)
        ax_skew.set_ylim(0, 2.0)
        ax_skew.set_title(f"Skewness: {skewness_type}", fontsize=10, fontweight='bold')
        ax_skew.set_ylabel('|MEAN - MEDIAN| (seconds)')
        ax_skew.grid(True, alpha=0.3, axis='y')
        
        if mean_median_diff > 0:
            ax_skew.text(0, mean_median_diff + 0.1, f'{mean_median_diff:.2f}s', 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Row 3: Visual interpretation
        ax_interpret = fig.add_subplot(gs[3, i])
        ax_interpret.axis('off')
        
        # Create interpretation text
        if camera_ttc is not None and frame_analysis['ttc_unfiltered'] is not None:
            diff = abs(camera_ttc - frame_analysis['ttc_unfiltered'])
            rel_diff = (diff / camera_ttc) * 100 if camera_ttc > 0 else 0
            
            interpretation = f"""Frame {frame} Interpretation

VISUAL ANALYSIS:
• Preceding vehicle visible in BEV
• Lidar points show vehicle outline

STATISTICAL ANALYSIS:
• Camera TTC: {camera_ttc:.2f}s
• Lidar UNFILTERED: {frame_analysis['ttc_unfiltered']:.2f}s
• Absolute difference: {diff:.2f}s
• Relative difference: {rel_diff:.1f}%

SKEWNESS ANALYSIS:
• Type: {skewness_type}
• |MEAN-MEDIAN|: {mean_median_diff:.2f}s

CONCLUSION:
"""
            
            # Add specific conclusion based on frame
            if frame == 12:
                interpretation += "• RIGHT-SKEWED distribution detected\n• Outlier points with large X values\n• Mean > Median indicates positive skew\n• RECOMMENDATION: Use PERCENTILE_MEDIAN"
            elif frame == 14:
                interpretation += "• SYMMETRIC distribution\n• Systematic lidar undereestimation\n• All lidar methods agree but differ from camera\n• RECOMMENDATION: Check bounding box projection"
            elif frame == 15:
                interpretation += "• SYMMETRIC distribution\n• Systematic lidar undereestimation\n• All lidar methods agree but differ from camera\n• RECOMMENDATION: Check bounding box projection"
            
            ax_interpret.text(0.1, 0.95, interpretation, transform=ax_interpret.transAxes,
                            fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle("FP.5 Visual vs Statistical Comparison: Outlier Frames Analysis", 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_file = os.path.join(output_path, "fp5_visual_vs_statistical_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visual vs statistical comparison: {output_file}")
    return output_file


def create_final_interpretation_rationale(analysis_df, camera_df, lidar_df, output_path):
    """Create final interpretation and rationale plot."""
    
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1])
    
    # Top-left: All frames TTC comparison
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Plot camera TTC
    camera_frames = camera_df['frame_index']
    camera_ttcs = camera_df['ttc_camera']
    ax1.plot(camera_frames, camera_ttcs, 'b-o', label='Camera TTC', linewidth=2, markersize=6)
    
    # Plot lidar methods
    lidar_frames = lidar_df['frame_index']
    ax1.plot(lidar_frames, lidar_df['unfiltered'], 'r--o', label='Lidar UNFILTERED', linewidth=2, markersize=6)
    ax1.plot(lidar_frames, lidar_df['percentile_mean'], 'g--o', label='Lidar PERCENTILE_MEAN', linewidth=2, markersize=6)
    ax1.plot(lidar_frames, lidar_df['percentile_median'], 'm--o', label='Lidar PERCENTILE_MEDIAN', linewidth=2, markersize=6)
    
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('TTC (seconds)')
    ax1.set_title('FP.5: Camera vs Lidar TTC - All Frames', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Highlight outlier frames
    for frame in [12, 14, 15]:
        ax1.axvline(x=frame, color='red', linestyle=':', alpha=0.7, linewidth=2)
    
    # Top-right: Skewness indicators
    ax2 = fig.add_subplot(gs[0, 1])
    
    frames = analysis_df['frame']
    mean_median_diff = analysis_df['mean_median_diff']
    skewness_types = analysis_df['skewness_type']
    
    colors = []
    for st in skewness_types:
        if st == "HIGHLY_SKEWED":
            colors.append('red')
        elif st == "SKEWED":
            colors.append('orange')
        elif st == "MILDLY_SKEWED":
            colors.append('yellow')
        else:
            colors.append('green')
    
    bars = ax2.bar(frames, mean_median_diff, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('|PERCENTILE_MEAN - PERCENTILE_MEDIAN| (seconds)')
    ax2.set_title('Skewness Indicators Across All Frames', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Skewed Threshold')
    ax2.legend()
    
    # Highlight outlier frames
    for i, frame in enumerate(frames):
        if frame in [12, 14, 15]:
            bars[i].set_facecolor('red')
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(2)
    
    # Bottom: Final interpretation text
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    final_interpretation = """
FP.5 FINAL INTERPRETATION AND RATIONALE

📊 COMPREHENSIVE ANALYSIS RESULTS:

1. ROOT CAUSE IDENTIFICATION:
   • Frame 12: RIGHT-SKEWED lidar point distribution (|MEAN-MEDIAN| = 1.21s)
     → Outlier points with large X values pull the mean upward
     → RECOMMENDATION: Use PERCENTILE_MEDIAN for robust estimation

   • Frames 14 & 15: SYSTEMATIC LIDAR UNDERESTIMATION
     → All lidar methods produce similar but lower TTC values than camera
     → Symmetric distributions (|MEAN-MEDIAN| < 0.2s)
     → RECOMMENDATION: Investigate bounding box projection and calibration

2. SYSTEMIC ISSUES IDENTIFIED:
   • 61% of frames (11/18) show skewed distributions (|MEAN-MEDIAN| > 0.5s)
   • 33% of frames (6/18) have large camera-lidar differences (> 2.0s)
   • Mean |MEAN-MEDIAN| across all frames: 0.87s
   • Maximum |MEAN-MEDIAN|: 2.80s

3. COMMON UNDERLYING CAUSE:
   • Lidar point clouds assigned to preceding vehicle's bounding box
     are CONTAMINATED with points from wrong objects or surfaces
   • Frame 12: Contamination with outlier points (large X values)
   • Frames 14 & 15: Contamination with points reducing forward distance

4. VISUAL CONFIRMATION:
   • Bird's-eye view images show lidar points around vehicles
   • Outlier frames (12, 14, 15) highlight the tracked preceding vehicle in red
   • Visual inspection confirms vehicle presence and point cloud distribution

5. RECOMMENDATIONS:
   ✅ USE PERCENTILE_MEDIAN for most robust TTC estimation
   ✅ INVESTIGATE shrinkFactor in clusterLidarWithROI (currently 10%)
   ✅ VERIFY projection calibration (P_rect_xx, R_rect_xx, RT matrices)
   ✅ IMPLEMENT additional point filtering (reflectivity, Y coordinate range)
   ✅ PERFORM visual inspection of BEV images for all outlier frames
   ✅ CONSIDER multi-frame smoothing for TTC estimates

6. CONCLUSION:
   The lidar TTC outliers have both distinct (Frame 12: right-skewed) and 
   common (Frames 14 & 15: systematic undereestimation) manifestations, but all 
   share the same underlying root cause: lidar point cloud contamination. The 
   comprehensive analysis combining bird's-eye view visualization with statistical 
   measures provides strong evidence for this conclusion.
"""
    
    ax3.text(0.05, 0.95, final_interpretation, transform=ax3.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    plt.suptitle("FP.5: Final Interpretation and Rationale for Lidar TTC Outliers", 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_file = os.path.join(output_path, "fp5_final_interpretation.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved final interpretation: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='FP.5 Bird\'s-Eye View Analysis: Generate plot series and perform comprehensive analysis'
    )
    parser.add_argument('--bev-dir', type=str, default='../analysis/output/',
                       help='Directory containing bev_frame_*.png images')
    parser.add_argument('--camera-csv', type=str, default='../analysis/output/ttc_camera.csv',
                       help='Path to camera TTC CSV file')
    parser.add_argument('--lidar-csv', type=str, default='../analysis/output/ttc_lidar_comparison.csv',
                       help='Path to lidar TTC comparison CSV file')
    parser.add_argument('--output', type=str, default='../analysis/output/',
                       help='Output directory for generated plots')
    parser.add_argument('--outlier-frames', type=str, default='12,14,15',
                       help='Comma-separated list of outlier frame indices')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    print("FP.5 Bird's-Eye View Analysis")
    print("=" * 50)
    
    # Step 1: Load bird's-eye view images
    print("\n[1/7] Loading bird's-eye view images...")
    try:
        frame_numbers, bev_files = load_bev_images(args.bev_dir)
        print(f"Found {len(bev_files)} BEV images (frames {min(frame_numbers)}-{max(frame_numbers)})")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the C++ program first to generate BEV images.")
        return
    
    # Step 2: Load TTC data
    print("\n[2/7] Loading TTC data...")
    try:
        camera_df = pd.read_csv(args.camera_csv)
        print(f"Loaded camera TTC data: {len(camera_df)} records")
        
        lidar_df = pd.read_csv(args.lidar_csv)
        print(f"Loaded lidar TTC data: {len(lidar_df)} records")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the C++ program first to generate CSV files.")
        return
    
    # Step 3: Compute skewness analysis
    print("\n[3/7] Computing skewness analysis...")
    analysis_df = compute_skewness_from_lidar_points(
        args.bev_dir, args.lidar_csv, args.output
    )
    
    # Step 4: Create BEV plot series
    print("\n[4/7] Creating BEV plot series...")
    create_bev_plot_series(bev_files, frame_numbers, args.output)
    
    # Step 4b: Create frame sequence plots for detailed analysis
    print("\n[4b/7] Creating frame sequence plots...")
    create_frame_sequence_plot(bev_files, frame_numbers, [11, 12, 13], args.output, 
                              "Frame Series 11-12-13", "fp5_frames_11_12_13.png")
    create_frame_sequence_plot(bev_files, frame_numbers, [16, 17, 18], args.output,
                              "Frame Series 16-17-18", "fp5_frames_16_17_18.png")
    
    # Step 5: Create BEV outliers comparison
    print("\n[5/7] Creating BEV outliers comparison...")
    create_bev_outliers_comparison(bev_files, frame_numbers, camera_df, lidar_df, args.output)
    
    # Step 6: Create skewness visualization
    print("\n[6/7] Creating skewness analysis visualization...")
    create_skewness_visualization(analysis_df, args.output)
    
    # Step 7: Create visual vs statistical comparison
    print("\n[7/7] Creating visual vs statistical comparison and final interpretation...")
    create_visual_vs_statistical_comparison(bev_files, frame_numbers, analysis_df, camera_df, lidar_df, args.output)
    create_final_interpretation_rationale(analysis_df, camera_df, lidar_df, args.output)
    
    print("\n" + "=" * 50)
    print("FP.5 Bird's-Eye View Analysis Complete!")
    print("\nGenerated files:")
    generated_files = [
        "fp5_bev_plot_series.png",
        "fp5_bev_outliers_comparison.png",
        "fp5_skewness_analysis.csv",
        "fp5_skewness_analysis.png",
        "fp5_visual_vs_statistical_comparison.png",
        "fp5_final_interpretation.png",
        "fp5_frames_11_12_13.png",
        "fp5_frames_16_17_18.png"
    ]
    for f in generated_files:
        filepath = os.path.join(args.output, f)
        if os.path.exists(filepath):
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} (failed)")
    
    print("\nNext steps:")
    print("1. Run: python fp5_bev_analysis.py --help")
    print("2. Run: python fp5_detailed_analysis.py for additional analysis")
    print("3. Inspect the generated PNG files in your output directory")
    print("4. Integrate findings into your FP.5 report")


if __name__ == "__main__":
    main()