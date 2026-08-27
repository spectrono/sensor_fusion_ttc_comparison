#!/usr/bin/env python3
"""
FP.1 Analysis Script

Plots the number of matches over frames for the current detector/descriptor combination
on the preceding vehicle on the ego lane using seaborn.

This script reads CSV data exported from the 3D_object_tracking executable and generates
visualizations of bounding box matching performance over the image sequence.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os


def plot_matches_over_frames(csv_path, output_dir=".", detector="SHITOMASI", descriptor="ORB", show=True):
    """
    Plot the number of matches over frames for the preceding vehicle.
    
    Args:
        csv_path: Path to the CSV file containing match data
        output_dir: Directory to save the plot
        detector: Detector type (for plot title)
        descriptor: Descriptor type (for plot title)
        show: Whether to display the plot interactively (default: True)
    """
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        print("Please run the 3D_object_tracking executable first to generate match data.")
        return
    
    # Filter for preceding vehicle (boxID 0 is typically the preceding vehicle on ego lane)
    # The CSV should contain: frame_index, prev_box_id, curr_box_id, match_count
    preceding_vehicle_matches = df[df['prev_box_id'] == 0]
    
    if preceding_vehicle_matches.empty:
        print("No matches found for preceding vehicle (boxID 0).")
        print("Available box IDs:", df['prev_box_id'].unique())
        # Use all data if boxID 0 not found
        preceding_vehicle_matches = df
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Plot matches over frames
    ax = sns.lineplot(
        data=preceding_vehicle_matches,
        x='frame_index',
        y='match_count',
        marker='o',
        markersize=8,
        linewidth=2.5,
        color='#1f77b4'
    )
    
    # Customize the plot
    plt.title(
        f"FP.1: Bounding Box Matches Over Frames\n"
        f"Detector: {detector}, Descriptor: {descriptor}",
        fontsize=14,
        pad=20
    )
    plt.xlabel("Frame Index", fontsize=12)
    plt.ylabel("Number of Keypoint Matches", fontsize=12)
    plt.xticks(preceding_vehicle_matches['frame_index'].unique())
    plt.grid(True, alpha=0.3)
    
    # Add value labels on points
    for line in ax.lines:
        for x, y in zip(line.get_xdata(), line.get_ydata()):
            ax.annotate(
                f'{int(y)}',
                xy=(x, y),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=9
            )
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"fp1_matches_{detector}_{descriptor}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Show the plot
    if show:
        plt.tight_layout()
        plt.show()
    else:
        plt.close()
    
    # Return the data for potential further analysis
    return preceding_vehicle_matches


def plot_all_boxes_matches(csv_path, output_dir=".", detector="SHITOMASI", descriptor="ORB"):
    """
    Plot matches for all bounding boxes over frames as separate lines.
    
    Args:
        csv_path: Path to the CSV file containing match data
        output_dir: Directory to save the plot
        detector: Detector type (for plot title)
        descriptor: Descriptor type (for plot title)
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Plot each box as a separate line
    ax = sns.lineplot(
        data=df,
        x='frame_index',
        y='match_count',
        hue='prev_box_id',
        style='prev_box_id',
        markers=True,
        dashes=False,
        palette='viridis'
    )
    
    plt.title(
        f"FP.1: Matches for All Bounding Boxes Over Frames\n"
        f"Detector: {detector}, Descriptor: {descriptor}",
        fontsize=14,
        pad=20
    )
    plt.xlabel("Frame Index", fontsize=12)
    plt.ylabel("Number of Keypoint Matches", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"fp1_all_boxes_{detector}_{descriptor}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"All boxes plot saved to: {output_path}")
    
    plt.tight_layout()
    plt.show()


def print_summary_statistics(df):
    """Print summary statistics for the match data."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total frames: {df['frame_index'].nunique()}")
    print(f"Total matches recorded: {len(df)}")
    print(f"Unique bounding box pairs: {df[['prev_box_id', 'curr_box_id']].nunique()}")
    print("\nPer-frame statistics:")
    print(df.groupby('frame_index')['match_count'].describe())
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="FP.1 Analysis: Plot bounding box matches over frames"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="output/bb_matches.csv",
        help="Path to CSV file containing match data (default: output/bb_matches.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for plots (default: output)"
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="SHITOMASI",
        help="Detector type for plot title (default: SHITOMASI)"
    )
    parser.add_argument(
        "--descriptor",
        type=str,
        default="ORB",
        help="Descriptor type for plot title (default: ORB)"
    )
    parser.add_argument(
        "--all-boxes",
        action="store_true",
        help="Also plot matches for all bounding boxes"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable interactive plot display"
    )
    
    args = parser.parse_args()
    
    # Plot matches for preceding vehicle
    df = plot_matches_over_frames(
        args.csv, 
        args.output, 
        args.detector, 
        args.descriptor,
        show=args.show if hasattr(args, 'show') else not args.no_show
    )
    
    if df is not None:
        # Print summary statistics
        print_summary_statistics(df)
        
        # Optionally plot all boxes
        if args.all_boxes:
            plot_all_boxes_matches(
                args.csv,
                args.output,
                args.detector,
                args.descriptor
            )


if __name__ == "__main__":
    main()
