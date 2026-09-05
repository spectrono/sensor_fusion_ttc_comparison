#!/usr/bin/env python3
"""
FP.3 Match Filtering Progression Analysis Script

Generates a comparison plot showing match counts across FP.1, FP.3, and FP.4 pipeline stages.
- FP.1: Raw matches (only bounding box containment filtering)
- FP.3: After Euclidean distance filtering  
- FP.4: Final matches used for camera TTC estimation

Usage:
    python fp3_match_filtering_progression.py [--fp1-csv output/bb_matches.csv] 
                                        [--fp3-csv output/kpt_matches_filtering.csv] 
                                        [--output output] [--show]
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os


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


def load_and_prepare_data(fp1_csv, fp3_csv, fp4_csv=None):
    """Load and prepare data from FP.1, FP.3, and FP.4 CSV files."""
    # Load CSV files
    df_fp1 = pd.read_csv(fp1_csv)
    df_fp3 = pd.read_csv(fp3_csv)
    
    # Try to load FP.4 progression data if provided
    if fp4_csv and os.path.exists(fp4_csv):
        df_fp4 = pd.read_csv(fp4_csv)
    else:
        df_fp4 = None
    
    # Get tracked vehicle track ID
    tracked_track_id = get_tracked_vehicle_track_id()
    
    if tracked_track_id is None:
        # Fallback: use most common track_id or track_id=1
        if 'track_id' in df_fp1.columns:
            tracked_track_id = df_fp1['track_id'].mode()[0] if not df_fp1['track_id'].mode().empty else 1
        else:
            tracked_track_id = 1
    
    # Filter for tracked vehicle
    fp1_tracked = df_fp1[df_fp1['track_id'] == tracked_track_id].copy()
    fp3_tracked = df_fp3[df_fp3['track_id'] == tracked_track_id].copy()
    
    if df_fp4 is not None:
        fp4_tracked = df_fp4[df_fp4['track_id'] == tracked_track_id].copy()
    else:
        fp4_tracked = None
    
    # Merge data on frame_index
    # FP.1: match_count (raw matches, only bounding box filtering)
    # FP.3: matches_before (raw), matches_after (displacement filtered)
    merged = pd.merge(
        fp1_tracked[['frame_index', 'match_count']], 
        fp3_tracked[['frame_index', 'matches_before', 'matches_after']], 
        on='frame_index', 
        how='inner'
    )
    
    # For FP.4, try to use actual FP.4 data if available, otherwise use FP.3 matches_after
    if df_fp4 is not None and not fp4_tracked.empty:
        # Merge with FP.4 data
        fp4_data = fp4_tracked[['frame_index', 'fp4_camera_ttc_count']].copy()
        fp4_data.columns = ['frame_index', 'fp4_matches']
        merged = pd.merge(merged, fp4_data, on='frame_index', how='left')
    else:
        # Use FP.3 matches_after as approximation for FP.4
        merged['fp4_matches'] = merged['matches_after']
    
    # Sort by frame_index
    merged = merged.sort_values('frame_index')
    
    return merged, tracked_track_id


def plot_filtering_progression(merged, tracked_track_id, output_dir=".", output_prefix="fp3_match_filtering_progression"):
    """Generate the filtering progression comparison plot."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(14, 8))
    
    frames = merged['frame_index']
    fp1_counts = merged['match_count']
    fp3_counts = merged['matches_after']  
    fp4_counts = merged['fp4_matches']
    
    # Plot FP.1 raw matches (blue)
    plt.plot(frames, fp1_counts, 'o-', label='FP.1: Raw matches (box containment only)', 
             color='blue', linewidth=2, markersize=8, alpha=0.9)
    
    # Plot FP.3 displacement-filtered matches (orange)  
    plt.plot(frames, fp3_counts, 's-', label='FP.3: After displacement filtering', 
             color='orange', linewidth=2, markersize=8, alpha=0.9)
    
    # Plot FP.4 final matches (green) - used for camera TTC
    plt.plot(frames, fp4_counts, '^-', label='FP.4: Final for camera TTC', 
             color='green', linewidth=2, markersize=8, alpha=0.9)
    
    # Add fill between FP.1 and FP.3 to show reduction
    plt.fill_between(frames, fp3_counts, fp1_counts, 
                     color='red', alpha=0.15, label='Matches removed by displacement threshold')
    
    # Customize plot
    plt.title('FP.4: Match Filtering Progression Across Pipeline Stages', fontsize=16, pad=20)
    plt.xlabel('Frame Index', fontsize=14)
    plt.ylabel('Number of Keypoint Matches', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=12)
    
    # Set integer ticks on x-axis
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Add value labels on points for first and last frame
    for i, (frame, count) in enumerate(zip(frames, fp1_counts)):
        if i == 0 or i == len(frames) - 1:
            plt.text(frame, count + 1, f'{int(count)}', ha='center', va='bottom', fontsize=10, color='blue')
    
    for i, (frame, count) in enumerate(zip(frames, fp3_counts)):
        if i == 0 or i == len(frames) - 1:
            plt.text(frame, count - 3, f'{int(count)}', ha='center', va='top', fontsize=10, color='orange')
    
    # Save plot
    output_path = os.path.join(output_dir, f"{output_prefix}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved filtering progression plot to: {output_path}")
    return output_path


def print_statistics(merged, tracked_track_id):
    """Print statistics for the filtering progression."""
    print("\n" + "="*80)
    print("FP.3 MATCH FILTERING PROGRESSION STATISTICS")
    print("="*80)
    
    fp1_counts = merged['match_count']
    fp3_counts = merged['matches_after']
    fp4_counts = merged['fp4_matches']
    
    print(f"\nTracked Vehicle: track_id={tracked_track_id}")
    print(f"Total frames: {len(merged)}")
    
    print(f"\nFP.1 (Raw matches - box containment only):")
    print(f"  Min: {fp1_counts.min():.0f}, Max: {fp1_counts.max():.0f}, Mean: {fp1_counts.mean():.1f}")
    
    print(f"\nFP.3 (After displacement filtering):")
    print(f"  Min: {fp3_counts.min():.0f}, Max: {fp3_counts.max():.0f}, Mean: {fp3_counts.mean():.1f}")
    
    print(f"\nFP.4 (Final for camera TTC):")
    print(f"  Min: {fp4_counts.min():.0f}, Max: {fp4_counts.max():.0f}, Mean: {fp4_counts.mean():.1f}")
    
    print(f"\nReduction Statistics:")
    reduction = fp1_counts - fp3_counts
    print(f"  Total matches removed by displacement threshold: {reduction.sum():.0f}")
    print(f"  Mean reduction per frame: {reduction.mean():.1f}")
    print(f"  Percentage reduction: {(reduction.sum() / fp1_counts.sum() * 100):.1f}%")
    
    # Check if FP.4 differs from FP.3
    fp4_diff = fp3_counts - fp4_counts
    print(f"  FP.4 vs FP.3 difference: {fp4_diff.sum():.0f} matches")
    
    # Check if FP.4 is same as FP.3
    fp4_same_as_fp3 = (fp4_counts == fp3_counts).all()
    print(f"\nFP.4 same as FP.3: {fp4_same_as_fp3}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='FP.3 Match Filtering Progression Analysis - Compare match counts across FP.1, FP.3, and FP.4'
    )
    parser.add_argument('--fp1-csv', type=str, 
                       default='output/bb_matches.csv',
                       help='Path to FP.1 BB matches CSV (default: output/bb_matches.csv)')
    parser.add_argument('--fp3-csv', type=str,
                       default='output/kpt_matches_filtering.csv',
                       help='Path to FP.3 keypoint displacement filtering CSV (default: output/kpt_matches_filtering.csv)')
    parser.add_argument('--fp4-csv', type=str,
                       default='output/filtering_progression.csv',
                       help='Path to FP.4 filtering progression CSV (default: output/filtering_progression.csv)')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory for plots (default: output)')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively')
    
    args = parser.parse_args()
    
    print("FP.3 Match Filtering Progression Analysis")
    print("="*50)
    
    # Load and prepare data
    merged, tracked_track_id = load_and_prepare_data(args.fp1_csv, args.fp3_csv, args.fp4_csv)
    
    if len(merged) == 0:
        print("Error: No matching data found for tracked vehicle")
        return
    
    print(f"Loaded {len(merged)} frames of data for track_id={tracked_track_id}")
    
    # Print statistics
    print_statistics(merged, tracked_track_id)
    
    # Generate plot
    plot_path = plot_filtering_progression(merged, tracked_track_id, args.output)
    
    # Show plot if requested
    if args.show:
        plt.figure(figsize=(14, 8))
        
        frames = merged['frame_index']
        fp1_counts = merged['match_count']
        fp3_counts = merged['matches_after']
        fp4_counts = merged['fp4_matches']
        
        plt.plot(frames, fp1_counts, 'o-', label='FP.1: Raw matches (box containment only)', 
                 color='blue', linewidth=2, markersize=8)
        plt.plot(frames, fp3_counts, 's-', label='FP.3: After distance filtering', 
                 color='orange', linewidth=2, markersize=8)
        plt.plot(frames, fp4_counts, '^-', label='FP.4: Final for camera TTC', 
                 color='green', linewidth=2, markersize=8)
        plt.fill_between(frames, fp3_counts, fp1_counts, color='red', alpha=0.15, 
                        label='Matches removed by distance filtering')
        
        plt.title('FP.3: Match Filtering Progression Across Pipeline Stages', fontsize=16, pad=20)
        plt.xlabel('Frame Index', fontsize=14)
        plt.ylabel('Number of Keypoint Matches', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=12)
        
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        plt.tight_layout()
        plt.show()
    
    print(f"\nAnalysis complete!")


if __name__ == '__main__':
    main()