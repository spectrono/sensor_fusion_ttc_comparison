#!/usr/bin/env python3
"""
FP.5 Performance Assessment 1 - BEV Plot Series Generator

This script generates a 6x3 thumbnail grid of bird's-eye view (BEV) images
for the FP.5 analysis. This is the only remaining functionality after 
consolidating all other FP.5 analyses into fp5_analysis_combined.py.

The generated plot is referenced in the main README.md at line 614.

Usage:
    python fp5_analysis.py
"""

import os
import glob
import re
import matplotlib.pyplot as plt
from PIL import Image


def load_bev_images(bev_dir='output'):
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
    
    return list(frame_numbers), list(bev_files)


def create_bev_plot_series(bev_files, frame_numbers, output_path='output'):
    """Create a plot series showing all BEV frames in a 6x3 grid."""
    n_frames = len(bev_files)
    
    # Use 6 rows x 3 columns for thumbnail grid (fits 18 frames)
    cols = 3
    rows = 6
    max_frames_to_show = rows * cols  # 18 frames
    
    # Limit to first 18 frames for 6x3 grid
    if n_frames > max_frames_to_show:
        bev_files = bev_files[:max_frames_to_show]
        frame_numbers = frame_numbers[:max_frames_to_show]
        n_frames = max_frames_to_show
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 20))
    
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
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "fp5_bev_plot_series.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved BEV plot series: {output_file}")
    return output_file


def main():
    """Main function to generate the BEV plot series."""
    print("FP.5 BEV Plot Series Generator")
    print("=" * 40)
    
    # Load BEV images
    print("Loading bird's-eye view images...")
    try:
        frame_numbers, bev_files = load_bev_images()
        print(f"Found {len(bev_files)} BEV images (frames {min(frame_numbers)}-{max(frame_numbers)})")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the C++ program first to generate BEV images.")
        return
    
    # Create BEV plot series
    print("Creating 6x3 BEV plot series...")
    output_file = create_bev_plot_series(bev_files, frame_numbers)
    
    print(f"\nDone! BEV plot series saved to: {output_file}")


if __name__ == '__main__':
    main()