#!/usr/bin/env python3
"""
Create Video from Keypoint Matches Images

This script creates a video from the keypoint_matches_*.png images for use on a website.
Supports MP4 and WebM formats (best for web compatibility).

Requirements:
    - ffmpeg (for MP4 and WebM)

Usage:
    python create_keypoint_video.py [--format mp4|webm|all] [--fps FPS] [--output OUTPUT_DIR]
    
    Examples:
        python create_keypoint_video.py --format mp4
        python create_keypoint_video.py --format all --fps 5
        python create_keypoint_video.py --format webm --output ../videos
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def extract_frame_number(path):
    """Extract frame number from a keypoint_matches_*.png filename."""
    name = path.stem
    # Extract number from keypoint_matches_1.png -> 1
    if name.startswith("keypoint_matches_"):
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return 0
    return 0


def get_sorted_image_paths(image_dir="output", pattern="keypoint_matches_*.png"):
    """Get sorted list of keypoint match image paths."""
    image_dir_path = Path(image_dir)
    
    # If the path doesn't exist, try to resolve it relative to the current working directory
    if not image_dir_path.exists():
        # Try absolute path construction
        cwd = Path.cwd()
        possible_path = cwd / image_dir
        if possible_path.exists():
            image_dir_path = possible_path
        else:
            raise FileNotFoundError(f"Image directory not found: {image_dir} (tried: {possible_path})")
    
    # Make it absolute
    image_dir_path = image_dir_path.absolute()
    
    image_paths = list(image_dir_path.glob(pattern))
    
    if not image_paths:
        raise FileNotFoundError(f"No images found matching pattern: {pattern} in {image_dir_path}")
    
    # Sort numerically by frame number
    image_paths = sorted(image_paths, key=extract_frame_number)
    
    return image_paths


def create_mp4_video(image_paths, output_path, fps=3, crf=23):
    """Create MP4 video using ffmpeg."""
    print(f"Creating MP4 video: {output_path}")
    
    # Use image sequence demuxer with %d pattern
    # Images must be named like keypoint_matches_1.png, keypoint_matches_2.png, etc.
    image_dir = image_paths[0].parent.absolute()
    image_pattern = str(image_dir / "keypoint_matches_%d.png")
    start_number = extract_frame_number(image_paths[0])
    
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite without asking
        '-framerate', str(fps),
        '-start_number', str(start_number),
        '-i', image_pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', str(crf),
        '-preset', 'slow',
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # Ensure even dimensions for libx264
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  ✓ MP4 created successfully")
        print(f"  Output: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ MP4 creation failed:")
        print(f"  {e.stderr}")
        return False


def create_webm_video(image_paths, output_path, fps=3, crf=30):
    """Create WebM video using ffmpeg."""
    print(f"Creating WebM video: {output_path}")
    
    # Use image sequence demuxer with %d pattern
    image_dir = image_paths[0].parent.absolute()
    image_pattern = str(image_dir / "keypoint_matches_%d.png")
    start_number = extract_frame_number(image_paths[0])
    
    cmd = [
        'ffmpeg',
        '-y',
        '-framerate', str(fps),
        '-start_number', str(start_number),
        '-i', image_pattern,
        '-c:v', 'libvpx-vp9',
        '-b:v', '0',
        '-crf', str(crf),
        '-row-mt', '1',
        '-cpu-used', '4',
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # Ensure even dimensions
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  ✓ WebM created successfully")
        print(f"  Output: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ WebM creation failed:")
        print(f"  {e.stderr}")
        return False


def create_gif_video(image_paths, output_path, fps=10):
    """Create animated GIF using ffmpeg."""
    print(f"Creating GIF: {output_path}")
    
    # Create a text file with the list of images
    list_file = output_path.parent / "image_list_gif.txt"
    with open(list_file, 'w') as f:
        for path in image_paths:
            f.write(f"file '{path}'\n")
    
    cmd = [
        'ffmpeg',
        '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(list_file),
        '-vf', f'fps={fps},scale=trunc(iw/2)*2:trunc(ih/2)*2',
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  ✓ GIF created successfully")
        print(f"  Output: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ GIF creation failed:")
        print(f"  {e.stderr}")
        return False
    finally:
        if list_file.exists():
            list_file.unlink()



def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Create video from keypoint_matches_*.png images for website'
    )
    parser.add_argument('--format', type=str, default='mp4',
                       choices=['mp4', 'webm', 'all'],
                       help='Video format: mp4, webm, or all')
    parser.add_argument('--fps', type=int, default=3,
                       help='Frames per second (default: 3)')
    parser.add_argument('--image-dir', type=str, default='output',
                       help='Directory containing keypoint_matches_*.png images')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for videos (default: output)')
    parser.add_argument('--crf', type=int, default=23,
                       help='CRF value for MP4/WebM (lower = better quality, default: 23)')
    parser.add_argument('--prefix', type=str, default='keypoint_matches',
                       help='Output filename prefix (default: keypoint_matches)')
    
    args = parser.parse_args()
    
    # Check ffmpeg
    if not check_ffmpeg():
        print("Error: ffmpeg is required but not found.")
        print("Install ffmpeg first:")
        print("  macOS: brew install ffmpeg")
        print("  Ubuntu: sudo apt install ffmpeg")
        sys.exit(1)
    
    # Get image paths
    print(f"\nLooking for images in: {args.image_dir}")
    try:
        image_paths = get_sorted_image_paths(args.image_dir)
        print(f"Found {len(image_paths)} images: {image_paths[0].name} to {image_paths[-1].name}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Create output directory (resolve relative to cwd)
    output_dir = Path(args.output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create videos based on format
    print(f"\nCreating videos with {args.fps} fps...")
    
    formats_to_create = []
    if args.format == 'all':
        formats_to_create = ['mp4', 'webm']
    else:
        formats_to_create = [args.format]
    
    created_files = []
    
    for fmt in formats_to_create:
        print(f"\n--- Creating {fmt.upper()} ---")
        
        if fmt == 'mp4':
            output_path = output_dir / f"{args.prefix}_video.mp4"
            if create_mp4_video(image_paths, output_path, args.fps, args.crf):
                created_files.append(output_path)
        
        elif fmt == 'webm':
            output_path = output_dir / f"{args.prefix}_video.webm"
            if create_webm_video(image_paths, output_path, args.fps, args.crf):
                created_files.append(output_path)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Input: {len(image_paths)} images from {args.image_dir}")
    print(f"FPS: {args.fps}")
    print(f"\nCreated files:")
    for f in created_files:
        if f.exists():
            size = f.stat().st_size
            size_str = f"{size/1024/1024:.2f} MB" if size > 1024*1024 else f"{size/1024:.1f} KB"
            print(f"  ✓ {f.name} ({size_str})")
        else:
            print(f"  ✗ {f.name} (failed)")
    
    print(f"\nUse these files for your website:")
    print(f"  <video controls autoplay loop>")
    print(f"    <source src=\"{args.prefix}_video.mp4\" type=\"video/mp4\">")
    print(f"    <source src=\"{args.prefix}_video.webm\" type=\"video/webm\">")
    print(f"    Your browser does not support the video tag.")
    print(f"  </video>")
    
    print(f"\nFor GIF (simpler but larger):")
    print(f"  <img src=\"{args.prefix}_video.gif\" alt=\"Keypoint Matches Animation\">")


if __name__ == '__main__':
    main()
