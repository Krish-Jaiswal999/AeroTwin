"""
AeroTwin — Script 01: Frame Extraction from Drone Video
========================================================
Extracts frames from a drone video, applies blur detection,
and saves only usable frames for 3D reconstruction.

Usage:
    python scripts/01_extract_frames.py --input data/input/drone_video.mp4 --project my_project

Output:
    data/output/<project>/frames/          — extracted frame images
    data/output/<project>/frame_report.txt — extraction statistics
"""

import argparse
import os
import shutil
import sys
import time
import json
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
from tqdm import tqdm
import yaml


def compute_blur_score(image_gray):
    """
    Compute image sharpness using Laplacian variance.
    Higher value = sharper image.
    
    This is a well-established method:
    Pech-Pacheco et al., "Diatom autofocusing in brightfield microscopy"
    """
    laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
    return laplacian.var()


def extract_frames(video_path, output_dir, config):
    """
    Extract frames from video with blur filtering.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        config: Dictionary with extraction parameters
    
    Returns:
        Dictionary with extraction statistics
    """
    frame_interval = config.get("frame_interval", 10)
    blur_threshold = config.get("blur_threshold", 100.0)
    max_frames = config.get("max_frames", 0)
    jpeg_quality = config.get("jpeg_quality", 95)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"{'='*60}")
    print(f"  AeroTwin — Frame Extraction")
    print(f"{'='*60}")
    print(f"  Video:       {video_path.name}")
    print(f"  Resolution:  {width}x{height}")
    print(f"  FPS:         {fps:.2f}")
    print(f"  Total frames:{total_frames}")
    print(f"  Duration:    {duration:.1f}s ({duration/60:.1f}min)")
    print(f"  Interval:    every {frame_interval}th frame")
    print(f"  Blur thresh: {blur_threshold}")
    print(f"{'='*60}")
    
    # Create a clean output directory so reruns do not retain stale frame files
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    stats = {
        "video_file": str(video_path),
        "video_resolution": f"{width}x{height}",
        "video_fps": fps,
        "total_video_frames": total_frames,
        "video_duration_sec": round(duration, 2),
        "frame_interval": frame_interval,
        "blur_threshold": blur_threshold,
        "frames_sampled": 0,
        "frames_blurry": 0,
        "frames_kept": 0,
        "blur_scores": [],
        "kept_frame_indices": [],
    }
    
    # Calculate frames to sample
    sample_indices = list(range(0, total_frames, frame_interval))
    if max_frames > 0 and len(sample_indices) > max_frames:
        # Evenly subsample
        step = len(sample_indices) / max_frames
        sample_indices = [sample_indices[int(i * step)] for i in range(max_frames)]
    
    stats["frames_sampled"] = len(sample_indices)
    kept_count = 0
    blurry_count = 0
    
    print(f"\n  Sampling {len(sample_indices)} frames...")
    print(f"  Applying blur detection (Laplacian variance >= {blur_threshold})")
    print()
    
    start_time = time.time()
    
    for idx in tqdm(sample_indices, desc="  Extracting", unit="frame"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Convert to grayscale for blur detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = compute_blur_score(gray)
        
        stats["blur_scores"].append({
            "frame_index": int(idx),
            "blur_score": float(round(blur_score, 2)),
            "kept": bool(blur_score >= blur_threshold)
        })
        
        if blur_score < blur_threshold:
            blurry_count += 1
            continue
        
        # Save frame
        frame_filename = f"frame_{kept_count:05d}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(
            frame_path,
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        )
        
        stats["kept_frame_indices"].append(idx)
        kept_count += 1
    
    cap.release()
    elapsed = time.time() - start_time
    
    stats["frames_blurry"] = blurry_count
    stats["frames_kept"] = kept_count
    stats["processing_time_sec"] = round(elapsed, 2)
    
    # Print report
    print(f"\n{'='*60}")
    print(f"  EXTRACTION REPORT")
    print(f"{'='*60}")
    print(f"  Total video frames:    {total_frames}")
    print(f"  Frames sampled:        {stats['frames_sampled']}")
    print(f"  Rejected (blurry):     {blurry_count}")
    print(f"  Frames kept:           {kept_count}")
    print(f"  Keep rate:             {kept_count/max(stats['frames_sampled'],1)*100:.1f}%")
    print(f"  Processing time:       {elapsed:.1f}s")
    print(f"  Output directory:      {output_dir}")
    print(f"{'='*60}")
    
    if kept_count < 10:
        print(f"\n  ⚠️  WARNING: Only {kept_count} frames kept.")
        print(f"  3D reconstruction typically needs ≥20 frames.")
        print(f"  Consider lowering blur_threshold or frame_interval.")
    
    if kept_count == 0:
        print(f"\n  ❌ ERROR: No frames passed blur filter!")
        print(f"  Try: --blur-threshold 50")
        sys.exit(1)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="AeroTwin: Extract frames from drone video for 3D reconstruction"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input drone video file"
    )
    parser.add_argument(
        "--project", "-p",
        default="default",
        help="Project name (used for output directory)"
    )
    parser.add_argument(
        "--interval", "-n",
        type=int,
        default=None,
        help="Extract every Nth frame (overrides config)"
    )
    parser.add_argument(
        "--blur-threshold", "-b",
        type=float,
        default=None,
        help="Minimum blur score to keep frame (overrides config)"
    )
    parser.add_argument(
        "--max-frames", "-m",
        type=int,
        default=None,
        help="Maximum number of frames to extract"
    )
    parser.add_argument(
        "--config", "-c",
        default="config/colmap_defaults.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output-base",
        default="data/output",
        help="Base output directory (default: data/output)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            full_config = yaml.safe_load(f)
        config = full_config.get("frame_extraction", {})
    else:
        print(f"  Config not found at {config_path}, using defaults.")
        config = {}
    
    # Override with CLI args
    if args.interval is not None:
        config["frame_interval"] = args.interval
    if args.blur_threshold is not None:
        config["blur_threshold"] = args.blur_threshold
    if args.max_frames is not None:
        config["max_frames"] = args.max_frames
    
    # Paths
    video_path = Path(args.input)
    if not video_path.exists():
        print(f"ERROR: Video file not found: {video_path}")
        sys.exit(1)
    
    project_dir = Path(args.output_base) / args.project
    frames_dir = project_dir / "frames"
    
    # Run extraction
    stats = extract_frames(video_path, str(frames_dir), config)
    
    # Save report
    report_path = project_dir / "frame_report.json"
    with open(report_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()
