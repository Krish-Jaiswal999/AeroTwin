"""
AeroTwin — Script 02: COLMAP 3D Reconstruction
===============================================
Runs COLMAP SfM + MVS pipeline on extracted frames to produce
a 3D point cloud and optionally a dense reconstruction.

Prerequisites:
    - COLMAP installed and accessible (in PATH or tools/ directory)
    - Frames extracted by 01_extract_frames.py

Usage:
    python scripts/02_run_colmap.py --project my_project

Output:
    data/output/<project>/sparse/   — sparse reconstruction (cameras + points)
    data/output/<project>/dense/    — dense point cloud (if enabled)
    data/output/<project>/colmap_report.json — reconstruction statistics
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import yaml


def find_colmap():
    """Find COLMAP executable. Prioritize native x64 colmap.exe."""
    # 1. Check tools/bin/colmap.exe (direct binary — fastest and most reliable on Windows)
    local_paths = [
        Path("tools/bin/colmap.exe"),
        Path("tools/colmap/bin/colmap.exe"),
        Path("tools/colmap/colmap.exe"),
        Path("tools/COLMAP/colmap.exe"),
        Path("tools/COLMAP.bat"),
    ]
    for p in local_paths:
        if p.exists():
            return str(p.resolve())

    # 2. Check system PATH
    colmap_path = shutil.which("colmap") or shutil.which("colmap.exe")
    if colmap_path:
        return colmap_path

    print("ERROR: COLMAP not found!")
    print("Install options:")
    print("  1. Download from https://colmap.github.io/ and add to PATH")
    print("  2. Extract to tools/colmap/ directory")
    sys.exit(1)


def get_colmap_env(colmap_exe: str) -> dict:
    """Build environment with DLL search directory and Qt plugin path."""
    env = os.environ.copy()
    p = Path(colmap_exe).resolve()
    bin_dir = p.parent if p.name.lower() == "colmap.exe" else p.parent / "bin"
    tools_dir = p.parent.parent if p.name.lower() == "colmap.exe" else p.parent
    plugins_dir = tools_dir / "plugins"

    path_parts = [str(bin_dir)]
    if "PATH" in env:
        path_parts.append(env["PATH"])
    env["PATH"] = ";".join(path_parts)

    if plugins_dir.exists():
        env["QT_PLUGIN_PATH"] = str(plugins_dir)

    return env


def run_colmap_command(colmap_exe, command, args_dict, description=""):
    """Run a COLMAP command with arguments using direct native process."""
    cmd = [colmap_exe, command]
    for key, value in args_dict.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))

    print(f"\n  {'─'*50}")
    print(f"  Running: {description or command}")
    print(f"  Command: {' '.join(cmd[:4])}...")

    start = time.time()
    env = get_colmap_env(colmap_exe)

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600  # 1 hour timeout
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  ❌ FAILED ({elapsed:.1f}s, exit code {result.returncode})")
        err_msg = result.stderr.strip() if result.stderr else (result.stdout.strip() if result.stdout else "(empty)")
        print(f"  LOG: {err_msg[-600:]}")
        return False, elapsed

    print(f"  ✓ Done ({elapsed:.1f}s)")
    return True, elapsed


def discover_sparse_models(sparse_dir: Path) -> list:
    """
    Universally discover all valid sparse models in sparse_dir.
    Checks all subdirectories as well as sparse_dir itself for points3D.bin / points3D.txt.
    """
    candidates = []
    if (sparse_dir / "points3D.bin").exists() or (sparse_dir / "points3D.txt").exists():
        candidates.append(sparse_dir)
    if sparse_dir.exists():
        for sub in sorted(sparse_dir.iterdir()):
            if sub.is_dir():
                if (sub / "points3D.bin").exists() or (sub / "points3D.txt").exists():
                    candidates.append(sub)
    return candidates


def ensure_model_converted_to_txt(colmap_exe: str, model_dir: Path) -> bool:
    """Ensure a model directory has cameras.txt, images.txt, and points3D.txt."""
    has_txt = (model_dir / "images.txt").exists() and (model_dir / "points3D.txt").exists()
    if has_txt:
        return True
    if (model_dir / "points3D.bin").exists():
        ok, _ = run_colmap_command(colmap_exe, "model_converter", {
            "input_path": str(model_dir),
            "output_path": str(model_dir),
            "output_type": "TXT",
        }, f"Convert model in {model_dir.name} to text format")
        return ok
    return False


def count_model_registered_images(model_dir: Path) -> int:
    """Count registered images in a model from images.txt or frames.bin."""
    img_txt = model_dir / "images.txt"
    if img_txt.exists():
        try:
            with open(img_txt, encoding="utf-8", errors="replace") as f:
                lines = [l for l in f if not l.startswith("#") and l.strip()]
            return len(lines) // 2
        except Exception:
            pass
    bin_f = model_dir / "frames.bin"
    if bin_f.exists():
        return max(1, int(bin_f.stat().st_size // 80))
    return 0


def count_model_points(model_dir: Path) -> int:
    """Count 3D points in a model."""
    pts_txt = model_dir / "points3D.txt"
    if pts_txt.exists():
        try:
            with open(pts_txt, encoding="utf-8", errors="replace") as f:
                lines = [l for l in f if not l.startswith("#") and l.strip()]
            return len(lines)
        except Exception:
            pass
    pts_bin = model_dir / "points3D.bin"
    if pts_bin.exists():
        return max(1, int(pts_bin.stat().st_size // 120))
    return 0


def read_colmap_sparse_stats(sparse_dir: Path, best_model: Path = None):
    """Read basic statistics from COLMAP sparse reconstruction."""
    stats = {
        "num_cameras": 0,
        "num_images_registered": 0,
        "num_points3D": 0,
        "model_dir": "0",
    }
    target = best_model
    if target is None:
        models = discover_sparse_models(sparse_dir)
        if models:
            target = max(models, key=lambda m: count_model_registered_images(m))

    if target and target.exists():
        stats["num_images_registered"] = count_model_registered_images(target)
        stats["num_points3D"] = count_model_points(target)
        cam_txt = target / "cameras.txt"
        if cam_txt.exists():
            try:
                with open(cam_txt, encoding="utf-8", errors="replace") as f:
                    c_lines = [l for l in f if not l.startswith("#") and l.strip()]
                stats["num_cameras"] = len(c_lines)
            except Exception:
                stats["num_cameras"] = 1
        else:
            stats["num_cameras"] = 1
        stats["model_dir"] = target.name

    return stats


def run_sparse_reconstruction(colmap_exe, project_dir, frames_dir, config):
    """Run COLMAP sparse reconstruction pipeline."""
    database_path = project_dir / "database.db"
    sparse_dir = project_dir / "sparse"
    os.makedirs(sparse_dir, exist_ok=True)
    
    fe_config = config.get("feature_extraction", {})
    fm_config = config.get("feature_matching", {})
    mapper_config = config.get("mapper", {})
    
    timings = {}
    
    # Step 1: Feature Extraction
    fe_args = {
        "database_path": str(database_path),
        "image_path": str(frames_dir),
        "ImageReader.camera_model": fe_config.get("image_reader_camera_model", "SIMPLE_RADIAL"),
        "ImageReader.single_camera": "1",
        "SiftExtraction.use_gpu": "1" if fe_config.get("use_gpu", True) else "0",
        "SiftExtraction.max_image_size": fe_config.get("max_image_size", 1600),
        "SiftExtraction.max_num_features": fe_config.get("max_num_features", 8192),
    }
    success, t = run_colmap_command(colmap_exe, "feature_extractor", fe_args, "Feature Extraction (SIFT)")
    timings["feature_extraction"] = t
    if not success:
        return None, timings

    # Step 2: Feature Matching
    match_method = fm_config.get("method", "sequential")
    if match_method == "sequential":
        match_command = "sequential_matcher"
        match_args = {
            "database_path": str(database_path),
            "SiftMatching.use_gpu": "1" if fm_config.get("use_gpu", True) else "0",
            "SequentialMatching.overlap": fm_config.get("sequential_overlap", 10),
        }
    else:
        match_command = "exhaustive_matcher"
        match_args = {
            "database_path": str(database_path),
            "SiftMatching.use_gpu": "1" if fm_config.get("use_gpu", True) else "0",
        }
    
    success, t = run_colmap_command(
        colmap_exe, match_command, match_args,
        f"Feature Matching ({match_method})"
    )
    timings["feature_matching"] = t
    if not success:
        return None, timings
    
    # Step 3: Sparse Reconstruction (Mapper)
    # Tuned for aerial / single-pass drone videos:
    # - Relaxed parallax angle (4.0 deg) for drone nadir flight
    # - Disable structure_less_fallback to prevent degenerate pose explosion
    # - Enable snapshots so model checkpoints are saved continuously
    # - multiple_models=0: optimize primary flight path directly without looping trials on outliers
    mapper_args = {
        "database_path": str(database_path),
        "image_path": str(frames_dir),
        "output_path": str(sparse_dir),
        "Mapper.min_num_matches": mapper_config.get("min_num_matches", 15),
        "Mapper.multiple_models": "0",
        "Mapper.max_num_models": "1",
        "Mapper.init_min_tri_angle": mapper_config.get("init_min_tri_angle", 4.0),
        "Mapper.filter_min_tri_angle": mapper_config.get("filter_min_tri_angle", 1.0),
        "Mapper.structure_less_registration_fallback": "0",
        "Mapper.ba_refine_focal_length": "0",
        "Mapper.snapshot_path": str(sparse_dir),
        "Mapper.snapshot_frames_freq": "20",
    }
    success, t = run_colmap_command(
        colmap_exe, "mapper", mapper_args,
        "Sparse Reconstruction (SfM)"
    )
    timings["sparse_reconstruction"] = t

    # Step 4: Universal Model Discovery across ALL models & snapshots
    models = discover_sparse_models(sparse_dir)
    if not models:
        print("  ⚠️  No sparse model directories found in sparse_dir.")
        return None, timings

    print(f"  Discovered {len(models)} reconstruction candidate(s) in {sparse_dir.name}:")
    for m in models:
        print(f"    - {m.name}")

    # Ensure all discovered models are converted to text format
    for m in models:
        ensure_model_converted_to_txt(colmap_exe, m)

    # Pick best model (most registered images, then most points)
    best_model_dir = max(models, key=lambda m: (count_model_registered_images(m), count_model_points(m)))
    reg_count = count_model_registered_images(best_model_dir)
    pts_count = count_model_points(best_model_dir)
    print(f"  ✓ Best model selected: {best_model_dir.name} ({reg_count} images, {pts_count:,} points)")

    # Ensure standard model 0 directory exists with best model content
    std_model_dir = sparse_dir / "0"
    if best_model_dir != std_model_dir:
        std_model_dir.mkdir(parents=True, exist_ok=True)
        for f in best_model_dir.glob("*"):
            if f.is_file():
                shutil.copy2(str(f), str(std_model_dir / f.name))
        print(f"  ✓ Copied best model to standard location: {std_model_dir}")

    # Export best model to PLY for 3D viewer
    ply_out = project_dir / "sparse_pointcloud.ply"
    run_colmap_command(colmap_exe, "model_converter", {
        "input_path": str(std_model_dir),
        "output_path": str(ply_out),
        "output_type": "PLY",
    }, f"Export best model to PLY ({reg_count} images)")

    return sparse_dir, timings


def run_dense_reconstruction(colmap_exe, project_dir, frames_dir, sparse_dir, config):
    """Run COLMAP dense reconstruction pipeline."""
    dense_config = config.get("dense", {})
    dense_dir = project_dir / "dense"
    os.makedirs(dense_dir, exist_ok=True)
    
    timings = {}
    models = discover_sparse_models(sparse_dir)
    if not models:
        print("  WARNING: No sparse model found, skipping dense reconstruction.")
        return None, timings

    best_model = max(models, key=lambda m: count_model_registered_images(m))
    
    # Step 1: Image undistortion
    success, t = run_colmap_command(colmap_exe, "image_undistorter", {
        "image_path": str(frames_dir),
        "input_path": str(best_model),
        "output_path": str(dense_dir),
        "output_type": "COLMAP",
        "max_image_size": dense_config.get("max_image_size", 1200),
    }, "Image Undistortion")
    timings["undistortion"] = t
    if not success:
        return None, timings
    
    # Step 2: Patch Match Stereo (photometric stereo on Windows to avoid geometric consistency driver bugs)
    success, t = run_colmap_command(colmap_exe, "patch_match_stereo", {
        "workspace_path": str(dense_dir),
        "workspace_format": "COLMAP",
        "PatchMatchStereo.geom_consistency": "false",
    }, "Patch Match Stereo (photometric depth maps)")
    timings["patch_match"] = t
    if not success:
        print("  ⚠️  Dense PatchMatch stereo skipped/failed — using high-resolution sparse cloud.")
        return None, timings
    
    # Step 3: Stereo Fusion
    fused_path = dense_dir / "fused.ply"
    success, t = run_colmap_command(colmap_exe, "stereo_fusion", {
        "workspace_path": str(dense_dir),
        "workspace_format": "COLMAP",
        "input_type": "photometric",
        "output_path": str(fused_path),
        "StereoFusion.min_num_pixels": dense_config.get("min_num_pixels", 5),
    }, "Stereo Fusion → Point Cloud")
    timings["fusion"] = t
    if not success:
        print("  ⚠️  Stereo fusion skipped/failed — using high-resolution sparse cloud.")
        return None, timings
    
    return dense_dir, timings


def export_sparse_ply(colmap_exe, sparse_dir, output_ply):
    """Export sparse point cloud to PLY format for visualization."""
    models = discover_sparse_models(sparse_dir)
    if not models:
        return False
    best_model = max(models, key=lambda m: count_model_registered_images(m))
    
    success, _ = run_colmap_command(colmap_exe, "model_converter", {
        "input_path": str(best_model),
        "output_path": str(output_ply),
        "output_type": "PLY",
    }, "Export sparse model to PLY")
    return success


def reset_project_outputs(project_dir: Path):
    """Remove previous COLMAP artifacts so reruns do not inherit stale state."""
    stale_paths = [
        project_dir / "database.db",
        project_dir / "sparse",
        project_dir / "dense",
        project_dir / "sparse_pointcloud.ply",
        project_dir / "colmap_report.json",
    ]
    for stale in stale_paths:
        if stale.is_dir():
            shutil.rmtree(stale)
        elif stale.exists():
            stale.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="AeroTwin: Run COLMAP 3D reconstruction on extracted frames"
    )
    parser.add_argument(
        "--project", "-p",
        default="default",
        help="Project name (matches frame extraction output)"
    )
    parser.add_argument(
        "--config", "-c",
        default="config/colmap_defaults.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--skip-dense",
        action="store_true",
        help="Skip dense reconstruction (only do sparse SfM)"
    )
    parser.add_argument(
        "--colmap-path",
        default=None,
        help="Explicit path to COLMAP executable"
    )
    parser.add_argument(
        "--output-base",
        default="data/output",
        help="Base output directory (default: data/output)"
    )
    
    args = parser.parse_args()
    
    # Find COLMAP
    colmap_exe = args.colmap_path or find_colmap()
    print(f"\n  Using COLMAP: {colmap_exe}")
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        print(f"  Config not found at {config_path}, using defaults.")
        config = {}
    
    # Paths
    project_dir = Path(args.output_base) / args.project
    frames_dir = project_dir / "frames"
    keyframes_dir = project_dir / "keyframes"

    # Remove stale COLMAP artifacts so reruns start from a clean slate.
    reset_project_outputs(project_dir)

    # Prefer selected keyframes for SfM when they exist; otherwise fall back to all extracted frames.
    frame_files = list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))
    keyframe_files = list(keyframes_dir.glob("*.jpg")) + list(keyframes_dir.glob("*.png"))
    if keyframe_files:
        frames_dir = keyframes_dir
        frame_files = keyframe_files
        print(f"  Using selected keyframes from {frames_dir}")
    else:
        print(f"  Using all extracted frames from {frames_dir}")

    if not frames_dir.exists():
        print(f"ERROR: Frames directory not found: {frames_dir}")
        print(f"Run 01_extract_frames.py first!")
        sys.exit(1)

    # Count frames
    num_frames = len(frame_files)

    print(f"\n{'='*60}")
    print(f"  AeroTwin — COLMAP 3D Reconstruction")
    print(f"{'='*60}")
    print(f"  Project:     {args.project}")
    print(f"  Frames:      {num_frames} images in {frames_dir}")
    print(f"  Dense:       {'disabled' if args.skip_dense else 'enabled'}")
    print(f"{'='*60}")
    
    if num_frames < 5:
        print(f"\n  ❌ ERROR: Only {num_frames} frames. Need at least 5 for reconstruction.")
        sys.exit(1)
    
    if num_frames < 20:
        print(f"\n  ⚠️  WARNING: Only {num_frames} frames. Results may be poor.")
        print(f"  Recommendation: ≥20 frames with good overlap.")
    
    total_start = time.time()
    all_timings = {}
    
    # ── SPARSE RECONSTRUCTION ──
    print(f"\n  ━━━ PHASE 1: Sparse Reconstruction ━━━")
    sparse_dir, sparse_timings = run_sparse_reconstruction(
        colmap_exe, project_dir, frames_dir, config
    )
    all_timings.update(sparse_timings)
    
    sparse_stats = {}
    if sparse_dir:
        sparse_stats = read_colmap_sparse_stats(sparse_dir)
        
        # Export sparse PLY
        sparse_ply = project_dir / "sparse_pointcloud.ply"
        export_sparse_ply(colmap_exe, sparse_dir, sparse_ply)
        
        print(f"\n  ✓ Sparse reconstruction complete!")
        print(f"    Registered images: {sparse_stats.get('num_images_registered', '?')}/{num_frames}")
        print(f"    3D points:         {sparse_stats.get('num_points3D', '?')}")
    else:
        print(f"\n  ❌ Sparse reconstruction FAILED.")
        print(f"  Possible causes:")
        print(f"    - Not enough overlap between frames")
        print(f"    - Too few features (textureless surfaces)")
        print(f"    - Blurry images")
        print(f"    - Try: reduce frame_interval, lower blur_threshold")
    
    # ── DENSE RECONSTRUCTION ──
    dense_stats = {}
    if sparse_dir and not args.skip_dense:
        dense_enabled = config.get("dense", {}).get("enabled", True)
        if dense_enabled:
            print(f"\n  ━━━ PHASE 2: Dense Reconstruction ━━━")
            dense_dir, dense_timings = run_dense_reconstruction(
                colmap_exe, project_dir, frames_dir, sparse_dir, config
            )
            all_timings.update(dense_timings)
            
            if dense_dir:
                fused_ply = dense_dir / "fused.ply"
                if fused_ply.exists():
                    size_mb = fused_ply.stat().st_size / (1024 * 1024)
                    dense_stats["fused_ply_size_mb"] = round(size_mb, 2)
                    print(f"\n  ✓ Dense point cloud: {fused_ply} ({size_mb:.1f} MB)")
    
    total_elapsed = time.time() - total_start
    
    # ── FINAL REPORT ──
    print(f"\n{'='*60}")
    print(f"  RECONSTRUCTION REPORT")
    print(f"{'='*60}")
    print(f"  Input frames:           {num_frames}")
    print(f"  Registered images:      {sparse_stats.get('num_images_registered', 'FAILED')}")
    print(f"  Sparse 3D points:       {sparse_stats.get('num_points3D', 'FAILED')}")
    if dense_stats:
        print(f"  Dense point cloud:      {dense_stats.get('fused_ply_size_mb', '?')} MB")
    print(f"  Total processing time:  {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"")
    for step, t in all_timings.items():
        print(f"    {step:30s} {t:8.1f}s")
    print(f"{'='*60}")
    
    # Save report
    report = {
        "project": args.project,
        "num_input_frames": num_frames,
        "sparse": sparse_stats,
        "dense": dense_stats,
        "timings": {k: round(v, 2) for k, v in all_timings.items()},
        "total_time_sec": round(total_elapsed, 2),
        "success": sparse_dir is not None,
    }
    
    report_path = project_dir / "colmap_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")
    
    # Return code
    if sparse_dir is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
