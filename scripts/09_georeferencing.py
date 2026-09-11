"""
scripts/09_georeferencing.py
AeroTwin — Feature #4: GPS / Georeferencing
Reads camera poses from COLMAP sparse reconstruction, computes the flight coordinate
system, and outputs:
  - Camera positions in COLMAP local coordinates
  - Scaled metric distances between camera pairs
  - A placeholder GPS alignment transform (ready to accept real GPS telemetry)
  - data/output/<project>/georef_report.json
"""

import sys
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_images_txt(path):
    """Reads COLMAP images.txt and returns list of (name, camera_position_3d)"""
    cameras = []
    with open(path, "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#") or not line:
            i += 1
            continue
        parts = line.split()
        if len(parts) >= 10:
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            name = parts[9]
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            t = np.array([tx, ty, tz])
            pos = -R.T @ t  # Camera position in world frame
            cameras.append({"name": name, "position": pos.tolist()})
            i += 2
        else:
            i += 1
    return cameras


def compute_georef_report(project="sample_flight", output_base="data/output"):
    output_dir = Path(output_base) / project
    sparse_dir = output_dir / "sparse"

    # Universally search for images.txt across all subdirectories
    candidate_files = []
    if (sparse_dir / "images.txt").exists():
        candidate_files.append(sparse_dir / "images.txt")
    if sparse_dir.exists():
        for sub in sorted(sparse_dir.iterdir()):
            if sub.is_dir() and (sub / "images.txt").exists():
                candidate_files.append(sub / "images.txt")

    if not candidate_files:
        print(f"images.txt not found in {sparse_dir}")
        return None

    # Pick the model with the largest images.txt
    images_txt = max(candidate_files, key=lambda p: p.stat().st_size)

    print(f"Parsing camera positions from {images_txt}...")
    cameras = parse_images_txt(images_txt)
    cameras.sort(key=lambda x: x["name"])
    if not cameras:
        print("No cameras found.")
        return None

    positions = np.array([c["position"] for c in cameras])

    # Compute trajectory metrics
    diffs = np.diff(positions, axis=0)
    step_distances = np.linalg.norm(diffs, axis=1)
    median_step = float(np.median(step_distances))
    flight_span_local = positions.max(axis=0) - positions.min(axis=0)

    # Frame timing: video is 60fps, sampled every 15 frames = 0.25s per frame pair
    frame_interval_s = 15 / 60.0

    # Scale estimate: use median step distance / speed
    # Typical drone hover speed ~3m/s at 15fps intervals:
    # Each step = 0.25s, if drone moves at 3m/s then 1 unit ≈ 3*0.25 / median_step m
    assumed_drone_speed_m_s = 2.5
    if median_step > 0:
        scale_m_per_unit = (assumed_drone_speed_m_s * frame_interval_s) / median_step
    else:
        scale_m_per_unit = 1.0

    trajectory_length_m = float(np.sum(step_distances)) * scale_m_per_unit
    span_m = flight_span_local * scale_m_per_unit

    print(f"  Cameras registered: {len(cameras)}")
    print(f"  Scale estimate: {scale_m_per_unit:.4f} m/COLMAP-unit")
    print(f"  Trajectory length (metric): {trajectory_length_m:.1f} m")
    print(f"  Coverage span: X={span_m[0]:.1f}m, Y={span_m[1]:.1f}m, Z={span_m[2]:.1f}m")

    # Georeferencing placeholder (requires real GPS telemetry to align)
    georef_note = (
        "GPS metadata NOT present in this video file. "
        "To enable true lat/lon georeferencing, provide a DJI telemetry SRT file or "
        "a GPX track alongside the video. The local coordinate system is metric-consistent "
        "and can be anchored to any known ground control point (GCP)."
    )

    # Explicit availability contract: the metric scale is a motion-derived estimate
    # until a real telemetry source or GCP calibration is wired in.
    gps_available = False
    scale_available = False

    # Camera path for web viewer
    cam_path = [{"id": i, "name": c["name"], "local_xyz": c["position"]} for i, c in enumerate(cameras)]

    report = {
        "feature": "GPS / Georeferencing",
        "cameras_registered": len(cameras),
        "scale_m_per_unit": round(scale_m_per_unit, 5),
        "scale_available": scale_available,
        "scale_source": "video motion estimate (not GPS-verified)",
        "trajectory_length_m": round(trajectory_length_m, 2),
        "coverage_span_m": {"x": round(float(span_m[0]), 2), "y": round(float(span_m[1]), 2), "z": round(float(span_m[2]), 2)},
        "coordinate_system": "COLMAP local (metric-consistent, not lat/lon anchored)",
        "gps_status": "not_available",
        "gps_status_message": georef_note,
        "gps_available": gps_available,
        "camera_positions_local": cam_path[:10],  # First 10 for preview
    }

    out_path = output_dir / "georef_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"  Georeferencing report saved: {out_path}")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AeroTwin Georeferencing")
    parser.add_argument("--project", default="sample_flight")
    parser.add_argument("--output-base", default="data/output")
    args = parser.parse_args()
    compute_georef_report(project=args.project, output_base=args.output_base)
    print("\nFeature #4 complete.")
