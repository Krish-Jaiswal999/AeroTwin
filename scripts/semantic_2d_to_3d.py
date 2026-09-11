"""
scripts/semantic_2d_to_3d.py

Project real 2D semantic masks into the dense 3D point cloud using COLMAP
cameras. Multi-view weighted voting. No HSV. No all-road fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CLASS_ID = {
    "Unknown": 0,
    "Road/Ground": 1,
    "Building/Roof": 2,
    "Vegetation": 3,
    "Water": 4,
    "Vehicle": 5,
}
N_CLASS = 6


def qvec_to_rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)


def parse_cameras_txt(path: Path) -> dict:
    cameras = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cameras[int(parts[0])] = {
                "model": parts[1],
                "width": int(parts[2]),
                "height": int(parts[3]),
                "params": list(map(float, parts[4:])),
            }
    return cameras


def parse_images_txt(path: Path) -> dict:
    images = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [l.rstrip() for l in f if not l.startswith("#") and l.strip()]
    i = 0
    while i < len(lines):
        meta = lines[i].split()
        if len(meta) < 10:
            i += 1
            continue
        qvec = list(map(float, meta[1:5]))
        tvec = np.array(list(map(float, meta[5:8])), dtype=np.float64)
        R = qvec_to_rotmat(qvec)
        C = -R.T @ tvec
        images[int(meta[0])] = {
            "name": meta[9],
            "cam_id": int(meta[8]),
            "qvec": qvec,
            "tvec": tvec,
            "R": R,
            "C": C,
        }
        i += 2
    return images


def discover_sparse_dir(job_dir: Path) -> Path | None:
    for root in [job_dir / "sparse", job_dir / job_dir.name / "sparse"]:
        if not root.exists():
            continue
        for preferred in ["0", "1"]:
            sub = root / preferred
            if (sub / "cameras.txt").exists() and (sub / "images.txt").exists():
                return sub
        for sub in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime if p.is_dir() else 0, reverse=True):
            if sub.is_dir() and (sub / "cameras.txt").exists() and (sub / "images.txt").exists():
                return sub
    return None


def discover_dense_ply(job_dir: Path) -> Path | None:
    search = [job_dir, job_dir / job_dir.name]
    names = [
        "dense_clean.ply",
        "dense/fused.ply",
        "fused.ply",
        "dense.ply",
        "dense_display.ply",
        "display_cloud.ply",
        "sparse_pointcloud.ply",
    ]
    for d in search:
        for name in names:
            p = d / name if "/" not in name else d.joinpath(*name.split("/"))
            if p.exists() and p.stat().st_size > 500:
                return p
    return None


def read_ply(path: Path):
    with open(path, "rb") as f:
        n_verts = 0
        is_binary = False
        has_rgb = False
        prop_order = []
        while True:
            line = b""
            while not line.endswith(b"\n"):
                c = f.read(1)
                if not c:
                    break
                line += c
            line = line.strip().decode("ascii", "replace")
            if line.startswith("format binary_little_endian"):
                is_binary = True
            elif line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif line.startswith("property float"):
                prop_order.append((line.split()[-1], "f4"))
            elif line.startswith("property uchar") or line.startswith("property uint8"):
                prop = line.split()[-1]
                prop_order.append((prop, "u1"))
                if prop in ("red", "r"):
                    has_rgb = True
            elif line == "end_header":
                break

        if is_binary:
            x_off = y_off = z_off = None
            r_off = g_off = b_off = None
            byte_offset = 0
            for p, typ in prop_order:
                if p == "x":
                    x_off = byte_offset
                    byte_offset += 4
                elif p == "y":
                    y_off = byte_offset
                    byte_offset += 4
                elif p == "z":
                    z_off = byte_offset
                    byte_offset += 4
                elif p in ("red", "green", "blue", "r", "g", "b", "alpha", "a"):
                    if p in ("red", "r"):
                        r_off = byte_offset
                    elif p in ("green", "g"):
                        g_off = byte_offset
                    elif p in ("blue", "b"):
                        b_off = byte_offset
                    byte_offset += 1
                else:
                    byte_offset += 4 if typ == "f4" else 1
            stride = byte_offset
            raw = f.read(n_verts * stride)
            pts = np.empty((n_verts, 3), dtype=np.float32)
            rgb = np.empty((n_verts, 3), dtype=np.uint8) if has_rgb else None
            for i in range(n_verts):
                base = i * stride
                pts[i, 0] = struct.unpack_from("<f", raw, base + x_off)[0]
                pts[i, 1] = struct.unpack_from("<f", raw, base + y_off)[0]
                pts[i, 2] = struct.unpack_from("<f", raw, base + z_off)[0]
                if has_rgb:
                    rgb[i, 0] = raw[base + r_off]
                    rgb[i, 1] = raw[base + g_off]
                    rgb[i, 2] = raw[base + b_off]
            return pts, rgb

        lines = f.read().decode("ascii", "replace").strip().split("\n")
        pts, rgb = [], []
        for l in lines:
            p = l.split()
            if len(p) >= 3:
                pts.append([float(p[0]), float(p[1]), float(p[2])])
                if len(p) >= 6:
                    rgb.append([int(p[3]), int(p[4]), int(p[5])])
        return np.array(pts, dtype=np.float32), (np.array(rgb, dtype=np.uint8) if rgb else None)


def voxel_downsample(pts: np.ndarray, voxel_size: float, rgb=None, target_max: int = 1_500_000):
    """Real voxel downsample (no invented points)."""
    if len(pts) <= target_max:
        return pts, rgb, np.arange(len(pts), dtype=np.int64)

    mins = pts.min(axis=0)
    voxel_ids = np.floor((pts - mins) / voxel_size).astype(np.int64)
    M = voxel_ids.max(axis=0) + 1
    M = np.maximum(M, 1)
    flat = (voxel_ids[:, 0] * M[1] * M[2] + voxel_ids[:, 1] * M[2] + voxel_ids[:, 2])
    _, first = np.unique(flat, return_index=True)
    if len(first) > target_max:
        rng = np.random.default_rng(0)
        first = rng.choice(first, size=target_max, replace=False)
    return pts[first], (None if rgb is None else rgb[first]), first


def load_mask_for_image(semantics_dir: Path, image_name: str):
    stem = Path(image_name).stem
    candidates = [
        semantics_dir / f"{stem}_mask.png",
        semantics_dir / f"{Path(image_name).stem}_mask.png",
    ]
    for p in candidates:
        if p.exists():
            mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            meta = {}
            jp = semantics_dir / f"{stem}.json"
            if jp.exists():
                with open(jp, encoding="utf-8") as f:
                    meta = json.load(f)
            return mask, meta
    return None, {}


def select_projection_views(images: dict, frames_dir: Path, semantics_dir: Path,
                            keyframe_names: set[str] | None) -> list:
    usable = []
    for img_id, info in images.items():
        name = info["name"]
        stem = Path(name).stem
        mask_ok = (semantics_dir / f"{stem}_mask.png").exists()
        frame_ok = (frames_dir / name).exists() if frames_dir else False
        if not mask_ok:
            continue
        kf_bonus = 1.0 if (not keyframe_names or name in keyframe_names or f"{stem}.jpg" in keyframe_names) else 0.0
        usable.append((img_id, kf_bonus, name, frame_ok))

    if not usable:
        return []

    # Prefer keyframes, then spread temporally by name
    usable.sort(key=lambda x: (x[2], -x[1]))
    n = len(usable)
    # Adaptive: use more views as reconstruction grows, cap for MVP cost
    target = int(np.clip(n, 12, 80))
    if n <= target:
        return [u[0] for u in usable]
    step = n / target
    chosen = []
    t = 0.0
    while t < n and len(chosen) < target:
        chosen.append(usable[int(t)][0])
        t += step
    return chosen


def project_points(pts, R, tvec, cam, mask_shape):
    W, H = cam["width"], cam["height"]
    params = cam["params"]
    model = cam.get("model", "SIMPLE_RADIAL")

    p_cam = (pts @ R.T) + tvec
    z = p_cam[:, 2]
    z_safe = np.clip(z, 1e-6, None)
    x_n = p_cam[:, 0] / z_safe
    y_n = p_cam[:, 1] / z_safe

    mH, mW = mask_shape

    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        f = params[0]
        cx = params[1] if len(params) > 1 else W / 2.0
        cy = params[2] if len(params) > 2 else H / 2.0
        k = params[3] if len(params) > 3 else 0.0
        r2 = x_n * x_n + y_n * y_n
        factor = 1.0 + k * r2
        u = f * x_n * factor + cx
        v = f * y_n * factor + cy
    else:
        # PINHOLE / OPENCV-style models can carry extra distortion terms.
        fx = params[0]
        fy = params[1] if len(params) > 1 else params[0]
        cx = params[2] if len(params) > 2 else W / 2.0
        cy = params[3] if len(params) > 3 else H / 2.0

        # Best-effort support for a radial distortion term if present.
        k = params[4] if len(params) > 4 else 0.0
        r2 = x_n * x_n + y_n * y_n
        factor = 1.0 + k * r2
        u = fx * x_n * factor + cx
        v = fy * y_n * factor + cy

    u_m = (u * (mW / W)).astype(np.int32)
    v_m = (v * (mH / H)).astype(np.int32)
    in_front = z > 0.05
    in_frame = in_front & (u_m >= 0) & (u_m < mW) & (v_m >= 0) & (v_m < mH)
    return in_frame, u_m, v_m, z


def run_projection(job_dir: Path) -> dict:
    t0 = time.time()
    sparse_dir = discover_sparse_dir(job_dir)
    ply_path = discover_dense_ply(job_dir)
    semantics_dir = job_dir / "semantics"
    if not semantics_dir.exists():
        nested = job_dir / job_dir.name / "semantics"
        if nested.exists():
            semantics_dir = nested

    frames_dir = None
    for c in [job_dir / "keyframes", job_dir / "frames",
              job_dir / job_dir.name / "keyframes", job_dir / job_dir.name / "frames"]:
        if c.exists():
            frames_dir = c
            break

    report = {
        "success": False,
        "source_ply": None,
        "source_points": 0,
        "labeled_points": 0,
        "views_used": 0,
        "coverage_pct": 0.0,
        "error": None,
        "classes": {},
    }

    if not sparse_dir:
        report["error"] = "COLMAP sparse model not found"
        _save(job_dir, report, None, None, None)
        return report
    if not ply_path:
        report["error"] = "No reconstruction PLY found"
        _save(job_dir, report, None, None, None)
        return report
    if not semantics_dir.exists():
        report["error"] = "Semantic masks not found (semantic AI did not produce masks)"
        _save(job_dir, report, None, None, None)
        return report

    cameras = parse_cameras_txt(sparse_dir / "cameras.txt")
    images = parse_images_txt(sparse_dir / "images.txt")
    pts, rgb = read_ply(ply_path)
    n_src = len(pts)
    report["source_ply"] = str(ply_path.name)
    report["source_points"] = int(n_src)

    # Real downsample for tractable voting (not amplification)
    spans = pts.max(axis=0) - pts.min(axis=0)
    voxel = max(1e-4, float(np.mean(spans)) / 400.0)
    pts_use, rgb_use, idx_map = voxel_downsample(pts, voxel, rgb, target_max=1_200_000)
    N = len(pts_use)
    print(f"  Source: {ply_path.name}  {n_src:,} pts → voting on {N:,} pts")

    kf_names = set()
    kf_report = job_dir / "keyframe_selection_report.json"
    if not kf_report.exists():
        kf_report = job_dir / job_dir.name / "keyframe_selection_report.json"
    if kf_report.exists():
        with open(kf_report, encoding="utf-8") as f:
            kf_names = set(json.load(f).get("keyframe_filenames", []))

    selected = select_projection_views(images, frames_dir, semantics_dir, kf_names)
    print(f"  Projection views: {len(selected)} registered cameras with masks")
    if not selected:
        report["error"] = "No registered cameras have semantic masks"
        labels = np.zeros(N, np.int32)
        conf = np.zeros(N, np.float32)
        votes_n = np.zeros(N, np.int32)
        _save(job_dir, report, labels, conf, votes_n)
        return report

    vote_w = np.zeros((N, N_CLASS), dtype=np.float32)
    vote_n = np.zeros(N, dtype=np.int32)
    pts64 = pts_use.astype(np.float64)

    for img_id in selected:
        info = images[img_id]
        cam = cameras.get(info["cam_id"])
        if not cam:
            continue
        mask, meta = load_mask_for_image(semantics_dir, info["name"])
        if mask is None:
            continue
        in_frame, u_m, v_m, z = project_points(pts64, info["R"], info["tvec"], cam, mask.shape[:2])
        idx = np.where(in_frame)[0]
        if len(idx) == 0:
            continue

        cls = mask[v_m[idx], u_m[idx]].astype(np.int32)
        cls = np.clip(cls, 0, N_CLASS - 1)

        # Weights: confidence from JSON mean, viewing angle (front-facing), inverse distance
        mean_conf = float(meta.get("mean_confidence") or 0.6)
        C = info["C"]
        view_dir = pts64[idx] - C
        dist = np.linalg.norm(view_dir, axis=1) + 1e-6
        # Optical axis in world: R maps world→cam, camera looks +Z_cam
        cam_fwd = R_to_forward(info["R"])
        view_n = view_dir / dist[:, None]
        cosang = np.clip(-(view_n @ cam_fwd), 0.0, 1.0)  # facing the camera
        w = (0.35 + 0.65 * mean_conf) * (0.25 + 0.75 * cosang) * (1.0 / (1.0 + dist / (np.median(dist) + 1e-6)))

        for c in range(N_CLASS):
            m = cls == c
            if np.any(m):
                vote_w[idx[m], c] += w[m]
        vote_n[idx] += 1

    labels = np.zeros(N, dtype=np.int32)
    conf = np.zeros(N, dtype=np.float32)
    voted = vote_n > 0
    if np.any(voted):
        labels[voted] = np.argmax(vote_w[voted], axis=1)
        total_w = vote_w[voted].sum(axis=1)
        best = vote_w[voted, labels[voted]]
        conf[voted] = np.where(total_w > 0, best / total_w, 0.0)
        # Require at least some support; keep Unknown if only 1 weak view of Unknown
        low = voted & (vote_n < 1)
        labels[low] = 0

    labeled = int(np.sum(labels > 0))
    coverage = 100.0 * labeled / max(N, 1)
    print(f"  Labeled (non-Unknown): {labeled:,}/{N:,}  coverage={coverage:.1f}%")

    classes = {}
    for name, cid in CLASS_ID.items():
        cnt = int(np.sum(labels == cid))
        classes[name] = {"count": cnt, "pct": round(100.0 * cnt / max(N, 1), 2)}

    report.update({
        "success": True,
        "labeled_points": labeled,
        "voting_points": int(N),
        "views_used": len(selected),
        "coverage_pct": round(coverage, 2),
        "classes": classes,
        "error": None if coverage > 0 else "Projection produced no non-Unknown labels",
        "downsample": {"source_points": int(n_src), "voting_points": int(N)},
        "elapsed_sec": round(time.time() - t0, 1),
    })
    _save(job_dir, report, labels, conf, vote_n)
    # Also store rgb/pts used so scene builder can consume aligned arrays
    np.savez_compressed(
        job_dir / "semantic_3d.npz",
        pts=pts_use,
        rgb=rgb_use if rgb_use is not None else np.zeros((N, 3), np.uint8),
        labels=labels,
        confidence=conf,
        vote_count=vote_n,
    )
    print(f"  ✓ semantic_3d.npz + semantic_3d_report.json")
    return report


def R_to_forward(R: np.ndarray) -> np.ndarray:
    """World-space camera optical axis (+Z in camera)."""
    return R.T[:, 2]


def _save(job_dir: Path, report: dict, labels, conf, vote_n):
    out = job_dir / "semantic_3d_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    if labels is not None:
        np.save(job_dir / "semantic_3d_labels.npy", labels)
        np.save(job_dir / "semantic_3d_conf.npy", conf)
        np.save(job_dir / "semantic_3d_votes.npy", vote_n)


def main():
    parser = argparse.ArgumentParser(description="AeroTwin 2D→3D semantic projection")
    parser.add_argument("--project", "-p", default="sample_flight")
    parser.add_argument("--output-base", default="data/output")
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--jobs-base", default="jobs")
    args = parser.parse_args()

    if args.job_id:
        root = Path(args.jobs_base) / args.job_id
        job_dir = root / args.job_id if (root / args.job_id).exists() else root
    else:
        job_dir = Path(args.output_base) / args.project

    print("\n" + "=" * 60)
    print("  AEROTWIN — 2D to 3D Semantic Projection")
    print("=" * 60)
    run_projection(job_dir)


if __name__ == "__main__":
    main()
