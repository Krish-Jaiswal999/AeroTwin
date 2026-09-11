"""
scripts/08_semantic_filter.py
AeroTwin — Semantic Classification & Dynamic Object Filtering

Performs:
  - HSV color-space semantic segmentation (road, building, vegetation, sky, water)
  - MOG2 background subtraction to detect dynamic objects
  - Saves per-frame GRAYSCALE MASK PNGs for 2D→3D projection

Designed for aerial/nadir drone imagery.

Outputs:
  <project_dir>/semantics/          — colorized semantic overlay images (visual JPEGs)
  <project_dir>/semantics/<stem>_mask.png  — grayscale class ID masks (for projection)
  <project_dir>/semantics/<stem>.json      — per-frame confidence/class JSON
  <project_dir>/dynamic_masks/      — binary foreground masks
  <project_dir>/semantic_classification_report.json
  <project_dir>/dynamic_object_report.json
"""

import argparse
import cv2
import json
import sys
import time
import numpy as np
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


CLASS_NAMES = {
    0: "Unknown",
    1: "Road/Ground",
    2: "Building/Roof",
    3: "Vegetation",
    4: "Water",
    5: "Vehicle",
}

# BGR colors for overlay visualization
CLASS_COLORS = {
    0: (60,  60,  60),   # Unknown       — dark gray
    1: (110, 115, 125),  # Road/Ground   — asphalt gray
    2: (50,  180, 220),  # Building/Roof — teal/cyan
    3: (35,  180, 60),   # Vegetation    — vibrant green
    4: (210, 140, 30),   # Water         — azure blue (BGR)
    5: (30,  120, 240),  # Vehicle       — orange (BGR)
}


def classify_pixel_hsv(bgr_img):
    """
    Per-pixel semantic classification using HSV color space.
    Tuned for aerial/nadir drone imagery across residential, urban, and landscape scenes.
    Returns a per-pixel class mask with class IDs (uint8).

    Classification order matters — higher specificity classes first.
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Convert to float for lab space analysis too
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2Lab)
    l_ch = lab[:, :, 0].astype(np.float32)
    a_ch = lab[:, :, 1].astype(np.float32)
    b_ch = lab[:, :, 2].astype(np.float32)

    mask = np.zeros(h.shape, dtype=np.uint8)  # Start all Unknown

    # 1. Vegetation (lawns, trees, shrubs, foliage)
    # Green-dominant in HSV: hue 30-90, saturation >30, decent brightness
    is_veg = (h >= 30) & (h <= 88) & (s >= 30) & (v >= 25) & (v <= 250)
    mask[is_veg] = 3

    # 2. Water (lakes, ponds, swimming pools, rivers) — blue dominant
    # Blue-cyan hue range (90-135), also dark reflective surfaces
    is_water_blue = (h >= 89) & (h <= 135) & (s >= 45) & (v >= 40)
    is_water_dark = (h >= 89) & (h <= 175) & (s >= 20) & (v >= 50) & (v <= 160) & (mask == 0)
    is_water = (is_water_blue | is_water_dark) & (mask == 0)
    mask[is_water] = 4

    # 3. Roads, Driveways, Sidewalks, Pavement (neutral asphalt/concrete)
    # Low saturation (gray), moderate-high brightness, NOT green
    # Also includes gravel, dirt paths (brown-gray)
    is_road_gray = (s < 30) & (v >= 80) & (v <= 240) & (mask == 0)
    # Brownish earthy tones (dirt, gravel) — warm low-sat hues
    is_road_brown = (h >= 10) & (h <= 30) & (s >= 10) & (s < 60) & (v >= 80) & (v <= 200) & (mask == 0)
    is_road = (is_road_gray | is_road_brown) & (mask == 0)
    mask[is_road] = 1

    # 4. Building / House Roofs (shingles, terracotta, slate, solar panels, walls)
    # What remains after vegetation, water, road: medium-bright non-green, non-blue areas
    # Roofs: reddish terracotta, slate gray, white reflective, brown shingles
    # Any area with moderate brightness that isn't vegetation/water/road
    is_roof = (v >= 40) & (v <= 230) & (mask == 0)
    # Exclude sky (very high brightness + low saturation)
    is_sky = (s < 20) & (v > 230)
    is_roof = is_roof & ~is_sky
    mask[is_roof] = 2

    # 5. High-brightness unknown = likely sky (don't classify as building)
    mask[is_sky & (mask == 2)] = 0

    return mask


def compute_class_confidence(mask, class_id):
    """Estimate per-class confidence based on region compactness."""
    binary = (mask == class_id).astype(np.uint8)
    if not np.any(binary):
        return 0.5
    # Use fill ratio as proxy for confidence
    total_px = binary.size
    class_px = int(np.sum(binary))
    ratio = class_px / total_px
    # More coherent segmentation = higher confidence
    return float(np.clip(0.5 + ratio * 2.0, 0.5, 0.95))


def run_semantic_segmentation(frames_dir: Path, output_dir: Path,
                               keyframes_dir: Path = None) -> dict:
    """
    Process frames for semantic segmentation.
    Saves:
      - Visual overlay JPEGs (sem_*.jpg) for display
      - Grayscale mask PNGs (<stem>_mask.png) for 2D→3D projection
      - Per-frame JSON (<stem>.json) with confidence metadata
    """
    # Prefer keyframes if available (better COLMAP alignment)
    if keyframes_dir and keyframes_dir.exists():
        frames = sorted(list(keyframes_dir.glob("*.jpg")))
        source_desc = f"keyframes"
    else:
        frames = sorted(list(frames_dir.glob("*.jpg")))
        source_desc = "frames"

    if not frames:
        print(f"  WARNING: No frames in {frames_dir}")
        return {"frames_analyzed": 0, "results": []}

    semantics_dir = output_dir / "semantics"
    semantics_dir.mkdir(exist_ok=True)

    print(f"  Semantic segmentation: {len(frames)} {source_desc}")
    print(f"  Output: {semantics_dir}")
    print(f"  Saving mask PNGs + visual JPEGs + per-frame JSON")

    results = []
    for frame_path in frames:
        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        # Work at slightly reduced resolution for speed, but keep detail
        h, w = img.shape[:2]
        target_w = min(960, w)
        target_h = int(h * target_w / w)
        small = cv2.resize(img, (target_w, target_h))

        # Compute semantic mask
        mask = classify_pixel_hsv(small)
        total = mask.size

        class_stats = {}
        class_conf = {}
        for cls_id, name in CLASS_NAMES.items():
            pct = round(float(np.sum(mask == cls_id) / total * 100), 2)
            class_stats[name] = pct
            class_conf[name] = compute_class_confidence(mask, cls_id) if pct > 0 else 0.0

        # ── CRITICAL: Save grayscale mask PNG for 2D→3D projection ──
        stem = frame_path.stem
        mask_path = semantics_dir / f"{stem}_mask.png"
        cv2.imwrite(str(mask_path), mask)  # uint8 class IDs 0-5

        # ── Save per-frame JSON with metadata ──
        frame_meta = {
            "frame": frame_path.name,
            "stem": stem,
            "mask_png": mask_path.name,
            "mean_confidence": float(np.mean(list(class_conf.values()))),
            "classes_pct": class_stats,
            "classes_conf": class_conf,
        }
        json_path = semantics_dir / f"{stem}.json"
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(frame_meta, jf, indent=2)

        # ── Generate colorized visual overlay (for display only) ──
        overlay = small.copy()
        for cls_id, color in CLASS_COLORS.items():
            region = mask == cls_id
            if np.any(region):
                overlay[region] = (
                    small[region] * 0.35 + np.array(color, dtype=np.float32) * 0.65
                ).clip(0, 255).astype(np.uint8)

        # Add class legend to overlay
        legend_h = 25 * len(CLASS_NAMES)
        legend = np.zeros((legend_h, target_w, 3), dtype=np.uint8)
        for i, (cls_id, name) in enumerate(CLASS_NAMES.items()):
            color = CLASS_COLORS[cls_id]
            pct = class_stats.get(name, 0)
            cv2.rectangle(legend, (0, i*25), (20, (i+1)*25-2), color, -1)
            cv2.putText(legend, f"{name}: {pct:.1f}%", (25, i*25+17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        # Side-by-side: original | semantic overlay
        combined = np.concatenate([small, overlay], axis=1)
        combined_with_legend = np.concatenate([combined, np.hstack([legend, legend])], axis=0)

        out_name = "sem_" + frame_path.name
        cv2.imwrite(str(semantics_dir / out_name), combined_with_legend,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

        results.append({
            "frame": frame_path.name,
            "stem": stem,
            "semantic_image": out_name,
            "mask_png": mask_path.name,
            "classes": class_stats,
        })

    # Compute averages across all frames
    if results:
        avg = {}
        for cls_name in CLASS_NAMES.values():
            avg[cls_name] = round(
                sum(r["classes"].get(cls_name, 0) for r in results) / len(results), 2
            )
    else:
        avg = {}

    report = {
        "method": "HSV_Semantic_Segmentation_with_Masks",
        "frames_analyzed": len(results),
        "total_frames": len(frames),
        "mask_format": "grayscale PNG class IDs (0=Unknown, 1=Road, 2=Building, 3=Vegetation, 4=Water, 5=Vehicle)",
        "class_averages": avg,
        "results": results,
    }

    out_path = output_dir / "semantic_classification_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"  ✓ Saved {len(results)} mask PNGs + visual overlays + JSONs")
    if avg:
        print("  Class averages:")
        for cls_name, pct in sorted(avg.items(), key=lambda x: -x[1]):
            if pct > 0.1:
                print(f"    {cls_name:20s}: {pct:.1f}%")

    return report


def run_dynamic_filtering(frames_dir: Path, output_dir: Path) -> dict:
    """Detect dynamic objects using MOG2 background subtraction."""
    frames = sorted(list(frames_dir.glob("*.jpg")))
    if not frames:
        return {"total_frames": 0, "high_motion_count": 0}

    masks_dir = output_dir / "dynamic_masks"
    masks_dir.mkdir(exist_ok=True)

    print(f"  Dynamic filtering on {len(frames)} frames (MOG2)...")

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=80, varThreshold=35, detectShadows=True
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    per_frame = []
    high_motion = []

    for idx, frame_path in enumerate(frames):
        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        small = cv2.resize(img, (640, 360))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        fg = bg_sub.apply(gray)
        fg_clean = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        dynamic_px = int(np.sum(fg_clean > 200))
        ratio = round(dynamic_px / fg_clean.size * 100, 3)

        per_frame.append({"frame": frame_path.name, "dynamic_pct": ratio})
        if ratio > 2.0:
            high_motion.append(frame_path.name)

        # Save sample masks every 20 frames
        if idx % 20 == 0:
            cv2.imwrite(str(masks_dir / ("mask_" + frame_path.name)), fg_clean)

    report = {
        "total_frames": len(frames),
        "high_motion_count": len(high_motion),
        "high_motion_frames": high_motion,
        "per_frame": per_frame,
    }

    out_path = output_dir / "dynamic_object_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"  ✓ High-motion frames: {len(high_motion)} / {len(frames)}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="AeroTwin: Semantic classification + dynamic object filtering"
    )
    parser.add_argument("--project", "-p", default="sample_flight",
                        help="Project name")
    parser.add_argument("--output-base", default="data/output",
                        help="Base output directory")
    args = parser.parse_args()

    project_dir = Path(args.output_base) / args.project
    frames_dir = project_dir / "frames"
    keyframes_dir = project_dir / "keyframes"

    if not frames_dir.exists() and not keyframes_dir.exists():
        print(f"ERROR: No frames directory found in {project_dir}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  AEROTWIN — Semantic & Dynamic Analysis")
    print("=" * 60)

    t0 = time.time()

    # Mirror COLMAP's input selection: prefer keyframes/ when present because
    # that is the directory COLMAP will register, then fall back to frames/.
    if keyframes_dir.exists() and any(keyframes_dir.glob("*.jpg")):
        print(f"\n  Processing keyframes/ directory (COLMAP-preferred source)...")
        run_semantic_segmentation(keyframes_dir, project_dir, keyframes_dir=keyframes_dir)
        print()
    elif frames_dir.exists():
        print(f"\n  Processing frames/ directory (fallback source)...")
        run_semantic_segmentation(frames_dir, project_dir, keyframes_dir=None)
        print()

    # Run dynamic filtering (always on frames)
    if frames_dir.exists():
        run_dynamic_filtering(frames_dir, project_dir)

    print(f"\n  Total time: {round(time.time()-t0, 1)}s")
    print("=" * 60)


if __name__ == "__main__":
    main()

