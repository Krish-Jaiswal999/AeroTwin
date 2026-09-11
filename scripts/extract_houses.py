import numpy as np
import sys
sys.path.append('.')
from scripts.test_multiview import pts, labels

# Class ids (see scripts/test_multiview.py): 1 = Road/Pavement, 2 = House/Roof
roof_pts = pts[labels == 2]
ground_pts = pts[labels == 1]

# Global ground reference (fallback for lots with no nearby road points).
# Use a low percentile rather than min() to stay robust to outlier/noise points.
if len(ground_pts) >= 10:
    global_ground_z = float(np.percentile(ground_pts[:, 2], 5))
else:
    # No classified ground points at all — fall back to the low end of all points
    global_ground_z = float(np.percentile(pts[:, 2], 5))

GROUND_SEARCH_RADIUS = 30.0   # m, how far to look for local ground points around a lot
MIN_LOCAL_GROUND_PTS = 8      # need at least this many nearby ground points to trust a local estimate
MIN_HOUSE_HEIGHT_M = 2.5      # sanity floor (garden shed / single low storey)
MAX_HOUSE_HEIGHT_M = 25.0     # sanity ceiling (small apartment block) — clips bad geometry, doesn't invent it

bin_size = 12.0
xmin, xmax = roof_pts[:, 0].min(), roof_pts[:, 0].max()
ymin, ymax = roof_pts[:, 1].min(), roof_pts[:, 1].max()
nx = int((xmax - xmin) / bin_size) + 1
ny = int((ymax - ymin) / bin_size) + 1

H, xedges, yedges = np.histogram2d(roof_pts[:, 0], roof_pts[:, 1], bins=[nx, ny])

# Find local maxima peaks
peaks = []
for i in range(1, nx-1):
    for j in range(1, ny-1):
        v = H[i, j]
        if v >= 12:
            neighborhood = H[i-1:i+2, j-1:j+2]
            if v == np.max(neighborhood):
                px = float((xedges[i] + xedges[i+1]) / 2.0)
                py = float((yedges[j] + yedges[j+1]) / 2.0)
                peaks.append((px, py, int(v)))

# Suppress peaks that are too close (merge within 20m)
filtered_peaks = []
for p in sorted(peaks, key=lambda x: x[2], reverse=True):
    if not any(np.hypot(p[0]-fp[0], p[1]-fp[1]) < 18.0 for fp in filtered_peaks):
        filtered_peaks.append(p)

print(f"Detected {len(filtered_peaks)} distinct residential house locations:")

houses = []
for px, py, cnt in filtered_peaks:
    dists = np.hypot(roof_pts[:, 0] - px, roof_pts[:, 1] - py)
    lot_pts = roof_pts[dists < 20.0]
    if len(lot_pts) >= 15:
        w = float(lot_pts[:, 0].max() - lot_pts[:, 0].min())
        d = float(lot_pts[:, 1].max() - lot_pts[:, 1].min())
        w = round(max(14.0, min(32.0, w)), 1)
        d = round(max(14.0, min(32.0, d)), 1)

        # Height: derived from real geometry — roof elevation minus local ground
        # elevation, not a random guess.
        roof_top_z = float(np.percentile(lot_pts[:, 2], 95))  # robust "top of roof"

        ground_dists = np.hypot(ground_pts[:, 0] - px, ground_pts[:, 1] - py) if len(ground_pts) else np.array([])
        local_ground_pts = ground_pts[ground_dists < GROUND_SEARCH_RADIUS] if len(ground_pts) else np.empty((0, 3))
        if len(local_ground_pts) >= MIN_LOCAL_GROUND_PTS:
            ground_z = float(np.percentile(local_ground_pts[:, 2], 25))
        else:
            ground_z = global_ground_z

        h_raw = roof_top_z - ground_z
        h = round(float(np.clip(h_raw, MIN_HOUSE_HEIGHT_M, MAX_HOUSE_HEIGHT_M)), 1)

        houses.append({
            'cx': round(px, 1),
            'cy': round(py, 1),
            'w': w,
            'd': d,
            'h': h,
            'pts': len(lot_pts)
        })

print(f"Extracted {len(houses)} residential houses:")
for idx, h in enumerate(houses[:20]):
    print(f"  House {idx+1:2d}: at ({h['cx']:6.1f}, {h['cy']:6.1f}) size {h['w']}m x {h['d']}m x {h['h']}m ({h['pts']} pts)")
