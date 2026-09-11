"""
scripts/06_mesh_and_analytics.py
AeroTwin — Mesh Generation & Analytics

Tries to generate a surface mesh from the reconstruction output.
Priority order:
  1. Dense point cloud (fused.ply from MVS) → Poisson mesh via Open3D
  2. Dense display cloud (downsampled) → Poisson mesh
  3. Sparse point cloud → alpha-shape / simple mesh

Outputs:
  data/output/<project>/aerotwin_mesh.ply   — mesh file (if generated)
  data/output/<project>/sparse_display.ply  — sparse cloud for viewer
  data/output/<project>/mesh_report.json    — stats
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def count_ply_points(ply_path: Path) -> int:
    """Read PLY header to count vertices."""
    try:
        with open(ply_path, "rb") as f:
            header = b""
            while True:
                line = f.readline()
                header += line
                if line.strip() == b"end_header":
                    break
                if b"element vertex" in line:
                    return int(line.split()[-1])
    except Exception:
        pass
    return 0


def try_open3d_mesh(ply_path: Path, output_mesh: Path, depth: int = 8):
    """Attempt Poisson surface reconstruction using Open3D."""
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(ply_path))
        if len(pcd.points) == 0:
            return None, 0

        print(f"  Open3D loaded {len(pcd.points):,} points")

        # Estimate normals if not present
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(30)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, linear_fit=False
        )
        # Remove low-density triangles (boundary artifacts)
        densities = np.asarray(densities)
        threshold = np.percentile(densities, 5)
        vertices_to_remove = densities < threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()

        faces = len(mesh.triangles)
        if faces == 0:
            return None, 0

        o3d.io.write_triangle_mesh(str(output_mesh), mesh)
        return output_mesh, faces
    except ImportError:
        print("  Open3D not available — skipping Poisson mesh")
        return None, 0
    except Exception as e:
        print(f"  Open3D mesh failed: {e}")
        return None, 0


def build_simple_ply_from_sparse(sparse_ply: Path, out_path: Path):
    """Copy sparse PLY as display cloud (no mesh)."""
    import shutil
    shutil.copy2(str(sparse_ply), str(out_path))


def process_mesh_and_analytics(project_dir: Path):
    print("=" * 60)
    print("  AEROTWIN — MESH GENERATION & ANALYTICS")
    print("=" * 60)

    t_start = time.time()
    report = {
        "project": project_dir.name,
        "dense_available": False,
        "mesh_generated": False,
        "mesh_faces": 0,
        "sparse_points": 0,
        "dense_points": 0,
        "display_cloud": None,
        "mesh_path": None,
        "processing_time_sec": 0,
    }

    # --- Priority 1: Dense fused cloud ---
    dense_paths = [
        project_dir / "dense" / "fused.ply",
        project_dir / "dense_clean.ply",
        project_dir / "dense_display.ply",
        project_dir / "dense_pointcloud_preview.ply",
    ]
    dense_ply = None
    for dp in dense_paths:
        if dp.exists() and dp.stat().st_size > 10000:
            dense_ply = dp
            print(f"  Found dense cloud: {dp.name} ({dp.stat().st_size // 1024}KB)")
            break

    # --- Sparse PLY ---
    sparse_paths = [
        project_dir / "sparse_pointcloud.ply",
        project_dir / "sparse_best_snapshot.ply",
    ]
    sparse_ply = None
    for sp in sparse_paths:
        if sp.exists() and sp.stat().st_size > 100:
            sparse_ply = sp
            print(f"  Found sparse cloud: {sp.name} ({sp.stat().st_size // 1024}KB)")
            break

    if dense_ply is None and sparse_ply is None:
        print("  WARNING: No point cloud found (neither dense nor sparse PLY).")
        print("  COLMAP reconstruction may have failed to export PLY.")
        # Try converting from binary model
        sparse_dir = project_dir / "sparse"
        colmap_exe = _find_colmap()
        if colmap_exe and sparse_dir.exists():
            candidates = []
            if (sparse_dir / "points3D.bin").exists() or (sparse_dir / "points3D.txt").exists():
                candidates.append(sparse_dir)
            for sub in sparse_dir.iterdir():
                if sub.is_dir() and ((sub / "points3D.bin").exists() or (sub / "points3D.txt").exists()):
                    candidates.append(sub)
            if candidates:
                best_candidate = max(candidates, key=lambda p: (p / "points3D.bin").stat().st_size if (p / "points3D.bin").exists() else 0)
                out_ply = project_dir / "sparse_pointcloud.ply"
                print(f"  Attempting COLMAP PLY export from model {best_candidate.name}...")
                import subprocess
                result = subprocess.run(
                    [colmap_exe, "model_converter",
                     "--input_path", str(best_candidate),
                     "--output_path", str(out_ply),
                     "--output_type", "PLY"],
                    capture_output=True, text=True, timeout=120
                )
                if result.returncode == 0 and out_ply.exists():
    sparse_ply = out_ply
    print(f"  ✓ Exported sparse PLY: {out_ply}")
else:
    print(f"  COLMAP PLY export failed: {result.stderr[-200:]}")

    # Count points
    if dense_ply:
        report["dense_available"] = True
        report["dense_points"] = count_ply_points(dense_ply)
        print(f"  Dense points: {report['dense_points']:,}")

    if sparse_ply:
        report["sparse_points"] = count_ply_points(sparse_ply)
        print(f"  Sparse points: {report['sparse_points']:,}")

    # --- Generate mesh ---
    mesh_output = project_dir / "aerotwin_mesh.ply"
    display_cloud = project_dir / "display_cloud.ply"

    if dense_ply and report["dense_points"] > 1000:
        print(f"\n  Attempting Poisson mesh from dense cloud...")
        mesh_path, faces = try_open3d_mesh(dense_ply, mesh_output, depth=8)
        if mesh_path and faces > 0:
            report["mesh_generated"] = True
            report["mesh_faces"] = faces
            report["mesh_path"] = str(mesh_output.name)
            print(f"  ✓ Mesh generated: {faces:,} faces → {mesh_output.name}")
        build_simple_ply_from_sparse(dense_ply, display_cloud)
        report["display_cloud"] = str(display_cloud.name)

    elif sparse_ply and report["sparse_points"] > 10:
        print(f"\n  Dense cloud unavailable — using sparse PLY for display")
        build_simple_ply_from_sparse(sparse_ply, display_cloud)
        report["display_cloud"] = str(display_cloud.name)

        # Try lightweight mesh from sparse
        if report["sparse_points"] > 50:
            print(f"  Attempting Poisson mesh from sparse cloud (depth=6)...")
            mesh_path, faces = try_open3d_mesh(sparse_ply, mesh_output, depth=6)
            if mesh_path and faces > 0:
                report["mesh_generated"] = True
                report["mesh_faces"] = faces
                report["mesh_path"] = str(mesh_output.name)
                print(f"  ✓ Sparse mesh generated: {faces:,} faces")
    else:
        print("  No usable point cloud available for mesh generation.")

    elapsed = round(time.time() - t_start, 2)
    report["processing_time_sec"] = elapsed

    # Save report
    report_path = project_dir / "mesh_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")

    print(f"  Total time: {elapsed}s")
    print("=" * 60)
    return report


def _find_colmap():
    import shutil
    p = shutil.which("colmap")
    if p:
        return p
    local_paths = [
        Path("tools/bin/colmap.exe"),
        Path("tools/COLMAP.bat"),
    ]
    for lp in local_paths:
        if lp.exists():
            return str(lp.resolve())
    return None


def main():
    parser = argparse.ArgumentParser(
        description="AeroTwin: Generate mesh and analytics from reconstruction output"
    )
    parser.add_argument("--project", "-p", default="sample_flight",
                        help="Project name")
    parser.add_argument("--output-base", default="data/output",
                        help="Base output directory")
    args = parser.parse_args()

    project_dir = Path(args.output_base) / args.project
    if not project_dir.exists():
        print(f"ERROR: Project directory not found: {project_dir}")
        sys.exit(1)

    process_mesh_and_analytics(project_dir)
    # Never sys.exit(1) here — allow pipeline to continue even on failure


if __name__ == "__main__":
    main()
