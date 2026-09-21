"""Apply the scene's GPS alignment to a PLY for upright visualization.

The source PGSR/COLMAP PLY stays untouched. Y-up is for Three.js/glTF viewers;
Z-up is for MATLAB/Open3D/ENU viewers. Geometry and vertex colors are preserved.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def contains_faces(path):
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PLY header is incomplete: {path}")
            if line.startswith(b"element face "):
                return int(line.split()[2]) > 0
            if line.strip() == b"end_header":
                return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--geo_transform", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--up_axis", choices=("y", "z"), default="y")
    args = ap.parse_args()

    if args.output.exists():
        ap.error(f"Output already exists: {args.output}")
    geo = json.loads(args.geo_transform.read_text(encoding="utf-8"))
    scale = float(geo["scale"])
    rotation = np.asarray(geo["R"], dtype=np.float64)
    translation = np.asarray(geo["t"], dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,) or scale <= 0:
        ap.error("Invalid geo_transform dimensions or scale")

    is_mesh = contains_faces(args.input)
    geometry = (o3d.io.read_triangle_mesh(str(args.input)) if is_mesh
                else o3d.io.read_point_cloud(str(args.input)))
    vertices = np.asarray(geometry.vertices if is_mesh else geometry.points)
    enu = scale * (vertices @ rotation.T) + translation
    upright = (np.column_stack((enu[:, 0], enu[:, 2], -enu[:, 1]))
               if args.up_axis == "y" else enu)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if is_mesh:
        geometry.vertices = o3d.utility.Vector3dVector(upright)
        geometry.compute_vertex_normals()
        ok = o3d.io.write_triangle_mesh(str(args.output), geometry)
    else:
        geometry.points = o3d.utility.Vector3dVector(upright)
        if geometry.has_normals():
            normals = np.asarray(geometry.normals) @ rotation.T
            if args.up_axis == "y":
                normals = np.column_stack((normals[:, 0], normals[:, 2], -normals[:, 1]))
            geometry.normals = o3d.utility.Vector3dVector(normals)
        ok = o3d.io.write_point_cloud(str(args.output), geometry)
    if not ok:
        raise RuntimeError(f"Failed to write {args.output}")
    print(f"Wrote {args.output} ({len(vertices)} vertices, {args.up_axis.upper()} up, ENU meters)")


if __name__ == "__main__":
    main()
