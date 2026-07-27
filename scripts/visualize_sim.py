"""
visualize_sim.py  ·  把路损/视距仿真结果画到 3D 模型上（Open3D）

在材质网格上叠加：
  · 接收点"覆盖球"(按路损 jet 上色：蓝=低损耗好，红=高损耗差)
  · 发射点 TX（品红大球）
  · (可选) TX→RX 射线（绿=LoS，红=NLoS，抽样）
导出为 PLY（MeshLab 里可转着看），并可离屏渲一张预览 PNG。

用法：
  python scripts\visualize_sim.py ^
    --mesh outputs\scene01_pgsr\material_seg\material_mesh.ply ^
    --grid_npz outputs\scene01_pgsr\pathloss_seg\pathloss_grid.npz ^
    --out_dir outputs\scene01_pgsr\pathloss_seg\viz3d --rays --screenshot
在 MeshLab 里同时打开 material_mesh.ply + viz3d\sim_*.ply 即可 3D 查看。
"""
import os, argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--grid_npz", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rays", action="store_true", help="额外画 TX→RX 射线")
    ap.add_argument("--ray_stride", type=int, default=6, help="射线抽样(越大越稀)")
    ap.add_argument("--screenshot", action="store_true", help="离屏渲一张预览 PNG")
    ap.add_argument("--show", action="store_true", help="弹出交互式 Open3D 窗口(需桌面)")
    args = ap.parse_args()

    import open3d as o3d
    import matplotlib

    d = np.load(args.grid_npz)
    PL = d["pathloss"]; LOS = d["los"]; TX = d["tx"]; g = PL.shape[0]
    mesh = o3d.io.read_triangle_mesh(args.mesh); mesh.compute_vertex_normals()
    V = np.asarray(mesh.vertices); lo, hi = V.min(0), V.max(0); ext = hi - lo
    up = int(d["up_axis"]) if "up_axis" in d else int(np.argmin(ext))

    # 接收点三维坐标：优先用 npz 里存的；否则向下 raycast 找地表
    if "rx" in d:
        RX = d["rx"].reshape(-1, 3)
    else:
        ax = [i for i in range(3) if i != up]
        A0 = d["A0"].ravel(); A1 = d["A1"].ravel()
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        st = np.zeros((g*g, 3)); st[:, ax[0]] = A0; st[:, ax[1]] = A1; st[:, up] = hi[up] + 0.5*ext[up]
        dn = np.zeros(3); dn[up] = -1.0
        rr = o3d.core.Tensor(np.hstack([st, np.tile(dn, (g*g, 1))]).astype(np.float32),
                             dtype=o3d.core.Dtype.Float32)
        t = sc.cast_rays(rr)["t_hit"].numpy()
        RX = st.copy(); RX[:, up] = st[:, up] - t

    pl = PL.ravel(); los = LOS.ravel()
    valid = np.isfinite(pl)
    rx = RX[valid].copy(); plv = pl[valid]; losv = los[valid]
    rx[:, up] += 0.02 * ext[up]                       # 抬高一点，浮在地表上方

    # 覆盖点：按路损 jet 上色
    norm = (plv - plv.min()) / (np.ptp(plv) + 1e-9)
    cov = o3d.geometry.PointCloud()
    cov.points = o3d.utility.Vector3dVector(rx)
    cov.colors = o3d.utility.Vector3dVector(matplotlib.colormaps["jet"](norm)[:, :3])

    # TX 标记
    r = 0.03 * float(np.linalg.norm(ext))
    tx = o3d.geometry.TriangleMesh.create_sphere(radius=r); tx.translate(TX)
    tx.paint_uniform_color([1.0, 0.0, 1.0]); tx.compute_vertex_normals()

    os.makedirs(args.out_dir, exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(args.out_dir, "sim_coverage.ply"), cov)
    o3d.io.write_triangle_mesh(os.path.join(args.out_dir, "sim_tx.ply"), tx)
    ls = None
    if args.rays:
        sub = np.arange(len(rx))[::args.ray_stride]
        pts = np.vstack([np.tile(TX, (len(sub), 1)), rx[sub]])
        lines = np.array([[i, len(sub) + i] for i in range(len(sub))])
        lc = np.array([[0, 0.8, 0] if losv[j] >= 0.5 else [0.9, 0, 0] for j in sub])
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector(lc)
        o3d.io.write_line_set(os.path.join(args.out_dir, "sim_rays.ply"), ls)
        print(f"[导出] sim_rays.ply ({len(sub)} 条)")
    print(f"[导出] {args.out_dir}\\sim_coverage.ply / sim_tx.ply")
    print("在 MeshLab 里同时打开 material_mesh.ply + 这些 sim_*.ply 即可 3D 查看")

    if args.show:
        geoms = [mesh, cov, tx] + ([ls] if ls is not None else [])
        print("弹出交互窗口：左键旋转 / 滚轮缩放 / 右键平移，按 Q 关闭")
        o3d.visualization.draw_geometries(geoms, window_name="通信仿真 3D 可视化")

    if args.screenshot:
        try:
            W, H = 1400, 1000
            rnd = o3d.visualization.rendering.OffscreenRenderer(W, H)
            rnd.scene.set_background([1, 1, 1, 1])
            ml = o3d.visualization.rendering.MaterialRecord(); ml.shader = "defaultLit"
            mp = o3d.visualization.rendering.MaterialRecord(); mp.shader = "defaultUnlit"; mp.point_size = 9.0
            rnd.scene.add_geometry("mesh", mesh, ml)
            rnd.scene.add_geometry("cov", cov, mp)
            rnd.scene.add_geometry("tx", tx, ml)
            if ls is not None:
                mln = o3d.visualization.rendering.MaterialRecord(); mln.shader = "unlitLine"; mln.line_width = 1.5
                rnd.scene.add_geometry("rays", ls, mln)
            center = (lo + hi) / 2.0
            ax = [i for i in range(3) if i != up]
            eye = center.copy()
            eye[up] = hi[up] + 1.3 * ext[up]
            eye[ax[0]] = hi[ax[0]] + 0.7 * ext[ax[0]]
            upv = np.zeros(3); upv[up] = 1.0
            rnd.setup_camera(60.0, center.astype(np.float64), eye.astype(np.float64), upv.astype(np.float64))
            outpng = os.path.join(args.out_dir, "sim_preview.png")
            o3d.io.write_image(outpng, rnd.render_to_image())
            print(f"[导出] 预览图 {outpng}")
        except Exception as e:
            print(f"[离屏渲染失败,跳过] {repr(e)[:140]}")
            print("  → PLY 已导出，直接用 MeshLab 打开查看即可。")


if __name__ == "__main__":
    main()
