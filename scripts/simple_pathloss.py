"""
simple_pathloss.py  ·  Step 5：电磁参数映射 + 射线路损（P0 简化版）

目的：用带材质的网格，算出一张"发射点→各接收点"的路径损耗热力图 + 视距(LoS)图。
原理：对每个接收点 RX，从发射点 TX 发射线：
      · 未被遮挡 → 视距，自由空间损耗 L=20log10(d_km)+20log10(f_MHz)+32.44
      · 被遮挡   → 非视距，自由空间损耗 + 沿途穿透的各面片材质穿透损耗(ITU-R P.2040)累加
      （P0 只算 直射+穿透，忽略反射/绕射；反射版后续再加。）

⚠️ 尺度：COLMAP 坐标是任意单位，不是米。绝对 dB 只有在 --scale(米/单位) 正确后才可信；
   默认 scale=1，此时热力图的"相对分布"(遮挡区更高)有意义，绝对值待地理配准后再定。
⚠️ 竖直轴：COLMAP 世界朝向任意，本脚本默认把 bbox 最小跨度的轴当竖直(航拍近似)，
   可用 --up_axis 指定。若热力图方位不对，多半是这里。

用法：
  python scripts\simple_pathloss.py ^
    --mesh outputs\scene01_pgsr\mesh\tsdf_fusion_post.ply ^
    --materials outputs\scene01_pgsr\material\face_materials.csv ^
    --radio_params configs\radio_params.yaml ^
    --out_dir outputs\scene01_pgsr\pathloss --grid 60
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from material_common import load_radio_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--materials", required=True, help="face_materials.csv (Step4 产物)")
    ap.add_argument("--radio_params", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "radio_params.yaml"))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--grid", type=int, default=60, help="接收点网格 NxN")
    ap.add_argument("--scale", type=float, default=1.0, help="米/COLMAP单位(地理配准后填真实值)")
    ap.add_argument("--up_axis", choices=["auto", "x", "y", "z"], default="auto")
    ap.add_argument("--cameras", default=None, help="cameras.json；用相机位置自动判定天空侧(强烈建议)")
    ap.add_argument("--rx_height_frac", type=float, default=0.03, help="接收点在地表上方高度=该比例*竖直跨度")
    ap.add_argument("--tx_up_frac", type=float, default=1.4, help="发射点高度=下界+该比例*竖直跨度")
    args = ap.parse_args()

    import open3d as o3d

    # ---- 频率 + 每面片电磁参数 ----
    cfg = {}
    try:
        import yaml; cfg = yaml.safe_load(open(args.radio_params, encoding="utf-8"))
    except Exception:
        pass
    freq_mhz = float(cfg.get("frequency_mhz", 3500))
    print(f"频率: {freq_mhz} MHz, 尺度: {args.scale} 米/单位")

    m = np.genfromtxt(args.materials, delimiter=",", names=True, dtype=None, encoding="utf-8")
    face_loss = np.asarray(m["penetration_loss_db"], np.float32)   # 每面片穿透损耗(dB)
    nF = len(face_loss)
    print(f"面片数: {nF}")

    # ---- 网格 + 光线场景 ----
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    INVALID = o3d.t.geometry.RaycastingScene.INVALID_ID
    V = np.asarray(mesh.vertices)
    lo, hi = V.min(0), V.max(0); ext = hi - lo

    # ---- 定竖直轴 + 两个水平轴 ----
    up = int(np.argmin(ext)) if args.up_axis == "auto" else {"x":0,"y":1,"z":2}[args.up_axis]
    ax = [i for i in range(3) if i != up]       # 两个水平轴
    print(f"竖直轴 = {'xyz'[up]}，水平轴 = {'xyz'[ax[0]]}{'xyz'[ax[1]]}，场景跨度 {ext.round(2)}")

    # ---- 确定"天空侧"：相机(无人机)一定在天空侧；据此定 up 的正负向 ----
    sgn = 1.0                                          # +1: 天空在 up 高处; -1: 在低处
    if args.cameras and os.path.isfile(args.cameras):
        import json
        cpos = np.array([c["position"] for c in json.load(open(args.cameras, encoding="utf-8"))])
        if cpos[:, up].mean() < (lo[up] + hi[up]) / 2:
            sgn = -1.0
        print(f"相机在 up 的{'高' if sgn > 0 else '低'}处 → 天空侧 sgn={sgn:+.0f}")
    sky = hi[up] if sgn > 0 else lo[up]                # 天空侧端点

    # ---- RX 网格：从天空侧上方向地面投射找地表 → 接收点放到地表上方一点 ----
    g = args.grid
    a0 = np.linspace(lo[ax[0]], hi[ax[0]], g)
    a1 = np.linspace(lo[ax[1]], hi[ax[1]], g)
    A0, A1 = np.meshgrid(a0, a1)
    top = sky + sgn * 0.5 * ext[up]                    # 天空侧上方起点
    down = np.zeros(3); down[up] = -sgn                # 朝地面
    starts = np.zeros((g * g, 3))
    starts[:, ax[0]] = A0.ravel(); starts[:, ax[1]] = A1.ravel(); starts[:, up] = top
    dr = o3d.core.Tensor(np.hstack([starts, np.tile(down, (g*g, 1))]).astype(np.float32),
                         dtype=o3d.core.Dtype.Float32)
    dt = scene.cast_rays(dr)["t_hit"].numpy()
    surf = top - sgn * dt                              # 地表在 up 轴的坐标
    valid = np.isfinite(dt)
    RX = starts.copy()
    RX[valid, up] = surf[valid] + sgn * args.rx_height_frac * ext[up]   # 地表上方(天空侧)
    TX = np.zeros(3)
    TX[ax[0]] = (lo[ax[0]] + hi[ax[0]]) / 2; TX[ax[1]] = (lo[ax[1]] + hi[ax[1]]) / 2
    TX[up] = sky + sgn * (args.tx_up_frac - 1.0) * ext[up]   # TX 在天空侧上方(模型上面)
    print(f"TX = {TX.round(2)}，RX 网格 {g}x{g}，有效接收点 {valid.sum()}/{g*g}")

    # ---- 自由空间损耗(米) ----
    d_unit = np.linalg.norm(RX - TX, axis=1)          # COLMAP 单位距离
    d_km = np.maximum(d_unit * args.scale, 1e-3) / 1000.0
    fspl = 20 * np.log10(d_km) + 20 * np.log10(freq_mhz) + 32.44   # [g*g]

    # ---- 第一遍(向量化)：判 LoS/NLoS ----
    dirs = (RX - TX); dist = np.linalg.norm(dirs, axis=1); dirs = dirs / dist[:, None]
    rays = o3d.core.Tensor(np.hstack([np.tile(TX, (g*g, 1)), dirs]).astype(np.float32),
                           dtype=o3d.core.Dtype.Float32)
    t_hit = scene.cast_rays(rays)["t_hit"].numpy()
    blocked = (t_hit < (dist - 1e-3)) & valid          # 首个命中在 RX 之前 → 被遮挡
    print(f"NLoS(遮挡) 比例: {blocked[valid].mean()*100:.1f}% (有效点内)")

    # ---- 第二遍：被遮挡点逐个累加穿透损耗 ----
    pen = np.zeros(g * g, np.float32)
    eps = ext[up] * 1e-3 + 1e-5
    blk_idx = np.where(blocked)[0]
    for k, i in enumerate(blk_idx):
        origin = TX.copy(); dir_i = dirs[i]; remain = dist[i]; total = 0.0
        for _ in range(30):
            r = o3d.core.Tensor([[*origin, *dir_i]], dtype=o3d.core.Dtype.Float32)
            a = scene.cast_rays(r)
            t = float(a["t_hit"].numpy()[0]); fid = int(a["primitive_ids"].numpy()[0])
            if not np.isfinite(t) or t >= remain - eps:
                break
            total += face_loss[fid]
            origin = origin + dir_i * (t + eps); remain -= (t + eps)
        pen[i] = total
        if (k + 1) % max(1, len(blk_idx)//5) == 0:
            print(f"  穿透累加 {k+1}/{len(blk_idx)}")

    pathloss = fspl + pen                              # 总路损 dB
    pathloss[~valid] = np.nan                          # 无地表处标无效
    PL = pathloss.reshape(g, g)
    LOS = (~blocked).astype(float); LOS[~valid] = np.nan; LOS = LOS.reshape(g, g)

    # ---- 导出 ----
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(os.path.join(args.out_dir, "pathloss_grid.npz"),
             A0=A0, A1=A1, pathloss=PL, los=LOS, tx=TX, rx=RX, up_axis=up,
             valid=valid.reshape(g, g), scale=args.scale, freq_mhz=freq_mhz)
    _save_heatmap(os.path.join(args.out_dir, "pathloss_heatmap.png"), PL,
                  f"Path Loss (dB) @ {freq_mhz:.0f}MHz  scale={args.scale}", cmap="jet")
    _save_heatmap(os.path.join(args.out_dir, "los_map.png"), LOS.astype(float),
                  "LoS(1) / NLoS(0)", cmap="gray")
    print(f"\n路损范围: {np.nanmin(PL):.1f} ~ {np.nanmax(PL):.1f} dB")
    print(f"[导出] {args.out_dir}\\pathloss_heatmap.png / los_map.png / pathloss_grid.npz")
    if args.scale == 1.0:
        print("[注意] scale=1（未配准），绝对 dB 仅供参考；相对分布(遮挡区更高)有效。")


def _save_heatmap(path, arr, title, cmap="jet"):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 5))
        plt.imshow(arr, origin="lower", cmap=cmap); plt.colorbar(); plt.title(title)
        plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
    except Exception:
        import cv2
        a = arr.astype(np.float32)
        amin, amax = float(np.nanmin(a)), float(np.nanmax(a))
        a = np.nan_to_num((a - amin) / (amax - amin + 1e-9), nan=0.0)
        img = cv2.applyColorMap((a * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(path, cv2.flip(img, 0))


if __name__ == "__main__":
    main()
