r"""
clean_gaussians.py  ·  清理 PGSR/3DGS 高斯点云（NaN / 离群 / 漂浮 / 低透明度）

关键区别（与用 open3d 读写的天真做法相反）：
  PGSR 的 point_cloud.ply 是【高斯】ply，每个点带 f_dc_* / f_rest_* / opacity /
  scale_* / rot_* 等属性。若用 o3d.io.read_point_cloud 读再写，会【只保留 xyz】，
  把所有高斯属性丢光——写出来的 ply 无法再被 render.py 的 GaussianModel.load_ply 读回。
  本脚本用 plyfile 读取整个 vertex 结构化数组，对行做布尔掩码后【原样写回】，
  保留全部属性，因此清理后的 ply 可直接覆盖 point_cloud.ply 重新出网格。

用法（在 E:\lowalt3d 下）：
  # 1) 先 dry-run 看会删多少、看俯视高度图定位漂浮区
  python scripts\clean_gaussians.py ^
    --ply outputs\scene02_pgsr_hq\point_cloud\iteration_30000\point_cloud.ply ^
    --cameras outputs\scene02_pgsr_hq\cameras.json ^
    --map outputs\scene02_pgsr_hq\cleaned_map.png --dry_run

  # 2) 确认后实际清理（写到 _cleaned.ply，不覆盖原文件）
  python scripts\clean_gaussians.py ^
    --ply outputs\scene02_pgsr_hq\point_cloud\iteration_30000\point_cloud.ply ^
    --cameras outputs\scene02_pgsr_hq\cameras.json ^
    --out outputs\scene02_pgsr_hq\point_cloud\iteration_30000\point_cloud_cleaned.ply
"""
import os
import json
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="清理高斯点云（保留全部高斯属性）")
    ap.add_argument("--ply", required=True, help="输入高斯 point_cloud.ply")
    ap.add_argument("--cameras", required=True, help="cameras.json（判定覆盖区 bbox / 竖直轴）")
    ap.add_argument("--out", default=None, help="输出清理后的高斯 ply（不填=只统计）")
    ap.add_argument("--map", default=None, help="导出俯视高度图（原始 vs 清理后，辅助定位漂浮区）")
    ap.add_argument("--dry_run", action="store_true", help="只统计、不写出")
    ap.add_argument("--outlier_margin", type=float, default=10.0,
                    help="覆盖区外离群阈值：超出相机 bbox 外 N 倍 bbox 跨度即判离群（默认 10）")
    ap.add_argument("--z_outlier_sigma", type=float, default=3.0,
                    help="竖直方向漂浮离群：均值 + N×std 以上判漂浮（默认 3）")
    ap.add_argument("--min_opacity", type=float, default=None,
                    help="可选：不透明度(sigmoid 后)低于该值的高斯判漂浮薄壳（如 0.02）；不填=不启用")
    ap.add_argument("--cut_above", default=None,
                    help="可选定点切除：X,Y,R,H（水平圆心+半径内、竖直 H 以上全切）")
    args = ap.parse_args()

    from plyfile import PlyData, PlyElement

    # ---- 读高斯 ply（保留整个 vertex 结构化数组）----
    ply = PlyData.read(args.ply)
    v = ply["vertex"]
    data = v.data                                   # 结构化 numpy 数组，含全部属性
    names = data.dtype.names
    N = len(data)
    pts = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    print(f"原始高斯数：{N:,}   属性列：{len(names)} 项")

    # ---- 读相机，定覆盖区 bbox 与竖直轴 ----
    cams = json.load(open(args.cameras, encoding="utf-8"))
    cam_pos = np.array([c["position"] for c in cams], dtype=np.float64)
    cam_lo, cam_hi = cam_pos.min(0), cam_pos.max(0)
    cam_ext = np.maximum(cam_hi - cam_lo, 1e-6)
    up = int(np.argmin(cam_ext))                    # 跨度最小的轴 = 竖直轴
    sgn = 1.0 if cam_pos[:, up].mean() > (cam_lo[up] + cam_hi[up]) / 2 else -1.0
    print(f"相机覆盖区：{cam_lo.round(2)} ~ {cam_hi.round(2)}  竖直轴={'xyz'[up]}  相机侧 sgn={sgn:+.0f}")

    keep = np.ones(N, dtype=bool)

    # ① NaN / Inf（检查所有 float 列，任一非有限即删）
    bad = np.zeros(N, dtype=bool)
    for nm in names:
        col = data[nm]
        if np.issubdtype(col.dtype, np.floating):
            bad |= ~np.isfinite(col)
    keep &= ~bad
    print(f"① NaN/Inf                 ：{bad.sum():>8,} ({bad.sum()/N*100:5.2f}%)")

    # ② 覆盖区外离群（xyz 超出相机 bbox 外 margin×跨度）
    margin = cam_ext * args.outlier_margin
    outlier = ((pts < cam_lo - margin) | (pts > cam_hi + margin)).any(1)
    keep &= ~outlier
    print(f"② 覆盖区外离群({args.outlier_margin:g}×)      ：{outlier.sum():>8,} ({outlier.sum()/N*100:5.2f}%)")

    # ③ 竖直方向漂浮离群（相机侧远超主体 均值+Nσ）
    z = pts[:, up]
    z_thresh = z.mean() + sgn * args.z_outlier_sigma * z.std()
    high = (sgn * z) > (sgn * z_thresh)
    keep &= ~high
    print(f"③ 竖直漂浮离群(>{z_thresh:8.2f})   ：{high.sum():>8,} ({high.sum()/N*100:5.2f}%)")

    # ④ 低不透明度薄壳（可选；opacity 存的是 logit，需 sigmoid）
    if args.min_opacity is not None and "opacity" in names:
        op = 1.0 / (1.0 + np.exp(-data["opacity"].astype(np.float64)))
        low = op < args.min_opacity
        keep &= ~low
        print(f"④ 低不透明度(<{args.min_opacity:g})       ：{low.sum():>8,} ({low.sum()/N*100:5.2f}%)")

    # ⑤ 定点切除（可选）
    if args.cut_above:
        cx, cy, r, h = map(float, args.cut_above.split(","))
        ax = [i for i in range(3) if i != up]
        dist = np.hypot(pts[:, ax[0]] - cx, pts[:, ax[1]] - cy)
        cut = (dist < r) & ((sgn * z) > (sgn * h))
        keep &= ~cut
        print(f"⑤ 定点切除(圆心({cx:g},{cy:g}) R={r:g} H>{h:g})：{cut.sum():>8,} ({cut.sum()/N*100:5.2f}%)")

    kept = int(keep.sum())
    print(f"\n保留：{kept:,} / {N:,} ({kept/N*100:.2f}%)   删除：{N-kept:,} ({(N-kept)/N*100:.2f}%)")

    # ---- 俯视高度图（原始 vs 清理后）----
    if args.map:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            ax_h = [i for i in range(3) if i != up]
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
            s1 = a1.scatter(pts[:, ax_h[0]], pts[:, ax_h[1]], c=z, s=0.1, cmap="jet")
            a1.set_title(f"俯视高度图·原始 ({N:,})"); a1.set_aspect("equal"); fig.colorbar(s1, ax=a1)
            s2 = a2.scatter(pts[keep, ax_h[0]], pts[keep, ax_h[1]], c=z[keep], s=0.1, cmap="jet")
            a2.set_title(f"俯视高度图·清理后 ({kept:,})"); a2.set_aspect("equal"); fig.colorbar(s2, ax=a2)
            plt.tight_layout(); plt.savefig(args.map, dpi=150); plt.close()
            print(f"[导出] {args.map}")
        except Exception as e:
            print(f"导出高度图失败：{repr(e)[:120]}")

    # ---- 写出（保留全部高斯属性）----
    if args.dry_run or not args.out:
        print("[Dry-run] 未写出。确认后去掉 --dry_run 并加 --out <路径> 实际清理。")
        return
    out_el = PlyElement.describe(data[keep], "vertex")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    PlyData([out_el], text=False).write(args.out)
    print(f"[写出] {args.out}（{kept:,} 高斯，属性完整，可覆盖 point_cloud.ply 重新出网格）")


if __name__ == "__main__":
    main()
