r"""
compare_engines.py · 三引擎覆盖对比：Python(simple_pathloss) vs UE(RFRayTracer) [vs InSite]
把 UE 导出的 coverage_ue.csv 重采样到 Python 的网格，画 并排热图 + 差值图 + 统计。
用于毕设“同一材质电磁地图、不同射线引擎结果一致性”验证。

运行：
  python scripts\compare_engines.py ^
    --py outputs\%SCENE%_pgsr\pathloss_gauss_cons\pathloss_grid.npz ^
    --ue outputs\%SCENE%_pgsr\ue\coverage_ue.csv ^
    --out_dir outputs\%SCENE%_pgsr\compare
说明：两引擎坐标系不同(Python=COLMAP轴, UE=米制ENU厘米)，本脚本按各自网格的
     归一化平面位置对齐(双线性重采样)，比较“相对分布”；绝对 dB 已由 --scale/配准统一到米制。
"""
import os, sys, argparse
import numpy as np


def load_ue(path):
    """coverage_ue.csv: x_cm,y_cm,z_cm,los,pathloss_db,prx_dbm → 规则网格重建。"""
    rec = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    x = np.asarray(rec["x_cm"], float); y = np.asarray(rec["y_cm"], float)
    pl = np.asarray(rec["pathloss_db"], float); los = np.asarray(rec["los"], float)
    # UE 网格是规则 NxN，用唯一坐标还原
    ux = np.unique(np.round(x, 1)); uy = np.unique(np.round(y, 1))
    gx, gy = len(ux), len(uy)
    PL = np.full((gy, gx), np.nan); LO = np.full((gy, gx), np.nan)
    xi = {v: i for i, v in enumerate(ux)}; yi = {v: i for i, v in enumerate(uy)}
    for k in range(len(x)):
        i = yi[round(y[k], 1)]; j = xi[round(x[k], 1)]
        PL[i, j] = pl[k]; LO[i, j] = los[k]
    return PL, LO


def resample(A, shape):
    """把 A 双线性重采样到 shape（忽略 NaN 用最近有效值）。"""
    from numpy import isfinite
    ys = np.linspace(0, A.shape[0] - 1, shape[0])
    xs = np.linspace(0, A.shape[1] - 1, shape[1])
    Yi, Xi = np.meshgrid(ys, xs, indexing="ij")
    y0 = np.floor(Yi).astype(int); x0 = np.floor(Xi).astype(int)
    y1 = np.minimum(y0 + 1, A.shape[0] - 1); x1 = np.minimum(x0 + 1, A.shape[1] - 1)
    B = A.copy()
    if np.isnan(B).any():                       # NaN 用列均值兜底，避免插值污染
        m = np.nanmean(B); B = np.where(np.isnan(B), m, B)
    wa = (y1 - Yi) * (x1 - Xi); wb = (y1 - Yi) * (Xi - x0)
    wc = (Yi - y0) * (x1 - Xi); wd = (Yi - y0) * (Xi - x0)
    return wa * B[y0, x0] + wb * B[y0, x1] + wc * B[y1, x0] + wd * B[y1, x1]


def save(path, arr, title, cmap="jet", vmin=None, vmax=None):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 5))
        plt.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(); plt.title(title); plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
    except Exception as e:
        print(f"[warn] 画图失败({e})，跳过 {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", required=True, help="Python 的 pathloss_grid.npz")
    ap.add_argument("--ue", required=True, help="UE 的 coverage_ue.csv")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    d = np.load(args.py, allow_pickle=True)
    PL_py = d["pathloss"].astype(float)
    UE_PL, _ = load_ue(args.ue)
    print(f"Python 网格 {PL_py.shape}，UE 网格 {UE_PL.shape}")

    UE_rs = resample(UE_PL, PL_py.shape)
    valid = np.isfinite(PL_py) & np.isfinite(UE_rs)
    diff = np.where(valid, UE_rs - PL_py, np.nan)

    os.makedirs(args.out_dir, exist_ok=True)
    lo = float(np.nanmin([np.nanmin(PL_py), np.nanmin(UE_rs)]))
    hi = float(np.nanmax([np.nanmax(PL_py), np.nanmax(UE_rs)]))
    save(os.path.join(args.out_dir, "py_pathloss.png"), PL_py, "Python simple_pathloss (dB)", vmin=lo, vmax=hi)
    save(os.path.join(args.out_dir, "ue_pathloss.png"), UE_rs, "UE RFRayTracer (dB)", vmin=lo, vmax=hi)
    save(os.path.join(args.out_dir, "diff.png"), diff, "UE - Python (dB)", cmap="bwr", vmin=-20, vmax=20)

    if valid.any():
        dd = diff[valid]
        print(f"差值(UE-Python): 均值 {dd.mean():.2f} dB，绝对均值 {np.abs(dd).mean():.2f} dB，"
              f"RMSE {np.sqrt((dd**2).mean()):.2f} dB")
        a, b = PL_py[valid], UE_rs[valid]
        if a.std() > 1e-6 and b.std() > 1e-6:
            print(f"相关系数 r = {np.corrcoef(a, b)[0,1]:.3f}")
    print(f"[导出] {args.out_dir}\\py_pathloss.png / ue_pathloss.png / diff.png")


if __name__ == "__main__":
    main()
