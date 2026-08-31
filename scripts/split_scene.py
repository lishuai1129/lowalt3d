r"""
split_scene.py  ·  大场景分块：一次全局 SfM → N 个可独立训练的 COLMAP 子模型

为什么要分块（scene03 = 1202 张 45MP，ENU 足迹 444×342 m）：
  单块跑必须靠 `--max_all_points 3000000 --max_abs_split_points 0 --opacity_cull_threshold 0.05`
  压显存，而这三个旋钮是【拿植被完整度换显存】。分块后每块都能跑 PGSR 默认稠密化，
  N 块 = N × 6M 高斯预算。产物合并回【一个高斯集 + 一个网格】，下游一行不改。

铁律：**只做一次全局 SfM**。每块单独跑 COLMAP 会得到互不兼容的坐标系，根本无法合并。

方法 = VastGaussian 四步，逐条对应（实测依据见 --note）：
  1. 按【相机数】均衡切网格（不是等面积——等面积在 scene02 上实测 2×2 得到 102/55/552/256）
  2. core / expanded 双框，expanded = core 外扩 --overlap
  3. 相机选取 = 中心在 expanded 内 ∪ 可见度≥--vis_thresh（该图观测点落在 core 内的比例）
  4. 点选取 = 落在 expanded 内的点

分块在 **ENU 米坐标**里做（复用 georeference.py 的 GPS→ENU + Umeyama）：上轴天然是 +U，
不必猜 COLMAP 的重力方向；块尺寸直接以米给定，可解释。

用法（在 E:\lowalt3d 下）：
  # 1) 先 dry-run 看每块相机数/点数/冗余度，不写盘。不满意就调 --grid，这一步免费
  python scripts\split_scene.py --sparse data\scene03\sparse --images data\scene03\input ^
      --out data\scene03\blocks --grid 3x2 --dry_run

  # 2) 确认后写出子模型
  python scripts\split_scene.py --sparse data\scene03\sparse --images data\scene03\input ^
      --out data\scene03\blocks --grid 3x2

  # 3) 逐块训练（-i 用 ../../images_2 指回共享图目录，os.path.join 会解析 ..）
  cd repos\pgsr
  python train.py -s ..\..\data\scene03\blocks\b00 -m ..\..\outputs\scene03_b00 ^
      -i ..\..\images_2 -r 2 --iterations 30000 --data_device cpu

  python scripts\split_scene.py --selfcheck     # 秒级自检

注意：
  · 写出的 images.txt 第二行（2D 观测）与 points3D.txt 的 track 都【留空】——PGSR 只用
    外参/内参 + 点坐标，观测表不读（见 scene/colmap_loader.py）。省每块数百 MB。
    代价：子模型不再是 COLMAP 自己可加载的完整模型，只喂 PGSR。自检会断言 PGSR 读得回。
  · 点先按 PGSR 自己的质量过滤（error≤2 且 track≥3，见 read_points3D_binary）筛一遍，
    保证块内初始点云 == 全局跑时会用的那批点的子集。
"""
import os
import sys
import json
import struct
import argparse

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# COLMAP 相机模型 → 参数个数（读 cameras.bin 要靠它定长）
_MODEL_NP = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
             3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
             6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
             9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12)}


# ---------------------------------------------------------------- COLMAP 二进制读取
def read_cameras_bin(path):
    """→ {cam_id: (model_name, w, h, params[])}"""
    out = {}
    with open(path, "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            cid, mid, w, h = struct.unpack("<iiQQ", f.read(24))
            name, npar = _MODEL_NP[mid]
            out[cid] = (name, w, h, np.array(struct.unpack("<" + "d" * npar, f.read(8 * npar))))
    return out


def read_images_bin(path):
    """→ [{id,q,t,cam,name,pids}]，pids 为该图观测到的 point3D_id（-1 表示无对应）"""
    _obs = np.dtype([("x", "<f8"), ("y", "<f8"), ("pid", "<i8")])
    out = []
    with open(path, "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            v = struct.unpack("<idddddddi", f.read(64))
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            m = struct.unpack("<Q", f.read(8))[0]
            pids = np.frombuffer(f.read(24 * m), dtype=_obs, count=m)["pid"].copy()
            out.append(dict(id=v[0], q=np.array(v[1:5]), t=np.array(v[5:8]),
                            cam=v[8], name=name.decode("utf-8"), pids=pids))
    return out


def read_points3d_bin(path):
    """→ pid[N], xyz[N,3], rgb[N,3], err[N], track_len[N]"""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        pid = np.empty(n, np.int64)
        xyz = np.empty((n, 3))
        rgb = np.empty((n, 3), np.uint8)
        err = np.empty(n)
        tlen = np.empty(n, np.int64)
        for i in range(n):
            a = struct.unpack("<QdddBBBd", f.read(43))
            pid[i] = a[0]
            xyz[i] = a[1:4]
            rgb[i] = a[4:7]
            err[i] = a[7]
            k = struct.unpack("<Q", f.read(8))[0]
            f.seek(8 * k, 1)
            tlen[i] = k
    return pid, xyz, rgb, err, tlen


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# ---------------------------------------------------------------- COLMAP 文本写出
def write_model_txt(out_dir, cams, imgs, pid, xyz, rgb, err):
    """写 PGSR 能读的最小 COLMAP 文本模型（观测表/track 留空，见模块 docstring）。"""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cameras.txt"), "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n"
                "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cid, (model, w, h, par) in sorted(cams.items()):
            f.write(f"{cid} {model} {w} {h} " + " ".join(repr(float(p)) for p in par) + "\n")

    with open(os.path.join(out_dir, "images.txt"), "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n"
                "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
                "#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for im in imgs:
            q = " ".join(repr(float(x)) for x in im["q"])
            t = " ".join(repr(float(x)) for x in im["t"])
            f.write(f"{im['id']} {q} {t} {im['cam']} {im['name']}\n\n")

    with open(os.path.join(out_dir, "points3D.txt"), "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n"
                "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        # numpy 2.x 的 repr(np.float64) 是 "np.float64(1.0)"，必须先转 python float
        for i in range(len(pid)):
            f.write(f"{int(pid[i])} {float(xyz[i,0])!r} {float(xyz[i,1])!r} {float(xyz[i,2])!r} "
                    f"{int(rgb[i,0])} {int(rgb[i,1])} {int(rgb[i,2])} {float(err[i])!r}\n")


# ---------------------------------------------------------------- 分块几何
def balanced_grid(xy, nx, ny, bbox):
    """按【相机数】均衡切网格（VastGaussian 渐进式划分）。

    先按 x 排序切 nx 列（每列相机数相等），每列内再按 y 排序切 ny 格。
    列/行边界取相邻两点中点；最外圈边界贴 bbox，保证并集覆盖全场景、不漏。
    返回 [(x0,x1,y0,y1), ...]，长度 nx*ny，行优先。
    """
    x0b, y0b, x1b, y1b = bbox

    def cuts(v, k):
        """把 v 切成 k 份等数量，返回 k+1 个边界（首尾由调用方替换成 bbox）。"""
        s = np.sort(v)
        b = [None]
        for i in range(1, k):
            j = int(round(i * len(s) / k))
            j = min(max(j, 1), len(s) - 1)
            b.append(float((s[j - 1] + s[j]) / 2))
        b.append(None)
        return b

    xb = cuts(xy[:, 0], nx)
    xb[0], xb[-1] = x0b, x1b
    cells = []
    for i in range(nx):
        m = (xy[:, 0] >= xb[i]) & (xy[:, 0] < xb[i + 1] if i < nx - 1 else xy[:, 0] <= xb[i + 1])
        col = xy[m, 1] if m.any() else xy[:, 1]
        yb = cuts(col, ny)
        yb[0], yb[-1] = y0b, y1b
        for j in range(ny):
            cells.append((xb[i], xb[i + 1], yb[j], yb[j + 1]))
    return cells


def expand(cell, overlap, ref=None):
    """core → expanded：按格子尺寸外扩 overlap 比例。

    ref 给定时按 ref 的尺寸算外扩量 —— 最外圈 core 会被拉到全场景 bbox（可能被 SfM
    离群点撑到上千米），拿它算外扩会让 expanded 吞掉整个场景（scene02 实测冗余度 3.34×）。
    """
    x0, x1, y0, y1 = cell
    r = ref if ref is not None else cell
    dx, dy = (r[1] - r[0]) * overlap, (r[3] - r[2]) * overlap
    return (x0 - dx, x1 + dx, y0 - dy, y1 + dy)


def snap_outer(cell, inner_bbox, outer_bbox):
    """最外圈边界从鲁棒 bbox 拉到全场景 bbox，保证 core 的并集不漏任何内容。"""
    x0b, y0b, x1b, y1b = inner_bbox
    X0, Y0, X1, Y1 = outer_bbox
    x0, x1, y0, y1 = cell
    return (X0 if x0 == x0b else x0, X1 if x1 == x1b else x1,
            Y0 if y0 == y0b else y0, Y1 if y1 == y1b else y1)


def _inbox(xy, box):
    x0, x1, y0, y1 = box
    return (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)


# ---------------------------------------------------------------- 自检
def _selfcheck():
    import tempfile
    print("[selfcheck] 分块几何 + COLMAP 文本模型往返")

    # 6x4 相机网格 + 平面点云
    gx, gy = np.meshgrid(np.linspace(0, 300, 12), np.linspace(0, 200, 8))
    cam_xy = np.stack([gx.ravel(), gy.ravel()], 1)
    px, py = np.meshgrid(np.linspace(0, 300, 40), np.linspace(0, 200, 30))
    pts = np.stack([px.ravel(), py.ravel()], 1)
    bbox = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())

    cells = balanced_grid(cam_xy, 3, 2, bbox)
    assert len(cells) == 6
    cnt = [int(_inbox(cam_xy, c).sum()) for c in cells]
    assert max(cnt) - min(cnt) <= 2, f"相机数不均衡: {cnt}"          # 均衡划分
    cov = np.zeros(len(pts), bool)
    for c in cells:
        cov |= _inbox(pts, c)
    assert cov.all(), "core 并集没覆盖全部点"                          # 无洞
    for c in cells:
        e = expand(c, 0.2)
        assert e[0] <= c[0] and e[1] >= c[1] and e[2] <= c[2] and e[3] >= c[3]  # expanded 包含 core
    print(f"  OK 6 块相机数 {cnt}，core 并集覆盖全部点，expanded 包含 core")

    # 离群点把全量 bbox 撑大 10 倍时，外扩量必须仍按鲁棒格尺寸算（scene02 实测踩过）
    outer = (-2000.0, -2000.0, 3000.0, 3000.0)
    c0 = cells[0]
    s0 = snap_outer(c0, bbox, outer)
    assert s0[0] == outer[0] and s0[2] == outer[1], "最外圈没拉到全量 bbox"
    assert s0[1] == c0[1] and s0[3] == c0[3], "内部边界被误改"
    e0 = expand(s0, 0.2, ref=c0)
    assert abs((e0[1] - s0[1]) - (c0[1] - c0[0]) * 0.2) < 1e-9, "外扩量跟着被撑大的 core 走了"
    cov2 = np.zeros(len(pts), bool)
    for c in cells:
        cov2 |= _inbox(pts, snap_outer(c, bbox, outer))
    assert cov2.all(), "snap_outer 后 core 并集出现空洞"
    print("  OK 最外圈拉到全量 bbox 后并集仍无洞，且外扩量仍按鲁棒格尺寸")

    # 文本模型能被 PGSR 的 colmap_loader 原样读回（观测表留空这条靠它兜住）
    pg = os.path.join(os.path.dirname(_HERE), "repos", "pgsr")
    if not os.path.isdir(pg):
        print("  ! 未找到 repos/pgsr，跳过往返断言")
        return
    sys.path.insert(0, pg)
    from scene.colmap_loader import (read_extrinsics_text, read_intrinsics_text,
                                     read_points3D_text)
    cams = {1: ("PINHOLE", 640, 480, np.array([500.0, 500.0, 320.0, 240.0]))}
    imgs = [dict(id=7, q=np.array([1.0, 0, 0, 0]), t=np.array([1.5, 2.5, 3.5]),
                 cam=1, name="DJI_0001.JPG", pids=np.array([], np.int64))]
    pid = np.array([3, 9], np.int64)
    xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    rgb = np.array([[10, 20, 30], [40, 50, 60]], np.uint8)
    err = np.array([0.5, 0.7])
    d = tempfile.mkdtemp()
    write_model_txt(d, cams, imgs, pid, xyz, rgb, err)
    ci = read_intrinsics_text(os.path.join(d, "cameras.txt"))
    ce = read_extrinsics_text(os.path.join(d, "images.txt"))
    cx, _, _ = read_points3D_text(os.path.join(d, "points3D.txt"))
    assert ci[1].width == 640 and abs(ci[1].params[0] - 500.0) < 1e-9
    assert ce[7].name == "DJI_0001.JPG" and abs(ce[7].tvec[2] - 3.5) < 1e-9
    assert ce[7].xys.shape == (0, 2), "空观测行没被解析成 0 行"
    assert cx.shape == (2, 3) and abs(cx[1, 2] - 6.0) < 1e-9
    print("  OK cameras/images/points3D.txt 均被 colmap_loader 正确读回（含空观测行）")
    print("[selfcheck] 全部通过")


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(
        description="大场景分块：全局 COLMAP 模型 → N 个可独立训练的子模型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--sparse", help=r"全局 COLMAP 模型目录（如 data\scene03\sparse）")
    ap.add_argument("--images", help=r"带 GPS EXIF 的原图目录（如 data\scene03\input），用于求 ENU 变换")
    ap.add_argument("--out", help=r"输出块目录（如 data\scene03\blocks）")
    ap.add_argument("--grid", default="3x2", help="网格 NxM（列数x行数），按相机数均衡切分")
    ap.add_argument("--overlap", type=float, default=0.2, help="expanded 相对 core 的外扩比例")
    ap.add_argument("--vis_thresh", type=float, default=0.25,
                    help="块外相机的入选阈值：该图观测点落在 core 内的比例")
    ap.add_argument("--geo_out", default=None, help="顺带写出 geo_transform.json 的路径")
    ap.add_argument("--dry_run", action="store_true", help="只打印每块统计，不写盘")
    ap.add_argument("--selfcheck", action="store_true", help="跑内置自检后退出")
    args = ap.parse_args()

    if args.selfcheck:
        _selfcheck()
        return
    for req in ("sparse", "images", "out"):
        if not getattr(args, req):
            ap.error(f"--{req} 必填（或用 --selfcheck）")

    from georeference import get_gps, geodetic_to_enu, umeyama

    nx, ny = (int(v) for v in args.grid.lower().split("x"))

    print(f"[读取] {args.sparse}")
    cams = read_cameras_bin(os.path.join(args.sparse, "cameras.bin"))
    imgs = read_images_bin(os.path.join(args.sparse, "images.bin"))
    pid, xyz, rgb, err, tlen = read_points3d_bin(os.path.join(args.sparse, "points3D.bin"))
    print(f"  相机模型 {len(cams)} 个 / 图 {len(imgs)} 张 / 点 {len(pid):,}")

    # 与 PGSR read_points3D_binary 同一套质量过滤，保证块内初始点 == 全局跑会用的那批
    keep = (err <= 2.0) & (tlen >= 3)
    pid, xyz, rgb, err = pid[keep], xyz[keep], rgb[keep], err[keep]
    print(f"  质量过滤(err≤2 且 track≥3) → {len(pid):,} 点（与 PGSR 一致）")

    # 相机中心
    C = np.array([-qvec2rotmat(im["q"]).T @ im["t"] for im in imgs])

    # GPS → ENU，Umeyama 求相似变换（复用 georeference.py）
    gps, ok = [], []
    for k, im in enumerate(imgs):
        g = get_gps(os.path.join(args.images, im["name"]))
        if g is not None:
            gps.append(g)
            ok.append(k)
    if len(ok) < 3:
        raise SystemExit(f"只有 {len(ok)} 张图有 GPS EXIF，至少要 3 张才能求 ENU 变换")
    G = np.array(gps)
    ref = (float(G[:, 0].mean()), float(G[:, 1].mean()), float(G[:, 2].min()))
    ENU = np.array([geodetic_to_enu(a, b, c, ref) for a, b, c in G])
    s, R, t = umeyama(C[ok], ENU)
    res = np.linalg.norm((s * (R @ C[ok].T).T + t) - ENU, axis=1)
    print(f"[配准] {len(ok)}/{len(imgs)} 张有 GPS；scale = {s:.4f} m/单位；"
          f"残差 中位 {np.median(res):.2f} m / 最大 {res.max():.2f} m")
    if np.median(res) > 5.0:
        print("  ! 残差 >5 m：消费级 GPS 噪声偏大，分块边界会随之抖动（不影响正确性）")

    def to_enu(P):
        return s * (R @ P.T).T + t

    Penu, Cenu = to_enu(xyz), to_enu(C)

    geo = dict(scale=float(s), R=R.tolist(), t=t.tolist(),
               ref_lat=ref[0], ref_lon=ref[1], ref_alt=ref[2],
               residual_m=float(res.mean()), num_matched=len(ok))  # mean，与 georeference.py 同定义
    if args.geo_out and not args.dry_run:
        os.makedirs(os.path.dirname(os.path.abspath(args.geo_out)), exist_ok=True)
        with open(args.geo_out, "w", encoding="utf-8") as f:
            json.dump(geo, f, indent=2)
        print(f"[写出] {args.geo_out}")

    # 分块：格子建在【鲁棒范围】上（SfM 离群点会把 min/max 撑大数倍，见 scene02 实测：
    # 点云 E 的 min/max 是 -549~1332 m，而 1~99 分位只有 -195~239 m）；
    # 最外圈 core 事后再拉到全场景 bbox，保证 core 并集不漏内容。
    full = (float(Penu[:, 0].min()), float(Penu[:, 1].min()),
            float(Penu[:, 0].max()), float(Penu[:, 1].max()))
    q1 = np.percentile(Penu[:, :2], 1, axis=0)
    q99 = np.percentile(Penu[:, :2], 99, axis=0)
    rob = (min(q1[0], Cenu[:, 0].min()), min(q1[1], Cenu[:, 1].min()),
           max(q99[0], Cenu[:, 0].max()), max(q99[1], Cenu[:, 1].max()))
    print(f"[范围] 鲁棒(1~99%+全部相机) E {rob[0]:.0f}~{rob[2]:.0f} m / N {rob[1]:.0f}~{rob[3]:.0f} m"
          f"；全量含离群 E {full[0]:.0f}~{full[2]:.0f} m / N {full[1]:.0f}~{full[3]:.0f} m")
    cells = balanced_grid(Cenu[:, :2], nx, ny, rob)

    # point3D_id → 行号（searchsorted，比 dict 快得多）
    order = np.argsort(pid)
    spid = pid[order]

    def obs_rows(ids):
        v = ids[ids >= 0]
        j = np.searchsorted(spid, v)
        j = np.clip(j, 0, len(spid) - 1)
        return order[j[spid[j] == v]]

    OBS = [obs_rows(im["pids"]) for im in imgs]

    print(f"\n[分块] {nx}x{ny} = {len(cells)} 块，overlap {args.overlap}，vis_thresh {args.vis_thresh}")
    print("（下表 E/N 是鲁棒格边界；最外圈块的外沿会被拉到全量 bbox，保证 core 并集不漏）")
    print("块    格子E 范围        格子N 范围       块内相机  选中相机     块内点")
    blocks, tot_cam = [], 0
    for bi, cell in enumerate(cells):
        core = snap_outer(cell, rob, full)      # 块真正拥有的范围（合并时按它裁）
        exp = expand(core, args.overlap, ref=cell)   # 外扩量按鲁棒格尺寸，不按拉长后的 core
        incore_p = _inbox(Penu[:, :2], core)
        inexp_p = _inbox(Penu[:, :2], exp)
        inexp_c = _inbox(Cenu[:, :2], exp)
        frac = np.array([incore_p[o].mean() if len(o) else 0.0 for o in OBS])
        sel = inexp_c | (frac >= args.vis_thresh)
        tot_cam += int(sel.sum())
        name = f"b{bi:02d}"
        blocks.append(dict(name=name, core=[float(v) for v in core],
                           expanded=[float(v) for v in exp],
                           n_cam=int(sel.sum()), n_cam_in=int(inexp_c.sum()),
                           n_pt=int(inexp_p.sum()),
                           images=[imgs[k]["name"] for k in np.where(sel)[0]]))
        print(f"{name}  {cell[0]:7.0f}~{cell[1]:7.0f}  {cell[2]:7.0f}~{cell[3]:7.0f}  "
              f"{int(inexp_c.sum()):8d}  {int(sel.sum()):8d}  {int(inexp_p.sum()):9,d}")

        if not args.dry_run:
            d = os.path.join(args.out, name, "sparse")
            write_model_txt(d, cams, [imgs[k] for k in np.where(sel)[0]],
                            pid[inexp_p], xyz[inexp_p], rgb[inexp_p], err[inexp_p])

    print(f"\n冗余度 = Σ块相机数 / 总图数 = {tot_cam}/{len(imgs)} = {tot_cam/len(imgs):.2f}×"
          "（1.5~2.0× 正常；<1.2× 说明重叠不足易出边界飘浮，>2.5× 说明块切太碎）")

    if args.dry_run:
        print("\n[dry-run] 未写盘。调 --grid / --overlap / --vis_thresh 直到数字满意，再去掉 --dry_run")
        return

    meta = dict(geo=geo, grid=[nx, ny], overlap=args.overlap,
                vis_thresh=args.vis_thresh, bbox_enu=list(full), bbox_robust_enu=list(rob),
                blocks=blocks)
    os.makedirs(args.out, exist_ok=True)
    mp = os.path.join(args.out, "blocks.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[写出] {mp}  （merge_blocks.py 读它做 core 裁剪）")
    print(f"[写出] {len(cells)} 个子模型 → {args.out}\\b*\\sparse\\")
    print("\n下一步（逐块训练，-i 指回共享图目录）：")
    o = args.out.replace("/", "\\")
    o = o if os.path.isabs(o) else "..\\..\\" + o
    print(r"  cd repos\pgsr && python train.py -s %s\b00 -m ..\..\outputs\scene03_b00 "
          r"-i ..\..\images_2 -r 2 --iterations 30000 --data_device cpu" % o)


if __name__ == "__main__":
    main()
