r"""
merge_blocks.py  ·  分块训练的产物合并回【一个高斯集】

配合 split_scene.py。每块训练时用的是 expanded 框（带重叠，防边界飘浮），
合并时只保留中心落在**自己 core 框**内的高斯 —— 重叠区恰好被切一刀，不重不漏。

合并后就是一个普通 PGSR 模型目录，下游一行不用改：
  cd repos\pgsr
  python render.py -m ..\..\outputs\scene03_merged -s ..\..\data\scene03 ^
      -i ..\..\images_2 -r 2 --iteration 30000 --voxel_size 0.01 --max_depth 10.0

关键（与用 open3d 读写的天真做法相反）：PGSR 的 point_cloud.ply 是【高斯】ply，
每点带 f_dc_* / f_rest_* / opacity / scale_* / rot_*。用 o3d 读再写会只留 xyz、
把高斯属性丢光。本脚本同 clean_gaussians.py，用 plyfile 读整个结构化数组、
掩码后原样拼接写回。

用法（在 E:\lowalt3d 下）：
  python scripts\merge_blocks.py --blocks data\scene03\blocks\blocks.json ^
      --models "outputs\scene03_b*" --iteration 30000 ^
      --out outputs\scene03_merged\point_cloud\iteration_30000\point_cloud.ply

  python scripts\merge_blocks.py --selfcheck      # 秒级自检
"""
import os
import re
import glob
import json
import shutil
import argparse

import numpy as np


def to_enu(P, geo):
    """COLMAP 坐标 → ENU 米（与 split_scene.py 写进 blocks.json 的变换一致）。"""
    R = np.array(geo["R"])
    return geo["scale"] * (R @ P.T).T + np.array(geo["t"])


def in_core(xyz_enu, core):
    e0, e1, n0, n1 = core
    return ((xyz_enu[:, 0] >= e0) & (xyz_enu[:, 0] <= e1)
            & (xyz_enu[:, 1] >= n0) & (xyz_enu[:, 1] <= n1))


def write_cfg_args(src_model, dst_model, source_path, images):
    """把块 0 的 cfg_args 抄到合并模型目录，并改写 source_path/images。

    render.py 靠 get_combined_args 读它；不改写的话，忘了传 -s 会静默用块 0 的相机出网格。
    """
    p = os.path.join(src_model, "cfg_args")
    if not os.path.isfile(p):
        return False
    s = open(p, encoding="utf-8").read()
    s = re.sub(r"source_path='[^']*'", "source_path=%r" % os.path.abspath(source_path), s)
    if images:
        s = re.sub(r"images='[^']*'", "images=%r" % images, s)
    with open(os.path.join(dst_model, "cfg_args"), "w", encoding="utf-8") as f:
        f.write(s)
    return True


def _selfcheck():
    import tempfile
    from plyfile import PlyData, PlyElement
    print("[selfcheck] core 裁剪 + 结构化数组拼接")

    dt = np.dtype([(n, "<f4") for n in
                   ("x", "y", "z", "f_dc_0", "opacity", "scale_0", "rot_0")])
    geo = dict(scale=1.0, R=np.eye(3).tolist(), t=[0.0, 0.0, 0.0])

    def mk(path, xs, tag):
        a = np.zeros(len(xs), dt)
        a["x"] = xs
        a["y"] = 0.0
        a["z"] = 0.0
        a["opacity"] = tag                      # 打标签，验证属性没串行
        PlyData([PlyElement.describe(a, "vertex")], text=False).write(path)

    d = tempfile.mkdtemp()
    # 两块 core 相邻: [0,10] 与 [10,20]; 各自训练时都长到了重叠区外
    mk(os.path.join(d, "a.ply"), np.arange(-3, 15, 1.0), 1.0)
    mk(os.path.join(d, "b.ply"), np.arange(5, 24, 1.0), 2.0)
    cores = [(0, 10, -1, 1), (10.0001, 20, -1, 1)]   # 边界错开半个体素,避免 x=10 双收

    parts, expect = [], 0
    for f, core, tag in zip(("a.ply", "b.ply"), cores, (1.0, 2.0)):
        a = PlyData.read(os.path.join(d, f))["vertex"].data
        xyz = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)
        k = in_core(to_enu(xyz, geo), core)
        assert (a["opacity"][k] == tag).all(), "属性串行"
        expect += int(k.sum())
        parts.append(a[k])
    m = np.concatenate(parts)

    assert len(m) == expect == 21, f"合并数 {len(m)} != 各块 core 内之和 {expect}"
    assert m.dtype == dt, "dtype 没保留 —— 高斯属性会丢"
    xs = np.sort(m["x"])
    assert np.all(np.diff(xs) > 0), "core 边界重复收点（重叠区被收了两次）"
    assert xs[0] == 0.0 and xs[-1] == 20.0, f"core 并集没覆盖 [0,20]: {xs[0]}~{xs[-1]}"
    print(f"  OK 合并 {len(m)} 高斯 = 各块 core 内之和，无重复、无空隙、dtype 保留")
    print("[selfcheck] 全部通过")


def main():
    ap = argparse.ArgumentParser(
        description="分块高斯合并：各块按自己的 core 框裁剪后拼成一个 point_cloud.ply",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--blocks", help=r"split_scene.py 产出的 blocks.json")
    ap.add_argument("--models", help=r'各块模型目录的 glob，如 "outputs\scene03_b*"')
    ap.add_argument("--iteration", type=int, default=30000, help="取哪一轮的 point_cloud.ply")
    ap.add_argument("--out", help=r"输出 ply（建议放 <merged>\point_cloud\iteration_N\point_cloud.ply）")
    ap.add_argument("--source_path", default=None,
                    help="全局场景目录，写进合并模型的 cfg_args（默认从 blocks.json 路径上推两级）")
    ap.add_argument("--images", default=None, help="写进 cfg_args 的图目录名，如 images_2")
    ap.add_argument("--selfcheck", action="store_true", help="跑内置自检后退出")
    args = ap.parse_args()

    if args.selfcheck:
        _selfcheck()
        return
    for req in ("blocks", "models", "out"):
        if not getattr(args, req):
            ap.error(f"--{req} 必填（或用 --selfcheck）")

    from plyfile import PlyData, PlyElement

    meta = json.load(open(args.blocks, encoding="utf-8"))
    geo, blocks = meta["geo"], meta["blocks"]

    dirs = sorted(d for d in glob.glob(args.models) if os.path.isdir(d))
    if not dirs:
        raise SystemExit(f"--models {args.models} 没匹配到任何目录")
    # 块名 → 模型目录（按目录名后缀匹配 b00/b01/...）
    m2b = {}
    for b in blocks:
        hit = [d for d in dirs if os.path.basename(d.rstrip("\\/")).endswith(b["name"])]
        if len(hit) != 1:
            raise SystemExit(f"块 {b['name']} 匹配到 {len(hit)} 个模型目录 {hit}，"
                             "请让模型目录名以块名结尾（如 outputs\\scene03_b00）")
        m2b[b["name"]] = hit[0]

    print(f"[合并] {len(blocks)} 块 @ iteration {args.iteration}")
    print("块    模型目录                        总高斯      core 内     裁掉")
    parts, dt0 = [], None
    for b in blocks:
        d = m2b[b["name"]]
        p = os.path.join(d, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
        if not os.path.isfile(p):
            raise SystemExit(f"缺 {p} —— 该块还没训到 {args.iteration} 轮？")
        a = PlyData.read(p)["vertex"].data
        if dt0 is None:
            dt0 = a.dtype
        elif a.dtype != dt0:
            raise SystemExit(f"{b['name']} 的 ply 字段与前面的块不一致（SH 阶数不同？）\n"
                             f"  {a.dtype.names}\n  vs\n  {dt0.names}")
        xyz = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)
        k = in_core(to_enu(xyz, geo), b["core"])
        parts.append(a[k])
        print(f"{b['name']}  {os.path.basename(d):28s} {len(a):10,d} {int(k.sum()):11,d} "
              f"{len(a)-int(k.sum()):10,d}")

    merged = np.concatenate(parts)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    PlyData([PlyElement.describe(merged, "vertex")], text=False).write(args.out)
    tot = sum(len(p) for p in parts)
    print(f"\n[写出] {args.out}")
    print(f"  合并 {len(merged):,} 高斯（各块 core 内之和 {tot:,}）、{len(dt0.names)} 个属性完整保留")

    # 合并模型目录补 cfg_args，render.py 才认
    mdir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(args.out)), "..", ".."))
    src = args.source_path or os.path.dirname(os.path.dirname(os.path.abspath(args.blocks)))
    if write_cfg_args(m2b[blocks[0]["name"]], mdir, src, args.images):
        print(f"  cfg_args 已写入 {mdir}（source_path 指向 {src}）")
    else:
        print(f"  ! 块 0 没有 cfg_args，未生成 —— render.py 时务必显式传 -s / -i / -r")

    print("\n下一步（TSDF 出网格，-s 指全局场景：1202 相机同一坐标系）：")
    print(r"  cd repos\pgsr && python render.py -m ..\..\%s -s ..\..\%s -i ..\..\images_2 -r 2 "
          r"--iteration %d --voxel_size 0.01 --max_depth 10.0"
          % (os.path.relpath(mdir, os.getcwd()).replace("/", "\\"),
             os.path.relpath(src, os.getcwd()).replace("/", "\\"), args.iteration))


if __name__ == "__main__":
    main()
