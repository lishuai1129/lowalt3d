r"""
merge_blocks.py  ·  分块训练的产物合并回【一个高斯集】

配合 split_scene.py。每块训练时用的是 expanded 框（带重叠，防边界飘浮），
合并时只保留中心落在**自己 core 框**内的高斯 —— 重叠区恰好被切一刀，不重不漏。

合并后就是一个普通 PGSR 模型目录，下游一行不用改：
  cd repos\pgsr
  python render.py -m ..\..\outputs\scene03_merged -s ..\..\data\scene03 ^
      -i images_4 -r 4 --iteration 30000 --voxel_size 0.01 --max_depth 10.0

同时从 --source_path 的全局 COLMAP 模型生成 cameras.json，包含全部注册视角。
相机位置/旋转保持 COLMAP 坐标，图像尺寸/焦距取去畸变原始内参，与 PGSR 格式一致。

关键（与用 open3d 读写的天真做法相反）：PGSR 的 point_cloud.ply 是【高斯】ply，
每点带 f_dc_* / f_rest_* / opacity / scale_* / rot_*。用 o3d 读再写会只留 xyz、
把高斯属性丢光。本脚本同 clean_gaussians.py，用 plyfile 读整个结构化数组、
掩码后原样拼接写回。

用法（在 E:\lowalt3d 下）：
  python scripts\merge_blocks.py --blocks data\scene03\blocks\blocks.json ^
      --models "outputs\scene03_b*" --iteration 30000 ^
      --source_path data\scene03 --images images_4 ^
      --out outputs\scene03_merged\point_cloud\iteration_30000\point_cloud.ply

  python scripts\merge_blocks.py --selfcheck      # 秒级自检
"""
import os
import re
import glob
import json
import shutil
import argparse
import importlib.util

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
    s = re.sub(r"source_path='[^']*'", lambda _: "source_path=%r" % os.path.abspath(source_path), s)
    if images:
        s = re.sub(r"images='[^']*'", lambda _: "images=%r" % images, s)
    with open(os.path.join(dst_model, "cfg_args"), "w", encoding="utf-8") as f:
        f.write(s)
    return True


def load_global_cameras(source_path):
    """全局去畸变 COLMAP 模型 → PGSR cameras.json 数据，无需加载照片或 CUDA。

    直接加载 colmap_loader.py，避免导入 scene 包触发 torch/CUDA；不调用
    readColmapSceneInfo，避免生成/覆盖源场景中的 points3D.ply。
    """
    loader_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "repos", "pgsr", "scene", "colmap_loader.py")
    if not os.path.isfile(loader_path):
        raise FileNotFoundError(f"缺少 PGSR 相机读取模块: {loader_path}")
    spec = importlib.util.spec_from_file_location("_merge_colmap_loader", loader_path)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)

    sparse = os.path.join(os.path.abspath(source_path), "sparse")
    camera_bin, image_bin = (os.path.join(sparse, n) for n in ("cameras.bin", "images.bin"))
    camera_txt, image_txt = (os.path.join(sparse, n) for n in ("cameras.txt", "images.txt"))
    if os.path.isfile(camera_bin) and os.path.isfile(image_bin):
        intrinsics = loader.read_intrinsics_binary(camera_bin)
        images = loader.read_extrinsics_binary(image_bin)
    elif os.path.isfile(camera_txt) and os.path.isfile(image_txt):
        intrinsics = loader.read_intrinsics_text(camera_txt)
        images = loader.read_extrinsics_text(image_txt)
    else:
        raise FileNotFoundError(f"{sparse} 中缺少成套 cameras/images.bin 或 .txt；"
                                "--source_path 必须指向全局去畸变场景")
    if not images:
        raise ValueError(f"{sparse} 中没有注册相机")

    result, seen = [], set()
    # PGSR 在创建 cameras.json 前按照片名排序，记录原始尺寸而非训练缩图尺寸。
    for image in sorted(images.values(), key=lambda im: os.path.basename(im.name).split(".")[0]):
        name = os.path.basename(image.name).split(".")[0]
        if name in seen:
            raise ValueError(f"相机名称重复: {name}，不能生成唯一的 img_name")
        seen.add(name)
        camera = intrinsics[image.camera_id]
        if camera.model == "PINHOLE":
            fx, fy = camera.params[:2]
        elif camera.model == "SIMPLE_PINHOLE":
            fx = fy = camera.params[0]
        else:
            raise ValueError(f"相机 {camera.id} 的模型为 {camera.model}，"
                              "需要先去畸变为 PINHOLE/SIMPLE_PINHOLE")
        if camera.width <= 0 or camera.height <= 0 or not np.isfinite([fx, fy]).all() or min(fx, fy) <= 0:
            raise ValueError(f"相机 {camera.id} 的尺寸或焦距无效")
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = loader.qvec2rotmat(image.qvec)
        world_to_camera[:3, 3] = image.tvec
        camera_to_world = np.linalg.inv(world_to_camera)
        if not np.isfinite(camera_to_world).all():
            raise ValueError(f"相机 {name} 的位姿无效")
        result.append(dict(id=len(result), img_name=name,
                           width=int(camera.width), height=int(camera.height),
                           position=camera_to_world[:3, 3].tolist(),
                           rotation=camera_to_world[:3, :3].tolist(),
                           fy=float(fy), fx=float(fx)))
    return result


def write_cameras_json(model_dir, cameras):
    """写入完整 JSON 后再替换目标，避免中断留下半个文件。"""
    import tempfile
    os.makedirs(model_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".cameras-", suffix=".json", dir=model_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cameras, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        path = os.path.join(model_dir, "cameras.json")
        os.replace(tmp, path)
        return path
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


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
    # 非单位旋转验证方向约定，避免把 world-to-camera 误写成 camera-to-world。
    with tempfile.TemporaryDirectory() as camera_test:
        sparse = os.path.join(camera_test, "scene", "sparse")
        os.makedirs(sparse)
        with open(os.path.join(sparse, "cameras.txt"), "w", encoding="utf-8") as f:
            f.write("1 PINHOLE 640 480 500 450 320 240\n")
        q = 2 ** -0.5
        with open(os.path.join(sparse, "images.txt"), "w", encoding="utf-8") as f:
            f.write(f"2 {q} 0 0 {q} 1 2 3 1 B.JPG\n\n"
                    "1 1 0 0 0 0 0 0 1 A.JPG\n\n")
        cameras = load_global_cameras(os.path.join(camera_test, "scene"))
        assert [c["img_name"] for c in cameras] == ["A", "B"]
        assert [c["id"] for c in cameras] == [0, 1]
        assert np.allclose(cameras[1]["position"], [-2, 1, -3])
        assert np.allclose(cameras[1]["rotation"], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        assert cameras[1]["width"] == 640 and cameras[1]["height"] == 480
        assert cameras[1]["fx"] == 500 and cameras[1]["fy"] == 450
        path = write_cameras_json(os.path.join(camera_test, "merged"), cameras)
        with open(path, encoding="utf-8") as f:
            assert json.load(f) == cameras
    print("  OK 全局 cameras.json：完整视角、排序、位姿方向、原始内参、JSON 往返")
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
                    help="全局去畸变场景目录，用于生成 cameras.json 和 cfg_args（默认从 blocks.json 路径上推两级）")
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

    mdir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(args.out)), "..", ".."))
    src = args.source_path or os.path.dirname(os.path.dirname(os.path.abspath(args.blocks)))
    # 写大 PLY 前先检查全局相机数据是否有效，避免最后才发现相机文件生成不了。
    cameras = load_global_cameras(src)

    print(f"[合并] {len(blocks)} 块 @ iteration {args.iteration}")
    print("块    模型目录                        总高斯      core 内     裁掉   NaN丢弃")
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
        # PGSR 开几何正则（7000 轮）后会有少量高斯整行变 NaN（scene02 各块约 0.7~0.9%），
        # 进了合并结果会污染 render.py 的光栅化
        fin = np.all([np.isfinite(a[f]) for f in a.dtype.names], axis=0)
        k = in_core(to_enu(xyz, geo), b["core"]) & fin
        parts.append(a[k])
        print(f"{b['name']}  {os.path.basename(d):28s} {len(a):10,d} {int(k.sum()):11,d} "
              f"{len(a)-int(k.sum()):10,d} {int((~fin).sum()):9,d}")

    merged = np.concatenate(parts)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    PlyData([PlyElement.describe(merged, "vertex")], text=False).write(args.out)
    tot = sum(len(p) for p in parts)
    print(f"\n[写出] {args.out}")
    print(f"  合并 {len(merged):,} 高斯（各块 core 内之和 {tot:,}）、{len(dt0.names)} 个属性完整保留")

    # 合并模型目录补 cfg_args，render.py 才认
    if write_cfg_args(m2b[blocks[0]["name"]], mdir, src, args.images):
        print(f"  cfg_args 已写入 {mdir}（source_path 指向 {src}）")
    else:
        print(f"  ! 块 0 没有 cfg_args，未生成 —— render.py 时务必显式传 -s / -i / -r")
    camera_path = write_cameras_json(mdir, cameras)
    print(f"  cameras.json 已写入 {camera_path}（全局 {len(cameras)} 个注册视角）")

    print(f"\n下一步（TSDF 出网格，-s 指全局场景：{len(cameras)} 相机同一坐标系）：")
    # 绝对路径兼容跨盘输出；分辨率沿用 cfg_args，不再硬编码 images_2 / -r 2。
    image_arg = f'-i "{args.images}" ' if args.images else ""
    print(r'  cd repos\pgsr && python render.py -m "%s" -s "%s" %s'
          r"--iteration %d --voxel_size 0.01 --max_depth 10.0"
          % (mdir, os.path.abspath(src), image_arg, args.iteration))


if __name__ == "__main__":
    main()
