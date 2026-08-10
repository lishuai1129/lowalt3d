r"""
downscale_images.py  ·  把去畸变原图预先降采样到磁盘，供【大图集】惰性加载训练

背景（scene03 实测，1202 张 8285x5483 原图）：
  · PGSR 的 preload_img=True 会把【全部】训练图解码进内存：401 张勉强、601 张就把
    系统内存榨干（表现为 CUDA 明明剩 14GB 却分配不出 116MB —— Windows WDDM 下显存
    分配需要系统内存做后备），1202 张直接 MemoryError。
  · 关掉预载改惰性加载虽然不吃内存，但每步都要解码 45MP 原图（~400ms），只有 1.8 it/s。
解法：预先把图降到【训练分辨率】写到磁盘。惰性加载解码的就是小图（~25ms），
  内存占用≈0、GPU 能跑满，于是可以【用满全部视角】而不必抽样。

用法（在 E:\lowalt3d 下）：
  python scripts\downscale_images.py --images data\scene03\images --out data\scene03\images_4 -r 4

之后训练指向新目录，**-r 仍然填 4**（PGSR 的分辨率取自 COLMAP 内参而非图片实际尺寸，
填 4 才会算出与本脚本一致的目标尺寸，使训练时的 resize 成为空操作；若填 1 会把小图
放大回原始 45MP）：
  python train.py -s ..\..\data\scene03 -m ..\..\outputs\scene03_pgsr_hq ^
      -i images_4 -r 4 --iterations 30000 --ncc_scale 0.5 ^
      --max_all_points 3000000 --max_abs_split_points 0 --opacity_cull_threshold 0.05

注意：文件名必须与 COLMAP 记录一致（含扩展名），本脚本原样保留。
"""
import os
import glob
import argparse

_EXTS = ("*.JPG", "*.jpg", "*.JPEG", "*.jpeg", "*.png", "*.PNG", "*.tif", "*.tiff")


def _resize_one(task):
    """降采样单张图。返回 (文件名, 原尺寸, 新尺寸, 错误信息或 None)。"""
    from PIL import Image
    src, dst, r, quality = task
    try:
        with Image.open(src) as im:
            w, h = im.size
            # 取整方式与 PGSR utils/camera_utils.py::loadCam 一致，
            # 使训练时 PILtoTorch 的 resize 成为空操作
            tw, th = round(w / r), round(h / r)
            is_jpeg = os.path.splitext(dst)[1].lower() in (".jpg", ".jpeg")
            if is_jpeg and im.mode != "RGB":
                im = im.convert("RGB")          # JPEG 不支持 alpha
            out = im.resize((tw, th), Image.LANCZOS)
            if is_jpeg:
                out.save(dst, quality=quality, subsampling=0)   # 4:4:4，避免色度下采样伤分割/材质
            else:
                out.save(dst)
        return (os.path.basename(src), (w, h), (tw, th), None)
    except Exception as e:
        return (os.path.basename(src), None, None, repr(e)[:120])


def main():
    ap = argparse.ArgumentParser(description="预降采样图像到训练分辨率（大图集惰性加载用）")
    ap.add_argument("--images", required=True, help=r"去畸变原图目录，如 data\scene03\images")
    ap.add_argument("--out", required=True, help=r"输出目录，如 data\scene03\images_4")
    ap.add_argument("-r", "--resolution", type=int, default=4,
                    help="降采样倍率，与训练时的 -r 保持一致（默认 4）")
    ap.add_argument("--quality", type=int, default=95, help="JPEG 质量（默认 95）")
    ap.add_argument("--workers", type=int, default=0, help="并行进程数，0=自动(CPU-1)")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出（默认跳过，便于断点续跑）")
    args = ap.parse_args()

    from concurrent.futures import ProcessPoolExecutor

    srcs = sorted({os.path.abspath(p) for e in _EXTS
                   for p in glob.glob(os.path.join(args.images, e))})
    if not srcs:
        print(f"[失败] {args.images} 下没找到图片")
        return
    os.makedirs(args.out, exist_ok=True)

    tasks, skipped = [], 0
    for s in srcs:
        dst = os.path.join(args.out, os.path.basename(s))    # 文件名原样保留，COLMAP 要靠它匹配
        if not args.overwrite and os.path.isfile(dst):
            skipped += 1
            continue
        tasks.append((s, dst, args.resolution, args.quality))

    print(f"源图 {len(srcs)} 张，待处理 {len(tasks)}，已存在跳过 {skipped}")
    if not tasks:
        print("[完成] 全部已存在，无需处理（要重做请加 --overwrite）")
        return

    nw = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 4) - 1)
    print(f"并行进程 {nw}，降采样 1/{args.resolution} …")

    done, errs, first = 0, [], None
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for name, osz, nsz, err in ex.map(_resize_one, tasks, chunksize=4):
            done += 1
            if err:
                errs.append((name, err))
            elif first is None:
                first = (osz, nsz)
            if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)}")

    if first:
        print(f"\n尺寸: {first[0][0]}x{first[0][1]} -> {first[1][0]}x{first[1][1]}")
    if errs:
        print(f"[警告] {len(errs)} 张失败：")
        for n, e in errs[:5]:
            print(f"  {n}: {e}")

    ok = len(tasks) - len(errs) + skipped
    print(f"\n[导出] {args.out}（可用 {ok} 张）")
    print(f"训练时加 -i {os.path.basename(os.path.normpath(args.out))} -r {args.resolution}"
          f"，并【不要】加 --preload_img（保持惰性加载，内存占用≈0）")


if __name__ == "__main__":
    main()
