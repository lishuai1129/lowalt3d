"""
labelme2mask.py  ·  方向A：把 LabelMe 标注(.json) 转成材质标签 PNG

目的：LabelMe 标注出的是多边形(.json)，训练分割模型需要每像素类别的 PNG 掩码。
      本脚本把每张标注 json 转成 <name>.png(uint8，像素值=类别id)。
      给出 --base_masks 时仅覆盖画出的区域；否则未标注区域=0(background)。
      LabelMe 的 point 可用 --point_radius 转成小圆形区域（半径为原图像素）。

用法：
  python scripts\labelme2mask.py ^
    --json_dir data\scene01\material\annot_images ^
    --classes  configs\classes.txt ^
    --out_dir  data\scene01\material\masks

局部修正已有掩码：再加 --base_masks data\scene01\material\masks_segformer
"""
import os, json, glob, argparse
import numpy as np
from material_common import _PALETTE


def load_classes(p):
    return [l.strip() for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", required=True, help="LabelMe 标注 json 所在目录(通常与图片同目录)")
    ap.add_argument("--classes", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_masks", default=None,
                    help="可选：已有类别掩码目录；只用 LabelMe 标注覆盖画出的区域")
    ap.add_argument("--point_radius", type=int, default=0,
                    help="将 point 标注填成圆形区域，半径为原图像素；0 表示跳过点")
    ap.add_argument("--color", action="store_true",
                    help="额外输出彩色预览到 <out_dir>_color")
    args = ap.parse_args()
    import cv2

    if args.point_radius < 0:
        ap.error("--point_radius 必须 >= 0")

    classes = load_classes(args.classes)
    idx = {c: i for i, c in enumerate(classes)}
    print(f"类别({len(classes)}): {classes}")
    os.makedirs(args.out_dir, exist_ok=True)
    if args.color:
        os.makedirs(args.out_dir + "_color", exist_ok=True)

    jsons = sorted(glob.glob(os.path.join(args.json_dir, "*.json")))
    if not jsons:
        print(f"[warn] {args.json_dir} 下没有 .json，请先用 LabelMe 标注"); return

    for jp in jsons:
        d = json.load(open(jp, encoding="utf-8"))
        H, W = int(d["imageHeight"]), int(d["imageWidth"])
        base = os.path.splitext(os.path.basename(jp))[0]
        if args.base_masks:
            base_path = os.path.join(args.base_masks, base + ".png")
            mask = cv2.imread(base_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"找不到基础掩码: {base_path}")
            if mask.shape != (H, W):
                raise ValueError(f"基础掩码尺寸 {mask.shape} 与标注 {(H, W)} 不一致: {base_path}")
            if np.any(mask >= len(classes)):
                raise ValueError(f"基础掩码含有超出类别范围的值: {base_path}")
        else:
            mask = np.zeros((H, W), np.uint8)      # 0 = background
        point_count = 0
        skipped_points = 0
        for sh in d.get("shapes", []):
            lb = sh["label"].strip()
            if lb not in idx:
                print(f"  [warn] 未知类别 '{lb}' in {os.path.basename(jp)} (跳过)"); continue
            pts = np.array(sh["points"], np.int32).reshape(-1, 1, 2)
            kind = sh.get("shape_type", "polygon")
            if kind == "rectangle" and len(pts) == 2:
                p1, p2 = pts.reshape(2, 2)
                cv2.rectangle(mask, tuple(p1), tuple(p2), int(idx[lb]), -1)
            elif kind == "polygon" and len(pts) >= 3:
                cv2.fillPoly(mask, [pts], int(idx[lb]))
            elif kind == "point" and len(pts) == 1:
                if args.point_radius:
                    x, y = sh["points"][0]
                    cv2.circle(mask, (round(x), round(y)), args.point_radius,
                               int(idx[lb]), -1)
                    point_count += 1
                else:
                    skipped_points += 1
            else:
                print(f"  [warn] 不支持的标注形状 '{kind}' in {os.path.basename(jp)} (跳过)")
        out = os.path.join(args.out_dir, os.path.splitext(os.path.basename(jp))[0] + ".png")
        cv2.imwrite(out, mask)
        if args.color:
            preview = _PALETTE[mask]
            cv2.imwrite(os.path.join(args.out_dir + "_color", base + ".png"),
                        cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        print(f"  {os.path.basename(out)}  (shapes={len(d.get('shapes', []))}, "
              f"point_applied={point_count}, point_skipped={skipped_points})")

    print(f"\n完成 {len(jsons)} 张 → {args.out_dir}")


if __name__ == "__main__":
    main()
