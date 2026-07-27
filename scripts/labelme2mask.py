"""
labelme2mask.py  ·  方向A：把 LabelMe 标注(.json) 转成材质标签 PNG

目的：LabelMe 标注出的是多边形(.json)，训练分割模型需要每像素类别的 PNG 掩码。
      本脚本把每张标注 json 转成 <name>.png(uint8，像素值=类别id)，未标注区域=0(background)。

用法：
  python scripts\labelme2mask.py ^
    --json_dir data\scene01\material\annot_images ^
    --classes  configs\classes.txt ^
    --out_dir  data\scene01\material\masks
"""
import os, json, glob, argparse
import numpy as np


def load_classes(p):
    return [l.strip() for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", required=True, help="LabelMe 标注 json 所在目录(通常与图片同目录)")
    ap.add_argument("--classes", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    import cv2

    classes = load_classes(args.classes)
    idx = {c: i for i, c in enumerate(classes)}
    print(f"类别({len(classes)}): {classes}")
    os.makedirs(args.out_dir, exist_ok=True)

    jsons = sorted(glob.glob(os.path.join(args.json_dir, "*.json")))
    if not jsons:
        print(f"[warn] {args.json_dir} 下没有 .json，请先用 LabelMe 标注"); return

    for jp in jsons:
        d = json.load(open(jp, encoding="utf-8"))
        H, W = int(d["imageHeight"]), int(d["imageWidth"])
        mask = np.zeros((H, W), np.uint8)          # 0 = background
        for sh in d.get("shapes", []):
            lb = sh["label"].strip()
            if lb not in idx:
                print(f"  [warn] 未知类别 '{lb}' in {os.path.basename(jp)} (跳过)"); continue
            pts = np.array(sh["points"], np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], int(idx[lb]))
        out = os.path.join(args.out_dir, os.path.splitext(os.path.basename(jp))[0] + ".png")
        cv2.imwrite(out, mask)
        print(f"  {os.path.basename(out)}  (shapes={len(d.get('shapes', []))})")

    print(f"\n完成 {len(jsons)} 张 → {args.out_dir}")


if __name__ == "__main__":
    main()
