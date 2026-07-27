"""
autolabel_color.py  ·  方向A：零下载 自动预标注（色彩/亮度 地物分类）

目的：不下载任何模型，只用 numpy/cv2，按颜色/亮度/植被指数把俯视航拍图分成材质类别。
      针对本 nadir 农田/公园场景调好：绿植 / 道路裸土 / 混凝土 / (少量)屋顶。

适用：HuggingFace 下不动、或想立刻出结果时。精度不如训练模型，但完全自动、可复核。
      之后若要更准：复核纠正少量 + train_seg 微调，或修好下载后用 autolabel.py。

用法：
  python scripts\autolabel_color.py --images data\scene01\images ^
    --out data\scene01\material\masks_auto --color
"""
import os, sys, glob, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from material_common import _PALETTE, _DEFAULT_CLASS_ORDER


def classify(rgb, class_order):
    """[H,W,3] float[0,1] → [H,W] 类别id。地物色彩规则(针对俯视航拍)。"""
    idx = {c: i for i, c in enumerate(class_order)}
    def I(n, d=0): return idx.get(n, d)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    V = rgb.max(-1); mn = rgb.min(-1); S = (V - mn) / (V + 1e-6)
    ExG = 2 * G - R - B                              # 植被指数(excess green)
    warm = R - B                                     # >0 暖(裸土/棕)；≈0 中性灰(混凝土/沥青)

    lab = np.full(rgb.shape[:2], I("background"), np.uint8)
    neutral = (S < 0.16) & (np.abs(warm) < 0.05)     # 中性灰
    # 1) 中性灰：亮→混凝土(停车场/水泥)，中→道路(沥青)
    lab[neutral & (V > 0.55)] = I("concrete")
    lab[neutral & (V >= 0.22) & (V <= 0.55)] = I("ground_road")
    # 2) 暖色(裸土/农田/土路) → 道路/地面
    lab[(warm > 0.04) & (V > 0.20) & (ExG < 0.06)] = I("ground_road")
    # 3) 植被(优先级最高，覆盖前面)
    lab[ExG > 0.06] = I("vegetation")
    # 4) 屋顶：很暗且非植被(阈值保守，阴影可能误入)
    lab[(V < 0.20) & (ExG < 0.03)] = I("roof")
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", default=None)
    ap.add_argument("--color", action="store_true", help="额外输出上色预览到 <out>_color(复核用)")
    args = ap.parse_args()
    import cv2

    class_order = (_DEFAULT_CLASS_ORDER if not args.classes
                   else [l.strip() for l in open(args.classes, encoding="utf-8") if l.strip()])
    os.makedirs(args.out, exist_ok=True)
    if args.color:
        os.makedirs(args.out + "_color", exist_ok=True)

    _seen, imgs = set(), []
    for p in sum([glob.glob(os.path.join(args.images, e))
                  for e in ("*.JPG", "*.jpg", "*.png", "*.PNG")], []):
        key = os.path.normcase(os.path.abspath(p))
        if key not in _seen:
            _seen.add(key); imgs.append(p)
    imgs = sorted(imgs)
    print(f"零下载自动预标注 {len(imgs)} 张 ...")
    for k, ip in enumerate(imgs):
        rgb = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        lab = classify(rgb, class_order)
        base = os.path.splitext(os.path.basename(ip))[0]
        cv2.imwrite(os.path.join(args.out, base + ".png"), lab)
        if args.color:
            col = _PALETTE[np.clip(lab, 0, len(_PALETTE) - 1)]
            cv2.imwrite(os.path.join(args.out + "_color", base + ".png"),
                        cv2.cvtColor(col, cv2.COLOR_RGB2BGR))
        if (k + 1) % 20 == 0 or k == len(imgs) - 1:
            print(f"  {k+1}/{len(imgs)}")
    print(f"\n掩码 → {args.out}" + (f"；上色预览 → {args.out}_color(先复核)" if args.color else ""))
    print("图例: 植被=绿 道路/农田=深灰 混凝土=浅灰 屋顶=红 背景=灰")


if __name__ == "__main__":
    main()
