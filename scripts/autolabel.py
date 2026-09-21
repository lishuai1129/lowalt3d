"""
autolabel.py  ·  方向A：用预训练 SegFormer(ADE20K) 自动预标注 → 映射到我们的材质类别

目的：不用人工画，直接对所有图跑一遍语义分割，把 ADE20K 的 150 类按关键词映射到
      我们的 7 类材质，输出掩码 + 上色预览，供人工/AI 复核。

⚠️ ADE20K 是"地面视角"训练，俯视航拍有域差；结果需复核。
   流程：autolabel → 看上色预览 → 满意就直接用；不满意就纠正少量 + train_seg 微调。

依赖：pip install transformers ; 国内下权重设 set HF_ENDPOINT=https://hf-mirror.com

用法：
  python scripts\autolabel.py --images data\scene01\images ^
    --out data\scene01\material\masks_auto --color
然后复核 masks_auto_color\，满意就：
  python scripts\assign_material.py ... --masks_dir data\scene01\material\masks_auto
"""
import os, sys, glob, argparse, re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from material_common import _PALETTE, _DEFAULT_CLASS_ORDER

# ADE20K 标签名 关键词 → 我们的类别（可按复核结果调整）
_KEYWORDS = {
    "vegetation":  ["tree", "grass", "plant", "palm", "flower", "bush", "hedge", "forest", "field"],
    "ground_road": ["road", "sidewalk", "pavement", "path", "earth", "ground",
                    "sand", "dirt", "soil", "runway", "land"],
    "concrete":    ["building", "wall", "house", "edifice", "skyscraper"],
    "roof":        ["roof"],
    "glass":       ["window", "windowpane", "glass"],
    # 其余(sky/person/car/water/...) → background
}


def build_lut(id2label, class_order):
    """ADE20K 类id → 我们的类id 查找表。"""
    idx = {c: i for i, c in enumerate(class_order)}
    bg = idx.get("background", 0)
    n = max(int(k) for k in id2label) + 1
    lut = np.full(n, bg, np.uint8)
    for ade_id, name in id2label.items():
        words = set(re.findall(r"[a-z]+", str(name).lower()))
        for cls, kws in _KEYWORDS.items():
            if cls in idx and any(k in words for k in kws):
                lut[int(ade_id)] = idx[cls]; break
    return lut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", default=None)
    ap.add_argument("--backbone", default="nvidia/segformer-b0-finetuned-ade-512-512",
                    help="更准可用 nvidia/segformer-b4-finetuned-ade-512-512(更慢更大)")
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--color", action="store_true", help="额外输出上色预览到 <out>_color(复核用)")
    args = ap.parse_args()

    import torch, cv2
    import torch.nn.functional as F
    from transformers import SegformerForSemanticSegmentation

    class_order = (_DEFAULT_CLASS_ORDER if not args.classes
                   else [l.strip() for l in open(args.classes, encoding="utf-8") if l.strip()])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"加载预训练模型 {args.backbone} ...")
    model = SegformerForSemanticSegmentation.from_pretrained(args.backbone).to(dev).eval()
    lut = build_lut(model.config.id2label, class_order)
    print(f"类别映射就绪；类别: {class_order}")

    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225]); S = args.img_size
    os.makedirs(args.out, exist_ok=True)
    if args.color:
        os.makedirs(args.out + "_color", exist_ok=True)

    imgs = sorted({os.path.normcase(os.path.abspath(p)): p
                   for e in ("*.JPG", "*.jpg", "*.png", "*.PNG")
                   for p in glob.glob(os.path.join(args.images, e))}.values())
    print(f"自动预标注 {len(imgs)} 张 ...")
    for k, ip in enumerate(imgs):
        im0 = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB); H, W = im0.shape[:2]
        im = cv2.resize(im0, (S, S)).astype(np.float32) / 255.0
        im = ((im - mean) / std).transpose(2, 0, 1)
        with torch.no_grad():
            logits = model(pixel_values=torch.from_numpy(im[None]).float().to(dev)).logits
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
            ade = logits.argmax(1)[0].cpu().numpy()
        lab = lut[np.clip(ade, 0, len(lut) - 1)].astype(np.uint8)     # → 我们的类别
        base = os.path.splitext(os.path.basename(ip))[0]
        cv2.imwrite(os.path.join(args.out, base + ".png"), lab)
        if args.color:
            col = _PALETTE[np.clip(lab, 0, len(_PALETTE) - 1)]
            cv2.imwrite(os.path.join(args.out + "_color", base + ".png"),
                        cv2.cvtColor(col, cv2.COLOR_RGB2BGR))
        if (k + 1) % 10 == 0 or k == len(imgs) - 1:
            print(f"  {k+1}/{len(imgs)}")
    print(f"\n掩码 → {args.out}" + (f"；上色预览 → {args.out}_color(先复核这个)" if args.color else ""))
    print("图例: 植被=绿 道路/农田=深灰 混凝土=浅灰 屋顶=红 玻璃=蓝 金属=橙 背景=灰")


if __name__ == "__main__":
    main()
