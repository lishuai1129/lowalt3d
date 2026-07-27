"""
infer_seg.py  ·  方向A：用训练好的 SegFormer 推理所有去畸变图 → 材质掩码

目的：把 Step1 去畸变的全部 87 张图跑一遍分割，得到每图的材质标签 PNG，
      供 assign_material.py --masks_dir 使用（把"占位启发式"换成"模型识别"）。

用法：
  python scripts\infer_seg.py ^
    --model  outputs\seg_model ^
    --images data\scene01\images ^
    --out    data\scene01\material\masks_all --color
然后重跑 Step4：
  python scripts\assign_material.py ... --masks_dir data\scene01\material\masks_all
"""
import os, glob, argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--color", action="store_true", help="额外输出上色预览到 <out>_color")
    args = ap.parse_args()

    import torch, cv2
    import torch.nn.functional as F
    from transformers import SegformerForSemanticSegmentation

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SegformerForSemanticSegmentation.from_pretrained(args.model).to(dev).eval()
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225]); S = args.img_size
    os.makedirs(args.out, exist_ok=True)
    if args.color:
        import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from material_common import _PALETTE
        os.makedirs(args.out + "_color", exist_ok=True)

    imgs = sorted(sum([glob.glob(os.path.join(args.images, e))
                       for e in ("*.JPG", "*.jpg", "*.png", "*.PNG")], []))
    print(f"推理 {len(imgs)} 张 ...")
    for k, ip in enumerate(imgs):
        im0 = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB); H, W = im0.shape[:2]
        im = cv2.resize(im0, (S, S)).astype(np.float32) / 255.0
        im = ((im - mean) / std).transpose(2, 0, 1)
        with torch.no_grad():
            logits = model(pixel_values=torch.from_numpy(im[None]).float().to(dev)).logits
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
            lab = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        base = os.path.splitext(os.path.basename(ip))[0]
        cv2.imwrite(os.path.join(args.out, base + ".png"), lab)
        if args.color:
            col = _PALETTE[np.clip(lab, 0, len(_PALETTE) - 1)]
            cv2.imwrite(os.path.join(args.out + "_color", base + ".png"),
                        cv2.cvtColor(col, cv2.COLOR_RGB2BGR))
        if (k + 1) % 10 == 0 or k == len(imgs) - 1:
            print(f"  {k+1}/{len(imgs)}")
    print(f"材质掩码 → {args.out}" + (f"（上色预览 → {args.out}_color）" if args.color else ""))


if __name__ == "__main__":
    main()
