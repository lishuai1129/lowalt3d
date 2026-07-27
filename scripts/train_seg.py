"""
train_seg.py  ·  方向A：微调 SegFormer 做材质分割

目的：用你 LabelMe 标注→labelme2mask 得到的 (图像, 掩码) 对，微调一个 SegFormer，
      学会把航拍图逐像素分成材质类别。

依赖：pip install transformers
      国内下载预训练权重慢/失败时，先设环境变量：set HF_ENDPOINT=https://hf-mirror.com

用法：
  python scripts\train_seg.py ^
    --images  data\scene01\material\annot_images ^
    --masks   data\scene01\material\masks ^
    --classes configs\classes.txt ^
    --out     outputs\seg_model --epochs 80 --img_size 512
"""
import os, glob, argparse
import numpy as np


def load_classes(p):
    return [l.strip() for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--backbone", default="nvidia/segformer-b0-finetuned-ade-512-512")
    args = ap.parse_args()

    import torch, cv2
    from torch.utils.data import Dataset, DataLoader
    from transformers import SegformerForSemanticSegmentation

    classes = load_classes(args.classes); K = len(classes)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # 配对 图像+掩码
    pairs = []
    for m in sorted(glob.glob(os.path.join(args.masks, "*.png"))):
        base = os.path.splitext(os.path.basename(m))[0]
        for ext in (".JPG", ".jpg", ".png", ".PNG"):
            ip = os.path.join(args.images, base + ext)
            if os.path.isfile(ip):
                pairs.append((ip, m)); break
    assert pairs, "没有配对的 图像+掩码，请先 LabelMe 标注 + 跑 labelme2mask"
    print(f"训练样本: {len(pairs)}，类别 {K}: {classes}")

    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225]); S = args.img_size

    class DS(Dataset):
        def __len__(self): return len(pairs)
        def __getitem__(self, i):
            ip, mp = pairs[i]
            im = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (S, S)).astype(np.float32) / 255.0
            im = ((im - mean) / std).transpose(2, 0, 1)
            mk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            mk = cv2.resize(mk, (S, S), interpolation=cv2.INTER_NEAREST).astype(np.int64)
            return torch.from_numpy(im).float(), torch.from_numpy(mk)

    dl = DataLoader(DS(), batch_size=args.batch, shuffle=True)
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.backbone, num_labels=K, ignore_mismatched_sizes=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for ep in range(args.epochs):
        tot = 0.0
        for im, mk in dl:
            im, mk = im.to(dev), mk.to(dev)
            loss = model(pixel_values=im, labels=mk).loss     # SegFormer 内部上采样对齐标签
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"epoch {ep+1}/{args.epochs}  loss {tot/len(dl):.4f}")

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    with open(os.path.join(args.out, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(classes))
    print(f"模型已保存 → {args.out}")


if __name__ == "__main__":
    main()
