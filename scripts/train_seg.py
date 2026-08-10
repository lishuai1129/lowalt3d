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


def compute_class_weights(mask_paths, K, cap=10.0, ignore_index=255):
    """逆频率(invsqrt)类别权重，均值归一、封顶 cap。少数类(glass/metal)权重明显>1。"""
    import numpy as np, cv2, torch
    cnt = np.zeros(K, np.float64)
    for p in mask_paths:
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE).reshape(-1)
        m = m[m != ignore_index]
        cnt += np.bincount(np.clip(m, 0, K - 1), minlength=K)
    inv = 1.0 / np.sqrt(cnt / max(cnt.sum(), 1) + 1e-6)
    return torch.tensor(np.minimum(inv / inv.mean(), cap), dtype=torch.float)


def seg_loss(logits, target, weight, gamma, ignore_index=255):
    """加权 + focal CE。logits[B,K,h,w] 上采样到 target[B,H,W]；gamma=0 即纯加权 CE。"""
    import torch.nn.functional as F
    logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    logp = F.log_softmax(logits, 1)
    ce = F.nll_loss(logp, target, weight=weight, ignore_index=ignore_index, reduction="none")
    if gamma > 0:
        pt = logp.gather(1, target.clamp(min=0).unsqueeze(1)).squeeze(1).exp()
        ce = (1 - pt).clamp(min=1e-6) ** gamma * ce
    valid = target != ignore_index
    return ce[valid].mean() if valid.any() else ce.mean()


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
    ap.add_argument("--backbone", default="nvidia/segformer-b2-finetuned-ade-512-512")
    ap.add_argument("--focal_gamma", type=float, default=2.0, help="focal γ；0=纯加权CE")
    ap.add_argument("--cls_weight_cap", type=float, default=10.0, help="逆频率类别权重上限")
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
    class_w = compute_class_weights([mp for _, mp in pairs], K, cap=args.cls_weight_cap).to(dev)
    print("类别权重:", {classes[i]: round(float(class_w[i]), 2) for i in range(K)})

    model.train()
    for ep in range(args.epochs):
        tot = 0.0
        for im, mk in dl:
            im, mk = im.to(dev), mk.to(dev)
            loss = seg_loss(model(pixel_values=im).logits, mk, class_w, args.focal_gamma)
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
