"""
eval_masks.py · 2D 掩码 per-class IoU/recall/precision vs 人工标注 held-out

没有量化评测就没法声称"更准了"。留一小批人工精标图(不参与训练)做真值，比对预测掩码。
用法(在 E:\lowalt3d 下):
  python scripts\eval_masks.py ^
    --pred data\scene02\material\masks_ft ^
    --gt   data\scene02\material\masks_gt ^
    --classes configs\classes.txt
"""
import os, argparse
import numpy as np
import cv2


def main():
    ap = argparse.ArgumentParser(description="2D 掩码 per-class 指标")
    ap.add_argument("--pred", required=True, help="预测掩码目录")
    ap.add_argument("--gt", required=True, help="人工真值掩码目录(held-out，不参与训练)")
    ap.add_argument("--classes", required=True)
    ap.add_argument("--ignore_index", type=int, default=255)
    a = ap.parse_args()

    classes = [l.strip() for l in open(a.classes, encoding="utf-8") if l.strip()]
    K = len(classes)
    conf = np.zeros((K, K), np.int64)                                # 行=GT, 列=Pred
    gts = [f for f in os.listdir(a.gt) if f.lower().endswith(".png")]
    assert gts, f"GT 目录无 png: {a.gt}"

    used = 0
    for f in gts:
        g = cv2.imread(os.path.join(a.gt, f), cv2.IMREAD_GRAYSCALE)
        pth = os.path.join(a.pred, f)
        if g is None or not os.path.isfile(pth):
            print("缺预测/读不了:", f); continue
        p = cv2.imread(pth, cv2.IMREAD_GRAYSCALE)
        if p.shape != g.shape:
            p = cv2.resize(p, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = g != a.ignore_index
        conf += np.bincount((g[m].astype(np.int64) * K + np.clip(p[m], 0, K - 1)).clip(0, K * K - 1),
                            minlength=K * K).reshape(K, K)
        used += 1

    tp = np.diag(conf).astype(np.float64)
    iou = tp / (conf.sum(1) + conf.sum(0) - tp + 1e-9)
    rec = tp / (conf.sum(1) + 1e-9)
    prec = tp / (conf.sum(0) + 1e-9)
    print(f"\n评测图数: {used}/{len(gts)}")
    print(f"{'class':12s} {'IoU':>7} {'recall':>7} {'prec':>7} {'GTpix':>11}")
    for i, n in enumerate(classes):
        print(f"{n:12s} {iou[i]:7.3f} {rec[i]:7.3f} {prec[i]:7.3f} {int(conf[i].sum()):11d}")
    present = conf.sum(1) > 0
    print(f"\nmIoU(present classes) = {iou[present].mean():.3f}")


if __name__ == "__main__":
    main()
