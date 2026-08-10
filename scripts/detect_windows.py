"""
detect_windows.py · 零样本窗检测(Grounding DINO 开放词表) → 并入现有掩码当 glass

思路：GDINO 用文本提示 "window" 检测窗户框(通用大模型，不吃 CMP 那种地面域限制)，
把窗框区域盖在现有 masks_auto 之上当 glass(class 5)，其余类(veg/road/concrete)沿用底图。
先 box-fill(窗≈矩形，够用且稳)；要更贴边再加 --sam(留待 box 验证 OK 后)。

⚠️ 必须配 --base_masks 当底图(GDINO 只出窗框，不认 veg/road)。
依赖：transformers>=4.40(已 5.14)；国内下权重 set HF_ENDPOINT=https://hf-mirror.com

用法(先探 20 张)：
  python scripts\detect_windows.py --images data\scene02\images_2 ^
    --base_masks data\scene02\material\masks_auto ^
    --out data\scene02\material\masks_win --classes configs\classes.txt --limit 20
看 masks_win_overlay\ 里窗户是否被蓝框/蓝块盖住；行就去掉 --limit 全量跑。
box 太少→降 --box_threshold(0.2)；误检大片→降 --max_box_frac。
"""
import os, sys, glob, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from material_common import _DEFAULT_CLASS_ORDER, _PALETTE


def gdino_boxes(proc, model, pil, text, box_thr, text_thr, dev, torch):
    """返回 [N,4] xyxy(像素)。兼容 transformers 几个版本的后处理签名。"""
    inp = proc(images=pil, text=text, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(**inp)
    W, H = pil.size
    kw = dict(target_sizes=[(H, W)], text_threshold=text_thr)
    for call in (
        lambda: proc.post_process_grounded_object_detection(out, inp["input_ids"], box_threshold=box_thr, **kw),
        lambda: proc.post_process_grounded_object_detection(out, inp["input_ids"], threshold=box_thr, **kw),
        lambda: proc.post_process_grounded_object_detection(out, threshold=box_thr, **kw),
    ):
        try:
            return call()[0]["boxes"].detach().cpu().numpy()
        except TypeError:
            continue
    raise RuntimeError("post_process_grounded_object_detection 签名不兼容，贴报错给我")


def filter_windows(boxes, W, H, max_frac, max_wh, contain_drop, min_win_px):
    """窗框过滤：裁边界 + 去过大(整面墙/长条) + 去外框(套住多个小窗) + 去碎点(像素过少)。
    返回 [(x0,y0,x1,y1),...]"""
    B = []
    for b in boxes:
        x0, y0, x1, y1 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W, x1), min(H, y1)
        if x1 > x0 and y1 > y0:
            B.append((x0, y0, x1, y1))
    cen = [((x0 + x1) / 2, (y0 + y1) / 2) for x0, y0, x1, y1 in B]
    keep = []
    for i, (x0, y0, x1, y1) in enumerate(B):
        w, h = x1 - x0, y1 - y0
        if w * h > max_frac * W * H:          continue     # 面积过大
        if w > max_wh * W or h > max_wh * H:  continue     # 单边过大(整面墙/长条)
        if w * h < min_win_px:                continue     # 面积过小(GDINO 置信度抖动的碎点)
        n_in = sum(1 for j, (mx, my) in enumerate(cen)
                   if j != i and x0 < mx < x1 and y0 < my < y1)
        if n_in >= contain_drop:              continue     # 外框：套住多个小窗
        keep.append((x0, y0, x1, y1))
    return keep


def sam_union(sam, sp, pil, boxes, dev, torch, batch=24):
    """SAM 用每个 GDINO 框做 prompt → 贴边掩码 → 并集 [H,W] bool（比 box-fill 清晰）。"""
    W, H = pil.size
    if not len(boxes):
        return np.zeros((H, W), bool)
    union = np.zeros((H, W), bool)
    for s in range(0, len(boxes), batch):                 # 分批: 防全尺寸掩码一次占爆内存
        e = min(s + batch, len(boxes))
        bb = [[list(map(float, b)) for b in boxes[s:e]]]
        inp = sp(pil, input_boxes=bb, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = sam(**inp, multimask_output=False)
        masks = sp.image_processor.post_process_masks(
            out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0]
        union |= masks[:, 0].numpy().astype(bool).any(0)
    return union


def main():
    if "--note" in sys.argv:
        print("""检测=候选，几何=裁决（两级管线，换场景不涂脏）：
  1) 本脚本只产出『窗候选』掩码(masks_win)，GDINO 低置信框请勿直接当玻璃硬监督——
     默认 --box_threshold 0.15（0.10 会引入大量误检，别降）。宁可漏检不可误涂。
  2) 掩码喂给 train_gaussian_material 时须开 --lambda_gate(水平面禁 glass)，
     学出的材质还要过 smooth_face_materials --geo_gate --horiz_forbid glass,metal,vegetation
     硬几何门(竖直面才保留玻璃)。
  3) 最终裁决用 label_windows_geom.py：凹坑(recess)+窗邻居支撑才判窗，纯几何、场景无关。
  4) smooth_face_materials --min_win_area 按连通块滤碎玻璃(小于阈值的整块回退邻居材质)。
  建议链：detect(0.15+min_win_px) → train(--lambda_gate) → gaussian_to_mesh →
         smooth(--geo_gate --min_win_area) → label_windows_geom(recess) → 导出""")
        return
    ap = argparse.ArgumentParser(description="GDINO 零样本窗检测 → 并入掩码当 glass（**候选层**）")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base_masks", required=True, help="底图掩码目录(masks_auto)，窗盖其上当 glass")
    ap.add_argument("--classes", default=None)
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--prompt", default="window. glass window.")
    ap.add_argument("--target_class", default="glass",
                    help="检测框填成哪个类(默认 glass；metal 时配金属提示词+放宽 --max_box_wh/--contain_drop)")
    ap.add_argument("--box_threshold", type=float, default=0.15,
                    help="GDINO 置信度阈值(0.15 平衡;越低召回越高但误检越多——不建议低于 0.12,宁可漏检不可误涂,误涂会让材质训练把非玻璃学成玻璃)")
    ap.add_argument("--text_threshold", type=float, default=0.20)
    ap.add_argument("--min_win_px", type=int, default=500,
                    help="保留的最小窗框像素数(滤 GDINO 置信度抖动产生的碎点;低分辨率图可下调)")
    ap.add_argument("--max_box_frac", type=float, default=0.1,
                    help="跳过面积>整图此比例的框(防把整面墙当一个窗)")
    ap.add_argument("--max_box_wh", type=float, default=0.15,
                    help="跳过单边>整图宽/高此比例的框(整面墙/长条=大框)")
    ap.add_argument("--contain_drop", type=int, default=3,
                    help="若一个框内含≥此数量的其它框中心→判为外框(整面墙)丢弃")
    ap.add_argument("--limit", type=int, default=0, help=">0 只跑前 N 张(探路)")
    ap.add_argument("--sam", action="store_true", help="用 SAM 把 GDINO 框精化成贴边掩码(窗更清晰，较慢)")
    ap.add_argument("--sam_model", default="facebook/sam-vit-base",
                    help="SAM 权重(base 快~375MB；huge 更准~2.5GB)")
    ap.add_argument("--note", action="store_true",
                    help="仅打印使用说明(检测=候选,材质由几何门裁决),不运行")
    args = ap.parse_args()

    import torch, cv2
    from PIL import Image
    from transformers import AutoProcessor, GroundingDinoForObjectDetection

    class_order = (_DEFAULT_CLASS_ORDER if not args.classes
                   else [l.strip() for l in open(args.classes, encoding="utf-8") if l.strip()])
    ci = {c: i for i, c in enumerate(class_order)}
    target_id = ci.get(args.target_class)
    assert target_id is not None, f"--target_class '{args.target_class}' 不在 {class_order}"
    ov_bgr = tuple(int(x) for x in _PALETTE[target_id % len(_PALETTE)][::-1])  # 叠加色=该类调色板(RGB→BGR)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"加载 Grounding DINO {args.model} ...（首次自动下载）")
    proc = AutoProcessor.from_pretrained(args.model)
    model = GroundingDinoForObjectDetection.from_pretrained(args.model).to(dev).eval()
    text = args.prompt if args.prompt.strip().endswith(".") else args.prompt.strip() + "."
    print(f"提示词: '{text}'  → 类 '{args.target_class}'(id={target_id})")

    sam = sp = None
    if args.sam:
        from transformers import SamModel, SamProcessor
        print(f"加载 SAM {args.sam_model} ...（首次自动下载）")
        sam = SamModel.from_pretrained(args.sam_model).to(dev).eval()
        sp = SamProcessor.from_pretrained(args.sam_model)

    os.makedirs(args.out, exist_ok=True)
    ov_dir = args.out + "_overlay"; os.makedirs(ov_dir, exist_ok=True)

    imgs = sorted({os.path.normcase(p): p for e in ("*.JPG", "*.jpg", "*.png", "*.PNG")
                   for p in glob.glob(os.path.join(args.images, e))}.values(),
                  key=os.path.normcase)   # dedup: Windows 大小写不敏感, *.jpg 与 *.JPG 会重复命中
    if args.limit > 0:
        imgs = imgs[:args.limit]
    print(f"处理 {len(imgs)} 张 ...")
    tot_box, tot_win = 0, 0
    for k, ip in enumerate(imgs):
        bgr = cv2.imread(ip)
        if bgr is None:
            continue
        H, W = bgr.shape[:2]
        pil = Image.open(ip).convert("RGB")
        boxes = gdino_boxes(proc, model, pil, text, args.box_threshold, args.text_threshold, dev, torch)
        stem = os.path.splitext(os.path.basename(ip))[0]

        bp = os.path.join(args.base_masks, stem + ".png")
        base = cv2.imread(bp, cv2.IMREAD_GRAYSCALE) if os.path.isfile(bp) else np.zeros((H, W), np.uint8)
        if base.shape != (H, W):
            base = cv2.resize(base, (W, H), interpolation=cv2.INTER_NEAREST)
        out = base.copy(); ov = bgr.copy()
        keep = filter_windows(boxes, W, H, args.max_box_frac, args.max_box_wh,
                              args.contain_drop, args.min_win_px)
        if args.sam and keep:
            mask = sam_union(sam, sp, pil, keep, dev, torch)     # SAM 贴边掩码(并集)
            out[mask] = target_id; ov[mask] = ov_bgr
        else:
            for x0, y0, x1, y1 in keep:
                out[y0:y1, x0:x1] = target_id
                ov[y0:y1, x0:x1] = ov_bgr
        kept = len(keep)
        tot_box += len(boxes); tot_win += kept
        cv2.imwrite(os.path.join(args.out, stem + ".png"), out)
        cv2.imwrite(os.path.join(ov_dir, stem + ".jpg"), cv2.addWeighted(bgr, 0.6, ov, 0.4, 0))
        if (k + 1) % 5 == 0 or k == len(imgs) - 1:
            print(f"  {k+1}/{len(imgs)}  本图窗框 {kept}(检出 {len(boxes)})")

    print(f"\n合并掩码 → {args.out}\n窗检测叠加预览 → {ov_dir}（先看这个）")
    print(f"总计: 保留窗框 {tot_win} / 检出 {tot_box}")
    if tot_win == 0:
        print("[结论] 没保留到窗框：降 --box_threshold 0.2 再试；仍 0→GDINO 也不吃这域，转手标微调")


def _selfcheck():
    base = np.full((10, 10), 3, np.uint8)
    out = base.copy()
    for x0, y0, x1, y1 in np.array([[2, 2, 5, 5]]):
        out[y0:y1, x0:x1] = 5
    assert out[3, 3] == 5 and out[0, 0] == 3, "box-fill 逻辑错"
    boxes = np.array([[0, 0, 80, 80],                         # 外框(单边过大+含多小框)→drop
                      [10, 10, 20, 20], [30, 30, 40, 40], [50, 50, 60, 60],  # 小窗→keep
                      [0, 0, 95, 8]], float)                  # 长条(宽过大)→drop
    keep = filter_windows(boxes, 100, 100, max_frac=0.1, max_wh=0.15, contain_drop=3,
                          min_win_px=25)
    assert len(keep) == 3, f"窗框过滤应留 3, got {len(keep)}"
    keep2 = filter_windows(boxes, 100, 100, max_frac=0.1, max_wh=0.15, contain_drop=3,
                           min_win_px=1000)                     # 全被 min_win_px 滤掉
    assert len(keep2) == 0, f"min_win_px 应滤光, got {len(keep2)}"
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
