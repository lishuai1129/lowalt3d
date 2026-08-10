"""
train_gaussian_material.py  ·  可微材质向量嵌入高斯（PGSR 之上）

方法：给训练好的每个 3D 高斯挂一个 K 维材质向量 material[N,K]（learnable）。
渲染时用 PGSR render() 的 override_color 钩子，把材质向量当"颜色"按每 3 通道一组
alpha-blend 渲出材质图 [K,H,W]（与几何/颜色共享同一光栅化与混合权重）→ softmax
→ 对 2D 材质掩码做交叉熵 → 反传，只更新 material（冻结几何/颜色/透明度）。
无需改 CUDA 光栅化器（Feature-3DGS/LangSplat 同款思路），梯度经 colors_precomp 回传。

前置：先有训练好的 PGSR 模型(iteration_15000) + 材质掩码(masks_auto)。
运行（在 E:\lowalt3d 下）：
  python scripts\train_gaussian_material.py ^
    -s data\scene01 -m outputs\scene01_pgsr -r 4 --load_iteration 15000 ^
    --masks_dir data\scene01\material\masks_auto --classes configs\classes.txt ^
    --mat_out outputs\scene01_pgsr\gaussian_material --epochs 30
输出：
  gaussian_material.pt   每高斯材质向量 [N,K]
  gaussian_material.ply  高斯按材质上色(点云，可 MeshLab 看)
"""
import os, sys, glob
import numpy as np
from argparse import ArgumentParser

_HERE = os.path.dirname(os.path.abspath(__file__))
PGSR = os.path.join(os.path.dirname(_HERE), "repos", "pgsr")
sys.path.insert(0, PGSR)      # 让 from scene / gaussian_renderer / arguments 生效
sys.path.insert(0, _HERE)     # 让 from material_common 生效


def load_classes(p):
    return [l.strip() for l in open(p, encoding="utf-8") if l.strip()]


def main():
    import torch
    import torch.nn.functional as F
    import cv2
    from scene import Scene
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from arguments import ModelParams, PipelineParams
    from material_common import _PALETTE

    parser = ArgumentParser(description="可微材质向量嵌入高斯")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--load_iteration", type=int, default=15000)
    parser.add_argument("--masks_dir", required=True)
    parser.add_argument("--classes", required=True)
    parser.add_argument("--mat_out", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--mat_lr", type=float, default=0.05)
    parser.add_argument("--ignore_bg", action="store_true", help="损失忽略 background(类0)")
    parser.add_argument("--lambda_cons", type=float, default=0.0, help=">0 开启多视角一致性 L_cons(KL)")
    parser.add_argument("--init_material", default=None, help="从已训练 gaussian_material.pt 续训")
    parser.add_argument("--focal_gamma", type=float, default=1.0, help="focal γ；0=纯加权CE")
    parser.add_argument("--cls_weight_cap", type=float, default=10.0, help="逆频率类别权重上限")
    parser.add_argument("--lambda_smooth", type=float, default=0.5, help=">0 开启空间平滑损失 L_smooth")
    parser.add_argument("--smooth_k", type=int, default=8, help="平滑图每高斯邻居数")
    parser.add_argument("--smooth_normal", action="store_true", help="平滑权重叠加法向一致(竖直墙面更稳)")
    parser.add_argument("--lambda_gate", type=float, default=0.0, help=">0 开启几何门：水平面(地/顶)禁 glass(治玻璃糊满地面)")
    parser.add_argument("--glass_gate_cos", type=float, default=0.6, help="|法向·竖直轴|>此值即判为水平面")
    parser.add_argument("--gate_classes", default="glass",
                        help="几何门作用的类(逗号分隔),这些类在水平面上被压概率。默认 glass;本场景金属只在栏杆(竖直)→设 glass,metal 清地面/屋顶金属误检")
    parser.add_argument("--gate_ground_classes", default="",
                        help="反向门:这些类只准水平面(地/顶),在竖直墙面被压概率→墙面回落 concrete。建议 asphalt,background 清墙上深灰")
    parser.add_argument("--crop_bbox", default=None,
                        help="只训练此框内高斯(裁掉没拍到的操场/远路):xmin,ymin,zmin,xmax,ymax,zmax(用 crop_scene.py 打印的串)")
    args = parser.parse_args()
    dataset = lp.extract(args); pipe = pp.extract(args)

    classes = load_classes(args.classes); K = len(classes)
    dev = "cuda"
    print(f"材质类别 K={K}: {classes}")

    # ---- 加载训练好的高斯 + 相机 ----
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.load_iteration, shuffle=False)
    cams = scene.getTrainCameras()
    if args.crop_bbox:
        v = list(map(float, args.crop_bbox.split(",")))
        lo = torch.tensor(v[:3], device=dev); hi = torch.tensor(v[3:], device=dev)
        keep = ((gaussians.get_xyz.detach() >= lo) & (gaussians.get_xyz.detach() <= hi)).all(1)
        for _a in ["_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"]:
            if hasattr(gaussians, _a) and getattr(gaussians, _a).shape[0] == keep.shape[0]:
                setattr(gaussians, _a, torch.nn.Parameter(getattr(gaussians, _a).detach()[keep], requires_grad=False))
        print(f"裁剪高斯(只留建筑区): {keep.shape[0]} → {int(keep.sum())}")
    N = gaussians.get_xyz.shape[0]
    print(f"高斯数 N={N}，相机数={len(cams)}")

    # 冻结高斯（只学材质）
    for a in ["_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"]:
        if hasattr(gaussians, a):
            getattr(gaussians, a).requires_grad_(False)

    # ---- 材质向量参数 ----
    material = torch.nn.Parameter(torch.zeros(N, K, device=dev))
    if args.init_material and os.path.isfile(args.init_material):
        material.data.copy_(torch.load(args.init_material, map_location=dev).to(dev).float())
        print(f"续训：载入 {args.init_material}")
    opt = torch.optim.Adam([material], lr=args.mat_lr)
    bg = torch.zeros(3, device=dev)
    Kpad = ((K + 2) // 3) * 3

    # 预加载每相机的掩码(按 image_name 匹配，resize 到相机分辨率)
    def get_mask(cam):
        base = cam.image_name
        for ext in (".png", ".PNG"):
            p = os.path.join(args.masks_dir, base + ext)
            if os.path.isfile(p):
                m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                m = cv2.resize(m, (cam.image_width, cam.image_height), interpolation=cv2.INTER_NEAREST)
                return np.clip(m, 0, K - 1).astype(np.uint8)   # 留 CPU、uint8：678 张掩码常驻内存 ~0.75GB 而非显存 ~6GB(int64)
        return None

    masks = {c.image_name: get_mask(c) for c in cams}
    valid_cams = [c for c in cams if masks[c.image_name] is not None]
    print(f"有掩码的相机: {len(valid_cams)}/{len(cams)}")
    assert valid_cams, "没有匹配到掩码，检查 --masks_dir 与 image_name"

    # ---- 类别平衡权重（少数类如 glass 不被 concrete 淹没）----
    cnt = np.zeros(K, np.float64)
    for m in masks.values():
        if m is not None:
            cnt += np.bincount(m.reshape(-1), minlength=K)
    present = cnt > 0                                    # 缺席类(roof/metal=0)不参与归一：否则其超大逆频权重会拉高均值、把在场稀有类(glass)压没
    inv = np.zeros(K)
    inv[present] = 1.0 / np.sqrt(cnt[present] / cnt.sum())
    inv[present] = inv[present] / inv[present].mean()    # 只在“在场类”内部归一
    cw = np.minimum(inv, args.cls_weight_cap)            # 缺席类保持 0
    if args.ignore_bg:
        cw[0] = 0.0
    cls_w = torch.tensor(cw, device=dev, dtype=torch.float)
    print("类别权重:", {classes[i]: round(float(cls_w[i]), 2) for i in range(K)})

    # ---- 几何门：gate_classes 只准竖直面(glass/metal 在水平面被压)；gate_ground_classes 只准水平面(asphalt/背景 在竖直墙被压→墙回落 concrete)----
    gate_ids = [classes.index(c.strip()) for c in args.gate_classes.split(",") if c.strip() in classes]
    ground_ids = [classes.index(c.strip()) for c in args.gate_ground_classes.split(",") if c.strip() in classes]
    horiz = None
    if args.lambda_gate > 0 and (gate_ids or ground_ids):
        try:
            from utils.general_utils import build_rotation
            cc = np.stack([torch.inverse(c.world_view_transform)[3, :3].detach().cpu().numpy() for c in cams])
            up_ax = int(cc.std(0).argmin())                              # 相机展布最小轴=竖直
            Rg = build_rotation(gaussians.get_rotation.detach())
            axmin = gaussians.get_scaling.detach().argmin(1)             # 最扁轴=法向
            ngg = torch.nn.functional.normalize(
                torch.gather(Rg, 2, axmin.view(-1, 1, 1).expand(-1, 3, 1)).squeeze(2), dim=1)
            horiz = ngg[:, up_ax].abs() > args.glass_gate_cos            # 法向≈竖直 → 水平面(地/顶)
            print(f"几何门: up轴={up_ax}  水平面 {100*float(horiz.float().mean()):.1f}%  "
                  f"→水平禁 {[classes[i] for i in gate_ids]}; 竖直禁 {[classes[i] for i in ground_ids]}")
        except Exception as e:
            print("[几何门] 跳过:", repr(e)[:80]); horiz = None

    def ce_loss(logits, mk, weight, gamma, ig):
        logp = F.log_softmax(logits, 0)                                  # [K,H,W]
        ce = F.nll_loss(logp.unsqueeze(0), mk.unsqueeze(0), weight=weight,
                        ignore_index=ig, reduction="none")                # [1,H,W]
        if gamma > 0:
            pt = logp.gather(0, mk.clamp(min=0).unsqueeze(0)).squeeze(0).exp()
            ce = (1 - pt).clamp(min=1e-6) ** gamma * ce
        valid = mk.unsqueeze(0) != ig
        return ce[valid].mean() if valid.any() else ce.mean()

    # ---- 平滑损失的高斯 kNN 图（相邻+颜色相近+(可选)同法向 → 材质应接近）----
    src_t = dst_t = w_edge = None
    if args.lambda_smooth > 0:
        from scipy.spatial import cKDTree
        xyz_np = gaussians.get_xyz.detach().cpu().numpy().astype(np.float64)
        dd, ii = cKDTree(xyz_np).query(xyz_np, k=args.smooth_k + 1)
        src = np.repeat(np.arange(N), args.smooth_k); dst = ii[:, 1:].reshape(-1)
        d = dd[:, 1:].reshape(-1); sig_d = float(np.median(d)) + 1e-9
        w_np = np.exp(-(d / sig_d) ** 2)
        dc = gaussians.get_features[:, 0, :].detach()                     # SH DC → 基础色
        rgb = (0.28209479 * dc + 0.5).clamp(0, 1).cpu().numpy()
        w_np *= np.exp(-((rgb[src] - rgb[dst]) ** 2).sum(1) / (2 * 0.10 ** 2))  # 颜色权重=天然护住暗窗玻璃(=G3)
        if args.smooth_normal:
            try:
                from utils.general_utils import build_rotation
                R = build_rotation(gaussians.get_rotation.detach())
                axis = gaussians.get_scaling.detach().argmin(1)
                ng = torch.gather(R, 2, axis.view(-1, 1, 1).expand(-1, 3, 1)).squeeze(2)
                ng = torch.nn.functional.normalize(ng, dim=1).cpu().numpy()
                w_np *= 0.5 + 0.5 * np.abs((ng[src] * ng[dst]).sum(1))
            except Exception as e:
                print("[L_smooth] 法向跳过(退化为颜色+距离):", repr(e)[:80])
        w_edge = torch.tensor(w_np, device=dev, dtype=torch.float)
        src_t = torch.tensor(src, device=dev); dst_t = torch.tensor(dst, device=dev)
        print(f"平滑图: {len(src)} 条边  smooth_normal={args.smooth_normal}")

    # ---- 渲染材质 logits [K,H,W]（override_color 每3通道一组） ----
    def render_material(cam):
        matpad = material if Kpad == K else torch.cat(
            [material, torch.zeros(N, Kpad - K, device=dev)], dim=1)
        imgs = [render(cam, gaussians, pipe, bg,
                       override_color=matpad[:, g:g + 3].contiguous(),
                       return_plane=False)["render"] for g in range(0, Kpad, 3)]
        return torch.cat(imgs, dim=0)[:K]

    # ---- 多视角一致性 L_cons：用参考深度把 near 视角材质重投影回来做 KL ----
    def cons_loss(cam_ref, logits_ref):
        if not cam_ref.nearest_id:
            return torch.zeros((), device=dev)
        cam_n = cams[int(np.random.choice(cam_ref.nearest_id))]
        with torch.no_grad():
            D = render(cam_ref, gaussians, pipe, bg, return_plane=True)["plane_depth"][0]  # [H,W]
            P_near = torch.softmax(render_material(cam_n), dim=0)                            # [K,H,W]
        P_ref = torch.softmax(logits_ref, dim=0)
        H, W = D.shape
        vv, uu = torch.meshgrid(torch.arange(H, device=dev).float(),
                                torch.arange(W, device=dev).float(), indexing="ij")
        xc = (uu - cam_ref.Cx) / cam_ref.Fx * D
        yc = (vv - cam_ref.Cy) / cam_ref.Fy * D
        camp = torch.stack([xc, yc, D], -1)                                                 # [H,W,3]
        Wr = cam_ref.world_view_transform
        Xw = (camp - Wr[3, :3]) @ torch.inverse(Wr[:3, :3])                                  # 世界坐标
        Wn = cam_n.world_view_transform
        cn = Xw @ Wn[:3, :3] + Wn[3, :3]                                                     # near 相机坐标
        z = cn[..., 2].clamp(min=1e-6)
        un = cam_n.Fx * cn[..., 0] / z + cam_n.Cx
        vn = cam_n.Fy * cn[..., 1] / z + cam_n.Cy
        grid = torch.stack([un / (W - 1) * 2 - 1, vn / (H - 1) * 2 - 1], -1)[None]
        warp = F.grid_sample(P_near[None], grid, align_corners=True, padding_mode="zeros")[0].clamp(1e-6)
        valid = (D > 0) & (cn[..., 2] > 0) & (un >= 0) & (un <= W - 1) & (vn >= 0) & (vn <= H - 1)
        kl = (P_ref * (torch.log(P_ref + 1e-6) - torch.log(warp))).sum(0)                    # [H,W]
        return (kl * valid).sum() / (valid.sum() + 1)

    # ---- 训练 ----
    ig = 0 if args.ignore_bg else -100
    order = list(range(len(valid_cams)))
    for ep in range(args.epochs):
        np.random.shuffle(order)
        totc, totk, tots, totg = 0.0, 0.0, 0.0, 0.0
        for i in order:
            cam = valid_cams[i]
            mk = torch.from_numpy(masks[cam.image_name]).to(dev).long()   # 用时才上卡（省显存）
            logits = render_material(cam)                                                   # [K,H,W]
            lce = ce_loss(logits, mk, cls_w, args.focal_gamma, ig)
            lk = cons_loss(cam, logits) if args.lambda_cons > 0 else torch.zeros((), device=dev)
            # softmax(material) 复用于平滑与几何门（避免重复算）
            P = torch.softmax(material, 1) if (args.lambda_smooth > 0 or horiz is not None) else None
            lsm = (w_edge * ((P[src_t] - P[dst_t]) ** 2).sum(1)).mean() if args.lambda_smooth > 0 else torch.zeros((), device=dev)
            lg = torch.zeros((), device=dev)
            if horiz is not None:
                if gate_ids and horiz.any():
                    lg = lg + P[horiz][:, gate_ids].mean()          # 水平面禁 glass/metal
                if ground_ids and (~horiz).any():
                    lg = lg + P[~horiz][:, ground_ids].mean()       # 竖直墙禁 asphalt/背景
            (lce + args.lambda_cons * lk + args.lambda_smooth * lsm + args.lambda_gate * lg).backward()
            opt.step(); opt.zero_grad()
            totc += lce.item(); totk += float(lk.detach()); tots += float(lsm.detach()); totg += float(lg.detach())
        n = len(valid_cams)
        print(f"epoch {ep+1}/{args.epochs}  CE {totc/n:.4f}  L_cons {totk/n:.4f}  L_smooth {tots/n:.5f}  L_gate {totg/n:.5f}")

    # ---- 保存 ----
    os.makedirs(args.mat_out, exist_ok=True)
    torch.save(material.detach().cpu(), os.path.join(args.mat_out, "gaussian_material.pt"))
    labels = material.detach().argmax(1).cpu().numpy()
    print("每高斯材质分布:")
    for i, n in enumerate(classes):
        c = int((labels == i).sum())
        if c: print(f"  {n:12s}: {c}")
    try:
        import open3d as o3d
        xyz = gaussians.get_xyz.detach().cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(_PALETTE[np.clip(labels, 0, len(_PALETTE)-1)] / 255.0)
        o3d.io.write_point_cloud(os.path.join(args.mat_out, "gaussian_material.ply"), pcd)
        print(f"[导出] {args.mat_out}\\gaussian_material.pt / gaussian_material.ply")
    except Exception as e:
        print("导出 ply 失败:", repr(e)[:100])


if __name__ == "__main__":
    main()
