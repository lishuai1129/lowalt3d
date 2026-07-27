"""
train_gaussian_material.py  ·  可微材质向量嵌入高斯（开题核心创新，PGSR 之上）

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
    args = parser.parse_args()
    dataset = lp.extract(args); pipe = pp.extract(args)

    classes = load_classes(args.classes); K = len(classes)
    dev = "cuda"
    print(f"材质类别 K={K}: {classes}")

    # ---- 加载训练好的高斯 + 相机 ----
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.load_iteration, shuffle=False)
    cams = scene.getTrainCameras()
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
                return torch.from_numpy(np.clip(m, 0, K - 1).astype(np.int64)).to(dev)
        return None

    masks = {c.image_name: get_mask(c) for c in cams}
    valid_cams = [c for c in cams if masks[c.image_name] is not None]
    print(f"有掩码的相机: {len(valid_cams)}/{len(cams)}")
    assert valid_cams, "没有匹配到掩码，检查 --masks_dir 与 image_name"

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
        totc, totk = 0.0, 0.0
        for i in order:
            cam = valid_cams[i]; mk = masks[cam.image_name]
            logits = render_material(cam)                                                   # [K,H,W]
            lce = F.cross_entropy(logits.unsqueeze(0), mk.unsqueeze(0), ignore_index=ig)
            lk = cons_loss(cam, logits) if args.lambda_cons > 0 else torch.zeros((), device=dev)
            (lce + args.lambda_cons * lk).backward()
            opt.step(); opt.zero_grad()
            totc += lce.item(); totk += float(lk)
        print(f"epoch {ep+1}/{args.epochs}  CE {totc/len(valid_cams):.4f}  L_cons {totk/len(valid_cams):.4f}")

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
