"""
gaussian_to_mesh_material.py  ·  可微高斯材质 → 网格面片（开题 §12 kNN 加权投票）

目的：把 train_gaussian_material.py 学出的每高斯材质向量，迁移到 PGSR 提取的三角面片上，
      生成通信仿真用的 face_materials.csv（与"直接投票版"同格式，可对比）。

原理（开题 §12）：对每个三角面片取中心 → kNN 找邻近高斯 → 以 不透明度×距离衰减 加权
      平均其材质概率 → argmax 得面片材质 → 按 ITU-R P.2040 映射电磁参数。

运行（在 E:\lowalt3d 下）：
  python scripts\gaussian_to_mesh_material.py ^
    --mesh outputs\scene01_pgsr\mesh\tsdf_fusion_post.ply ^
    --gaussian_ply outputs\scene01_pgsr\point_cloud\iteration_15000\point_cloud.ply ^
    --material_pt outputs\scene01_pgsr\gaussian_material\gaussian_material.pt ^
    --classes configs\classes.txt --out_dir outputs\scene01_pgsr\material_gauss
"""
import os, sys, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "repos", "pgsr"))
sys.path.insert(0, _HERE)
from material_common import load_radio_params, _PALETTE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--gaussian_ply", required=True)
    ap.add_argument("--material_pt", required=True)
    ap.add_argument("--classes", default=None)
    ap.add_argument("--radio_params", default=os.path.join(
        os.path.dirname(_HERE), "configs", "radio_params.yaml"))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k", type=int, default=8, help="每面片取最近的 k 个高斯")
    ap.add_argument("--chunk", type=int, default=1000000, help="面片分块(控内存)")
    args = ap.parse_args()

    import torch, open3d as o3d
    from scipy.spatial import cKDTree
    from scene.gaussian_model import GaussianModel

    class_order, em = load_radio_params(args.radio_params)
    K = len(class_order)
    print(f"材质类别 K={K}: {class_order}")

    # ---- 高斯：中心 + 不透明度 + 材质概率 ----
    g = GaussianModel(3); g.load_ply(args.gaussian_ply)
    xyz = g.get_xyz.detach().cpu().numpy().astype(np.float64)
    op = g.get_opacity.detach().cpu().numpy()[:, 0].astype(np.float64)      # sigmoid后
    mat = torch.load(args.material_pt, map_location="cpu")                  # [N,K]
    prob = torch.softmax(mat.float(), dim=1).numpy().astype(np.float32)     # [N,K]
    assert prob.shape[0] == xyz.shape[0] == len(op), "高斯数不一致"
    assert prob.shape[1] == K, f"材质维度 {prob.shape[1]} != K {K}"
    print(f"高斯数 N={len(xyz)}")

    # ---- 网格面片中心 ----
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    V = np.asarray(mesh.vertices); Tri = np.asarray(mesh.triangles)
    nF = len(Tri)
    centers = V[Tri].mean(axis=1)                                          # [F,3]
    print(f"面片数 F={nF}")

    # ---- kNN 加权投票(分块) ----
    tree = cKDTree(xyz)
    face_mat = np.empty(nF, np.int64)
    for s in range(0, nF, args.chunk):
        e = min(s + args.chunk, nF)
        dist, idx = tree.query(centers[s:e], k=args.k)                     # [c,k]
        if args.k == 1:
            dist = dist[:, None]; idx = idx[:, None]
        sig = dist.mean(1, keepdims=True) + 1e-9
        w = op[idx] * np.exp(-(dist / sig) ** 2)                           # [c,k] 不透明度×距离衰减
        fp = (w[:, :, None] * prob[idx]).sum(1) / (w.sum(1, keepdims=True) + 1e-9)  # [c,K]
        face_mat[s:e] = fp.argmax(1)
        print(f"  面片 {e}/{nF}")

    # ---- 导出 CSV(与直接投票版同格式) ----
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "face_materials.csv")
    with open(csv_path, "w") as f:
        f.write("face_id,material_id,material_name,eps_r,sigma,penetration_loss_db\n")
        for i in range(nF):
            name = class_order[face_mat[i]]; p = em.get(name, {})
            f.write(f"{i},{face_mat[i]},{name},{p.get('eps_r',1.0)},"
                    f"{p.get('sigma',0.0)},{p.get('penetration_loss_db',0.0)}\n")
    print(f"[导出] {csv_path}")

    # ---- 上色网格(顶点=相邻面片材质众数) ----
    vcount = np.zeros((len(V), K), np.float32)
    np.add.at(vcount, Tri.reshape(-1), np.repeat(np.eye(K, dtype=np.float32)[face_mat], 3, axis=0))
    vmat = vcount.argmax(1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(_PALETTE[np.clip(vmat, 0, len(_PALETTE)-1)] / 255.0)
    ply_path = os.path.join(args.out_dir, "material_mesh.ply")
    o3d.io.write_triangle_mesh(ply_path, mesh)
    print(f"[导出] {ply_path}")

    print("\n面片材质分布(可微高斯版):")
    for i, n in enumerate(class_order):
        c = int((face_mat == i).sum())
        if c: print(f"  {n:12s}: {c}")


if __name__ == "__main__":
    main()
