"""
assign_material.py  ·  Step 4：给网格面片分配材质（多视角光线投票）

目的：把三角网格的每个面片打上材质标签，为通信仿真提供电磁属性。
原理：对每个相机，从相机穿过每个像素向网格投射光线(Open3D RaycastingScene)，
      取首个命中面片(天然处理遮挡) → 在该像素的 2D 材质分割图上采样材质概率
      → 累加到该面片的投票。所有相机投完后，面片材质 = argmax 投票。
      再按 ITU-R P.2040 映射为电磁参数。

⚠️ 2D 材质分割目前是占位启发式(颜色规则)，之后替换为毕设训练的分割模型
   （见 heuristic_material_prob，一处替换即可）。

用法：
  python scripts\assign_material.py ^
    --mesh   outputs\scene01_pgsr\mesh\tsdf_fusion_post.ply ^
    --cameras outputs\scene01_pgsr\cameras.json ^
    --images data\scene01\images ^
    --out_dir outputs\scene01_pgsr\material
输出：
  face_materials.csv   每个面片: face_id, material_id, name, eps_r, sigma, loss
  material_mesh.ply     按材质上色的网格(MeshLab 可看)
"""
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from material_common import load_radio_params, _PALETTE   # 复用类别表/调色板


# ============================================================================
# 占位 2D 材质分割（PLACEHOLDER）—— 之后换毕设分割模型
# ============================================================================
def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)

def heuristic_material_prob(rgb, class_order):
    """[H,W,3] float[0,1] → [H,W,K] 材质概率。航拍简易规则，占位用。
    正式实验请替换为: prob = trained_seg_model(rgb)  # 通道顺序须与 class_order 一致
    """
    H, W, _ = rgb.shape
    idx = {n: i for i, n in enumerate(class_order)}
    K = len(class_order)
    logit = np.full((H, W, K), 0.1, np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    gray = rgb.mean(-1)
    L = lambda n: idx.get(n, 0)
    logit[(g > r * 1.05) & (g > b * 1.05) & (g > 0.20), L("vegetation")] += 2.5   # 绿=植被
    logit[gray < 0.22, L("ground_road")] += 1.5                                    # 很暗=沥青路面
    logit[gray > 0.80, L("glass")] += 1.0                                          # 很亮=玻璃/金属反光
    logit[(gray >= 0.22) & (gray <= 0.80), L("concrete")] += 1.0                   # 中灰=混凝土
    return _softmax(logit, -1)


def load_mask_prob(masks_dir, name, W, H, K):
    """从分割掩码目录读 <name>.png(uint8 类别id) → one-hot 概率 [H,W,K]。找不到返回 None。"""
    import cv2
    base = os.path.splitext(name)[0]
    for ext in (".png", ".PNG"):
        p = os.path.join(masks_dir, base + ext)
        if os.path.isfile(p):
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            prob = np.zeros((H, W, K), np.float32)
            lab = np.clip(m, 0, K - 1).astype(np.int64)
            np.put_along_axis(prob, lab[..., None], 1.0, axis=2)
            return prob
    return None


# ============================================================================
# 相机 (3DGS cameras.json 约定：position=相机中心, rotation=C2W, 主点在中心)
# ============================================================================
def load_cameras(path):
    cams = json.load(open(path, "r", encoding="utf-8"))
    out = []
    for c in cams:
        out.append(dict(
            name=c["img_name"], W=int(c["width"]), H=int(c["height"]),
            fx=float(c["fx"]), fy=float(c["fy"]),
            pos=np.array(c["position"], np.float64),
            R=np.array(c["rotation"], np.float64),   # 3x3 C2W
        ))
    return out

def find_image(images_dir, name):
    for ext in ("", ".jpg", ".JPG", ".png", ".PNG", ".jpeg"):
        p = os.path.join(images_dir, name + ext)
        if os.path.isfile(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--radio_params", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "radio_params.yaml"))
    ap.add_argument("--pixel_stride", type=int, default=3, help="光线像素抽样步长(越大越快越糙)")
    ap.add_argument("--default_material", default="concrete", help="没被任何视角看到的面片默认材质")
    ap.add_argument("--masks_dir", default=None,
                    help="材质分割掩码目录(每图 <name>.png 类别id)；给了用真模型，否则占位启发式")
    args = ap.parse_args()

    import open3d as o3d
    import cv2

    class_order, em = load_radio_params(args.radio_params)
    K = len(class_order)
    idx = {n: i for i, n in enumerate(class_order)}
    print(f"材质类别(K={K}): {class_order}")

    # ---- 网格 + 光线投射场景 ----
    print("读取网格 ...")
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    tris = np.asarray(mesh.triangles)
    nF = len(tris)
    print(f"面片数: {nF}, 顶点数: {len(mesh.vertices)}")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    INVALID = o3d.t.geometry.RaycastingScene.INVALID_ID

    votes = np.zeros((nF, K), np.float32)      # 每面片材质投票累加

    cams = load_cameras(args.cameras)
    print(f"相机数: {len(cams)}，逐相机投射光线投票 ...")
    s = args.pixel_stride
    for ci, cam in enumerate(cams):
        img_path = find_image(args.images, cam["name"])
        if img_path is None:
            print(f"  [跳过] 找不到图像 {cam['name']}"); continue
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (cam["W"], cam["H"])).astype(np.float32) / 255.0
        prob = None
        if args.masks_dir:
            prob = load_mask_prob(args.masks_dir, cam["name"], cam["W"], cam["H"], K)   # 真模型掩码
        if prob is None:
            prob = heuristic_material_prob(img, class_order)      # 占位启发式(兜底)

        W, H, fx, fy = cam["W"], cam["H"], cam["fx"], cam["fy"]
        cx, cy = W / 2.0, H / 2.0
        us = np.arange(0, W, s); vs = np.arange(0, H, s)
        uu, vv = np.meshgrid(us, vs)                              # [h,w]
        dcam = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu, np.float64)], -1)  # 相机系方向
        dworld = dcam.reshape(-1, 3) @ cam["R"].T                 # C2W: world dir
        dworld /= np.linalg.norm(dworld, axis=1, keepdims=True)
        origins = np.tile(cam["pos"], (dworld.shape[0], 1))
        rays = o3d.core.Tensor(np.hstack([origins, dworld]).astype(np.float32),
                               dtype=o3d.core.Dtype.Float32)
        ans = scene.cast_rays(rays)
        fid = ans["primitive_ids"].numpy()                       # [N] 面片id 或 INVALID
        hit = fid != INVALID
        if not hit.any():
            continue
        probs_hit = prob[vv.reshape(-1)[hit], uu.reshape(-1)[hit]]  # [M,K]
        np.add.at(votes, fid[hit].astype(np.int64), probs_hit)
        if (ci + 1) % 10 == 0 or ci == len(cams) - 1:
            print(f"  {ci+1}/{len(cams)} 相机")

    # ---- 面片材质：argmax；无票面片给默认 ----
    seen = votes.sum(1) > 0
    face_mat = np.full(nF, idx[args.default_material], np.int64)
    face_mat[seen] = votes[seen].argmax(1)
    print(f"被看到的面片: {seen.sum()}/{nF} ({seen.mean()*100:.1f}%)")

    os.makedirs(args.out_dir, exist_ok=True)
    # CSV：面片材质 + 电磁参数
    csv_path = os.path.join(args.out_dir, "face_materials.csv")
    with open(csv_path, "w") as f:
        f.write("face_id,material_id,material_name,eps_r,sigma,penetration_loss_db\n")
        for i in range(nF):
            name = class_order[face_mat[i]]
            p = em.get(name, {})
            f.write(f"{i},{face_mat[i]},{name},{p.get('eps_r',1.0)},"
                    f"{p.get('sigma',0.0)},{p.get('penetration_loss_db',0.0)}\n")
    print(f"[导出] {csv_path}")

    # 上色网格：顶点色 = 其相邻面片材质的众数
    vcount = np.zeros((len(mesh.vertices), K), np.float32)
    np.add.at(vcount, tris.reshape(-1), np.repeat(np.eye(K, dtype=np.float32)[face_mat], 3, axis=0))
    vmat = vcount.argmax(1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(_PALETTE[np.clip(vmat,0,len(_PALETTE)-1)] / 255.0)
    ply_path = os.path.join(args.out_dir, "material_mesh.ply")
    o3d.io.write_triangle_mesh(ply_path, mesh)
    print(f"[导出] {ply_path}")

    # 材质分布
    print("\n面片材质分布:")
    for i, n in enumerate(class_order):
        c = int((face_mat == i).sum())
        if c: print(f"  {n:18s}: {c}")
    print(f"\n完成。用 MeshLab 打开 {ply_path} 看材质上色。")


if __name__ == "__main__":
    main()
