"""
label_windows_geom.py · 几何法判窗:竖直立面上"凹陷的面"= 窗户 → glass(不重训/不靠2D)

动机:俯视航拍从没拍进窗洞里,凹坑内表面无观测 → 2D检测/光线投票/降阈值重训都填不进。
但凹坑本身就是几何信号:立面墙是沿法向最靠前的面,窗户凹进去 → 相对本地墙面"回退"。
做法:取竖直面,对每个面沿其法向投影,用邻域高分位(墙前沿)减自身投影 = 回退深度 recess;
recess 在 [min,max] 且有足够窗邻居支撑 → 判 glass。纯几何后处理,读干净版 csv,写新 csv+ply。

假设:mesh 法向朝外一致(TSDF 通常如此)。若结果反了(墙被判成窗),加 --flip。
  用法:
    python scripts\label_windows_geom.py ^
      --mesh outputs\scene02_pgsr_hq\mesh\tsdf_fusion_post.ply ^
      --materials outputs\scene02_pgsr_hq\material_clean\face_materials.csv ^
      --classes configs\classes.txt --up_axis 2 ^
      --out_dir outputs\scene02_pgsr_hq\material_win
  先跑一次看打印的 recess 分位直方图,再调 --recess_min/--recess_max(COLMAP单位)。
  ⚠️ 单位换算:1 COLMAP 单位 = geo_transform.json 的 scale(scene02 = 33.06 m)。
     缺省 0.01~0.06 = 0.33~2 m,对应常见窗洞进深;别按"厘米"理解这些数。
"""
import os, sys, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from material_common import load_radio_params, _PALETTE


def find_windows(centers, normals, up_axis, gate_cos, k, recess_min, recess_max, min_support,
                 chunk=500000):
    from scipy.spatial import cKDTree
    up = np.zeros(3); up[up_axis] = 1.0
    vmask = np.abs(normals @ up) < gate_cos                 # 竖直面(法向≈水平)
    vidx = np.nonzero(vmask)[0]
    C = centers[vidx]; Nrm = normals[vidx]
    k = min(k, len(vidx))
    tree = cKDTree(C)
    idx = np.empty((len(C), k), np.int32)                   # int32: 235万×60 省一半内存
    recess = np.empty(len(C))
    for s in range(0, len(C), chunk):                       # 分块: 否则 C[idx] 单块 3.4GB
        e = min(s + chunk, len(C))
        _, ii = tree.query(C[s:e], k=k)                     # [c,k] 邻域(含自身)
        idx[s:e] = ii
        p = np.einsum("ijk,ik->ij", C[ii], Nrm[s:e])       # 邻居中心投到"本面法向"
        # 本地墙前沿(最靠外) - 自身投影 = 回退深度
        recess[s:e] = np.percentile(p, 90, axis=1) - np.einsum("ik,ik->i", C[s:e], Nrm[s:e])
    win = (recess > recess_min) & (recess < recess_max)     # 回退适中 = 窗(太深=楼边/缺口,排除)
    win &= win[idx].sum(1) >= min_support                   # 邻域窗支撑(去孤立假阳)
    return vidx, win, recess


def main():
    ap = argparse.ArgumentParser(description="几何法判窗(凹陷竖直面→glass)")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--materials", required=True, help="干净版 face_materials.csv")
    ap.add_argument("--classes", default=None)
    ap.add_argument("--radio_params", default=os.path.join(os.path.dirname(_HERE), "configs", "radio_params.yaml"))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--up_axis", type=int, default=2, help="上轴 0/1/2")
    ap.add_argument("--gate_cos", type=float, default=0.35, help="|n·up|<此值判竖直面")
    ap.add_argument("--k", type=int, default=60, help="邻域面数(需大到跨过窗、够到墙;窗漏判就调大)")
    # scene02: 1 COLMAP 单位 = 33.06 m(geo_transform.json 的 scale)。窗洞进深 0.3~2 m → 0.01~0.06
    ap.add_argument("--recess_min", type=float, default=0.01, help="回退下限(COLMAP单位;×scale=米。0.01≈0.33m,滤墙面噪声)")
    ap.add_argument("--recess_max", type=float, default=0.06, help="回退上限(0.06≈2m,排楼边/大缺口)")
    ap.add_argument("--min_support", type=int, default=8, help="邻域窗支撑数(去孤立假阳)")
    ap.add_argument("--relabel_from", default="concrete,background", help="只把这些类的凹陷面改判 glass")
    ap.add_argument("--flip", action="store_true", help="法向朝内时翻转(墙被判成窗时加)")
    args = ap.parse_args()

    import open3d as o3d
    class_order, em = load_radio_params(args.radio_params)
    if args.classes:   # 该开关此前只解析不生效:真源是 radio_params.yaml。两者不一致会让 material_id 整体错位
        want = [l.strip() for l in open(args.classes, encoding="utf-8") if l.strip()]
        assert want == class_order, f"--classes 与 radio_params 类别顺序不一致:\n  {want}\n  {class_order}"
    K = len(class_order); gid = class_order.index("glass")
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    mesh.compute_triangle_normals()
    V = np.asarray(mesh.vertices); Tri = np.asarray(mesh.triangles)
    centers = V[Tri].mean(1)
    normals = np.asarray(mesh.triangle_normals) * (-1.0 if args.flip else 1.0)
    nF = len(Tri)
    print(f"面片 F={nF}  类别 K={K}: {class_order}  glass id={gid}")

    data = np.genfromtxt(args.materials, delimiter=",", names=True, dtype=None, encoding="utf-8")
    mat = data["material_id"].astype(np.int64)
    assert len(mat) == nF, f"csv 行数 {len(mat)} != 网格面数 {nF}"

    vidx, win, recess = find_windows(centers, normals, args.up_axis, args.gate_cos,
                                     args.k, args.recess_min, args.recess_max, args.min_support)
    q = np.percentile(recess, [50, 75, 90, 95, 99])
    print(f"竖直面 {len(vidx)}  recess 分位 50/75/90/95/99 = " + "/".join(f"{v:.3f}" for v in q))
    print(f"  (调参: recess_min/max 参考上面分位; 现窗候选 {int(win.sum())} 面)")

    from_ids = {class_order.index(x) for x in args.relabel_from.split(",") if x in class_order}
    tgt = vidx[win]
    tgt = tgt[np.isin(mat[tgt], list(from_ids))]           # 只改指定类的凹陷面
    print(f"改判 glass: {len(tgt)} 面 (原为 {args.relabel_from})")
    mat[tgt] = gid

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "face_materials.csv")
    with open(csv_path, "w") as f:
        f.write("face_id,material_id,material_name,eps_r,sigma,penetration_loss_db\n")
        for i in range(nF):
            name = class_order[mat[i]]; p = em.get(name, {})
            f.write(f"{i},{mat[i]},{name},{p.get('eps_r',1.0)},{p.get('sigma',0.0)},{p.get('penetration_loss_db',0.0)}\n")
    vcount = np.zeros((len(V), K), np.float32)
    np.add.at(vcount, Tri.reshape(-1), np.repeat(np.eye(K, dtype=np.float32)[mat], 3, axis=0))
    mesh.vertex_colors = o3d.utility.Vector3dVector(_PALETTE[np.clip(vcount.argmax(1), 0, len(_PALETTE)-1)] / 255.0)
    ply_path = os.path.join(args.out_dir, "material_mesh.ply")
    o3d.io.write_triangle_mesh(ply_path, mesh)
    print(f"[导出] {csv_path}\n[导出] {ply_path}")


def _selfcheck():
    # 5x5 竖直墙(法向+x, up=z), 中间 3x3 回退 0.2 → 应判窗; 平墙面不判
    pts, win_gt = [], []
    for y in range(5):
        for z in range(5):
            rec = 1 <= y <= 3 and 1 <= z <= 3
            pts.append([-0.2 if rec else 0.0, y, z]); win_gt.append(rec)
    C = np.array(pts, float); N = np.tile([1.0, 0, 0], (25, 1)); win_gt = np.array(win_gt)
    vidx, win, _ = find_windows(C, N, up_axis=2, gate_cos=0.35, k=25,
                                recess_min=0.05, recess_max=0.6, min_support=3)
    got = np.zeros(25, bool); got[vidx[win]] = True
    assert (got == win_gt).all(), f"凹陷 3x3 应判窗, 得 {got.sum()} 应 {win_gt.sum()}"
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
