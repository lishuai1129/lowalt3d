"""
smooth_face_materials.py · 面片材质空间众数滤波(去椒盐/去伪金属) → 干净版 face_materials.csv + mesh

问题：可微高斯材质投到网格后，墙面上散落孤立的 glass/veg/metal 面片(椒盐噪声)，
且无标签的 metal 会靠 argmax 噪声"假阳"出一堆(RF 假反射体)。
做法：对每个面片取"最近 k 个面片材质的众数"，迭代 N 次。孤立噪声被邻居多数票抹掉，
成片的真实材质(窗阵列/树冠)保留。纯后处理，不重训。

用法：
  python scripts\smooth_face_materials.py ^
    --mesh outputs\scene02_pgsr_hq\mesh\tsdf_fusion_closed.ply ^
    --materials outputs\scene02_pgsr_hq\material_gauss_cons\face_materials.csv ^
    --classes configs\classes.txt --out_dir outputs\scene02_pgsr_hq\material_clean --k 12 --iters 2
可选 --drop_unlabeled metal 把某类(无标签的伪金属)直接并回多数邻居。
"""
import os, sys, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from material_common import load_radio_params, _PALETTE


def mode_filter(centers, mat, K, k, iters, chunk=1000000):
    from scipy.spatial import cKDTree
    tree = cKDTree(centers)
    for it in range(iters):
        new = np.empty_like(mat)
        for s in range(0, len(mat), chunk):
            e = min(s + chunk, len(mat))
            _, idx = tree.query(centers[s:e], k=k)                 # [c,k] 含自身
            nb = mat[idx]                                          # [c,k]
            oh = (nb[:, :, None] == np.arange(K)).sum(1)           # [c,K] 邻域各类计数
            new[s:e] = oh.argmax(1)
        changed = int((new != mat).sum()); mat = new
        print(f"  pass {it+1}/{iters}: 改动 {changed} 面 ({100*changed/len(mat):.2f}%)")
    return mat


def _edge_pairs(tri):
    """返回所有共享边的面对 (f1[], f2[])，全向量化(无 Python 循环/无逐面 set)。"""
    N = len(tri)
    E = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    F = np.tile(np.arange(N), 3)
    order = np.lexsort((E[:, 1], E[:, 0]))
    E, F = E[order], F[order]
    same = (E[1:] == E[:-1]).all(1)                      # 排序后相邻两行=同一条边
    return F[:-1][same], F[1:][same]


def drop_small_clusters(tri, mat, K, min_area, cls_ids=None):
    """把某类(默认 glass)的小连通块【整块】回退为该块周边非目标类的多数材质。
    连通按共享边——成片的真窗/真树不受影响，孤立噪点被整块抹平。
    场景无关的尺度先验：真窗(即使小)也≥几十面，检测层抖动产生的碎玻璃远小于此。"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    N = len(mat)
    tgt = np.isin(mat, list(cls_ids or []))
    if not tgt.any():
        return mat, 0
    a, b = _edge_pairs(tri)
    both = tgt[a] & tgt[b]                               # 只在目标类内部连边
    g = coo_matrix((np.ones(int(both.sum()), np.int8), (a[both], b[both])), shape=(N, N))
    ncomp, lab = connected_components(g, directed=False)
    size = np.bincount(lab, weights=tgt, minlength=ncomp)   # 每块的目标面数
    drop = tgt & (size[lab] < min_area)
    if not drop.any():
        return mat, 0
    # 整块回退：按【连通块】统计周边非目标类邻居的众数，全块统一改成它
    src = np.concatenate([a, b]); dst = np.concatenate([b, a])
    m = drop[src] & ~tgt[dst]
    votes = np.zeros((ncomp, K), np.int32)
    np.add.at(votes, (lab[src[m]], mat[dst[m]]), 1)
    ok = votes.sum(1) > 0                                # 全被目标类包住的块无从判断→不动
    sel = drop & ok[lab]
    new = mat.copy()
    new[sel] = votes.argmax(1)[lab[sel]]
    return new, int(sel.sum())


def infer_up(normals):
    # 上轴 = 法向最常对齐的坐标轴(屋顶+地面这类大片水平面法向≈±up)
    return int(np.argmax([np.mean(np.abs(normals[:, a]) > 0.85) for a in range(3)]))


def geo_gate(centers, normals, mat, class_order, K, up_axis, gate_cos, gate_voxel,
             horiz_forbid, vert_forbid):
    """硬几何门(网格级,确定性)：水平面禁 glass/metal、竖直面禁 asphalt/bg/veg。
    违规面回填其所在【体素】的多数类——体素是固定空间尺度,与面数无关:
    墙上绿/橘椒盡所在体素多数是混凝土→抹平;真树体素多数植被→整棵留;真天窗体素多数玻璃→留。
    (旧版用 k 近邻判"局部多数",但 9.7M 面里 k=48 空间半径极小,一坨椒盐即自成多数被误保护)"""
    up = np.zeros(3); up[up_axis] = 1.0
    nu = np.abs(normals @ up)                                  # |n·up|: ≈1 水平, ≈0 竖直
    horiz = nu > (1 - gate_cos); vert = nu < gate_cos          # 中间(斜面)不管,安全
    forb = np.zeros(len(mat), bool)
    for c in horiz_forbid: forb |= horiz & (mat == c)
    for c in vert_forbid:  forb |= vert & (mat == c)
    key = np.floor(centers / gate_voxel).astype(np.int64)     # 体素分桶
    _, inv = np.unique(key, axis=0, return_inverse=True)
    counts = np.zeros((int(inv.max()) + 1, K), np.int64)
    np.add.at(counts, (inv, mat), 1)
    dom = counts.argmax(1)                                     # 每体素多数类
    new = dom[inv[forb]]                                       # 违规面回填其体素多数
    n = int((new != mat[forb]).sum())                         # 只数真改动(体素本就该类的不计)
    mat = mat.copy(); mat[forb] = new
    hn = [class_order[c] for c in horiz_forbid]; vn = [class_order[c] for c in vert_forbid]
    print(f"几何门(体素{gate_voxel}): up轴={up_axis} 水平{np.mean(horiz)*100:.0f}%禁{hn} 竖直{np.mean(vert)*100:.0f}%禁{vn} → 改 {n} 面")
    return mat


def main():
    ap = argparse.ArgumentParser(description="面片材质众数滤波(去椒盐/伪金属)")
    ap.add_argument("--mesh", required=True, help="几何网格(face 顺序须与 csv 的 face_id 一致)")
    ap.add_argument("--materials", required=True, help="face_materials.csv")
    ap.add_argument("--classes", default=None)
    ap.add_argument("--radio_params", default=os.path.join(os.path.dirname(_HERE), "configs", "radio_params.yaml"))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k", type=int, default=12, help="每面片取最近 k 个面片投票")
    ap.add_argument("--iters", type=int, default=2, help="迭代轮数(2-3 够)")
    ap.add_argument("--drop_unlabeled", default=None, help="某类名(如 metal)先清零再滤波,防无标签假阳")
    ap.add_argument("--geo_gate", action="store_true", help="开硬几何门(网格级确定性,清屋顶玻璃/墙上绿椒盐)")
    ap.add_argument("--gate_cos", type=float, default=0.35, help="法向阈:|n·up|>1-cos 判水平,<cos 判竖直,中间不管")
    ap.add_argument("--gate_voxel", type=float, default=0.3, help="几何门体素边长(COLMAP单位):>椒盐坨尺度且<真树/真天窗;椒盐没清干净就调大")
    ap.add_argument("--up_axis", type=int, default=-1, help="上轴 0/1/2,-1=自动从法向推断")
    ap.add_argument("--horiz_forbid", default="glass,metal", help="水平面禁用类(屋顶不该有玻璃/金属)")
    ap.add_argument("--vert_forbid", default="asphalt,background,vegetation", help="竖直面禁用类(墙不该有沥青/背景/植被)")
    ap.add_argument("--min_win_area", type=int, default=0,
                    help=">0: 该面数以下的玻璃连通块(共享边连通)整块回退为邻居多数材质。滤检测层碎玻璃,场景无关(真窗≥几十面)")
    args = ap.parse_args()

    import open3d as o3d
    class_order, em = load_radio_params(args.radio_params)
    if args.classes:   # 该开关此前只解析不生效:真源是 radio_params.yaml。两者不一致会让 material_id 整体错位
        want = [l.strip() for l in open(args.classes, encoding="utf-8") if l.strip()]
        assert want == class_order, f"--classes 与 radio_params 类别顺序不一致:\n  {want}\n  {class_order}"
    K = len(class_order)
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    V = np.asarray(mesh.vertices); Tri = np.asarray(mesh.triangles)
    centers = V[Tri].mean(1)
    nF = len(Tri)
    print(f"面片 F={nF}  类别 K={K}: {class_order}")

    data = np.genfromtxt(args.materials, delimiter=",", names=True, dtype=None, encoding="utf-8")
    mat = data["material_id"].astype(np.int64)
    assert len(mat) == nF, f"csv 行数 {len(mat)} != 网格面数 {nF}(mesh 与 csv 不配对)"

    def dist(tag):
        print(tag, {class_order[i]: int((mat == i).sum()) for i in range(K) if (mat == i).sum()})
    dist("滤波前:")

    if args.drop_unlabeled and args.drop_unlabeled in class_order:
        did = class_order.index(args.drop_unlabeled)
        n = int((mat == did).sum()); mat[mat == did] = -1     # 标记待众数重投
        # -1 面用众数滤波补：临时设为最近非 -1 的多数——直接进滤波，众数会把 -1 邻域按多数填
        # 简化：先把 -1 当作一个"空"类不参与计数 → 用一次滤波填充
        print(f"清零 {args.drop_unlabeled}: {n} 面 待邻域重投")
        from scipy.spatial import cKDTree
        tree = cKDTree(centers); bad = mat < 0
        if bad.any():
            _, idx = tree.query(centers[bad], k=args.k + 1)
            oh = (mat[idx][:, :, None] == np.arange(K)).sum(1)    # -1 不在 [0,K) 不计
            fill = oh.argmax(1)
            novote = oh.sum(1) == 0        # 邻居全是被清类(大块) → argmax 会默认落到 id0=background
            if novote.any():               # 改用全局多数,别把整块金属silently变成背景
                fill[novote] = np.bincount(mat[mat >= 0], minlength=K).argmax()
            mat[bad] = fill
        assert (mat >= 0).all(), "仍有未回填面(-1);写 CSV 时 class_order[-1] 会静默变成最后一类"

    if args.geo_gate:
        mesh.compute_triangle_normals()
        normals = np.asarray(mesh.triangle_normals)
        up_axis = args.up_axis if args.up_axis >= 0 else infer_up(normals)
        ids = lambda s: [class_order.index(x) for x in s.split(",") if x and x in class_order]
        mat = geo_gate(centers, normals, mat, class_order, K, up_axis,
                       args.gate_cos, args.gate_voxel, ids(args.horiz_forbid), ids(args.vert_forbid))
        dist("几何门后:")

    print("众数滤波:")
    mat = mode_filter(centers, mat, K, args.k, args.iters)
    dist("滤波后:")

    if args.min_win_area > 0:
        gid = class_order.index("glass") if "glass" in class_order else -1
        mat, nd = drop_small_clusters(Tri, mat, K, args.min_win_area, [gid] if gid >= 0 else None)
        if nd:
            print(f"碎玻璃连通块过滤(min_area={args.min_win_area}): 回退 {nd} 面")
        dist("碎块过滤后:")

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
    # 3x3 全 concrete(3) 里塞 1 个孤立 glass(4) → 众数滤波应抹掉
    centers = np.array([[x, y, 0] for y in range(3) for x in range(3)], float)
    mat = np.full(9, 3, np.int64); mat[4] = 4
    out = mode_filter(centers, mat, K=6, k=5, iters=1)
    assert out[4] == 3, "孤立 glass 应被众数抹成 concrete"
    # 几何门: 水平屋顶(法向+z)上 1 个 glass 被体素多数(concrete)票走; 整片 glass 天窗应保留
    cls = ["background", "vegetation", "asphalt", "concrete", "glass", "metal"]
    nrm = np.tile([0.0, 0.0, 1.0], (9, 1))                 # 全水平面(屋顶)
    m2 = np.full(9, 3, np.int64); m2[4] = 4                # 中心 1 块 glass, 体素=3 使 9 面同桶
    g = geo_gate(centers, nrm, m2, cls, 6, 2, 0.35, 3.0, [4, 5], [2, 0, 1])
    assert g[4] == 3, "屋顶孤立 glass 应被体素多数票走"
    m3 = np.full(9, 4, np.int64)                           # 整片 glass(真天窗)
    g3 = geo_gate(centers, nrm, m3, cls, 6, 2, 0.35, 3.0, [4, 5], [2, 0, 1])
    assert (g3 == 4).all(), "整片水平 glass(天窗)应被体素多数保护"
    # 碎块过滤: 孤立 1 个 glass(面0,四邻皆 concrete) + 连成片的 4 个 glass(面1-4) → 小块回退, 大片保留
    tri = np.array([[0,1,2],[3,4,5],[3,5,6],[4,6,7],[4,5,6],[0,1,8]], int)
    #                              F1        F2        F3       F4   F5(邻F0,concrete)
    m4 = np.full(6, 3, np.int64); m4[0] = 4                 # 面0 孤立 glass
    m4[1] = m4[2] = m4[3] = m4[4] = 4                          # 面1..4 成块 glass
    o4, n4 = drop_small_clusters(tri, m4, 6, 4, [4])
    assert o4[0] == 3 and (o4[[1,2,3,4]] == 4).all(), f"孤立玻璃应回退、整片保留, got {o4}"
    assert n4 == 1
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
