"""
export_dae.py  ·  材质网格 → Wireless InSite 用的 .dae（面片按材质分组 + 命名） + InSite 材质表

目的：把我们的 mesh + face_materials.csv 导成 Wireless InSite 能读的模型：
      面片按【InSite 材质名】分组绑定，射线命中即知材质；并附一份 InSite 格式材质表。

我们的类 → InSite 材质映射(用参考 materials.csv 的值)：
      concrete→Concrete, asphalt→Asphalt, glass→Glass, metal→Metal(PEC),
      vegetation→Wood(近似,InSite 无植被), background→Concrete(兜底)

依赖：pip install pycollada
运行：
  python scripts\export_dae.py ^
    --mesh outputs\%SCENE%_pgsr\mesh\tsdf_fusion_post.ply ^
    --materials outputs\%SCENE%_pgsr\material_gauss_cons\face_materials.csv ^
    --out_dir outputs\%SCENE%_pgsr\insite --target_faces 500000 --scale 1.0
输出：insite\scene.dae + insite\materials_insite.csv

⚠️ 尺度：InSite 用【米】。我们坐标是 COLMAP 任意单位；`--scale` 填 米/单位（地理配准后
   得到），否则几何形状对但绝对尺寸不对，路损/绕射会偏。
⚠️ 朝向：InSite 一般 +Z 向上；COLMAP 朝向任意，导入后可能需在 InSite 里旋转摆正。
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from material_common import _DEFAULT_CLASS_ORDER

# 我们的类 → (InSite 材质名, 电导率σ, 相对介电εr, 粗糙度, 厚度m, 模型, 颜色RGB)
INSITE_MAP = {
    "concrete":    ("Concrete", 0.015,  7.0,  0.0, 0.3,   "LayeredDielectric",   (0.72, 0.72, 0.72)),
    "asphalt":     ("Asphalt",  0.0005, 5.72, 0.0, 0.3,   "DielectricHalfspace", (0.30, 0.30, 0.30)),
    "glass":       ("Glass",    0.0,    2.4,  0.0, 0.003, "LayeredDielectric",   (0.35, 0.60, 0.90)),
    "metal":       ("Metal",    None,   None, 0.0, 0.0,   "PEC",                 (0.90, 0.72, 0.20)),
    "vegetation":  ("Wood",     0.0,    5.0,  0.0, 0.03,  "LayeredDielectric",   (0.40, 0.62, 0.32)),
    "background":  ("Concrete", 0.015,  7.0,  0.0, 0.3,   "LayeredDielectric",   (0.72, 0.72, 0.72)),
}
assert set(INSITE_MAP) >= set(_DEFAULT_CLASS_ORDER), \
    f"INSITE_MAP 缺类(会静默落到 background→Concrete): {set(_DEFAULT_CLASS_ORDER) - set(INSITE_MAP)}"


def write_city(path, V, T, face_insite_names, used, name2params):
    """写 Wireless InSite 原生 .city：每面片直接带正确材质【索引】，绕过 COLLADA 导入丢材质的坑。

    格式(逆向自 InSite 生成的 .city)：头部 → N 个材质定义(各带 Material <idx>) →
    一个 structure_group/structure/sub_structure 包住所有面片，每面片一行 Material <idx>。
    """
    idx_of = {n: i for i, n in enumerate(used)}
    face_idx = np.array([idx_of[n] for n in face_insite_names], dtype=int)
    with open(path, "w", encoding="utf-8") as f:
        f.write("Format type:keyword version: 1.1.0\n")
        f.write("begin_<city> scene\n")
        f.write("begin_<reference> \ncartesian\nlongitude 0.000000000000000\n"
                "latitude 0.000000000000000\nvisible no\nterrain\nend_<reference>\n")
        # 材质定义(全部用 LayeredDielectric 单层——语法与 InSite 生成的完全一致，保证可读)
        for i, name in enumerate(used):
            sg, er, ro, th, col = name2params[name]
            f.write(f"begin_<Material> {name}\nMaterial {i}\nLayeredDielectric\n")
            f.write("begin_<Color> \n")
            f.write(f"ambient {col[0]:.6f} {col[1]:.6f} {col[2]:.6f} 1.000000\n")
            f.write(f"diffuse {col[0]:.6f} {col[1]:.6f} {col[2]:.6f} 1.000000\n")
            f.write(f"specular {col[0]:.6f} {col[1]:.6f} {col[2]:.6f} 1.000000\n")
            f.write("emission 0.000000 0.000000 0.000000 0.000000\nshininess 5.000000\nend_<Color>\n")
            f.write("diffuse_scattering_model none\nfields_diffusively_scattered 0.400000\n")
            f.write("cross_polarized_power 0.400000\ndirective_alpha 4\ndirective_beta 4\n")
            f.write("directive_lambda 0.750000\nsubdivide_facets yes\n")
            f.write("reflection_coefficient_options do_not_use\nnLayers 1\n")
            f.write(f"begin_<DielectricLayer> {name}\n")
            f.write(f"conductivity {sg:.3e}\npermittivity {er:.6f}\n")
            f.write(f"roughness {ro:.3e}\nthickness {th:.3e}\n")
            f.write("end_<DielectricLayer>\nBackingLayer NoBackingLayer\nend_<Material>\n")
        # 面片(显式坐标，与 InSite .city 一致：begin_<face>/Material idx/nVertices 3/三顶点)
        f.write("begin_<structure_group> \nbegin_<structure> \nbegin_<sub_structure> \n")
        buf = []
        for fi in range(len(T)):
            a, b, c = T[fi]
            buf.append(
                f"begin_<face> \nMaterial {int(face_idx[fi])}\nnVertices 3\n"
                f"{V[a, 0]:.10f} {V[a, 1]:.10f} {V[a, 2]:.10f}\n"
                f"{V[b, 0]:.10f} {V[b, 1]:.10f} {V[b, 2]:.10f}\n"
                f"{V[c, 0]:.10f} {V[c, 1]:.10f} {V[c, 2]:.10f}\n"
                f"end_<face>\n")
            if len(buf) >= 20000:
                f.write("".join(buf)); buf = []
        f.write("".join(buf))
        f.write("end_<sub_structure>\nend_<structure>\nend_<structure_group>\nend_<city>\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--materials", required=True, help="face_materials.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_faces", type=int, default=500000, help="抽面到此数量(InSite 性能);0=不抽")
    ap.add_argument("--scale", type=float, default=1.0, help="米/COLMAP单位(地理配准后填;无 geo_transform 时用)")
    ap.add_argument("--geo_transform", default=None, help="georeference.py 的 geo_transform.json(优先;含米制+ENU朝向)")
    ap.add_argument("--ground_z0", action="store_true",
                    help="把地面移到 Z≈0（ENU 原点在无人机飞行高度→地面会在 z≈-80；加此项便于放 TX/RX，不影响相对几何/路损）")
    ap.add_argument("--clean_floaters", type=float, default=0.0,
                    help=">0 时移除小的悬空碎块：删掉面数占比 < 此阈值的连通分量(如 0.02=删<2%的碎块)，去重建噪声、加速射线追踪")
    args = ap.parse_args()

    import open3d as o3d
    import collada
    from scipy.spatial import cKDTree

    # ---- 网格 + 每面片我们的类别 ----
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    V = np.asarray(mesh.vertices); T = np.asarray(mesh.triangles)
    rec = np.genfromtxt(args.materials, delimiter=",", names=True, dtype=None, encoding="utf-8")
    face_cls = np.asarray(rec["material_name"]).astype(str)     # 我们的类名/面
    assert len(face_cls) == len(T), f"face 数不一致: csv {len(face_cls)} vs mesh {len(T)}"
    print(f"原始面片 {len(T)}，顶点 {len(V)}")

    # ---- (可选)清理悬空碎块：删掉小连通分量(重建噪声)，去毛边 + 加速射线追踪 ----
    if args.clean_floaters > 0:
        tri_clusters, cl_ntri, _ = mesh.cluster_connected_triangles()
        tri_clusters = np.asarray(tri_clusters); cl_ntri = np.asarray(cl_ntri)
        thresh = args.clean_floaters * len(T)
        keep_mask = cl_ntri[tri_clusters] >= thresh            # 每面片：其所属连通块够大才留
        mesh.remove_triangles_by_mask(~keep_mask)
        mesh.remove_unreferenced_vertices()
        face_cls = face_cls[keep_mask]
        V = np.asarray(mesh.vertices); T = np.asarray(mesh.triangles)
        print(f"清理悬空碎块(阈值 {thresh:.0f} 面): 删 {int((~keep_mask).sum())} 面，剩 {len(T)}")

    # ---- 抽面(减小 .dae) + 材质重投票(最近原面片) ----
    if args.target_faces and len(T) > args.target_faces:
        sm = mesh.simplify_quadric_decimation(args.target_faces)
        V2 = np.asarray(sm.vertices); T2 = np.asarray(sm.triangles)
        oc = V[T].mean(1); nc = V2[T2].mean(1)
        _, idx = cKDTree(oc).query(nc)
        face_cls = face_cls[idx]; V, T = V2, T2
        print(f"抽面后 {len(T)} 面 / {len(V)} 顶点")

    # ---- 米制/朝向变换：优先 geo_transform(米制+ENU摆正)，否则仅 scale ----
    if args.geo_transform and os.path.isfile(args.geo_transform):
        import json
        gt = json.load(open(args.geo_transform, encoding="utf-8"))
        s = float(gt["scale"]); Rg = np.array(gt["R"]); tg = np.array(gt["t"])
        V = s * (V @ Rg.T) + tg                                  # → ENU 米制(+Z 向上，正合 InSite)
        print(f"应用地理配准: scale={s:.4f} m/单位，已转 ENU 米制+摆正")
    elif args.scale != 1.0:
        V = V * float(args.scale)                                # → 米(手动尺度)

    # ---- (可选)接地：把地面移到 Z≈0（ENU 原点在无人机高度→地面 z≈-80，放收发机不便）----
    if args.ground_z0:
        z0 = float(np.percentile(V[:, 2], 1.0))                  # 1%分位≈地面(避开个别低于地面的噪点)
        V[:, 2] -= z0
        print(f"接地: Z 整体 {-z0:+.2f} m → 地面≈0（仅平移，不影响相对几何/路损）")

    # ---- 我们的类 → InSite 材质名，按 InSite 名分组 ----
    insite = np.array([INSITE_MAP.get(c, INSITE_MAP["background"])[0] for c in face_cls])
    used = [n for n in ["Concrete", "Asphalt", "Glass", "Metal", "Wood", "Brick", "Dry Earth"] if (insite == n).any()]
    print(f"用到的 InSite 材质: {used}")

    # ---- 构建 COLLADA(.dae)：一个几何，多个按材质分组的 triangleset ----
    dae = collada.Collada()
    vsrc = collada.source.FloatSource("verts-array", V.astype(np.float32).flatten(), ('X', 'Y', 'Z'))
    geom = collada.geometry.Geometry(dae, "geom0", "model", [vsrc])
    matnodes = []
    # 颜色取该 InSite 名对应(找一个映射到它的类)
    name2color = {}
    for c, v in INSITE_MAP.items():
        name2color.setdefault(v[0], v[6])
    for name in used:
        col = name2color.get(name, (0.7, 0.7, 0.7))
        eff = collada.material.Effect("eff_" + name, [], "phong",
                                      diffuse=(col[0], col[1], col[2], 1.0), specular=(0, 0, 0, 1))
        mat = collada.material.Material(name, name, eff)         # id=name=显示名 → InSite 读到
        dae.effects.append(eff); dae.materials.append(mat)
        il = collada.source.InputList(); il.addInput(0, 'VERTEX', "#verts-array")
        idx = T[insite == name].reshape(-1).astype(np.int32)     # 该材质的面片顶点索引
        triset = geom.createTriangleSet(idx, il, name)           # material symbol = name
        geom.primitives.append(triset)
        matnodes.append(collada.scene.MaterialNode(name, mat, inputs=[]))
    dae.geometries.append(geom)
    node = collada.scene.Node("node0", children=[collada.scene.GeometryNode(geom, matnodes)])
    scn = collada.scene.Scene("scene0", [node]); dae.scenes.append(scn); dae.scene = scn

    os.makedirs(args.out_dir, exist_ok=True)
    dae_path = os.path.join(args.out_dir, "scene.dae")
    dae.write(dae_path)
    print(f"[导出] {dae_path}")

    # ---- InSite 原生 .city（推荐！材质按面片写死，不经 COLLADA 导入器）----
    name2params = {}
    for c, v in INSITE_MAP.items():
        nm = v[0]
        sg = 1e7 if v[1] is None else v[1]   # PEC/金属 → 高导体近似(用 LayeredDielectric 语法表达)
        er = 1.0 if v[2] is None else v[2]
        name2params.setdefault(nm, (sg, er, v[3], v[4], name2color.get(nm, (0.7, 0.7, 0.7))))
    city_path = os.path.join(args.out_dir, "scene.city")
    write_city(city_path, V, T, insite, used, name2params)
    print(f"[导出] {city_path}  <- InSite 直接 File>Open 加载它（材质已正确分组）")

    # ---- InSite 格式材质表 ----
    csv_path = os.path.join(args.out_dir, "materials_insite.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("material_id,material_name,conductivity,permittivity,roughness,thickness,model\n")
        for i, name in enumerate(used):
            v = next(x for x in INSITE_MAP.values() if x[0] == name)
            cond = "" if v[1] is None else v[1]
            perm = "" if v[2] is None else v[2]
            f.write(f"{i},{name},{cond},{perm},{v[3]},{v[4]},{v[5]}\n")
    print(f"[导出] {csv_path}")
    print("\n面片材质分布(InSite 名):")
    for name in used:
        print(f"  {name:10s}: {int((insite==name).sum())}")
    geo_applied = bool(args.geo_transform and os.path.isfile(args.geo_transform))
    if args.scale == 1.0 and not geo_applied:
        print("\n[注意] scale=1（未配准），尺寸非米制；InSite 路损需先地理配准后重导(--geo_transform)。")


if __name__ == "__main__":
    main()
