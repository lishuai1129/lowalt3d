r"""
export_gltf_ue.py  ·  材质网格 → Cesium for Unreal / UE 用的 .glb（按电磁类分材质槽）
                        + UE 材质表(P.2040) + Cesium 锚点经纬度 + TX 建议位

目的：把我们的 mesh + face_materials.csv 导成 UE 能直接用的模型：
      · 面片按【电磁类名】拆成独立材质槽 EM_<class> → UE 里每槽绑一个 PhysicalMaterial，
        LineTrace 命中即可 GetPhysicalMaterial 查电磁参数（射频射线追踪的关键）。
      · 应用地理配准 geo_transform → 米制；坐标轴转成 glTF(Y-up,右手)，导入 UE 后直立不镜像。
      · 反算模型质心的经纬高 → Cesium GlobeAnchor 锚点（把模型落到真实地球位置）。

坐标链：COLMAP任意单位 --geo--> ENU米(E,N,U,Z-up) --轴换--> glTF(x=E, y=U, z=-N, Y-up右手)
        UE 导入后：UE_X=E, UE_Y=N, UE_Z=U（左手Z-up），乘 100 得厘米。

依赖：pip install trimesh  （open3d/scipy 已装）
运行：
  python scripts\export_gltf_ue.py ^
    --mesh outputs\%SCENE%_pgsr\mesh\tsdf_fusion_post.ply ^
    --materials outputs\%SCENE%_pgsr\material_gauss_cons\face_materials.csv ^
    --geo_transform outputs\%SCENE%_pgsr\geo_transform.json ^
    --out_dir outputs\%SCENE%_pgsr\ue --target_faces 300000
输出：ue\scene_ue.glb + ue\ue_materials.csv + ue\ue_anchor.json
"""
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from material_common import load_radio_params, _PALETTE, _DEFAULT_CLASS_ORDER

R_E = 6378137.0            # 地球半径(米)，与 georeference.py 一致
D2R = np.pi / 180.0


def enu_to_geodetic(E, N, U, ref):
    """本地 ENU(米) → 经纬高，geodetic_to_enu 的逆(小场景切平面近似)。"""
    lat0, lon0, alt0 = ref
    lat = lat0 + (N / R_E) / D2R
    lon = lon0 + (E / (R_E * np.cos(lat0 * D2R))) / D2R
    alt = alt0 + U
    return float(lat), float(lon), float(alt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default=None, help="几何网格(tsdf_fusion_post.ply)；用 --materials 时需要")
    ap.add_argument("--materials", default=None, help="face_materials.csv（每面片类名）")
    ap.add_argument("--material_ply", default=None,
                    help="带材质颜色的网格(material_mesh.ply)：几何+材质一体，从顶点色反推类别；"
                         "可替代 --mesh/--materials（当 csv 被占用时用）")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_faces", type=int, default=300000, help="抽面到此数(UE交互性);0=不抽")
    ap.add_argument("--geo_transform", default=None, help="geo_transform.json(优先;米制+ENU)")
    ap.add_argument("--scale", type=float, default=1.0, help="米/COLMAP单位(无 geo_transform 时用)")
    ap.add_argument("--radio_params", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "radio_params.yaml"))
    ap.add_argument("--tx_up_frac", type=float, default=1.4,
                    help="TX 高度 = 顶部 + (该值-1)*竖直跨度；与 simple_pathloss 对齐")
    args = ap.parse_args()

    import open3d as o3d
    import trimesh
    from scipy.spatial import cKDTree

    class_order, em = load_radio_params(args.radio_params)
    freq_mhz = 3500.0
    try:
        import yaml
        freq_mhz = float(yaml.safe_load(open(args.radio_params, encoding="utf-8")).get("frequency_mhz", 3500))
    except Exception:
        pass
    cls2idx = {c: i for i, c in enumerate(_DEFAULT_CLASS_ORDER)}

    # ---- 网格 + 每面片电磁类：两种来源 ----
    classes_arr = np.array(_DEFAULT_CLASS_ORDER)
    if args.material_ply:
        mesh = o3d.io.read_triangle_mesh(args.material_ply)
        V = np.asarray(mesh.vertices); T = np.asarray(mesh.triangles)
        assert mesh.has_vertex_colors(), "material_ply 无顶点色，无法反推材质"
        vc = (np.asarray(mesh.vertex_colors) * 255.0).round().astype(int)     # [Nv,3]
        pal = _PALETTE.astype(int)                                            # [7,3]
        vidx = ((vc[:, None, :] - pal[None, :, :]) ** 2).sum(-1).argmin(1)    # 每顶点最近调色板类
        a, b, c = vidx[T[:, 0]], vidx[T[:, 1]], vidx[T[:, 2]]                 # 面片三顶点类
        face_idx = np.where(a == b, a, np.where(a == c, a, np.where(b == c, b, a)))  # 多数票(向量化)
        face_cls = classes_arr[face_idx]
        print(f"从 material_ply 顶点色反推材质：面片 {len(T)}，顶点 {len(V)}")
    else:
        assert args.mesh and args.materials, "需 --material_ply，或同时给 --mesh 与 --materials"
        mesh = o3d.io.read_triangle_mesh(args.mesh)
        V = np.asarray(mesh.vertices); T = np.asarray(mesh.triangles)
        rec = np.genfromtxt(args.materials, delimiter=",", names=True, dtype=None, encoding="utf-8")
        face_cls = np.asarray(rec["material_name"]).astype(str)
        assert len(face_cls) == len(T), f"face 数不一致: csv {len(face_cls)} vs mesh {len(T)}"
        print(f"原始面片 {len(T)}，顶点 {len(V)}")

    # ---- 抽面 + 材质重投票(最近原面片) ----
    if args.target_faces and len(T) > args.target_faces:
        sm = mesh.simplify_quadric_decimation(args.target_faces)
        V2 = np.asarray(sm.vertices); T2 = np.asarray(sm.triangles)
        oc = V[T].mean(1); nc = V2[T2].mean(1)
        _, idx = cKDTree(oc).query(nc)
        face_cls = face_cls[idx]; V, T = V2, T2
        print(f"抽面后 {len(T)} 面 / {len(V)} 顶点")

    # ---- 米制 + ENU：优先 geo_transform ----
    ref = None
    if args.geo_transform and os.path.isfile(args.geo_transform):
        gt = json.load(open(args.geo_transform, encoding="utf-8"))
        s = float(gt["scale"]); Rg = np.array(gt["R"]); tg = np.array(gt["t"])
        V = s * (V @ Rg.T) + tg                       # → ENU 米(E,N,U)
        ref = (gt["ref_lat"], gt["ref_lon"], gt["ref_alt"])
        print(f"应用地理配准: scale={s:.4f} m/单位 → ENU 米制")
    elif args.scale != 1.0:
        V = V * float(args.scale)
        print(f"应用手动尺度: {args.scale} m/单位（无经纬锚点）")

    # ---- 质心居中；反算锚点经纬高 ----
    C = V.mean(0)                                     # ENU 质心(米)
    Vc = V - C                                        # 居中(米，仍 E,N,U)
    E, N, U = Vc[:, 0], Vc[:, 1], Vc[:, 2]
    lo, hi = Vc.min(0), Vc.max(0); ext = hi - lo
    anchor = None
    if ref is not None:
        anchor = enu_to_geodetic(C[0], C[1], C[2], ref)   # 质心处经纬高
        print(f"Cesium 锚点(质心): lat={anchor[0]:.7f} lon={anchor[1]:.7f} alt={anchor[2]:.2f} m")

    # ---- 轴换到 glTF(Y-up,右手): x=E, y=U, z=-N ----
    Vg = np.column_stack([E, U, -N]).astype(np.float32)   # 米

    # ---- 按电磁类拆 submesh，各绑一个命名 PBR 材质(颜色=调色板) ----
    scene = trimesh.Scene()
    used = []
    for cls in class_order:
        fmask = face_cls == cls
        if not fmask.any():
            continue
        f = T[fmask]
        uv, inv = np.unique(f, return_inverse=True)
        sub = trimesh.Trimesh(vertices=Vg[uv], faces=inv.reshape(-1, 3), process=False)
        col = _PALETTE[cls2idx.get(cls, 0)]
        mat = trimesh.visual.material.PBRMaterial(
            name=f"EM_{cls}",
            baseColorFactor=[float(col[0])/255, float(col[1])/255, float(col[2])/255, 1.0],
            metallicFactor=(1.0 if cls == "metal" else 0.0),
            roughnessFactor=0.9)
        sub.visual = trimesh.visual.TextureVisuals(material=mat)
        scene.add_geometry(sub, geom_name=f"EM_{cls}", node_name=f"EM_{cls}")
        used.append(cls)
    print(f"材质槽: {['EM_'+c for c in used]}")

    os.makedirs(args.out_dir, exist_ok=True)
    glb_path = os.path.join(args.out_dir, "scene_ue.glb")
    scene.export(glb_path)
    print(f"[导出] {glb_path}  (米制；UE 导入设 Uniform Scale 100 或勾 Convert Scene Unit)")

    # ---- UE 材质表(P.2040 + InSite 模型) ----
    def insite_model(cls, p):
        if cls == "metal":
            return "PEC"
        if cls in ("asphalt",):
            return "DielectricHalfspace"
        return "LayeredDielectric"
    csv_path = os.path.join(args.out_dir, "ue_materials.csv")
    with open(csv_path, "w", encoding="utf-8") as fp:
        fp.write("Name,eps_r,sigma,penetration_loss_db,model,R,G,B\n")
        for cls in used:
            p = em.get(cls, {})
            col = _PALETTE[cls2idx.get(cls, 0)]
            fp.write(f"EM_{cls},{p.get('eps_r',1.0)},{p.get('sigma',0.0)},"
                     f"{p.get('penetration_loss_db',0.0)},{insite_model(cls,p)},"
                     f"{int(col[0])},{int(col[1])},{int(col[2])}\n")
    print(f"[导出] {csv_path}")

    # ---- 锚点 + TX 建议(UE 厘米，X=E,Y=N,Z=U) ----
    tx_up_m = hi[2] + (args.tx_up_frac - 1.0) * ext[2]      # 顶部上方(米)
    tx_ue_cm = [0.0, 0.0, float(tx_up_m) * 100.0]
    meta = dict(
        freq_mhz=freq_mhz,
        units="glb=meters; UE uses cm (×100). Import: Uniform Scale 100 或 Convert Scene Unit.",
        cesium_anchor=(dict(latitude=anchor[0], longitude=anchor[1], height=anchor[2])
                       if anchor else None),
        tx_suggest_ue_cm=tx_ue_cm,
        bbox_ue_cm=dict(min=[float(x)*100 for x in lo], max=[float(x)*100 for x in hi]),
        extent_m=[float(x) for x in ext],
        material_slots=[f"EM_{c}" for c in used],
        note="UE_X=E, UE_Y=N, UE_Z=U（导入后）。TX 在质心正上方、模型顶上。",
    )
    meta_path = os.path.join(args.out_dir, "ue_anchor.json")
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[导出] {meta_path}")
    print(f"\n场景水平跨度 {ext[0]:.1f} x {ext[1]:.1f} m，高 {ext[2]:.1f} m")
    print(f"TX 建议(UE cm): {tx_ue_cm}")
    if anchor is None:
        print("[注意] 无 geo_transform：无 Cesium 锚点，模型可放在 UE 世界原点做纯射线实验。")


if __name__ == "__main__":
    main()
