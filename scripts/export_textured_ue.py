r"""
export_textured_ue.py  ·  网格 + 原始照片 → 带【照片纹理 + PBR】的分块 .glb（UE / Cesium 展示用）

export_gltf_ue.py 只给每个材质类一个纯色，导进 UE 是色块。本脚本补上真实外观：
  1. 抽面（材质按最近原面片重投，同 export_gltf_ue）；
  2. 按 ENU 米制网格分块（大场景每块单独展 UV、单独一张贴图）；
  3. xatlas 自动展 UV；
  4. 纹理烘焙：每个纹素 → 3D 点 → 投到所有可见相机（Open3D 射线求深度图做遮挡判断），
     按 |cos 入射角| × 分辨率 × 离图像边缘距离 加权（取 p 次方，偏向最好的视角又能平滑接缝）；
  5. 由 face_materials.csv 生成逐纹素 PBR：metallicRoughness 贴图（glTF：G=粗糙度, B=金属度，
     值查 configs/visual_pbr.yaml）+ 单通道 material_id.png（UE 自定义材质可按类替换，如伪玻璃）；
  6. 坐标链与 export_gltf_ue 完全一致：COLMAP → ENU 米 → glTF(x=E, y=U, z=-N)，
     所有块共用同一个质心原点 → 在 UE 里都放世界原点即可严丝合缝；
  7. 可选：每块再抽一份碰撞网格 *_collision.glb（给无人机/机器狗用）。

输入相机：cameras.json（PGSR 约定：position=相机中心，rotation=相机→世界，主点在图像中心，
         fx/fy 对应 width/height；--images 里的图若被降采样，按实际尺寸自动缩放内参）。
         --images 必须是【去畸变后】的图（COLMAP/PGSR 的 data\<scene>\images 或 images_2）。

依赖：pip install xatlas trimesh   （open3d / scipy / opencv-python / pillow 已装）
运行：
  python scripts\export_textured_ue.py ^
    --mesh outputs\%SCENE%_pgsr_hq\mesh\tsdf_fusion_post.ply ^
    --materials outputs\%SCENE%_pgsr_hq\material_win\face_materials.csv ^
    --cameras outputs\%SCENE%_pgsr_hq\cameras.json --images data\%SCENE%\images_2 ^
    --geo_transform outputs\%SCENE%_pgsr_hq\geo_transform.json ^
    --out_dir outputs\%SCENE%_pgsr_hq\ue_textured --target_faces 2000000 --tile_m 100 --tex 4096
自检：python scripts\export_textured_ue.py --selfcheck
输出：<out_dir>\tile_<i>_<j>\{tile.glb, basecolor.png, orm.png, material_id.png, [collision.glb]}
      <out_dir>\tiles.json（块清单、覆盖率、Cesium 锚点）
UE 导入：Interchange 导入 tile.glb（Uniform Scale 100 或 Convert Scene Unit），全部放原点；
         开 Nanite；material_id.png 用 Nearest 采样。
"""
import os, sys, json, argparse, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from material_common import _DEFAULT_CLASS_ORDER

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R_E = 6378137.0
D2R = np.pi / 180.0
IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG")


def enu_to_geodetic(E, N, U, ref):
    lat0, lon0, alt0 = ref
    lat = lat0 + (N / R_E) / D2R
    lon = lon0 + (E / (R_E * np.cos(lat0 * D2R))) / D2R
    return float(lat), float(lon), float(alt0 + U)


def load_pbr(path, class_order):
    table = {"default": {"roughness": 0.85, "metallic": 0.0}}
    if path and os.path.isfile(path):
        import yaml
        table.update(yaml.safe_load(open(path, encoding="utf-8")) or {})
    rough = np.array([float(table.get(c, table["default"])["roughness"]) for c in class_order], np.float32)
    metal = np.array([float(table.get(c, table["default"])["metallic"]) for c in class_order], np.float32)
    return rough, metal


def imread_any(path):
    import cv2
    data = np.fromfile(path, dtype=np.uint8)          # 兼容中文路径
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"读图失败: {path}")
    return img[:, :, ::-1].copy()                     # BGR → RGB


def index_images(images_dir):
    idx = {}
    for fn in os.listdir(images_dir):
        stem, ext = os.path.splitext(fn)
        if ext in IMG_EXT:
            idx[stem] = os.path.join(images_dir, fn)
    return idx


# ----------------------------------------------------------------------------- UV 光栅化
def rasterize_uv(uv_px, F, W, H, budget=4_000_000):
    """在 UV 像素空间光栅化三角形。uv_px: [Nv,2]（x 向右, y 向下=行）。
    返回 (tri_id, x, y, bary[n,3])，每个被覆盖的纹素中心一条（共边处可能重复，后写覆盖）。"""
    T = uv_px[F].astype(np.float64)                                   # [M,3,2]
    mn = np.floor(T.min(1)).astype(np.int64)
    mx = np.ceil(T.max(1)).astype(np.int64)
    mn = np.clip(mn, 0, [W - 1, H - 1]); mx = np.clip(mx, 1, [W, H])
    size = np.maximum(mx - mn, 1)
    area = size[:, 0] * size[:, 1]
    order = np.argsort(-area, kind="stable")
    outs = []
    i, M = 0, len(F)
    while i < M:
        a0 = int(area[order[i]])
        n = max(1, budget // a0)
        idx = order[i:i + n]; i += n
        bw, bh = int(size[idx, 0].max()), int(size[idx, 1].max())
        oy, ox = np.mgrid[0:bh, 0:bw]
        px = mn[idx, 0, None, None] + ox[None] + 0.5
        py = mn[idx, 1, None, None] + oy[None] + 0.5
        a, b, c = T[idx, 0], T[idx, 1], T[idx, 2]
        v0, v1 = b - a, c - a
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        ok = np.abs(den) > 1e-12
        den = np.where(ok, den, 1.0)[:, None, None]
        dx = px - a[:, 0, None, None]; dy = py - a[:, 1, None, None]
        l1 = (dx * v1[:, 1, None, None] - v1[:, 0, None, None] * dy) / den
        l2 = (v0[:, 0, None, None] * dy - dx * v0[:, 1, None, None]) / den
        l0 = 1.0 - l1 - l2
        e = -1e-4
        inside = ((l0 >= e) & (l1 >= e) & (l2 >= e) & ok[:, None, None]
                  & (px < W) & (py < H))
        ti, yy, xx = np.nonzero(inside)
        bary = np.stack([l0[ti, yy, xx], l1[ti, yy, xx], l2[ti, yy, xx]], 1)
        outs.append((idx[ti], (mn[idx[ti], 0] + xx).astype(np.int32),
                     (mn[idx[ti], 1] + yy).astype(np.int32), np.clip(bary, 0, 1).astype(np.float32)))
    if not outs:
        return (np.zeros(0, np.int64), np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros((0, 3), np.float32))
    return tuple(np.concatenate([o[k] for o in outs]) for k in range(4))


# ----------------------------------------------------------------------------- 烘焙
def bilinear(img, u, v):
    h, w = img.shape[:2]
    x0 = np.floor(u - 0.5).astype(np.int64); y0 = np.floor(v - 0.5).astype(np.int64)
    fx = (u - 0.5 - x0)[:, None]; fy = (v - 0.5 - y0)[:, None]
    x0 = np.clip(x0, 0, w - 1); y0 = np.clip(y0, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1); y1 = np.clip(y0 + 1, 0, h - 1)
    im = img.astype(np.float32)
    return ((im[y0, x0] * (1 - fx) + im[y0, x1] * fx) * (1 - fy)
            + (im[y1, x0] * (1 - fx) + im[y1, x1] * fx) * fy)


class DepthCache:
    """每个相机一张低分辨率深度图（z 深度），用 Open3D 射线求交整网格（含块外遮挡物）。"""

    def __init__(self, V, F, scale):
        import open3d as o3d
        self.o3d = o3d
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
        self.scale = scale
        self.cache = {}

    def get(self, cam, w, h, fx, fy):
        key = cam["img_name"]
        if key in self.cache:
            return self.cache[key]
        s = self.scale
        wd, hd = max(8, int(round(w * s))), max(8, int(round(h * s)))
        xs = (np.arange(wd) + 0.5) / wd * w; ys = (np.arange(hd) + 0.5) / hd * h
        X, Y = np.meshgrid(xs, ys)
        d_cam = np.stack([(X - w / 2) / fx, (Y - h / 2) / fy, np.ones_like(X)], -1).reshape(-1, 3)
        R = np.asarray(cam["rotation"], np.float64); C = np.asarray(cam["position"], np.float64)
        d_w = d_cam @ R.T                                  # |z_cam|=1 → t_hit 就是 z 深度
        rays = np.concatenate([np.broadcast_to(C, d_w.shape), d_w], 1).astype(np.float32)
        t = self.scene.cast_rays(self.o3d.core.Tensor(rays))["t_hit"].numpy().reshape(hd, wd)
        self.cache[key] = (t, wd / w, hd / h)
        if len(self.cache) > 64:                          # 简单限内存
            self.cache.pop(next(iter(self.cache)))
        return self.cache[key]


def bake_tile(P, Nrm, cams, img_index, depth, args, bbox):
    """P/Nrm: 纹素 3D 点与面法向（COLMAP 坐标）。返回 (rgb[n,3] float, 覆盖 bool[n], 用到的视角数)。"""
    n = len(P)
    acc = np.zeros((n, 3), np.float64); wsum = np.zeros(n, np.float64)
    corners = np.array([[x, y, z] for x in bbox[:, 0] for y in bbox[:, 1] for z in bbox[:, 2]])
    used = 0
    for cam in cams:
        path = img_index.get(cam["img_name"])
        if path is None:
            continue
        R = np.asarray(cam["rotation"], np.float64); C = np.asarray(cam["position"], np.float64)
        W0, H0 = cam["width"], cam["height"]
        # 快速剔除：块包围盒投影与图像不相交
        cc = (corners - C) @ R
        if (cc[:, 2] > 1e-6).all():
            uu = cam["fx"] * cc[:, 0] / cc[:, 2] + W0 / 2; vv = cam["fy"] * cc[:, 1] / cc[:, 2] + H0 / 2
            if uu.max() < 0 or uu.min() > W0 or vv.max() < 0 or vv.min() > H0:
                continue
        elif (cc[:, 2] <= 1e-6).all():
            continue
        img = imread_any(path)
        h, w = img.shape[:2]
        fx = cam["fx"] * w / W0; fy = cam["fy"] * h / H0
        Xc = (P - C) @ R
        z = Xc[:, 2]
        front = z > 1e-6
        zs = np.where(front, z, 1.0)
        u = fx * Xc[:, 0] / zs + w / 2; v = fy * Xc[:, 1] / zs + h / 2
        m = front & (u >= 1) & (u < w - 1) & (v >= 1) & (v < h - 1)
        if not m.any():
            continue
        ii = np.nonzero(m)[0]
        vd = C - P[ii]; dist = np.linalg.norm(vd, axis=1) + 1e-12
        cos = np.abs((Nrm[ii] * vd).sum(1)) / dist
        keep = cos > args.min_cos
        ii, cos = ii[keep], cos[keep]
        if len(ii) == 0:
            continue
        dmap, sx, sy = depth.get(cam, w, h, fx, fy)
        du = np.clip((u[ii] * sx).astype(np.int64), 0, dmap.shape[1] - 1)
        dv = np.clip((v[ii] * sy).astype(np.int64), 0, dmap.shape[0] - 1)
        vis = z[ii] <= dmap[dv, du] * (1.0 + args.depth_tol)
        ii, cos = ii[vis], cos[vis]
        if len(ii) == 0:
            continue
        uu, vv = u[ii], v[ii]
        edge = np.minimum.reduce([uu, w - uu, vv, h - vv]) / (args.edge_frac * min(w, h))
        wgt = (cos * (fx / z[ii]) * np.clip(edge, 0.05, 1.0)) ** args.blend_power
        acc[ii] += bilinear(img, uu, vv) * wgt[:, None]
        wsum[ii] += wgt
        used += 1
    cov = wsum > 0
    rgb = np.zeros((n, 3), np.float32)
    rgb[cov] = (acc[cov] / wsum[cov, None]).astype(np.float32)
    return rgb, cov, used


def fill_nearest(img, mask):
    """未覆盖像素（UV 缝隙/未拍到）填最近的已覆盖像素，避免 mip 渗黑边。"""
    from scipy import ndimage
    if mask.all() or not mask.any():
        return img
    _, (iy, ix) = ndimage.distance_transform_edt(~mask, return_indices=True)
    return img[iy, ix]


# ----------------------------------------------------------------------------- 主流程
def export_tiles(V, F, face_cls_idx, cams, img_index, args, geo, class_order, out_dir, log=print):
    import xatlas, trimesh
    from PIL import Image
    rough_t, metal_t = load_pbr(args.pbr, class_order)

    # COLMAP → ENU 米
    if geo is not None:
        s = float(geo["scale"]); Rg = np.array(geo["R"]); tg = np.array(geo["t"])
        V_enu = s * (V @ Rg.T) + tg
        ref = (geo["ref_lat"], geo["ref_lon"], geo["ref_alt"])
    else:
        s = float(args.scale); V_enu = V * s; ref = None
    Cg = V_enu.mean(0)
    Vc = V_enu - Cg
    V_gltf = np.column_stack([Vc[:, 0], Vc[:, 2], -Vc[:, 1]]).astype(np.float32)

    # 分块（按面片质心 E/N）
    fc = Vc[F].mean(1)
    if args.tile_m > 0:
        e0, n0 = fc[:, 0].min(), fc[:, 1].min()
        ti = np.floor((fc[:, 0] - e0) / args.tile_m).astype(int)
        tj = np.floor((fc[:, 1] - n0) / args.tile_m).astype(int)
    else:
        ti = tj = np.zeros(len(F), int)
    keys = sorted(set(zip(ti.tolist(), tj.tolist())))
    log(f"分块: {len(keys)} 块（tile_m={args.tile_m}）")

    depth = DepthCache(V, F, args.depth_scale)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12

    manifest = []
    W = H = int(args.tex)
    for (i, j) in keys:
        t0 = time.time()
        fsel = np.nonzero((ti == i) & (tj == j))[0]
        if len(fsel) == 0:
            continue
        Ft = F[fsel]
        vids, inv = np.unique(Ft, return_inverse=True)
        Fl = inv.reshape(-1, 3).astype(np.uint32)
        Vl = V[vids]
        # UV
        atlas = xatlas.Atlas()
        atlas.add_mesh(Vl.astype(np.float32), Fl)
        po = xatlas.PackOptions()
        po.resolution = W; po.padding = args.padding; po.bilinear = True
        atlas.generate(pack_options=po)
        vmap, Fa, uvs = atlas[0]                       # 新顶点 → 旧顶点索引, 新面, uv∈[0,1]
        Fa = Fa.astype(np.int64)
        # xatlas 保持面顺序 → 面 k 对应 fsel[k]
        uv_px = uvs.astype(np.float64) * [W, H]
        tri, xx, yy, bary = rasterize_uv(uv_px, Fa, W, H)
        Va = Vl[vmap]
        P = (Va[Fa[tri]] * bary[:, :, None]).sum(1)
        Nrm = fn[fsel[tri]]
        bbox = np.stack([Va.min(0), Va.max(0)])
        rgb, cov, used = bake_tile(P, Nrm, cams, img_index, depth, args, bbox)

        mask = np.zeros((H, W), bool); mask[yy[cov], xx[cov]] = True
        base = np.zeros((H, W, 3), np.float32); base[yy[cov], xx[cov]] = rgb[cov]
        base = fill_nearest(base, mask)
        if not mask.any():
            base[:] = 128
        cls = face_cls_idx[fsel[tri]]
        cmask = np.zeros((H, W), bool); cmask[yy, xx] = True
        cid = np.zeros((H, W), np.uint8); cid[yy, xx] = cls
        cid = fill_nearest(cid, cmask)
        orm = np.zeros((H, W, 3), np.uint8)
        orm[..., 0] = 255                                   # R=AO(无)
        orm[..., 1] = (rough_t[cid] * 255).round().astype(np.uint8)
        orm[..., 2] = (metal_t[cid] * 255).round().astype(np.uint8)

        tdir = os.path.join(out_dir, f"tile_{i:02d}_{j:02d}")
        os.makedirs(tdir, exist_ok=True)
        base_img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
        orm_img = Image.fromarray(orm)
        base_img.save(os.path.join(tdir, "basecolor.png"))
        orm_img.save(os.path.join(tdir, "orm.png"))
        Image.fromarray(cid).save(os.path.join(tdir, "material_id.png"))

        uv_gl = np.column_stack([uvs[:, 0], 1.0 - uvs[:, 1]])   # trimesh 用 OpenGL 约定，导出 glTF 时再翻回
        mat = trimesh.visual.material.PBRMaterial(
            name=f"PH_tile_{i:02d}_{j:02d}", baseColorTexture=base_img,
            metallicRoughnessTexture=orm_img, metallicFactor=1.0, roughnessFactor=1.0)
        tm = trimesh.Trimesh(vertices=V_gltf[vids][vmap], faces=Fa, process=False,
                             visual=trimesh.visual.TextureVisuals(uv=uv_gl, material=mat))
        glb = os.path.join(tdir, "tile.glb")
        tm.export(glb)

        entry = dict(tile=[i, j], dir=os.path.relpath(tdir, out_dir), faces=int(len(fsel)),
                     texels=int(len(tri)), coverage=float(cov.mean()) if len(cov) else 0.0,
                     views_used=int(used),
                     bbox_enu_m=[(Vc[vids].min(0)).tolist(), (Vc[vids].max(0)).tolist()],
                     class_faces={class_order[k]: int((face_cls_idx[fsel] == k).sum())
                                  for k in range(len(class_order)) if (face_cls_idx[fsel] == k).any()})
        if args.collision_ratio > 0:
            col = collision_mesh(V_gltf[vids], Fl.astype(np.int64), args.collision_ratio)
            col.export(os.path.join(tdir, "collision.glb"))
            entry["collision_faces"] = int(len(col.faces))
        manifest.append(entry)
        log(f"  块({i},{j}): {len(fsel)} 面, 纹素 {len(tri)}, 覆盖 {entry['coverage']*100:.1f}%, "
            f"视角 {used}, {time.time()-t0:.1f}s")

    meta = dict(
        units="glb=米, glTF Y-up(x=E,y=U,z=-N)；UE 导入 Uniform Scale 100，全部块放世界原点",
        origin_enu_m=Cg.tolist(),
        cesium_anchor=(dict(zip(["latitude", "longitude", "height"], enu_to_geodetic(*Cg, ref)))
                       if ref else None),
        tex=W, class_order=class_order,
        pbr_table={c: dict(roughness=float(rough_t[k]), metallic=float(metal_t[k]))
                   for k, c in enumerate(class_order)},
        tiles=manifest)
    json.dump(meta, open(os.path.join(out_dir, "tiles.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return meta


def collision_mesh(V, F, ratio):
    import open3d as o3d, trimesh
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V.astype(np.float64)),
                                  o3d.utility.Vector3iVector(F.astype(np.int32)))
    target = max(4, int(len(F) * ratio))
    m = m.simplify_quadric_decimation(target)
    m.remove_degenerate_triangles(); m.remove_duplicated_triangles()
    m.remove_duplicated_vertices(); m.remove_unreferenced_vertices()
    return trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.triangles), process=False)


def load_mesh_and_classes(args, class_order, log=print):
    import open3d as o3d
    from scipy.spatial import cKDTree
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    V = np.asarray(mesh.vertices, np.float64); F = np.asarray(mesh.triangles, np.int64)
    log(f"原始网格: {len(F)} 面 / {len(V)} 顶点")
    cls2idx = {c: k for k, c in enumerate(class_order)}
    if args.materials:
        rec = np.genfromtxt(args.materials, delimiter=",", names=True, dtype=None, encoding="utf-8")
        names = np.atleast_1d(np.asarray(rec["material_name"]).astype(str))
        assert len(names) == len(F), f"face 数不一致: csv {len(names)} vs mesh {len(F)}"
        dflt = cls2idx.get("concrete", 0)
        fcls = np.array([cls2idx.get(nm, dflt) for nm in names], np.uint8)
    else:
        fcls = np.full(len(F), cls2idx.get("concrete", 0), np.uint8)
        log("未给 --materials：全部按 concrete 赋 PBR")
    if args.target_faces and len(F) > args.target_faces:
        sm = mesh.simplify_quadric_decimation(args.target_faces)
        V2 = np.asarray(sm.vertices, np.float64); F2 = np.asarray(sm.triangles, np.int64)
        _, idx = cKDTree(V[F].mean(1)).query(V2[F2].mean(1))
        fcls = fcls[idx]; V, F = V2, F2
        log(f"抽面后: {len(F)} 面 / {len(V)} 顶点")
    return V, F, fcls


def main(argv=None):
    ap = argparse.ArgumentParser(description="网格 + 照片 → 带纹理/PBR 的分块 glb（UE 展示）")
    ap.add_argument("--mesh"); ap.add_argument("--materials", default=None, help="face_materials.csv（建议 material_win）")
    ap.add_argument("--cameras", help="cameras.json"); ap.add_argument("--images", help="去畸变影像目录")
    ap.add_argument("--geo_transform", default=None)
    ap.add_argument("--scale", type=float, default=1.0, help="无 geo_transform 时的 米/COLMAP单位")
    ap.add_argument("--out_dir")
    ap.add_argument("--target_faces", type=int, default=2_000_000, help="先整体抽面;0=不抽")
    ap.add_argument("--tile_m", type=float, default=100.0, help="分块边长(米);0=不分块")
    ap.add_argument("--tex", type=int, default=4096, help="每块贴图边长(像素)")
    ap.add_argument("--padding", type=int, default=4, help="UV 岛间距(像素)")
    ap.add_argument("--depth_scale", type=float, default=0.25, help="遮挡深度图相对影像的分辨率")
    ap.add_argument("--depth_tol", type=float, default=0.02, help="遮挡判断的相对深度容差")
    ap.add_argument("--min_cos", type=float, default=0.15, help="入射角余弦下限（剔除掠射视角）")
    ap.add_argument("--edge_frac", type=float, default=0.1, help="离图像边缘此比例内降权")
    ap.add_argument("--blend_power", type=float, default=4.0, help="视角权重的次方：大=更锐利/接缝更硬")
    ap.add_argument("--collision_ratio", type=float, default=0.1, help="碰撞网格面数比例;0=不出")
    ap.add_argument("--pbr", default=os.path.join(ROOT, "configs", "visual_pbr.yaml"))
    ap.add_argument("--classes", default=os.path.join(ROOT, "configs", "classes.txt"))
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args(argv)
    if args.selfcheck:
        return selfcheck()
    for k in ("mesh", "cameras", "images", "out_dir"):
        assert getattr(args, k), f"缺少 --{k}"

    class_order = list(_DEFAULT_CLASS_ORDER)
    if args.classes and os.path.isfile(args.classes):
        file_order = [l.strip() for l in open(args.classes, encoding="utf-8") if l.strip()]
        assert file_order == class_order, f"classes.txt {file_order} 与 material_common {class_order} 不一致"
    V, F, fcls = load_mesh_and_classes(args, class_order)
    cams = json.load(open(args.cameras, encoding="utf-8"))
    img_index = index_images(args.images)
    miss = sum(c["img_name"] not in img_index for c in cams)
    print(f"相机 {len(cams)}，影像 {len(img_index)}，缺图 {miss}")
    assert miss < len(cams), "cameras.json 的 img_name 在 --images 里一张都找不到"
    geo = json.load(open(args.geo_transform, encoding="utf-8")) if args.geo_transform else None
    if geo is None:
        print("[注意] 无 geo_transform：分块边长按 --scale 换算，且无 Cesium 锚点")
    os.makedirs(args.out_dir, exist_ok=True)
    meta = export_tiles(V, F, fcls, cams, img_index, args, geo, class_order, args.out_dir)
    cov = [t["coverage"] for t in meta["tiles"]]
    print(f"[完成] {len(meta['tiles'])} 块 → {args.out_dir}；平均纹理覆盖 {np.mean(cov)*100:.1f}%")
    return 0


# ----------------------------------------------------------------------------- 自检
def selfcheck():
    """合成场景：地面 + 一栋楼，颜色是世界坐标的平滑函数 f(P)；用射线渲染出相机影像，
    再走完整导出流程，检查 ① 烘焙颜色 ≈ f(P) ② 楼后被遮挡处不串色 ③ glb 往返 uv/贴图一致。"""
    import tempfile, open3d as o3d, trimesh
    from PIL import Image

    def color_fn(P):
        return np.stack([127 + 100 * np.sin(P[:, 0] * 0.6), 127 + 100 * np.cos(P[:, 1] * 0.5),
                         127 + 100 * np.sin(P[:, 2] * 0.9 + 1.0)], 1)

    ground = o3d.geometry.TriangleMesh.create_box(20, 20, 0.2).translate((-10, -10, -0.2))
    bld = o3d.geometry.TriangleMesh.create_box(4, 4, 6).translate((-2, -2, 0))
    m = ground + bld
    m = m.subdivide_midpoint(3)
    V = np.asarray(m.vertices); F = np.asarray(m.triangles)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))

    def look_at(C, target):
        z = target - C; z /= np.linalg.norm(z)
        x = np.cross(z, [0, 0, 1.0]); x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0, 0])
        y = np.cross(z, x)
        return np.stack([x, y, z], 1)           # 列 = 相机轴 → 相机到世界

    td = tempfile.mkdtemp(prefix="txcheck_")
    imgdir = os.path.join(td, "images"); os.makedirs(imgdir)
    cams = []
    Wc, Hc, f = 320, 240, 260.0
    poses = [(r * np.cos(a), r * np.sin(a), 12.0) for r in (9, 14) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    poses.append((0.1, 0.1, 18.0))
    for k, Cp in enumerate(poses):
        C = np.array(Cp, float); Rcw = look_at(C, np.array([0, 0, 1.0]))
        X, Y = np.meshgrid(np.arange(Wc) + 0.5, np.arange(Hc) + 0.5)
        d = np.stack([(X - Wc / 2) / f, (Y - Hc / 2) / f, np.ones_like(X)], -1).reshape(-1, 3) @ Rcw.T
        rays = np.concatenate([np.broadcast_to(C, d.shape), d], 1).astype(np.float32)
        t = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        hit = np.isfinite(t)
        P = C + d * np.where(hit, t, 0)[:, None]
        img = np.where(hit[:, None], color_fn(P), 0).reshape(Hc, Wc, 3)
        name = f"img{k:03d}"
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(os.path.join(imgdir, name + ".png"))
        cams.append(dict(id=k, img_name=name, width=Wc * 2, height=Hc * 2,   # 模拟"影像被降采样 2 倍"
                         position=C.tolist(), rotation=Rcw.tolist(), fx=f * 2, fy=f * 2))
    mesh_path = os.path.join(td, "mesh.ply"); o3d.io.write_triangle_mesh(mesh_path, m)
    fc = V[F].mean(1)
    names = np.where(fc[:, 2] > 0.05, "glass", "asphalt")
    with open(os.path.join(td, "fm.csv"), "w", encoding="utf-8") as fp:
        fp.write("face_id,material_id,material_name\n")
        for i, nm in enumerate(names):
            fp.write(f"{i},{_DEFAULT_CLASS_ORDER.index(nm)},{nm}\n")
    cam_path = os.path.join(td, "cameras.json"); json.dump(cams, open(cam_path, "w"))
    out = os.path.join(td, "out")
    rc = main(["--mesh", mesh_path, "--materials", os.path.join(td, "fm.csv"), "--cameras", cam_path,
               "--images", imgdir, "--out_dir", out, "--target_faces", "0", "--tile_m", "10",
               "--tex", "512", "--scale", "1.0"])
    assert rc == 0
    meta = json.load(open(os.path.join(out, "tiles.json"), encoding="utf-8"))
    # 20 m 场景 / 10 m 块 → 4 个主块；x 或 y 恰在 +10 的侧面会落进第 3 行/列的边缘块
    assert len(meta["tiles"]) >= 4, f"至少应分 4 块，得 {len(meta['tiles'])}"

    # ① 颜色误差：重新加载每块 glb，按其 uv 采样贴图，与 f(世界点) 比较
    errs, glass_rough = [], []
    Cg = np.array(meta["origin_enu_m"])
    for t in meta["tiles"]:
        sc = trimesh.load(os.path.join(out, t["dir"], "tile.glb"))
        g = list(sc.geometry.values())[0] if hasattr(sc, "geometry") else sc
        base = np.asarray(Image.open(os.path.join(out, t["dir"], "basecolor.png")).convert("RGB"), np.float32)
        orm = np.asarray(Image.open(os.path.join(out, t["dir"], "orm.png")))
        uv = np.asarray(g.visual.uv); Vg = np.asarray(g.vertices); Fg = np.asarray(g.faces)
        # 取面重心
        cuv = uv[Fg].mean(1); cP = Vg[Fg].mean(1)
        Pw = np.column_stack([cP[:, 0], -cP[:, 2], cP[:, 1]]) + Cg       # glTF → ENU(=世界，scale 1)
        px = np.clip((cuv[:, 0] * base.shape[1]).astype(int), 0, base.shape[1] - 1)
        py = np.clip(((1 - cuv[:, 1]) * base.shape[0]).astype(int), 0, base.shape[0] - 1)
        # 只评相机拍得到的面：排除楼底下的地面、地面板 0.2 m 厚的外侧边（掠射/几乎拍不到）
        under = (np.abs(Pw[:, 0]) < 2.5) & (np.abs(Pw[:, 1]) < 2.5) & (Pw[:, 2] < 0.05)
        rim = np.abs(Pw[:, :2]).max(1) > 9.5
        valid = ~under & ~rim & (Pw[:, 2] > -0.001)
        errs.append(np.abs(base[py, px] - color_fn(Pw))[valid].mean(1))
        isg = Pw[:, 2] > 0.5
        glass_rough.append(orm[py[isg], px[isg], 1])
        assert os.path.isfile(os.path.join(out, t["dir"], "collision.glb"))
    e = np.concatenate(errs)
    med, p90 = np.median(e), np.percentile(e, 90)
    print(f"  颜色误差（0-255）：中位 {med:.2f}，90% 分位 {p90:.2f}")
    assert med < 6 and p90 < 25, "烘焙颜色与真值偏差过大（投影/坐标/uv 约定可能有误）"
    gr = np.concatenate(glass_rough)
    assert np.median(gr) < 40, f"glass 面 roughness 应≈0.06*255，得 {np.median(gr)}"
    print(f"  OK 分块 {len(meta['tiles'])} 块、颜色烘焙、遮挡、降采样内参缩放、PBR(orm)、glb 往返、碰撞网格")
    print(f"  自检产物：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
