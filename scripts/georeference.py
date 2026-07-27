"""
georeference.py  ·  地理配准：用 GPS EXIF 把 COLMAP 坐标对齐到本地 ENU 米坐标

目的：COLMAP 重建是任意单位/朝向。用无人机图的 GPS EXIF（每张的经纬高）→ 本地 ENU（米），
      与 COLMAP 相机位置做 Umeyama 相似变换，求出 尺度s + 旋转R + 平移t。
      一次同时解决：① 米制尺度（InSite/路损要米）② 朝向（ENU 的 +Z 向上，正合 InSite）。

输出 geo_transform.json：{scale, R(3x3), t, ref_lat/lon/alt, residual_m, num_matched}
下游用法：
  export_dae.py --geo_transform geo_transform.json   → 输出米制+摆正的 .dae 给 InSite
  simple_pathloss.py --scale <scale>                 → 路损绝对 dB 变可信

运行：
  python scripts\georeference.py --images data\%SCENE%\input --cameras outputs\%SCENE%_pgsr\cameras.json --out outputs\%SCENE%_pgsr\geo_transform.json
"""
import os, sys, json, glob, argparse
import numpy as np


def get_gps(path):
    """读单张图 GPS EXIF → (lat, lon, alt) 十进制度/米；无则 None。"""
    try:
        from PIL import Image, ExifTags
        exif = Image.open(path).getexif()
        gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
        if not gps or 2 not in gps or 4 not in gps:
            return None
        def dms(v): return float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
        lat = dms(gps[2]); lat = -lat if str(gps.get(1, "N")).upper().startswith("S") else lat
        lon = dms(gps[4]); lon = -lon if str(gps.get(3, "E")).upper().startswith("W") else lon
        alt = float(gps.get(6, 0.0))
        aref = gps.get(5, 0)
        if isinstance(aref, (bytes, bytearray)):
            aref = aref[0] if len(aref) else 0
        try:
            if int(aref) == 1:
                alt = -alt
        except Exception:
            pass
        return lat, lon, alt
    except Exception:
        return None


def geodetic_to_enu(lat, lon, alt, ref):
    """经纬高 → 本地 ENU(米)，局部切平面近似(小场景足够)。"""
    R_E = 6378137.0
    lat0, lon0, alt0 = ref
    d2r = np.pi / 180
    E = (lon - lon0) * d2r * R_E * np.cos(lat0 * d2r)
    N = (lat - lat0) * d2r * R_E
    U = alt - alt0
    return np.array([E, N, U])


def umeyama(src, dst):
    """求 dst ≈ s*R*src + t（带尺度）。src,dst: [N,3]。返回 s,R,t。"""
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    cov = (D.T @ S) / n
    U, d, Vt = np.linalg.svd(cov)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt
    var_s = (S ** 2).sum() / n
    s = np.trace(np.diag(d) @ W) / var_s
    t = mu_d - s * R @ mu_s
    return float(s), R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="原始带 GPS EXIF 的图目录(data\\<SCENE>\\input)")
    ap.add_argument("--cameras", required=True, help="cameras.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # COLMAP 相机位置
    cams = json.load(open(args.cameras, encoding="utf-8"))
    cam_pos = {c["img_name"]: np.array(c["position"], float) for c in cams}

    # 逐图 GPS
    _seen, imgs = set(), []
    for p in sum([glob.glob(os.path.join(args.images, e))
                  for e in ("*.JPG", "*.jpg", "*.png", "*.tif", "*.tiff")], []):
        k = os.path.normcase(os.path.abspath(p))
        if k not in _seen:
            _seen.add(k); imgs.append(p)
    imgs = sorted(imgs)
    gps = {}
    for p in imgs:
        g = get_gps(p)
        if g:
            gps[os.path.splitext(os.path.basename(p))[0]] = g
    print(f"读到 GPS 的图: {len(gps)}/{len(imgs)}")
    if len(gps) < 3:
        print("[失败] 带 GPS 的图 < 3，无法配准。请确认原图含 GPS EXIF（或用 GCP 手动配准）。")
        sys.exit(1)

    # 匹配 COLMAP 位置 <-> GPS
    names = [n for n in gps if n in cam_pos]
    if len(names) < 3:
        # cameras.json 的 img_name 可能带扩展名，做个宽松匹配
        stem2cam = {os.path.splitext(k)[0]: v for k, v in cam_pos.items()}
        names = [n for n in gps if n in stem2cam]
        cam_pos = stem2cam
    assert len(names) >= 3, f"匹配到的 图 <-> 相机 < 3 (gps {len(gps)}, cam {len(cam_pos)})"
    print(f"匹配到 {len(names)} 对")

    # 参考点 = GPS 均值；转 ENU
    latlonalt = np.array([gps[n] for n in names])
    ref = latlonalt.mean(0)
    src = np.array([cam_pos[n] for n in names])                 # COLMAP
    dst = np.array([geodetic_to_enu(*gps[n], ref) for n in names])  # ENU 米

    s, R, t = umeyama(src, dst)
    resid = np.linalg.norm((s * (src @ R.T) + t) - dst, axis=1)
    print(f"\n尺度 scale = {s:.4f} 米/单位")
    print(f"配准残差: 均值 {resid.mean():.2f} m，中位 {np.median(resid):.2f} m，最大 {resid.max():.2f} m")

    out = dict(scale=s, R=R.tolist(), t=t.tolist(),
               ref_lat=float(ref[0]), ref_lon=float(ref[1]), ref_alt=float(ref[2]),
               residual_m=float(resid.mean()), num_matched=len(names))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[导出] {args.out}")
    print("下游：export_dae.py --geo_transform 本文件 → 米制+摆正的 .dae；simple_pathloss.py --scale "
          f"{s:.4f} → 真实 dB")
    if resid.mean() > 5:
        print("[注意] 残差偏大——普通 GPS 精度有限；要更准用 RTK/PPK 或 GCP。")


if __name__ == "__main__":
    main()
