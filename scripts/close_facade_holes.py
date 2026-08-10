"""
close_facade_holes.py · 补几何空洞(俯视没拍到的立面下部/凹窗/底面) → RF 射线不漏

背景：无人机俯视航拍，建筑下部立面/底面/凹窗从未成像 → TSDF 网格留空洞。
对射线仿真，空洞会让射线"漏"进建筑。用 Open3D fill_holes 把开边界环补成三角面
(相当于把观测到的立面平面向缺口延展/封盖)——这是"外推已观测结构"，不是生成式幻觉，RF 安全。
补出的面无需手工标材质：随后重跑 gaussian_to_mesh_material.py，kNN 会从周围高斯把
concrete/glass 投票上去。

⚠️ 只补"边界环<hole_size"的洞。hole_size 小=只补凹窗/小缝；大=连建筑底面大开口也封。
   建议先默认值看补了多少、是否变 watertight，再按需调 --hole_size。

用法(在 E:\lowalt3d 下)：
  python scripts\close_facade_holes.py ^
    --mesh outputs\scene02_pgsr_hq\mesh\tsdf_fusion_post.ply ^
    --out  outputs\scene02_pgsr_hq\mesh\tsdf_fusion_closed.ply --hole_size 0.1
然后用 closed 网格重出材质：
  python scripts\gaussian_to_mesh_material.py --mesh ...tsdf_fusion_closed.ply ... --use_normal ...
"""
import os, argparse


def close_holes(mesh, hole_size):
    """Open3D t.geometry.fill_holes：补 <hole_size 的开边界环。返回 legacy 网格。"""
    import open3d as o3d
    tm = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    filled = tm.fill_holes(hole_size=float(hole_size))   # 需 open3d>=0.17
    return filled.to_legacy()


def main():
    ap = argparse.ArgumentParser(description="补建筑几何空洞(俯视缺拍)→ RF 射线不漏")
    ap.add_argument("--mesh", required=True, help="输入网格(tsdf_fusion_post.ply)")
    ap.add_argument("--out", required=True, help="输出补洞后网格")
    ap.add_argument("--hole_size", type=float, default=0.1,
                    help="补边界环阈值(COLMAP 单位；scale≈33m/单位 → 0.1≈3.3m)。小=凹窗/缝，大=封底面大口")
    ap.add_argument("--min_component", type=int, default=0,
                    help=">0 先删小于此面数的孤立连通块(去悬浮碎片；主体建筑连通块很大不受影响)")
    args = ap.parse_args()

    import open3d as o3d, numpy as np
    m = o3d.io.read_triangle_mesh(args.mesh)
    m.remove_duplicated_vertices(); m.remove_degenerate_triangles(); m.remove_duplicated_triangles()
    if args.min_component > 0:
        ti, nt, _ = m.cluster_connected_triangles()
        ti = np.asarray(ti); nt = np.asarray(nt)
        small = nt[ti] < args.min_component
        m.remove_triangles_by_mask(small); m.remove_unreferenced_vertices()
        print(f"去浮块: 删 {int(small.sum())} 面(<{args.min_component} 面的孤立连通块)")
    n0 = len(m.triangles)
    print(f"输入: 顶点 {len(m.vertices)}  面 {n0}  watertight={m.is_watertight()}")

    try:
        out = close_holes(m, args.hole_size)
    except AttributeError:
        raise SystemExit("当前 open3d 无 t.geometry.fill_holes(需 >=0.17)；升级或改用 MeshLab 'Close Holes'")
    except Exception as e:
        raise SystemExit(f"fill_holes 失败(网格可能非流形)：{repr(e)[:120]}\n"
                         f"可先跑 --clean_only 清一遍，或用较小 --hole_size 重试")

    n1 = len(out.triangles)
    print(f"输出: 顶点 {len(out.vertices)}  面 {n1}  watertight={out.is_watertight()}  (+{n1-n0} 面补洞)")
    if n1 == n0:
        print("[提示] 没补上任何面：hole_size 可能太小(调大)，或洞不是闭合边界环(得用立面延展，另说)")
    o3d.io.write_triangle_mesh(args.out, out)
    print(f"[导出] {args.out}\n下一步：gaussian_to_mesh_material.py 用 {os.path.basename(args.out)} 重出材质(补面自动投票为 concrete/glass)")


def _selfcheck():
    import open3d as o3d, numpy as np
    box = o3d.geometry.TriangleMesh.create_box()
    assert box.is_watertight()
    box.triangles = o3d.utility.Vector3iVector(np.asarray(box.triangles)[1:])  # 挖一个洞
    assert not box.is_watertight(), "挖洞失败"
    filled = close_holes(box, hole_size=10.0)
    assert filled.is_watertight(), "fill_holes 没把洞补上"
    print("selfcheck ok")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
