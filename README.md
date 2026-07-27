# lowalt3d — 无人机影像 → 材质感知 3DGS → 电磁三维地图 → 射线路损仿真

从无人机航拍照片出发，重建带**电磁材质属性**（ITU-R P.2040）的三维场景模型，
用于低空无线信道的射线追踪仿真：内置 Python 路损仿真器，并可一键导出
**Wireless InSite**（原生 `.city`，材质不丢）与 **Cesium for Unreal**（`.glb`）。

```
影像 → COLMAP 位姿 → PGSR(3DGS)训练 + TSDF 网格 → GPS 地理配准(米制+ENU)
     → 自动材质标注 → 可微材质向量嵌入高斯(+多视角一致性 L_cons) → 材质迁移到网格面片
     → P.2040 电磁参数 → 射线路损 / 3D 可视化 → 导出 InSite (.city) / UE (.glb)
```

## 特性

- **可微材质学习**：把 K 维材质向量作为每高斯可学习参数，经 3DGS 渲染管线
  （`override_color` 通道，无需改 CUDA）投影到 2D 与分割掩码做 CE 监督，
  并支持跨视角重投影一致性损失 `L_cons`；提供 2D 直接投票基线作消融对比。
- **地理配准**：无人机 GPS EXIF ↔ COLMAP 相机位姿做 Umeyama 相似变换，
  一次恢复米制尺度 + ENU 朝向（+Z 向上），残差可到亚米级。
- **仿真器直通**：
  - Wireless InSite：直写原生 `.city`（逐面片材质索引，绕过 COLLADA 导入丢材质的问题）；
  - Cesium for Unreal：`.glb` 按电磁类分材质槽 + 经纬高锚点 + C++ LineTrace 追踪器（`ue/`）；
  - Python：内置 P.2040 射线路损图（LoS/穿透损耗），三方结果可互相对比。

## 环境

| 项 | 要求 |
|---|---|
| OS | Windows 10/11（原生，无需 WSL） |
| GPU | NVIDIA（已验证 RTX 5070 Ti / Blackwell sm_120） |
| Python | conda 环境，Python 3.10 + PyTorch 2.8 (cu128) |
| 编译 | VS 2022 (C++ 工具链) + CUDA Toolkit 12.8（编译 PGSR CUDA 扩展） |
| 依赖 | `numpy scipy opencv-python open3d pillow pycollada matplotlib` |

第三方（不随仓库分发，放入下列目录）：

- `repos/pgsr` — [PGSR](https://github.com/zju3dv/PGSR) 克隆后按其说明编译子模块
  （Blackwell 显卡编译时 `set TORCH_CUDA_ARCH_LIST=12.0`，pip 安装子模块加 `--no-build-isolation`）；
- `tools/` — [COLMAP](https://github.com/colmap/colmap/releases) Windows-CUDA 预编译版解压于此。

## 快速开始

把无人机原图（保留 GPS EXIF）放到 `data\<场景名>\input\`，然后按
**[docs/通用流程文档_换场景复用.md](docs/通用流程文档_换场景复用.md)** 从上到下执行——
全流程已参数化，换场景只改一个 `SCENE` 变量。核心步骤速览：

```cmd
set SCENE=myscene

:: 1) COLMAP 位姿 + 去畸变（在 repos\pgsr 下）
python scripts\preprocess\convert.py --data_path E:\lowalt3d\data

:: 2) PGSR 训练 + 网格提取（在 repos\pgsr 下）
python train.py -s ..\..\data\%SCENE% -m ..\..\outputs\%SCENE%_pgsr -r 4 --iterations 15000 --max_abs_split_points 0 --opacity_cull_threshold 0.05
python render.py -m ..\..\outputs\%SCENE%_pgsr --max_depth 10.0 --voxel_size 0.01 --use_depth_filter

:: 3) 地理配准（GPS EXIF → 米制 + ENU）
python scripts\georeference.py --images data\%SCENE%\input --cameras outputs\%SCENE%_pgsr\cameras.json --out outputs\%SCENE%_pgsr\geo_transform.json

:: 4) 自动材质标注（零下载，颜色/植被指数分类）
python scripts\autolabel_color.py --images data\%SCENE%\images --out data\%SCENE%\material\masks_auto --color

:: 5) 可微材质嵌入高斯（+ 可选 L_cons 续训），再迁移到网格面片
python scripts\train_gaussian_material.py -s data\%SCENE% -m outputs\%SCENE%_pgsr -r 4 --load_iteration 15000 --masks_dir data\%SCENE%\material\masks_auto --classes configs\classes.txt --mat_out outputs\%SCENE%_pgsr\gaussian_material --epochs 30
python scripts\gaussian_to_mesh_material.py --mesh outputs\%SCENE%_pgsr\mesh\tsdf_fusion_post.ply --gaussian_ply outputs\%SCENE%_pgsr\point_cloud\iteration_15000\point_cloud.ply --material_pt outputs\%SCENE%_pgsr\gaussian_material\gaussian_material.pt --classes configs\classes.txt --out_dir outputs\%SCENE%_pgsr\material_gauss

:: 6) P.2040 射线路损 + 3D 可视化
python scripts\simple_pathloss.py --mesh outputs\%SCENE%_pgsr\mesh\tsdf_fusion_post.ply --materials outputs\%SCENE%_pgsr\material_gauss\face_materials.csv --cameras outputs\%SCENE%_pgsr\cameras.json --out_dir outputs\%SCENE%_pgsr\pathloss --grid 60
python scripts\visualize_sim.py --mesh outputs\%SCENE%_pgsr\material_gauss\material_mesh.ply --grid_npz outputs\%SCENE%_pgsr\pathloss\pathloss_grid.npz --out_dir outputs\%SCENE%_pgsr\viz3d --rays --show

:: 7) 导出 Wireless InSite（加载 scene.city，勿用 .dae）/ Cesium UE
python scripts\export_dae.py --mesh outputs\%SCENE%_pgsr\mesh\tsdf_fusion_post.ply --materials outputs\%SCENE%_pgsr\material_gauss\face_materials.csv --out_dir outputs\%SCENE%_pgsr\insite --geo_transform outputs\%SCENE%_pgsr\geo_transform.json --ground_z0 --target_faces 500000
python scripts\export_gltf_ue.py --material_ply outputs\%SCENE%_pgsr\material_gauss\material_mesh.ply --geo_transform outputs\%SCENE%_pgsr\geo_transform.json --out_dir outputs\%SCENE%_pgsr\ue --target_faces 300000
```

## 目录结构

| 路径 | 内容 |
|---|---|
| `scripts/` | 全部管线脚本（见下表） |
| `configs/` | `classes.txt` 材质类别、`radio_params.yaml` P.2040 电磁参数表 |
| `docs/` | 流程/执行手册（**通用流程文档**为主入口，含常见坑速查表） |
| `ue/` | UE C++ 射线追踪器 `RFRayTracer.{h,cpp}` |
| `materials.csv` | Wireless InSite 材质参数参考表 |
| `data/` `outputs/` | 场景输入/输出（不入库，运行时生成） |
| `repos/` `tools/` | PGSR / COLMAP（不入库，自行获取） |

## 脚本一览

| 脚本 | 作用 |
|---|---|
| `autolabel_color.py` | 零下载地物自动材质预标注（ExG/色彩） |
| `autolabel.py` / `train_seg.py` / `infer_seg.py` / `labelme2mask.py` | 分割备选路线（SegFormer / 人工标注训练） |
| `georeference.py` | GPS EXIF 地理配准 → 米制尺度 + ENU 朝向 |
| `train_gaussian_material.py` | 可微材质向量嵌入高斯 + 多视角一致性 L_cons |
| `gaussian_to_mesh_material.py` | 高斯材质 → 网格面片（kNN 加权投票） |
| `assign_material.py` | 2D 分割直接投票到面片（基线） |
| `simple_pathloss.py` | 材质 → P.2040 → 射线路损图 |
| `export_dae.py` | 导出 Wireless InSite `.city`/`.dae` + 材质表 |
| `export_gltf_ue.py` | 导出 Cesium/UE `.glb` + 材质槽 + 地理锚点 |
| `compare_engines.py` | Python ↔ UE 覆盖结果对比（RMSE/相关系数） |
| `visualize_sim.py` | 路损/视距/射线 3D 可视化 |
| `material_common.py` | 共用：类别表 / 电磁参数 / 调色板 |

## 已知要点（详见通用流程文档"常见坑速查"）

- **InSite 请加载 `scene.city`**：其 COLLADA 导入器会把多材质合并成单一材质，`.city` 为逐面片材质索引的原生格式；
- 模型躺平/尺寸不对 → 导出时加 `--geo_transform`；地面悬在 z≈-80 → 加 `--ground_z0`；
- InSite 日志 `Failed to load OptiX` = 该版对新架构显卡无 GPU 加速，降 Reflections 至 2、Ray spacing 至 1.0° 提速；
- 显存不足 → 训练分辨率 `-r 4` 改 `-r 8`。

## 许可与致谢

依赖并致谢开源项目 [PGSR](https://github.com/zju3dv/PGSR)、[COLMAP](https://colmap.github.io/)、Open3D、pycollada。
电磁参数取自 ITU-R P.2040 建议书。
