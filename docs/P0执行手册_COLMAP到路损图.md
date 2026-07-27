# P0 执行手册：从无人机照片到材质电磁路损图

> 单场景纯无人机端到端闭环。每步含 **目的 / 原理 / 命令 / 参数 / 成功标志 / 记录要点**，便于截图存档。
> 场景：`E:\lowalt3d\data\scene01\input\`（87 张无人机 JPG）
> 环境：conda `E:\condaenvs\emmap`（torch 2.8 cu128）+ COLMAP + PGSR（CUDA 扩展已编译）

---

## 全流程总览

```
input/ 87张照片
   │  Step1 COLMAP (SfM)         恢复相机位姿+内参+稀疏点
   ▼
images/ + sparse/0/
   │  Step2 PGSR 训练            学稠密几何(平面约束高斯场)
   ▼
point_cloud.ply (高斯场)
   │  Step3 网格提取 (TSDF)      高斯→水密三角网格
   ▼
mesh .ply
   │  Step4 材质投票 (待写脚本)   给每个面片打材质标签
   ▼
face_materials.csv
   │  Step5 电磁映射+路损 (待写)  材质→P.2040→射线→路损图
   ▼
路损图 / LoS 图
```

**每一步都在这个窗口里跑**（已激活 emmap 的 conda 命令行，且能用 COLMAP）：

```cmd
conda activate E:\condaenvs\emmap
cd /d E:\lowalt3d\repos\pgsr
```

---

## Step 1 · COLMAP 位姿估计（Structure-from-Motion）

**目的**：照片只有像素，没有"相机在哪、朝哪"。本步从 87 张无序照片反推出**每张的相机位姿(外参)+ 相机内参 + 一片稀疏三维点**。这是三维重建的地基——位姿不准，后面全错。

**原理**：SfM。① 每张图提取 SIFT 特征点；② 图与图之间匹配同名点；③ 增量式三角化，联合求解相机位姿与三维点（Bundle Adjustment 优化）；④ 按解出的畸变参数把图像去畸变成理想针孔。

**命令**：

```cmd
python scripts\preprocess\convert.py --data_path E:\lowalt3d\data
```

> 注意是父目录 `data`（convert.py 把每个子目录当一个场景）。

**内部四阶段**（会依次在屏幕滚动）：
| 阶段 | 作用 |
|---|---|
| feature_extractor | 提 SIFT 特征 |

使用 GPU SIFT 算法从每张输入图像中提取局部图像特征点（关键点），并计算描述子，用于后续图像间的匹配；每张图提取了 约 8000~14000 个 SIFT 特征点并将这些特征数据写入 database.db数据库中。

![](C:\Users\Administrator\AppData\Roaming\marktext\images\2026-07-20-15-50-58-image.png)| exhaustive_matcher | 所有图两两匹配 |
对数据库中的所有图像进行**两两之间的穷尽匹配**，通过特征相似度找到同名点，并进行几何校验（如极线约束）以剔除误匹配；建立了 **3206 对有效图像匹配**（`3206 in 0.024s`），这意味着这些图像对之间拥有足够多的共视区域，为后面的三维重建奠定了基础![](C:\Users\Administrator\AppData\Roaming\marktext\images\2026-07-20-15-52-04-image.png)| mapper | 增量式 SfM，解位姿+稀疏点 |
利用匹配结果，通过增量式的方式（从一个初始图像对开始，逐步注册新图像），恢复每张相机的精确位姿（旋转 R 和平移 t），并生成稀疏的三维点云。过程中会反复执行局部和全局光束法平差（BA）优化。![](C:\Users\Administrator\AppData\Roaming\marktext\images\2026-07-20-15-52-28-image.png)| image_undistorter | 去畸变，整理成 3DGS 输入格式 |

利用 `mapper` 估计出的**相机内参（焦距、主点、畸变参数）**，将原始的带畸变图像（OPENCV 模型）校正为符合**理想小孔成像（PINHOLE）模型**的无畸变图像，并同时转换对应的相机参数文件。

![](C:\Users\Administrator\AppData\Roaming\marktext\images\2026-07-20-15-52-42-image.png)

**成功标志**：生成

```
data\scene01\images\        ← 去畸变后的图
data\scene01\sparse\0\      ← cameras.bin / images.bin / points3D.bin
```

**记录要点**（截图）：mapper 输出的 `Registered images: XX`（越接近 87 越好；太少说明重叠不足或纹理弱）。

**耗时**：约 5–15 分钟。

---

## Step 2 · PGSR 训练（平面约束 3D 高斯重建）

**目的**：用 Step1 的位姿+图像，学出场景的**稠密几何与外观**，表示成一堆"平面化的 3D 高斯椭球"。

**原理**：3DGS 用各向异性高斯椭球表示场景，通过可微光栅化把椭球渲染成图像、与真实照片比对、反向传播优化椭球参数。PGSR 额外加**平面约束**，逼高斯变扁、贴合建筑表面——这样后面才能提出干净的三角网格。

**命令**：

```cmd
python train.py -s E:\lowalt3d\data\scene01 -m E:\lowalt3d\outputs\scene01_pgsr -r 2 --max_abs_split_points 0 --opacity_cull_threshold 0.05
```

**参数含义**：
| 参数 | 作用 |
|---|---|
| `-s` | 输入（Step1 的 scene01 目录）|
| `-m` | 输出模型目录 |
| `-r 2` | 图像降采样 2 倍，**加速训练**（无人机图很大）|
| `--max_abs_split_points 0` | 弱纹理场景（屋顶/道路）关闭该切分，**防过拟合** |
| `--opacity_cull_threshold 0.05` | 剔除低不透明度高斯，**减少浮点噪声** |

**成功标志**：`outputs\scene01_pgsr\point_cloud\iteration_30000\point_cloud.ply` 等文件生成。

![](C:\Users\Administrator\AppData\Roaming\marktext\images\2026-07-20-17-49-07-image.png)

**记录要点**（截图）：训练进度条、loss 下降趋势、最终迭代数（默认 30000）。

**耗时**：约 30–60 分钟（取决于 `-r`）。

---

## Step 3 · 网格提取（TSDF 融合）

**目的**：把隐式的高斯场转成**显式三角网格 (mesh)**。原因：通信仿真靠射线与三角面片求交，高斯椭球不能直接做射线检测。

**原理**：从优化好的高斯渲染出每个视角的深度图/法线图 → 用 TSDF（截断符号距离场）融合成一张水密三角网格 → 清理浮点/断片。

**命令**（先用默认，跑完看效果再调）：

```cmd
python render.py -m E:\lowalt3d\outputs\scene01_pgsr --max_depth 10.0 --voxel_size 0.01 --use_depth_filter
```

**参数含义**：
| 参数 | 作用 |
|---|---|
| `--max_depth` | TSDF 融合的最大深度。⚠️ **单位是 COLMAP 归一化坐标，不是米**；场景不同要调。太小→网格残缺，太大→纳入远处噪声 |
| `--voxel_size` | 体素分辨率。越小越细、越慢；建筑场景 0.005~0.02 常用 |
| `--use_depth_filter` | 过滤不可靠深度（无人机视角不足时**建议开**）|

**成功标志**：模型目录下生成融合网格 `.ply`（控制台会打印实际路径，**记下来**）。

**记录要点**（截图）：网格顶点数；用 **MeshLab** 打开看形状是否成建筑轮廓。若网格空/破 → 调 `--max_depth`（先看 Step1 稀疏点云的尺度范围再定）。

---

## Step 4 · 材质分配（多视角光线投票）✅ 已完成

**目的**：给网格每个三角面片打材质标签，作为电磁属性来源。

**原理**：对 87 个相机，从相机穿过每像素向网格投射光线(Open3D RaycastingScene)，取**首个命中面片**(自动处理遮挡) → 在该像素的 2D 材质分割图上取材质 → 加权投票。所有相机投完，面片材质 = 得票最高类。⚠️ 2D 分割当前为**占位启发式**(颜色规则)，方向 A 换成训练好的分割模型。

**命令**：

```cmd
python E:\lowalt3d\scripts\assign_material.py --mesh E:\lowalt3d\outputs\scene01_pgsr\mesh\tsdf_fusion_post.ply --cameras E:\lowalt3d\outputs\scene01_pgsr\cameras.json --images E:\lowalt3d\data\scene01\images --out_dir E:\lowalt3d\outputs\scene01_pgsr\material
```

**成功标志**：`material\face_materials.csv`(每面片材质ID+电磁参数) + `material_mesh.ply`(材质上色)。
**本次结果**：植被正确判绿、地面判灰、亮路面被占位规则误判为玻璃(蓝)——占位分割的正常表现，链路验证通过。

---

## Step 5 · 电磁映射 + 射线路损 ✅ 已完成

**目的**：材质 → ITU-R P.2040 电磁参数 → 射线传播算路损 → 出路损图/LoS 图（P0 最终成果）。

**原理**：每个接收点 RX(向下投射找地表、放地表上方)，从发射点 TX 发线：未遮挡=视距，自由空间损耗；被遮挡=自由空间 + 沿途穿透面片 P.2040 损耗累加。（P0 只算直射+穿透，忽略反射/绕射。）

**命令**：

```cmd
python E:\lowalt3d\scripts\simple_pathloss.py --mesh E:\lowalt3d\outputs\scene01_pgsr\mesh\tsdf_fusion_post.ply --materials E:\lowalt3d\outputs\scene01_pgsr\material\face_materials.csv --out_dir E:\lowalt3d\outputs\scene01_pgsr\pathloss --grid 60
```

**成功标志**：`pathloss\pathloss_heatmap.png` / `los_map.png` / `pathloss_grid.npz`。
**本次结果**：有效接收点 2727/3600，NLoS 13.6%，路损 52–148 dB @3.5GHz。⚠️ `scale=1` 未地理配准，**绝对 dB 仅供参考，相对分布(遮挡区更高)有效**；配准后填 `--scale` 得真值。

---

## P0 完成 · 阶段成果存档

**闭环链路**（87 张照片 → 路损图，全通）：

```
87张无人机照片
 → COLMAP 位姿 (87/87 注册)
 → PGSR 高斯场 (280万高斯, -r4/15000iter)
 → TSDF 网格 (420万顶点 / 805万面片)
 → 多视角材质投票 (占位分割: 植被/地面/混凝土)
 → ITU-R P.2040 电磁参数
 → 射线路损图 (NLoS 13.6%, 52–148 dB @3.5GHz)
```

**结论**：验证了开题核心链路"材质感知三维地图 → 通信仿真"可打通。

**已知 P0 局限**（后续改进方向）：
- 材质是占位启发式 → **方向 A：换训练的分割模型**（进行中）
- 网格较糙(-r4/15000) → 方向 B：`-r 2 --iterations 30000` 重训
- 未地理配准(scale=1) → 方向 C：GCP/EXIF 求真实 `--scale`
- 只算直射+穿透 → 方向 D：补一次镜面反射
- 纯 UAV 无多源 / 未上 Cesium-UE → 方向 E/F（对齐开题创新①③）

**关键脚本**（均在 `scripts/`）：`assign_material.py`(材质投票) · `simple_pathloss.py`(路损) · `material_em_fusion.py`(体素融合, 备用)

---

## 常见问题速记

| 现象                                | 原因 / 处理                             |
| --------------------------------- | ----------------------------------- |
| COLMAP 报某选项 `unrecognised option` | 版本差异，去掉该选项即可（发我）                    |
| mapper Registered images 很少       | 重叠/纹理不足；换更连续的图或降 SIFT 阈值            |
| 网格空/破碎                            | `--max_depth` 不对（单位是归一化，不是米），按场景尺度调 |
| 训练/编译报显存不足                        | 加大 `-r`（如 `-r 4`）降分辨率               |
