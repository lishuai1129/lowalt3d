# CLAUDE.md

给 Claude Code（以及第一次接手这个仓库的人）看的说明。先读"一分钟看懂"和"五条铁律"，
其余部分遇到相关问题再查。

---

## 一分钟看懂

**这个项目做什么：** 用无人机航拍照片重建城市三维模型，并给模型的每个面标上材质
（混凝土、玻璃、植被……）和对应的电磁参数（ITU-R P.2040），然后拿去做低空无线信号的
路损/射线仿真。这是一个毕设研究管线。

**一条主线：**

```
航拍照片
 → COLMAP 求相机位姿
 → PGSR（3DGS 的一个变体）训练 + TSDF 出三角网格
 → GPS 地理配准（换算成米、东北天朝向）
 → 2D 材质掩码（SegFormer）+ 窗户候选（Grounding DINO + SAM）
 → 把材质"学"进每个 3D 高斯 → 投票到网格面片
 → 几何规则清洗（材质裁决链）
 → 得到 face_materials.csv（每个面一行材质 + 电磁参数）
 → 三个仿真器：Python 路损 / Wireless InSite（.city）/ UE 的 C++ 射线追踪
```

**代码长什么样：** 没有构建系统、没有测试、没有 lint。`scripts/` 下每个文件都是一个独立的
命令行工具，按顺序一个个跑。各步骤之间**只通过文件传递数据**，没有共享的运行时。
另有 `ue/RFRayTracer.{h,cpp}` 是 UE5 的 C++ 射线追踪器（拷进 UE 工程 `Source/` 用，这里不编译）。

代码注释、文档、提交信息都用中文；标识符和专业术语保留英文原样。

---

## 五条铁律（改代码前必须知道）

1. **材质类别顺序不能乱。** 6 类固定为
   `background(0) vegetation(1) asphalt(2) concrete(3) glass(4) metal(5)`。
   2D 掩码的像素值就是这个序号。下面三处必须完全一致：
   `configs/classes.txt`、`configs/radio_params.yaml` 的 `class_order`、
   `scripts/material_common.py::_DEFAULT_CLASS_ORDER`。
   （`autolabel.py::_KEYWORDS` 是按类名做键的字典，顺序不影响序号，但键名必须拼对，拼错会被静默跳过。）
   只有 `smooth_face_materials.py`、`label_windows_geom.py`、`export_textured_ue.py` 会 assert 检查，
   其他脚本不检查——顺序一错，所有 `material_id` 整体错位，**不报错**。
2. **`face_materials.csv` 是全管线的公共接口。** 第 i 行 = 网格的第 i 个面，所以它**只对生成它的那份网格有效**。
   换网格（补洞、抽面、重新出网格）就必须从"高斯材质 → 网格"那一步重跑。
3. **COLMAP 的单位不是米。** 1 个 COLMAP 单位 = `geo_transform.json` 里的 `scale` 米（scene02 约 33.06 m）。
   很多阈值参数吃的是 COLMAP 单位（见"单位与坐标"一节），按米或厘米去填会全错。
4. **全局只做一次 SfM。** 大场景分块训练时，所有块必须共用同一次 COLMAP 结果；各块单独跑 COLMAP 的坐标系对不上，没法合并。
5. **不要"造"不存在的东西。** 不用纯颜色规则产玻璃标签；不把低置信度检测直接当训练硬标签；
   不用生成式补全（NeRFiller/GaussFiller 之类）去填没拍到的表面——对幻觉内容做电磁仿真是正确性问题，不是画质问题。

---

## 目录和路径

仓库实际位置：**`E:\rebuiuld3d\lowalt3d`**。（很多脚本 docstring、README、`AGENTS.md`、
`RFRayTracer.h` 里还写着旧路径 `E:\lowalt3d`，那是过时的。）

| 路径 | 内容 | 是否入 git |
|---|---|---|
| `scripts/` | 全部 Python 管线脚本 | 是 |
| `configs/classes.txt` | 6 个材质类别（顺序有意义） | 是 |
| `configs/radio_params.yaml` | 每类的电磁参数（eps_r、sigma、穿透损耗）+ 频率等仿真参数 | 是 |
| `configs/visual_pbr.yaml` | 每类的**视觉**粗糙度/金属度，只给 `export_textured_ue.py` 用，和电磁参数无关 | 新文件，未提交 |
| `ue/` | UE5 C++ 射线追踪器 | 是 |
| `patches/pgsr_local.patch` | 对 PGSR 的本地改动（见下） | 新文件，未提交 |
| `assets/`、`dataset.png` | README 用的图 | 是 |
| `.kiro/specs/literature-collection/` | 一份"文献收集系统"的需求/设计文档，不是管线代码 | 是 |
| `data/`、`outputs/`、`repos/`、`tools/` | 数据、产物、第三方代码。**是符号链接，指向 `D:\lowalt3d\lowalt3d\...`** | 否（gitignore） |

- `repos/pgsr` = [PGSR](https://github.com/zju3dv/PGSR)，已按 `patches/pgsr_local.patch` 打过补丁：
  - `train.py` 新增 `--extent_override`（分块训练必需，见"分块训练"）；
  - `scene/gaussian_model.py` 去掉 pytorch3d 依赖；
  - `scripts/preprocess/convert.py` 重写为 `subprocess` 调用，会自动转绝对路径，`--data_path` 可以给单个场景目录或其父目录。
  重新 clone PGSR 时要重新打这个补丁。
- `tools/COLMAP` = COLMAP Windows-CUDA 预编译版。
- `.gitignore` 还忽略了 `docs/ experiments/ checkpoints/ reports/ data01/ outputs01/` 和
  `*.ply *.glb *.pt *.npz *.city *.dae` 等大文件。**不要把这些加进 git。**
- 当前 `outputs/` 里有 scene02 的分块结果（`scene02_b00` … `b05`，其中 b02 再拆成 `b02a/b02b`）和合并结果 `scene02_merged`。

---

## 环境和怎么跑

- **Windows 原生。** README 示例是 cmd 语法（`set SCENE=`、`^` 续行、`\` 路径）。这里的交互 shell 是 bash，
  用 `/` 和 `$SCENE` 也行。
- **先激活 conda 环境**（Python 3.10 + PyTorch 2.8 cu128，已在 RTX 5070 Ti / Blackwell sm_120 上验证）。
  系统默认的 Python 连 numpy 都没有，脚本和 `--selfcheck` 都跑不起来。
- 依赖：`numpy scipy opencv-python open3d pillow pycollada trimesh matplotlib pyyaml plyfile`；
  可选 `transformers`（SegFormer / GDINO+SAM）、`xatlas`（`export_textured_ue.py`）。
- **`scripts/*.py` 在仓库根目录运行。** 脚本自己按 `__file__` 找 `material_common` 和 `repos/pgsr`，
  不依赖当前目录；但 README 里的参数都是相对仓库根的路径。
- **PGSR 自己的 `convert.py`、`train.py`、`render.py` 在 `repos\pgsr` 下运行。**
- 完整配方在 `README.md` 的"快速开始"，全程只用一个 `SCENE` 变量切换场景。
- 重型库（torch、open3d、trimesh、cv2）放在 `main()` 里导入，让 `--help` 秒出、导入错误局部化，**新脚本请照做**。
  例外：`train_gaussian_material.py` 的参数表依赖 PGSR，`--help` 也会加载 torch 和 PGSR；`eval_masks.py` 在顶部导入 cv2。
- 带 `--selfcheck`（用合成数据自测）的脚本：`close_facade_holes`、`detect_windows`、`split_scene`、`merge_blocks`、
  `smooth_face_materials`、`label_windows_geom`、`export_textured_ue`。它是直接查 `sys.argv` 实现的，`--help` 里看不到。

---

## 每一步用哪个脚本

按 README 的执行顺序：

| 步骤 | 脚本 | 输入 → 输出（大意） |
|---|---|---|
| 1 位姿 | PGSR `convert.py`（在 `repos\pgsr` 下） | `data/<scene>/input/` 原图 → 去畸变 `images/` + `sparse/` |
| 2 降采样 | `downscale_images.py` | `images/` → `images_N/`（文件名不变；训练的 `-r` 要填同一个 N） |
| 3 训练出网格 | PGSR `train.py` + `render.py` | → 高斯 `point_cloud.ply`、`cameras.json`、`mesh/tsdf_fusion_post.ply` |
| （可选）清高斯 | `clean_gaussians.py` | 删 NaN/离群/漂浮/低透明度的高斯，属性完整保留 |
| （可选）补洞 | `close_facade_holes.py` | 用 Open3D `fill_holes` 封小洞（**不是**平面拟合；默认不去浮块）。用了就要从第 7 步重跑 |
| （大场景）分块 | `split_scene.py` → 逐块训练 → `merge_blocks.py` | 见"分块训练" |
| 4 配准 | `georeference.py` | GPS EXIF + `cameras.json` → `geo_transform.json`（`ENU米 ≈ scale·R·colmap + t`） |
| 5 2D 掩码 | `autolabel.py`（SegFormer 基线）→ `detect_windows.py`（加窗户候选） | → `masks_auto/` → `masks_win/` |
| 5' 人工标注路线 | `labelme2mask.py` → `train_seg.py` → `infer_seg.py`；`eval_masks.py` 评估 IoU/召回 | 提高召回的正路 |
| 6 材质学进高斯 | `train_gaussian_material.py` | 掩码 + 训练好的高斯 → `gaussian_material.pt` |
| 7 高斯 → 面片 | `gaussian_to_mesh_material.py` | → `face_materials.csv` + `material_mesh.ply`（原始投票，未清洗） |
| 8 材质裁决 | `smooth_face_materials.py` → `label_windows_geom.py` | → `material_clean/` → **`material_win/`** |
| 9 仿真 + 可视化 | `simple_pathloss.py`、`visualize_sim.py` | → `pathloss_grid.npz`、热力图、3D 视图 |
| 10 导出仿真用 | `export_dae.py`（InSite）、`export_gltf_ue.py`（UE/Cesium） | → `scene.city`/`.dae`；`.glb` + 材质槽 + 经纬高锚点 |
| 11 导出展示用 | `export_textured_ue.py` | → 分块的带照片纹理 + PBR 的 `tile.glb`、`collision.glb`、`tiles.json` |
| 对比 | `compare_engines.py` | UE `coverage_ue.csv` vs Python 网格 → RMSE / 相关系数 |

**下游一律用 `material_win/`**（裁决链的输出）。第 7 步的 `material_gauss_cons/` 是原始票数：墙面有椒盐、屋顶有玻璃、metal 全是噪声。

`scripts/material_common.py`：共用的类别表、P.2040 参数加载（`load_radio_params`，yaml 缺失时有内置回退）、调色板 `_PALETTE`。
调色板也用来给 `material_mesh.ply` 上色，`export_gltf_ue.py` 再按颜色反推类别。

`scripts/md2docx.py` 不是管线的一部分，只是把 Markdown 报告转成 Word（需要 pypandoc 和 mermaid-cli，`mmdc` 路径是写死的）。

---

## 关键设计

### 两个接口约定

1. **`face_materials.csv`**：列为 `face_id,material_id,material_name,eps_r,sigma,penetration_loss_db`。
   `gaussian_to_mesh_material.py` 的输出多两列 `eps_r_mix,sigma_mix`（软混合参数），经过 smooth/label 之后只剩 6 列。
   读的一方按**行号**对应面片：`smooth`/`label` 只检查"行数 = 面数"，`simple_pathloss` 连这个都不检查。
2. **类别顺序**：见铁律 1。

### 材质怎么学出来的（主线，"方向A"）

- `train_gaussian_material.py` 给每个 3D 高斯加一个可学习的 K 维材质向量，几何/颜色/透明度全部冻结，只训材质。
- 技巧：借用 PGSR 渲染器的 `override_color` 接口，把材质 logits 当"颜色"每次 3 个通道渲出来，
  和 2D 掩码做交叉熵——**不用改 CUDA**（Feature-3DGS / LangSplat 的做法）。
- 可选项：类别平衡 + Focal、空间平滑（`--lambda_smooth` 默认就是 0.5，关掉要显式传 0）、
  跨视角一致性 `L_cons`（用渲染深度把邻近视角的材质投过来算 KL）、训练侧几何门 `--lambda_gate`。
- `gaussian_to_mesh_material.py` 再按"透明度 × 距离（× 法向一致度）"加权的 kNN 投票，把材质转到网格面片上。
- 旧的直接投票基线 `assign_material.py` 和颜色规则标注 `autolabel_color.py` **都已删除**，不要恢复
  （颜色规则会把阴影和暗地面标成玻璃，航拍上召回≈0）。

### 玻璃（窗户）：检测只是候选，几何说了算

- **候选层**：`detect_windows.py`（Grounding DINO + 可选 SAM）给出"可能是窗"的 2D 区域。
  `--box_threshold` 不低于 0.12（推荐 0.15，宁可漏不可误），`--min_win_px` 去掉小碎点。
- **裁决层**（与场景无关的几何规则，最终拍板）：
  - 玻璃/金属只能在竖直面上（`smooth --geo_gate --horiz_forbid glass,metal,vegetation`）；
  - 窗是立面上凹进去的坑（`label_windows_geom.py --recess_*`）；
  - 真窗成片出现，太小的连通块退回邻居的多数类（`smooth --min_win_area`，本网格一个窗约 30–60 面，取 20）。
- 推荐链路：`detect(0.15 + min_win_px)` → `train_gaussian_material --lambda_gate` → `gaussian_to_mesh_material`
  → `smooth --geo_gate --min_win_area` → `label_windows_geom` → 导出。
- metal 没有 2D 标签，残留的 metal 面全是噪声，用 `smooth --drop_unlabeled metal` 清掉，否则等于在仿真里凭空造出导体反射面。

### 分块训练（大场景）

为什么分块：单次训练要靠 `--max_all_points 3000000 --max_abs_split_points 0 --opacity_cull_threshold 0.05`
压显存，代价是植被和细结构变残。分成 N 块就有 N 倍的高斯预算。

- 流程：`split_scene.py` → 逐块 PGSR 训练 → `merge_blocks.py` 合成一份 `point_cloud.ply` → 后面的 TSDF/材质/导出完全不变。
- `split_scene.py` 在 ENU 米制坐标下分块（复用 `georeference.py` 的函数），按 VastGaussian 的思路：
  `--grid NxM` 按**相机数**均衡切块（不是按米），`--overlap` 是**比例**；每块有 `core`（核心区）和 `expanded`（扩展区）。
  块编号先按 E 方向列、再按 N 方向排。可用 `--split_block` 把单个过大的块再拆（如 b02 → b02a/b02b）。
- 每块是 COLMAP **文本**子模型，2D 观测表留空（只给 PGSR 用，COLMAP 自己打不开）；
  `-i ../../images_4` 指回共享图目录，不复制图片。
- **逐块训练要加 `--extent_override <整场景 cameras_extent>`**（`split_scene.py` 会打印这个值并写进 `blocks.json`）。
  否则各块按自己的相机算 extent，块越窄致密化越细，高斯数暴涨。这个参数来自本地补丁，原版 PGSR 没有。
- `merge_blocks.py` 只保留中心落在各块 `core` 里的高斯，重叠区只算一次。注意：
  - 一定要传 `--images`（如 `images_4`），否则合并后的 `cfg_args` 里图片路径还是块目录的相对写法，`render.py` 找不到图；
  - `--out` 要写成 `<合并目录>/point_cloud/iteration_N/point_cloud.ply`，`cfg_args`/`cameras.json` 会写到它往上两级；
  - 需要 `plyfile`。
- 不要用 `colmap model_cropper/model_splitter`：scene02 上裁到 55% 的点仍保留 674/678 张图，等于没切。

### 单位与坐标（最容易出错）

- COLMAP 世界坐标的单位和朝向都是任意的。`georeference.py` 用 Umeyama 相似变换（尺度 + 旋转 + 平移）
  把 COLMAP 相机中心对齐到 GPS 换算的 ENU 米坐标，写出 `geo_transform.json`。至少要 3 张带 GPS 的图；
  残差 > 5 m 说明是消费级 GPS 噪声，需要 RTK/像控点。
- **吃 COLMAP 单位的参数**：`label_windows_geom --recess_min/--recess_max`、`smooth --gate_voxel`、
  `close_facade_holes --hole_size`、`clean_gaussians --cut_above`、`train_gaussian_material/gaussian_to_mesh_material --crop_bbox`。
  先跑一次看打印的分位数再定阈值。
- **竖直轴的判断各脚本不一样**：`train_gaussian_material` 按相机位置自动推断；`smooth` 默认按法向自动推断；
  `label_windows_geom` 默认写死 `--up_axis 2`；`simple_pathloss` 取包围盒最短的轴，天空在哪侧要靠 `--cameras` 判断（不传就默认正方向）。
  README 给 smooth/label 都传了 `--up_axis 2`——如果场景的竖直方向不是 Z，几何门和判窗会整体失效，先看打印的"水平/竖直面数"确认。
- `simple_pathloss.py` 要传 `--scale`（取 `geo_transform.json` 的值）绝对 dB 才有意义；不传只能看相对分布。
- 导出：`export_dae.py`/`export_gltf_ue.py` 传 `--geo_transform` 才会是米制、Z 朝上。UE 导出还会转成 glTF 的 Y-up，并算出 Cesium 用的经纬高锚点。
  - 模型躺平/尺寸不对 → 没传或传错了 `--geo_transform`（**路径写错会静默退回 scale=1**，只有 dae 会在最后打印一句警告）；
  - 地面悬在 z≈-80（ENU 原点在无人机高度）→ 加 `--ground_z0`（纯平移，不影响路损）。
- `split_scene.py` 写的 `geo_transform.json` 原点高度取 GPS 最低值，`georeference.py` 取平均值，两份文件的 `t` 不同，不要混用。
- `cameras.json`（PGSR/3DGS 约定）：`position` = 相机中心，`rotation` = 3×3 相机到世界，主点在图像中心。

### 三个仿真器

- `simple_pathloss.py`：Python，自由空间损耗 + P.2040 穿透损耗，Open3D 射线判断视距。**目前只有直射 + 穿透，没有反射/绕射。**
  电磁参数全部来自 CSV，yaml 只读频率（读不到就静默用 3500 MHz）。
- `ue/RFRayTracer.cpp`：UE LineTrace，同样的物理模型。它**不读 CSV**，而是按命中处的材质槽名（`EM_<类名>`，由 `export_gltf_ue.py` 生成）
  查 DataTable 或 C++ 里写死的回退表。Details 面板有 `RunProbe`（建议先用）、`RunCoverage`、`RunRayFan`、`ClearDebug` 四个按钮。
- Wireless InSite：**加载 `scene.city`，不要加载 `scene.dae`**。InSite 的 COLLADA 导入会把多材质合成一种；
  `.city` 是按 InSite 输出逆向出来的原生格式，逐面写材质索引——改写入代码时保持关键字结构原样。
- `compare_engines.py` 只对比 Python 和 UE（没有 InSite 输入），而且**只按格子序号**对齐：
  UE 里的 `GridCenter/GridExtent` 必须手动配成和 Python 网格一样的范围，否则对比没有意义。

---

## 踩坑清单

**PGSR 训练 / 显存**
- 显存要在第 7000 轮之后再判断：两个几何正则在 7000 轮才开启，之前看着很省，之后会断崖。一旦超显存不是变慢一点，
  而是溢出到共享内存，PCIe 来回倒腾，慢 100 倍以上且不会恢复。
- 显存 ≈ 单图像素数 × 几何正则缓冲 + 高斯数量，两项都可能单独撑爆：
  高斯太多 → 分块（别用上面那三个省显存参数）；单图像素太多 → 降 `-r`（分块帮不上）。先判断是哪一项。
  实测（16 GB）：scene03 `-r 2` 的一个块（每图 11.36 Mpx）只到 285 万高斯就挂了；scene03 `-r 8` 整场景（每图 0.71 Mpx）
  过 7000 轮还有 3.4 it/s，致密化到 530 万后掉到 1.97 s/it。
- 分块后第 7000 轮高斯数偏高，先检查 `--extent_override` 有没有传，再考虑继续切块。
- densification 参数保持默认，调它们会加重植被空洞。
- `--ncc_scale` 是**除数**（0.5 反而把 NCC 图放大 4 倍像素），保持 1.0。
- `-i` 是拼在 `-s` 后面的相对路径：从块目录（`data/<scene>/blocks/bXX`）训练写 `../../images_4`，`-s` 是场景根时写 `images_4`。
- `--preload_img` 在 `-r 2` + 678 张时会撑爆内存，别加。
- `convert.py` 保持 exhaustive 匹配：放开特征数 + `spatial_matcher` 实测把 PSNR 从 30.2 降到 28.8。

**材质**
- 网格空洞是采集缺口（俯拍拍不到建筑底部、窗洞内壁、植被背面），任何重建方法都补不出没拍到的面。
- 几何门是"软"的：违规面改成所在体素的多数类，如果多数类本身就是禁止类，就保持不变。
- `--horiz_forbid`/`--gate_classes`/`--drop_unlabeled` 等参数里的类名拼错会被静默忽略。
- `label_windows_geom` 假设法向朝外；墙被大面积判成窗时试 `--flip`。
- `smooth`/`label` 的 `--k` 是邻居面个数，不是半径：网格越密，覆盖的空间越小。
- `train_gaussian_material` 和 `gaussian_to_mesh_material` 的 `--crop_bbox` 必须一致，否则高斯数对不上直接 assert。
- `gaussian_to_mesh_material` 的类别顺序取自 `radio_params.yaml`（`--classes` 参数其实没用到），
  `train_gaussian_material` 取自 `classes.txt`——两者不一致时不会报错。
- 掩码里如果有 255（忽略值）：`train_gaussian_material` 会把它截断成 5=metal 参与训练；`train_seg` 开 focal 时会越界崩溃。
  `labelme2mask` 产出的掩码没有 255，没画到的区域全是 0=background。

**导出 / 仿真**
- `--target_faces` 抽面：新面继承最近原始面的类别，一般不丢类，但很小的类可能被抽没；会磨平窗洞这类小几何。
  `0` = 不抽（9.7M 面约 1.7 GB 的 `.city`，InSite 基本吃不下），实用约 200 万。三个导出脚本的默认值各不相同（50 万 / 30 万 / 200 万）。
- InSite 日志出现 `Failed to load OptiX` = 这一版对新显卡没有 GPU 加速，把 Reflections 降到 2、Ray spacing 设 1.0° 提速。
- `export_textured_ue.py`：`--images` 必须是和 `cameras.json` 同一套的去畸变图（降采样图会自动缩放内参）；
  不传 `--geo_transform` 时 `--tile_m` 的单位就不是米；UE 里 Uniform Scale 100 导入，全部放世界原点即可拼合，开 Nanite，碰撞用 `collision.glb`。
  已知局限：没做曝光统一，没拍到的地方没纹理。
- UE C++ 里材质名一个关键词都匹配不上时，穿透损耗默认 0 dB（墙对信号完全透明），且不报警。


