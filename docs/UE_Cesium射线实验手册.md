# UE + Cesium for Unreal 射线实验手册（详版 · 从建项目开始）

> 目标：把材质感知 3DGS 电磁地图导入 **Unreal Engine 5.4.4 + Cesium for Unreal**，
> 用 **UE 几何射线（LineTrace）** 做射频覆盖、读每次命中的电磁材质（ITU‑R P.2040），
> 并与 Python `simple_pathloss` / Wireless InSite 做三引擎对比。
> 面向 UE 新手，逐菜单、逐数值。示例场景 `scene02`（深圳城区）。

> 🈶 **中文版 UE 说明**：下面菜单项写成 **中文（English）**。注意两类**仍显示英文**、不用找中文：
> ① **Cesium 插件**的界面（CesiumGeoreference、Origin Latitude… 插件未本地化）；
> ② **`RFRayTracer` 自己的属性/按钮**（Run Coverage、Frequency MHz、Grid N… 是我们写的 C++，UE 不翻译）。
> 找不到某个自带选项时，善用**面板顶部的搜索框**（项目设置、细节面板都有）。

---

## ⚠️ 先理解一个关键概念（写论文别搞错）

**UE 的硬件光线追踪（Lumen / RTX / Hardware Ray Tracing）是给光学渲染/光照用的，不是电磁 RF 传播。**
UE 没有内置射频射线追踪。我们用 UE 的**碰撞射线** `LineTraceByChannel` 做**几何射线投射**：
判 LoS/NLoS、一阶反射，并在每次命中时**查该面片的电磁材质**。这就是本实验里 “UE 射线追踪” 的
确切含义（几何意义）。物理模型（自由空间损耗 FSPL + 沿途穿透损耗）与 `simple_pathloss.py`
**同源**，因此三引擎结果可对比——这正是论文的验证逻辑。

---

## 0. 前置产物（已由 `export_gltf_ue.py` 生成，位于 `outputs\scene02_pgsr\ue\`）

| 文件                 | 内容                                                                  | 关键值（scene02） |
| ------------------ | ------------------------------------------------------------------- | ------------ |
| `scene_ue.glb`     | 米制网格，按电磁类分材质槽 `EM_background/vegetation/ground_road/concrete/glass` | 30万面，5 槽     |
| `ue_materials.csv` | 每类 P.2040 参数（eps_r,sigma,penetration_loss_db,model,RGB）             | 见文件          |
| `ue_anchor.json`   | Cesium 锚点经纬高 + TX 建议位 + bbox                                        | 见下           |

`ue_anchor.json` 里要用到的数（scene02）：

- **锚点（质心）**：纬度 `22.5329830`，经度 `113.9352358`，高 `21.49` m
- **TX 建议**（UE 厘米，模型本地坐标，质心正上方 ~112m）：`X=0, Y=0, Z=11182`
- **场景尺寸**：679.9(东西) × 524.3(南北) × 99.2(高) m
- **bbox 半宽**（做 RX 网格用）：X≈34000 cm、Y≈26000 cm
- 坐标约定：导入 UE 后 `UE_X=东, UE_Y=北, UE_Z=上`

> 换场景时打开各自的 `ue_anchor.json` 取对应数即可。

---

## 1. 创建 UE 项目（C++ 空白工程）

我们要写 C++ Actor，所以从一开始就建 **C++** 工程（省去以后蓝图转 C++）。

1. 打开 **Epic Games 启动程序 →「虚幻引擎」→「库」→ 启动 5.4.4**（或直接开已装的 UE 5.4.4）。
2. 弹出的项目浏览器里：左侧选 **游戏（Games）** → 模板选 **空白（Blank）**。
3. 右侧 **项目默认设置（Project Defaults）**：
   - **蓝图（Blueprint）/ C++** 切到 **C++**（重要）。
   - 目标平台 **桌面（Desktop）**；质量预设 **最大（Maximum）**；
     **初学者内容包（Starter Content）** 可不勾；**光线追踪（Raytracing）** 不用勾。
4. 底部设 **项目名称（Project Name）**（如 `LowAltRF`，别用中文/空格）与位置（如 `E:\ue\LowAltRF`）→ **创建（Create）**。
5. UE 会生成 C++ 工程并自动打开 **Visual Studio 2022** + **UE 编辑器**（首次编译稍等）。

## 2. 启用 Cesium for Unreal 插件

1. UE 里 **编辑（Edit）→ 插件（Plugins）**。
2. 搜索框输入 **Cesium** → 找到 **Cesium for Unreal** → 勾选 **已启用（Enabled）** → 按提示 **重启（Restart）编辑器**。
3. 重启后菜单栏出现 **Cesium** 页；若没看到：**窗口（Window）→ Cesium**。
4. （全球底图才需要）点 Cesium 面板的 **Connect to Cesium ion**（插件按钮，英文）→ 浏览器登录/注册**免费**
   Cesium ion 账号 → 授权 → token 自动写入工程。
   
   > 说明：ion token 只为加载**全球地形/影像底图**。我们自己的模型 + 射频实验**不依赖** ion，
   > 先不连也能跑（第 5–9 步）；数字孪生可视化（第 10 步）才需要。

## 3. 添加 C++ 射线追踪器类

1. UE 菜单 **工具（Tools）→ 新建 C++ 类（New C++ Class）**。
2. 父类选 **Actor** → **下一步（Next）** → 类名填 **`RFRayTracer`**（位置保持 **公有 / Public**）→ **创建类（Create Class）**。
3. UE 生成 `Source\LowAltRF\RFRayTracer.h` 和 `.cpp`（骨架），并触发一次编译。
4. **用我们的实现覆盖骨架**：把
   `E:\lowalt3d\ue\RFRayTracer.h` 和 `E:\lowalt3d\ue\RFRayTracer.cpp`
   的**全部内容**分别复制，粘贴替换掉刚生成的两个同名文件（路径 `Source\LowAltRF\`）。
   
   > 我的文件里 `#include "RFRayTracer.generated.h"`、`class ARFRayTracer : public AActor` 都与
   > 向导一致，直接替换内容即可（单模块工程无需 `_API` 导出宏）。
5. 编译：在 **Visual Studio** 里按 **Ctrl+Shift+B**（生成，配置选 *Development Editor / Win64*）；
   或在 UE 里点右下角 **编译（Compile）** / 实时编码 **Ctrl+Alt+F11**。
6. 编译通过后，在 UE 的 **快速添加（放置Actor / Place Actors）** 面板搜 `RFRayTracer` 能搜到。
   
   > 依赖模块 `Core/CoreUObject/Engine` 默认就有；万一链接报 `UPhysicalMaterial` 未解析，
   > 在 `Source\LowAltRF\LowAltRF.Build.cs` 的 `PublicDependencyModuleNames` 里补 `"PhysicsCore"` 再编。

## 4. 一次性项目设置

1. **编辑（Edit）→ 项目设置（Project Settings）**。
2. 左侧选 **引擎（Engine）→ 物理（Physics）**；在**顶部搜索框输入 `Support UV`**
   → 勾上 **Support UV From Hit Results**（中文版可能显示为 **“支持从命中结果得出UV”** 之类）
   → 按提示 **重启编辑器**。
   
   > 作用：射线命中才会返回**面片索引**，据此查材质槽名。不勾则材质全走兜底。

## 5. 导入模型 glb

1. 在 **内容浏览器（Content Browser）** 里新建/进入一个文件夹（如 `Content\Scene02`）。
2. 把 `E:\lowalt3d\outputs\scene02_pgsr\ue\scene_ue.glb` **拖进内容浏览器**（走 5.4 的 Interchange glTF 导入）。
3. 导入对话框默认即可（静态网格体 / Static Mesh；不勾 Nanite）→ **导入（Import）**。
   - 结果：内容浏览器里出现 **5 个静态网格体（带 `1` 后缀：`EM_background1 / EM_vegetation1 /
     EM_ground_road1 / EM_concrete1 / EM_glass1`）** + 5 个同名**材质实例**（无后缀）。**要摆进关卡的是那 5 个
     静态网格体（`…1`）**。它们**共享同一原点坐标**。
4. **摆进关卡并对齐尺度**：
   - 框选 5 个静态网格体（`EM_xxx1`）→ 一起拖进视口 → 它们**同原点重叠**，自动拼回完整场景。
   - 在 **大纲（Outliner）** 里框选这 5 个 Actor → **细节（Details）** 面板里设
     **位置（Location）=(0,0,0)、旋转（Rotation）=(0,0,0)**。
   - **缩放（Scale）**：看你怎么用——
     · **纯世界原点跑（无 Cesium）**：glb 是米、UE 是厘米 → 设 **100**。
     · **已用 Cesium 锚定到真实地形**：设成**视觉上与地形吻合的值**（实测 **1** 合适）。
     · 用 **8.0 点对点探针** 时**缩放随便**——它按实际 Actor 位置算，与缩放无关。
   - 看不到就按 **F** 键聚焦；仍看不到多半是缩放不对。
5. **设碰撞（让射线能命中）**——**不用属性矩阵**（它不列碰撞复杂度）。射线用的是逐三角碰撞
   （我的代码 `bTraceComplex=true`，导入网格默认自带三角碰撞），**只需保证网格挡 Visibility 通道**：
   - 大纲里框选这 5 个 Actor → 细节面板顶部**搜索框输入 `碰撞`** → 找 **碰撞预设（Collision Presets）**
     → 选 **BlockAll**（5 个一次改完）。
     
     > 备用（万一射线全穿透）：双击每个 `EM_xxx1` → 静态网格体编辑器右侧细节搜 **碰撞复杂度
     > （Collision Complexity）** → 选 **将复杂碰撞用作简单碰撞（Use Complex Collision As Simple）** → 保存。
     > 现在世界坐标单位=厘米，与 `ue_anchor.json` 的 cm 数一致：UE_X=东、UE_Y=北、UE_Z=上。

## 6. 材质 → 电磁参数绑定（二选一）

**A. 简单法（推荐，零额外操作）**：什么都不做。`RFRayTracer` 会用命中面片的**材质槽名**
（`EM_concrete`…）匹配脚本内置的 **P.2040 回退表**。前提：第 4 步勾了 **Support UV From Hit Results**。

**B. 严谨法（论文更规范）**：导入数据表（DataTable）

1. 把 `ue\ue_materials.csv` 拖进 **内容浏览器** → 弹出 **选择数据表行结构（Choose DataTable Row Type）** →
   选 **`FRFMaterialRow`**（编译后才会出现）→ 确定。
   
   > CSV 里多余列 model/R/G/B 会被忽略，只吃 eps_r/sigma/penetration_loss_db；RowName 用 `EM_*`。
2. 第 8 步把这个数据表拖给 `RFRayTracer` 细节面板里的 **Material Table** 属性（此属性是我们的 C++，显示英文）。

## 7. 放置发射机并编译检查

1. 从 **快速添加 / 放置Actor（Place Actors）** 面板搜 `RFRayTracer` → 拖进关卡。它的**位置即 TX**。
2. **细节（Details）** 面板里设 **位置（Location）=(0, 0, 11182)**
   （= `ue_anchor.json` 的 `tx_suggest_ue_cm`，质心正上方 ~112m）。
3. 核对细节面板其余参数（**以下都是 `RFRayTracer` 的 C++ 属性，中文版里也显示英文**）：
   - `Frequency MHz = 3500`，`Tx Power Dbm = 23`
   - `Grid N = 60`；`Grid Extent X = 34000`、`Grid Extent Y = 26000`（= bbox 半宽，取自 `ue_anchor.json`）
   - `Grid Center = (0,0)`，`Rx Height Cm = 150`，`Sky Trace Top Cm = 30000`
   - `Output Csv Path = E:/lowalt3d/outputs/scene02_pgsr/ue/coverage_ue.csv`
   - （用了严谨法才设）`Material Table` = 上一步的数据表

## 8. 跑射线实验

`RFRayTracer` **细节（Details）** 面板顶部有四个按钮（我们的 C++ 按钮，英文；**不用 Play、直接点**）。

### 8.0 最简测试：点对点射线 + 材质识别（★推荐先做这个）

只要几条楼间射线、能报出命中材质，就达到"射线识别材质"的目标——不用铺满屏的网格。

1. **放几个目标点**：**放置Actor（Place Actors）** 面板搜 **目标点（Target Point）** → 拖到视口，
   移动到**几栋不同建筑的墙面/楼顶**上（放 2~4 个即可）。
2. 选中 **RFRayTracer**（TX 位置就放在某栋楼顶或空中一点）→ 细节面板找 **Probe Targets**（数组）→
   点 **+** 加几个元素 → 每个下拉/拖入一个**目标点**。
3. 点 **Run Probe**：
   - 视口画出射线（**品红球=TX，青球=目标，黄=首个命中段，灰=命中后示意**）；命中处标材质名。
   - **窗口（Window）→ 输出日志（Output Log）** 里每条射线打印：
     `[RF][Probe i] 首击=concrete d=..m eps_r=.. 穿透=..dB` 和 `穿过 N 层: glass -> concrete ...`。
     
     > 材质识别以**输出日志**为准（一定有）；视口那行浮字是附带的，个别情况不显示不影响。
     > 本模式**用实际 Actor 位置、与缩放无关**——所以不必纠结模型缩放是 1 还是 100。

### 8.1 覆盖热图 / 射线扇（可选，做完最简测试再看）

- 生成 RX 网格 → 逐点判 LoS/NLoS → 算 FSPL + 穿透损耗 → Prx；
- 视口里按路损 jet 上色打点（蓝=低损、红=高损），TX 处一个品红球；
- 写 `coverage_ue.csv`（列：x_cm,y_cm,z_cm,los,pathloss_db,prx_dbm）；
- **窗口（Window）→ 输出日志（Output Log）** 里打印 `valid / LoS 比例 / PL 范围`。
  2. **Run Ray Fan**（射线扇，可视化）：从 TX 向下半球发射线——
     绿=打到地面/植被、红=打到建筑、橙=建筑上的一次镜面反射。复现你那张“射线源往四周发散”的图，
     但**读了真实材质**。
  3. **Clear Debug**：清除视口里的点/线。

> **先在世界原点跑通**（本步），确认覆盖点/射线正常、日志 LoS 比例合理，再去做第 10 步 Cesium 摆地球。
> 卡顿就把 `Grid N` 从 60 降到 40。

## 9. 三引擎对比（论文验证核心）

回到命令行（`E:\lowalt3d`，`conda activate E:\condaenvs\emmap`）：

```cmd
python scripts\compare_engines.py ^
  --py outputs\scene02_pgsr\pathloss_gauss_cons\pathloss_grid.npz ^
  --ue outputs\scene02_pgsr\ue\coverage_ue.csv ^
  --out_dir outputs\scene02_pgsr\compare
```

产出 `py_pathloss.png / ue_pathloss.png / diff.png` + 打印 **均值差 / 绝对均值 / RMSE / 相关系数 r**。
物理模型同源 → 两图分布应高度一致（r 高、RMSE 小），作为“方法自洽”证据。
InSite（`insite\scene.dae` 导入 Wireless InSite 跑）作**第三方独立引擎**佐证。

## 10. Cesium 地球锚定（数字孪生可视化，可选）

让模型按**真实经纬度**落到全球地形上，做数字孪生展示。（本节 **Cesium 相关名词都是插件英文**，不用找中文。）

> ⚠️ **顺序很重要**：务必**先跑完第 7–9 步的 RF 实验**（模型在世界原点即可），再做本节。
> **加 Cesium 前，先在大纲里删掉/隐藏**：模板自带的 **`Landscape`**（你模型自带地面，不需要它）
> 和模板自带的 **`DirectionalLight`/`SkyAtmosphere`/天空球**（否则和 `CesiumSunSky` 抢光，报"多个定向光竞争"）。
> 若加完 Cesium World Terrain 后模型变得极小/地形乱铺，就是**下面第 1–2 步的锚点没设**——地形默认加载在别处，
> 你的模型还在原点。设好锚点 + GlobeAnchor 后两者才对齐。

1. 第 2 步已连 Cesium ion。Cesium 面板里点 **Cesium World Terrain + Bing Maps Aerial**（或 Google 3D Tiles）
   加全球底图——会自动在关卡里放一个 **CesiumGeoreference**（插件 Actor）。
2. 在 **大纲（Outliner）** 选中 **CesiumGeoreference**，**细节（Details）** 里设原点为锚点：
   `Origin Latitude=22.5329830`、`Origin Longitude=113.9352358`、`Origin Height=21.49`（这些是插件属性，英文）。
3. **让模型贴到该位置**：
   - **放置Actor（Place Actors）** 面板搜 **空Actor（Empty Actor）** 拖进关卡，命名 `SceneRoot`，位置=(0,0,0)。
   - 在大纲里把 5 个 `EM_*` 模型 Actor 拖到 `SceneRoot` 下作**子物体**（保持它们 缩放=100）。
   - 选中 `SceneRoot` → 细节面板 **添加组件（Add Component）→ CesiumGlobeAnchor**（插件组件）→ 设
     `Latitude=22.5329830`、`Longitude=113.9352358`、`Height=21.49` → 整个模型吸附到深圳真实位置、+Z 朝天。
   - 朝向若不对：微调 `SceneRoot` 旋转的 **Yaw（偏航/Z轴）**（90°/180°）；镜像了就把某个水平轴缩放取负。
4. 也给 **RFRayTracer** 添加组件 `CesiumGlobeAnchor`：`Latitude/Longitude` 同锚点、`Height=133.3`
   （= 锚点 21.49m + TX 高 112m），TX 就悬在模型正上方。
5. **放置Actor** 面板搜 **CesiumSunSky** 拖进关卡补光照。
6. 现在既能看真实地球上的模型，又能在其上点 **Run Coverage / Run Ray Fan**。

---

## 坑表

| 现象                           | 处理                                                                                             |
| ---------------------------- | ---------------------------------------------------------------------------------------------- |
| 射线全“未命中”/材质全兜底               | 没勾 **Support UV From Hit Results**（项目设置搜 `Support UV`）；或网格没挡 Visibility（细节里 **碰撞预设=BlockAll**） |
| 模型看不见/极小                     | Actor **缩放（Scale）设 100**；按 F 聚焦；确认拖进了关卡不是只在内容浏览器                                               |
| `RFRayTracer` 搜不到            | C++ 没编过：先 新建C++类 生成再替换内容重编（Ctrl+Shift+B）                                                       |
| 编译报某 API 签名不符                | 已按 5.4.4 核对；若仍报，把报错行发我                                                                         |
| VS 报 `MSB307 退出码 6` 但已生成 dll | 外层误报，无视；进 UE 编辑器即可                                                                             |
| 覆盖点太稀/太密/卡                   | 调 `Grid N`（40 快 / 60 中 / 100 密慢）                                                               |
| 模型悬空/朝向错（第10步）               | `CesiumGlobeAnchor` 经纬高对上 `ue_anchor.json`；`SceneRoot` 调 Yaw                                   |
| glb 导不出（csv 被占用）             | `export_gltf_ue.py` 用 `--material_ply material_mesh.ply`（已内置，本例就是这么导的）                         |
| 链接报 PhysicalMaterial         | Build.cs 加 `"PhysicsCore"`                                                                     |
| VS 一堆 `NU1301` NuGet 报错      | 只生成 `LowAltRF` 项目（不是整个解决方案）；或删全局 NuGet.Config 里的坏本地源                                           |
| 加 Cesium 后模型极小/地形乱铺          | `CesiumGeoreference` 原点没设成锚点；且模型要加 `CesiumGlobeAnchor`（见第10步）                                  |
| 黄字"多个定向光竞争"/天空球警告            | 模板的 `DirectionalLight`/`SkyAtmosphere`/天空球 与 `CesiumSunSky` 冲突：删掉模板那几个，只留一套                    |
| 用了 Open World 模板多出 Landscape | RF 实验不需要，隐藏或删除 `Landscape`（你模型自带地面）                                                            |

## 与 Python / InSite 的对应

| 环节    | Python            | Wireless InSite        | UE                       |
| ----- | ----------------- | ---------------------- | ------------------------ |
| 几何    | tsdf 网格           | `scene.dae`            | `scene_ue.glb`           |
| 材质→电磁 | radio_params.yaml | `materials_insite.csv` | `ue_materials.csv` / 内置表 |
| 射线    | open3d raycast    | InSite SBR             | LineTrace 几何射线           |
| 尺度    | `--scale 33.03`   | `--geo_transform`      | glb 米制 ×100              |
| 产物    | pathloss_grid.npz | InSite 覆盖              | coverage_ue.csv          |

## 文件位置速查

| 用途             | 路径                                       |
| -------------- | ---------------------------------------- |
| C++ 追踪器        | `E:\lowalt3d\ue\RFRayTracer.{h,cpp}`     |
| glb + 材质表 + 锚点 | `E:\lowalt3d\outputs\scene02_pgsr\ue\`   |
| 导出脚本           | `E:\lowalt3d\scripts\export_gltf_ue.py`  |
| 对比脚本           | `E:\lowalt3d\scripts\compare_engines.py` |
