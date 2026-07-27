# 方向 A 执行手册：占位材质 → 训练分割模型

> 目标：把 `assign_material.py` 里的**占位启发式**换成**训练好的 SegFormer 材质分割**，让材质从"颜色猜"变"模型识别"。
> 类别体系（折中统一，`configs/classes.txt`）：`background / vegetation / ground_road / concrete / roof / glass / metal`
> 环境：`conda activate E:\condaenvs\emmap`

## 流程总览

```
① LabelMe 标注 22 张 (你手动, 1~2h)
② labelme2mask.py  → 标签 PNG
③ train_seg.py     → 微调 SegFormer
④ infer_seg.py     → 全部 87 张的材质掩码
⑤ assign_material.py --masks_dir  → 面片材质(真模型)
⑥ simple_pathloss.py              → 新路损图
```

已就绪：`scripts/{labelme2mask,train_seg,infer_seg}.py`、`assign_material.py`(已加 `--masks_dir`)、`configs/classes.txt`、`configs/radio_params.yaml`(7类)、`data/scene01/material/annot_images/`(22 张待标)。

---

## Step A1 · LabelMe 标注（你手动，唯一的人工步骤）

**目的**：给模型提供"这块是什么材质"的监督信号。

**装 + 打开**：
```cmd
pip install labelme
labelme data\scene01\material\annot_images --labels configs\classes.txt --nodata
```

**怎么标**：
- 左侧 **Create Polygons**，沿材质区域画多边形，弹窗选类别（7 类之一）。
- **不用像素级精确**，圈住主要区域即可：树/草→`vegetation`、道路/裸地→`ground_road`、水泥硬化面/墙→`concrete`、屋顶→`roof`、玻璃窗→`glass`、金属面→`metal`。
- **不确定的、天空、人车杂物就不画**（默认算 background）。
- 每张存成同名 `.json`（自动，就在图片旁）。约 5~10 分钟/张。

**记录要点**：本公园场景以 vegetation/ground_road/concrete 为主，`glass/metal/roof` 可能很少或没有——**正常**，模型学到啥算啥；材质多样性留给日后的建筑场景。

**成功标志**：`annot_images/` 里出现 22 个 `.json`。

---

## Step A2 · 标注转掩码

```cmd
python scripts\labelme2mask.py --json_dir data\scene01\material\annot_images --classes configs\classes.txt --out_dir data\scene01\material\masks
```
**成功标志**：`masks/` 下生成 22 个 `.png`（像素值=类别 id）。

---

## Step A3 · 微调 SegFormer

```cmd
pip install transformers
set HF_ENDPOINT=https://hf-mirror.com
python scripts\train_seg.py --images data\scene01\material\annot_images --masks data\scene01\material\masks --classes configs\classes.txt --out outputs\seg_model --epochs 80
```
- `HF_ENDPOINT` 是国内下预训练权重的镜像（首次会下 SegFormer-B0 权重）。
- 打印 loss 下降；**成功标志**：`outputs\seg_model\` 生成模型文件。约 10~30 分钟。

---

## Step A4 · 推理全部 87 张 → 材质掩码

```cmd
python scripts\infer_seg.py --model outputs\seg_model --images data\scene01\images --out data\scene01\material\masks_all --color
```
**成功标志**：`masks_all\` 下 87 个 `.png`；`masks_all_color\` 是上色预览——**先打开看看分割质量**（好不好、要不要多标几张重训）。

---

## Step A5 · 用真掩码重跑材质 + 路损（对比占位版）

```cmd
python scripts\assign_material.py --mesh outputs\scene01_pgsr\mesh\tsdf_fusion_post.ply --cameras outputs\scene01_pgsr\cameras.json --images data\scene01\images --out_dir outputs\scene01_pgsr\material_seg --masks_dir data\scene01\material\masks_all

python scripts\simple_pathloss.py --mesh outputs\scene01_pgsr\mesh\tsdf_fusion_post.ply --materials outputs\scene01_pgsr\material_seg\face_materials.csv --out_dir outputs\scene01_pgsr\pathloss_seg --grid 60
```
（输出到 `material_seg\` / `pathloss_seg\`，与占位版 `material\` / `pathloss\` 并存，**方便对比**。）

**成功标志**：`material_seg\material_mesh.ply` 材质比占位版更准；新路损图。

---

## 迭代提质（可选）

分割不理想时：多标几张（尤其漏标的类）→ 重跑 A2/A3/A4；或把 `--backbone` 换成 `nvidia/segformer-b2-finetuned-ade-512-512`（更强，更慢）。
