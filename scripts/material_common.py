"""
material_common.py · 当前毕设主线共用：材质类别 / ITU-R P.2040 电磁参数 / 调色板

从早期 material_em_fusion.py 抽出的共享部分，供主线脚本
(assign_material / simple_pathloss / infer_seg) import。
（流式体素融合类 MaterialEMVoxelMap 属于已暂缓的流式线，留在 archive/。）
"""
import os
import numpy as np

# 类别顺序（材质体系 v2：6 类，drop roof，ground_road→asphalt；三文件须一致）
_DEFAULT_CLASS_ORDER = [
    "background", "vegetation", "asphalt",
    "concrete", "glass", "metal",
]

# 各类别 ITU-R P.2040 近似电磁参数（缺 yaml 时的内置回退）
_DEFAULT_EM = {
    "background":  dict(eps_r=1.0,  sigma=0.0,    penetration_loss_db=0.0),
    "vegetation":  dict(eps_r=1.5,  sigma=0.05,   penetration_loss_db=6.0),
    "asphalt":     dict(eps_r=8.0,  sigma=0.02,   penetration_loss_db=30.0),
    "concrete":    dict(eps_r=5.31, sigma=0.0326, penetration_loss_db=12.0),
    "glass":       dict(eps_r=6.27, sigma=0.0043, penetration_loss_db=4.0),
    "metal":       dict(eps_r=1.0,  sigma=1e7,    penetration_loss_db=40.0),
}

# 可视化调色板（与 class_order 对应）
_PALETTE = np.array([
    [128, 128, 128],   # background  灰
    [ 60, 170,  70],   # vegetation  绿
    [ 90,  90,  90],   # asphalt     深灰
    [200, 200, 200],   # concrete    浅灰
    [ 80, 160, 230],   # glass       蓝
    [230, 180,  60],   # metal       橙黄
], dtype=np.uint8)


def load_radio_params(yaml_path=None):
    """读取电磁参数表，返回 (class_order, em_table)。yaml 缺失/失败时用内置默认表。"""
    if yaml_path is None or not os.path.isfile(yaml_path):
        return list(_DEFAULT_CLASS_ORDER), dict(_DEFAULT_EM)
    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return list(cfg.get("class_order", _DEFAULT_CLASS_ORDER)), dict(cfg.get("materials", _DEFAULT_EM))
    except Exception as e:
        print(f"[radio_params] 读取失败，用默认表: {e}")
        return list(_DEFAULT_CLASS_ORDER), dict(_DEFAULT_EM)
