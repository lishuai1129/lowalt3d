# Literature Collection System - Spec Summary

## 概述

这是一个为 lowalt3d 项目设计的文献收集与管理系统的完整规格说明。该系统旨在帮助研究人员系统化地收集、分类和管理与玻璃/材质识别相关的学术文献。

## 文档结构

本规格包含三个核心文档：

### 1. requirements.md - 需求文档
定义了系统的7个主要功能需求：
- **Requirement 1**: 文献搜索与收集
- **Requirement 2**: 文献分类与组织（7个主题类别）
- **Requirement 3**: 元数据管理与索引
- **Requirement 4**: BibTeX引用生成
- **Requirement 5**: 文献摘要与相关性标注
- **Requirement 6**: 优先文献列表生成
- **Requirement 7**: 文献库维护与更新

每个需求包含详细的验收标准（Acceptance Criteria），使用EARS模式编写。

### 2. design.md - 设计文档
包含系统的详细设计：
- **架构图**: 展示5个核心模块的交互关系
- **组件接口**: 详细的Python接口定义
- **数据模型**: PaperMetadata, Paper, Category等
- **文件组织**: `docs/literature/`下的目录结构
- **12个正确性属性**: 用于property-based testing
- **错误处理策略**: 网络失败、API限流、文件系统错误等
- **测试策略**: 单元测试 + 属性测试 + 集成测试

关键设计决策：
- 使用arXiv, Semantic Scholar, CrossRef三个API源
- 7个预定义分类类别（立面分割、零样本、3DGS材质等）
- 评分系统：管线契合度(5分) + 可实现性(5分) + 影响力(5分) + 新近度(5分)
- 支持增量更新和去重

### 3. tasks.md - 实现任务
将设计分解为14个主要任务，共60+个子任务：
1. 项目结构和数据模型
2. Metadata Extractor实现
3. Literature Collector实现
4. **检查点1** - 验证收集功能
5. Category Classifier实现
6. 注释和摘要生成
7. Citation Manager实现
8. **检查点2** - 验证分类和引用
9. Priority Scorer实现
10. 更新和维护功能
11. 命令行界面
12. 初始配置
13. 首次文献收集
14. **最终检查** - 文档和测试

所有测试都是必做任务，包括12个property tests和多个单元测试。

## 核心特性

### 多源文献搜索
- arXiv: 预印本论文
- Semantic Scholar: 引用和影响力数据
- CrossRef: DOI元数据
- Web Scraping: CVPR、ISPRS等会议

### 智能分类系统
7个主题分类，基于关键词自动分类：
```python
CATEGORIES = {
    "facade-seg": "立面分割与窗检测",
    "zero-shot": "零样本学习与开放词汇",
    "3dgs-material": "3D高斯材质学习",
    "synthetic-data": "合成数据与数据增强",
    "uav-processing": "航拍图像处理",
    "glass-detection": "玻璃表面检测",
    "multi-view": "多视角一致性"
}
```

### 优先级评分
自动计算每篇文献的相关性评分（0-20分）：
- 与lowalt3d管线的契合度: 0-5分
- 方法的可实现性: 0-5分
- 引用次数和影响力: 0-5分
- 发表时间新近度: 0-5分

### 自动化工作流
1. 搜索 → 2. 下载PDF → 3. 提取元数据 → 4. 分类 → 5. 生成注释 → 6. 计算评分 → 7. 导出BibTeX

## 预期输出

执行完整流程后，`docs/literature/`将包含：

```
docs/literature/
├── README.md                    # 主索引
├── PRIORITY_LIST.md             # Top 20优先文献
├── metadata.json                # 结构化元数据
├── bibliography.bib             # BibTeX引用
├── by-category/
│   ├── facade-seg/
│   │   ├── README.md
│   │   ├── paper1.pdf
│   │   ├── paper1.md          # 中文注释
│   │   └── ...
│   ├── zero-shot/
│   ├── 3dgs-material/
│   └── ...
└── by-year/
    ├── 2026/
    ├── 2025/
    └── ...
```

## 技术栈

- **语言**: Python 3.10+
- **核心库**:
  - `arxiv` - arXiv API
  - `semanticscholar` - Semantic Scholar
  - `requests` / `aiohttp` - HTTP请求
  - `bibtexparser` - BibTeX处理
  - `hypothesis` - Property-based testing
  - `jsonschema` - 数据验证
- **测试**: pytest + hypothesis (100+ iterations per property)

## 下一步行动

### 如果你想开始实现：
1. 打开 `tasks.md`
2. 从任务1开始："设置项目结构和核心数据模型"
3. 点击IDE中的"Start task"按钮

### 如果你想直接获取文献（不实现系统）：
我可以帮你：
1. 直接使用Web搜索收集文献URL
2. 手动创建分类文件夹
3. 生成初始的文献列表和BibTeX文件

请告诉我你希望的方向！

## 与调研结果的对应

该系统设计直接支持你调研中识别的所有关键方案：

| 调研方案 | 系统中的分类 | 预期文献数 |
|---------|-------------|-----------|
| 立面正射图 + DINO+SAM2 | facade-seg, zero-shot | 8-12篇 |
| SegFormer立面微调 | facade-seg, synthetic-data | 5-8篇 |
| 3DGS噪声标签鲁棒 | 3dgs-material, multi-view | 6-10篇 |
| 合成数据增强 | synthetic-data, uav-processing | 4-6篇 |
| 玻璃反射检测 | glass-detection | 3-5篇 |

总计预期收集：30-50篇高相关性文献
