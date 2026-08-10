# Requirements Document

## Introduction

本需求文档定义了一个文献收集与整理系统，用于收集和组织与 lowalt3d 项目中玻璃/材质识别优化相关的学术文献、技术报告和在线资源。该系统将帮助研究人员快速访问相关文献，支持毕业设计的理论支撑和方法改进。

## Glossary

- **Literature_Collector**: 文献收集系统，负责搜索、下载和整理文献
- **Citation_Manager**: 引用管理器，负责维护文献的元数据和引用信息
- **Category_Classifier**: 分类器，根据文献主题和方法将文献分类到不同文件夹
- **Metadata_Extractor**: 元数据提取器，从文献中提取标题、作者、年份、DOI等信息
- **lowalt3d_Pipeline**: 目标项目管线，包含COLMAP、PGSR、材质识别等阶段

## Requirements

### Requirement 1: 文献搜索与收集

**User Story:** 作为研究人员，我想要搜索和收集与玻璃/材质识别相关的文献，以便了解最新的研究进展和可行方案。

#### Acceptance Criteria

1. WHEN 用户提供关键词列表（如"facade segmentation", "window detection", "material classification"）THEN THE Literature_Collector SHALL 返回相关文献的列表
2. WHEN 用户指定文献来源（如arxiv.org、CVPR、ISPRS）THEN THE Literature_Collector SHALL 从指定来源搜索文献
3. WHEN 搜索到文献 THEN THE Literature_Collector SHALL 提取文献的DOI、标题、作者、年份、摘要等元数据
4. WHEN 文献可公开获取 THEN THE Literature_Collector SHALL 下载PDF文件并保存到指定目录
5. WHEN 文献无法直接下载 THEN THE Literature_Collector SHALL 保存文献的URL和元数据以供手动下载

### Requirement 2: 文献分类与组织

**User Story:** 作为研究人员，我想要将文献按照主题和方法分类，以便快速找到特定类型的解决方案。

#### Acceptance Criteria

1. WHEN 文献被收集 THEN THE Category_Classifier SHALL 根据文献内容将其分类到相应的主题文件夹
2. THE Category_Classifier SHALL 支持以下分类类别：
   - 立面分割与窗检测（Facade Segmentation & Window Detection）
   - 零样本学习与开放词汇（Zero-shot & Open Vocabulary）
   - 3D高斯材质学习（3D Gaussian Material Learning）
   - 合成数据与数据增强（Synthetic Data & Augmentation）
   - 航拍图像处理（UAV Image Processing）
   - 玻璃表面检测（Glass Surface Detection）
   - 多视角一致性（Multi-view Consistency）
3. WHEN 文献属于多个类别 THEN THE Category_Classifier SHALL 在多个文件夹中创建符号链接或副本
4. WHEN 文献被分类 THEN THE System SHALL 在分类文件夹中生成README文件，说明该类别的研究方向和关键文献

### Requirement 3: 元数据管理与索引

**User Story:** 作为研究人员，我想要维护所有文献的结构化元数据，以便进行搜索、引用和分析。

#### Acceptance Criteria

1. WHEN 文献被收集 THEN THE Metadata_Extractor SHALL 创建包含以下字段的元数据记录：
   - 标题（title）
   - 作者列表（authors）
   - 发表年份（year）
   - 会议/期刊名称（venue）
   - DOI或arXiv ID
   - 摘要（abstract）
   - 关键词（keywords）
   - PDF文件路径（pdf_path）
   - URL
   - 相关性评分（relevance_score）
2. WHEN 元数据被提取 THEN THE System SHALL 将元数据保存为JSON或CSV格式
3. WHEN 用户查询文献 THEN THE System SHALL 支持按标题、作者、年份、关键词等字段搜索
4. WHEN 元数据被更新 THEN THE System SHALL 保持数据一致性并记录修改历史

### Requirement 4: BibTeX引用生成

**User Story:** 作为研究人员，我想要为收集的文献自动生成BibTeX引用，以便在论文写作时引用。

#### Acceptance Criteria

1. WHEN 文献元数据可用 THEN THE Citation_Manager SHALL 生成标准格式的BibTeX条目
2. WHEN 用户请求导出引用 THEN THE System SHALL 生成包含所有文献的.bib文件
3. WHEN BibTeX条目生成 THEN THE System SHALL 确保引用键（cite key）唯一且符合命名规范（如"author2024title"）
4. WHEN 文献更新 THEN THE System SHALL 自动更新对应的BibTeX条目

### Requirement 5: 文献摘要与相关性标注

**User Story:** 作为研究人员，我想要为每篇文献添加中文摘要和与lowalt3d项目的相关性说明，以便快速评估文献价值。

#### Acceptance Criteria

1. WHEN 文献被收集 THEN THE System SHALL 为每篇文献创建一个Markdown注释文件
2. THE Markdown注释文件 SHALL 包含以下内容：
   - 文献基本信息（标题、作者、年份、来源）
   - 中文摘要或简介
   - 核心方法说明
   - 与lowalt3d的相关性（如何应用到立面正射、窗检测、材质学习等）
   - 关键图表或结果摘录
   - 实现难度评估
3. WHEN 用户浏览文献库 THEN THE System SHALL 提供README索引文件，列出所有文献及其相关性评分
4. WHEN 文献注释更新 THEN THE System SHALL 保持注释文件与元数据的同步

### Requirement 6: 优先文献列表生成

**User Story:** 作为研究人员，我想要根据相关性和实现可行性生成优先阅读列表，以便高效地推进研究。

#### Acceptance Criteria

1. WHEN 所有文献被收集和分类 THEN THE System SHALL 根据以下标准计算相关性评分：
   - 与lowalt3d管线的契合度（0-5分）
   - 方法的可实现性（0-5分）
   - 文献的引用次数和影响力（0-5分）
   - 发表时间新近度（0-5分）
2. WHEN 评分计算完成 THEN THE System SHALL 生成按评分排序的优先文献列表
3. THE 优先文献列表 SHALL 包含Top 10-20篇文献的详细信息
4. WHEN 用户查看优先列表 THEN THE System SHALL 提供每篇文献的快速访问链接和实施建议

### Requirement 7: 文献库维护与更新

**User Story:** 作为研究人员，我想要定期更新文献库，添加新发表的相关文献，以便保持文献库的时效性。

#### Acceptance Criteria

1. WHEN 用户触发更新操作 THEN THE System SHALL 搜索自上次更新以来的新文献
2. WHEN 发现新文献 THEN THE System SHALL 自动添加到文献库并进行分类
3. WHEN 文献库更新 THEN THE System SHALL 生成更新日志，记录新增文献数量和主要更新内容
4. WHEN 重复文献被检测到 THEN THE System SHALL 避免重复添加并合并元数据
