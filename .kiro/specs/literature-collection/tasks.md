# Implementation Plan: Literature Collection System

## Overview

本实现计划将文献收集系统分解为可执行的离散任务。每个任务专注于特定功能模块的实现，并包含相应的测试。任务按依赖关系组织，确保增量构建和早期验证。

## Tasks

- [ ] 1. 设置项目结构和核心数据模型
  - 创建`scripts/literature/`目录作为新模块
  - 定义数据类: `PaperMetadata`, `Paper`, `Category`, `CollectionResult`
  - 实现数据模型的序列化/反序列化（JSON）
  - 创建配置文件`literature_config.yaml`（包含API keys、分类定义等）
  - _Requirements: 3.1, 3.2_

- [ ] 1.1 编写数据模型的单元测试
  - 测试`PaperMetadata`的各字段验证
  - 测试JSON序列化边界情况（None值、特殊字符）
  - _Requirements: 3.1, 3.2_

- [ ] 1.2 编写Property test: 元数据序列化往返一致性
  - **Property 3: 元数据序列化往返一致性**
  - **Validates: Requirements 3.2**
  - 使用`hypothesis`生成随机`PaperMetadata`对象
  - 验证JSON序列化后反序列化的一致性
  - _Requirements: 3.2_

- [ ] 2. 实现Metadata Extractor模块
  - [ ] 2.1 实现arXiv API集成
    - 使用`arxiv` Python包搜索和获取元数据
    - 处理API限流和错误重试
    - 提取标题、作者、摘要、arXiv ID
    - _Requirements: 1.1, 1.2, 1.3_

  - [ ] 2.2 实现Semantic Scholar API集成
    - 获取引用次数和影响力指标
    - 增强已有元数据（通过DOI或标题查询）
    - _Requirements: 1.3, 6.1_

  - [ ] 2.3 实现CrossRef API集成
    - 通过DOI获取完整元数据
    - 处理DOI解析失败的情况
    - _Requirements: 1.3_

  - [ ] 2.4 编写单元测试：API响应解析
    - Mock API响应测试元数据提取
    - 测试缺失字段的默认值处理
    - 测试特殊字符处理（作者名中的üöä）
    - _Requirements: 1.3_

  - [ ] 2.5 编写Property test: 搜索结果格式一致性
    - **Property 1: 搜索结果格式一致性**
    - **Validates: Requirements 1.1, 1.3**
    - 验证所有搜索结果都包含必需字段
    - _Requirements: 1.1, 1.3_

- [ ] 3. 实现Literature Collector模块
  - [ ] 3.1 实现文献搜索功能
    - 整合多个API源的搜索结果
    - 实现关键词组合查询
    - 支持年份范围和来源过滤
    - _Requirements: 1.1, 1.2_

  - [ ] 3.2 实现PDF下载功能
    - 支持从arXiv、直接URL下载PDF
    - 实现异步并发下载（使用`aiohttp`，限制5-10并发）
    - 处理下载失败和保存元数据
    - _Requirements: 1.4, 1.5_

  - [ ] 3.3 实现批量收集流程
    - 读取查询配置文件
    - 执行搜索→下载→保存的完整流程
    - 生成`CollectionResult`报告
    - _Requirements: 1.1, 1.4_

  - [ ] 3.4 编写单元测试：下载错误处理
    - 测试网络超时、404错误的处理
    - 验证失败记录到`failed_downloads.log`
    - _Requirements: 1.5_

  - [ ] 3.5 编写Property test: 来源过滤正确性
    - **Property 2: 来源过滤正确性**
    - **Validates: Requirements 1.2**
    - 验证指定来源参数后，结果只包含该来源
    - _Requirements: 1.2_

- [ ] 4. 检查点 - 验证文献收集功能
  - 运行收集器搜索5-10篇测试文献
  - 验证元数据完整性和PDF下载
  - 确保所有测试通过，询问用户是否继续

- [ ] 5. 实现Category Classifier模块
  - [ ] 5.1 实现基于关键词的分类逻辑
    - 加载`CATEGORY_KEYWORDS`配置
    - 计算文献标题/摘要与各类别的匹配度
    - 支持多分类（阈值可配置）
    - _Requirements: 2.1, 2.2_

  - [ ] 5.2 实现文件组织功能
    - 创建分类文件夹结构
    - 复制或链接PDF到多个分类文件夹
    - 同步复制Markdown注释文件
    - _Requirements: 2.3_

  - [ ] 5.3 实现分类README生成
    - 为每个类别文件夹生成README
    - 列出该类别的所有文献和简介
    - 包含类别的研究方向说明
    - _Requirements: 2.4, 5.3_

  - [ ] 5.4 编写单元测试：关键词匹配
    - 测试不同关键词密度的分类结果
    - 测试多分类阈值逻辑
    - _Requirements: 2.1_

  - [ ] 5.5 编写Property test: 分类器返回有效类别
    - **Property 4: 分类器返回有效类别**
    - **Validates: Requirements 2.1, 2.2**
    - 验证所有返回类别都在预定义集合中
    - _Requirements: 2.1, 2.2_

  - [ ] 5.6 编写Property test: 多分类文件同步
    - **Property 5: 多分类文件同步**
    - **Validates: Requirements 2.3**
    - 验证N个分类对应N个文件夹中都有该文献
    - _Requirements: 2.3_

- [ ] 6. 实现文献注释和摘要生成
  - [ ] 6.1 设计Markdown注释模板
    - 定义注释文件的结构和必需section
    - 实现模板填充逻辑
    - _Requirements: 5.1, 5.2_

  - [ ] 6.2 实现基本信息自动填充
    - 从元数据自动生成"基本信息"section
    - 生成BibTeX引用并嵌入
    - _Requirements: 5.2_

  - [ ] 6.3 实现与lowalt3d相关性模板生成
    - 根据分类自动生成相关性说明框架
    - 预填充可能的应用场景
    - 标记为"待人工补充"
    - _Requirements: 5.2_

  - [ ] 6.4 编写单元测试：Markdown生成
    - 测试模板渲染的正确性
    - 测试特殊字符转义
    - _Requirements: 5.2_

  - [ ] 6.5 编写Property test: 注释文件与元数据同步
    - **Property 11: 注释文件与元数据同步**
    - **Validates: Requirements 5.4**
    - 验证Markdown中的基本信息与JSON元数据一致
    - _Requirements: 5.4_

- [ ] 7. 实现Citation Manager模块
  - [ ] 7.1 实现cite key生成逻辑
    - 格式: `{first_author_lastname}{year}{first_title_word}`
    - 处理特殊字符（使用`unidecode`）
    - 实现去重（添加a/b/c后缀）
    - _Requirements: 4.1, 4.3_

  - [ ] 7.2 实现BibTeX条目生成
    - 根据元数据生成标准BibTeX格式
    - 处理不同文献类型（article, inproceedings, misc）
    - _Requirements: 4.1_

  - [ ] 7.3 实现批量导出bibliography.bib
    - 遍历所有元数据生成.bib文件
    - 维护cite key的一致性
    - _Requirements: 4.2_

  - [ ] 7.4 编写单元测试：cite key冲突处理
    - 测试同名作者同年的后缀添加
    - 测试特殊字符转换
    - _Requirements: 4.3_

  - [ ] 7.5 编写Property test: BibTeX cite key唯一性
    - **Property 6: BibTeX cite key唯一性**
    - **Validates: Requirements 4.3**
    - 验证生成的所有cite key互不相同
    - _Requirements: 4.3_

- [ ] 8. 检查点 - 验证分类和引用功能
  - 对已收集的文献进行分类
  - 生成README和BibTeX文件
  - 验证文件组织结构正确
  - 确保所有测试通过，询问用户是否继续

- [ ] 9. 实现Priority Scorer模块
  - [ ] 9.1 实现相关性评分算法
    - 实现管线契合度评分（基于分类和关键词）
    - 实现可实现性评分（检测CUDA、硬件等关键词）
    - 实现影响力评分（归一化引用次数）
    - 实现新近度评分（基于年份）
    - _Requirements: 6.1_

  - [ ] 9.2 实现优先列表生成
    - 根据总评分排序文献
    - 生成Top N列表（默认20）
    - 为每篇文献生成实施建议
    - _Requirements: 6.2, 6.3, 6.4_

  - [ ] 9.3 生成PRIORITY_LIST.md文件
    - 使用Markdown表格展示优先列表
    - 包含评分、分类、关键信息
    - 提供快速访问链接
    - _Requirements: 6.4_

  - [ ] 9.4 编写单元测试：评分计算
    - 测试各子分数的计算逻辑
    - 测试边界情况（零引用、未来年份）
    - _Requirements: 6.1_

  - [ ] 9.5 编写Property test: 评分范围有效性
    - **Property 7: 评分范围有效性**
    - **Validates: Requirements 6.1**
    - 验证所有评分在0-20范围内
    - _Requirements: 6.1_

  - [ ] 9.6 编写Property test: 优先列表排序单调性
    - **Property 8: 优先列表排序单调性**
    - **Validates: Requirements 6.2**
    - 验证列表按评分降序排列
    - _Requirements: 6.2_

- [ ] 10. 实现更新和维护功能
  - [ ] 10.1 实现增量更新逻辑
    - 读取上次更新时间戳
    - 只搜索新发表的文献
    - 合并到现有文献库
    - _Requirements: 7.1, 7.2_

  - [ ] 10.2 实现重复检测和合并
    - 基于DOI精确匹配
    - 基于标题归一化匹配（去空格、小写、去标点）
    - 合并元数据（保留更完整的）
    - _Requirements: 7.4_

  - [ ] 10.3 实现更新日志生成
    - 记录每次更新的时间、新增数量
    - 列出新增文献的标题和分类
    - 追加到`update_log.md`
    - _Requirements: 7.3_

  - [ ] 10.4 编写单元测试：时间戳过滤
    - 测试不同时间范围的过滤逻辑
    - _Requirements: 7.1_

  - [ ] 10.5 编写Property test: 更新操作时间过滤
    - **Property 9: 更新操作时间过滤**
    - **Validates: Requirements 7.1**
    - 验证返回的文献都晚于上次更新时间
    - _Requirements: 7.1_

  - [ ] 10.6 编写Property test: 去重逻辑正确性
    - **Property 10: 去重逻辑正确性**
    - **Validates: Requirements 7.4**
    - 验证基于DOI/标题的去重保证唯一性
    - _Requirements: 7.4_

- [ ] 11. 实现命令行界面
  - [ ] 11.1 创建主CLI脚本`literature_manager.py`
    - 使用`argparse`定义子命令
    - 子命令: search, classify, export, update, score
    - 全局参数: --config, --verbose, --output-dir
    - _Requirements: 所有_

  - [ ] 11.2 实现`search`子命令
    - 参数: --keywords, --sources, --year-range, --max-results
    - 调用Literature Collector
    - 显示搜索进度和结果摘要
    - _Requirements: 1.1, 1.2_

  - [ ] 11.3 实现`classify`子命令
    - 参数: --input (metadata JSON), --threshold
    - 调用Category Classifier
    - 生成分类报告
    - _Requirements: 2.1_

  - [ ] 11.4 实现`export`子命令
    - 参数: --format (bibtex/csv/json), --output
    - 调用Citation Manager或导出元数据
    - _Requirements: 4.2_

  - [ ] 11.5 实现`update`子命令
    - 参数: --since (时间戳)
    - 执行增量更新流程
    - 生成更新报告
    - _Requirements: 7.1, 7.2_

  - [ ] 11.6 实现`score`子命令
    - 参数: --top-n
    - 调用Priority Scorer
    - 生成和显示优先列表
    - _Requirements: 6.1, 6.2_

- [ ] 12. 创建初始文献收集配置
  - [ ] 12.1 定义玻璃/材质识别的关键词集合
    - 根据调研结果整理搜索关键词
    - 包含: facade segmentation, window detection, 3D Gaussian, material classification等
    - _Requirements: 1.1_

  - [ ] 12.2 创建种子文献列表
    - 根据调研结果手动添加已知重要文献的DOI/arXiv ID
    - 包含CVPR 2021 Glass Detection, ISPRS 2026 Window Detection等
    - _Requirements: 1.1_

  - [ ] 12.3 配置API credentials
    - 申请Semantic Scholar API key（可选，无key有限流）
    - 配置到`literature_config.yaml`
    - _Requirements: 1.2_

- [ ] 13. 执行初次完整文献收集
  - [ ] 13.1 运行搜索命令收集文献
    - 执行: `python scripts/literature/literature_manager.py search --keywords "facade segmentation" "window detection" ...`
    - 预期收集30-50篇文献
    - _Requirements: 1.1, 1.4_

  - [ ] 13.2 分类和组织文献
    - 执行: `... classify`
    - 验证7个分类文件夹都有文献
    - _Requirements: 2.1, 2.3_

  - [ ] 13.3 生成优先列表和BibTeX
    - 执行: `... score` 和 `... export --format bibtex`
    - 审查Top 20列表是否合理
    - _Requirements: 4.2, 6.2_

  - [ ] 13.4 人工补充注释文件
    - 选择Top 5-10篇文献
    - 填充中文摘要和相关性说明
    - 完善实施建议
    - _Requirements: 5.2_

- [ ] 14. 最终检查点和文档生成
  - [ ] 14.1 生成主README索引
    - 创建`docs/literature/README.md`
    - 列出所有文献、分类统计、使用指南
    - _Requirements: 5.3_

  - [ ] 14.2 验证所有Property tests通过
    - 运行完整测试套件: `pytest tests/literature/ --hypothesis-seed=random`
    - 确保至少100次迭代
    - 修复发现的边界情况bug
    - _Requirements: 所有_

  - [ ] 14.3 编写用户使用文档
    - 创建`docs/literature/USAGE.md`
    - 包含安装依赖、配置、常用命令示例
    - 包含常见问题FAQ
    - _Requirements: 所有_

  - [ ] 14.4 与用户最终确认
    - 演示文献库的浏览和查询
    - 确认优先列表符合预期
    - 询问是否需要调整评分权重或添加更多文献

## Notes

- 所有测试任务都是必做的，确保完整的测试覆盖
- 每个任务引用具体需求编号以保证追溯性
- 检查点任务（4, 8, 14）确保增量验证
- Property tests每个至少100次迭代，使用`hypothesis`库
- 命令行工具放在`scripts/literature/literature_manager.py`，遵循项目风格
- 文献库输出到`docs/literature/`，与现有文档结构一致
