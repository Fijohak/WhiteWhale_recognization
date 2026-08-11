<!-- ppt-master-schema: design-spec/v1 -->
# 中华白海豚个体识别 — 前期准备与后续方向 - Design Spec

## I. Project Information

| Item | Value |
| --- | --- |
| Project Name | 中华白海豚个体识别 — 前期准备与后续方向周报 |
| Canvas Format | PPT 16:9 (1280×720) |
| Page Count | 12 |
| Target Audience | 项目负责人 / 导师及团队成员（本周汇报），对本项目背景已有了解，关心进展、数据事实与下一步计划 |
| Communication Intent | 汇报前期准备（数据盘点与索引）完成情况，并基于数据现状明确技术方向（跨物种迁移学习 + 散图分拣利用）与行动计划，获得对下一步工作的认可。以报告与对齐为主。 |
| Desired Audience Outcome | 听众掌握数据规模与标签语义（2863 张、43 个体 199 张 labeled、207 张散图 loose_known），理解数据特点如何决定两大技术方向（迁移学习四组对照实验、散图分拣即标签扩充），明确下一步行动（散图分拣 / 外部数据下载 / Pilot Set） |
| Core Message / Ask / Action | 数据盘点与索引已完成并写入文档；基于数据现状确定两大技术方向：① 跨物种迁移学习（MegaDescriptor / Happywhale / 多源鲸豚预训练 + 本地微调，四组对照实验验证）② 散图利用（分拣扩充标签 + candidate-label 弱监督）；下一步启动散图分拣、外部数据下载与 Pilot Set 基线 |
| Delivery Context | 主讲汇报，约 10-15 分钟，现场或线上投屏；投影用 PPT |
| Artifact Afterlife | 会后留存作为项目进度存档与后续阶段汇报的参照 |
| Reading Mode | balanced |
| Content Strategy | balanced default |
| Design Style | 学术编辑风（editorial）：深蓝主色 + 金色装饰线 + 红色强调，参照 AI-Internship 周5汇报样式 |
| Formula Policy | text-only |
| AI Image Acquisition Path | not applicable |
| Generation Mode | continuous |
| Spec Refinement | disabled |
| Created Date | 2026-08-07 |

## II. Canvas Specification

| Property | Value |
| --- | --- |
| Format | PPT 16:9 |
| Dimensions | 1280 × 720 |
| viewBox | 0 0 1280 720 |
| Margins | 60px 安全边距 |
| Content Area | x: 60–1220, y: 50–670 |

## III. Visual Theme

### Theme Style

- **Mode**: briefing
- **Visual style**: editorial
- **Theme**: 学术编辑风，深蓝主色 + 金色装饰线，红色强调关键数字
- **Tone**: 专业、清晰、信息密度中高

### Color Scheme

| Role | HEX | Purpose |
| --- | --- | --- |
| Background | #FFFFFF | 内容页背景 |
| Secondary background | #F2F5F9 | 表格交替行、信息卡片底 |
| Primary | #1E3A5F | 封面背景、标题条、表头 |
| Accent | #E8B339 | 装饰线、页码、关键分隔 |
| Secondary accent | #BF0000 | 关键数据、强调结论、待办标记 |
| Body text | #2B3440 | 正文 |

## IV. Typography System

### Font Plan

| Role | Chinese | English | Fallback tail |
| --- | --- | --- | --- |
| Title | 微软雅黑 | Microsoft YaHei | sans-serif |
| Body | 微软雅黑 | Microsoft YaHei | sans-serif |
| Data | 微软雅黑 | Microsoft YaHei | sans-serif |
| Annotation | 微软雅黑 | Microsoft YaHei | sans-serif |
| Card Title | 微软雅黑 | Microsoft YaHei | sans-serif |
| Card Body | 微软雅黑 | Microsoft YaHei | sans-serif |

- **Title stack**: 微软雅黑, Microsoft YaHei, sans-serif
- **Body stack**: 微软雅黑, Microsoft YaHei, sans-serif
- **Data stack**: 微软雅黑, Microsoft YaHei, sans-serif
- **Annotation stack**: 微软雅黑, Microsoft YaHei, sans-serif
- **Card Title stack**: 微软雅黑, Microsoft YaHei, sans-serif
- **Card Body stack**: 微软雅黑, Microsoft YaHei, sans-serif

### Font Size Hierarchy

| Purpose | Anchor Size (px) |
| --- | ---: |
| KPI Value | 42 |
| Body | 22 |
| Title | 34 |
| Subtitle | 26 |
| Annotation | 14 |
| Card Title | 19 |
| Card Body | 17 |

## V. Layout Principles

### Page Structure

- **Header area**: 顶部深蓝标题条（高 90px），左侧金色竖条装饰
- **Content area**: 标题条下方至页脚上方
- **Footer area**: 左下角章节名 + 右下角页码

### Spacing Specification

| Element | Current Project |
| --- | --- |
| Safe margin | 60px 左右 / 50px 上下 |
| Content block gap | 20px |
| Icon-text gap | 12px |

## VI. Icon Usage Specification

- **Primary bundled library**: none

| Purpose | Icon Path | Page |
| --- | --- | --- |

## VII. Visualization Reference List

（无目录引用）

## VIII. Image Resource List

| Filename | Dimensions | Ratio | Purpose | Type | Layout pattern | Crop Policy | Acquire Via | Status | Reference | text_policy | page_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| label_status.png | 1400x900 | 1.56 | 标签状态分布（labeled 199 / loose_known 207 / ignored 2457） | Chart | #6 bottom-band image + top title + middle text | no-crop | user | Existing | 标签状态分布真实数据 | keep | content |
| session_dist.png | 1200x800 | 1.50 | 两个调查批次图片量分布（01：818 / 03：2045） | Chart | #7 top-and-bottom symmetric split | no-crop | user | Existing | 调查批次分布真实数据 | keep | content |
| quality_band.png | 1400x900 | 1.56 | 评分区间分布（unknown 2018 / 60_69 401 / 70_79 223 / 80_and_above 183 / 50_59 33 / below_50 5） | Chart | #6 bottom-band image + top title + middle text | no-crop | user | Existing | 评分区间分布真实数据 | keep | content |

## IX. Content Outline

> 叙事主线：数据现状 → 数据特点 → 技术方向（迁移学习 + 散图利用）→ 行动计划。
> 每个数据特点对应一个具体应对方案，不罗列约束。

### Part 1: 本周进展

#### Slide 01 - 封面

- **Audience move**: 不知晓项目 → 明确汇报主题与定位
- **Layout**: 深蓝全幅背景 + 白色大标题 + 金色装饰线 + 副标题信息
- **Title**: 中华白海豚个体识别 — 前期准备与后续方向
- **Core message**: 数据盘点完成，基于数据现状明确迁移学习与散图利用两大技术方向
- **Content**: 大标题「中华白海豚个体识别」;副标题「前期准备与后续方向 · 数据盘点 → 技术路线」;项目定位：利用背鳍特征做个体识别，当前处于数据准备阶段;日期 2026-08-07
- **Cover impact**: 深蓝底白字 + 金色装饰线 + 项目定位一句话

#### Slide 02 - 本周进展总览

- **Audience move**: 了解本周完成的核心事项 → 明确四件事全部落地
- **Layout**: 四行横向进度条（编号 + 标题 + 说明），突出第 04 项
- **Title**: 本周进展 — 盘点索引与方案成文
- **Core message**: 数据盘点、索引、文档与迁移学习方案四件事全部完成
- **Content**: 01 数据盘点 — 全量扫描 I:\01 与 I:\03，建立 2863 张图片索引，SHA-256 精确重复检测;02 标签体系 — 解析评分区间与个体分组语义，labeled / loose_known / ignored 三级标签，含 candidate-label 弱监督;03 文档整合 — README / TASKS 全面更新，数据语义、假设清单、待办事项全部落档;04 迁移方案成文 — 跨物种迁移学习与少样本适配方案落档，明确四组对照实验（docs/cetacean_reid_transfer_learning_plan.md）;汇报主线：基于数据现状定方向

### Part 2: 数据现状

#### Slide 03 - 数据规模总览

- **Audience move**: 了解数据总体规模 → 掌握核心数字
- **Layout**: 四个 KPI 数字卡（左上大数字 + 说明），下方两列分布
- **Title**: 数据规模 — 2863 张野外调查照片
- **Core message**: 数据规模清晰：总照片 2863 张，可用数据 406 张（199 labeled + 207 散图）
- **Content**: KPI：总图片 2863 张（01：818，03：2045）;已分组个体 43 个 / 199 张;70-79 散图 207 张;低于 70 分 / 拍摄者目录 2457 张;备注：个体数按标签 {session}_{group_id} 计，跨调查同名编号未合并;可读图片 2863 / 损坏 0;精确重复组 1 组 2 张

#### Slide 04 - 数据分布图表

- **Audience move**: 通过图表直观掌握数据分布 → 三个维度一目了然
- **Layout**: 上标签状态分布图 + 下两图并排（调查批次分布 / 评分区间分布）
- **Title**: 数据分布 — 真实扫描统计
- **Core message**: 数据分布三张图全部来自实际扫描结果
- **Content**: 标签状态分布：labeled 199 / loose_known 207 / ignored 2457;调查批次分布：01 818 / 03 2045;评分区间分布：unknown 2018 / 60-69 401 / 70-79 223 / 80+ 183 / 50-59 33 / below 50 5
- **Visualization**: label_status.png / session_dist.png / quality_band.png（真实数据）
- **Native-ready**: no

#### Slide 05 - 数据特点 → 技术方向

- **Audience move**: 认识数据特殊性 → 理解这些特点直接决定了技术路线
- **Layout**: 四列「特点 → 应对」对应卡
- **Title**: 数据特点决定技术方向
- **Core message**: 少样本 / 散图 / 左右侧 / 跨调查四个特点，各自指向明确的技术应对
- **Content**: 极端少样本（43 个体各 1 张）→ 迁移学习：外部鲸豚数据预训练 + 本地微调;207 张散图归属未确认 → 分拣扩充标签 + candidate-label 弱监督;背鳍左右两面 → 朝向标注，左右侧分别比较;同一天两群 / 跨调查未核验 → 调查内比较，跨调查仅作候选

### Part 3: 技术方向（大头）

#### Slide 06 - 散图利用方案

- **Audience move**: 理解 207 张散图不是负担而是资产 → 明确两条利用路径
- **Layout**: 上左右两栏（分拣路径 / 弱监督路径）+ 底部小结条
- **Title**: 散图利用 — 207 张候选图的两种用法
- **Core message**: 散图分拣后直接扩充个体标签（59+148），分拣前作 candidate-label 弱监督参与训练
- **Content**: 路径 A 人工分拣（01：59 / 03：148）— 分配到对应个体文件夹 → 更新 Manifest → 个体标签扩充，缓解 one-shot;路径 B candidate-label 弱监督 — 未分拣前参与训练，按「候选集合」约束（同调查同评分区间）;小结：分拣即扩样本，每分拣一张散图，可用训练样本 +1

#### Slide 07 - 迁移学习方案 I（核心实验）

- **Audience move**: 理解迁移学习是解决少样本的关键 → 掌握四组对照实验与判定标准
- **Layout**: 四组实验卡（A/B/C/D 对照）+ 底部判定标准条
- **Title**: 迁移学习 — 四组对照实验
- **Core message**: 用「通用预训练 vs 鲸豚监督迁移 vs 多源鲸豚」四组实验，验证外部鲸豚数据是否真能帮助本地少样本识别
- **Content**: A 基线 — ImageNet / DINOv2 / MegaDescriptor 预训练 + 本地; B 鲸豚监督迁移 — Happywhale 预训练 + 本地; C 自监督鲸豚迁移 — Happywhale / NDD20 self-supervised + 本地; D 多源鲸豚 — Happywhale + NDD20 + NOAA + Beluga + 本地;判定：C/B > A 说明迁移有效；C < B 说明存在 source cue bias / 负迁移;推荐执行顺序 A → B → D → C;核心判断：迁移的是 cetacean dorsal-fin representation，不是源物种个体分类规则

#### Slide 08 - 迁移学习方案 II（数据源与风险）

- **Audience move**: 了解外部数据从哪来、风险在哪 → 明确执行顺序
- **Layout**: 左数据源列表 + 右上风险卡 + 右下执行顺序
- **Title**: 迁移学习 — 外部数据源与执行顺序
- **Core message**: 四类外部数据覆盖轮廓与纹理两种判别线索，按规模从小到大执行，验证负迁移风险
- **Content**: 数据源（P0）：Happywhale 5.1 万张 / 1.56 万个体 / 30 物种;NDD20 4402 张（背鳍识别 + 分割标注）;NOAA 宽吻海豚 photo-ID（188 个体，背鳍形态）;Beluga ID 5902 张 / 788 个体（疤痕纹理）;负迁移风险：源域靠背鳍轮廓、目标域靠斑点纹理，直接分类迁移可能忽略纹理;执行顺序：NDD20 → NOAA → Beluga → Happywhale → 多源融合

### Part 4: 路线与行动

#### Slide 09 - 技术路线总览

- **Audience move**: 看清全链路 → 理解各环节如何衔接
- **Layout**: 全管线横向流程（8 步，箭头连接）+ 底部两类输出说明
- **Title**: 技术路线 — 从裁剪到个体数据库
- **Core message**: 从背鳍裁剪 → 特征 → 检索 → 聚类 → 人工核验 → 伪标签 → 个体数据库 的完整链路
- **Content**: 管线：背鳍裁剪与朝向标注 → 预训练 embedding（MegaDescriptor）→ L2 归一化 + 余弦相似度 → Faiss Top-K 检索 → HDBSCAN 候选聚类 → 人工核验（FiftyOne）→ 伪标签 → 个体数据库;输出：Top-K 候选 / 未知个体 / 需人工核验三类;闭环：核验结果回流微调模型

#### Slide 10 - 阶段规划

- **Audience move**: 了解 0-6 阶段整体规划 → 明确当前进度与下一步
- **Layout**: 阶段 0-6 横向进度条 + 底部「当前」标记
- **Title**: 阶段规划 — 当前处于阶段 2 起点
- **Core message**: 阶段 0-1（语义确认 + 数据索引）已完成，阶段 2（Pilot Set）开始
- **Content**: 阶段0 语义确认 [x] → 阶段1 数据索引 [x] → 阶段2 Pilot Set 与最小基线 [~] → 阶段3 人工基准集 → 阶段4 自监督与轮廓特征 → 阶段5 伪标签与度量学习 → 阶段6 自动化系统;当前：2.1 选择 Pilot Set（进行中）;底部：阶段 0-1 交付物（Manifest / Stats / Tree / unreadable）

#### Slide 11 - 交付与行动清单

- **Audience move**: 明确已交付 + 下一步行动 + 待确认 → 知道该干什么
- **Layout**: 上索引四件套交付卡 + 下行动清单（3 项）与待确认（压缩为 1 行）
- **Title**: 交付与行动 — 索引已就绪，行动待启动
- **Core message**: 索引四件套已交付；下一步三件事：散图分拣、外部数据下载、朝向标注
- **Content**: 交付：dataset_manifest.csv（2863×28）/ dataset_stats.json / dataset_tree.txt / unreadable_files.csv;行动：① 散图分拣（207 张，人工）→ ② 外部数据下载（NDD20 / Happywhale）→ ③ 朝向标注规范;待确认：数据授权 / individual.docx / 50-69 区间含义（一行）

#### Slide 12 - 结尾

- **Audience move**: 总结汇报 → 明确下一步行动
- **Layout**: 深蓝背景 + 居中白色大字 + 金色装饰线 + 下一步行动
- **Title**: 下一步 — 数据已就绪，方向已明确
- **Core message**: 数据盘点与索引已完成，迁移学习与散图利用方向已定，下一步构建 Pilot Set 并启动外部数据迁移实验
- **Content**: 核心信息回顾：数据现状清晰，技术方向明确（迁移学习 + 散图利用）;下一步行动：① 散图分拣扩充标签 ② 下载外部数据启动迁移实验 ③ Pilot Set 基线;等待确认：数据授权;谢谢，欢迎讨论
- **Closing impact**: 深蓝底白字 + 下一步行动三行 + 金色装饰线收尾

## X. Speaker Notes Requirements

- **Filename**: match each SVG filename under `notes/`
- **Content**: 每页 3-5 句口语化讲解，数据引用来源（dataset_stats.json / README §5 / docs/cetacean_reid_transfer_learning_plan.md）
- **Total duration**: 约 12 分钟（12 页，每页 ~1 分钟）
- **Notes style**: conversational
- **Presentation purpose**: report
