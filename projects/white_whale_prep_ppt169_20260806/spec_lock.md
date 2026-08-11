<!-- ppt-master-schema: spec-lock/v1 -->
# Execution Lock

## canvas
- viewBox: 0 0 1280 720
- format: PPT 16:9

## communication
- audience: 项目负责人 / 导师及团队成员（本周汇报），对本项目背景已有了解，关心进展、数据事实与下一步计划
- objective: 让听众掌握前期准备（数据盘点与索引）完成情况，理解数据特点如何决定两大技术方向（跨物种迁移学习四组对照实验、散图分拣即标签扩充），明确下一步行动（散图分拣 / 外部数据下载 / Pilot Set 基线）
- core_message: 数据盘点与索引已完成并全面写入文档；基于数据现状确定两大技术方向：① 跨物种迁移学习（外部鲸豚预训练 + 本地微调，四组对照实验验证）② 散图利用（分拣扩充标签 + candidate-label 弱监督）；下一步启动散图分拣、外部数据下载与 Pilot Set 基线
- consumption_mode: balanced

## mode
- mode: briefing

## visual_style
- visual_style: editorial

## colors
- background: #FFFFFF
- secondary_bg: #F2F5F9
- primary: #1E3A5F
- accent: #E8B339
- secondary_accent: #BF0000
- body_text: #2B3440

## typography
- font_family: 微软雅黑, Microsoft YaHei, sans-serif
- title_family: 微软雅黑, Microsoft YaHei, sans-serif
- body_family: 微软雅黑, Microsoft YaHei, sans-serif
- data_family: 微软雅黑, Microsoft YaHei, sans-serif
- annotation_family: 微软雅黑, Microsoft YaHei, sans-serif
- card_title_family: 微软雅黑, Microsoft YaHei, sans-serif
- card_body_family: 微软雅黑, Microsoft YaHei, sans-serif
- body: 22
- kpi_value: 42
- title: 34
- subtitle: 26
- annotation: 14
- card_title: 19
- card_body: 17

## icons
- library: none
- inventory: none

## images
- p04_label: images/label_status.png | source=user | pattern=#6 bottom-band image | crop=no-crop
- p04_session: images/session_dist.png | source=user | pattern=#7 top-and-bottom symmetric split | crop=no-crop
- p04_quality: images/quality_band.png | source=user | pattern=#6 bottom-band image | crop=no-crop

## page_rhythm
- P01: anchor
- P02: dense
- P03: anchor
- P04: dense
- P05: dense
- P06: dense
- P07: dense
- P08: dense
- P09: dense
- P10: dense
- P11: dense
- P12: breathing

## pptx_structure
- mode: flat

## forbidden
- `mask`, `<style>`, `class`, external CSS, `<foreignObject>`, `textPath`, `@font-face`, `<animate*>`, `<set>`, `<script>` / event attributes, `<iframe>`
- HTML named entities in text; write typography as raw Unicode and escape XML reserved characters
