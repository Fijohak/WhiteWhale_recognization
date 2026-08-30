# 任务单：完整同串口径下的未命中逐查询诊断（r4）

> 状态（2026-08-29）：session-local、完整同串口径的真实图片证据已重建。当前产物包含
> 18 个 Top-1 未命中个体、35 张逐 query 卡片；其中 7 个个体、10 张 query 未进入 Top-5。
> CSV 中 query/SAME/T1–T3 的 image_id 与路径均非空，35 张证据图均存在。
> 旧格式产物在 `outputs/reports/missed_diag_legacy_20260827/`，旧拆分残留图在
> `outputs/reports/missed_diag_stale_splits_20260829/`；旧跨 session v2 包在
> `outputs/reports/missed_diag_r4_v2_full_manifest_v2/`。三者均不得用于当前结论。

## 一、诊断口径

- `individual_id` 是批次内已确认个体，不是候选标签；跨批次同名编号不自动合并。
- 当前展示使用保留来源目录名的 canonical ID（`source_label_preserving_v2`）；旧版
  `1.0`/`5.0` 等 ID 仅可通过 `outputs/pilot/individual_id_migration_v1_to_v2.csv`
  回溯，不能与当前 ID 直接按字符串混用。
- 同串定义为 `session_id + 文件名序列键 + 连续帧段`，连续帧段按相邻帧号差
  `<=2` 的传递闭包生成。无法解析连拍信息的照片各自独立。
- 每张 query 打分前，图库中与其相同且非空 `series_id` 的照片必须剔除。
- 每张 query 只对所属 session 内的批次内已确认身份打分；跨 session 身份没有对齐
  真值，不作候选或负类。
- 现有多图样本时间间隔短。本诊断只能解释当前短间隔跨串失败，不能据此声称
  已验证跨月或跨年识别能力。

稳定拆分后的评估器报告：43 个可评 probe 中 18 个 Top-1 未命中，对应 35 张相关 query；
其中 7 个个体的 10 张 query 未进入 Top-5；另有 11 张 query 因同 session 图库中没有
跨串正样本而 skipped，不进入指标分母。诊断目录已按这一口径重建，人工视觉裁决
仍待至少 3 人独立完成。

## 二、重建与复现证据

当前验收产物是不可覆盖的 v3 目录。需要重跑时必须换一个新版本目录（下面以 v4
为例），不能覆盖 v3：

```bash
python experiments/diagnose_missed.py \
  --feats-stem embeddings_eval51_all_r4_v2 \
  --out outputs/reports/missed_diag_r4_v2_full_manifest_v4
python experiments/build_missed_viewer.py \
  --detail outputs/reports/missed_diag_r4_v2_full_manifest_v4/missed_query_detail.csv \
  --out outputs/reports/missed_diag_r4_v2_full_manifest_v4/index.html
```

生成物：

| 文件 | 用途 |
|---|---|
| `outputs/reports/missed_diag_r4_v2_full_manifest_v3/missed_query_detail.csv` | 每张 query 的精确分数、真实 argmax 图像与路径 |
| `outputs/reports/missed_diag_r4_v2_full_manifest_v3/miss_*__qNN.png` | 逐 query 证据图：Q、SAME、T1–T3 |
| `outputs/reports/missed_diag_r4_v2_full_manifest_v3/summary.json` | 18 个体 / 35 query 未命中及 Top-5、跳过数汇总 |
| `outputs/reports/missed_diag_r4_v2_full_manifest_v3/index.html` | 分类、备注、本地保存与 JSON 导出 |

新版 CSV 必须包含 `cluster_true_rank`、`cluster_topk_hit`、`cluster_n_candidates`、
`q_series_id`、`same_image_id`、`same_path`、
`top1_image_id`、`top1_path`、`evidence_tile` 等字段。查看器会拒绝旧格式 CSV；
出现拒绝时应先重跑诊断脚本，不能手工伪造缺失列。

证据图中的 SAME 和 T1–T3 都是该 query 对应分数的实际图库 argmax 图像，
不是“每个个体随便取第一张”的代表图。表格中的分数、image_id、路径和拼图
必须能一一对应。

## 三、逐 query 审核方法

每张证据图依次判断：

1. Q 与 SAME 是否确为同一只；比较背鳍轮廓缺口、凹陷、伤痕、斑点和侧别。
2. T1–T3 中谁与 Q 最像；视觉判断须与对应 image_id 和分数一起记录。
3. 检查检测框、裁剪范围、清晰度、遮挡及 `det_conf`，区分检测问题与 ReID 问题。
4. 检查 `q_series_id` 与图库序列，若发现同串遗漏或误分，单独标记序列问题。

失败原因允许多选：

- 同体外观差异：Q 与 SAME 同体，但角度、侧别、姿态或可见特征差异大；
- 异体顶替：某个 T 候选在当前特征下压过 SAME；
- 检测/裁剪/画质：背鳍未检出、框偏、回退整图、模糊、远小或遮挡；
- 序列划分问题：文件名解析、连续帧段或 session 信息不正确；
- 标签异常待复核：仅当视觉证据明确冲突时提出。`individual_id` 默认仍按已确认标签
  处理，不能只因余弦分数低就改判标签；
- 无法判断：图像证据不足时保留，不强行归类。

## 四、多人复核要求

- 每位审核人独立完成 Q-SAME（三态：同体/异体/不确定）、最相似候选、原因多选、
  特征描述和备注；不得查看他人的未汇总结论。
- 导出 JSON 后按 `query image_id + reviewer` 保留原始判断；至少 3 人且结论完全
  一致才形成确认建议。人数不足、任意分歧或任一关键项“不确定”均进入专家复核，
  不自动改标签。
- 看不到原图或证据图时必须写“无法判断”，禁止根据 cos 数值脑补视觉描述。
- 跨批次 possibly_same 只能作为候选关系；未经过独立复核与裁决，不得写成跨年同体成果。

## 五、交付物与完成标准

交付：查看器导出的逐 query JSON、多人投票汇总、争议样本清单和 3–5 句失败模式
小结。完成前必须核对：

- 新版 CSV 的必需列齐全，记录数与查看器卡片数一致；
- 每条判断可回溯到 query/SAME/T1–T3 的 image_id、路径和证据图；
- `missed_diag_legacy_20260827/` 的旧格式包与
  `missed_diag_stale_splits_20260829/` 的旧拆分残留图、跨 session v2 包均未混入
  当前 35 张结果；
- 检测问题、序列问题、ReID 问题和标签复核建议没有被混为同一类；
- 结论明确限定在当前短时间间隔数据，不外推跨年能力。
