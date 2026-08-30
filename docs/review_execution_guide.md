# 人工核验执行指南：r4 未命中诊断与历史库一致性

> 本指南把当前最优先的两项人工工作整理为可交接的执行卡。详细字段和异常处理分别以
> [未命中诊断任务单](task_missed_diag.md) 和
> [历史库核验手册](history_verify_crossyear.md) 为准。
>
> 两项工作都只核验现有证据：不得把跨 session 同名编号当同一只，也不得由模型分数
> 自动确认身份。任何结论只适用于当前批次内、短时间间隔数据。

## 一、开始前的共同约定

1. 指定一名资料管理员，只负责分发材料、收集原始结果和汇总，不参与其他审核人的
   判断讨论。
2. 每名审核人使用唯一且固定的 `reviewer` 名称，例如 `reviewer_A`；结果文件不能
   覆盖他人的文件。
3. 审核期间不展示其他人的结论。未命中查看器使用浏览器 `localStorage` 暂存结果，
   因此不同审核人必须使用不同电脑、不同浏览器 profile，或在开始前清空该网页的站点
   数据，避免看到前一人的选择。
4. 原图含拍摄时间和地点信息，只通过项目原图盘、U 盘或局域网传递，不上传公共网盘。
5. “不确定”是正确且有价值的结果；任何分歧都不强行归并或修改标签。

## 二、任务 A：三人复核 r4 的 35 张未命中证据

### A.1 目标与材料

目标是区分模型未命中的主要来源：同体外观差异、异体顶替、检测/裁剪/画质、序列
划分问题或标签异常待复核。它不是重新训练，也不直接修改 `pilot_set.csv`。

使用当前验收包：

| 材料 | 路径 |
|---|---|
| 查看器 | `outputs/reports/missed_diag_r4_v2_full_manifest_v3/index.html` |
| 逐 query 明细 | `outputs/reports/missed_diag_r4_v2_full_manifest_v3/missed_query_detail.csv` |
| 35 张证据拼图 | 同目录 `miss_*__qNN.png` |
| 原图 | `src_dataset/` 下 CSV 记录的路径 |

当前共有 35 张 Top-1 未命中 query；另有 11 张因图库没有跨串同体正样本被跳过，
不纳入本次 35 张人工结论。

### A.2 每位审核人的步骤

1. 在各自独立的浏览器 profile 中双击打开 `index.html`，先填唯一审核人名称。
2. 按卡片顺序核对 Q、SAME、T1–T3：
   - Q 与 SAME 是否确为同一只；
   - T1–T3 中谁视觉上最像 Q；
   - 背鳍缺刻、凹陷、伤痕、斑点、侧别，以及裁剪和画质问题；
   - `q_series_id` 是否提示同串遗漏或误分。
3. 每张卡片填写 Q-SAME 判断、可多选的原因、可见特征、最相似候选和证据备注。
   看不清时选择“无法判断”，不要根据余弦分数猜测。
4. 审完 35 张后，点击“复制到剪贴板”，将 JSON 保存为
   `outputs/review/missed_diag_r4_v3/<reviewer>.json`。保存前检查 JSON 中 `reviewer`
   非空、共有 35 条记录且每条都有 `q_image_id`。
5. 把 JSON 交给资料管理员；审核人之间在管理员公布汇总前不讨论。

### A.3 管理员汇总与完成标准

1. 保留三份原始 JSON，不覆盖、不改写。
2. 以 `q_image_id` 对齐三人结果，生成 `consensus.csv` 和 `disputes.csv`。
3. 只有三人对 Q-SAME 判断和关键原因完全一致，才能写入 `consensus.csv`；任一
   “不确定”或分歧写入 `disputes.csv`，交专家复核。
4. 输出 3–5 句总结：每种失败原因的数量、是否存在序列问题、是否需要修正检测裁剪，
   以及哪些项目不应改标签。

建议交付目录：

```text
outputs/review/missed_diag_r4_v3/
├── reviewer_A.json
├── reviewer_B.json
├── reviewer_C.json
├── consensus.csv
├── disputes.csv
└── summary.md
```

## 三、任务 B：历史库 43 组、202 张二次一致性核验

### B.1 目标与前提

这不是否定数据提供方已确认的批次内 `individual_id`，而是建立独立可信的历史基准，
排查组内异常照片。完成前，不能把历史库用于跨年确认结论。

材料：

| 材料 | 路径 |
|---|---|
| 核验清单 | `outputs/review_package/batches_history/history_verify.csv` |
| 簇内特征 | `outputs/review_package/batches_emb/history_verify.npy` |
| 历史对照图 | `outputs/review_package/history_lookup/` |
| 原图 | `src_dataset/20140806 01`、`src_dataset/20140806 03` |
| ID 迁移表 | `outputs/pilot/individual_id_migration_v1_to_v2.csv` |

### B.2 启动与逐组核验

在仓库根目录执行（每名审核人使用独立 `--reviewer` 和 annotations 文件）：

```powershell
python scripts/launch_review.py `
  --clusters outputs/review_package/batches_history/history_verify.csv `
  --images-root src_dataset `
  --batch-embeddings outputs/review_package/batches_emb/history_verify.npy `
  --annotations outputs/review/history_verify_<reviewer>.csv `
  --reviewer <reviewer> --port 8002
```

浏览器打开 `http://127.0.0.1:8002` 后，逐组检查：

1. 整组显然都是同一只：整簇确认，名称必须保持该组原名；
2. 某图疑似混入：逐图选择确认、不确定或排除，并写明背鳍细节/画质原因；
3. 看不清：选择不确定，不得因模型相似度高而确认。

每人可中途退出；标注会保存到其独立 CSV。应先完成 202/202 覆盖；出现不确定或
排除的图交第二名审核人复看。此前撤回的 6 张跨年照片必须由至少 3 人独立审核且
完全一致才可能解除 `uncertain` 状态，具体见历史库核验手册第二节。

### B.3 汇总、回填与停止条件

资料管理员完成 `outputs/review/history_verify_summary.csv`，至少包含：

```text
group,n_images,n_confirmed,n_uncertain,n_reject,结论
```

结论只能是 `通过`、`需拆分` 或 `需复核`。任何组只要有一张不确定或排除，就不能写
“通过”。检查无误后才可执行：

```powershell
python scripts/finalize_history_verify.py `
  --summary outputs/review/history_verify_summary.csv `
  --pilot outputs/pilot/pilot_set.csv --out outputs/review
```

该命令会更新 `pilot_set.csv` 的 `review_status` 并创建备份；因此只在汇总表审定后由
资料管理员执行一次。产出 `history_verified_individuals.csv` 后，仍需完成撤回照片的
多人复审，才可以讨论任何跨年匹配。

## 四、每天结束前检查

- 未命中复核：每位审核人恰有 35 条 JSON 记录，`q_image_id` 无重复；
- 历史库核验：累计标注数、组数和未处理数可统计，原始 CSV 保留；
- 不分享其他审核人的判断，不覆盖原始票；
- 没有把 `possibly_same`、模型候选或跨 session 同名编号写成确认同体。
