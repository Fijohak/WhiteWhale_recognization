# Anchor / Pool 数据语义（2026-08-11 确认）

## 1. 背景

历史数据整理时，工作人员从某个个体的照片中挑选一张质量或评分最高的**代表照片**单独保存，其余照片没有严格按照个体归档，可能混入 `70-79` 等公共照片池。因此：

**原始文件夹结构不能直接转换成 `individual_id`。**

## 2. 数据模型

```text
Anchor / Representative Image     （代表照片，可作检索查询）
    +
Unresolved Image Pool             （未归属照片池，作检索 Gallery）
```

- `80 and above`、`70-79` 目录主要表示**历史筛选或评分结果**，不表示个体身份。
- 数字子文件夹中的照片 = 候选个体的 Anchor；在没有进一步证据前，**不自动认为目录中其他照片属于该个体**。

## 3. 概念定义

| 概念 | 定义 |
|---|---|
| Anchor | 历史挑选的代表照片。可作检索查询（query），未经人工确认前不代表身份 |
| Unresolved Image Pool | 未归属照片池（如 `70-79` 散图、低于 70 分照片等）。作检索 Gallery（gallery） |
| Candidate Cluster | 检索或聚类产生的候选分组，仅作人工审核候选 |
| Confirmed Individual | 经人工核验确认的个体（进入 Confirmed Individual Catalogue） |

## 4. 使用规则

### 允许

- 用 Anchor 作 query，对 Pool（及同调查其他 Anchor）作 Top-K 检索；
- 检索结果为**候选**，供人工审核 same / different / uncertain；
- 人工确认后可逐步扩充 Confirmed Set，用多个确认样本再检索（prototype / 融合排名）。

### 禁止（在可信身份数据建立以前）

- 把数字文件夹当作个体 ID；
- 把 `70-79`、`80+` 等目录当作个体类别；
- 直接训练 individual classifier；
- 用 Source Group 训练 ArcFace；
- 为训练强行制造伪标签；
- 把 HDBSCAN 聚类结果直接称为真实个体；
- 因某张照片与 Anchor 相似就自动确认身份。

## 5. 扩展流程

```text
Seed（Anchor A）
→ Retrieve（Top-K）
→ Review（人工确认 A2 / A3）
→ Expand（Confirmed Set：A + A2 + A3）
→ Retrieve Again（prototype / 融合排名）
```

每次扩展必须经过人工确认，避免错误身份传播。

## 6. 数据划分要求

- 同一次连续拍摄产生的照片**不能**被随机拆到 query 和 gallery；
- 优先建立 cross-sequence / cross-encounter / cross-date 测试；
- 若只能做到同一次拍摄内匹配，必须明确标记，不得夸大为长期个体识别效果。

## 7. 当前 Pilot

- Anchor：高分目录 43 组（199 张）
- Pool：`70-79` 散图 207 张（01：59，03：148）
- 当前流程：预训练 embedding → Anchor Top-K 检索 → 人工审核 → 逐步恢复目录
