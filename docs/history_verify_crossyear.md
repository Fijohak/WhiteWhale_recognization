# 历史库核验与跨年匹配重建操作手册

> 目标读者：接手此项工作的组员/同学。按本手册操作即可独立完成，不需要问开发人员。
> 三步骤串行依赖：**3.6 历史库核验 → 3.7 撤回照片重审 → 3.8 跨年匹配重建**。
> 对应 README 待办：3.6 / 3.7 / 3.8。

---

## 〇、背景（为什么要做）

- 项目里 43 个体、202 张"历史库"照片（2014-08-06 一天拍的），`individual_id`
  是数据提供方确认的批次内个体身份，但不是跨批次全局 ID。3.6 是独立的二次一致性
  核验，用于发现异常照片并建立更严格的跨时间基准，不是否定原始身份语义。
- 当前 pilot 使用保留来源目录名的 canonical ID（例如 `20140806 01_01`）；早期
  review package 目录仍保留 `20140806 01_1.0` 等 legacy ID。对应关系以
  `outputs/pilot/individual_id_migration_v1_to_v2.csv` 为准。正式回填工具会兼容旧 ID，
  但新建汇总表建议直接写 canonical ID；禁止靠去掉 `.0` 等字符串猜测手工映射。
- 2026-08-23 的教训：初审在没核验历史库的情况下确认了 3 条"跨年匹配"，数据提供方复核后**全部撤回**（6 张照片判定为审核失误）。所以红线是：**历史库核验通过之前，一切跨年匹配都不算数**。
- 2026-08-29 的 r5 只在短时间间隔、批次内完整跨串协议上训练和评估；全量同协议结果
  低于 r4，未切换生产。该实验没有提供跨年正样本，也不改变当前“跨年匹配 0 条确认”
  的事实。
- 本手册完成 3.6+3.7 后，历史库才是可信基准，3.8 才有意义。

---

## 一、3.6 历史库核验（43 组 202 张）

### 1.1 需要的材料

| 材料 | 位置（仓库内） | 说明 |
|---|---|---|
| 审核程序 | `outputs/review_package/code/review_app.py` + `review_app.html` | 独立网页程序，不依赖仓库其他代码 |
| 核验清单 | `outputs/review_package/batches_history/history_verify.csv` | 202 行 = 202 张待核验照片，按组排列 |
| 批次特征 | `outputs/review_package/batches_emb/history_verify.npy` | 簇内相似度辅助（混入者自动沉底标红） |
| 历史照片目录 | `outputs/review_package/history_lookup/` | 43 组，按个体分目录；目录名是历史 legacy ID（如 `20140806 01_1.0`），须经迁移表对应当前 canonical ID |
| 原图 | `src_dataset/20140806 01`、`src_dataset/20140806 03` | 审核时看原图背鳍细节 |
| 质量表 | `outputs/review_package/history_quality.csv` | 模糊/未检出背鳍的对照图标记（核验时低质图会变灰，可取消隐藏） |

> 若给别人核验：把 `review_package/` 整个目录 + `src_dataset/` 下两个批次目录拷到对方电脑（U 盘/局域网，**不走外网网盘**——照片含拍摄时间地点信息）。对方电脑上两者放同一层，例如 `D:\whale\20140806 01` + `D:\whale\review_package`。

### 1.2 环境（一次性）

```powershell
pip install fastapi uvicorn pandas numpy pillow
```

### 1.3 启动审核网页

```powershell
cd D:\whale\review_package
python code/review_app.py --clusters "batches_history/history_verify.csv" --images-root D:\whale --batch-embeddings "batches_emb/history_verify.npy" --annotations "outputs/review/review_annotations_history_verify.csv" --reviewer 你的名字 --port 8002
```

浏览器打开 http://127.0.0.1:8002

### 1.4 怎么核验（判定规则）

- 页面**左侧每栏 = 一个历史组**（如 "1.0" = `20140806 01_1.0` 组）；右侧无对照——**只判断组内是否同一只**。
- 组内照片按相似度排序，**可疑混入者沉底红框**（左下有簇内相似度数值）。
- 操作（三选一，整组判定优先）：
  1. **组内全是同一只** → 审核旧包时可输入该组 legacy 原名（如 `20140806 01_1.0`）→「整簇确认」；汇总/回填时工具按迁移表解析为 canonical ID；
  2. **组内混入了不同个体** → 点缩略图放大原图，逐张「确认 / 不确定 / 排除」（原图背鳍后缘的缺刻、凹陷、缺口形状是个体间差异最大的特征）；
  3. **看不清** → 「不确定」；**确认不是该组** → 「排除」。
- 红线：
  - **同名 = 同一只**：组内确认的个体名必须与组名一致，否则后续会合并错；
  - **不硬凑**：宁可「不确定」多留待定，不可错合并（错合并会让种群统计低估，是最危险错误）；
  - 每张照片标注即时写入 CSV，中途可随时关掉，下次接着审。

### 1.5 完成标准（什么算核验通过）

全部 202 张审完，且满足：

1. **审核覆盖率 = 100%**：`outputs/review/review_annotations_history_verify.csv` 内 202 个 image_id 全部有标注（没有遗漏）；
2. **组级结论齐全**：每组都明确归属「整组确认 / 部分排除 / 整组不确定」，并把结果整理到一张汇总表 `outputs/review/history_verify_summary.csv`（列：`group`、`n_images`、`n_confirmed`、`n_uncertain`、`n_reject`、`结论`，结论 ∈ 通过/需拆分/需复核）；
3. **冲突处理**：某张「不确定」或「排除」的，记录原因（模糊/混入/看不清），需第二人复核的进 3.7 流程；
4. **回填系统**：把通过核验的组（组内无排除、无不确定）登记为可信基准，用回填脚本一键完成（自动校验 + 改前备份）：
   ```powershell
   python scripts/finalize_history_verify.py `
       --summary outputs/review/history_verify_summary.csv `
       --pilot outputs/pilot/pilot_set.csv --out outputs/review
   ```
   - 脚本产出：`outputs/review/history_verified_individuals.csv`（列：`individual_id`（组名）、`n_images`、`verified_date`）；pilot_set 中对应照片 `review_status` 更新为 `verified`；
   - 内置防护：结论=通过但组内有不确定/排除 → 拒绝；通过组名不在 pilot_set → 拒绝；回填照片数与汇总表登记数不一致 → 拒绝；改前自动备份 `pilot_set.csv.bak_时间戳`。

> 判定原则：**任何一组只要有一张「排除」或「不确定」，该组就不算核验通过**，跨年匹配时该组降级为"参考组"（不进可信基准），直到重审完成。

---

## 二、3.7 撤回照片重审（6 张，多人投票）

### 2.1 是哪 6 张

2026-08-23 被撤回的 3 条跨年匹配涉及的 6 张照片（当前标注均为 `uncertain`，位于 `outputs/review_package/outputs/review/review_annotations.csv`）：

| image_id | 原候选匹配 | 张数 |
|---|---|---|
| IMG_4e8e3dde958060ec | 20140806 03_1.0（20151017 03 簇 0.0） | 3 |
| IMG_6a42e0705a6dd481 | ↑ | ↑ |
| IMG_f8b660e5bbd81c52 | ↑ | ↑ |
| IMG_864adc308e704832 | 20140806 03_5.0（20151017 03 簇 1.0） | 2 |
| IMG_31ca2021bb88c9f4 | ↑ | ↑ |
| IMG_3bc18b6aa6c5c78e | 20140806 01_2.0（20151017 02 噪声池） | 1 |

### 2.2 流程（关键判定，多人把关）

1. **至少 3 名审核人独立审**：每人打开审核网页（命令同上，`--port` 换不同端口、`--reviewer` 写各自名字），对 6 张照片单独判定「确认（填个体名）/ 不确定 / 排除」；
2. **独立标注**：审完前互相不通气，避免从众；
3. **保守裁决**：至少 3 名命名审核人且结论完全一致才确认；2:1、1:1:1、
   人数不足或任意分歧均维持 `uncertain`，不强行归档；
4. **结果落库**：每人的原始票保留在 `review_annotations.csv`，程序另行导出
   `review_vote_summary.csv` 和仅含一致票的 `confirmed_individuals.csv`；争议票不覆盖原始记录。

> 6.11 已完成：仓库正式入口 `scripts/launch_review.py` 支持同一 annotations
> 文件内多人独立票、盲审隔离、并发合并和保守裁决。每人必须传唯一 `--reviewer`；
> 最后不带 `--reviewer` 执行 `--export --min-reviewers 3`，个人参数不能绕过共识导出。

### 2.3 红线

- 单人不能定论（E9 教训：初审者自审会受模型候选先入影响）；
- 撤回照片在重新裁决前**不得**出现在任何跨年匹配结果里。

---

## 三、3.8 跨年匹配重建（前置：3.6 通过 + 3.7 完成）

### 3.1 前置条件确认（不满足则停下）

- [ ] 3.6 核验完成：`history_verified_individuals.csv` 已生成，历史库可信基准就绪；
- [ ] 3.7 重审完成：6 张照片裁决已定论，不确定者已排除出匹配候选；
- [x] 多人复核机制就绪（至少 3 人独立审核、完全一致才确认，6.11 工具已完成）。

### 3.2 重跑跨时间管线（历史库 → 7 个新批次）

```powershell
# 仓库根目录
python scripts/run_cross_time_batch.py
```

- 流程：历史库（20140806 01/03 labeled）→ YOLO 裁剪 + 当前生产 r4 特征 → 逐个新批次跑批内归档 + 匹配历史库（E7 已验证可跑通）；
- 可只跑指定批次：`python scripts/run_cross_time_batch.py --sessions "20140419 02"`；
- `python scripts/run_cross_time_batch.py --only-gallery` 只读校验当前活动 gallery，不会原地重建；
- 当前活动严格产物为 `outputs/artifacts/r4_yolocrop_v3/`。如需重建，使用
  `python scripts/rebuild_r4_artifacts.py --out outputs/artifacts/<新版本目录>` 发布新目录，
  验证后再显式修改配置；不得把 r5 候选手工替换进生产链路。

### 3.3 人工审核匹配候选（防错并优先）

1. 用审核网页逐批次审：左侧 = 子簇，右侧 = 模型建议的历史个体对照（`--history-lookup` 指到核验后的 `history_lookup/`）；
2. **候选 ≠ 确认**：模型建议仅供参考，以人工看背鳍为准；
3. 确认的跨年匹配必须满足：
   - 匹配对的目标个体**已通过 3.6 核验**（在 `history_verified_individuals.csv` 中）；
   - 判定走多人流程：至少 3 人独立审核且结果完全一致（用 6.11 投票）；
4. 裁决原则（E9 结论）：**宁可拆分不可错并**；不确定一律 `possibly_same`（不进确认库）。

### 3.4 成功标准与记录

- 产出：`outputs/review/confirmed_individuals.csv`（含跨年匹配行）；
- 每条确认的跨年匹配写入 `EXPERIMENT_LOG.md`（新实验 E12+）：照片对、分数、投票、核验依据；
- 工程完成标准是管线重跑、候选全部完成多人裁决且证据可追溯；确认匹配数可以为 0。
  0 条确认是**合法科研结果**（可能数据覆盖不足，未确认 ≠ 不存在，不得为了有成果强行确认）。

---

## 四、常见问题

| 问题 | 处理 |
|---|---|
| 审核网页打不开 | 检查端口被占用（换 `--port`）；检查 `--images-root` 路径层级（批次目录要在其下） |
| 照片加载不出来 | 原图在 `src_dataset/` 被移动/只读权限 → 确认路径存在；历史对照图在 `history_lookup/` |
| 组里混入者标红但看起来像同体 | 以原图背鳍细节为准，红框只是相似度提示不是结论 |
| 核验中途退出 | 标注已即时写入 CSV，重新启动同一 `--annotations` 路径继续 |
| 不确定太多 | 正常。宁可多留待定，不要为了"完成"硬确认 |
| 跨年匹配 0 条 | 合法。记录结果即可（E9 归因：历史库全在 2014-08-06 一天，跨年覆盖不足） |

## 五、操作红线汇总

1. 历史库核验通过前，不产出任何跨年匹配结论；
2. 同一核验组内确认名必须与组名一致；跨 session 同名编号不代表同一只；
3. 关键判定（撤回照片、跨年匹配）多人独立 + 投票，单人不得定论；
4. 不确定/排除是合法审核结论，不强行归档；
5. 照片含地点时间信息，传递走 U 盘/局域网，不走外网。
