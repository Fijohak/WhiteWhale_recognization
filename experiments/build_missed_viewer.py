"""
未命中 query 诊断查看器生成（一次性工具）。

视觉通道不可用（模型无视觉），改为：
1. 数值驱动自动归类（同体断裂 / 跨批高相似 / 同批混淆 / 簇内不一致 / 平稳低分）；
2. 生成本地 HTML 查看器：每张 query 一张卡片，含证据拼图、数值表与结构化
   人工分类控件（localStorage 记忆 + 导出 JSON），供人工在浏览器里过一遍，
   为 3.6 历史库核验提供重点样本清单。

输入：outputs/reports/missed_diag/missed_query_detail.csv
      outputs/reports/cluster_retrieval_v2/per_individual_conservative__r4.csv
输出：outputs/reports/missed_diag/diagnose_viewer.html
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DIAG = REPO_ROOT / "outputs" / "reports" / "missed_diag"
OUT = DIAG / "diagnose_viewer.html"
REQUIRED_QUERY_COLUMNS = {
    "individual", "q_path", "q_image_id", "q_det_conf", "q_fallback",
    "q_series_id", "same_cos_max", "same_gallery_n", "evidence_tile",
    "same_image_id", "same_path",
    "top1_ind", "top1_cos", "top2_ind", "top2_cos", "top3_ind", "top3_cos",
    "top1_image_id", "top1_path", "top2_image_id", "top2_path",
    "top3_image_id", "top3_path",
}


def rule_of(ind: str, rows: list[dict]) -> list[str]:
    """数值驱动归类（供人工复核，非定论）。"""
    same = [float(r["same_cos_max"]) for r in rows
            if pd.notna(r["same_cos_max"])]
    top1s = [str(r.get("top1_ind", "")) for r in rows]
    ind_ses = ind.split("_")[0] if "_" in ind else ""
    tags = []
    if same and min(same) < 0.25:
        tags.append("同体特征断裂(外观/图质/标签待复核)")
    if any(t != ind_ses for t in top1s):
        hi = [float(r["top1_cos"]) for r in rows
              if str(r.get("top1_ind", "")).split("_")[0] != ind_ses
              and pd.notna(r.get("top1_cos"))]
        if hi and max(hi) >= 0.6:
            tags.append("跨批高相似顶替(possibly_same 候选)")
    same_ses = [float(r["top1_cos"]) - float(r["same_cos_max"])
                for r in rows
                if str(r.get("top1_ind", "")).split("_")[0] == ind_ses
                and pd.notna(r.get("top1_cos"))
                and pd.notna(r.get("same_cos_max"))]
    if same_ses and max(same_ses) >= 0.15:
        tags.append("同批混淆(同群相似个体)")
    if same and max(same) - min(same) > 0.35:
        tags.append("簇内不一致(部分query同体、部分断裂)")
    if not tags:
        tags.append("平稳低分(特征区分度不足)")
    return tags


def build_cards(det: pd.DataFrame) -> list[dict]:
    """校验逐 query 证据字段并构造一张 query 一张卡片的数据。"""
    missing = sorted(REQUIRED_QUERY_COLUMNS - set(det.columns))
    if missing:
        raise ValueError(
            "missed_query_detail.csv 是旧格式或不完整，需先重跑 "
            f"diagnose_missed.py；缺少列：{missing}")
    cards = []
    for ind, g in det.groupby("individual"):
        rows = g.to_dict("records")
        group_rules = rule_of(ind, rows)
        for row in rows:
            cards.append({
                "individual": ind,
                "n_query": int(len(rows)),
                "q_no": int(row.get("q_no", 0)),
                "rules": group_rules,
                "query": row,
            })
    return cards


def main():
    """从逐 query 诊断明细生成可离线审核的 HTML 查看器。"""
    parser = argparse.ArgumentParser(description="生成保守口径未命中诊断查看器")
    parser.add_argument(
        "--detail", type=Path, default=DIAG / "missed_query_detail.csv",
        help="diagnose_missed.py 生成的逐 query 明细 CSV",
    )
    parser.add_argument(
        "--out", type=Path, default=OUT,
        help="HTML 输出路径",
    )
    args = parser.parse_args()

    det = pd.read_csv(args.detail)
    cards = build_cards(det)

    body = []
    for c in cards:
        body.append(json.dumps(c, ensure_ascii=False))
    html = TEMPLATE.replace("__DATA__", "[\n" + ",\n".join(body) + "\n]")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"[done] {args.out}（{len(cards)} 张 query 卡片）")


TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>保守口径未命中诊断（r4）</title>
<style>
  body { font-family: "Microsoft YaHei", sans-serif; margin: 16px auto; max-width: 1100px;
         color: #1a1a1a; background: #fafafa; }
  h1 { font-size: 20px; } p.desc { color: #555; font-size: 13px; }
  .card { background: #fff; border: 1px solid #ddd; border-radius: 6px; margin: 14px 0;
          padding: 12px; }
  .card h2 { font-size: 15px; margin: 0 0 6px; }
  .tag { display: inline-block; background: #eef2ff; color: #1e3a5f; border-radius: 3px;
         font-size: 12px; padding: 1px 8px; margin-right: 6px; }
  .tag.warn { background: #fff3e0; color: #b45309; }
  .tile { width: 320px; max-width: 100%; border: 1px solid #eee; margin: 4px 0; }
  table { border-collapse: collapse; font-size: 12px; margin: 6px 0; }
  th, td { border: 1px solid #ddd; padding: 2px 6px; text-align: left; }
  th { background: #f2f4f8; }
  .cls label { margin-right: 10px; font-size: 13px; }
  textarea { width: 100%; height: 120px; margin-top: 8px; font-size: 11px; }
  input[type=text] { margin-top: 4px; }
  .btn { margin: 4px 4px 0 0; padding: 4px 12px; cursor: pointer; }
</style>
</head>
<body>
<h1>保守口径未命中 query 诊断（r4 特征）</h1>
<p class="desc">
  每张 query 的证据拼图都严格绑定实际产生 cosine 分数的 SAME/T gallery 照片。
  自动归类标签仅供筛选（<span class="tag warn">黄标签</span>），人工结论支持多选原因和结构化证据，
  结果存本地并可导出，供历史库核验使用。
</p>
<p><label>审核人（必填）：<input id="reviewer" type="text" placeholder="姓名或唯一代号"></label></p>
<div id="cards"></div>
<h3>导出诊断结果（JSON）</h3>
<textarea id="export" readonly></textarea>
<button class="btn" onclick="copyExport()">复制到剪贴板</button>
<script>
const DATA = __DATA__;
const CLS = ["同体外观差异", "异体顶替", "检测/裁剪/画质", "序列划分问题", "标签异常待复核", "无法判断", "其他"];
const QSAME = ["未判断", "同体", "异体", "不确定"];
const reviewerInput = document.getElementById("reviewer");
reviewerInput.value = localStorage.getItem("missed_diag_reviewer") || "";
reviewerInput.addEventListener("input", () => {
  localStorage.setItem("missed_diag_reviewer", reviewerInput.value.trim());
  updateExport();
});
function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch =>
    ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
}
function fmt(value, digits=3) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "NA";
}
function shortPath(path) {
  return String(path ?? "").replaceAll("\\", "/").split("/").slice(-2).join("/");
}
function savedFor(key) {
  try { return JSON.parse(localStorage.getItem(key) || "null"); }
  catch (_) { return null; }
}
function card(c) {
  const q = c.query;
  const key = "missed_diag_query_" + q.q_image_id;
  const saved = savedFor(key);
  const tags = c.rules.map(t => `<span class="tag warn">${esc(t)}</span>`).join("");
  const checks = CLS.map(reason =>
    `<label><input type="checkbox" name="classify" value="${esc(reason)}" ${saved && (saved.classify || []).includes(reason) ? "checked" : ""}>${esc(reason)}</label>`).join("");
  const qsame = QSAME.map(value =>
    `<option value="${esc(value)}" ${saved && saved.q_same === value ? "selected" : ""}>${esc(value)}</option>`).join("");
  return `<div class="card" data-key="${esc(key)}">
    <h2>${esc(c.individual)} — query ${c.q_no}/${c.n_query} ${tags}</h2>
    <table><tr><th>query 图</th><th>对应证据拼图</th><th>series</th><th>同体cos/图</th><th>top1</th><th>top2</th><th>top3</th><th>det</th></tr>
      <tr><td title="${esc(q.q_path)}">${esc(shortPath(q.q_path))}</td>
      <td><a href="${esc(q.evidence_tile)}" target="_blank"><img class="tile" src="${esc(q.evidence_tile)}" alt="${esc(q.q_image_id)}"></a></td>
      <td>${esc(q.q_series_id || "无法解析")}</td>
      <td>${fmt(q.same_cos_max)} / ${esc(q.same_image_id || "无")}</td>
      <td>${esc(q.top1_ind)} (${fmt(q.top1_cos)})<br>${esc(q.top1_image_id)}</td>
      <td>${esc(q.top2_ind)} (${fmt(q.top2_cos)})<br>${esc(q.top2_image_id)}</td>
      <td>${esc(q.top3_ind)} (${fmt(q.top3_cos)})<br>${esc(q.top3_image_id)}</td>
      <td>${q.q_fallback ? "回退" : fmt(q.q_det_conf, 2)}</td></tr></table>
    <div class="cls">原因（可多选）：${checks}<br>
      Q-SAME：<select name="q_same">${qsame}</select><br>
      <input type="text" name="features" placeholder="背鳍特征（缺口/斑点/伤痕/侧别）" style="width:95%" value="${esc(saved ? (saved.features || "") : "")}"><br>
      <input type="text" name="most_similar" placeholder="与 T 谁最像" style="width:45%" value="${esc(saved ? (saved.most_similar || "") : "")}">
      <input type="text" name="evidence" placeholder="判断证据" style="width:48%" value="${esc(saved ? (saved.evidence || "") : "")}">
    </div>
  </div>`;
}
document.getElementById("cards").innerHTML = DATA.map(card).join("");
document.querySelectorAll(".card").forEach(el => {
  const save = () => {
    const classify = [...el.querySelectorAll("input[name=classify]:checked")].map(x => x.value);
    const q_same = el.querySelector("select[name=q_same]").value;
    const features = el.querySelector("input[name=features]").value;
    const most_similar = el.querySelector("input[name=most_similar]").value;
    const evidence = el.querySelector("input[name=evidence]").value;
    localStorage.setItem(el.dataset.key, JSON.stringify({classify, q_same, features, most_similar, evidence}));
    updateExport();
  };
  el.querySelectorAll("input,select").forEach(input => {
    input.addEventListener(input.type === "text" ? "input" : "change", save);
  });
});
function updateExport() {
  const reviewer = reviewerInput.value.trim();
  const out = DATA.map(c => {
    const q = c.query;
    const s = savedFor("missed_diag_query_" + q.q_image_id);
    return {reviewer, individual: c.individual, q_no: c.q_no,
      q_image_id: q.q_image_id, q_path: q.q_path, q_series_id: q.q_series_id,
      evidence_tile: q.evidence_tile, same_image_id: q.same_image_id, same_path: q.same_path,
      top1_image_id: q.top1_image_id, top1_path: q.top1_path,
      top2_image_id: q.top2_image_id, top2_path: q.top2_path,
      top3_image_id: q.top3_image_id, top3_path: q.top3_path, rules: c.rules,
      classify: s ? (s.classify || []) : [], q_same: s ? (s.q_same || "未判断") : "未判断",
      features: s ? (s.features || "") : "", most_similar: s ? (s.most_similar || "") : "",
      evidence: s ? (s.evidence || "") : ""};
  });
  document.getElementById("export").value = JSON.stringify(out, null, 1);
}
function copyExport() {
  if (!reviewerInput.value.trim()) { alert("请先填写审核人唯一标识"); return; }
  const t = document.getElementById("export"); t.select();
  document.execCommand("copy");
  alert("已复制（如失败请手动复制文本框内容）");
}
updateExport();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
