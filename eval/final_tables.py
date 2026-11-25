"""最终判定（任务 1–3）：
  没送 AI 的图 → OCR 判定（第 1 级）；送 AI 的图 → GPT（gpt-6.2-sol medium）判定。人工判过的图只用来算一致率（USE_HUMAN=False）。
  --rubric（定稿用）：GPT 按 rubric.md 逐字打 0/1/2，规则见 codex_judge.rubric_score（逐字平均 >= 1 算写对），读 codex/rubric.jsonl。
  不加 --rubric（旧版）：单字 = 盲读等于目标且是标准字；场景 / 多区域 = 文字读对且质量 >= 1，读 codex/stage1.jsonl。
输出 final/final_<任务>.csv（逐张）和 final/summary_final.csv（任务 × 精度 × 分组，和纯 OCR 对照）。"""
import csv, json, os, sys, collections, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grok_judge as G  # noqa: E402
import codex_judge as C  # noqa: E402

H = G.OUT
HUMAN = os.environ.get("HB_HUMAN", os.path.expanduser("~/Downloads/review_gpt40.csv"))
RUBRIC = "--rubric" in sys.argv
USE_HUMAN = False   # 用户决定：最终统一用 GPT 的标准，人工 40 张只用来算一致率


def gpt_ok(r):
    a = r["ai"]
    if r["task"] == "1_single_char":
        return int(G.stage1_match(r))
    if r["task"] == "2_simple_scene":
        hit = G.norm(r["target"]) in G.norm(a.get("text"))
    else:
        signs = [G.norm(s) for s in a.get("signs") or []]
        pos = [next((k for k, s in enumerate(signs) if G.norm(t) in s), None) for t in r["targets"]]
        hit = None not in pos and pos == sorted(pos)
    return int(bool(hit) and (a.get("quality") or 0) >= 1)


def main():
    src_file = os.path.join(C.OUT, "rubric.jsonl" if RUBRIC else "stage1.jsonl")
    gpt = {json.loads(l)["file"]: json.loads(l) for l in open(src_file, encoding="utf-8") if '"ai"' in l}
    hum = {r["file"]: int(r["人工三档"]) for r in csv.DictReader(open(HUMAN, encoding="utf-8-sig")) if r["人工三档"]} if USE_HUMAN else {}
    sent = {it["file"] for it in G.items()}

    def rd(n):
        return list(csv.DictReader(open(f"{G.T1DIR}/score_{n}.tsv", encoding="utf-8"), delimiter="\t"))
    spec = {"1_single_char": ("写对", lambda r: r["组"]), "2_simple_scene": ("逐字正确", lambda r: r["文字类别"]),
            "3_multi_region": ("全对且位置对", lambda r: f"{r['区域数']}区")}
    os.makedirs(f"{H}/final", exist_ok=True)
    summ, out = [], {}
    for task, (col, grp) in spec.items():
        rows = []
        for r in rd(task):
            f = r["file"]
            ocr = int(r[col])
            score, detail = "", ""
            if USE_HUMAN and f in hum:
                fin, src = int(hum[f] >= 1), "人工"
            elif f in sent:
                if f not in gpt:
                    raise SystemExit(f"{src_file} 里缺 {f}（送 AI 的图没判完）")
                if RUBRIC:
                    score, detail, fin = C.rubric_score(gpt[f])
                else:
                    fin = gpt_ok(gpt[f])
                src = "GPT"
            else:
                fin, src = ocr, "OCR"
            rows.append({"file": f, "精度": r["精度"], "分组": grp(r), "目标": r.get("字") or r.get("目标") or r.get("题号"),
                         "OCR判定": ocr, "GPT平均分": score, "GPT逐字档": detail, "最终判定": fin, "判定来源": src})
        out[task] = rows
        with open(f"{H}/final/final_{task}{'_rubric' if RUBRIC else ''}.csv", "w", encoding="utf-8-sig", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        g = collections.defaultdict(list)
        for x in rows:
            g[(x["精度"], "全部")].append(x)
            g[(x["精度"], x["分组"])].append(x)
        for (q, gg), xs in sorted(g.items()):
            summ.append({"任务": task, "精度": q, "分组": gg, "张数": len(xs),
                         "最终写对率": round(sum(x["最终判定"] for x in xs) / len(xs), 3),
                         "纯OCR写对率": round(sum(x["OCR判定"] for x in xs) / len(xs), 3),
                         "来源OCR/GPT/人工": "/".join(str(sum(x["判定来源"] == s for x in xs)) for s in ("OCR", "GPT", "人工"))})
    with open(f"{H}/final/summary_final{'_rubric' if RUBRIC else ''}.csv", "w", encoding="utf-8-sig", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    # 简繁配对（McNemar 精确检验）
    t1 = {x["file"]: x for x in out["1_single_char"]}
    pidx = [l.rstrip("\n").split("\t") for l in open(f"{G.REPO}/chars/pairs_index.tsv", encoding="utf-8") if not l.startswith("#")] \
        if os.path.exists(f"{G.REPO}/chars/pairs_index.tsv") else []
    return summ, out, pidx


if __name__ == "__main__":
    summ, out, pidx = main()
    Q = ["F16", "Q8_0", "Q4_K_M", "Q2_K"]
    tab = {(s["任务"], s["分组"], s["精度"]): s for s in summ}
    for task in ("1_single_char", "2_simple_scene", "3_multi_region"):
        groups = sorted({s["分组"] for s in summ if s["任务"] == task}, key=lambda g: (g != "全部", g))
        print(f"\n== {task}  最终写对率（括号内为纯 OCR）")
        for g in groups:
            print(f"  {g:14}" + "".join(f"  {tab[(task, g, q)]['最终写对率']:.2f}({tab[(task, g, q)]['纯OCR写对率']:.2f})" for q in Q))
