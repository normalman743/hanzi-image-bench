"""rubric 版 GPT vs 人工三档 vs 旧版 GPT（同一批人工看过的图）。rubric 版用 codex_judge.rubric_score：
逐字档取平均，平均 >= 1 算写对。比较"对错"（人工 >= 1 算对）。"""
import csv
import difflib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grok_judge as G  # noqa: E402
import codex_judge as C  # noqa: E402

HUMAN = os.environ.get("HB_HUMAN", os.path.expanduser("~/Downloads/review_gpt40.csv"))
NEW = os.environ.get("HB_RUBRIC", os.path.join(G.OUT, "rubric_test", "gpt_rubric.jsonl"))
OLD = os.environ.get("HB_OLD", os.path.join(G.OUT, "codex", "stage1.jsonl"))


def old_grade(rec):
    return rec["ai"].get("quality", 0) if G.stage1_match(rec) else 0


def main():
    hum = {r["file"]: (int(r["人工三档"]), r["备注"]) for r in csv.DictReader(open(HUMAN, encoding="utf-8-sig"))}
    new = {json.loads(l)["file"]: json.loads(l) for l in open(NEW, encoding="utf-8") if '"error"' not in l}
    old = {}
    for l in open(OLD, encoding="utf-8"):
        r = json.loads(l)
        if "error" not in r and r["file"] in hum:
            old[r["file"]] = r
    rows, agree = [], {"新对错": 0, "旧对错": 0}
    for f, (h, note) in hum.items():
        ng, detail, nok = C.rubric_score(new[f])
        og = old_grade(old[f])
        agree["新对错"] += nok == (h >= 1)
        agree["旧对错"] += (og >= 1) == (h >= 1)
        a = new[f]["ai"]
        rows.append({"file": f, "目标": new[f].get("target") or " | ".join(new[f].get("targets", [])), "人工": h, "新GPT平均分": ng, "新GPT写对": nok, "旧GPT": og,
                     "新读到": a.get("read") or "".join(x.get("c", "") for x in a.get("chars") or []) or
                     " | ".join("".join(x.get("c", "") for x in s) for s in a.get("signs") or []),
                     "逐字档": detail, "新把握": a.get("confidence"), "新备注": a.get("note"), "人工备注": note})
    n = len(rows)
    print(f"与人工对错一致（{n} 张）：" + "  ".join(f"{k} {v}/{n}" for k, v in agree.items()))
    out = os.path.join(os.path.dirname(NEW), "compare.csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["新GPT写对"] != (r["人工"] >= 1):
            print(f'{r["file"]}  目标{r["目标"]}  人{r["人工"]} 新{r["新GPT平均分"]:.2f} 旧{r["旧GPT"]}  读到 {r["新读到"]} {r["逐字档"]}  把握{r["新把握"]}  GPT:{r["新备注"]}  人:{r["人工备注"]}')
    print("→", out)


if __name__ == "__main__":
    main()
