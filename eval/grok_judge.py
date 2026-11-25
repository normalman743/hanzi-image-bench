"""第 2–3 级（AI）判分：用 Cursor Agent 里的 Grok 看图。
  第 2 级 grok-4.7-medium 盲读全部送 AI 的图；第 3 级 grok-4.7-high 重读 medium 把握分数 < 0.8 的图（python3 grok_judge.py high）。
  high 仍 < 0.8 的交给 Claude（第 4 级），Claude 把握 < 0.8 的交给人工（第 5 级）。
  阶段 1 盲读：不告诉目标，读出图里写的字，并判断是不是标准字（笔画缺 / 多 / 错算不是）。
    AI 判对 = 读出的字等于目标，且是标准字。（低档 low-fast 太宽松，看不出笔画错，所以全部用 medium。）
  阶段 2（可选）逐笔画核对：只对阶段 1 判对的图，告诉目标，问是否一笔不差。
送 AI 的图（第 1 级用朋友的 PP-OCRv5 结果 7_ocr_scores）：
  任务 1：OCR 判错、或字典外、或 3/4 级字、或 OCR 判对但置信度 < 0.9
  任务 2：OCR 逐字判错；任务 3：OCR 判"全对且位置对"为否
原始回复全部追加到 grok/<阶段>.jsonl，已判过的跳过，中断后重跑即可。

用法：python3 grok_judge.py stage1 [--par 6]；python3 grok_judge.py stage2 [--par 6]；python3 grok_judge.py merge"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import threading
import time
import unicodedata
import zlib
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def need(name):
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"请设置环境变量 {name}（见 eval/README.md）")
    return v


# 路径用环境变量给：HB_RES = 含 1_single_char/ 2_simple_scene/ 3_multi_region/ 的结果目录；
# HB_T1DIR = 7_ocr_scores 目录（第 1 级 OCR 结果）；HB_REPO = 本仓库根目录；HB_OUT = 结果输出目录（默认 eval/out）
RES = need("HB_RES")
T1DIR = need("HB_T1DIR")
REPO = os.environ.get("HB_REPO", os.path.dirname(HERE))
OUT = os.environ.get("HB_OUT", os.path.join(HERE, "out"))
MODEL = {"stage1": "grok-4.7-medium", "high": "grok-4.7-high", "stage2": "grok-4.7-medium"}
BATCH = {"1_single_char": 10, "2_simple_scene": 5, "3_multi_region": 5}
lock = threading.Lock()
AI_CONF = 0.8   # Grok 自报的把握分数低于这个就算"没把握"


def norm(s):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s or ""))


def items():
    """送 AI 的图：[{task, file, target(单字/词) 或 targets(多区域，从左到右)}]"""
    conf = {}
    for p in glob.glob(f"{T1DIR}/ocr_g*.jsonl"):
        for l in open(p, encoding="utf-8"):
            d = json.loads(l)
            ss = [x["score"] for x in d["lines"]] + ([d["crop"]["score"]] if "crop" in d else [])
            conf[d["file"].split("/")[-1]] = min(ss) if ss else 0.0

    def rd(n):
        return list(csv.DictReader(open(f"{T1DIR}/score_{n}.tsv", encoding="utf-8"), delimiter="\t"))
    out = []
    for r in rd("1_single_char"):
        lv = int(r["组"].split("_L")[1][0])
        why = []
        if r["写对"] == "0":
            why.append("OCR判错")
        if r["OCR字典内"] == "0":
            why.append("字典外")
        if lv >= 3:
            why.append("3/4级")
        if r["写对"] == "1" and conf[r["file"]] < 0.9:
            why.append("置信度低")
        if why:
            out.append({"task": "1_single_char", "file": r["file"], "target": r["字"], "ocr_ok": int(r["写对"]), "why": "；".join(why)})
    for r in rd("2_simple_scene"):
        if r["逐字正确"] == "0":
            out.append({"task": "2_simple_scene", "file": r["file"], "target": r["目标"], "ocr_ok": 0, "why": "OCR判错"})
    bank = {r["题号"]: r for r in csv.DictReader(open(f"{REPO}/data/multi-region-text/eval.csv", encoding="utf-8-sig"))}
    for r in rd("3_multi_region"):
        if r["全对且位置对"] == "0":
            regs = sorted(json.loads(bank[r["题号"]]["区域标注"]), key=lambda x: x["位置"])
            out.append({"task": "3_multi_region", "file": r["file"], "targets": [x["目标文字"] for x in regs], "ocr_ok": 0, "why": "OCR判错"})
    return out


def prompt_stage1(task, files):
    fl = " ".join(files)
    if task == "1_single_char":
        return ("You are grading AI-generated images of single Chinese characters. For EACH image file below (in the current directory), "
                "open the image and read the ONE character written in it. Do not guess what was intended: report exactly the glyph that is drawn, "
                "distinguishing simplified vs traditional forms. If the glyph is not a real standard character (missing/extra strokes, invented, garbled), say so. "
                f"Files: {fl}\nReply with ONLY a JSON array, one object per file in the same order: "
                '[{"file": "...", "read": "<character you see, or empty>", "real_char": true/false, "confidence": <0.0-1.0, how sure you are about this judgment>, "note": "<very short>"}]')
    if task == "2_simple_scene":
        return ("You are grading AI-generated photos that should contain a sign with Chinese text. For EACH image file below (in the current directory), "
                "open the image and read ALL legible text on the sign(s), exactly as drawn (simplified vs traditional as drawn). Do not guess what was intended. "
                "If any glyph is not a real standard character, mark it as ? in the text and set malformed true. "
                f"Files: {fl}\nReply with ONLY a JSON array, one object per file in the same order: "
                '[{"file": "...", "text": "<all sign text>", "malformed": true/false, "confidence": <0.0-1.0, how sure you are about this judgment>, "note": "<very short>"}]')
    return ("You are grading AI-generated photos with several signs in a row. For EACH image file below (in the current directory), open the image and read the text "
            "on each sign from LEFT to RIGHT, exactly as drawn (simplified vs traditional as drawn). Do not guess what was intended. If a glyph is not a real standard "
            "character, write ? for it. "
            f"Files: {fl}\nReply with ONLY a JSON array, one object per file in the same order: "
            '[{"file": "...", "signs": ["<sign 1 text>", "<sign 2 text>", ...], "malformed": true/false, "confidence": <0.0-1.0, how sure you are about this judgment>, "note": "<very short>"}]')


def prompt_stage2(task, batch):
    lines = "\n".join(f"- {it['file']}: should be {'「' + it['target'] + '」' if 'target' in it else ' | '.join('「' + t + '」' for t in it['targets']) + ' (left to right)'}"
                      for it in batch)
    return ("You are strictly checking AI-generated images of Chinese text. For EACH image file below (in the current directory), open the image and check, "
            "stroke by stroke and component by component, whether the drawn text is EXACTLY the expected text (same simplified/traditional form). "
            "Any missing, extra, broken or wrong stroke/component counts as NOT exact.\n" + lines +
            '\nReply with ONLY a JSON array, one object per file in the same order: [{"file": "...", "exact": true/false, '
            '"error_type": "none|wrong_char|simp_trad_swap|missing_stroke|extra_stroke|wrong_component|garbled|missing_text|other", '
            '"confidence": <0.0-1.0, how sure you are about this judgment>, "note": "<very short>"}]')


def call(model, cwd, prompt, tries=3):
    for k in range(tries):
        t0 = time.time()
        p = subprocess.run(["cursor-agent", "-p", "--trust", "--mode", "ask", "--model", model, "--output-format", "json",
                            "--workspace", cwd, prompt], cwd=cwd, capture_output=True, text=True, timeout=3600)
        try:
            d = json.loads(p.stdout.strip().splitlines()[-1])
            txt = d["result"]
            arr = json.loads(txt[txt.find("["): txt.rfind("]") + 1])
            return arr, {"sec": round(time.time() - t0), "usage": d.get("usage")}
        except Exception as e:
            err = f"{e!r} | stdout={p.stdout[-300:]!r} | stderr={p.stderr[-300:]!r}"
            if k == tries - 1:
                return None, {"sec": round(time.time() - t0), "error": err}
            time.sleep(10)


def run_stage(stage, todo, par):
    path = os.path.join(OUT, f"{stage}.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["file"] for l in open(path, encoding="utf-8") if '"error"' not in l}
    todo = [it for it in todo if it["file"] not in done]
    batches = []
    for task in BATCH:
        ts = [it for it in todo if it["task"] == task]
        batches += [(task, ts[i:i + BATCH[task]]) for i in range(0, len(ts), BATCH[task])]
    print(f"{stage}：{len(todo)} 张待判，{len(batches)} 次调用，{par} 路并行，模型 {MODEL[stage]}", flush=True)
    t0, n = time.time(), [0]

    def one(tb):
        task, batch = tb
        files = [it["file"] for it in batch]
        pr = prompt_stage2(task, batch) if stage == "stage2" else prompt_stage1(task, files)
        arr, meta = call(MODEL[stage], os.path.join(RES, task), pr)
        by = {}
        if arr:
            for o in arr:
                if isinstance(o, dict) and o.get("file") in files:
                    by[o["file"]] = o
        with lock:
            with open(path, "a", encoding="utf-8") as f:
                for it in batch:
                    rec = {**it, "stage": stage, "model": MODEL[stage], **meta}
                    if it["file"] in by:
                        rec["ai"] = by[it["file"]]
                    else:
                        rec["error"] = meta.get("error", "回复里没有这张图")
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n[0] += 1
            el = time.time() - t0
            print(f"  {n[0]}/{len(batches)} 次  {el / 60:.0f} 分钟  剩余约 {el / n[0] * (len(batches) - n[0]) / 60:.0f} 分钟  "
                  f"本次 {meta['sec']}s {'出错' if 'error' in meta else ''}", flush=True)

    with ThreadPoolExecutor(par) as ex:
        list(ex.map(one, batches))
    errs = sum(1 for l in open(path, encoding="utf-8") if '"error"' in l)
    print(f"{stage} 完成。文件里出错记录 {errs} 条（重跑会补判）", flush=True)


def stage1_match(rec):
    a = rec["ai"]
    if rec["task"] == "1_single_char":
        return norm(a.get("read")) == norm(rec["target"]) and a.get("real_char") is True
    if rec["task"] == "2_simple_scene":
        return norm(rec["target"]) in norm(a.get("text")) and a.get("malformed") is not True
    signs = [norm(s) for s in a.get("signs") or []]
    pos = [next((k for k, s in enumerate(signs) if norm(t) in s), None) for t in rec["targets"]]
    return None not in pos and pos == sorted(pos) and len(set(pos)) == len(pos) and a.get("malformed") is not True


def latest(stage):
    path = os.path.join(OUT, f"{stage}.jsonl")
    d = {}
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            if "error" not in r:
                d[r["file"]] = r
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["count", "stage1", "high", "stage2", "merge"])
    ap.add_argument("--par", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="只判前 N 张（试跑用）")
    ap.add_argument("--shard", default="0/1", help="k/n：只判按文件名固定分到第 k 份（共 n 份）的图，例如 0/2、1/2")
    ap.add_argument("--files", default=None, help="只判这个清单里的图（每行一个文件名），例如 grok/shard_teammate.txt")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    k, n = (int(x) for x in a.shard.split("/"))
    its = items()
    if a.cmd in ("stage1", "high", "stage2") and n > 1:
        its = [it for it in its if zlib.crc32(it["file"].encode()) % n == k]
        print(f"分片 {k}/{n}：{len(its)} 张", flush=True)
    if a.files and a.cmd in ("stage1", "high", "stage2"):
        keep = {l.strip() for l in open(a.files, encoding="utf-8") if l.strip()}
        its = [it for it in its if it["file"] in keep]
        print(f"清单 {a.files}：{len(its)} 张", flush=True)
    if a.cmd == "count":
        from collections import Counter
        print(Counter(it["task"] for it in its))
        return
    if a.cmd == "stage1":
        run_stage("stage1", its[: a.limit] if a.limit else its, a.par)
    elif a.cmd == "high":
        # medium 把握分数 < AI_CONF（或没给分数）的图，用 grok-4.7-high 重新盲读（同一提示词）
        s1 = latest("stage1")
        low = {f for f, r in s1.items() if not isinstance(r["ai"].get("confidence"), (int, float)) or r["ai"]["confidence"] < AI_CONF}
        todo = [it for it in its if it["file"] in low]
        run_stage("high", todo[: a.limit] if a.limit else todo, a.par)
    elif a.cmd == "stage2":
        s1 = latest("stage1")
        todo = [s1[f] for f in s1 if stage1_match(s1[f])]
        todo = [{k: r[k] for k in ("task", "file", "target", "targets", "ocr_ok", "why") if k in r} for r in todo]
        run_stage("stage2", todo[: a.limit] if a.limit else todo, a.par)
    else:
        s1, s2 = latest("stage1"), latest("stage2")
        rows = []
        for it in its:
            f = it["file"]
            r1 = s1.get(f)
            if r1 is None:
                rows.append({**it, "AI判定": "", "AI来源": "未判"})
                continue
            m = stage1_match(r1)
            a1 = r1["ai"]
            if not m:
                ai_ok, src, conf = 0, "stage1", a1.get("confidence")
            elif f in s2:
                ai_ok, src, conf = int(bool(s2[f]["ai"].get("exact"))), "stage2", s2[f]["ai"].get("confidence")
            else:
                ai_ok, src, conf = 1, "stage1", a1.get("confidence")
            rows.append({"task": it["task"], "file": f, "目标": it.get("target") or " | ".join(it["targets"]), "送AI原因": it["why"],
                         "OCR判定": it["ocr_ok"], "AI盲读": a1.get("read") or a1.get("text") or " | ".join(a1.get("signs") or []),
                         "AI判定": ai_ok, "AI来源": src, "AI把握分数": conf,
                         "AI错误类型": s2[f]["ai"].get("error_type") if f in s2 else "",
                         "AI备注": (s2[f]["ai"].get("note") if f in s2 else a1.get("note")) or ""})
        with open(os.path.join(OUT, "tier2.csv"), "w", encoding="utf-8-sig", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=list(rows[0]) if "AI盲读" in rows[0] else
                               ["task", "file", "目标", "送AI原因", "OCR判定", "AI盲读", "AI判定", "AI来源", "AI把握分数", "AI错误类型", "AI备注"],
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"写出 {len(rows)} 行 → grok/tier2.csv")


if __name__ == "__main__":
    main()
