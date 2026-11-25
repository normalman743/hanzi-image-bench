"""裁判对比：让 GPT（Codex CLI，默认 gpt-6.2-sol）判一部分送 AI 的图，和 Grok 逐张比较。
  抽样：在 grok_judge.items()（送 AI 的 1321 张）里按层（任务 × 精度 × 组）分层随机抽样，每层抽 FRAC（默认 25%），至少 MIN 张，种子固定。
  提示词、输出格式和 Grok 阶段 1 完全一样（盲读），只是图片直接附在提示词里（codex exec -i），不让 agent 自己读文件。
  结果追加到 codex/stage1.jsonl，判过的跳过，中断后重跑即可。

用法：python3 codex_judge.py sample [--frac 0.25 --min 2]   → codex/sample.txt
      python3 codex_judge.py run [--par 6 --model gpt-6.2-sol --effort medium --limit N]"""
import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grok_judge as G  # noqa: E402

OUT = os.path.join(G.OUT, "codex")
BACKEND = "codex"
SHEET = 0
RUB = False
WEEKLY_STOP = 101   # 不按额度停（用户要求不看额度）
_stop = None
lock = threading.Lock()


def layer(it):
    f = it["file"]
    q = next(x for x in ("Q8_0", "Q4_K_M", "Q2_K", "F16") if f"_{x}_" in f)
    if it["task"] == "1_single_char":
        grp = f.split("_")[1] + "_" + f.split("_")[2] if f.split("_")[1] == "pair" else f.split("_")[1] + "_" + f.split("_")[2]
        return (it["task"], q, grp)
    return (it["task"], q, "")


def sample(frac, mn, seed=42):
    strata = defaultdict(list)
    for it in G.items():
        strata[layer(it)].append(it["file"])
    rng = random.Random(seed)
    pick = []
    for k in sorted(strata):
        fs = sorted(strata[k])
        n = min(len(fs), max(mn, round(len(fs) * frac)))
        pick += rng.sample(fs, n)
    return pick, strata


QUALITY = (' Also give "quality": 2 = the glyph is a fully standard form (every component and stroke correct); '
           '1 = all components correct but with stroke-level flaws (broken, merged, mis-sized strokes) and still readable as that character; '
           '0 = a component is wrong, it is a different character, or it is not a real character.')


def prompt(task, files):
    p = G.prompt_stage1(task, files)
    p = p.replace("Reply with ONLY a JSON array", QUALITY.strip() + " For multi-character text, quality is the worst character's grade.\nReply with ONLY a JSON array")
    p = p.replace('"confidence": <0.0-1.0', '"quality": 0/1/2, "confidence": <0.0-1.0')
    # 图片是附在提示词里的，按文件名顺序
    return p.replace("For EACH image file below (in the current directory), open the image and",
                     "The images are attached in the same order as the file names listed below. For EACH attached image,")


# rubric 版（见 rubric.md）：每个字单独打 0/1/2
RUBRIC = (
    "Grade EACH character with this rubric:\n"
    "  2 = standard: instantly recognizable, every component and stroke correct, normal structure. Calligraphic/artistic style is NOT a flaw.\n"
    "  1 = flawed but acceptable: clearly recognizable as that character, all components present and none replaced; flaws are stroke-level only "
    "(broken or merged strokes, wrong stroke length/angle, an extra or missing small dot that does not turn it into another character), "
    "or components shifted/rescaled but still recognizable.\n"
    "  0 = wrong: a component is replaced (e.g. 雨 drawn as 兩), added/missing strokes make it a different or non-existent character, "
    "garbled/invented glyph, or too blurry to recognize.\n"
    "If unsure between 1 and 2 give 1; if unsure between 0 and 1 give 0 and explain in note.\n"
    "Report the character exactly as drawn: simplified and traditional forms are different characters, variant forms are not merged. "
    "Do not guess what was intended.\n"
    "confidence = 0.0-1.0, how sure you are about your judgment of the whole image; give < 0.8 if blurry or hard to decide.\n")


def prompt_rubric(task, files):
    fl = " ".join(files)
    head = "The images are attached in the same order as the file names listed below. "
    if task == "1_single_char":
        return ("You are grading AI-generated images of single Chinese characters. " + head +
                "For EACH image, read the ONE character written in it and grade it.\n" + RUBRIC + f"Files: {fl}\n"
                'Reply with ONLY a JSON array, one object per file in the same order: '
                '[{"file": "...", "read": "<character you see, or □ if not a real character>", "grade": 0/1/2, "confidence": 0.0-1.0, "note": "<very short>"}]')
    if task == "2_simple_scene":
        return ("You are grading AI-generated photos that should contain a sign with Chinese text. " + head +
                "For EACH image, read ALL Chinese characters on the sign(s) in reading order and grade each one. "
                "Garbled extra glyphs are listed as □ with g 0.\n" + RUBRIC + f"Files: {fl}\n"
                'Reply with ONLY a JSON array, one object per file in the same order: '
                '[{"file": "...", "chars": [{"c": "<char>", "g": 0/1/2}, ...], "confidence": 0.0-1.0, "note": "<very short>"}]')
    return ("You are grading AI-generated photos with several signs in a row. " + head +
            "For EACH image, read the Chinese characters on each sign from LEFT to RIGHT (one list per sign) and grade each character. "
            "Garbled extra glyphs are listed as □ with g 0.\n" + RUBRIC + f"Files: {fl}\n"
            'Reply with ONLY a JSON array, one object per file in the same order: '
            '[{"file": "...", "signs": [[{"c": "<char>", "g": 0/1/2}, ...], ...], "confidence": 0.0-1.0, "note": "<very short>"}]')


def _per_char(target, chars):
    """目标里每个字的档：按 difflib 对齐到读出的字，对上了取 GPT 给的档，漏字 0。"""
    import difflib
    t = G.norm(target)
    rc = [(G.norm(x.get("c")), x.get("g")) for x in chars if isinstance(x, dict) and G.norm(x.get("c"))]
    s = "".join(c for c, _ in rc)
    gs = [0] * len(t)
    for a, b, n in difflib.SequenceMatcher(None, t, s, autojunk=False).get_matching_blocks():
        for k in range(n):
            g = rc[b + k][1]
            gs[a + k] = g if g in (0, 1, 2) else 0
    return gs


def rubric_score(rec):
    """rubric 版判定 → (平均分 0–2, 逐字档字符串, 写对 0/1)。
    单字：读到的字等于目标才取 grade，否则 0；写对 = 分 >= 1。
    场景：目标逐字对齐取档，平均；写对 = 平均 >= 1。
    多区域：每块目标招牌找读得最像的那块，逐字取档，全部字平均；写对 = 平均 >= 1 且招牌顺序对（读到的招牌从左到右不乱、不重复）。"""
    a = rec["ai"]
    if rec["task"] == "1_single_char":
        g = a.get("grade") if G.norm(a.get("read")) == G.norm(rec["target"]) else 0
        g = g if g in (0, 1, 2) else 0
        return float(g), str(g), int(g >= 1)
    if rec["task"] == "2_simple_scene":
        gs = _per_char(rec["target"], a.get("chars") or [])
        m = sum(gs) / len(gs)
        return round(m, 3), "".join(map(str, gs)), int(m >= 1)
    signs = [s for s in a.get("signs") or [] if isinstance(s, list)]
    allg, parts, pos = [], [], []
    for t in rec["targets"]:
        cand = [(sum(g > 0 for g in _per_char(t, sg)), -k, k) for k, sg in enumerate(signs)]
        hit, _, k = max(cand) if cand else (0, 0, None)
        gs = _per_char(t, signs[k]) if hit else [0] * len(G.norm(t))
        if hit:
            pos.append(k)
        allg += gs
        parts.append("".join(map(str, gs)))
    m = sum(allg) / len(allg)
    order_ok = pos == sorted(pos) and len(set(pos)) == len(pos)
    return round(m, 3), "|".join(parts), int(m >= 1 and order_ok)


SHEET_DIR = os.path.join(G.OUT, "sheets")


def make_sheet(task, files, cell=300, cols=5):
    """把单字图的中间一半裁出来缩到 cell 像素，拼成带编号（#1…）的网格图，返回文件名。"""
    import uuid
    from PIL import Image, ImageDraw, ImageFont
    os.makedirs(SHEET_DIR, exist_ok=True)
    rows = (len(files) + cols - 1) // cols
    lab = 34
    sheet = Image.new("RGB", (cols * cell, rows * (cell + lab)), "white")
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 26)
    for i, f in enumerate(files):
        im = Image.open(os.path.join(G.RES, task, f)).convert("RGB")
        c = im.crop((im.width // 4, im.height // 4, im.width * 3 // 4, im.height * 3 // 4)).resize((cell - 6, cell - 6))
        x, y = (i % cols) * cell, (i // cols) * (cell + lab)
        d.text((x + 6, y + 2), f"#{i + 1}", fill="red", font=font)
        sheet.paste(c, (x + 3, y + lab))
        d.rectangle([x, y, x + cell - 1, y + cell + lab - 1], outline=(180, 180, 180))
    name = f"sheet_{uuid.uuid4().hex[:10]}.png"
    sheet.save(os.path.join(SHEET_DIR, name))
    return name


def prompt_sheet(name, n, attached):
    where = "The attached image" if attached else f"The image file {name} in the current directory (open it)"
    return ((("" if attached else "Do NOT write plans or create/modify any files. Just view the image directly and answer. ")) +
            f"You are grading AI-generated images of single Chinese characters. {where} is a grid of {n} numbered cells "
            f"(#1 to #{n}, left to right, top to bottom; the red number is the cell label, not part of the character). "
            "Each cell shows ONE character. For EACH cell, read the character written in it. Do not guess what was intended: report exactly "
            "the glyph that is drawn, distinguishing simplified vs traditional forms. If the glyph is not a real standard character "
            "(missing/extra strokes, invented, garbled), say so." + QUALITY +
            '\nReply with ONLY a JSON array, one object per cell in order: [{"cell": 1, "read": "<character you see, or empty>", '
            '"real_char": true/false, "quality": 0/1/2, "confidence": <0.0-1.0, how sure you are about this judgment>, "note": "<very short>"}]')


def prompt_files(task, files):
    """Gemini（agy）版：图片不附在提示词里，让 agent 读当前目录的文件。"""
    return ("Do NOT write plans or create/modify any files. Just view the images directly and answer. "
            + prompt(task, files).replace("The images are attached in the same order as the file names listed below. For EACH attached image,",
                                          "For EACH image file below (in the current directory), open the image and"))


def call_agy(model, cwd, pr, tries=3):
    for k in range(tries):
        t0 = time.time()
        p = subprocess.run(["agy", "--model", model, "--mode", "auto", "--sandbox", "--output-format", "json", "--print", pr],
                           cwd=cwd, capture_output=True, text=True, timeout=3600, stdin=subprocess.DEVNULL)
        try:
            d = json.loads(p.stdout.strip().splitlines()[-1])
            txt = d["response"]
            arr = json.loads(txt[txt.find("["): txt.rfind("]") + 1])
            return arr, {"sec": round(time.time() - t0), "usage": d.get("usage")}
        except Exception as e:
            err = f"{e!r} | stdout={p.stdout[-300:]!r} | stderr={p.stderr[-300:]!r}"
            if k == tries - 1:
                return None, {"sec": round(time.time() - t0), "error": err}
            time.sleep(10)


def call(model, effort, cwd, files, pr, tries=3):
    for k in range(tries):
        t0 = time.time()
        with tempfile.NamedTemporaryFile("r", suffix=".txt", delete=False) as tf:
            last = tf.name
        cmd = ["codex", "exec", "--json", "--skip-git-repo-check", "--ephemeral", "--sandbox", "read-only", "-m", model,
               "-c", f'model_reasoning_effort="{effort}"', "-C", cwd, "-o", last]
        for f in files:
            cmd += ["-i", os.path.join(cwd, f)]
        cmd += ["--", pr]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, stdin=subprocess.DEVNULL)
        usage, rl = None, None
        for line in p.stdout.splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            u = ev.get("usage") or (ev.get("msg") or {}).get("info", {}) or {}
            if isinstance(ev.get("usage"), dict):
                usage = ev["usage"]
            r = ev.get("rate_limits") or (ev.get("msg") or {}).get("rate_limits")
            if isinstance(r, dict):
                rl = r
        try:
            txt = open(last, encoding="utf-8").read()
            arr = json.loads(txt[txt.find("["): txt.rfind("]") + 1])
            meta = {"sec": round(time.time() - t0), "usage": usage}
            if rl:
                meta["rate_limits"] = {k: (rl.get(k) or {}).get("used_percent") for k in ("primary", "secondary")}
            return arr, meta
        except Exception as e:
            err = f"{e!r} | rc={p.returncode} | stderr={p.stderr[-400:]!r}"
            if k == tries - 1:
                return None, {"sec": round(time.time() - t0), "error": err}
            time.sleep(10)
        finally:
            if os.path.exists(last):
                os.remove(last)


def run(par, model, effort, limit, files=None, out_name="stage1.jsonl"):
    keep = [l.strip() for l in open(files or os.path.join(OUT, "sample.txt"), encoding="utf-8") if l.strip()]
    keep = set(keep)
    its = [it for it in G.items() if it["file"] in keep]
    path = os.path.join(OUT, out_name)
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["file"] for l in open(path, encoding="utf-8") if '"error"' not in l}
    its = [it for it in its if it["file"] not in done]
    if limit:
        its = its[:limit]
    batches = []
    for task, bs in G.BATCH.items():
        if SHEET and task == "1_single_char":
            bs = SHEET
        ts = [it for it in its if it["task"] == task]
        batches += [(task, ts[i:i + bs]) for i in range(0, len(ts), bs)]
    print(f"{BACKEND}：{len(its)} 张待判，{len(batches)} 次调用，{par} 路并行，模型 {model}（effort {effort}）", flush=True)
    t0, n = time.time(), [0]

    def one(tb):
        task, batch = tb
        files = [it["file"] for it in batch]
        if SHEET and task == "1_single_char":
            name = make_sheet(task, files)
            if BACKEND == "agy":
                arr, meta = call_agy(model, SHEET_DIR, prompt_sheet(name, len(files), False))
            else:
                arr, meta = call(model, effort, SHEET_DIR, [name], prompt_sheet(name, len(files), True))
            meta["sheet"] = name
            arr = [{**o, "file": files[int(o["cell"]) - 1]} for o in (arr or [])
                   if isinstance(o, dict) and str(o.get("cell", "")).isdigit() and 1 <= int(o["cell"]) <= len(files)]
        elif BACKEND == "agy":
            arr, meta = call_agy(model, os.path.join(G.RES, task), prompt_files(task, files))
        else:
            arr, meta = call(model, effort, os.path.join(G.RES, task), files, (prompt_rubric if RUB else prompt)(task, files))
        by = {o.get("file"): o for o in (arr or []) if isinstance(o, dict)}
        with lock:
            with open(path, "a", encoding="utf-8") as f:
                for it in batch:
                    rec = {**it, "stage": BACKEND, "model": model, "effort": effort, **meta}
                    if it["file"] in by:
                        rec["ai"] = by[it["file"]]
                    else:
                        rec["error"] = meta.get("error", "回复里没有这张图")
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n[0] += 1
            el = time.time() - t0
            rl = meta.get("rate_limits") or {}
            u = meta.get("usage") or {}
            print(f"  {n[0]}/{len(batches)} 次  {el / 60:.0f} 分钟  本次 {meta['sec']}s  tokens 入{u.get('input_tokens')} 出{u.get('output_tokens')}  "
                  f"额度 5h={rl.get('primary')}% 周={rl.get('secondary')}% {'出错 ' + meta['error'][:120] if 'error' in meta else ''}", flush=True)
            if isinstance(rl.get("secondary"), (int, float)) and rl["secondary"] >= WEEKLY_STOP:
                _stop.set()

    stop = threading.Event()

    def guarded(tb):
        if stop.is_set():
            return
        one(tb)

    global _stop
    _stop = stop
    with ThreadPoolExecutor(par) as ex:
        list(ex.map(guarded, batches))
    if stop.is_set():
        print(f"每周额度超过 {WEEKLY_STOP}%，已停止；剩下的批次没有判。", flush=True)
    errs = sum(1 for l in open(path, encoding="utf-8") if '"error"' in l)
    print(f"{BACKEND} 完成。文件里出错记录 {errs} 条（重跑会补判）", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "run"])
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--min", type=int, default=2)
    ap.add_argument("--par", type=int, default=6)
    ap.add_argument("--model", default="gpt-6.2-sol")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--files", default=None, help="只判这个清单（默认 codex/sample.txt）")
    ap.add_argument("--out_name", default="stage1.jsonl")
    ap.add_argument("--backend", default="codex", choices=["codex", "agy"], help="agy = Gemini（Antigravity CLI）")
    ap.add_argument("--out_dir", default=None, help="结果目录（默认 codex/；Gemini 建议 gemini/）")
    ap.add_argument("--rubric", action="store_true", help="用 rubric.md 的逐字三档提示词")
    ap.add_argument("--sheet", type=int, default=0, help="单字题拼图：每张拼图放 N 个字（0 = 不拼，逐张读文件）")
    a = ap.parse_args()
    global OUT, BACKEND, SHEET, RUB
    RUB = a.rubric
    BACKEND = a.backend
    SHEET = a.sheet
    if a.out_dir:
        OUT = a.out_dir
    os.makedirs(OUT, exist_ok=True)
    if a.cmd == "sample":
        pick, strata = sample(a.frac, a.min)
        open(os.path.join(OUT, "sample.txt"), "w", encoding="utf-8").write("\n".join(pick) + "\n")
        by_task = defaultdict(int)
        for f in pick:
            by_task[next(t for t in G.BATCH if os.path.exists(os.path.join(G.RES, t, f)))] += 1
        print(f"{len(strata)} 层，共 {sum(len(v) for v in strata.values())} 张 → 抽 {len(pick)} 张 {dict(by_task)}")
    else:
        run(a.par, a.model, a.effort, a.limit, a.files, a.out_name)


if __name__ == "__main__":
    main()
