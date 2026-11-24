"""按题库 CSV 里的完整提示词批量生成图片（简单场景 / 多区域 / 整页长文）。
每张卡一个工作线程，按行号分题；每道题依次跑完所有精度和种子。已有的图跳过，中断后重跑同一条命令即可。

用法：python3 scripts/gen_csv.py data/simple-scene-text/eval.csv
      python3 scripts/gen_csv.py data/full-page-text/eval.csv --quants "Q8_0 Q2_K" --seeds "42 43"

画面尺寸默认按题库目录名取（建议值，可用 --size 改）：
  simple-scene-text  2048x2048（1:1，一块牌子）
  multi-region-text  2752x1536（16:9，最多 5 块牌子横排）
  full-page-text     1792x2400（3:4，竖版打印页）
都在 Qwen-Image-2.1 官方支持的尺寸里。其他题库必须用 --size 指定。"""
import argparse
import csv
import os
import subprocess
import sys
import threading
import time

DEFAULT_SIZE = {"simple-scene-text": "2048x2048", "multi-region-text": "2752x1536", "full-page-text": "1792x2400"}


def gpu_count():
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    return len([l for l in out.splitlines() if l.strip()])


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default=None, help="默认 out_<题库目录名>/")
    ap.add_argument("--id_col", default="题号")
    ap.add_argument("--prompt_col", default="提示词")
    ap.add_argument("--quants", default="Q8_0 Q4_K_M Q2_K")
    ap.add_argument("--seeds", default="42", help="同一组种子用于所有题，对照题因此种子相同")
    ap.add_argument("--size", default=None, help="宽x高，例如 2048x2048")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--cfg", default="1.0")
    ap.add_argument("--ngpu", type=int, default=None, help="默认用所有显卡")
    ap.add_argument("--models", default=os.path.join(root, "models"))
    ap.add_argument("--sd_bin", default=os.path.join(root, "stable-diffusion.cpp/build/bin/sd-cli"))
    ap.add_argument("--extra", default="", help="额外传给 sd-cli 的参数，例如 --vae-tiling")
    a = ap.parse_args()

    bank = os.path.basename(os.path.dirname(os.path.abspath(a.csv)))
    size = a.size or DEFAULT_SIZE.get(bank)
    if not size:
        raise SystemExit(f"题库 {bank} 没有默认尺寸，请用 --size 指定，例如 --size 2048x2048")
    w, h = size.lower().split("x")
    out = a.out or os.path.join(root, f"out_{bank}")
    quants, seeds = a.quants.split(), a.seeds.split()
    ngpu = a.ngpu or gpu_count()

    if not os.access(a.sd_bin, os.X_OK):
        raise SystemExit(f"找不到 sd-cli：{a.sd_bin}（先跑 scripts/setup.sh）")
    for q in quants:
        p = os.path.join(a.models, f"qwen-image-2.1-{q}.gguf")
        if not os.path.exists(p):
            raise SystemExit(f"缺模型 {p}")
    with open(a.csv, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for col in (a.id_col, a.prompt_col):
        if col not in rows[0]:
            raise SystemExit(f"{a.csv} 没有列 {col!r}，实际列：{list(rows[0])}")
    empty = [r[a.id_col] for r in rows if not r[a.prompt_col].strip()]
    if empty:
        raise SystemExit(f"{len(empty)} 道题提示词为空，第一个：{empty[0]}")
    ids = [r[a.id_col] for r in rows]
    if len(set(ids)) != len(ids):
        dup = next(i for i in ids if ids.count(i) > 1)
        raise SystemExit(f"题号重复：{dup}")

    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    print(f"{a.csv}：{len(rows)} 题，{ngpu} 张卡，精度 {quants}，种子 {seeds}，{w}x{h}，{a.steps} 步，cfg {a.cfg}，输出 {out}", flush=True)
    lock = threading.Lock()

    def worker(g):
        man = os.path.join(out, f"manifest_g{g}.tsv")
        for i, r in enumerate(rows):
            if i % ngpu != g:
                continue
            for q in quants:
                for s in seeds:
                    name = f"{r[a.id_col]}_{q}_s{s}"
                    png = os.path.join(out, name + ".png")
                    if os.path.exists(png):
                        continue
                    cmd = [a.sd_bin, "--diffusion-model", os.path.join(a.models, f"qwen-image-2.1-{q}.gguf"),
                           "--vae", os.path.join(a.models, "qwen_image_2.1_vae_bf16.safetensors"),
                           "--llm", os.path.join(a.models, "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf"),
                           "-p", r[a.prompt_col], "--steps", str(a.steps), "--cfg-scale", a.cfg,
                           "--sampling-method", "euler", "-W", w, "-H", h, "--diffusion-fa", "--seed", s,
                           *a.extra.split(), "-o", png]
                    t0 = time.time()
                    with open(os.path.join(out, "logs", name + ".log"), "w") as lf:
                        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}).returncode
                    st = "OK" if rc == 0 and os.path.exists(png) else "FAIL"
                    line = f"{i + 1}\t{r[a.id_col]}\t{q}\t{s}\t{name}.png\t{time.time() - t0:.0f}s\t{st}"
                    with lock:
                        print(line, flush=True)
                        with open(man, "a", encoding="utf-8") as f:
                            f.write(line + "\n")

    ts = [threading.Thread(target=worker, args=(g,)) for g in range(ngpu)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    latest = {}
    for g in range(ngpu):
        p = os.path.join(out, f"manifest_g{g}.tsv")
        if os.path.exists(p):
            for l in open(p, encoding="utf-8"):
                f = l.rstrip("\n").split("\t")
                latest[f[4]] = f                     # 同一张图重跑过的，只看最后一条
    fails = [f for f in latest.values() if f[6] == "FAIL"]
    print(f"完成：成功 {len(latest) - len(fails)} 张，失败 {len(fails)} 张", flush=True)
    if fails:
        log = os.path.join(out, "logs", fails[0][4][:-4] + ".log")
        print("第一个失败：" + "\t".join(fails[0]))
        print("它的 sd-cli 输出最后 10 行：")
        print("".join(open(log, encoding="utf-8", errors="replace").readlines()[-10:]))
        sys.exit(1)
    print("GEN_DONE")


if __name__ == "__main__":
    main()
