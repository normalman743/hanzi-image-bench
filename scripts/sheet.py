"""把 gen.sh 的输出拼成对照图：每行一个字，每列一个（精度, 种子）。
用法：python3 scripts/sheet.py out sheet.png [--cell 256]"""
import argparse
import csv
import glob
import os

from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("sheet")
    ap.add_argument("--cell", type=int, default=256)
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.out_dir, "manifest_g*.tsv"))):
        rows += [r for r in csv.reader(open(f, encoding="utf-8"), delimiter="\t") if r]
    if not rows:
        raise SystemExit(f"{a.out_dir} 里没有 manifest_g*.tsv")
    chars = sorted({(int(r[0]), r[1], r[2]) for r in rows})
    cols = sorted({(r[3], r[4]) for r in rows})
    cell, label = a.cell, 160
    sheet = Image.new("RGB", (label + cell * len(cols), 30 + cell * len(chars)), "white")
    d = ImageDraw.Draw(sheet)
    for j, (q, s) in enumerate(cols):
        d.text((label + j * cell + 5, 8), f"{q} s{s}", fill="black")
    by = {(int(r[0]), r[3], r[4]): r for r in rows}
    for i, (idx, grp, ch) in enumerate(chars):
        y = 30 + i * cell
        d.text((5, y + cell // 2), f"{idx} {grp}", fill="black")
        for j, (q, s) in enumerate(cols):
            r = by.get((idx, q, s))
            x = label + j * cell
            if r is None or r[7] != "OK":
                d.text((x + 5, y + cell // 2), "FAIL" if r else "-", fill="red")
                continue
            im = Image.open(os.path.join(a.out_dir, r[5])).convert("RGB").resize((cell, cell))
            sheet.paste(im, (x, y))
    sheet.save(a.sheet)
    print(f"{len(chars)} 个字 × {len(cols)} 列 → {a.sheet}")


if __name__ == "__main__":
    main()
