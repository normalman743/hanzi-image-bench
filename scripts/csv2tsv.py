"""把 data/singlewords/*.csv（列：字,复杂度,级别）转成 gen.sh 读的 TSV。
输出每行：组名<TAB>字<TAB>复杂度<TAB>级别<TAB>来源文件。组名 = 字体前缀_L级别，例如 simp_L1、trad_L3、dial_L4。
行的顺序：各组之间轮流排（simp_L1、simp_L2、…、dial_L4、simp_L1、…）；组内按复杂度"跳着取"（中位、四分位、八分位……）。
这样中途停下时，每组完成的字数差不多，而且覆盖从简单到复杂的整个范围。

用法：python3 scripts/csv2tsv.py data/singlewords/*.csv -o chars/all.tsv"""
import argparse
import csv
import os
from collections import OrderedDict

PREFIX = {"simplified": "simp", "traditional": "trad", "dialect": "dial"}
HEADER = ["字", "复杂度", "级别"]


def prefix_of(path):
    hits = [p for k, p in PREFIX.items() if k in os.path.basename(path)]
    if len(hits) != 1:
        raise ValueError(f"{path}：文件名里要有且只有一个 {list(PREFIX)}，实际匹配 {hits}")
    return hits[0]


def spread_order(n):
    """0..n-1 的一个排列，任意前缀都在 [0, n) 上分布均匀（按二进制位反转排序）。"""
    bits = max(1, (n - 1).bit_length())
    return sorted(range(n), key=lambda i: int(format(i, f"0{bits}b")[::-1], 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    groups = OrderedDict()
    seen = {}
    errors = []
    for path in a.csvs:
        pre = prefix_of(path)
        with open(path, encoding="utf-8-sig", newline="") as f:
            rd = csv.reader(f)
            head = next(rd)
            if head != HEADER:
                raise ValueError(f"{path}：表头 {head}，预期 {HEADER}")
            for n, row in enumerate(rd, start=2):
                if not row:
                    continue
                if len(row) != 3:
                    errors.append(f"{path}:{n} 列数 {len(row)}，预期 3：{row}")
                    continue
                ch, cx, lv = row
                if len(ch) != 1:
                    errors.append(f"{path}:{n} 字 {ch!r} 不是单字")
                try:
                    cx_v, lv_v = float(cx), int(lv)
                except ValueError:
                    errors.append(f"{path}:{n} 复杂度 {cx!r} / 级别 {lv!r} 不是数字")
                    continue
                if ch in seen:
                    errors.append(f"{path}:{n} 字 {ch} 和 {seen[ch]} 重复")
                seen[ch] = f"{path}:{n}"
                groups.setdefault(f"{pre}_L{lv_v}", []).append((ch, cx, lv, os.path.basename(path)))
    if errors:
        raise ValueError(f"{len(errors)} 处错误，第一处：{errors[0]}")
    names = sorted(groups)
    for g in names:
        rows = sorted(groups[g], key=lambda r: float(r[1]))
        groups[g] = [rows[i] for i in spread_order(len(rows))]
    out_rows = []
    for k in range(max(len(v) for v in groups.values())):
        for g in names:
            if k < len(groups[g]):
                out_rows.append((g,) + groups[g][k])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# 组名\t字\t复杂度\t级别\t来源（由 scripts/csv2tsv.py 生成）\n")
        for r in out_rows:
            f.write("\t".join(r) + "\n")
    print(f"{len(out_rows)} 个字 → {a.out}")
    for g in names:
        print(f"  {g}: {len(groups[g])}")


if __name__ == "__main__":
    main()
