"""把 data/singlewords/*.csv 转成 gen.sh 读的 TSV。

两种输入：
  单字表（列：字,复杂度,级别），文件名含 simplified / traditional / dialect。组名 = simp_L1、trad_L3、dial_L4 这种。
  配对表（列：简体字,繁体字,级别,简体周长复杂度,繁体周长复杂度），文件名含 pairs。级别：1 = 大陆 1 级配台湾甲表，2 = 2 级配乙表，3 = 3 级配丙表。
    每对的两个字连着排，组名 pair_L1_simp / pair_L1_trad；已经在单字表里的字不重复生成（分析时按字去找）。
    另写一份配对索引（-p）：组号、级别、简体、繁体、两边复杂度、两边各在哪个组。

输出每行：组名<TAB>字<TAB>复杂度<TAB>级别<TAB>来源文件。
行的顺序：各组之间轮流排；组内按复杂度"跳着取"（中位、四分位、八分位……；配对按两字复杂度的平均）。
这样中途停下时，每组完成的字数差不多，而且覆盖从简单到复杂的整个范围。

用法：python3 scripts/csv2tsv.py data/singlewords/*.csv -o chars/all.tsv -p chars/pairs_index.tsv"""
import argparse
import csv
import os

PREFIX = {"simplified": "simp", "traditional": "trad", "dialect": "dial"}
HEADER = ["字", "复杂度", "级别"]
PAIR_HEADER = ["简体字", "繁体字", "级别", "简体周长复杂度", "繁体周长复杂度"]


def spread_order(n):
    """0..n-1 的一个排列，任意前缀都在 [0, n) 上分布均匀（按二进制位反转排序）。"""
    bits = max(1, (n - 1).bit_length())
    return sorted(range(n), key=lambda i: int(format(i, f"0{bits}b")[::-1], 2))


def read_rows(path, header, errors):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rd = csv.reader(f)
        head = next(rd)
        if head != header:
            raise ValueError(f"{path}：表头 {head}，预期 {header}")
        for n, row in enumerate(rd, start=2):
            if not row:
                continue
            if len(row) != len(header):
                errors.append(f"{path}:{n} 列数 {len(row)}，预期 {len(header)}：{row}")
                continue
            yield n, row


def check(path, n, ch, nums, errors):
    if len(ch) != 1:
        errors.append(f"{path}:{n} 字 {ch!r} 不是单字")
        return False
    try:
        float(nums[0]), int(nums[1])
    except ValueError:
        errors.append(f"{path}:{n} 复杂度 {nums[0]!r} / 级别 {nums[1]!r} 不是数字")
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("-p", "--pairs_index", default=None, help="有配对表时必填")
    a = ap.parse_args()
    singles = [p for p in a.csvs if "pairs" not in os.path.basename(p)]
    pair_files = [p for p in a.csvs if "pairs" in os.path.basename(p)]
    if pair_files and not a.pairs_index:
        raise ValueError("输入里有配对表，要用 -p 指定配对索引的输出文件")
    groups = {}      # 组名 → [单元]，单元 = (排序用复杂度, [(字, 复杂度, 级别, 来源), ...])
    where = {}       # 字 → (组名, 位置)
    errors = []

    for path in singles:
        hits = [p for k, p in PREFIX.items() if k in os.path.basename(path)]
        if len(hits) != 1:
            raise ValueError(f"{path}：文件名里要有且只有一个 {list(PREFIX)}，实际匹配 {hits}")
        for n, (ch, cx, lv) in read_rows(path, HEADER, errors):
            if not check(path, n, ch, (cx, lv), errors):
                continue
            if ch in where:
                errors.append(f"{path}:{n} 字 {ch} 和 {where[ch][1]} 重复")
                continue
            g = f"{hits[0]}_L{int(lv)}"
            where[ch] = (g, f"{path}:{n}")
            groups.setdefault(g, []).append((float(cx), [(ch, cx, lv, os.path.basename(path))]))

    index = []
    for path in pair_files:
        for n, (s, t, lv, cs, ct) in read_rows(path, PAIR_HEADER, errors):
            if not (check(path, n, s, (cs, lv), errors) and check(path, n, t, (ct, lv), errors)):
                continue
            if s == t:
                errors.append(f"{path}:{n} 简繁同形 {s}")
                continue
            pid = f"P{len(index) + 1:03d}"
            unit, gs = [], {}
            for ch, cx, side in ((s, cs, "simp"), (t, ct, "trad")):
                if ch in where:                       # 已在单字表 / 别的配对里：不重复生成
                    gs[side] = where[ch][0]
                    continue
                g = f"pair_L{int(lv)}_{side}"
                where[ch] = (g, f"{path}:{n}")
                gs[side] = g
                unit.append((ch, cx, lv, os.path.basename(path)))
            index.append((pid, lv, s, t, cs, ct, gs["simp"], gs["trad"]))
            if unit:
                groups.setdefault(f"pair_L{int(lv)}", []).append(((float(cs) + float(ct)) / 2, unit))

    if errors:
        raise ValueError(f"{len(errors)} 处错误，第一处：{errors[0]}")
    names = sorted(groups)
    for g in names:
        units = sorted(groups[g], key=lambda u: u[0])
        groups[g] = [units[i] for i in spread_order(len(units))]
    out_rows = []
    for k in range(max(len(v) for v in groups.values())):
        for g in names:
            if k < len(groups[g]):
                for ch, cx, lv, src in groups[g][k][1]:
                    out_rows.append((where[ch][0], ch, cx, lv, src))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# 组名\t字\t复杂度\t级别\t来源（由 scripts/csv2tsv.py 生成）\n")
        for r in out_rows:
            f.write("\t".join(r) + "\n")
    print(f"{len(out_rows)} 个字 → {a.out}")
    counts = {}
    for r in out_rows:
        counts[r[0]] = counts.get(r[0], 0) + 1
    for g in sorted(counts):
        print(f"  {g}: {counts[g]}")
    if index:
        with open(a.pairs_index, "w", encoding="utf-8") as f:
            f.write("# 组号\t级别\t简体\t繁体\t简体复杂度\t繁体复杂度\t简体所在组\t繁体所在组\n")
            for r in index:
                f.write("\t".join(r) + "\n")
        reused = sum((r[6] != f"pair_L{int(r[1])}_simp") + (r[7] != f"pair_L{int(r[1])}_trad") for r in index)
        print(f"{len(index)} 对 → {a.pairs_index}（其中 {reused} 个字复用单字表里的图）")


if __name__ == "__main__":
    main()
