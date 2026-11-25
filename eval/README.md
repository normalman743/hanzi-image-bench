# 判分脚本

评估方法见 [EVAL.md](../EVAL.md)，评分细则见 [rubric.md](rubric.md)。

| 文件 | 作用 |
|---|---|
| `grok_judge.py` | 公共部分（哪些图送 AI、目标文字、结果比对），以及 Grok 裁判 |
| `codex_judge.py` | GPT 裁判（Codex CLI）；`--rubric` 是定稿用的逐字三档提示词，`--backend agy` 是 Gemini |
| `final_tables.py` | 合成最终判定和汇总表 |
| `rubric_compare.py` | GPT 和人工逐张对比，算一致率 |
| `rubric.md` | 评分细则 |
| `shards/` | Grok 分工清单 |

## 准备

- 生成结果目录：要有 `1_single_char/`、`2_simple_scene/`、`3_multi_region/` 三个子目录（Drive 上的结构）。如果还是生成时的目录名，建软链接：
  ```bash
  mkdir -p hb && ln -s /路径/out hb/1_single_char && ln -s /路径/out_simple-scene-text hb/2_simple_scene && ln -s /路径/out_multi-region-text hb/3_multi_region
  ```
- `7_ocr_scores/` 目录（第 1 级 OCR 结果，在共享 Drive）。
- Python 3，只用标准库（`--sheet` 拼图模式另需 Pillow）。
- 路径都用环境变量给：

| 变量 | 内容 |
|---|---|
| `HB_RES` | 上面的生成结果目录（必填） |
| `HB_T1DIR` | `7_ocr_scores` 目录（必填） |
| `HB_OUT` | 输出目录，默认 `eval/out` |
| `HB_HUMAN` | 人工判分 CSV（列 `file, 人工三档, 备注`），只有 `rubric_compare.py` 用 |

下面的命令都先设好：
```bash
export HB_RES=/路径/hb HB_T1DIR=/路径/7_ocr_scores
```

## 1. GPT 判分（定稿）

装好并登录 Codex CLI，确认能用 `gpt-6.2-sol`。

```bash
# 送 AI 的全部 1321 张的清单
python3 -c "import sys; sys.path.insert(0,'eval'); import grok_judge as G; print('\n'.join(i['file'] for i in G.items()))" > all.txt
python3 -u eval/codex_judge.py run --rubric --files all.txt --out_name rubric.jsonl --par 8 | tee gpt.log
```

- 结果写到 `$HB_OUT/codex/rubric.jsonl`，每行一张图。GPT 的回答在 `ai` 字段：
  - 单字：`read`、`grade`；
  - 场景：`chars`，逐字 `c` / `g`；
  - 多区域：`signs`，每块招牌一组 `c` / `g`；
  - 都带 `confidence` 和 `note`。
- 约 140 次调用。8 路并行时每批约 1 分钟，全部大约 20 分钟。
- 中断了重跑同一条命令即可：判过的图跳过，出错的补判。

## 2. 最终表

```bash
python3 eval/final_tables.py --rubric
```

输出到 `$HB_OUT/final/`：逐张的 `final_<任务>_rubric.csv`，和汇总的 `summary_final_rubric.csv`。屏幕上会打印各任务 × 精度的写对率，括号里是纯 OCR 的写对率。

不加 `--rubric` 时用的是旧提示词的结果（`codex/stage1.jsonl`），只用于对照。

## 3. 和人工对比

```bash
HB_HUMAN=/路径/review.csv HB_RUBRIC=$HB_OUT/codex/rubric.jsonl HB_OLD=$HB_OUT/codex/stage1.jsonl python3 eval/rubric_compare.py
```

脚本打印对错一致率，并逐张列出不一致的图。逐张对比写在 `HB_RUBRIC` 所在目录的 `compare.csv` 里。

## 4. 其他裁判（裁判对比用）

### Grok

需要 Cursor Agent 命令行，并确认能用 Grok：`cursor-agent --list-models | grep grok-4.7-medium`。

前 200 张已经判完。剩下 1121 张按顺序交替分成两份：

| 清单 | 张数 | 谁跑 |
|---|---|---|
| `shards/shard_0_me.txt` | 561 | normalman743 |
| `shards/shard_1_teammate.txt` | 560 | 队友 |

```bash
python3 -u eval/grok_judge.py stage1 --par 10 --files eval/shards/shard_1_teammate.txt | tee grok.log
```

- 开头应打印 `清单 …：560 张`。10 路并行，每批 10 张约 10 分钟，共约 1 小时。
- 如果日志里大量出错（网络 / TLS 断开），把 `--par` 降到 6。
- 中断了重跑同一条命令即可。
- 耗的是自己 Cursor 账号的额度。
- 跑完把 `$HB_OUT/stage1.jsonl` 传到共享 Drive（建议放 `8_grok/`），不要提交到仓库。

### Gemini

```bash
python3 -u eval/codex_judge.py run --backend agy --model gemini-4-hydrogen-medium --files all.txt --out_dir $HB_OUT/gemini
```

试用账号很快会被限速（429）。

## 注意

判分结果（`*.jsonl`、`final/`）都放 Drive，不提交到仓库。
