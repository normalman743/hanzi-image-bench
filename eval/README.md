# 第 2 级判分：Grok 看图

按 [EVAL.md](../EVAL.md) 的三级判分，第 1 级 OCR 拿不准的图交给 Grok（Cursor Agent 里的 `grok-4.7-medium`）盲读：不告诉它目标文字，只问图里写的是什么、是不是标准字、有多大把握（0–1）。

送 Grok 的图（共 1321 张）：

- 任务 1 单字：OCR 判错、OCR 字典外、3 / 4 级字、或 OCR 判对但置信度 < 0.9；
- 任务 2 简单场景：OCR 判逐字错；
- 任务 3 多区域：OCR 判"全对且位置对"为否。

AI 判对 = 读出的文字等于目标，且没有不规范的字。把握分数低的会再往上送（grok-4.7-high → Claude → 人工），阈值看分数分布后再定。

## 分工

前 200 张已经判完。剩下的 1121 张按顺序交替分成两份：

| 清单 | 张数 | 谁跑 |
|---|---|---|
| `shards/shard_0_me.txt` | 561 | normalman743 |
| `shards/shard_1_teammate.txt` | 560 | 队友 |

## 运行

1. 装好并登录 Cursor Agent 命令行，确认能用 Grok：`cursor-agent --list-models | grep grok-4.7-medium`
2. 准备：
   - 生成结果目录：要有 `1_single_char/`、`2_simple_scene/`、`3_multi_region/` 三个子目录（Drive 上的结构）。如果还是生成时的目录名，建软链接：
     ```bash
     mkdir -p hb && ln -s /路径/out hb/1_single_char && ln -s /路径/out_simple-scene-text hb/2_simple_scene && ln -s /路径/out_multi-region-text hb/3_multi_region
     ```
   - `7_ocr_scores/` 目录（第 1 级 OCR 结果，在共享 Drive）。
   - Python 3，只用标准库。
3. 运行（换成你的路径）：
   ```bash
   HB_RES=/路径/hb HB_T1DIR=/路径/7_ocr_scores \
   python3 -u eval/grok_judge.py stage1 --par 10 --files eval/shards/shard_1_teammate.txt | tee grok.log
   ```
   - 开头应打印 `清单 …：560 张`。10 路并行，每批 10 张约 10 分钟，共约 1 小时。
   - 日志里大量"出错"（网络 / TLS 断开）就把 `--par` 降到 6。
   - 中断了重跑同一条命令即可，判过的图跳过，出错的批次补判。
   - 耗的是你自己 Cursor 账号的额度。
4. 跑完把 `eval/out/stage1.jsonl` 传到共享 Drive（建议放 `8_grok/`），不要提交到仓库。

`stage1.jsonl` 每行一张图：题号 / 文件、目标、送 AI 原因，以及 Grok 的回答 `ai`（`read` 或 `text` / `signs`、`real_char` 或 `malformed`、`confidence`、`note`）。
