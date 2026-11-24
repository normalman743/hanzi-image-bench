# hanzi-image-bench

测 Qwen-Image-2.1 在不同量化精度下生成图中文字的能力。单字题按字表级别、字形复杂度和用字分组。另有 160 道试测题，覆盖简单场景文字、多区域文字和整页长文。

## 测什么

- **模型**：[Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)（7B 去噪网络 + Qwen3-VL-8B 文本编码器）。
- **精度**：只换去噪网络，编码器和 VAE 所有精度共用同一份，所以差别只来自去噪网络的量化。

  | 精度 | 文件（[unsloth/Qwen-Image-2.1-GGUF](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF)） | 大小 | 是否必测 |
  |---|---|---|---|
  | Q2_K | `qwen-image-2.1-Q2_K.gguf` | 2.47 GB | 必测 |
  | Q4_K_M | `qwen-image-2.1-Q4_K_M.gguf` | 4.20 GB | 必测 |
  | Q8_0 | `qwen-image-2.1-Q8_0.gguf` | 7.64 GB | 必测 |
  | F16 | `qwen-image-2.1-F16.gguf` | 14.23 GB | 建议：不量化的参照 |

  共用：文本编码器 `Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf`（5.15 GB，[unsloth/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF)），VAE `qwen_image_2.1_vae_bf16.safetensors`（0.68 GB，[unsloth/Qwen-Image-2.1-FP8](https://huggingface.co/unsloth/Qwen-Image-2.1-FP8) 的 `vae/`）。

- **字**：463 个（单字表 350 + 配对表新增 113），见下面「字表」。分组维度：
  - 常用等级：简体用大陆《通用规范汉字表》1 / 2 / 3 级，繁体用台湾甲 / 乙 / 丙表，方言字两边都不收的算 4 级。
  - 复杂度：周长复杂度（Pelli et al., 2006；算法同 [imWo](https://github.com/SebastianKorinth/imWo)）。
  - 字体：简体、繁体（只取和简体字形不同的字，简繁配对记组号）、方言字。
- **生成设置**（官方配置）：2048×2048，40 步，不开 guidance（`--cfg-scale 1.0`），euler。每个字每个精度用同一组种子，同一个字在不同精度下可以逐张对比。
- **提示词**（固定，只换引号里的字，繁体 / 方言不额外注明）：
  `白色背景上，用黑色楷体端正地写着一个大大的汉字“X”，字居中，画面中没有其他任何文字或图案。`

## 搭建

需要：Linux，NVIDIA 驱动，CUDA 工具包（`nvcc`，12.x），`cmake`、`g++`、`git`、`curl`；Python 3 + Pillow（只用来拼图）。Ubuntu 上缺什么就装：

```bash
sudo apt-get install -y cmake g++ build-essential git curl
pip install pillow
```

一条命令下载模型并编译 [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)（固定在已验证的 commit）：

```bash
bash scripts/setup.sh
```

- 模型放在 `models/`，sd.cpp 编译在 `stable-diffusion.cpp/build/bin/sd-cli`。中断后重新执行会接着下载。
- 默认下载 Q8_0、Q4_K_M、Q2_K。加 F16 参照：`QUANTS="Q8_0 Q4_K_M Q2_K F16" bash scripts/setup.sh`。
- 默认按 RTX 4090 / L4 编译（`CUDA_ARCH=89`）。其他显卡改这个变量，例如 A100 用 `CUDA_ARCH=80`、H100 用 `CUDA_ARCH=90`。
- 硬盘：三个量化版加共用文件约 20 GB，加 F16 约 35 GB。

## 生成单字图

```bash
bash scripts/gen.sh chars/pilot.tsv
```

- 自动使用所有显卡：每张卡一个进程，按行号分字。也可以用 `NGPU=4` 指定卡数。
- 每个字依次跑完所有精度和种子，再换下一个字。这样中途停下，已经出图的字各精度都是齐的。
- 已经存在的图会跳过，中断后重新执行同一条命令即可接着跑。
- 可调的环境变量（括号里是默认值）：`QUANTS`（`Q8_0 Q4_K_M Q2_K`）、`SEEDS`（`42`，多个种子用空格分开）、`W` / `H`（`2048`）、`STEPS`（`40`）、`CFG`（`1.0`）、`OUT`（`out/`）、`EXTRA`（额外传给 sd-cli 的参数）。
- 显存不够（OOM）时加 `EXTRA="--vae-tiling"`。还不够再加 `--clip-on-cpu`（编码器放到 CPU，会慢一些）。
- 输出：`out/<行号>_<组>_<精度>_s<种子>.png`，每张卡一个记录文件 `out/manifest_g<卡号>.tsv`，列为：行号、组、字、精度、种子、文件名、耗时、状态。出错不中断：失败的那张记为 `FAIL`，sd-cli 的原始输出存在 `out/logs/`，跑完会打印失败数量和第一个失败。

耗时（估算）：L4 上 1024×1024、20 步、cfg 6 实测每张 98 秒。按官方 2048 / 40 步算，4090 上每张约 2–3 分钟，4 张卡每小时约 80–120 张。

## 字表

正式字表在 `data/singlewords/`（CSV，列：`字,复杂度,级别`）：

| 文件 | 字数 | 级别的含义 |
|---|---|---|
| `eval_simplified_120.csv` | 1 / 2 / 3 级各 40 | 大陆《通用规范汉字表》1 / 2 / 3 级 |
| `eval_traditional_120.csv` | 1 / 2 / 3 级各 40 | 台湾甲 / 乙 / 丙表（1 = 甲，2 = 乙，3 = 丙）；只收和简体字形不同的字 |
| `eval_dialect_110.csv` | 2 级 24、3 级 12、4 级 74 | 2 / 3 = 大陆字表级别；4 = 两边字表都不收 |
| `eval_simplified_traditional_pairs_60.csv` | 60 对，1 / 2 / 3 级各 20 对 | 简繁配对，两边常用程度对齐：1 = 大陆 1 级配台湾甲表，2 = 2 级配乙表，3 = 3 级配丙表 |

同一文件内各级的复杂度中位数基本一致（简体约 550，繁体约 675），等级和复杂度是分开的。

单字表是各自随机抽的，简繁两边天然配上的只有 1 对，所以另抽了配对表，用来比较同一个字的简繁两种写法（McNemar 检验，按写对率差 20% 左右估计需要约 60 对）。配对表的前 5 对（国國、门門、书書、马馬、龙龍）是试点时手选的，其余为随机抽取。

待补的列：方言字所属方言。

`gen.sh` 读 TSV（`组名<TAB>字`，后面的列不读，`#` 开头的行跳过）。先把 CSV 转成 TSV：

```bash
python3 scripts/csv2tsv.py data/singlewords/*.csv -o chars/all.tsv -p chars/pairs_index.tsv
bash scripts/gen.sh chars/all.tsv
```

- 组名为字体前缀加级别：`simp_L1` … `trad_L3`、`dial_L2` … `dial_L4`；配对表的字为 `pair_L1_simp`、`pair_L1_trad` 等，每对的两个字连着排。
- 配对表里已经在单字表中的字（7 个）不重复生成。`chars/pairs_index.tsv` 记录每对的组号、级别、两个字和它们各自所在的组，分析时按字找图。
- 行的顺序：各组之间轮流排，组内按复杂度分散排。中途停下时，每组完成的字数差不多，而且已完成的字覆盖从简单到复杂的整个范围。
- 合计 463 个字。工作量（估算）：463 × 3 个精度 ≈ 1390 张，4 张 4090 按官方 2048 设置约 14–15 小时，加 F16 约 19 小时。

`chars/pilot.tsv` 是在 L4 上做可行性测试用的 19 个字。

## 看结果

```bash
python3 scripts/sheet.py out sheet.png
```

拼成一张对照图：每行一个字，每列一个精度（多个种子时每个种子一列）。

## 评测（待定）

- **主指标**：写对率，用 OCR 判断，必须和目标字完全一致才算写对。按组 × 精度统计，看它随常用等级、复杂度怎样下降。
  - 每个字先用字体渲染一遍，OCR 读不对的字剔除，免得把 OCR 的错算到模型头上。剔掉的数量按简 / 繁 / 方言分别记。
  - OCR 选用 PP-OCRv5（支持简繁）还是多模态大模型盲读（不告诉它目标字）还没定。不要用 Qwen3-VL：它就是被测模型的文本编码器。
- **辅助指标**：同一个字、同一个种子，各量化版和参照（F16，没有就用 Q8_0）的 LPIPS 距离。

## 简单场景文字试测

[题库 CSV](data/simple-scene-text/eval.csv) 有 40 题。以 20 道自然场景题为基础，加上 6 道场景变体、5 对书面／口语题和 2 对简繁题。对照题用相同的种子和生成设置。场景搭配题固定目标文字，语体和简繁题固定牌子及场景。香港的两对语体题比较书面中文与书面粤语，措辞和语体一起变化。场景和文字的设计参考 [TextInVision](https://arxiv.org/html/2503.13730#S3)。

- `目标文字语体` 有正式书面、中性、口语化三个值。菜名和场馆名称归入中性。`少见但合理` 是设题判断。公开用例的链接和自拟题的标记见 `用例来源` 与 `用例说明`。
- 字形复杂度按目标文字中的汉字求和与平均，重复字逐次计入。遇到缺评分字时保留题目，复杂度两列留空。
- 判分时分别记录牌面文字是否逐字正确、目标文字是否出现在指定牌面。文字错误可记录为简繁替换、同义改写、字形缺损、漏写等。比如目标是「食得晒至好嗌」，结果写成「請按食量點餐」，逐字判错，并记录为同义改写。

## 多区域文字试测

[题库 CSV](data/multi-region-text/eval.csv) 有 60 题。青海西宁、黑龙江、四川成都、香港、台湾客家地区和通用对照各 10 题。每组各有一个餐饮场景和一个文化展览场景。地方用词有公开出处，照片场景按用例拟写。选题参考 [CVTG-2K](https://arxiv.org/html/2503.23461v6)。

- 每个场景有 2、3、4、5 区域各一题，再加一道两区互换题。题号尾缀 `-1` 是两区基准题，`-2` 是互换题，`-3` 到 `-5` 依次增加一块同用途牌子。同一场景各题用同一种子，牌子都正面横排，样式相同。
- `区域标注` 是按从左到右排列的 JSON 数组，每块牌子都有位置、目标文字、汉字数、周长复杂度和用例来源。原有左右区字段适用于两区题。
- 分别记录目标词是否写对、是否出现在指定位置，以及两区题是否发生完整互换。区域数随总字数增加，地区和用字习惯也随场景变化。按区域数和场景报告结果时，同时给出总字数。

## 整页长文试测

[题库 CSV](data/full-page-text/eval.csv) 有 60 题，取自 5 篇 [STRICT 公开的中文百科原文](https://github.com/tianyu-z/STRICT-Bench/blob/main/text_data/zh.json) 和 5 篇本项目自拟的生活说明。每篇各做简体、繁体，再从每个版本开头取 50、100、200 字符。各长度档 20 题，包含 10 道简体、10 道繁体。每个长度、字形、题材的组合有 5 题。

- 先清理百科原文中的抽取残留，再用 OpenCC 转写字形并核对专名。简繁使用同一版式和提示词模板，原文编号用于配对。正文长度含标点、空格、数字及英文字母，字形复杂度只计算汉字。
- 按正文顺序比对转写结果与目标正文，记录全文字符错误率和末尾漏写的长度。正文会在指定字符位置截断。

第 2–4 项的 CSV 提供完整的 `提示词`，可逐题用于图像生成。上面的 `scripts/gen.sh` 批处理入口读取单字 TSV。

场景和长文题的新字沿用 R + imWo 的周长复杂度流程。统一用 BabelStone Han 字体、48 点、600 × 600 像素，在 300 dpi 下渲染并按 `P²/A` 计分。补算值为嘥 559.4714327、粄 573.3429777、佈 458.864892、錄 638.8553187、淨 576.439256、閒 581.3959594。

## 参考

- Qwen-Image 技术报告（ChineseWord 三级字测试）：https://arxiv.org/abs/2508.02324
- Ideogram 4.0 的 INT8 / GGUF 量化与 OCR 写字评测：https://arxiv.org/abs/2606.12280
- Pelli, D. G., Burns, C. W., Farell, B., & Moore-Page, D. C. (2006). Feature detection and letter identification. *Vision Research*, 46(28), 4646–4674.
