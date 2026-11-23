#!/bin/bash
# 用法：bash scripts/gen.sh chars/pilot.tsv
# 每张卡一个进程，按行号分字；每个字依次跑完所有精度和种子。已有的图跳过，中断后重跑同一条命令即可。
set -u
CHARS=${1:?用法：bash scripts/gen.sh <字表.tsv>}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
M=${MODELS:-$ROOT/models}
BIN=${SD_BIN:-$ROOT/stable-diffusion.cpp/build/bin/sd-cli}
OUT=${OUT:-$ROOT/out}
QUANTS=${QUANTS:-"Q8_0 Q4_K_M Q2_K"}
SEEDS=${SEEDS:-"42"}
W=${W:-2048}; H=${H:-2048}; STEPS=${STEPS:-40}; CFG=${CFG:-1.0}
EXTRA=${EXTRA:-}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
TMPL='白色背景上，用黑色楷体端正地写着一个大大的汉字“{ch}”，字居中，画面中没有其他任何文字或图案。'

[ -x "$BIN" ] || { echo "找不到 sd-cli：${BIN}（先跑 scripts/setup.sh）"; exit 1; }
for q in $QUANTS; do [ -f "$M/qwen-image-2.1-$q.gguf" ] || { echo "缺模型 $M/qwen-image-2.1-$q.gguf"; exit 1; }; done
mkdir -p "$OUT/logs"
echo "字表 ${CHARS}，${NGPU} 张卡，精度 [${QUANTS}]，种子 [${SEEDS}]，${W}x${H}，${STEPS} 步，cfg ${CFG}，输出 ${OUT}"

worker() {
  local g=$1 i=0 grp ch rest q seed name f t0 st
  local man=$OUT/manifest_g$g.tsv
  while IFS=$'\t' read -r grp ch rest; do
    [ -z "${grp// }" ] && continue
    [[ $grp == \#* ]] && continue
    i=$((i + 1))
    [ $(( (i - 1) % NGPU )) -eq $g ] || continue
    for q in $QUANTS; do
      for seed in $SEEDS; do
        name=$(printf %04d $i)_${grp}_${q}_s${seed}
        f=$OUT/$name.png
        [ -f "$f" ] && continue
        t0=$(date +%s)
        if CUDA_VISIBLE_DEVICES=$g "$BIN" --diffusion-model "$M/qwen-image-2.1-$q.gguf" \
             --vae "$M/qwen_image_2.1_vae_bf16.safetensors" --llm "$M/Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf" \
             -p "${TMPL//\{ch\}/$ch}" --steps $STEPS --cfg-scale $CFG --sampling-method euler \
             -W $W -H $H --diffusion-fa --seed $seed $EXTRA -o "$f" > "$OUT/logs/$name.log" 2>&1 && [ -f "$f" ]; then
          st=OK
        else
          st=FAIL
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%ss\t%s\n' $i "$grp" "$ch" $q $seed "$name.png" $(( $(date +%s) - t0 )) $st | tee -a "$man"
      done
    done
  done < "$CHARS"
}

for g in $(seq 0 $((NGPU - 1))); do worker $g & done
wait
# 同一张图重跑过的，只看最后一条记录
latest=$(cat "$OUT"/manifest_g*.tsv 2>/dev/null | awk -F'\t' '{last[$6]=$0} END {for (k in last) print last[k]}' | sort)
nfail=$(echo "$latest" | awk -F'\t' '$8=="FAIL"' | wc -l)
nok=$(echo "$latest" | awk -F'\t' '$8=="OK"' | wc -l)
echo "完成：成功 ${nok} 张，失败 ${nfail} 张"
if [ "$nfail" -gt 0 ]; then
  first=$(echo "$latest" | awk -F'\t' '$8=="FAIL"' | head -1)
  echo "第一个失败：${first}"
  echo "它的 sd-cli 输出最后 10 行："
  tail -10 "$OUT/logs/$(echo "$first" | cut -f6 | sed 's/\.png$//').log"
  exit 1
fi
echo "GEN_DONE"
