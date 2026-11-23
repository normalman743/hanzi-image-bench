#!/bin/bash
# 下载模型 + 编译 stable-diffusion.cpp（CUDA）。中断后重新执行会接着下载。
# 环境变量：QUANTS（默认 "Q8_0 Q4_K_M Q2_K"，可加 F16）、CUDA_ARCH（默认 89 = RTX 4090 / L4）、MODELS（默认 ./models）
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
M=${MODELS:-$ROOT/models}
QUANTS=${QUANTS:-"Q8_0 Q4_K_M Q2_K"}
CUDA_ARCH=${CUDA_ARCH:-89}
SD_COMMIT=0d71f963af9c94941f4a67e8ea145935651dd756
HF=https://huggingface.co
mkdir -p "$M"

dl() {   # dl <repo 内路径> <本地文件>
  echo "下载 $(basename "$2")"
  curl -sSfL --retry 5 -C - -o "$2" "$HF/$1"
}
dl unsloth/Qwen-Image-2.1-FP8/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors "$M/qwen_image_2.1_vae_bf16.safetensors"
dl unsloth/Qwen3-VL-8B-Instruct-GGUF/resolve/main/Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf "$M/Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf"
for q in $QUANTS; do
  dl unsloth/Qwen-Image-2.1-GGUF/resolve/main/qwen-image-2.1-$q.gguf "$M/qwen-image-2.1-$q.gguf"
done
ls -la "$M"

cd "$ROOT"
[ -d stable-diffusion.cpp ] || git clone --recursive https://github.com/leejet/stable-diffusion.cpp
cd stable-diffusion.cpp
git fetch -q origin
git checkout -q $SD_COMMIT
git submodule update --init --recursive -q
export PATH=/usr/local/cuda/bin:$PATH
cmake -B build -DSD_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH
cmake --build build --config Release -j "$(nproc)"
ls -la build/bin/sd-cli
echo "SETUP_DONE"
