#!/usr/bin/env bash
# run_demo.sh - 启动摄像头仪表读数 Demo(RK3576)
# 用法: bash run_demo.sh                 (摄像头)
#       bash run_demo.sh --image-dir assets/demo_images   (回放模式)
set -euo pipefail

BASE="$(cd "$(dirname "$0")" && pwd)"

# 激活环境
if [ -f "$BASE/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "$BASE/.venv/bin/activate"
else
  echo "[ERROR] .venv 不存在,请先执行 bash install.sh"; exit 1
fi

# 可选:设备自检(不阻断)
if [ "${SKIP_CHECK:-0}" != "1" ]; then
  echo "--- 设备自检 (SKIP_CHECK=1 可跳过) ---"
  bash "$BASE/scripts/check_device.sh" || echo "(自检有缺项,继续尝试启动 Demo)"
fi

cd "$BASE"
# 默认摄像头;若传入 --image-dir 则交给 demo_camera 回放模式
exec python -m style_reader.demo_camera "$@"
