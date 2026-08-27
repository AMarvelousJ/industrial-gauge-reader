#!/usr/bin/env bash
# install.sh - RK3576 部署包安装脚本(Ubuntu 24.04 arm64)
# 步骤:创建venv -> 检查环境 -> 安装依赖 -> 模型校验
set -euo pipefail

BASE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"

echo "==========================================================="
echo " gauge_demo_rk3576_ubuntu - install"
echo "==========================================================="

# 0) 前置:环境探测(架构/版本)
bash "$BASE/scripts/check_arm_env.sh" || true

# 1) 创建 venv
if [ ! -d "$BASE/.venv" ]; then
  echo "[1/4] 创建虚拟环境 .venv ..."
  "$PYTHON" -m venv "$BASE/.venv"
else
  echo "[1/4] .venv 已存在,跳过创建"
fi
# shellcheck disable=SC1091
. "$BASE/.venv/bin/activate"

# 2) 升级 pip
echo "[2/4] 升级 pip ..."
python -m pip install --upgrade pip setuptools wheel

# 3) 安装依赖(优先离线 wheels/,否则走 index-url)
echo "[3/4] 安装依赖 ..."
if [ -d "$BASE/wheels" ] && [ -n "$(ls -A "$BASE/wheels" 2>/dev/null)" ]; then
  echo "  使用离线 wheels/ 目录"
  python -m pip install --no-index --find-links "$BASE/wheels" -r "$BASE/requirements_rk3576.txt"
else
  echo "  wheels/ 为空或缺失,尝试在线安装(要求设备可访问 PyPI)"
  python -m pip install -r "$BASE/requirements_rk3576.txt"
fi

# 4) 环境自检
echo "[4/4] 环境自检 ..."
python - <<'PY'
import importlib, sys
ok=True
for name, attr in (("torch", "__version__"), ("torchvision", "__version__"),
                   ("cv2", "__version__"), ("onnxruntime", "__version__"),
                   ("numpy", "__version__"), ("ultralytics", "__version__"),
                   ("rapidocr", "__version__")):
    try:
        mod = importlib.import_module(name)
        print("  OK  %-14s %s" % (name, getattr(mod, attr, "?")))
    except Exception as e:
        ok=False
        print("  MISS %-14s %s" % (name, e))
if not ok:
    sys.exit("dependency check failed")
import torch
print("  CUDA available (arm64 CPU 预期 False):", torch.cuda.is_available())
print("STACK OK")
PY

# 模型校验(SHA256 列于 README)
echo "--- 模型文件校验 ---"
for f in models/meter_detector.pt models/scale_segment.pt; do
  if [ -f "$BASE/$f" ]; then
    echo "  OK  $f $(du -h "$BASE/$f"|cut -f1)"
  else
    echo "  MISS $f"; exit 1
  fi
done

echo ""
echo "安装完成。可执行: bash run_demo.sh   (或 bash scripts/check_device.sh 检查设备)"
