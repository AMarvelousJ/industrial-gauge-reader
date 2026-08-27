#!/usr/bin/env bash
# check_arm_env.sh - RK3576 ARM64 环境探测 + 依赖版本建议
# 用于部署前确定 torch/torchvision/opencv/onnxruntime 的 arm64 可用版本。
set -u

echo "==========================================================="
echo " RK3576 ARM64 环境探测"
echo "==========================================================="

echo "--- 1. 架构 ---"
uname -m
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "arm64" ]; then
  echo "[WARN] 当前架构 $ARCH 不是 aarch64/arm64;预期 RK3576 为 aarch64"
fi

echo ""
echo "--- 2. Ubuntu / 内核版本 ---"
head -n2 /etc/os-release || true
uname -r

echo ""
echo "--- 3. Python ---"
if command -v python3 >/dev/null 2>&1; then
  python3 --version
  PYV=$(python3 -c 'import sys;print("{}.{}".format(sys.version_info[0],sys.version_info[1]))')
  echo "Python found: $PYV"
else
  echo "[ERROR] python3 未找到;请先安装 python3 / python3-venv"
fi

echo ""
echo "--- 4. glibc ---"
if command -v ldd >/dev/null 2>&1; then
  ldd --version | head -1
fi

echo ""
echo "--- 5. 内存/磁盘 ---"
free -h | head -2 || true
df -h . | tail -1 || true

echo ""
echo "==========================================================="
echo " 依赖版本建议 (基于探测结果)"
echo "==========================================================="
cat <<'SUGGEST'
[arm64 / aarch64 + Python 3.10-3.12 + Ubuntu 24.04]:

  torch        : 2.4.1            (PyTorch 官方 arm64 CPU wheel;勿用 cu126)
  torchvision  : 0.19.1           (与 torch 2.4.1 配对)
  opencv-python: 4.10.0.84        (arm64 wheel;若 5.x 无轮子则用此)
  onnxruntime  : 1.19.2           (arm64 官方 wheel;RapidOCR 依赖)
  numpy        : 1.26.4           (arm64 稳定线)
  ultralytics   : 8.4.106         (纯 python)
  rapidocr      : 3.8.1           (纯 python + onnxruntime)

  GPU/NPU       : 若设备无独立 GPU,请确认不安装任何 +cu* 轮子
  (若 RK3576 有 NPU,当前 ultralytics 流程不直接利用;RKNN 导出为后续工作)

说明: 版本优先级 = 官方 arm64 wheel 可获得性 > 最新;追新(如
opencv 5.0 / numpy 2.5)在 arm64 上可能缺 wheel。
建议以 requirements_rk3576.txt 为准(可在本包内调整)。
SUGGEST

echo ""
echo "--- 自检命令 ---"
echo "  python3 -c \"import torch,torchvision,cv2,onnxruntime;print(torch.__version__,cv2.__version__,onnxruntime.__version__)\""
echo "  python3 -c \"from ultralytics import YOLO; import rapidocr; print('stack ok')\""
