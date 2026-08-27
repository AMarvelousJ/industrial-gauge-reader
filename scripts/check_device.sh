#!/usr/bin/env bash
# check_device.sh - RK3576 部署前设备自检(架构/Python/摄像头/依赖/模型)
set -u
FAIL=0

echo "==========================================================="
echo " RK3576 设备自检"
echo "==========================================================="

echo "[1] ARM 架构"
if [ "$(uname -m)" = "aarch64" ] || [ "$(uname -m)" = "arm64" ]; then
  echo "  OK: $(uname -m)"
else
  echo "  FAIL: $(uname -m) (预期 aarch64)"; FAIL=1
fi

echo "[2] Python"
if command -v python3 >/dev/null 2>&1; then
  python3 --version
else
  echo "  FAIL: python3 缺失"; FAIL=1
fi

echo "[3] 摄像头权限 (+usb/video)"
if [ -e /dev/video0 ]; then
  ls -l /dev/video0
  if groups | grep -q video; then echo "  OK: 当前用户属于 video 组"; else
    echo "  WARN: 不在 video 组(需: sudo usermod -aG video \$USER 后重登)"; fi
else
  echo "  WARN: 未发现 /dev/video0(可使用 --image-dir 回放模式)"
fi

echo "[4] 依赖导入"
python3 - <<'PY' 2>&1 || FAIL=1
import importlib
missing=[]
for name in ("torch","torchvision","cv2","onnxruntime","numpy","ultralytics","rapidocr","yaml","PIL"):
    try:
        mod=importlib.import_module(name)
        print("  OK  %-12s %s" % (name, getattr(mod,"__version__","?")))
    except Exception as e:
        missing.append(name)
        print("  MISS %-12s %s" % (name, e))
if missing: raise SystemExit("missing: "+",".join(missing))
print("  CUDA(arm64 CPU 预期 False):", importlib.import_module('torch').cuda.is_available())
PY

echo "[5] 模型文件"
BASE="$(cd "$(dirname "$0")/.." && pwd)"
for f in models/meter_detector.pt models/scale_segment.pt models/pointer_keypoints.pt \
         models/keypoint_threshold.json models/ocr_models/ch_PP-OCRv4_rec_mobile.onnx; do
  if [ -f "$BASE/$f" ]; then
    echo "  OK  $f ($(du -h "$BASE/$f"|cut -f1))"
  else
    echo "  MISS $f"; FAIL=1
  fi
done

echo "[6] 源码完整性"
for d in style_reader pointer_keypoints; do
  if [ -d "$BASE/src/$d" ]; then echo "  OK  src/$d"; else echo "  MISS src/$d"; FAIL=1; fi
done

echo ""
if [ "$FAIL" -eq 0 ]; then echo "== 自检通过 =="; else echo "== 存在缺失项(见上) =="; fi
exit $FAIL
