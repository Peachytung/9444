#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-/opt/anaconda3/bin/python}
DATA_ROOT=${DATA_ROOT:-../Comp9444_gw/UECFOODPIX/data}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_3000_val500_384x320}

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}/logs"

common=(
  --data-root "${DATA_ROOT}"
  --split-dir splits
  --output-dir "${OUTPUT_ROOT}"
  --epochs 30
  --width 384
  --height 320
  --batch-size 2
  --lr 0.0003
  --weight-decay 0.0001
  --dropout 0.2
  --num-workers 0
  --seed 9444
  --device mps
  --log-every 50
  --resume
)

echo "started_at=$(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "python=${PYTHON_BIN}"
"${PYTHON_BIN}" - <<'PY'
import torch, torchvision
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("mps_available", torch.backends.mps.is_available())
PY

echo "=== ResNet50-FCN ==="
"${PYTHON_BIN}" src/train_resnet.py "${common[@]}" 2>&1 | tee "${OUTPUT_ROOT}/logs/resnet50_fcn.log"

echo "=== VGG16-FCN ==="
"${PYTHON_BIN}" src/train_vgg.py "${common[@]}" 2>&1 | tee "${OUTPUT_ROOT}/logs/vgg16_fcn.log"

echo "=== Aggregate comparison ==="
"${PYTHON_BIN}" src/aggregate_results.py --outputs "${OUTPUT_ROOT}" --output-dir "${OUTPUT_ROOT}/comparison"

echo "completed_at=$(date '+%Y-%m-%dT%H:%M:%S%z')"
