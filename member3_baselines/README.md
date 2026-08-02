# COMP9444 Project 005 - Member 3 Baselines

This folder contains the Member 3 baseline contribution for
**005 Intelligent Food Image Recognition**.

It includes two semantic-segmentation baseline models:

- ResNet50-FCN
- VGG16-FCN

Both models use the same experimental protocol:

- Dataset: UECFoodPIX
- Number of classes: 103, including background
- Train split: 3,000 images
- Validation split: 500 images
- Official test split: 1,000 images
- Input size: 384 x 320
- Batch size: 2
- Epochs: 30
- Learning rate: 3e-4
- Weight decay: 1e-4
- Dropout: 0.2
- Seed: 9444
- Optimizer: AdamW
- Loss: cross-entropy

The train and validation images use the provided 3,000/500 split JSON. The
official test set is evaluated after selecting the best checkpoint by validation
mIoU.

## Folder Structure

```text
notebooks/
  baseline_experiments.ipynb

src/
  data_uecfoodpix.py
  metrics.py
  model_resnet.py
  model_vgg.py
  models.py
  train.py
  train_resnet.py
  train_vgg.py
  smoke_test.py
  create_splits.py
  import_split_json.py
  aggregate_results.py

splits/
  uecfoodpix_split_train3000_val500.json
  train_3000.txt
  val_500.txt
  test_official_1000.txt
  split_manifest.json

outputs/
  comparison/
  resnet50_fcn/
  vgg16_fcn/
```

Large model checkpoints, logs, report files, slides, and local HPC files are not
included in this GitHub folder.

## Results

| Model | Best Epoch | Test Pixel Accuracy | Test mIoU | Test FG mIoU | Test Macro F1 |
|---|---:|---:|---:|---:|---:|
| ResNet50-FCN | 25 | 0.6259 | 0.1178 | 0.1112 | 0.1896 |
| VGG16-FCN | 29 | 0.5896 | 0.0693 | 0.0625 | 0.1128 |

ResNet50-FCN performs better than VGG16-FCN under the fixed baseline protocol.

## How to Run

Place the UECFoodPIX data under a data root containing `category.txt`,
`train9000.txt`, `test1000.txt`, and the `UECFoodPIX/` image folder.

```bash
pip install -r requirements.txt

python src/import_split_json.py \
  --data-root /path/to/UECFOODPIX/data \
  --split-json splits/uecfoodpix_split_train3000_val500.json \
  --output-dir splits

python src/smoke_test.py \
  --data-root /path/to/UECFOODPIX/data \
  --image-width 384 \
  --image-height 320 \
  --dropout 0.2

python src/train_resnet.py \
  --data-root /path/to/UECFOODPIX/data \
  --epochs 30 \
  --image-width 384 \
  --image-height 320 \
  --batch-size 2 \
  --lr 0.0003 \
  --weight-decay 0.0001 \
  --dropout 0.2 \
  --amp

python src/train_vgg.py \
  --data-root /path/to/UECFOODPIX/data \
  --epochs 30 \
  --image-width 384 \
  --image-height 320 \
  --batch-size 2 \
  --lr 0.0003 \
  --weight-decay 0.0001 \
  --dropout 0.2 \
  --amp
```
