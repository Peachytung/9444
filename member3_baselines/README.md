# COMP9444 Project 005 - ResNet50 and VGG16 Baselines

This folder is the complete, merge-ready baseline contribution for the
**Intelligent Food Image Recognition** group project. It trains and evaluates
two classical CNN semantic-segmentation baselines on UECFoodPIX:

- ResNet50-FCN
- VGG16-FCN

Both models are trained from scratch with the same simple recipe. No learning
rate scheduler, early stopping, class weighting, pretrained weights, or other
model optimisation is used.

The delivered runs follow the group-shared protocol exactly: the supplied
3,000/500 train/validation JSON split, seed 9444, 30 epochs, 384 x 320 inputs,
batch size 2, AdamW (`lr=3e-4`, `weight_decay=1e-4`), dropout 0.2, and automatic
mixed precision. The official 1,000-image test set remains held out until the
best validation checkpoint has been selected.

## What is included

- Reproducible source code and deterministic train/validation/test splits.
- An executed baseline notebook for merging into the group notebook.
- Best checkpoints, metric histories, test metrics, confusion matrices,
  training curves, per-class results, and qualitative predictions.
- A report section in Markdown and Word format.
- A PowerPoint section following the supplied COMP9444 slide template.
- HPC2 scripts, logs, environment evidence, and a completion checklist.

The dataset itself is deliberately excluded from the submission folder. Use
the supplied `UECFOODPIX.zip` and extract it so that `category.txt`,
`train9000.txt`, `test1000.txt`, and the `UECFoodPIX/` directory are under one
data root.

## Reproduce locally or on a GPU machine

```bash
python src/import_split_json.py --data-root /path/to/UECFOODPIX/data --split-json splits/uecfoodpix_split_train3000_val500.json --output-dir splits
python src/smoke_test.py --data-root /path/to/UECFOODPIX/data --image-width 384 --image-height 320 --dropout 0.2
python src/train_resnet.py --data-root /path/to/UECFOODPIX/data --epochs 30 --image-width 384 --image-height 320 --batch-size 2 --lr 0.0003 --weight-decay 0.0001 --dropout 0.2 --amp
python src/train_vgg.py --data-root /path/to/UECFOODPIX/data --epochs 30 --image-width 384 --image-height 320 --batch-size 2 --lr 0.0003 --weight-decay 0.0001 --dropout 0.2 --amp
```

The exact full-run commands and environment versions are saved under
`reproducibility/` after training.

See `RUN_SUMMARY.md` for the completed official test metrics and HPC2 execution
evidence, and `FINAL_QA_REPORT.md` for the final machine-checkable acceptance
report.

## Submission / integration

For the final group submission, merge the executed notebook section, the
report section, and the baseline slide into the corresponding whole-group
artifacts. Do not submit `UECFOODPIX.zip`. Check the final instructions in
`SUBMISSION_CHECKLIST.md`.

The formal HPC2 runs, official held-out test evaluation, result aggregation,
executed notebook, report section, slide, and final acceptance checks are all
complete. The authoritative headline results are ResNet50-FCN test mIoU
`0.1178` and VGG16-FCN test mIoU `0.0693`.
