# Baseline run summary

## Delivered protocol

- Dataset: UECFoodPIX semantic segmentation, 103 pixel labels.
- Split: 3,000 training / 500 validation from official train9000; official test1000 held out.
- Shared recipe: seed 9444, 30 epochs, 384 x 320, batch size 2, AdamW (lr=3e-4, weight decay=1e-4), dropout 0.2.
- No pretrained weights, scheduler, early stopping, class weighting, or model-specific tuning.
- Best checkpoint selected by validation mIoU; official test evaluated only after training finishes.

## Official held-out test results

Re-run the notebook or scripts with the updated protocol to regenerate this
section. Previous results are retained under `outputs/` as historical artifacts
only.

## Execution evidence

- Smoke-test Slurm job(s): 10089610.
- Training chunk job log(s): 10089712, 10089745, 10089746, 10089747, 10089748, 10089749, 10089750, 10089751, 10089752.
- Exact commands and software/GPU versions: `reproducibility/`.
- Curves, confusion matrices, per-class CSVs, qualitative predictions and best checkpoints: `outputs_3000_val500_384x320/`.

## Interpretation boundary

These are deliberately simple baselines, not tuned performance claims. Any
improved group model should be compared on the same split and metrics without
using the official test set for model choices.
