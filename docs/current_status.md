# Current Project Status

## Goal

- Build an AI-generated video detector.
- Base architecture: `VideoMAE encoder -> MLP classifier -> binary classification`.
- Add explainability support so the system can provide evidence for its decision.

## Current Code State

- Core package: [src/ai_video_detector](/G:/내 드라이브/project g/src/ai_video_detector)
- Data pipeline already exists:
  - video loading
  - temporal sampling
  - resize and normalization
  - train / eval / infer flow
- Model structure currently supports:
  - pretrained `VideoMAE`
  - explicit local fallback encoder when `use_pretrained=False`
  - MLP binary classifier head
  - optional XAI output during inference

## Important Model Changes Applied

- Removed silent fallback when `use_pretrained=True`.
- If `VideoMAE` loading fails, the code now leaves an error log and raises an exception.
- Added `--encoder-name` CLI option for both training and inference.
- Added `--freeze-encoder` CLI option.
- Added `freeze_encoder` to `VideoClassifierConfig`.
- Added encoder-side XAI path:
  - `predict_with_xai(...)`
  - frame importance output
  - attention rollout for `VideoMAE`
  - activation-energy fallback explanation for the local fallback encoder

## Dataset Status

- Main local dataset root:
  - `D:\genVideo-100k\GenVideo-100K\extracted`
- Confirmed folder structure:
  - `Real`
  - `train_pika`
  - `train_SVD`
  - `train_VideoCrafter`
  - `I2VGEN_XL`
- Current loader supports:
  - `.mp4`, `.avi`, `.mov`, `.mkv`, `.npy`, `.pt`
- `I2VGEN_XL` is currently excluded from training manifests because it is stored as `.gif`.

## Manifest Files

- Full manifests:
  - [train_manifest.json](/G:/내 드라이브/project g/manifests/train_manifest.json)
  - [val_manifest.json](/G:/내 드라이브/project g/manifests/val_manifest.json)
  - [manifest_summary.json](/G:/내 드라이브/project g/manifests/manifest_summary.json)
- Full manifest summary:
  - train: `54,000`
  - val: `6,000`
  - balanced binary labels
  - real label: `0`
  - AI label: `1`
- Smoke-test manifests:
  - [train_manifest_smoke.json](/G:/내 드라이브/project g/manifests/smoke/train_manifest_smoke.json)
  - [val_manifest_smoke.json](/G:/내 드라이브/project g/manifests/smoke/val_manifest_smoke.json)
  - smoke train count: `8`
  - smoke val count: `4`

## VideoMAE Model Status

- Downloaded local pretrained model:
  - [videomae-base](/G:/내 드라이브/project g/models/videomae-base)
- Verified load result:
  - `uses_transformers=True`
  - encoder type: `VideoMAEModel`
  - hidden size: `768`

## Execution Results So Far

### 1. Test Input Inference With Untrained Fallback Model

- Ran inference on several `test_input` samples before trained checkpoints existed.
- Result was only for smoke validation of the pipeline.
- Outputs stayed near `0.5`, which is expected for an untrained model.

### 2. VideoMAE Smoke Training

- Output directory:
  - [videomae_smoke](/G:/내 드라이브/project g/outputs/videomae_smoke)
- Result:
  - epoch `1`
  - `train_loss=0.7456`
  - `val_loss=0.6924`
  - `accuracy=0.5`
  - `f1_score=0.0`
- Meaning:
  - pipeline works end-to-end
  - performance is not meaningful because the dataset is extremely small

### 3. Checkpoint-Based Inference After VideoMAE Smoke Training

- Result file:
  - [test_input_infer.json](/G:/내 드라이브/project g/outputs/videomae_smoke/test_input_infer.json)
- Ran on `test_input` mp4 files.
- All samples were predicted as `real` with confidence roughly in the `0.42-0.47` range.
- This is expected because the smoke training set is too small to form a useful boundary.

### 4. Head-Only Smoke Training

- Output directory:
  - [videomae_head_only_smoke](/G:/내 드라이브/project g/outputs/videomae_head_only_smoke)
- Result:
  - epoch `1`
  - `train_loss=0.7111`
  - `val_loss=0.7076`
  - `accuracy=0.5`
  - `f1_score=0.0`
- Meaning:
  - pretrained `VideoMAE` loaded correctly
  - encoder freeze worked
  - MLP head-only training path works

## Test Status

- Automated tests passed after running them in a writable execution mode.
- Current verified coverage includes:
  - dataset shape
  - metrics output
  - train / eval / infer pipeline
  - JSON output
  - `freeze_encoder` behavior

## Known Constraints

- Current environment is CPU-only.
- Full training on the entire manifest will be slow in the current environment.
- `.gif` support is not implemented in the current video loader, so `I2VGEN_XL` is excluded for now.
- Some Python file-write operations in the sandbox required escalated execution for tests and generated outputs.

## Practical Next Steps

1. Run a larger subset experiment with pretrained `VideoMAE`.
2. Compare:
   - full fine-tuning
   - head-only training
3. Decide whether to keep encoder frozen for early epochs and then unfreeze later.
4. Add explicit XAI output saving and visualization.
5. Add `.gif` loading if `I2VGEN_XL` should be included in training.

## Recommended Baseline Direction

- For stable baseline:
  - pretrained `VideoMAE`
  - MLP classifier
  - binary classification
  - start with head-only or light fine-tuning
- For project differentiation:
  - attach XAI output
  - present frame-level evidence or attention-derived rationale
