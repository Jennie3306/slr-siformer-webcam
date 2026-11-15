# Webcam-Based Isolated ASL Recognition with Siformer

Recognises 100 isolated American Sign Language words (WLASL100) from a laptop webcam, using
[Siformer](https://github.com/mpuu00001/Siformer) (Pu et al., ACM MM 2024) on MediaPipe keypoints.
Runs in real time on a laptop CPU.

This repository contains everything **around** the model: data preparation, MediaPipe keypoint
extraction in the authors' format, a rebuilt hand pose rectification step, evaluation, robustness and
speed tests, a webcam application, and a test on unseen signers (ASL Citizen). **The model and the
training loop are the authors' original code**, cloned into `external/Siformer` and not modified
apart from two small compatibility patches.

Multidisciplinary Project (Artificial Intelligence Track), Ho Chi Minh City University of Technology, 2026.

## Results

Official WLASL100 split (about half of the videos are still available: 709 train / 176 val / 128 test),
MediaPipe keypoints, mean ± std over three seeds, checkpoints selected on validation:

| Model | Top-1 (%) | Top-5 (%) |
|---|---|---|
| **Siformer** (rectification `dip_only`) | **64.3 ± 1.6** | **85.9 ± 2.1** |
| Siformer (no rectification) | 64.1 ± 0.8 | 85.4 ± 0.5 |
| SPOTER, same data and settings | 43.2 ± 2.4 | 75.5 ± 0.5 |

Unseen signers: 490 home-webcam videos from 11 signers of the ASL Citizen test split (98 words):

| Model | No mirroring | Mirror per video | Mirror per signer |
|---|---|---|---|
| **Siformer** | 50.5 / 66.8 | **58.6 / 80.1** | 57.6 / 78.0 |
| SPOTER | 33.1 / 56.9 | 36.1 / 65.4 | 36.5 / 65.5 |

(top-1 / top-5, %). On right-handed signers Siformer reaches 62.8%, close to its WLASL result; most
of the loss comes from left-handed signers, which mirroring largely recovers.

Other findings:
- The original result is reproduced on the authors' data: 89.25% (paper: 86.50%).
- Siformer keeps 92% of its accuracy without body keypoints (SPOTER: 72%), and loses less than one
  point when a hand is randomly missing in 20–40% of the frames.
- On an Intel Core Ultra 5 laptop CPU the model takes about 22 ms per sign; the prediction appears
  about 0.1 s after the sign ends. MediaPipe (12–14 frames per second) is the slowest stage.
- Hand pose rectification brings no measurable gain on 2D MediaPipe keypoints (the paper reports
  +2.45 points on its own data).

The full analysis is in the report (`report/`).

## Repository layout

```
src/        project code (data, keypoints, rectification, evaluation, webcam app)
tests/      self-test on synthetic data (no MediaPipe or PyTorch needed)
tables/     range-of-motion tables for hand pose rectification (rebuilt, see tables/README.md)
results/    metrics, training logs and figures
notebooks/  Kaggle training notebooks
report/     LaTeX source of the report
external/   authors' code, cloned here (not part of this repository)
data/       videos, keypoints, csv files (not part of this repository)
```

Note: comments and console messages in the code are in Vietnamese.

## Setup

Tested on Windows 11, Python 3.10, PyTorch 2.x (CPU), MediaPipe 0.10.14.

```
conda create -n slr python=3.10 -y
conda activate slr
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
git clone https://github.com/mpuu00001/Siformer external/Siformer
python src/patch_siformer.py      # torch.load(weights_only=False), disable a broken test helper
python tests/selftest.py
python src/check_env.py
```

Do not install `external/Siformer/requirements.txt`: it is a conda export that pins torch 2.1.0.

## Pipeline

**1. Video list.** Download the "WLASL processed" videos (named by `video_id`) with
`WLASL_v0.3.json` and `nslt_100.json`, then build the manifest. Classes that have lost all their
validation or test videos receive one video moved from train.

```
python src/prepare_wlasl.py --json data/WLASL_v0.3.json --videos data/videos --nslt data/nslt_100.json
```

**2. Keypoints.** MediaPipe Holistic, mapped to the 54 keypoints of the authors' format
(y-axis up, 25 fps, padded to 204 frames). Can run in parallel with `--part i --nparts n`.

```
python src/extract_dataset.py --manifest data/manifest.csv --out data/keypoints
python src/build_csv.py --manifest data/manifest.csv --kp data/keypoints --out data/csv
```

**3. Hand pose rectification.** Calls the authors' functions in `external/Siformer/rectifications`
with rebuilt tables. `tables/dip_only` (default) corrects only implausible DIP flexion;
`tables/full` follows Table 1 of the paper and over-corrects on 2D keypoints.

```
python src/rectify_hands.py --inp data/csv/train.csv --out data/csv_rect_dip/train.csv
python src/oversample.py --inp data/csv_rect_dip/train.csv --out data/csv_rect_dip/train_smote.csv
```

(repeat the first command for `val` and `test`).

**4. Training** (GPU; Kaggle T4, about 45 minutes per run), with the authors' `train.py`.
SPOTER is the same script with `--FIM ""`.

```
python -m train --experiment_name siformer_s0 --num_classes 100 --seed 0 \
  --training_set_path data/csv_rect_dip/train_smote.csv \
  --validation_set from-file --validation_set_path data/csv_rect_dip/val.csv \
  --testing_set_path data/csv_rect_dip/test.csv
```

The exact runs are in `notebooks/`: version 1 trains Siformer (seed 0) on the three data versions
(no rectification, `dip_only`, `full`) to choose one on validation; version 2 trains the remaining
Siformer seeds and SPOTER on `dip_only`. The keypoint data they read is not public.

`train.py` reports the best checkpoint on the test set and evaluates with dropout on. This project
instead selects the checkpoint on validation (`src/select_checkpoint.py`) and evaluates once in
eval mode.

**5. Evaluation.**

```
python src/predict_split.py --which siformer --ckpt results/siformer_s0/best.pth --csv data/csv_rect_dip/test.csv --out results/siformer_s0/pred.npz
python src/evaluate.py --pred results/siformer_s0/pred.npz --out results/siformer_s0
python src/evaluate.py --aggregate "results/siformer_s*/metrics.json" --out results/summary_siformer.json
python src/robustness.py --inp data/csv_rect_dip/test.csv --outdir data/csv_rect_dip/robust
python src/benchmark.py --video data/videos/<video>.mp4 --ckpt results/siformer_s0/best.pth --csv data/csv_rect_dip/test.csv --rectify --out results/benchmark_siformer.json
python src/choose_threshold.py --pred results/siformer_s0/pred_val.npz
```

**6. Webcam application.**

```
python src/webcam_app.py --ckpt results/siformer_s0/best.pth --rectify
```

SPACE starts and ends a sign, A toggles automatic segmentation, Q quits. The three best candidates are
shown; below the confidence threshold (0.30) the application answers "not sure".

**7. Unseen signers (ASL Citizen).** Reads only the needed videos from the 43 GB archive on
Microsoft's server, then runs the same pipeline as the webcam.

```
pip install remotezip
python src/external_test.py fetch --per_gloss 5
python src/external_test.py extract
python src/external_test.py build --mirror none
python src/external_test.py build --mirror signer     # or --mirror auto (per video)
```

Then score `data/asl_citizen/test_rect_dip_mirror_*.csv` with `predict_split.py` and `evaluate.py`.

## Data and licences

This repository contains no videos, keypoints or trained checkpoints.

- **WLASL** (Li et al., 2020) is distributed under the C-UDA licence; download it from the official
  sources.
- **ASL Citizen** (Desai et al., 2023, Microsoft) is for non-commercial research only and may not be
  redistributed; `external_test.py fetch` downloads the needed videos directly from Microsoft.
- **Siformer** and **SPOTER** code belong to their authors; clone them as shown in Setup.

## References

- Pu, M., Lim, M. K., Chong, C. Y. Siformer: Feature-isolated Transformer for Efficient Skeleton-based
  Sign Language Recognition. ACM Multimedia 2024.
- Boháček, M., Hrúz, M. Sign Pose-based Transformer for Word-level Sign Language Recognition. WACV
  Workshops 2022.
- Li, D., Rodriguez, C., Yu, X., Li, H. Word-level Deep Sign Language Recognition from Video. WACV 2020.
- Desai, A., et al. ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language
  Recognition. NeurIPS 2023 Datasets and Benchmarks.
- Lugaresi, C., et al. MediaPipe: A Framework for Building Perception Pipelines. 2019.
