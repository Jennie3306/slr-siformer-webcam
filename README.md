# Đồ án: nhận diện ngôn ngữ ký hiệu qua webcam với Siformer

Bộ code này lo mọi phần **xung quanh** mô hình: chuẩn bị dữ liệu WLASL100, trích keypoint MediaPipe đúng định dạng
Siformer cần, đánh giá, đo tốc độ, demo webcam, thử nghiệm thật và vẽ hình cho báo cáo.
**Mô hình và vòng huấn luyện dùng nguyên code của tác giả** (github.com/mpuu00001/Siformer), không viết lại.

Mọi lệnh chạy từ thư mục gốc của dự án. Kiểm tra nhanh bộ code: `python tests/selftest.py` (không cần mediapipe/torch).

```
slr_project/
  src/            code của đồ án
  tests/          tự kiểm tra bằng dữ liệu giả
  external/       repo của tác giả: Siformer/, spoter/
  data/           video, json, keypoint, csv
  results/        checkpoint, dự đoán, chỉ số, hình
```

---

Hai nơi gõ lệnh: **[Anaconda Prompt]** trên laptop (luôn bắt đầu bằng `conda activate slr` và
`cd /d D:\Semester\261\DATH\slr_project`) và **[Colab]** cho huấn luyện bằng GPU.

## Bước 0. Cài đặt — [Anaconda Prompt]

```
conda create -n slr python=3.10 -y
conda activate slr
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install tqdm scikit-learn "numpy<2"
git clone https://github.com/mpuu00001/Siformer external/Siformer
python tests/selftest.py        # phải in "tất cả đều đạt"
python src/check_env.py         # phải in "môi trường sẵn sàng"
pip freeze > results/env_laptop.txt
```
KHÔNG cài `external/Siformer/requirements.txt` (danh sách conda, ép torch 2.1.0, sẽ làm hỏng môi trường).

---

## Bước 1. Chạy lại kết quả gốc (tuần 1)

### 1a. Huấn luyện trên dữ liệu của tác giả — [Colab, GPU T4]
Thêm thư mục dữ liệu trong README của Siformer vào My Drive (Add shortcut). Rồi:
```python
from google.colab import drive; drive.mount('/content/drive')
!git clone https://github.com/mpuu00001/Siformer /content/Siformer
%cd /content/Siformer
s = open('train.py').read()
s = s.replace('+ str(i) + ".pth")', '+ str(i) + ".pth", weights_only=False)')            # torch >= 2.6
s = s.replace('_, _, top_val_acc = evaluate_top_k(slr_model, val_loader, device)', 'pass')   # hàm này lỗi
open('train.py', 'w').write(s)
!grep -n "weights_only\|evaluate_top_k(slr" train.py
```
```python
DATA = '/content/drive/MyDrive/<thư mục dữ liệu>'
!python -m train --experiment_name repro_wlasl100 --num_classes 100 --num_worker 2 \
  --training_set_path "$DATA/WLASL100_train_25fps.csv" \
  --validation_set from-file --validation_set_path "$DATA/WLASL100_val_25fps.csv" \
  --testing_set_path "$DATA/WLASL100_val_25fps.csv" 2>&1 | tee repro_train.log
!mkdir -p /content/drive/MyDrive/DATH_results/repro && cp -r out-checkpoints out-img *.log /content/drive/MyDrive/DATH_results/repro/
```
Cấu hình mặc định của train.py chính là cấu hình của bài (100 epoch, lr 1e-4, AdamW, giảm lr ở epoch 60 và 80,
3 encoder, 2 decoder, patience 1). **Đạt** khi dòng "The top result was recorded at" khoảng 0,83–0,87.
Lưu ý: con số đó là checkpoint tốt nhất *chấm trên tập test*, giống cách bài báo làm.

Ghi chú về train.py: tham số kiểu bool chỉ tắt được bằng chuỗi rỗng (`--FIM ""`), `--FIM False` vẫn là True;
dòng log "VALIDATION Top 5 acc" thật ra là val cao nhất, không phải Top-5; hình loss của repo bị trống.

### 1b. Đọc định dạng dữ liệu — [Anaconda Prompt]
Tải file train csv của tác giả từ Drive về `data/siformer_wlasl100/`, rồi:
```
python src/inspect_template.py --csv data/siformer_wlasl100/WLASL100_train_25fps.csv
python src/visualize_template.py --csv data/siformer_wlasl100/WLASL100_train_25fps.csv --row 0 --out results/check_author.png
```
Phải thấy: 54 keypoint, hai tay ứng với hai bên khác nhau, độ dài cố định (dự kiến 204), đầu ở trên trong hình.

### 1c. Kiểm tra adapter và đo tốc độ — [Anaconda Prompt]
Tải một checkpoint (`out-checkpoints/repro_wlasl100/checkpoint_v_*.pth`) từ Drive về `results/repro/`:
```
python src/check_adapter.py --which siformer --ckpt results/repro/checkpoint_v_9.pth --csv data/siformer_wlasl100/WLASL100_val_25fps.csv
python src/benchmark.py --ckpt results/repro/checkpoint_v_9.pth --csv data/siformer_wlasl100/WLASL100_val_25fps.csv --out results/benchmark_repro.json
```
Top-1 của check_adapter phải bằng dòng `checkpoint_v_9  ->  ...` trong log Colab (cùng checkpoint, cùng file).

## Bước 2. Tải và lập danh sách video WLASL100 (tuần 2)

1. Lấy `WLASL_v0.3.json` (và `nslt_100.json` nếu có) từ repo github.com/dxli94/WLASL, đặt vào `data/`.
2. Tải video bằng script của repo WLASL vào `data/videos/` (tên file = video_id). Link chết nhiều thì dùng bản
   đã tải sẵn trên Kaggle, nhưng tên file vẫn phải là video_id.
3. Lập danh sách:
   ```bash
   python src/prepare_wlasl.py --json data/WLASL_v0.3.json --videos data/videos [--nslt data/nslt_100.json]
   ```
   Chép bảng "trong_json / tai_duoc" vào báo cáo.

## Bước 3. Trích keypoint MediaPipe (tuần 3)

```bash
python src/extract_dataset.py --manifest data/manifest.csv --out data/keypoints     # thêm --no_trim nếu video đã cắt sẵn
python src/build_csv.py --manifest data/manifest.csv --kp data/keypoints --out data/csv
```

- Trích chậm (MediaPipe chạy từng frame): mở 2–3 cửa sổ với `--part i --nparts n` để chạy song song. Chạy lại được, không mất phần đã xong.
- Xem `data/extract_log_part*.csv`: video nào có `any_hand_ratio` rất thấp thì mở ra xem.
- **Kiểm tra bằng mắt**: chọn một video có trong cả dữ liệu của tác giả và dữ liệu của mình, vẽ cả hai bằng `visualize_template.py`:
  bố cục giống nhau, tay đỏ/xanh cùng phía. Sai ở đây thì mọi bước sau đều sai.

## Bước 3b. Hiệu chỉnh tư thế tay (kinematic hand pose rectification)

Dùng lại code của tác giả trong `external/Siformer/rectifications/`. Hai bộ bảng biên độ (`tables/full`, `tables/dip_only`) do đồ án tự xây, vì repo không công bố; xem `tables/README.md`.

1. **[Anaconda Prompt]** Thống kê (không ghi file): `python src/rectify_hands.py --inp data/csv/train.csv --tables tables/full --stats results/rectify_stats_train.json`
2. **[Anaconda Prompt]** Tạo 2 bản dữ liệu đã hiệu chỉnh (α = 0,4):
   ```
   for %s in (train val test) do python src\rectify_hands.py --inp data\csv\%s.csv --tables tables\dip_only --out data\csv_rect_dip\%s.csv
   for %s in (train val test) do python src\rectify_hands.py --inp data\csv\%s.csv --tables tables\full --out data\csv_rect_full\%s.csv
   ```
3. Mô hình học trên dữ liệu đã hiệu chỉnh thì lúc chạy webcam phải thêm `--rectify tables/dip_only` (hoặc `tables/full`), đúng bộ bảng đã dùng khi huấn luyện.

## Bước 4. Huấn luyện trên dữ liệu MediaPipe (tuần 4–5)

1. **[Anaconda Prompt]** Cân bằng lớp cho tập train (code Siformer không tự làm SMOTE):
   `python src/oversample.py --inp data/csv/train.csv --out data/csv/train_smote.csv`
2. **[File Explorer]** Đưa `data/csv/` lên Drive, ví dụ `MyDrive/DATH_data/csv/`.
3. **[Colab]** Huấn luyện 3 seed cho Siformer và 3 seed cho SPOTER (SPOTER = cùng repo, `--FIM ""`):
   ```python
   CSV = '/content/drive/MyDrive/DATH_data/csv'
   OUT = '/content/drive/MyDrive/DATH_results'
   for model, fim in (("siformer", "True"), ("spoter", "")):
       for seed in (0, 1, 2):
           name = f"{model}_s{seed}"
           !python -m train --experiment_name {name} --num_classes 100 --num_worker 2 --seed {seed} --FIM "{fim}" \
             --training_set_path "{CSV}/train_smote.csv" \
             --validation_set from-file --validation_set_path "{CSV}/val.csv" \
             --testing_set_path "{CSV}/test.csv" 2>&1 | tee {name}.log
           !mkdir -p "{OUT}/{name}" && cp -r out-checkpoints/{name} {name}.log "{OUT}/{name}/"
   ```
   Con số "top result" train.py in ra lần này **không dùng cho báo cáo** (chọn bằng tập test).
4. **[Anaconda Prompt]** Tải các thư mục `DATH_results/<tên>` về `results/<tên>/`, chọn checkpoint theo **val**:
   ```
   python src/select_checkpoint.py --log results/siformer_s0/siformer_s0.log --ckpt_dir results/siformer_s0/siformer_s0
   ```
   Sinh `results/siformer_s0/best.pth`. Làm cho cả 6 lần chạy.

## Bước 5. Đánh giá (tuần 6)

```bash
# Dự đoán + chỉ số cho từng seed (đổi which/ckpt cho SPOTER)
python src/predict_split.py --which siformer --ckpt results/siformer_s0/best.pth --csv data/csv/test.csv --out results/siformer_s0/pred_test.npz
python src/evaluate.py --pred results/siformer_s0/pred_test.npz --out results/siformer_s0
# Gộp 3 seed -> trung bình ± độ lệch chuẩn (số chính của báo cáo)
python src/evaluate.py --aggregate "results/siformer_s*/metrics.json" --out results/siformer_summary.json
python src/evaluate.py --aggregate "results/spoter_s*/metrics.json"   --out results/spoter_summary.json

# Độ bền: tạo tập test bị xóa keypoint, dự đoán từng file (dùng seed 0)
python src/robustness.py --inp data/csv/test.csv --outdir data/csv/robust
for f in data/csv/test.csv data/csv/robust/*.csv; do
  n=$(basename $f .csv)
  python src/predict_split.py --which siformer --ckpt results/siformer_s0/best.pth --csv $f --out results/robust/siformer/$n/pred.npz
  python src/evaluate.py --pred results/robust/siformer/$n/pred.npz --out results/robust/siformer/$n
done   # làm lại với spoter

# Tốc độ trên laptop (đóng các ứng dụng khác, cắm sạc)
python src/benchmark.py --video data/videos/<một video>.mp4 --ckpt results/siformer_s0/best.pth \
  --csv data/csv/test.csv --out results/benchmark_siformer.json

# Hình cho báo cáo
python src/make_figures.py curves  --log results/siformer_s0/siformer_s0.log --out results/fig_curves_siformer.png
python src/make_figures.py robust  --siformer results/robust/siformer --spoter results/robust/spoter --out results/fig_robust.png
python src/make_figures.py latency --bench results/benchmark_siformer.json --out results/fig_latency.png
```
Hình ma trận nhầm và độ chính xác theo lớp nằm sẵn trong `results/siformer_s0/` sau khi chạy `evaluate.py`.

## Bước 6. Demo webcam (tuần 7–8)

1. Chọn ngưỡng tin cậy trên tập val:
   `python src/predict_split.py ... --csv data/csv/val.csv --out results/siformer_s0/pred_val.npz` rồi
   `python src/choose_threshold.py --pred results/siformer_s0/pred_val.npz`; sửa `CONF_THRESHOLD` trong `src/config.py`.
2. Chạy: `python src/webcam_app.py --ckpt results/siformer_s0/best.pth`
   - SPACE: bắt đầu/kết thúc một ký hiệu (chế độ chính, luôn dùng được). A: bật tự động tách. Q: thoát.
   - Chế độ tự động: chỉnh `SEG` trong `src/config.py` (start_frames, end_frames, raise_ratio) tới khi không bắt nhầm lúc đưa tay lên.
   - `max_len` trong `SEG` đặt bằng số frame lớn nhất của tập train (xem `template_config.json` hoặc dữ liệu của mình).
3. Quay một video demo dự phòng.

## Bước 7. Thử nghiệm thật (tuần 9)

1. Tạo `data/real_words.txt`: 10–20 gloss có trong `data/label_map.json`, mỗi dòng một từ.
2. Mỗi điều kiện chạy một lần (mỗi từ ký 5 lần):
   ```bash
   python src/webcam_app.py --ckpt results/siformer_s0/best.pth --targets data/real_words.txt --repeats 5 --condition du_sang
   python src/webcam_app.py --ckpt results/siformer_s0/best.pth --targets data/real_words.txt --repeats 5 --condition thieu_sang
   python src/webcam_app.py --ckpt results/siformer_s0/best.pth --targets data/real_words.txt --repeats 5 --condition nen_roi
   ```
3. Tổng hợp: `python src/real_test_summary.py` (Top-1, Top-3, độ trễ theo từng điều kiện).

## Bước 8. Viết báo cáo (tuần 10)

Xem `report_outline.md`: mỗi mục ghi rõ lấy số / hình từ file nào.

---

## Lỗi hay gặp

| Hiện tượng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| `check_adapter` lệch với train.py | khác checkpoint hoặc khác file csv | so đúng dòng `checkpoint_v_k  ->` trong log |
| Top-1 trên dữ liệu MediaPipe rất thấp | ánh xạ tay trái/phải hoặc trục y sai | chạy lại `inspect_template.py`, so hình `visualize_template.py` |
| `ImportError: mp.solutions` | mediapipe quá mới | `pip install mediapipe==0.10.14` (Python 3.10) |
| Demo nhận nhầm khi chỉ đưa tay lên | ngưỡng tách ký hiệu quá nhạy | tăng `start_frames`, `min_len` hoặc dùng chế độ SPACE |
| FPS thấp | MediaPipe chiếm phần lớn thời gian | giảm `--width/--height`, đóng ứng dụng khác, đo lại bằng benchmark |

## Thử nghiệm trên người ký mới (ASL Citizen)

ASL Citizen (Microsoft, 2023): video quay bằng webcam tại nhà, người ký Điếc/khiếm thính, không trùng với WLASL.

1. **[Anaconda Prompt]** `pip install remotezip`
2. `python src/external_test.py fetch --per_gloss 5` (chỉ tải các video cần, không tải cả file zip 43 GB)
3. `python src/external_test.py extract` (có thể chạy 2 cửa sổ: `--part 0 --nparts 2` và `--part 1 --nparts 2`)
4. `python src/external_test.py build --mirror none` và `python src/external_test.py build --mirror signer` (lật ngang mọi video của người ký thuận tay trái; `--mirror auto` quyết định riêng từng video, kém ổn định hơn)
5. Chấm bằng `predict_split.py` + `evaluate.py` trên `data/asl_citizen/test_rect_dip_mirror_*.csv`

## Dữ liệu và giấy phép

Repo này chỉ chứa code, bảng biên độ (`tables/`), số liệu kết quả và mã nguồn báo cáo. Không chứa video,
keypoint hay checkpoint:

- **WLASL** (Li et al., 2020) theo giấy phép C-UDA: tự tải bộ "WLASL processed" rồi chạy từ Bước 2.
- **ASL Citizen** (Desai et al., 2023, Microsoft): chỉ dùng cho nghiên cứu phi thương mại, không phân phối lại;
  `src/external_test.py fetch` tự tải đúng các video cần từ máy chủ Microsoft.
- **Siformer** (Pu et al., 2024): code của tác giả, clone vào `external/Siformer` như Bước 0, không chép vào repo.
