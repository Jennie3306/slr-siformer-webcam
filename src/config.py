"""Cấu hình chung: đường dẫn, ánh xạ keypoint, tham số tách ký hiệu."""
import sys
from pathlib import Path

# Windows: cửa sổ lệnh / pipe mặc định dùng cp1252, không in được tiếng Việt -> ép UTF-8
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
EXTERNAL = ROOT / "external"
SIFORMER_DIR = EXTERNAL / "Siformer"   # git clone https://github.com/mpuu00001/Siformer external/Siformer
SPOTER_DIR = EXTERNAL / "spoter"       # không bắt buộc: baseline SPOTER chạy ngay trong repo Siformer (--FIM "")

TEMPLATE_CFG = DATA / "template_config.json"   # do inspect_template.py sinh ra
LABEL_MAP = DATA / "label_map.json"            # do prepare_wlasl.py sinh ra
NUM_CLASSES = 100
TARGET_FPS = 25          # dữ liệu Siformer là "..._25fps.csv": mọi chuỗi được đưa về 25 frame/giây

# ---- MediaPipe Pose: chỉ số các điểm dùng tới (theo người trong ảnh: left = tay trái của người ký)
MP_POSE = {
    "nose": 0,
    "left_eye": 2, "right_eye": 5,
    "left_ear": 7, "right_ear": 8,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
}

# ---- Tên điểm thân trong dữ liệu SPOTER/Siformer (kiểu Apple Vision) -> phần cơ thể
# "leftEye" -> side "left", part "eye". "nose", "neck" không có bên.
BODY_PARTS = {"Eye": "eye", "Ear": "ear", "Shoulder": "shoulder", "Elbow": "elbow", "Wrist": "wrist"}

# ---- Tên điểm tay kiểu Apple Vision -> chỉ số MediaPipe Hands (0..20). Ghép theo TÊN, không theo thứ tự.
HAND_INDEX = {
    "wrist": 0,
    "thumbCMC": 1, "thumbMP": 2, "thumbIP": 3, "thumbTip": 4,
    "indexMCP": 5, "indexPIP": 6, "indexDIP": 7, "indexTip": 8,
    "middleMCP": 9, "middlePIP": 10, "middleDIP": 11, "middleTip": 12,
    "ringMCP": 13, "ringPIP": 14, "ringDIP": 15, "ringTip": 16,
    "littleMCP": 17, "littlePIP": 18, "littleDIP": 19, "littleTip": 20,
}

# ---- Tách ký hiệu tự động từ webcam (đơn vị: frame). Chỉnh sau khi thử thực tế.
SEG = {
    "start_frames": 3,     # tay "đang ký" liên tiếp bao nhiêu frame thì bắt đầu ghi
    "end_frames": 8,       # tay hạ xuống / biến mất liên tiếp bao nhiêu frame thì kết thúc
    "min_len": 12,         # đoạn ngắn hơn thì bỏ (thường là đưa tay lên rồi hạ)
    "max_len": 200,        # tính theo frame webcam; sau khi đổi về 25 frame/giây phải không quá độ dài cố định (204)
    "raise_ratio": 1.2,    # tay coi là "đang ký" nếu cổ tay cao hơn: giữa vai + raise_ratio * độ rộng vai
}
CONF_THRESHOLD = 0.30      # xác suất top-1 thấp hơn thì hiện "không chắc"; chọn lại theo tập val
