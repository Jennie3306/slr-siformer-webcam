"""Trích keypoint bằng MediaPipe và chuyển sang 54 điểm theo khuôn của Siformer.

Dùng CHUNG cho dữ liệu huấn luyện (video) và webcam để tránh lệch dữ liệu.
Quy ước nội bộ: tọa độ chuẩn hóa [0,1] theo ảnh, y hướng xuống, 'left' = tay trái của người ký,
giá trị thiếu = NaN. Việc đổi sang quy ước của khuôn chỉ làm ở to_template().
"""
import numpy as np

from config import HAND_INDEX, MP_POSE
from template import classify_base

N_POSE, N_HAND = 33, 21


def _lm_to_arr(lms, n):
    if lms is None:
        return np.full((n, 2), np.nan, dtype=np.float32)
    return np.array([[p.x, p.y] for p in lms.landmark], dtype=np.float32)


def assign_hands(pose, hand_a, hand_b):
    """Gán hai bàn tay cho bên trái/phải của người ký dựa vào khoảng cách tới cổ tay của pose.

    hand_a, hand_b: nhãn left/right mà MediaPipe trả về. Không tin hoàn toàn nhãn này
    (dễ nhầm khi ảnh bị lật), nên kiểm tra lại bằng vị trí cổ tay trên pose.
    """
    lw, rw = pose[MP_POSE["left_wrist"]], pose[MP_POSE["right_wrist"]]
    if np.isnan(lw).any() or np.isnan(rw).any():
        return hand_a, hand_b
    have_a, have_b = not np.isnan(hand_a).any(), not np.isnan(hand_b).any()

    def d(h, w):
        return float(np.hypot(*(h[0] - w)))

    if have_a and have_b:
        keep = d(hand_a, lw) + d(hand_b, rw)
        swap = d(hand_a, rw) + d(hand_b, lw)
        return (hand_a, hand_b) if keep <= swap else (hand_b, hand_a)
    empty = np.full((N_HAND, 2), np.nan, dtype=np.float32)
    if have_a:
        return (hand_a, empty) if d(hand_a, lw) <= d(hand_a, rw) else (empty, hand_a)
    if have_b:
        return (hand_b, empty) if d(hand_b, lw) <= d(hand_b, rw) else (empty, hand_b)
    return hand_a, hand_b


class Extractor:
    """Bọc MediaPipe Holistic. Gọi extractor(frame_bgr) -> (pose (33,2), left (21,2), right (21,2))."""

    def __init__(self, model_complexity=1, static_image_mode=False):
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            raise ImportError(
                "Bản mediapipe này không còn mp.solutions. Cài một bản 0.10.x còn Holistic "
                "(ví dụ: pip install 'mediapipe==0.10.14') hoặc viết lại hàm này bằng MediaPipe Tasks."
            )
        import cv2
        self._cv2 = cv2
        self._h = mp.solutions.holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def __call__(self, frame_bgr):
        rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._h.process(rgb)
        pose = _lm_to_arr(res.pose_landmarks, N_POSE)
        a = _lm_to_arr(res.left_hand_landmarks, N_HAND)
        b = _lm_to_arr(res.right_hand_landmarks, N_HAND)
        left, right = assign_hands(pose, a, b)
        return pose, left, right

    def close(self):
        self._h.close()


def extract_video(path, start=0, end=None, model_complexity=1):
    """Trích keypoint cho frame [start, end) của một video. Trả về dict mảng + thông tin video."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Không mở được video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ex = Extractor(model_complexity=model_complexity)
    poses, lefts, rights = [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i >= start and (end is None or i < end):
            p, l, r = ex(frame)
            poses.append(p); lefts.append(l); rights.append(r)
        i += 1
        if end is not None and i >= end:
            break
    cap.release()
    ex.close()
    if not poses:
        z = np.zeros((0, N_POSE, 2), np.float32)
        return {"pose": z, "left": z[:, :N_HAND], "right": z[:, :N_HAND], "w": w, "h": h, "fps": fps, "total_frames": total}
    return {
        "pose": np.stack(poses), "left": np.stack(lefts), "right": np.stack(rights),
        "w": w, "h": h, "fps": fps, "total_frames": total,
    }


def resample(pose, left, right, src_fps, dst_fps):
    """Đổi tốc độ khung hình bằng cách chọn frame gần nhất theo thời gian (ví dụ 30 -> 25 frame/giây)."""
    T = len(pose)
    if T == 0 or not src_fps or abs(src_fps - dst_fps) < 0.5:
        return pose, left, right
    n = max(1, int(round(T * dst_fps / src_fps)))
    idx = np.clip(np.round(np.arange(n) * src_fps / dst_fps).astype(int), 0, T - 1)
    return pose[idx], left[idx], right[idx]


def _other(side):
    return "right" if side == "left" else "left"


def to_template(pose, left, right, cfg, w=1, h=1):
    """Chuyển mảng MediaPipe (T,...) sang dict base -> (T,2) theo đúng quy ước của khuôn."""
    T = pose.shape[0]
    hands = {"left": left, "right": right}
    out = {}
    for base in cfg["keypoint_bases"]:
        kind, a, b = classify_base(base)
        if kind == "body":
            side, part = a, b
            if part == "nose":
                arr = pose[:, MP_POSE["nose"]]
            elif part == "neck":
                arr = (pose[:, MP_POSE["left_shoulder"]] + pose[:, MP_POSE["right_shoulder"]]) / 2
            else:
                person = _other(side) if cfg["swap_lr"] else side
                arr = pose[:, MP_POSE[f"{person}_{part}"]]
        elif kind == "hand":
            name, suffix = a, b
            tside = cfg["hand_suffix_side"].get(suffix)
            if tside is None:
                raise KeyError(f"Không biết hậu tố tay '{suffix}' ứng với bên nào; xem lại template_config.json")
            person = _other(tside) if cfg["swap_lr"] else tside
            arr = hands[person][:, HAND_INDEX[name]]
        else:
            # điểm mà Dataset của tác giả không đọc (ví dụ 'root'): ghi giá trị thiếu cho đủ cột
            out[base] = np.full((T, 2), cfg["missing_value"], dtype=np.float64)
            continue
        arr = np.array(arr, dtype=np.float64, copy=True).reshape(T, 2)
        miss = np.isnan(arr).any(axis=1)
        if cfg["y_up"]:
            arr[:, 1] = 1.0 - arr[:, 1]
        if cfg["scale"] == "pixel":
            arr[:, 0] *= w
            arr[:, 1] *= h
        arr[miss] = cfg["missing_value"]
        out[base] = arr
    return out
