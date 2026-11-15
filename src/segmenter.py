"""Tách từng ký hiệu từ luồng webcam liên tục.

Hai chế độ:
- ManualRecorder: bấm phím để bắt đầu / kết thúc (luôn chạy được, dùng làm demo chính).
- AutoSegmenter: tự phát hiện. Tay coi là "đang ký" khi thấy tay và cổ tay cao hơn một đường ngang
  dưới vai. Bắt đầu khi "đang ký" đủ start_frames frame liên tiếp; kết thúc khi không "đang ký"
  đủ end_frames frame liên tiếp, hoặc khi đoạn dài tới max_len.
"""
from collections import deque

import numpy as np

from config import MP_POSE, SEG


def hand_active(pose, left, right, raise_ratio=SEG["raise_ratio"]):
    hands = [h for h in (left, right) if not np.isnan(h).any()]
    if not hands:
        return False
    ls, rs = pose[MP_POSE["left_shoulder"]], pose[MP_POSE["right_shoulder"]]
    if np.isnan(ls).any() or np.isnan(rs).any():
        return True                          # không thấy vai: chỉ cần thấy tay
    width = float(np.hypot(*(ls - rs)))
    line_y = (ls[1] + rs[1]) / 2 + raise_ratio * width   # y hướng xuống
    return any(h[0, 1] < line_y for h in hands)


def _stack(frames):
    return tuple(np.stack(x) for x in zip(*frames))   # (pose, left, right)


class ManualRecorder:
    def __init__(self, max_len=SEG["max_len"]):
        self.max_len = max_len
        self.recording = False
        self.buf = []

    def toggle(self):
        """Bấm lần 1: bắt đầu. Bấm lần 2: kết thúc, trả về đoạn đã ghi (hoặc None)."""
        if not self.recording:
            self.recording, self.buf = True, []
            return None
        self.recording = False
        return _stack(self.buf) if self.buf else None

    def update(self, pose, left, right):
        if self.recording:
            self.buf.append((pose, left, right))
            if len(self.buf) >= self.max_len:
                self.recording = False
                return _stack(self.buf)
        return None


class AutoSegmenter:
    def __init__(self, start_frames=SEG["start_frames"], end_frames=SEG["end_frames"],
                 min_len=SEG["min_len"], max_len=SEG["max_len"], raise_ratio=SEG["raise_ratio"]):
        self.start_frames, self.end_frames = start_frames, end_frames
        self.min_len, self.max_len, self.raise_ratio = min_len, max_len, raise_ratio
        self.reset()

    def reset(self):
        self.recording = False
        self.buf = []
        self.pre = deque(maxlen=self.start_frames)
        self.active_run = 0
        self.idle_run = 0

    @property
    def state(self):
        return "đang ghi" if self.recording else "chờ"

    def update(self, pose, left, right):
        """Gọi mỗi frame. Trả về (pose, left, right) của một ký hiệu khi vừa kết thúc, ngược lại None."""
        act = hand_active(pose, left, right, self.raise_ratio)
        frame = (pose, left, right)
        if not self.recording:
            self.pre.append(frame)
            self.active_run = self.active_run + 1 if act else 0
            if self.active_run >= self.start_frames:
                self.recording, self.buf, self.idle_run = True, list(self.pre), 0
            return None
        self.buf.append(frame)
        self.idle_run = 0 if act else self.idle_run + 1
        if self.idle_run >= self.end_frames or len(self.buf) >= self.max_len:
            seg = self.buf[: len(self.buf) - self.idle_run] if self.idle_run else self.buf
            self.reset()
            if len(seg) >= self.min_len:
                return _stack(seg)
        return None
