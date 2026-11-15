"""Bước 7: demo nhận diện ký hiệu qua webcam.

    python src/webcam_app.py --ckpt results/siformer_s0/best.pth [--camera 0] [--threads 4]

Phím:  SPACE = bắt đầu/kết thúc ghi (chế độ tay)   A = bật/tắt tự động tách ký hiệu   Q = thoát
Mỗi lần nhận diện được ghi vào results/webcam_log.csv (dùng cho phần thử nghiệm thực tế và đo trễ).
"""
import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from config import CONF_THRESHOLD, LABEL_MAP, RESULTS, TARGET_FPS, TEMPLATE_CFG
from keypoints import Extractor, resample, to_template
from model_adapter import ModelAdapter
from segmenter import AutoSegmenter, ManualRecorder
from template import load_cfg

POSE_LINKS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
HAND_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def draw(frame, pose, left, right):
    h, w = frame.shape[:2]

    def pts(a):
        return [(int(x * w), int(y * h)) if not np.isnan(x) else None for x, y in a]

    for arr, links, col in ((pose, POSE_LINKS, (200, 200, 200)), (left, HAND_LINKS, (60, 60, 230)), (right, HAND_LINKS, (230, 120, 40))):
        p = pts(arr)
        for a, b in links:
            if p[a] and p[b]:
                cv2.line(frame, p[a], p[b], col, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--which", default="siformer", choices=["siformer", "spoter"])
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--condition", default="", help="ghi chú điều kiện thử nghiệm, ví dụ 'du_sang'")
    ap.add_argument("--targets", default=None, help="file txt, mỗi dòng một gloss cần ký (chế độ thử nghiệm thật)")
    ap.add_argument("--repeats", type=int, default=5, help="mỗi từ ký bao nhiêu lần")
    ap.add_argument("--rectify", nargs="?", const="tables/dip_only", default=None, metavar="BANG",
                    help="hiệu chỉnh tư thế tay trước khi dự đoán, dùng ĐÚNG bộ bảng của dữ liệu huấn luyện "
                         "(tables/dip_only hoặc tables/full)")
    args = ap.parse_args()

    cfg = load_cfg(TEMPLATE_CFG)
    names = {int(k): v for k, v in json.load(open(LABEL_MAP, encoding="utf-8")).items()}
    model = ModelAdapter(args.which, args.ckpt, threads=args.threads)
    rect_tables = None
    if args.rectify:
        from rectify_hands import load_tables, rectify_points
        from config import ROOT
        rect_tables = load_tables(ROOT / args.rectify)
    ex = Extractor()
    manual, auto = ManualRecorder(), AutoSegmenter()
    use_auto = False
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    log_path = RESULTS / "webcam_log.csv"
    new_log = not log_path.exists()
    logf = open(log_path, "a", newline="", encoding="utf-8")
    logw = csv.writer(logf)
    if new_log:
        logw.writerow(["time", "condition", "mode", "target", "n_frames", "top1", "p1", "top2", "top3",
                       "mediapipe_ms_mean", "model_ms", "end_to_result_ms"])

    queue = []
    if args.targets:
        words = [w.strip() for w in open(args.targets, encoding="utf-8") if w.strip()]
        unknown = [w for w in words if w not in names.values()]
        if unknown:
            raise SystemExit(f"Các từ không có trong WLASL100: {unknown}")
        queue = [w for w in words for _ in range(args.repeats)]
    n_targets = len(queue)
    mp_ms = deque(maxlen=200)
    fps_t = deque(maxlen=30)
    result = "SPACE: ghi/dung   A: tu dong   Q: thoat"
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        pose, left, right = ex(frame)               # ảnh KHÔNG lật đưa vào MediaPipe
        mp_ms.append((time.perf_counter() - t0) * 1000)
        fps_t.append(time.perf_counter())

        seg = auto.update(pose, left, right) if use_auto else manual.update(pose, left, right)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("a"):
            use_auto, result = not use_auto, ("Tu dong: BAT" if not use_auto else "Tu dong: TAT")
            auto.reset()
        if key == ord(" ") and not use_auto:
            seg = manual.toggle()

        if seg is not None:
            t1 = time.perf_counter()
            P, L, R = seg
            cam_fps = (len(fps_t) - 1) / (fps_t[-1] - fps_t[0]) if len(fps_t) > 1 else TARGET_FPS
            P, L, R = resample(P, L, R, cam_fps, TARGET_FPS)          # đưa về 25 frame/giây như dữ liệu huấn luyện
            pts = to_template(P, L, R, cfg, frame.shape[1], frame.shape[0])
            if rect_tables is not None:
                pts = rectify_points(pts, cfg, rect_tables, aspect=frame.shape[1] / frame.shape[0])
            tm = time.perf_counter()
            prob = model.predict_points(pts, cfg)
            model_ms = (time.perf_counter() - tm) * 1000
            total_ms = (time.perf_counter() - t1) * 1000
            top = np.argsort(-prob)[:3]
            if prob[top[0]] < CONF_THRESHOLD:
                result = f"Khong chac ({names.get(int(top[0]), top[0])} {prob[top[0]]:.0%})"
            else:
                result = "  ".join(f"{names.get(int(i), i)} {prob[i]:.0%}" for i in top)
            target = queue.pop(0) if queue else ""
            logw.writerow([time.strftime("%H:%M:%S"), args.condition, "auto" if use_auto else "manual", target, len(P),
                           names.get(int(top[0]), top[0]), f"{prob[top[0]]:.3f}",
                           names.get(int(top[1]), top[1]), names.get(int(top[2]), top[2]),
                           f"{np.mean(mp_ms):.1f}", f"{model_ms:.1f}", f"{total_ms:.1f}"])
            logf.flush()

        draw(frame, pose, left, right)
        view = cv2.flip(frame, 1)                   # chỉ lật để HIỂN THỊ như gương
        fps = (len(fps_t) - 1) / (fps_t[-1] - fps_t[0]) if len(fps_t) > 1 else 0
        recording = manual.recording or (use_auto and auto.recording)
        status = f"{'TU DONG' if use_auto else 'BAM PHIM'} | {'GHI' if recording else 'cho'} | {fps:.0f} FPS | MediaPipe {np.mean(mp_ms):.0f} ms"
        cv2.putText(view, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if recording else (0, 160, 0), 2)
        cv2.putText(view, str(result), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if n_targets:
            msg = f"Hay ky: {queue[0]} ({n_targets - len(queue) + 1}/{n_targets})" if queue else "Xong! Bam Q de thoat"
            cv2.putText(view, msg, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        cv2.imshow("SLR - Siformer", view)

    cap.release()
    ex.close()
    logf.close()
    cv2.destroyAllWindows()
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
