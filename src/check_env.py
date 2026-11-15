"""Bước 0: kiểm tra môi trường trên laptop.

    python src/check_env.py            # kiểm tra thư viện, repo, webcam, MediaPipe
    python src/check_env.py --no_cam   # bỏ qua webcam

Mỗi dòng in OK hoặc LỖI kèm cách sửa. Gửi toàn bộ phần in ra nếu cần hỏi.
"""
import argparse
import platform
import sys
import time

from config import SIFORMER_DIR

bad = []


def report(ok, msg, fix=""):
    print(("OK   " if ok else "LỖI ") + msg + ("" if ok or not fix else f"\n     -> {fix}"))
    if not ok:
        bad.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_cam", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    v = sys.version_info
    report(v[:2] == (3, 10), f"Python {platform.python_version()} ({sys.executable})",
           "chưa vào đúng môi trường: chạy 'conda activate slr' rồi thử lại")
    report(all(ord(c) < 128 for c in str(SIFORMER_DIR)), f"đường dẫn dự án không dấu: {SIFORMER_DIR.parent.parent}",
           "MediaPipe có thể lỗi với đường dẫn có dấu tiếng Việt; chuyển thư mục sang ví dụ D:\\DATH\\slr_project")

    for name in ("numpy", "pandas", "matplotlib", "cv2", "torch", "mediapipe"):
        try:
            m = __import__(name)
            report(True, f"{name} {getattr(m, '__version__', '')}")
        except Exception as e:
            fix = {"cv2": "pip install opencv-python", "torch": "pip install torch --index-url https://download.pytorch.org/whl/cpu",
                   "mediapipe": "pip install mediapipe==0.10.14"}.get(name, f"pip install {name}")
            report(False, f"{name}: {e}", fix)

    try:
        import mediapipe as mp
        report(hasattr(mp, "solutions"), "mediapipe còn mp.solutions (Holistic)",
               "pip install mediapipe==0.10.14")
    except Exception:
        pass

    report((SIFORMER_DIR / "train.py").exists(), f"repo Siformer ở {SIFORMER_DIR}",
           "git clone https://github.com/mpuu00001/Siformer external/Siformer (thư mục phải tên đúng 'Siformer')")
    for f in ("datasets/czech_slr_dataset.py", "siformer/model.py", "siformer/utils.py"):
        report((SIFORMER_DIR / f).exists(), f"có file {f} trong repo Siformer", "clone lại repo Siformer")

    if args.no_cam:
        print("Bỏ qua kiểm tra webcam.")
    else:
        try:
            import cv2
            cap = cv2.VideoCapture(args.camera)
            ok, frame = cap.read()
            cap.release()
            report(ok, f"webcam {args.camera} đọc được frame" + (f" {frame.shape[1]}x{frame.shape[0]}" if ok else ""),
                   "đóng Zoom/Meet/Camera; Windows: Settings > Privacy > Camera > cho phép ứng dụng desktop; thử --camera 1")
            if ok:
                from keypoints import Extractor
                ex = Extractor()
                for _ in range(5):
                    ex(frame)
                t = time.perf_counter()
                for _ in range(20):
                    pose, left, right = ex(frame)
                ms = (time.perf_counter() - t) / 20 * 1000
                ex.close()
                seen = not (pose != pose).all()
                report(True, f"MediaPipe chạy được: khoảng {ms:.0f} ms mỗi frame; "
                             f"{'thấy người' if seen else 'không thấy người (ngồi trước webcam rồi chạy lại để chắc)'}")
        except Exception as e:
            report(False, f"webcam / MediaPipe: {e}")

    print("\nKẾT QUẢ:", "môi trường sẵn sàng" if not bad else f"{len(bad)} mục cần sửa")


if __name__ == "__main__":
    main()
