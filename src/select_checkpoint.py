"""Chọn checkpoint theo tập VAL từ file log của train.py (không chọn theo tập test).

    python src/select_checkpoint.py --log results/siformer_s0/train.log --ckpt_dir results/siformer_s0/out-checkpoints/siformer_s0

train.py của Siformer lưu checkpoint_v_<k>.pth = mô hình có val cao nhất trong mỗi khối 10 epoch, rồi chấm TẤT CẢ
trên tập test và báo con số cao nhất (tức là chọn bằng tập test, con số bị lạc quan). Script này chỉ đọc val:
tìm epoch có val cao nhất, lấy đúng file checkpoint đã lưu ở epoch đó và chép thành best.pth.
"""
import argparse
import re
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--ckpt_dir", required=True, help="thư mục chứa checkpoint_v_*.pth của lần chạy này")
    ap.add_argument("--out", default=None, help="mặc định: <thư mục của log>/best.pth")
    args = ap.parse_args()

    val, saved = {}, {}
    for line in open(args.log, encoding="utf-8", errors="ignore"):
        m = re.search(r"\[(\d+)\]\s*VALIDATION\s+acc:\s*([0-9.eE+-]+)", line)
        if m:
            val[int(m.group(1))] = float(m.group(2))
        m = re.search(r"Save checkpoint for \[(\d+)\] as .*?(checkpoint_v_\d+\.pth)", line)
        if m:
            saved[int(m.group(1))] = m.group(2)
    if not val or not saved:
        raise SystemExit("Không đọc được log: cần các dòng 'VALIDATION  acc' và 'Save checkpoint for'.")
    cand = {e: v for e, v in val.items() if e in saved}
    best_epoch = max(cand, key=lambda e: (cand[e], -e))
    src = Path(args.ckpt_dir) / saved[best_epoch]
    out = Path(args.out) if args.out else Path(args.log).parent / "best.pth"
    shutil.copy(src, out)
    print(f"Val cao nhất {cand[best_epoch]:.4f} ở epoch {best_epoch} -> {src.name} -> {out}")


if __name__ == "__main__":
    main()
