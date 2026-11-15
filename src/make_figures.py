"""Bước 8: vẽ các hình cho báo cáo.

Đường học từ log huấn luyện (dạng SPOTER: "[epoch] TRAIN  loss: .. acc: .." và "[epoch] VALIDATION  acc: .."):
    python src/make_figures.py curves --log results/siformer_s0/train.log --out results/fig_curves_siformer.png
Độ bền (đọc metrics.json của từng tập test bị xóa keypoint):
    python src/make_figures.py robust --siformer results/robust/siformer --spoter results/robust/spoter --out results/fig_robust.png
Độ trễ:
    python src/make_figures.py latency --bench results/benchmark_siformer.json --out results/fig_latency.png
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, GRAY = "#2a78d6", "#9a9a9a"


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5"); ax.set_axisbelow(True)


def curves(args):
    tr_loss, tr_acc, va_acc = {}, {}, {}
    num = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    for line in open(args.log, encoding="utf-8", errors="ignore"):
        m = re.search(r"\[(\d+)\]\s*TRAIN\s+loss:\s*" + num + r"\s+acc:\s*" + num, line)
        if m:
            e = int(m.group(1)); tr_loss[e] = float(m.group(2)); tr_acc[e] = float(m.group(3))
        m = re.search(r"\[(\d+)\]\s*VALIDATION\s+acc:\s*" + num, line)
        if m:
            va_acc[int(m.group(1))] = float(m.group(2))
    if not tr_acc:
        raise SystemExit("Không đọc được log. Mở file log xem định dạng dòng rồi sửa biểu thức trong hàm curves().")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    a1.plot(list(tr_loss), list(tr_loss.values()), color=BLUE); a1.set_title("Loss huấn luyện"); a1.set_xlabel("Epoch")
    a2.plot(list(tr_acc), [v * 100 for v in tr_acc.values()], color=GRAY, label="train")
    if va_acc:
        a2.plot(list(va_acc), [v * 100 for v in va_acc.values()], color=BLUE, label="val")
    a2.set_title("Độ chính xác (%)"); a2.set_xlabel("Epoch"); a2.legend(frameon=False)
    for a in (a1, a2):
        _clean(a)
    plt.tight_layout(); plt.savefig(args.out, dpi=160)
    print(f"Đã lưu {args.out}")


def robust(args):
    conds = ["test", "test_drop_left_hand", "test_drop_right_hand", "test_drop_body", "test_drop_random20", "test_drop_random40"]
    labels = ["Đầy đủ", "Mất tay trái", "Mất tay phải", "Mất thân", "Mất tay 20%", "Mất tay 40%"]
    models = [("Siformer", args.siformer, BLUE), ("SPOTER", args.spoter, GRAY)]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    w = 0.38
    for k, (name, folder, col) in enumerate(models):
        if not folder:
            continue
        vals = []
        for c in conds:
            f = Path(folder) / c / "metrics.json"
            vals.append(json.load(open(f))["pi_top1"] * 100 if f.exists() else 0)
        xs = [i + (k - 0.5) * w for i in range(len(conds))]
        ax.bar(xs, vals, w, color=col, label=name)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.8, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Top-1 (%)"); ax.legend(frameon=False); _clean(ax)
    plt.tight_layout(); plt.savefig(args.out, dpi=160)
    print(f"Đã lưu {args.out}")


def latency(args):
    b = json.load(open(args.bench))
    parts = [("MediaPipe / frame", "mediapipe_per_frame_ms"), ("Mô hình / ký hiệu", "model_forward_ms"),
             ("Sau khi ký xong (đầy đủ)", "after_sign_total_ms")]
    parts = [(n, b[k]) for n, k in parts if k in b]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    y = list(range(len(parts)))[::-1]
    ax.barh(y, [p[1]["mean"] for p in parts], color=BLUE, height=0.5, label="trung bình")
    ax.scatter([p[1]["p95"] for p in parts], y, color="black", zorder=3, label="P95")
    for yi, p in zip(y, parts):
        ax.text(p[1]["p95"] * 1.03 + 1, yi, f"{p[1]['mean']:.1f} / {p[1]['p95']:.1f} ms", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels([p[0] for p in parts]); ax.set_xlabel("ms")
    ax.set_xlim(0, max(p[1]["p95"] for p in parts) * 1.45)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2); _clean(ax)
    plt.tight_layout(); plt.savefig(args.out, dpi=160)
    print(f"Đã lưu {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("curves"); c.add_argument("--log", required=True); c.add_argument("--out", required=True)
    r = sub.add_parser("robust"); r.add_argument("--siformer"); r.add_argument("--spoter"); r.add_argument("--out", required=True)
    l = sub.add_parser("latency"); l.add_argument("--bench", required=True); l.add_argument("--out", required=True)
    args = ap.parse_args()
    {"curves": curves, "robust": robust, "latency": latency}[args.cmd](args)


if __name__ == "__main__":
    main()
