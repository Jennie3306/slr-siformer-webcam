"""Vá repo Siformer để huấn luyện được trên laptop (giống các bản vá trên Colab). Chạy lại nhiều lần không sao.

    python src/patch_siformer.py

1. torch.load(..., weights_only=False): PyTorch mới mặc định weights_only=True, không nạp được checkpoint của train.py.
2. Bỏ dòng evaluate_top_k(...) bị lỗi IndexError (label[0][0]).
3. Thêm __init__.py để không đụng tên với gói 'datasets' của HuggingFace.
"""
from config import SIFORMER_DIR

train = SIFORMER_DIR / "train.py"
if not train.exists():
    raise SystemExit(f"Không thấy {train}. Clone repo vào external/Siformer trước.")
s = train.read_text(encoding="utf-8")
fixes = [('+ str(i) + ".pth")', '+ str(i) + ".pth", weights_only=False)'),
         ("_, _, top_val_acc = evaluate_top_k(slr_model, val_loader, device)", "pass")]
for old, new in fixes:
    if new in s and old not in s:
        print("đã vá từ trước:", new[:50])
    elif old in s:
        s = s.replace(old, new)
        print("vá:", old[:50], "->", new[:50])
    else:
        print("[CẢNH BÁO] không tìm thấy:", old[:60])
train.write_text(s, encoding="utf-8")
for d in ("datasets", "normalization", "augmentations", "siformer"):
    p = SIFORMER_DIR / d
    if p.is_dir() and not (p / "__init__.py").exists():
        (p / "__init__.py").write_text("", encoding="utf-8")
        print("thêm", p / "__init__.py")
print("Xong.")
