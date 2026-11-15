"""Lớp trung gian giữa code của đồ án và code của tác giả Siformer.

Mọi bước chạy mô hình (dự đoán tập test, đo tốc độ, webcam) đều đi qua ĐÚNG Dataset của tác giả
(datasets/czech_slr_dataset.py: chuẩn hóa thân, chuẩn hóa tay, trừ 0,5), nên dữ liệu giống hệt lúc huấn luyện.

Đã đối chiếu với code thật của repo:
- Dataset: CzechSLRDataset(path) trả về (tay_trái, tay_phải, thân, nhãn); nhãn dùng nguyên giá trị trong csv.
- Gọi mô hình: model(l_hand[None], r_hand[None], body[None], training=False) như hàm evaluate trong siformer/utils.py.
- Checkpoint: torch.save(cả mô hình) -> nạp bằng torch.load(..., weights_only=False).
Baseline SPOTER cũng huấn luyện trong repo Siformer (--FIM ""), nên dùng chung adapter này.
Chạy check_adapter.py để chắc: Top-1 qua adapter phải bằng con số train.py in ra.
"""
import importlib
import sys
import tempfile
from pathlib import Path

import numpy as np

from config import SIFORMER_DIR

REPOS = {
    "siformer": {"dir": SIFORMER_DIR, "dataset_module": "datasets.czech_slr_dataset",
                 "dataset_class": "CzechSLRDataset", "dataset_kwargs": {}},
    # baseline: lớp SpoTer có sẵn trong repo Siformer, huấn luyện bằng train.py với --FIM ""
    "spoter": {"dir": SIFORMER_DIR, "dataset_module": "datasets.czech_slr_dataset",
               "dataset_class": "CzechSLRDataset", "dataset_kwargs": {}},
}


class ModelAdapter:
    def __init__(self, which, ckpt, device="cpu", threads=None, seed=0):
        import torch
        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        self.cfg = REPOS[which]
        repo = str(Path(self.cfg["dir"]).resolve())
        # Thư mục 'datasets' của tác giả thiếu __init__.py nên có thể bị gói 'datasets' của HuggingFace che mất
        for d in ("datasets", "normalization", "augmentations", "siformer"):
            p = Path(repo) / d
            if p.is_dir() and not (p / "__init__.py").exists():
                (p / "__init__.py").write_text("")
        if repo not in sys.path:
            sys.path.insert(0, repo)   # đặt trước để không lẫn với gói 'datasets' của HuggingFace
        mod = importlib.import_module(self.cfg["dataset_module"])
        self.Dataset = getattr(mod, self.cfg["dataset_class"])
        self.device = device
        # ProbSparse attention lấy mẫu ngẫu nhiên mỗi lần chạy -> cố định seed trước mỗi dự đoán để kết quả lặp lại được
        self.seed = seed
        self.model = self.load_model(ckpt)

    # ---- 3. nạp checkpoint
    def load_model(self, ckpt):
        obj = self.torch.load(ckpt, map_location=self.device, weights_only=False)
        if isinstance(obj, self.torch.nn.Module):
            model = obj
        else:
            raise RuntimeError("Checkpoint không phải cả mô hình; hãy dùng file .pth do train.py của Siformer lưu.")
        model.to(self.device)
        model.eval()
        return model

    def dataset(self, csv_path):
        return self.Dataset(str(csv_path), **self.cfg["dataset_kwargs"])

    # ---- 2. gọi mô hình: giống hàm evaluate trong siformer/utils.py
    def forward(self, item):
        """item: bộ (tay_trái, tay_phải, thân, nhãn) mà Dataset trả về cho MỘT mẫu. Trả về logits 1 chiều (numpy)."""
        l_hand, r_hand, body = (t.float().unsqueeze(0).to(self.device) for t in item[:3])
        if self.seed is not None:
            self.torch.manual_seed(self.seed)
        with self.torch.no_grad():
            out = self.model(l_hand, r_hand, body, training=False)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.reshape(-1).float().cpu().numpy()

    @staticmethod
    def label_of(item):
        return int(np.asarray(item[3]).reshape(-1)[0])

    def predict_csv(self, csv_path, limit=None):
        """Dự đoán mọi mẫu trong một csv. Trả về list (nhãn đúng, logits, thời gian ms)."""
        import time
        ds = self.dataset(csv_path)
        n = len(ds) if limit is None else min(limit, len(ds))
        res = []
        for i in range(n):
            item = ds[i]
            t = time.perf_counter()
            logits = self.forward(item)
            res.append((self.label_of(item), logits, (time.perf_counter() - t) * 1000))
        return res

    def predict_points(self, points, cfg):
        """Dự đoán một chuỗi keypoint (dict base -> (T,2), đã ở 25 frame/giây) qua đúng Dataset của tác giả.

        write_csv tự đệm/cắt về độ dài cố định của khuôn (204 frame) nếu có."""
        from template import write_csv
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "one.csv"
            write_csv([{"label": 0, "points": points, "w": 1, "h": 1, "fps": 30}], cfg, p)
            item = self.dataset(p)[0]
        logits = self.forward(item)
        z = logits - logits.max()
        prob = np.exp(z) / np.exp(z).sum()
        return prob
