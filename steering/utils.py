"""Вспомогательное: журнал с временем и фиксация seed."""

import os, random, time
import numpy as np


class Logger:
    """Печатает строку с временем и дописывает её в run.log."""
    def __init__(self, out_dir):
        """Создаёт папку результатов."""
        self.out_dir, self.t0 = out_dir, time.time()
        os.makedirs(out_dir, exist_ok=True)

    def __call__(self, *a):
        """Пишет сообщение в консоль и в run.log."""
        msg = f"[{time.strftime('%H:%M:%S')} +{(time.time() - self.t0) / 60:5.1f}м] " + " ".join(str(x) for x in a)
        print(msg, flush=True)
        with open(f"{self.out_dir}/run.log", "a") as f:
            f.write(msg + "\n")


def set_seed(seed):
    """Фиксирует seed в random, numpy и torch."""
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
