import os
import random
import numpy as np
import torch


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TorchVarLogger:
    def __init__(self, filepath, overwrite=False, fsync=False):
        self.filepath = filepath
        self.fsync = bool(fsync)
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        mode = "w" if overwrite else "a"
        self.f = open(filepath, mode, encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def log(self, step, *values, precision=9):
        floats = []
        for v in values:
            if isinstance(v, torch.Tensor):
                v = float(v.detach().cpu())
            else:
                v = float(v)
            floats.append(v)
        vec = ", ".join(f"{v:.{precision}g}" for v in floats)
        self.f.write(f"{int(step)} [{vec}]\n")
        self.f.flush()
        if self.fsync:
            try:
                os.fsync(self.f.fileno())
            except Exception:
                pass

    def close(self):
        try:
            if not self.f.closed:
                self.f.close()
        except Exception:
            pass
