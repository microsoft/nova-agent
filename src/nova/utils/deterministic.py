import numpy as np
import torch


def _set_deterministic(SEED: int = 42):
    """
    Set random seeds for reproducibility across numpy and PyTorch.
    """
    import random

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
