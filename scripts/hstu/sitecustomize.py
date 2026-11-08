from __future__ import annotations

import os
import random
import logging


seed = int(os.environ.get("HSTU_REPRO_SEED", "42"))
random.seed(seed)
os.environ.setdefault("PYTHONHASHSEED", str(seed))

try:
    import numpy as np

    np.random.seed(seed)
except ImportError:
    pass

try:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.getLogger().setLevel(logging.INFO)
except ImportError:
    pass
