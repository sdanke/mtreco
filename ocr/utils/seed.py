import random
import torch
import torch.utils.data
import numpy as np


def apply_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
