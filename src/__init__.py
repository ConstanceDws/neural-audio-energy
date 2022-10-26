from .preprocess import get_audio_preprocess
from .checkpoint import PeriodicCheckpoint
import numpy as np


def accumulate_metric_dict(accumulated_dict, new_dict, iteration):
    new_dict = {k: np.mean(v) for k, v in new_dict.items()}
    if accumulated_dict is not None:
        accumulated_dict = {
            k: v + (new_dict[k] - v) / (iteration + 1)
            for k, v in accumulated_dict.items()
        }
    else:
        accumulated_dict = new_dict
    return accumulated_dict
