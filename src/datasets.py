import os
import torch
from torch.utils.data import random_split
from torch.utils.data import Subset

import numpy as np

from udls import SimpleDataset
from .preprocess import get_audio_preprocess

datasets = {
    "ljspeech": "/data/datasets/waveform/ljspeech",
    "sc09": "/data/datasets/waveform/sc09"
}


def get_dataset(dataset_name, sr, n_signal, n_mel, stride, deterministic=False):
    
    if deterministic:
        dataset_path = f"{dataset_name}_{sr}{n_signal}{n_mel}{stride}_det"
    else:
        dataset_path = f"{dataset_name}_{sr}{n_signal}{n_mel}{stride}"

    dataset = SimpleDataset(
        os.path.join(
            "/data/tmp/",
            dataset_path,
        ),
        datasets[dataset_name],
        preprocess_function=get_audio_preprocess(
            sr,
            n_signal,
            n_mel,
            stride,
        ),
        split_set="full",
    )

    min_value = 0
    max_value = 0
    for _, S in dataset:
        min_value = min(min_value, np.min(S))
        max_value = max(max_value, np.max(S))

    def normalize(data):
        x, S = data
        S = (S - min_value) / (max_value - min_value)
        return x, S

    dataset.transforms = normalize

    if deterministic:
        print("\n[INFO] Fixing the train/test split\n")
        train, val = deterministic_split(dataset, num_test_examples=(len(dataset) // 5))
    else:
        print("\n[INFO] Shuffling the train/test split\n")
        train, val = random_split(
            dataset,
            [len(dataset) - len(dataset) // 5,
             len(dataset) // 5],
            generator=torch.Generator().manual_seed(42),
        )

    return train, val


def deterministic_split(dataset, num_test_examples):
    num_train_examples = len(dataset) - num_test_examples
    train_indices = range(num_train_examples)
    val_indices = range(num_train_examples, len(dataset))
    assert(len(train_indices) + len(val_indices) == len(dataset))
    return [Subset(dataset, train_indices), Subset(dataset, val_indices)]