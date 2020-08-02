from model import HatefulMemesModel
import pytorch_lightning as pl
import json
import logging
from pathlib import Path
import random
import tarfile
import tempfile
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_path  # Path style access for pandas
from tqdm import tqdm
import torch
import torchvision
import fasttext
data_dir = Path.cwd().parent/ "data"
print(data_dir)
img_tar_path = data_dir / "img.tar.gz"
train_path = data_dir / "train.jsonl"
dev_path = data_dir / "dev.jsonl"
test_path = data_dir / "test.jsonl"
hparams = {

    # Required hparams
    "train_path": train_path,
    "dev_path":dev_path ,
    "img_dir": data_dir,

    # Optional hparams
    "embedding_dim": 150,
    "language_feature_dim": 300,
    "vision_feature_dim": 300,
    "fusion_output_size": 256,
    "output_path": "../train/model-outputs",
    "dev_limit": None,
    "lr": 0.00005,
    "max_epochs": 10,
    "n_gpu": 1,
    "batch_size": 4,
    # allows us to "simulate" having larger batches
    "accumulate_grad_batches": 16,
    "early_stop_patience": 3,
}

hateful_memes_model = HatefulMemesModel(hparams=hparams)
hateful_memes_model.fit()
