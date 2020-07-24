import json
import logging
from pathlib import Path
import random
import tarfile
import tempfile
import warnings
import numpy as np
import pandas as pd
import pandas_path  # Path style access for pandas
from tqdm import tqdm
import torch
import torchvision
import fasttext

data_dir = Path.cwd().parent / "data"
print(data_dir)
img_tar_path = data_dir / "img.tar.gz"
train_path = data_dir / "train.jsonl"
dev_path = data_dir / "dev.jsonl"
test_path = data_dir / "test.jsonl"

