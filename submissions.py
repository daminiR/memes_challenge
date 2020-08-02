import pytorch_lightning as pl
import json
import logging
from pathlib import Path
from lav import LanguageAndVisionConcat
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
from dataset import HatefulMemesDataset
from model import HatefulMemesModel
checkpoints = list(Path("../train/model-outputs").glob("*.ckpt"))
print(checkpoints)
assert len(checkpoints) == 1
data_dir = Path.cwd().parent/ "data"
test_path = data_dir / "test.jsonl"
hateful_memes_model = HatefulMemesModel.load_from_checkpoint(checkpoint_path="../train/model-outputs/epoch=0.ckpt")
submission = hateful_memes_model.make_submission_frame(
    test_path
)
print(submission.head())
