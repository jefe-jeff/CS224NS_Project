
from collections import OrderedDict
from itertools import chain

import h5py
import math
import json
import torch
import wandb
import numpy as np
import pytorch_lightning as pl
from glob import glob
import librosa
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn.metrics import f1_score
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


from wavy_bert import *

WANDB_NAME = 'jefe-jeff' 

config = {
    'datapath':"/content/gdrive/MyDrive/CS224NS_Data/",
    'p' :0.15,
    'lr' : 4e-5, 
    'lr_decay' : 0.96,
    'weight_decay' : 1e-5,
    'lambda_AC': 1,
    'lambda_LM' : 0.2,
    'max_label_length' : 400,
    'max_w2v_length' : 16000*15,
    'num_warmup_steps' : 8000,
    'num_train_steps' :42000,
    'batch_size': 1
}

run(system="LightningWavyBert", config=config, ckpt_dir='wavy-bert', epochs=20, 
    use_gpu=True)