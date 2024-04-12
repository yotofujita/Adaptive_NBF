#! /usr/bin/env python3
# coding: utf-8

import argparse
import yaml

import torch
import pytorch_lightning as pl
from pytorch_lightning.accelerators import find_usable_cuda_devices

from utils.dict_struct import Struct
from utils.import_utils import instantiate


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./src/config/train.yaml")
parser.add_argument("--resume_from_last", action="store_true")
parser.add_argument("--resume_from_ckpt", action="store_true")
parser.add_argument("--save_root_dir", type=str, default="./data/IROS2022/")
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()
    
with open(args.config) as f:
    config = vars(args) | Struct(yaml.load(f, yaml.SafeLoader))

torch.cuda.empty_cache()
pl.seed_everything(0)

dm = instantiate(config | config.data_module)

model = instantiate(config | config.lightning_module)

default_root_dir = f"{config.save_root_dir}/{str(model)}/"

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor="valid_sep_loss", save_last=True, every_n_epochs=30)

trainer = pl.Trainer(
    accelerator=config.device, 
    devices=find_usable_cuda_devices(config.n_gpus), 
    # strategy="ddp",
    callbacks=[checkpoint_callback],
    default_root_dir=default_root_dir,
    gradient_clip_val=1.0,
    max_epochs=config.epochs
)

trainer.fit(model, dm)
