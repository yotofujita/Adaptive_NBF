#! /usr/bin/env python3
# coding:utf-8

import os, sys
from glob import glob

root_dir = "/home/sekiguch/IROS/lstm-BF=True-SV=False-DAN=True-VAD=False/lightning_logs/version_0/checkpoints/"
for wpe_dir in glob(f"{root_dir}/wpe_incremental*ratio=0.5*"):
    for ckpt_fname in glob(f"{wpe_dir}/checkpoints/*-v1.ckpt"):
        os.system(f"mv {ckpt_fname} {ckpt_fname.replace('-v1.ckpt', '.ckpt')}")