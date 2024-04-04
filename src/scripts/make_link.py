#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import torch
from Lightning_CSS import Lightning_CSS
import pytorch_lightning as pl
import json
import h5py
import os
from glob import glob
import torchaudio
import utility
import time


if __name__ == "__main__":
    mix_dir="/n/work3/sekiguch/data_for_paper/IROS2022/FastMNMF-M=5-Ns=5-Nn=0-K=8-it=100-itIVA=50-bit=64_reset_wpe/"
    sep_dir="/n/work3/sekiguch/data_for_paper/IROS2022/FastMNMF-M=5-Ns=5-Nn=0-K=8-it=100-itIVA=50-bit=64_reset/"

    for fname in glob(f"{sep_dir}/sep*.wav"):
        target = fname.replace(sep_dir, mix_dir)
        os.system(f"ln -s {fname} {target}")