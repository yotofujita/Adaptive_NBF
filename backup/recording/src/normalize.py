#! /usr/bin/env python3
# coding:utf-8

import os, sys
import soundfile as sf
import random
from glob import glob
import numpy as np

data_dir = f"{os.environ['HOME']}/dataset/librispeech/test-clean/"

# for filename in glob(f"{data_dir}/*/*/*.flac"):
#     wav, sr = sf.read(filename)
#     os.system(f"rm {filename}")
#     sf.write(filename.replace(".flac", ".wav"), wav, sr)
# exit()

for filename in glob(f"{data_dir}/*/*/*.wav"):
    wav, sr = sf.read(filename)
    wav /= np.abs(wav).max() * 1.2
    sf.write(filename, wav, sr)
