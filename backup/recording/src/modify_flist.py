#! /usr/bin/env python3
# coding:utf-8

import os, sys
import soundfile as sf
import random
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type= str, default=f"{os.environ['HOME']}/dataset/recording_15th_bldg/0deg/")
args = parser.parse_args()

# test-cleanの性別情報を取得
ID_gender_dict = {}
with open("../SPEAKERS.TXT") as f:
    for line in f.readlines():
        if ";" not in line:
            ID, gender, dataset, length = line.replace(" ", "").split("|")[:4]
            if dataset == "test-clean":
                ID_gender_dict[ID] = gender

# target_flist_fname = "../flist/target_fname_list.txt"
# target_flist_fname_new = "../flist/target_fname_list_new.txt"
# with open(target_flist_fname, "r") as f:
#     with open(target_flist_fname_new, "w") as f2:
#         for line in f.readlines():
#             tmp = line.replace(" ", "").split(",")
#             spk_id = tmp[0].split("-")[0]
#             gender = ID_gender_dict[spk_id]
#             print(gender)
#             f2.write(f"{tmp[0]}, {gender}, {tmp[2]}, {tmp[3]}, {tmp[4]}")

for fname in glob(args.data_dir + "*_F_*.wav"):
    if "tsp" in fname:
        continue
    spk_id = fname.split("/")[-1].split("-")[0]
    gender = ID_gender_dict[spk_id]
    if gender == "M":
        fname2 = fname.replace("_F_", f"_{gender}_")
        print(f"mv {fname} {fname2}")
        os.system(f"mv {fname} {fname2}")
