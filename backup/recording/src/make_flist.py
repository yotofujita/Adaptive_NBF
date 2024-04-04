#! /usr/bin/env python3
# coding:utf-8

import os, sys
import soundfile as sf
import random
from glob import glob

data_dir = f"{os.environ['HOME']}/dataset/librispeech/test-clean/"

# for filename in glob(f"{data_dir}/*/*/*.flac"):
    # wav, sr = sf.read(filename)
    # os.system(f"rm {filename}")
    # sf.write(filename.replace(".flac", ".wav"), wav, sr)
# exit()

# test-cleanの性別情報を取得
ID_gender_dict = {}
with open(f"{os.environ['HOME']}/dataset/librispeech/SPEAKERS.TXT") as f:
    for line in f.readlines():
        if ";" not in line:
            ID, gender, dataset, length = line.replace(" ", "").split("|")[:4]
            if dataset == "test-clean":
                ID_gender_dict[ID] = gender

# F, Mを４人ずつ選択
ID_list  =list(ID_gender_dict.keys())
random.shuffle(ID_list)
count_dict = {"F": 0, "M": 0}
selected_ID_dict = {"F": [], "M": []}
for i in range(len(ID_list)):
    id = ID_list[i]
    gender = ID_gender_dict[id]
    if count_dict[gender] < 4:
        selected_ID_dict[gender].append(id)
        count_dict[gender] += 1
    if count_dict["F"] > 4 and count_dict["M"] > 4:
        break
F_known = selected_ID_dict["F"][:2]
F_unknown = selected_ID_dict["F"][2:]
M_known = selected_ID_dict["M"][:2]
M_unknown = selected_ID_dict["M"][2:]
target_list = F_known + F_unknown + M_known + M_unknown

# 残りから雑音に使う21人(3人* 7方向)を選択
interference_list = []
interference_count = 0
for id in ID_list:
    if id not in target_list:
        interference_list.append(id)
        interference_count += 1
    if interference_count > 21:
        break

with open("target_fname_list.txt", "w") as f:
    for tmp in ["F_known", "F_unknown", "M_known", "M_unknown"]:
        gender, known = tmp.split("_")
        length = 0
        for i in range(2):
            spk = eval(tmp)[i]
            for fname in glob(f"{data_dir}/{spk}/*/*.wav"):
                wav, sr = sf.read(fname)
                length += len(wav) / sr
                data_id = fname.split("/")[-1].split(".")[0]
                f.write(f"{data_id}, {gender}, {known}, {round(len(wav) / sr, 1)}, {fname}\n")
        print(tmp, " : ", length)

for i in range(7):
    length = 0
    with open(f"interference_fname_list_{i+1}.txt", "w") as f:
        for spk in interference_list[i*3:(i+1)*3]:
            for fname in glob(f"{data_dir}/{spk}/*/*.wav"):
                wav, sr = sf.read(fname)
                length += len(wav) / sr
                data_id = fname.split("/")[-1].split(".")[0]
                f.write(f"{data_id}, {round(len(wav) / sr, 1)}, {fname}\n")
        print(i, " : ", length)

with open("noise_fname_list.txt", "w") as f:
    for fname in glob("/home/xks/dataset/CHiME3/backgrounds/*.wav"):
        fname = fname.strip()
        situ = fname.split(".")[0].split("_")[-1]
        wav, sr = sf.read(fname)
        wav = wav[:sr * 60 * 2]
        sf.write(fname, wav, sr)
        data_id = fname.split("/")[-1].split(".")[0]
        f.write(f"{data_id}, {fname}\n")

print(target_list, interference_list)

