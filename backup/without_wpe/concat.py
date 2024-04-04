#! /usr/bin/env python3
# coding:utf-8

import os, sys
from glob import glob
import soundfile as sf
import numpy as np

data_dir = "/n/work3/sekiguch/data_for_paper/IROS2022/test/FastMNMF-M=5-Ns=5-Nn=0-K=8-it=100-itIVA=50-bit=64/"
data_root_dir = "/n/work3/sekiguch/dataset/Hololens2_RealData_15th_bldg/test/"

spk_id_list = set()
for fname in glob(data_dir + "/sep*.wav"):
    spk_id = fname.split("/")[-1].split("_", 1)[-1].rsplit("_", 3)[0]
    spk_id_list.add(spk_id)

for spk_id in spk_id_list:
    fname_list = []
    for fname in glob(data_dir + f"/sep_{spk_id}*.wav"):
        fname_list.append(fname)
    
    wav_list = []
    for i in range(len(fname_list)):
        fname = data_dir + f"/sep_{spk_id}_{i+1}_azim=0_elev=0.wav"
        wav, sr = sf.read(fname)
        wav_list.append(wav)

    wav_all = np.concatenate(wav_list, axis=0)
    print("wav_all : ",  wav_all.shape)
    save_fname =  f"./sep_{spk_id}_all_azim=0_elev=0.wav"
    sf.write(save_fname, wav_all, sr)

wav_fname = "wav_list_FastMNMF.txt"
with open(wav_fname, "w") as f:
    for fname in glob("./sep*.wav"):
        spk_id = fname.split("sep_")[-1].split("_")[0].split("=")[-1]
        output_transcript_file = fname.replace("sep-", "text-").replace(".wav", ".trn")
        mixture_fname = fname
        json_fname = f"{data_root_dir}/test_target_0deg_{spk_id}.json"
        f.write(f"{spk_id},{output_transcript_file},{fname},{mixture_fname},{json_fname}\n")

os.system(f"CUDA_VISIBLE_DEVICES=5 python3 asr_speechbrain_noSync.py --wav_list {wav_fname}")
ref_trn_fname = "/n/work3/sekiguch/dataset/Hololens2_RealData_15th_bldg/test/ref_test_target_0deg.trn"
for hyp_fname in glob("./hyp_*"):
    output_fname = hyp_fname.replace("hyp_", "sclite_").replace(".trn", ".txt")
    os.system(
        f"sclite -r '{ref_trn_fname}' trn -h '{hyp_fname}' trn -i rm -o all stdout > '{output_fname}'"
    )

