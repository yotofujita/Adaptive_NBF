#! /usr/bin/env python3
# coding:utf-8

from glob import glob
import os, sys
import numpy as np
import collections
import json

if len(sys.argv) == 3:
    speech_root_path = sys.argv[1]
    noise_root_path = sys.argv[2]
else:
    speech_root_path = "/n/work3/sekiguch/dataset/librispeech/LibriSpeech/"
    noise_root_path = "/n/work3/sekiguch/dataset/DEMAND/split_20sec/"

speech_dir_dict = {
    "train":"train-clean-100",
    "valid":"dev-clean",
}
noise_idx_dict = {
    "train": list(range(1, 13)),
    "valid": list(range(13, 16))
}

noise_dir_list = glob(f"{noise_root_path}/*")

for stage in ["train", "valid"]:
    speech_dir = speech_dir_dict[stage]
    speech_list = []
    for speech_filename in glob(f"{speech_root_path}/{speech_dir}/*/*/*.flac"):
        speech_list.append(speech_filename.strip())

    noise_list = [f"{noise_dir}/ch01_{i}.wav" for i in noise_idx_dict[stage] for noise_dir in noise_dir_list]

    speech_idx_list = list(range(len(speech_list)))
    noise_idx_list = list(range(len(noise_list)))

    data_all = collections.OrderedDict()
    for i in range(len(speech_list)):
        data = collections.OrderedDict()
        data["ID"] = i + 1
        data["target"] = speech_list[i]
        speaker_id = speech_list[i].split(".")[0].split("/")[-1]

        interference_idx_list = np.random.choice(
            speech_idx_list, 3, replace=False
        ).tolist()
        try:
            interference_idx_list.remove(i)
        except:
            pass
        data["noise_list"] = [
            speech_list[interference_idx_list[0]],
            speech_list[interference_idx_list[1]],
            noise_list[np.random.choice(noise_idx_list)]
        ]
        data_all[speaker_id] = data

    with open(f"./{stage}_librispeech.json", "w") as f:
        json.dump(data_all, f, indent=4)


# for stage in ["train", "valid"]:
#     speech_dir = speech_dir_dict[stage]
#     speech_list = []
#     for speech_filename in glob(f"{speech_root_path}/{speech_dir}/*/*/*.flac"):
#         speech_list.append(speech_filename.strip())

#     noise_list = [f"{noise_dir}/ch01_{i}.wav" for i in noise_idx_dict[stage] for noise_dir in noise_dir_list]

#     speech_idx_list = list(range(len(speech_list)))
#     noise_idx_list = list(range(len(noise_list)))

#     data_all = collections.OrderedDict()
#     for i in range(20):
#         data = collections.OrderedDict()
#         data["ID"] = i + 1
#         data["target"] = speech_list[i]
#         speaker_id = speech_list[i].split(".")[0].split("/")[-1]

#         interference_idx_list = np.random.choice(
#             speech_idx_list, 3, replace=False
#         ).tolist()
#         try:
#             interference_idx_list.remove(i)
#         except:
#             pass
#         data["noise_list"] = [
#             speech_list[interference_idx_list[0]],
#             speech_list[interference_idx_list[1]],
#             noise_list[np.random.choice(noise_idx_list)]
#         ]
#         data_all[speaker_id] = data

#     with open(f"../data/{stage}_small_librispeech.json", "w") as f:
#         json.dump(data_all, f, indent=4)
