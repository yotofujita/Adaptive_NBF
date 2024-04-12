#!/usr/bin/env python3
# coding:utf-8
import numpy as np
import speechbrain as sb
import soundfile as sf
import resampy
import sys, os
import json
from tqdm import tqdm
from utils.utility import eval_SDR

EPS = 1e-7


def eval(model, hparams, fname, data_dir, save_dir,
    iteration=1, localization_loss=False, speaker_loss=False, queue_size=10, bp_freq=3
):
    save_dir += f"/{str(model)}_it={iteration}"
    if localization_loss or speaker_loss:
        save_dir += \
            f"_FineTune_loc={localization_loss}_spk={speaker_loss}_" \
            f"queue={queue_size}_bp={bp_freq}"
    save_dir += "/"


    RT60 = fname.split(".")[0].split("_")[-1]
    json_fname = f"{save_dir}/SDR_{RT60}.json"
    if os.path.isfile(json_fname):
        SDR_dict = json.load(open(json_fname, "r"))
    else:
        SDR_dict = {
            "each": {"MPDR": {}, "MVDR": {}},
            "average": {"MPDR": {}, "MVDR": {}},
        }

    with open(fname, "r") as f:
        for line in f.readlines():
            line = line.replace("{data_dir}", data_dir)
            noisy_fname, clean_fname = line.strip().split(" ")
            file_id = noisy_fname.split("/")[-1].split(".wav")[0].replace("mixture_", "")

            if file_id in SDR_dict["each"]["MPDR"].keys():
                continue

            save_MPDR_fname = f"{save_dir}/sep_MPDR__{noisy_fname.split('/')[-1]}"
            save_MVDR_fname = f"{save_dir}/sep_MVDR__{noisy_fname.split('/')[-1]}"
            length_list_fname = \
                clean_fname.replace('clean', 'target_start_idx').replace('wav', 'txt')

            sep_MPDR, sr = sf.read(save_MPDR_fname)
            sep_MVDR, sr = sf.read(save_MVDR_fname)
            clean, sr_clean = sf.read(clean_fname)
            clean = clean[:, 0]
            if sr_clean != 16000:
                clean = resampy.resample(clean, sr_clean, sr, axis=0)

            SDR = eval_SDR(sep_MPDR, clean, length_list_fname)
            SDR_dict["each"]["MPDR"][file_id] = SDR
            SDR_dict["average"]["MPDR"][file_id] = float(np.array(SDR).mean())

            SDR = eval_SDR(sep_MVDR, clean, length_list_fname)
            SDR_dict["each"]["MVDR"][file_id] = SDR
            SDR_dict["average"]["MVDR"][file_id] = float(np.array(SDR).mean())

    print("SDR_dict : ", SDR_dict["average"])
    with open(json_fname, "w") as fj:
        json.dump(SDR_dict, fj, indent=4)


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    device = "cpu"

    from hyperpyyaml import load_hyperpyyaml
    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f, overrides)

    if hparams["model_name"] == "SB_ver1":
        from model_SB_ver1 import SB_ver1
        model = SB_ver1(
            input_length=hparams["segment_size"],
            z_dim = hparams["z_dim"] if "z_dim" in overrides else 128,
        ).eval().to(device)

    elif hparams["model_name"] == "SB_orig":
        from model_SB_orig import SB_orig
        model = SB_orig(
            z_dim = hparams["z_dim"] if "z_dim" in overrides else 30,
            hidden_channel = hparams["hidden_channel"],
            RNN_or_TF =  hparams["RNN_or_TF"],
        ).eval().to(device)

    RT60 = 200
    fname = f"/n/work3/sekiguch/dataset/pyroomacoustics_librispeech/test_data_{RT60}.flist"
    data_dir = f"/n/work3/sekiguch/dataset/pyroomacoustics_librispeech/"
    save_dir = f"/n/work3/sekiguch/data_for_paper/ICASSP2022/pyroomacoustics_librispeech/"

    eval(
        model, hparams, fname, data_dir, save_dir
    )
