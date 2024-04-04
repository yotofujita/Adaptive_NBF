#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import torch
import os
import itertools as it
from glob import glob


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--gpu_start", type=int, default=0, help="args.gpu_start ~ args.gpu_start+3 are used")
    args = parser.parse_args()

    ckpt_root_dir = "/n/work3/sekiguch/data_for_paper/IROS2022/"
    max_epochs = 50
    model_name = "lstm"
    settings = [
        [True, False, True, False],
    ]  # use_BF, use_SV, use_DAN, use_VAD

    for setting in settings:
        use_BF, use_SV, use_DAN, use_VAD = setting

        if use_DAN:
            finetune_list = [True]
        else:
            finetune_list = [True]

        # tuning_data_minute_list = [3, *range(6, 50, 6)]
        tuning_data_minute_list = [[3, 6, 12], [18, 24, 30], [36, 42, 48]][args.idx]
        tuning_data_ratio_list = [0.5]
        batch_size_list = [16]
        lr_list = [1e-3]

        for config in it.product(finetune_list, batch_size_list, lr_list, tuning_data_ratio_list):
            finetune, batch_size, lr, tuning_data_ratio = config

            ckpt_dir = (
                f"{ckpt_root_dir}/{model_name}-BF={use_BF}-SV={use_SV}-DAN={use_DAN}-VAD={use_VAD}/lightning_logs/version_0/checkpoints/"
            )

            ckpt_list = glob(f"{ckpt_dir}/epoch=*.ckpt")
            assert len(ckpt_list) == 1, f"There are multiple ckpt files : {ckpt_list}"
            ckpt = ckpt_list[0]

            for i, tuning_data_minute in enumerate(tuning_data_minute_list):
                if i + 1 == len(tuning_data_minute_list):
                    suffix = ""
                else:
                    suffix = "&"

                os.system(
                    f"CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES={args.gpu_start + i} ./finetune.py {ckpt} --tuning_data_minute {tuning_data_minute} --tuning_data_ratio {tuning_data_ratio} --batch_size {batch_size} --max_epochs {max_epochs} --model_name {model_name}  {'--use_BF' if use_BF else ''} {'--use_SV' if use_SV else ''} {'--use_DAN' if use_DAN else ''} {'--use_VAD' if use_VAD else ''} --finetune {finetune} --gpus 1 {suffix}"
                )
            # exit()
