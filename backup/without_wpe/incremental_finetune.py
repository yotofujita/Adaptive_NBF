#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import torch
from lightning_modules.Lightning_CSS import Lightning_CSS
import pytorch_lightning as pl
import json
import h5py
import os
from glob import glob
from tqdm import tqdm
import torchaudio
import utils.utility as utility


class FinetuneDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        count,
        tuning_data_dir,
        use_BF,
        use_SV,
        use_DAN,
        use_VAD,
        window_len=48128,
        overlap_ratio=0.5,
        tuning_data_interval_minute=3,
        tuning_data_minute=12,
        tuning_data_ratio=0.5,
        dataset=["wsj0_chime3", "librispeech"][0],
        step=["train", "valid"][0],
    ):
        second_per_wav = 9
        n_sample_per_wav = window_len * 3

        self.use_BF = use_BF
        self.use_SV = use_SV
        self.use_DAN = use_DAN
        self.use_VAD = use_VAD
        shift = int(window_len * (1 - overlap_ratio))

        f = h5py.File("../data/SV_for_HL2.h5", "r")
        self.SV_EAFM = torch.from_numpy(np.asarray(f["SV_EAFM"], dtype=np.complex64))
        norm_EAF = torch.linalg.norm(self.SV_EAFM, axis=3)
        self.SV_EAFM /= norm_EAF[..., None]
        _, _, _, n_mic = self.SV_EAFM.shape

        if dataset == "wsj0_chime3":
            root = "/n/work3/sekiguch/dataset/Hololens2_SimData_WSJ0_CHiME3/"
            # fname = "../data/train_wsj0_chime3_2spk.json"
        elif dataset == "librispeech":
            root = "/n/work3/sekiguch/dataset/Hololens2_SimData_Librispeech/"
            # fname = "../data/train_librispeech_2spk.json"
        # flist = json.load(open(fname, "r"))
        # id_list = np.random.permutation(list(flist.keys()))

        self.tuning_mix = []
        self.tuning_sep = []
        self.tuning_angle = []

        if count * tuning_data_interval_minute < tuning_data_minute:
            wav_start_idx = 0
            start_idx_in_first_wav = 0
            n_wav = int(np.ceil((count * tuning_data_interval_minute * 60) / second_per_wav))
            second_remainder = int(n_wav * second_per_wav - count * tuning_data_interval_minute * 60)
            end_idx_in_last_wav = int(((second_per_wav - second_remainder) * n_sample_per_wav) / second_per_wav)
        else:
            start_sec = int((count * tuning_data_interval_minute - tuning_data_minute) * 60)
            wav_start_idx = start_sec // second_per_wav
            first_unused_sec = start_sec - wav_start_idx * second_per_wav
            start_idx_in_first_wav = int(first_unused_sec * n_sample_per_wav / second_per_wav)

            if first_unused_sec > 0:
                n_wav_use_full = int((tuning_data_minute * 60 - (second_per_wav - first_unused_sec)) / second_per_wav)
                second_remainder_last = tuning_data_minute * 60 - (
                    n_wav_use_full * second_per_wav + (second_per_wav - first_unused_sec)
                )
            else:
                n_wav_use_full = int((tuning_data_minute * 60) / second_per_wav)
                second_remainder_last = tuning_data_minute * 60 - (n_wav_use_full * second_per_wav)

            if (second_remainder_last > 0) and (first_unused_sec > 0):
                n_wav = n_wav_use_full + 2
            elif (second_remainder_last > 0) and (first_unused_sec == 0):
                n_wav = n_wav_use_full + 1
            elif (second_remainder_last == 0) and (first_unused_sec > 0):
                n_wav = n_wav_use_full + 1
            elif (second_remainder_last == 0) and (first_unused_sec == 0):
                n_wav = n_wav_use_full

            if second_remainder_last > 0:
                end_idx_in_last_wav = int(second_remainder_last * n_sample_per_wav / second_per_wav)
            else:
                end_idx_in_last_wav = n_sample_per_wav

        print(f"\n In dataset, from {wav_start_idx} to {wav_start_idx + n_wav} is used in the {count} iteration")
        sep_fname_list = sorted(glob(tuning_data_dir + "/sep*.wav"))
        for idx, sep_fname in enumerate(sep_fname_list[wav_start_idx : wav_start_idx + n_wav]):
            sep, fs_sep = torchaudio.load(sep_fname)
            mix_fname = sep_fname.replace("sep", "mix").split("_azim")[0] + ".wav"
            mix, fs_mix = torchaudio.load(mix_fname)
            angle = int(sep_fname.split("azim=")[1].split("_")[0])
            angle = int(np.round(angle / 5))

            assert fs_sep == 16000, f"Sampling rate of sep should be 16kHz, but got {fs_sep}"
            assert fs_mix == 16000, f"Sampling rate of mix should be 16kHz, but got {fs_mix}"
            assert len(mix) == n_mic, "The number of microphone is different from that of train data"
            assert sep.shape[1] == n_sample_per_wav, "-----------------Error------------\n\n"

            if idx == 0:
                sep = sep[:, start_idx_in_first_wav:]
                mix = mix[:, start_idx_in_first_wav:]
            if idx == len(sep_fname_list) - 1:
                sep = sep[:, :end_idx_in_last_wav]
                mix = mix[:, :end_idx_in_last_wav]

            n_window = (sep.shape[1] - window_len) // shift + 1
            if n_window == 0:
                continue

            self.tuning_mix.extend(
                torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256, power=None)(
                    torch.as_strided(mix, size=(n_mic, n_window, window_len), stride=(n_sample_per_wav, shift, 1))
                ).permute(1, 0, 3, 2)
            )
            self.tuning_sep.extend(torch.as_strided(sep, size=(n_window, window_len), stride=(shift, 1)))
            self.tuning_angle.extend([[1, angle] for i in range(n_window)])

        self.tuning_sep = torch.stack(self.tuning_sep)
        self.tuning_mix = torch.stack(self.tuning_mix)
        self.tuning_angle = torch.as_tensor(self.tuning_angle)

        print("\n\n tuning_sep.shape = ", self.tuning_sep.shape)

        if step == "train":
            n_window_train = int((1 - tuning_data_ratio) / tuning_data_ratio * len(self.tuning_sep))
            n_window_total = 0
            for i in range(24):
                h5_fname = root + f"/h5/2spk/train/{dataset}_train_{window_len}_0.25-{i+1}.h5"
                with h5py.File(h5_fname, "r") as f:
                    n_window_per_block = f["image"].shape[0]
                    n_window_tmp = min(n_window_train - n_window_total, n_window_per_block)
                    self.tuning_sep = torch.cat(
                        [self.tuning_sep, torch.as_tensor(np.array(f["image"][:n_window_tmp]))]
                    )
                    self.tuning_mix = torch.cat(
                        [self.tuning_mix, torch.as_tensor(np.array(f["mixture"][:n_window_tmp]))]
                    )
                    self.tuning_angle = torch.cat(
                        [self.tuning_angle, torch.as_tensor(np.array(f["angle"][:n_window_tmp]))]
                    )
                    n_window_total += n_window_tmp
                    if n_window_total >= n_window_train:
                        break

    def set_param(self, use_BF=None, use_SV=None, use_DAN=None, use_VAD=None):
        if use_BF is not None:
            self.use_BF = use_BF
        if use_SV is not None:
            self.use_SV = use_SV
        if use_DAN is not None:
            self.use_DAN = use_DAN
        if use_VAD is not None:
            self.use_VAD = use_VAD

    def __len__(self):
        return len(self.tuning_sep)

    def __getitem__(self, idx):
        mixture_MTF = self.tuning_mix[idx]
        elev_idx, azim_idx = self.tuning_angle[idx]

        feature = utility.make_feature(
            mixture_MTF,
            self.SV_EAFM[elev_idx, azim_idx],
            azim_idx,
            use_BF=self.use_BF,
            use_SV=self.use_SV,
            use_DAN=self.use_DAN,
        )

        if self.use_DAN:
            feature, doa = feature
            return feature, self.tuning_sep[idx], mixture_MTF.to(torch.cdouble), doa
        else:
            return feature, self.tuning_sep[idx], mixture_MTF.to(torch.cdouble)


class FinetuneDataModule(pl.LightningDataModule):
    def __init__(
        self,
        use_BF,
        use_SV,
        use_DAN,
        use_VAD,
        batch_size,
        tuning_data_dir,
        tuning_data_minute=12,
        tuning_data_interval_minute=3,
        tuning_data_ratio=0.5,
        overlap_ratio=0.5,
        window_len=48128,
        num_workers=2,
        **kwargs,
    ):
        super().__init__()
        self.count = 0
        self.use_BF = use_BF
        self.use_SV = use_SV
        self.use_DAN = use_DAN
        self.use_VAD = use_VAD
        self.save_hyperparameters()  # ignore=["overwrite", "hoge"]
        self.batch_size = batch_size
        self.tuning_data_dir = tuning_data_dir
        self.tuning_data_minute = tuning_data_minute
        self.tuning_data_interval_minute = tuning_data_interval_minute
        self.tuning_data_ratio = tuning_data_ratio
        self.overlap_ratio = overlap_ratio
        self.window_len = window_len
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def train_dataloader(self):
        pl.seed_everything(0)
        self.count += 1
        print(f"\n LightningDataModule is called  count = {self.count}\n")
        self.train_dataset = FinetuneDataset(
            count=self.count,
            use_BF=self.use_BF,
            use_SV=self.use_SV,
            use_DAN=self.use_DAN,
            use_VAD=self.use_VAD,
            window_len=self.window_len,
            overlap_ratio=self.overlap_ratio,
            tuning_data_dir=self.tuning_data_dir,
            tuning_data_minute=self.tuning_data_minute,
            tuning_data_interval_minute=self.tuning_data_interval_minute,
            tuning_data_ratio=self.tuning_data_ratio,
            step="train",
        )
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )


def main(args):
    n_iteration = 48 // args.tuning_data_interval_minute
    max_epochs = n_iteration * args.reload_epoch_interval

    ckpt_dir = args.ckpt.rsplit("/", 1)[0]
    version = f"incremental-n_epoch={args.reload_epoch_interval}-intv_minute={args.tuning_data_interval_minute}-minute={args.tuning_data_minute}-ratio={args.tuning_data_ratio}-finetune={args.finetune}-batch={args.batch_size}-lr={args.lr}"
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=ckpt_dir, name="", version=version)
    default_root_dir = ckpt_dir + f"/{version}/"

    previous_ckpt_last = glob(default_root_dir + f"/checkpoints/epoch={max_epochs-1}*.ckpt")
    if len(previous_ckpt_last) >= 1:
        print(previous_ckpt_last, " already exist. Skip...")
        return 0

    pl.seed_everything(0)

    model = Lightning_CSS(
        save_hparam=False,
        **vars(args),
    )

    ckpt_dict = torch.load(args.ckpt)
    model.load_state_dict(ckpt_dict["state_dict"])

    dm = FinetuneDataModule(
        tuning_data_dir=args.tuning_data_dir,
        tuning_data_minute=args.tuning_data_minute,
        tuning_data_interval_minute=args.tuning_data_interval_minute,
        tuning_data_ratio=args.tuning_data_ratio,
        use_BF=args.use_BF,
        use_SV=args.use_SV,
        use_DAN=args.use_DAN,
        use_VAD=args.use_VAD,
        batch_size=args.batch_size,
        num_workers=3,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="train_loss",
        save_top_k=n_iteration,
        every_n_epochs=args.reload_epoch_interval,
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        gpus=1,
        max_epochs=max_epochs,
        # strategy=pl.plugins.DDPPlugin(find_unused_parameters=False),
        reload_dataloaders_every_n_epochs=args.reload_epoch_interval,
        callbacks=[checkpoint_callback],
        logger=tb_logger,
        default_root_dir=default_root_dir,
        gradient_clip_val=1.0,
        deterministic=True,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("ckpt", type=str, default=None)
    parser.add_argument("--finetune", type=str, default=True, help="True or dan")
    parser.add_argument(
        "--tuning_data_minute", type=int, default=12, help="How many minutes of data is used at maximum"
    )
    parser.add_argument(
        "--tuning_data_interval_minute", type=int, default=3, help="The interval [min] for updating parameters"
    )
    parser.add_argument(
        "--tuning_data_ratio", type=float, default=0.5, help="Ratio between tuning and original train data"
    )
    parser.add_argument(
        "--reload_epoch_interval", type=int, default=5, help="The number of epochs to reload datamodule"
    )
    parser.add_argument(
        "--tuning_data_dir",
        type=str,
        default="/n/work3/sekiguch/data_for_paper/IROS2022/FastMNMF-M=5-Ns=5-Nn=0-K=8-it=100-itIVA=50-bit=64_reset/",
    )
    parser.add_argument("--model_name", type=str, default="lstm")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--memo", type=str, default="")
    parser.add_argument("--use_BF", action="store_true", default=False)
    parser.add_argument("--use_SV", action="store_true", default=False)
    parser.add_argument("--use_DAN", action="store_true", default=False)
    parser.add_argument("--use_VAD", action="store_true", default=False)
    args_tmp, _ = parser.parse_known_args()
    parser = Lightning_CSS.add_model_specific_args(parser, model_name=args_tmp.model_name)
    parser = pl.Trainer.add_argparse_args(parser)
    args, _ = parser.parse_known_args()

    torch.cuda.empty_cache()

    # make_flist(args.tuning_data_dir)

    main(args)
