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


class FinetuneDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tuning_data_dir,
        use_BF,
        use_SV,
        use_DAN,
        use_VAD,
        window_len=48128,
        overlap_ratio=0.5,
        tuning_data_minute=3,
        tuning_data_ratio=0.5,
        dataset=["wsj0_chime3", "librispeech"][0],
        step=["train", "valid"][0],
        BF_input=["DSBF", "MPDR"][0],
    ):
        second_per_wav = 9
        n_sample_per_wav = window_len * 3

        self.use_BF = use_BF
        self.use_SV = use_SV
        self.use_DAN = use_DAN
        self.use_VAD = use_VAD
        self.BF_input = BF_input

        self.window_len = window_len
        self.overlap_ratio = overlap_ratio
        self.shift = int(self.window_len * (1 - overlap_ratio))
        self.dataset = dataset
        self.tuning_data_minute = tuning_data_minute
        self.tuning_data_ratio = tuning_data_ratio
        self.tuning_data_dir = tuning_data_dir

        f = h5py.File("../data/SV_for_HL2.h5", "r")
        self.SV_EAFM = torch.from_numpy(np.asarray(f["SV_EAFM"], dtype=np.complex64))
        norm_EAF = torch.linalg.norm(self.SV_EAFM, axis=3)
        self.SV_EAFM /= norm_EAF[..., None]
        _, _, self.n_freq, self.n_mic = self.SV_EAFM.shape

        if self.dataset == "wsj0_chime3":
            self.root = "/n/work3/sekiguch/dataset/Hololens2_SimData_WSJ0_CHiME3/"
            fname = f"../data/train_wsj0_chime3_2spk.json"

        elif self.dataset == "librispeech":
            self.root = "/n/work3/sekiguch/dataset/Hololens2_SimData_Librispeech/"
            fname = f"../data/train_librispeech_2spk.json"

        self.flist = json.load(open(fname, "r"))
        self.id_list = np.random.permutation(list(self.flist.keys()))

        self.tuning_mix = []
        self.tuning_sep = []
        self.tuning_angle = []

        n_wav = int(np.ceil(self.tuning_data_minute * 60 / second_per_wav))

        sep_fname_list = sorted(glob(self.tuning_data_dir + "/sep*.wav"))[:n_wav]
        for idx, sep_fname in enumerate(sep_fname_list):
            sep, fs_sep = torchaudio.load(sep_fname)
            mix_fname = sep_fname.replace("sep", "mix").split("_azim")[0] + ".wav"
            mix, fs_mix = torchaudio.load(mix_fname)
            angle = int(sep_fname.split("azim=")[1].split("_")[0])
            angle = int(np.round(angle / 5))

            assert fs_sep == 16000, f"Sampling rate of sep should be 16kHz, but got {fs_sep}"
            assert fs_mix == 16000, f"Sampling rate of mix should be 16kHz, but got {fs_mix}"
            assert len(mix) == self.n_mic, "The number of microphone is different from that of train data"

            if idx == len(sep_fname_list) - 1:  # If last iteration
                second_remainder = self.tuning_data_minute * 60 - idx * second_per_wav
                n_sample_last = int(second_remainder * n_sample_per_wav / second_per_wav)
                sep = sep[:, :n_sample_last]
                mix = mix[:, :n_sample_last]

            n_window = (sep.shape[1] - self.window_len) // self.shift + 1
            if n_window == 0:
                continue

            self.tuning_mix.extend(
                torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256, power=None)(
                    torch.as_strided(
                        mix, size=(self.n_mic, n_window, window_len), stride=(n_sample_per_wav, self.shift, 1)
                    )
                ).permute(1, 0, 3, 2)
            )
            self.tuning_sep.extend(torch.as_strided(sep, size=(n_window, window_len), stride=(self.shift, 1)))
            self.tuning_angle.extend([[1, angle] for i in range(n_window)])

        self.tuning_sep = torch.stack(self.tuning_sep)
        self.tuning_mix = torch.stack(self.tuning_mix)
        self.tuning_angle = torch.as_tensor(self.tuning_angle)

        if step == "train":
            n_window_train = int((1 - self.tuning_data_ratio) / self.tuning_data_ratio * len(self.tuning_sep))
            n_window_total = 0
            for i in range(24):
                h5_fname = self.root + f"/h5/2spk/train/{self.dataset}_train_{self.window_len}_0.25-{i+1}.h5"
                with h5py.File(h5_fname, "r") as f:
                    n_data_per_block = f["image"].shape[0]
                    n_window_tmp = min(n_window_train - n_window_total, n_data_per_block)
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
            BF=self.BF_input,
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
        tuning_data_minute=3,
        tuning_data_ratio=0.5,
        overlap_ratio=0.5,
        window_len=48128,
        num_workers=2,
        BF_input=["DSBF", "MPDR"][0],
        **kwargs,
    ):
        super().__init__()

        self.use_BF = use_BF
        self.use_SV = use_SV
        self.use_DAN = use_DAN
        self.use_VAD = use_VAD
        self.BF_input = BF_input
        self.save_hyperparameters()  # ignore=["overwrite", "hoge"]
        self.batch_size = batch_size
        self.tuning_data_dir = tuning_data_dir
        self.tuning_data_minute = tuning_data_minute
        self.tuning_data_ratio = tuning_data_ratio
        self.overlap_ratio = overlap_ratio
        self.window_len = window_len
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = FinetuneDataset(
            use_BF=self.use_BF,
            use_SV=self.use_SV,
            use_DAN=self.use_DAN,
            use_VAD=self.use_VAD,
            window_len=self.window_len,
            overlap_ratio=self.overlap_ratio,
            tuning_data_dir=self.tuning_data_dir,
            tuning_data_minute=self.tuning_data_minute,
            tuning_data_ratio=self.tuning_data_ratio,
            step="train",
            BF_input=self.BF_input,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )


class CheckpointCallback_SaveAtSpecificEpoch(pl.Callback):
    def __init__(self, save_dir, save_epoch_list=[5, *range(10, 101, 10)]):
        """
        Args:
            save_epoch_list: when to save in epochs
        """
        self.save_epoch_list = save_epoch_list
        self.save_dir = save_dir

    def on_train_epoch_end(self, trainer: pl.Trainer, _):
        """Check if we should save a checkpoint after train epoch"""
        epoch = trainer.current_epoch
        if epoch + 1 in self.save_epoch_list:
            filename = f"epoch={epoch}-step={trainer.global_step}.ckpt"
            ckpt_path = os.path.join(self.save_dir, filename)
            trainer.save_checkpoint(ckpt_path, weights_only=True)


def main(args):

    if (args.BF_input == "DSBF") or (args.use_BF is False):
        version = (
            f"wpe_minute={args.tuning_data_minute}-ratio={args.tuning_data_ratio}-finetune={args.finetune}"
            + f"-batch={args.batch_size}-lr={args.lr}"
        )
    else:
        version = (
            f"wpe_minute={args.tuning_data_minute}-ratio={args.tuning_data_ratio}-finetune={args.finetune}"
            + f"-batch={args.batch_size}-lr={args.lr}-BF={args.BF_input}"
        )
    ckpt_dir = args.ckpt.rsplit("/", 1)[0]
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=ckpt_dir, name="", version=version)
    default_root_dir = ckpt_dir + f"/{version}/checkpoints/"

    previous_ckpt_list = glob(default_root_dir + f"/epoch={args.max_epochs-1}-*.ckpt")
    if len(previous_ckpt_list) >= 1:
        print(previous_ckpt_list, " already exist. Skip...")
        return 0

    pl.seed_everything(0)

    start_load_model = time.time()

    model = Lightning_CSS(
        save_hparam=False,
        **vars(args),
    )
    ckpt_dict = torch.load(args.ckpt)
    model.load_state_dict(ckpt_dict["state_dict"])

    start_load_data = time.time()

    dm = FinetuneDataModule(
        tuning_data_dir=args.tuning_data_dir,
        tuning_data_minute=args.tuning_data_minute,
        tuning_data_ratio=args.tuning_data_ratio,
        use_BF=args.use_BF,
        use_SV=args.use_SV,
        use_DAN=args.use_DAN,
        use_VAD=args.use_VAD,
        batch_size=args.batch_size,
        num_workers=2,
        BF_input=args.BF_input,
    )

    save_epoch_list = [1, 3, 5, *range(10, args.max_epochs + 1, 10)]
    checkpoint_callback = CheckpointCallback_SaveAtSpecificEpoch(
        save_epoch_list=save_epoch_list, save_dir=default_root_dir
    )

    start_train_data = time.time()

    trainer = pl.Trainer.from_argparse_args(
        args,
        gpus=1,
        max_epochs=args.max_epochs,
        # strategy=pl.plugins.DDPPlugin(find_unused_parameters=False),
        callbacks=[checkpoint_callback],
        logger=tb_logger,
        gradient_clip_val=1.0,
        checkpoint_callback=False,
        deterministic=True,
    )
    trainer.fit(model, dm)
    end_train_data = time.time()

    with open(f"{default_root_dir}/elapsed_time.txt", "a") as f:
        f.write(f"max_epochs = {args.max_epochs}\n")
        f.write(f"start_load_model = {start_load_model}\n")
        f.write(f"start_load_data = {start_load_data}\n")
        f.write(f"start_train_data = {start_train_data}\n")
        f.write(f"end_train_data = {end_train_data}\n")
        f.write(f"elapsed_time_per_epoch = {(end_train_data-start_train_data) / args.max_epochs}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("ckpt", type=str, default=None)
    parser.add_argument("--finetune", type=str, default=True, help="True or dan")
    parser.add_argument(
        "--tuning_data_minute", type=float, default=3, help="How many minutes of data is used for fine-tune"
    )
    parser.add_argument(
        "--tuning_data_ratio", type=float, default=0.5, help="Ratio between tuning and original train data"
    )
    parser.add_argument(
        "--tuning_data_dir",
        type=str,
        default="/n/work3/sekiguch/data_for_paper/IROS2022/FastMNMF-M=5-Ns=5-Nn=0-K=8-it=100-itIVA=50-bit=64_reset_wpe/",
    )
    # parser.add_argument("--max_epochs", type=int, default=100, help="The number of epochs")
    parser.add_argument("--model_name", type=str, default="lstm")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--memo", type=str, default="")
    parser.add_argument("--BF_input", type=str, default="DSBF")
    parser.add_argument("--use_BF", action="store_true", default=False)
    parser.add_argument("--use_SV", action="store_true", default=False)
    parser.add_argument("--use_DAN", action="store_true", default=False)
    parser.add_argument("--use_VAD", action="store_true", default=False)
    args_tmp, _ = parser.parse_known_args()
    parser = Lightning_CSS.add_model_specific_args(parser, model_name=args_tmp.model_name)
    parser = pl.Trainer.add_argparse_args(parser)
    args, _ = parser.parse_known_args()

    torch.cuda.empty_cache()

    main(args)
