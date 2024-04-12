#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import torch
from Lightning_CSS_VAD_DAN import Lightning_CSS_VAD_DAN
import pytorch_lightning as pl
import json
import h5py
import os

import utils.utility as utility


class MultichannelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        window_len,
        dataset=["wsj0_chime3", "librispeech"][0],
        step=["train", "valid"][0],
        angle_std=0,
        dan_std=0,
        n_spk=2,
        overlap_ratio=0.25,
    ):
        self.step = step
        self.window_len = window_len
        self.overlap_ratio = overlap_ratio
        self.shift = int(self.window_len * (1 - overlap_ratio))
        self.dataset = dataset
        self.n_spk = n_spk
        self.angle_std = angle_std
        self.dan_std = dan_std

        f = h5py.File("../data/SV_for_HL2.h5", "r")
        self.SV_EAFM = torch.from_numpy(np.asarray(f["SV_EAFM"], dtype=np.complex64))
        norm_EAF = torch.linalg.norm(self.SV_EAFM, axis=3)
        self.SV_EAFM /= norm_EAF[..., None]

        if self.dataset == "wsj0_chime3":
            self.root = "/n/work3/sekiguch/dataset/Hololens2_SimData_WSJ0_CHiME3/"
            fname = f"../data/{self.step}_wsj0_chime3_{self.n_spk}spk.json"

        elif self.dataset == "librispeech":
            self.root = "/n/work3/sekiguch/dataset/Hololens2_SimData_Librispeech/"
            fname = f"../data/{self.step}_librispeech_{self.n_spk}spk.json"

        self.flist = json.load(open(fname, "r"))
        self.id_list = np.random.permutation(list(self.flist.keys()))

        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        if 24 % world_size != 0:
            raise ValueError("WORLD_SIZE should be a divisor of 24 (number of data block)")
        n_block = 24 // world_size

        self.image_list = []
        self.mixture_list = []
        self.spk_id_list = []
        self.angle_list = []

        for i in range(n_block):
            block_idx = local_rank * n_block + i + 1
            h5_fname = (
                self.root
                + f"/h5/{self.n_spk}spk/{self.step}/"
                + f"{self.dataset}_{self.step}_{self.window_len}_{self.overlap_ratio:.2}-{block_idx}.h5"
            )
            print("Start loading ", h5_fname)
            with h5py.File(h5_fname, "r") as f:
                self.image_list.append(torch.as_tensor(np.array(f["image"])))
                self.mixture_list.append(torch.as_tensor(np.array(f["mixture"])))
                # self.spk_id_list.append(f["spk_id"])
                self.angle_list.append(torch.as_tensor(np.array(f["angle"])))

        self.mixture_list = torch.cat(self.mixture_list, dim=0)
        self.image_list = torch.cat(self.image_list, dim=0)
        self.angle_list = torch.cat(self.angle_list, dim=0)

    def __len__(self):
        return len(self.mixture_list)

    def __getitem__(self, idx):
        mixture_MTF = self.mixture_list[idx]
        elev_idx, azim_idx = self.angle_list[idx]
        azim_idx_SV = (azim_idx + np.random.choice(np.arange(-1 * self.angle_std, self.angle_std + 1))) % 72
        azim_idx_dan = azim_idx + np.random.normal(scale=self.dan_std)
        feature, doa = utility.make_feature(mixture_MTF, self.SV_EAFM[elev_idx, azim_idx_SV], azim_idx_dan)

        return feature, self.image_list[idx], mixture_MTF.to(torch.cdouble), doa


class LightningDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers=2, angle_std=0, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # ignore=["overwrite", "hoge"]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.angle_std = angle_std

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = MultichannelDataset(window_len=48128, step="train", angle_std=self.angle_std)
        self.valid_dataset = MultichannelDataset(window_len=48128, step="valid", angle_std=self.angle_std)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    """
    TODO:
        * maskの範囲を制限（過去、未来のデータから現在を予測)
    """
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--version", default=0, help="test")
    parser.add_argument("--resume_from_last", action="store_true")
    parser.add_argument("--resume_from_ckpt", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="lstm")
    parser.add_argument("--batch_sizse", type=int, default=12)
    parser.add_argument("--angle_std", type=int, default=2)
    parser.add_argument("--dan_std", type=float, default=0)
    parser.add_argument("--memo", type=str, default="")
    args_tmp, _ = parser.parse_known_args()
    parser = Lightning_CSS_VAD_DAN.add_model_specific_args(parser, model_name=args_tmp.model_name)
    # --max_epochs, --terminate_on_nan, --auto_lr_find
    parser = pl.Trainer.add_argparse_args(parser)
    args, _ = parser.parse_known_args()

    torch.cuda.empty_cache()

    save_root_dir = "/n/work3/sekiguch/data_for_paper/IROS2022/CSS_VAD_DAN/"
    if args.resume_from_ckpt is not None:
        ckpt_path = args.resume_from_ckpt
        ckpt_dir = args.resume_from_ckpt.rsplit("/", 1)[0]
        version = ckpt_dir.rsplit("/", 1)[0].split(f"_")[-1]
        default_root_dir = None
        tb_logger = pl.loggers.TensorBoardLogger(
            ckpt_dir.rsplit("/", 3)[0], name=f"lightning_logs/{args.model_name}", version=version
        )
    elif args.resume_from_last:
        from glob import glob

        version = max(
            [
                int(fname.split("_")[-1])
                for fname in glob(save_root_dir + f"/lightning_logs/{args.model_name}/version*")
            ]
        )
        ckpt_path = save_root_dir + f"/lightning_logs/{args.model_name}/version_{version}/checkpoints/last.ckpt"
        ckpt_dir = ckpt_path.rsplit("/", 1)[0]
        default_root_dir = None
        tb_logger = pl.loggers.TensorBoardLogger(
            save_root_dir, name=f"lightning_logs/{args.model_name}", version=version
        )
    else:
        ckpt_path = None
        ckpt_dir = None
        default_root_dir = save_root_dir
        tb_logger = pl.loggers.TensorBoardLogger(save_root_dir, name=f"lightning_logs/{args.model_name}")

    pl.seed_everything(0)
    dm = LightningDataModule(batch_size=16, num_workers=3, angle_std=args.angle_std, dan_std=args.dan_std)
    model = Lightning_CSS_VAD_DAN(save_hparam=True, **vars(args))
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="valid_loss", save_last=True, dirpath=ckpt_dir, every_n_epochs=10
    )
    trainer = pl.Trainer.from_argparse_args(
        args,
        gpus=args.gpus,
        strategy=pl.plugins.DDPPlugin(find_unused_parameters=False),
        callbacks=[checkpoint_callback],
        logger=tb_logger,
        default_root_dir=default_root_dir,
        gradient_clip_val=1.0,
    )
    trainer.fit(model, dm, ckpt_path=ckpt_path)
