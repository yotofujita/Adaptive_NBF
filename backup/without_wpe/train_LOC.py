#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import torch
import json
import h5py
import os
import pytorch_lightning as pl

import utility
from Lightning_LOC import Lightning_LOC


class MultichannelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        use_BF=False,
        use_SV=False,
        use_DAN=False,
        window_len=48128,
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

        self.use_BF = use_BF
        self.use_SV = use_SV
        self.use_DAN = use_DAN

        f = h5py.File("../data/SV_for_HL2.h5", "r")
        self.SV_EAFM = torch.from_numpy(np.asarray(f["SV_EAFM"], dtype=np.complex64))
        norm_EAF = torch.linalg.norm(self.SV_EAFM, axis=3)
        self.SV_EAFM /= norm_EAF[..., None]

        if self.dataset == "wsj0_chime3":
            self.root = "/n/work3/sekiguch/dataset/Hololens2_SimData_WSJ0_CHiME3/"
            fname = f"../data/{self.step}_wsj0_chime3_{self.n_spk}spk.json"

        elif self.dataset == "librispeech":
            raise ValueError
            # self.root = "/n/work3/sekiguch/dataset/Hololens2_SimData_Librispeech/"
            # fname = f"../data/{self.step}_librispeech_{self.n_spk}spk.json"

        self.flist = json.load(open(fname, "r"))
        self.id_list = np.random.permutation(list(self.flist.keys()))

        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        if 24 % world_size != 0:
            raise ValueError("WORLD_SIZE should be a divisor of 24 (number of data block)")
        # n_block = 24 // world_size
        n_block = 1

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
                self.angle_list.append(torch.as_tensor(np.array(f["angle"])))

        self.mixture_list = torch.cat(self.mixture_list, dim=0)
        self.image_list = torch.cat(self.image_list, dim=0)
        self.angle_list = torch.cat(self.angle_list, dim=0)

    def set_param(self, use_BF=None, use_SV=None, use_DAN=None, angle_std=None, dan_std=None):
        if use_BF is not None:
            self.use_BF = use_BF
        if use_SV is not None:
            self.use_SV = use_SV
        if use_DAN is not None:
            self.use_DAN = use_DAN
        if angle_std is not None:
            self.angle_std = angle_std
        if dan_std is not None:
            self.dan_std = dan_std

    def __len__(self):
        return len(self.mixture_list)

    def __getitem__(self, idx):
        mixture_MTF = self.mixture_list[idx]
        elev_idx, azim_idx = self.angle_list[idx]
        azim_idx_SV = (azim_idx + np.random.choice(np.arange(-1 * self.angle_std, self.angle_std + 1))) % 72
        azim_idx_dan = azim_idx + np.random.normal(scale=self.dan_std)

        feature = utility.make_feature(
            mixture_MTF,
            self.SV_EAFM[elev_idx, azim_idx_SV],
            azim_idx_dan,
            use_BF=self.use_BF,
            use_SV=self.use_SV,
            use_DAN=self.use_DAN,
        )

        if self.use_DAN:
            feature, doa = feature
            return (
                feature,
                self.image_list[idx],
                mixture_MTF,
                self.SV_EAFM[elev_idx, azim_idx].to(torch.complex64),
                doa,
            )
        else:
            return (
                feature,
                self.image_list[idx],
                mixture_MTF,
                self.SV_EAFM[elev_idx, azim_idx].to(torch.complex64),
            )


class LightningDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers=2, angle_std=0, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # ignore=["overwrite", "hoge"]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.angle_std = angle_std

    def set_param(self, use_BF=None, use_SV=None, use_DAN=None, angle_std=None, dan_std=None):
        if use_BF is not None:
            self.use_BF = use_BF
        if use_SV is not None:
            self.use_SV = use_SV
        if use_DAN is not None:
            self.use_DAN = use_DAN
        if angle_std is not None:
            self.angle_std = angle_std
        if dan_std is not None:
            self.dan_std = dan_std

        try:
            self.train_dataset.set_param(
                use_BF=self.use_BF,
                use_SV=self.use_SV,
                use_DAN=self.use_DAN,
                angle_std=self.angle_std,
                dan_std=self.dan_std,
            )
            self.valid_dataset.set_param(
                use_BF=self.use_BF,
                use_SV=self.use_SV,
                use_DAN=self.use_DAN,
                angle_std=self.angle_std,
                dan_std=self.dan_std,
            )
        except:
            pass

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = MultichannelDataset(
            use_BF=self.use_BF,
            use_SV=self.use_SV,
            use_DAN=self.use_DAN,
            window_len=48128,
            step="valid",
            angle_std=self.angle_std,
        )
        self.valid_dataset = MultichannelDataset(
            use_BF=self.use_BF,
            use_SV=self.use_SV,
            use_DAN=self.use_DAN,
            window_len=48128,
            step="valid",
            angle_std=self.angle_std,
        )

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
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--setting", type=int, default=0)
    parser.add_argument("--version", default=0, help="test")
    parser.add_argument("--resume_from_last", action="store_true")
    parser.add_argument("--resume_from_ckpt", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="lstm")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--memo", type=str, default="")
    args_tmp, _ = parser.parse_known_args()
    parser = Lightning_LOC.add_model_specific_args(parser, model_name=args_tmp.model_name)
    parser = pl.Trainer.add_argparse_args(parser)
    args, _ = parser.parse_known_args()

    torch.cuda.empty_cache()
    save_root_dir = "/n/work3/sekiguch/data_for_paper/IROS2022/"

    # if args.resume_from_ckpt is not None:
    #     ckpt_path = args.resume_from_ckpt
    #     ckpt_dir = args.resume_from_ckpt.rsplit("/", 1)[0]
    #     version = ckpt_dir.rsplit("/", 1)[0].split(f"_")[-1]
    #     default_root_dir = None
    #     tb_logger = pl.loggers.TensorBoardLogger(
    #         ckpt_dir.rsplit("/", 3)[0], name=f"lightning_logs/{args.model_name}", version=version
    #     )
    # elif args.resume_from_last:
    #     from glob import glob

    #     version = max(
    #         [
    #             int(fname.split("_")[-1])
    #             for fname in glob(save_root_dir + f"/lightning_logs/{args.model_name}/version*")
    #         ]
    #     )
    #     ckpt_path = save_root_dir + f"/lightning_logs/{args.model_name}/version_{version}/checkpoints/last.ckpt"
    #     ckpt_dir = ckpt_path.rsplit("/", 1)[0]
    #     default_root_dir = None
    #     tb_logger = pl.loggers.TensorBoardLogger(
    #         save_root_dir, name=f"lightning_logs/{args.model_name}", version=version
    #     )
    # else:
    #     ckpt_path = None
    #     ckpt_dir = None
    #     default_root_dir = save_root_dir
    #     tb_logger = pl.loggers.TensorBoardLogger(save_root_dir, name=f"lightning_logs/{args.model_name}")

    settings = [
        [True, False, True],
        [True, True, False],
        [True, False, False],
        [True, True, True],
        [False, True, False],
        [False, False, True],
        [False, True, True],
    ]  # use_BF, use_SV, use_DAN

    pl.seed_everything(0)

    use_BF, use_SV, use_DAN = settings[args.setting]
    if use_DAN:
        angle_std = 2
    else:
        angle_std = 0
    dan_std = 0

    dm = LightningDataModule(
        batch_size=args.batch_size,
        num_workers=3,
        angle_std=angle_std,
        dan_std=dan_std,
    )
    dm.set_param(use_BF=use_BF, use_SV=use_SV, use_DAN=use_DAN)

    model = Lightning_LOC(
        use_BF=use_BF,
        use_SV=use_SV,
        use_DAN=use_DAN,
        save_hparam=True,
        fine_tune=False,
        **vars(args),
    )

    default_root_dir = f"{save_root_dir}/{str(model)}/"
    # if os.path.isdir(default_root_dir):
    #     print(f"{default_root_dir} already exist. Skip this configuration.")
    #     exit()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="valid_sep_loss", save_last=True, every_n_epochs=30)
    trainer = pl.Trainer.from_argparse_args(
        args,
        gpus=args.gpus,
        strategy=pl.plugins.DDPPlugin(find_unused_parameters=False),
        callbacks=[checkpoint_callback],
        default_root_dir=default_root_dir,
        gradient_clip_val=0.1,
        max_epochs=500,
    )
    trainer.fit(model, dm)