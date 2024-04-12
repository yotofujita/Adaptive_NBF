#!/usr/bin/env python3
# coding:utf-8
import numpy as np
import torch
import speechbrain as sb
import soundfile as sf
import sys
import pdb
import librosa
from numpy.lib.stride_tricks import as_strided

from utils.utility import *
import resampy

class SEBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        if stage != sb.Stage.TEST:
            noisy_sig = batch.noisy_sig.data.to(self.device)
            enhanced_sig = batch.enhanced_sig.data.to(self.device)
            clean_sig = batch.clean_sig.data.to(self.device)
            speaker_label = batch.speaker_label.data.to(self.device)

            if self.model_name == "SB_ver1":
                est, clean_est, spk_emb, spk_emb_mean, clean_spk_emb, clean_spk_emb_mean, reliability = self.modules.model.forward(noisy_sig, enhanced_sig, clean_sig, speaker_label)
                return {"est": est, "spk_emb": spk_emb, "spk_emb_mean": spk_emb_mean, "clean_est": clean_est, "clean_spk_emb": clean_spk_emb, "clean_spk_emb_mean": clean_spk_emb_mean}
            elif self.model_name == "SB_orig":
                est, clean_est, spk_emb, clean_spk_emb = self.modules.model.forward(noisy_sig, enhanced_sig, clean_sig, speaker_label)
                return {"est": est, "spk_emb": spk_emb, "clean_est": clean_est, "clean_spk_emb": clean_spk_emb, "spk_emb_mean": None, "clean_spk_emb_mean": None}


    def calculate_speaker_vec_loss(self, spk_emb, spk_emb_mean, clean_spk_emb, clean_spk_emb_mean, loss_type="type1"):
        if loss_type == "type1":
            loss = sb.nnet.losses.mse_loss(spk_emb, clean_spk_emb)
            return loss
        elif loss_type == "type2":
            loss1 = sb.nnet.losses.mse_loss(spk_emb, clean_spk_emb)
            loss2 = sb.nnet.losses.mse_loss(spk_emb_mean, clean_spk_emb_mean)
            return loss1 + loss2
        elif loss_type == "type3":
            loss = sb.nnet.losses.mse_loss(spk_emb_mean, clean_spk_emb_mean)
            return loss


    def compute_objectives(self, predictions, batch, stage):
        if stage in [sb.Stage.TRAIN, sb.Stage.VALID]:
            # prepare clean targets for comparison
            B, F, T = batch.noisy_sig.data.shape
            loss_recons = sb.nnet.losses.mse_loss(
                predictions["est"].permute(0, 2, 1),
                batch.clean_sig.data.to(self.device).permute(0, 2, 1)
            )
            loss_recons_clean = sb.nnet.losses.mse_loss(
                predictions["clean_est"].permute(0, 2, 1),
                batch.clean_sig.data.to(self.device).permute(0, 2, 1)
            )
            # loss_speaker_vec = self.modules.model.calculate_speaker_vec_loss(
            loss_speaker_vec = self.calculate_speaker_vec_loss(
                predictions["spk_emb"], 
                predictions["spk_emb_mean"], 
                predictions["clean_spk_emb"],
                predictions["clean_spk_emb_mean"],
                loss_type=self.hparams.loss_spk_emb
            )

            if stage == sb.Stage.TRAIN:
                self.train_loss = {
                    "loss_sum": loss_recons + loss_recons_clean + loss_speaker_vec,
                    "loss_recons": loss_recons, 
                    "loss_recons_clean": loss_recons_clean,
                    "loss_speaker_vec": loss_speaker_vec
                }
            elif stage == sb.Stage.VALID:
                self.valid_loss = {
                    "loss_sum": loss_recons + loss_recons_clean + loss_speaker_vec,
                    "loss_recons": loss_recons, 
                    "loss_recons_clean": loss_recons_clean,
                    "loss_speaker_vec": loss_speaker_vec
                }
            
            return loss_recons + loss_recons_clean * 0.5 + loss_speaker_vec * 0.1


    def on_stage_start(self, stage, epoch=None):
        pass


    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.VALID:
            self.train_logger_tfboard.log_stats(
                {"Epoch": epoch}, train_stats=self.train_loss,
                valid_stats=self.valid_loss
            )
            self.train_logger_text.log_stats(
                {"Epoch": epoch}, train_stats=self.train_loss,
                valid_stats=self.valid_loss
            )

            self.checkpointer.save_and_keep_only(meta=self.valid_loss, num_to_keep=2, min_keys=["loss_sum"])

        if stage == sb.Stage.VALID:
            if epoch % 10 == 0:
                masked_list = self.test()
                for i in range(len(masked_list)):
                    self.train_logger_tfboard.writer.add_audio(
                        f"audio_{i+1}", masked_list[i] / np.abs(masked_list[i]).max(), epoch,
                        sample_rate=16000
                    )



    def test(self):
        data_dir = f"/n/work3/sekiguch/dataset/pyroomacoustics_librispeech/"
        flist = f"/n/work3/sekiguch/dataset/pyroomacoustics_librispeech/test_data_200.flist"
        masked_list = []
        with torch.no_grad():
            with open(flist, "r") as f:
                for line in f.readlines():
                    file_id = line.split(" ")[0].split("/")[-1]
                    fname_MPDR = f"{data_dir}/geometric_IR/sep_MPDR__{file_id}"
                    noisy_fname, clean_fname = line.replace("{data_dir}", data_dir).split(" ")
                    noisy, sr = sf.read(noisy_fname)
                    if sr != 16000:
                        noisy = resampy.resample(noisy, sr, 16000, axis=0)
                    sep_MPDR, sr = sf.read(fname_MPDR)

                    sep_MPDR_SFT = segmentation(
                        MultiSTFT(sep_MPDR, n_fft=hparams["n_fft"]),
                        segment_size=hparams["segment_size"],
                        overlap_ratio=hparams["overlap_ratio"]
                    ).to(self.device)
                    sep_MPDR_pwr_SFT = torch.abs(sep_MPDR_SFT) ** 2
                    S = len(sep_MPDR_pwr_SFT)

                    noisy_SFT = segmentation(
                        MultiSTFT(noisy[:, 0], n_fft=hparams["n_fft"]),
                        segment_size=hparams["segment_size"],
                        overlap_ratio=hparams["overlap_ratio"]
                    )[:S].to(self.device)
                    noisy_pwr_SFT = torch.abs(noisy_SFT) ** 2
                    print("shape : ", noisy_pwr_SFT.shape, sep_MPDR_pwr_SFT.shape)

                    masked_SFT = torch.zeros_like(sep_MPDR_SFT)
                    spk_emb = None
                    for s in range(S):
                        # pdb.set_trace()
                        if type(self.modules.model) == torch.nn.DataParallel:
                            mask_FT, spk_emb = self.modules.model.module.estimate_mask_segment(
                                noisy_pwr_SFT[s], sep_MPDR_pwr_SFT[s], spk_emb
                            )
                        else:
                            mask_FT, spk_emb = self.modules.model.estimate_mask_segment(
                                noisy_pwr_SFT[s], sep_MPDR_pwr_SFT[s], spk_emb
                            )
                        masked_SFT[s] = noisy_SFT[s] * mask_FT
                    masked_FT = overlap_add(masked_SFT, overlap_ratio=hparams["overlap_ratio"])
                    masked_list.append(MultiISTFT(masked_FT))
        return masked_list



def dataio_prep(hparams):

    @sb.utils.data_pipeline.takes("target", "noise_list")
    # @sb.utils.data_pipeline.provides("clean_pwr_spec", "noisy_pwr_spec", "enhanced_pwr_spec")
    @sb.utils.data_pipeline.provides("clean_sig", "noisy_sig", "enhanced_sig")
    def audio_pipeline(target, noise_list):
        # clean_sig = sb.dataio.dataio.readaudio(wav)
        clean_sig = sf.read(target)[0].astype(np.float32)
        noise_sig_list = []
        N = len(noise_list)
        for noise_path in noise_list:
            noise_sig_list.append(sf.read(noise_path)[0])

        min_length = min([len(x) for x in [clean_sig, *noise_sig_list]])
        clean_sig = clean_sig[:min_length]
        clean_sig = (clean_sig / np.abs(clean_sig).max() * np.random.rand()).astype(np.float32)
        noise_sig_list = [x[:min_length] / np.abs(x[:min_length]).max() for x in noise_sig_list]
        yield clean_sig

        noise = np.einsum("nt, n -> t", np.asarray([x[:min_length] for x in noise_sig_list]), np.random.rand(N))
        noise_enhanced = np.einsum("nt, n -> t", np.asarray([x[:min_length] for x in noise_sig_list]),  np.random.rand(N))

        noisy_SNR = np.random.rand() * 20 - 15 # -15 ~ 5
        SNR = 10 * np.log10((np.abs(clean_sig) ** 2).mean() / (np.abs(noise) ** 2).mean())
        scale = np.sqrt(10 ** ((SNR - noisy_SNR) / 10))
        noisy = (clean_sig + scale * noise).astype(np.float32)
        yield noisy

        enhanced_SNR = noisy_SNR + (np.random.rand() * 10 + 5) # noisy_SNR + (5~15)
        SNR = 10 * np.log10((np.abs(clean_sig) ** 2).mean() / (np.abs(noise_enhanced) ** 2).mean())
        scale = np.sqrt(10 ** ((SNR - enhanced_SNR) / 10))
        enhanced = (clean_sig + scale * noise_enhanced).astype(np.float32)
        yield enhanced


    datasets  ={}
    # hparams["dataloader_options"]["shuffle"] = False
    for dataset in ["train", "valid"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_json"],
            # replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline],
            output_keys={"id", "noisy_sig", "enhanced_sig", "clean_sig"}
        )#.filtered_sorted(sort_key="length")
    return datasets




if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    from hyperpyyaml import load_hyperpyyaml
    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f, overrides)

    datasets = dataio_prep(hparams)

    if hparams["model_name"] == "SB_ver1":
        from model_SB_ver1 import SB_ver1
        model = SB_ver1(
            input_length=hparams["segment_size"],
            z_dim = hparams["z_dim"] if "z_dim" in overrides else 128,
        )
        hparams["loss_spk_emb"] = hparams["loss_spk_emb"] if "loss_spk_emb" in overrides else "type3"
        output_folder = f"/n/work3/sekiguch/data_for_paper/ICASSP2022/train_log/{hparams['model_name']}/T{hparams['segment_size']}-Z{model.z_dim}-loss_{hparams['loss_spk_emb']}/"

    elif hparams["model_name"] == "SB_orig":
        from model_SB_orig import SB_orig
        model = SB_orig(
            z_dim = hparams["z_dim"] if "z_dim" in overrides else 30,
            hidden_channel = hparams["hidden_channel"],
            RNN_or_TF =  hparams["RNN_or_TF"],
        )
        hparams["loss_spk_emb"] = hparams["loss_spk_emb"] if "loss_spk_emb" in overrides else "type1"
        output_folder = f"/n/work3/sekiguch/data_for_paper/ICASSP2022/train_log/{hparams['model_name']}/T{hparams['segment_size']}-Z{model.z_dim}-H{hparams['hidden_channel']}-loss_{hparams['loss_spk_emb']}/"


    save_folder = f"{output_folder}/save"
    train_log = f"{output_folder}/train_log.txt"

    checkpointer = sb.utils.checkpoints.Checkpointer(
        checkpoints_dir=f"{save_folder}/{hparams['model_name']}",
        recoverables={
            "model": model,
            "counter": hparams["epoch_counter"]
        }
    )

    se_brain = SEBrain(
        modules={"model":model},
        opt_class=hparams["opt_class"],
        hparams=hparams,
        checkpointer=checkpointer,
        run_opts=run_opts
    )
    se_brain.model_name = model.model_name
    se_brain.train_logger_text = sb.utils.train_logger.FileTrainLogger(
        save_file=train_log
    )
    se_brain.train_logger_tfboard = sb.utils.train_logger.TensorboardLogger(
        save_dir=f"{save_folder}/tensorboard"
    )

    import collections
    from speechbrain.utils.data_utils import mod_default_collate
    from speechbrain.utils.data_utils import batch_pad_right
    from torch.utils.data._utils.collate import default_convert

    PaddedData = collections.namedtuple("PaddedData", ["data", "lengths"])

    class MyPaddedBatch(sb.dataio.batch.PaddedBatch):
        """Collate_fn when examples are dicts and have variable-length sequences.

        Different elements in the examples get matched by key.
        All numpy tensors get converted to Torch (PyTorch default_convert)
        Then, by default, all torch.Tensor valued elements get padded and support
        collective pin_memory() and to() calls.
        Regular Python data types are just collected in a list.

        Arguments
        ---------
        examples : list
            List of example dicts, as produced by Dataloader.
        padded_keys : list, None
            (Optional) List of keys to pad on. If None, pad all torch.Tensors
        device_prep_keys : list, None
            (Optional) Only these keys participate in collective memory pinning and moving with
            to().
            If None, defaults to all items with torch.Tensor values.
        padding_func : callable, optional
            Called with a list of tensors to be padded together. Needs to return
            two tensors: the padded data, and another tensor for the data lengths.
        padding_kwargs : dict
            (Optional) Extra kwargs to pass to padding_func. E.G. mode, value
        apply_default_convert : bool
            Whether to apply PyTorch default_convert (numpy to torch recursively,
            etc.) on all data. Default:True, usually does the right thing.
        nonpadded_stack : bool
            Whether to apply PyTorch-default_collate-like stacking on values that
            didn't get padded. This stacks if it can, but doesn't error out if it
            cannot. Default:True, usually does the right thing.

        """

        def __init__(
            self,
            examples,
            padded_keys=None,
            device_prep_keys=None,
            padding_func=batch_pad_right,
            padding_kwargs={},
            apply_default_convert=True,
            nonpadded_stack=True,
            hparams=hparams
            # segment_size=hparams["segment_size"]
            # segment_size=hparams["segment_size"]
        ):
            segment_size = hparams["segment_size"]
            overlap_ratio = hparams["overlap_ratio"]
            overlap = int(segment_size * overlap_ratio)
            shift = segment_size - overlap

            self.__length = len(examples)
            self.__keys = list(examples[0].keys())
            self.__padded_keys = []
            self.__device_prep_keys = []
            for key in self.__keys:
                values = [example[key] for example in examples]
                if "_sig" in key:

                    values = [
                        np.abs(librosa.core.stft(value, n_fft=1024, hop_length=256)) ** 2
                        for value in values
                    ]

                    F, T = values[0].shape
                    s1, s2 = values[0].strides
                    itemsize = values[0].itemsize

                    len_list = [(value.shape[1] - segment_size) // shift + 1    for value in values]

                    while 1:
                        if 0 in len_list:
                            idx = len_list.index(0)
                            del values[idx]
                            del len_list[idx]
                        else:
                            break

                    values = np.concatenate(
                        [as_strided(
                            values[i],
                            shape=(F, len_list[i], segment_size),
                            strides=(s1, itemsize * shift, itemsize)
                        ).transpose(1, 0, 2) for i in range(len(values))],
                        axis=0
                    )

                    if "noisy_sig" == key:
                        speaker_label = mod_default_collate(np.concatenate([
                            [i+1] * len_list[i] for i in range(len(len_list))
                        ]))
                        setattr(self, "speaker_label", speaker_label)


                # Default convert usually does the right thing (numpy2torch etc.)
                if apply_default_convert:
                    values = default_convert(values)
                if (padded_keys is not None and key in padded_keys) or (
                    padded_keys is None and isinstance(values[0], torch.Tensor)
                ):
                    # Padding and PaddedData
                    self.__padded_keys.append(key)
                    padded = PaddedData(*padding_func(values, **padding_kwargs))
                    setattr(self, key, padded)
                else:
                    # Default PyTorch collate usually does the right thing
                    # (convert lists of equal sized tensors to batch tensors, etc.)
                    if nonpadded_stack:
                        values = mod_default_collate(values)
                    setattr(self, key, values)
                if (device_prep_keys is not None and key in device_prep_keys) or (
                    device_prep_keys is None and isinstance(values[0], torch.Tensor)
                ):
                    self.__device_prep_keys.append(key)

    checkpointer.recover_if_possible(device=se_brain.device)
    hparams["dataloader_options"].update({"collate_fn": MyPaddedBatch})

    # test_stats = se_brain.evaluate(
    #     test_set=datasets["valid"],
    #     min_key="loss",
    #     test_loader_kwargs=hparams["dataloader_options"],
    # )
    # exit()
    # pdb.set_trace()

    se_brain.fit(
        epoch_counter=se_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )


