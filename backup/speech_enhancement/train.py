#!/usr/bin/env python3
# coding:utf-8
import numpy as np
import torch
import speechbrain as sb
import soundfile as sf
import sys
import pdb

class SEBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        if stage != sb.Stage.TEST:
            noisy_pwr_spec = self.compute_feats(batch.noisy_sig.data.to(self.device)) # B x F x T
            enhanced_pwr_spec = self.compute_feats(batch.enhanced_sig.data.to(self.device)) # B x F x T
            return self.modules.model(noisy_pwr_spec, enhanced_pwr_spec) # B x 2F x T
        else:
            noisy_sig = batch.noisy_sig.data.to(self.device)
            noisy_pwr_spec = self.compute_feats(noisy_sig) # B x F x T
            enhanced_pwr_spec = self.compute_feats(batch.enhanced_sig.data.to(self.device)) # B x F x T
            output = self.modules.model(noisy_pwr_spec, enhanced_pwr_spec).permute(0, 2, 1) # B x 2F x T
            wav = self.hparams.resynth(torch.sqrt(output), noisy_sig)
            sf.write("input_noisy.wav", batch.noisy_sig.data.detach().numpy().T, 16000)
            sf.write("clean.wav", batch.clean_sig.data.detach().numpy().T, 16000)
            sf.write("input_enhanced.wav", batch.enhanced_sig.data.detach().numpy().T, 16000)
            sf.write("enhanced.wav", wav.to("cpu").detach().numpy().T, 16000)


    # def compute_forward_finetune(self, batch, stage):
    #     batch  = batch.to(self.device)
    #     noisy_pwr_spec = batch.noisy_pwr_spec.data # B x F x T
    #     enhanced_pwr_spec = batch.enhanced_pwr_spec.data # B x F x T
    #     x = torch.cat([noisy_pwr_spec, enhanced_pwr_spec], dim=1) # B x 2F x T

    #     mask = self.modules.model(x).squeeze(0) # B x F x T
    #     masked_speech = batch.noisy_multi_spec.data * mask.unsqueeze(1) # B x M x F x T
    #     masked_noise = batch.noisy_multi_spec.data * (1-mask).unsqueeze(1) # B x M x F x T
    #     speech_SCM = torch.einsum("bfti, bftj -> bfij", masked_speech, masked_speech.conj())
    #     noise_SCM = torch.einsum("bfti, bftj -> bfij", masked_noise, masked_noise.conj())
    #     eig_val, eig_vec = torch.linalg.eigh(speech_SCM) # B x F x M x M
    #     print(eig_val[0, 10])
    #     steering_vec = torch.linalg.eigh(speech_SCM)[1][..., -1] # B x F x M x M

    #     filter_BFM = torch.einsum("bfim, bfm", torch.linalg.inv(noise_SCM), steering_vec)
    #     filter_BFM /= (steering_vec.conj() * filter_BFM).sum(axis=-1).unsqueeze(-1)

    #     return torch.einsum("bfm, bftm -> bft", filter_BFM.conj(), bath.noisy_multi_spec)

    def compute_objectives(self, predictions, batch, stage):
        if stage in [sb.Stage.TRAIN, sb.Stage.VALID]:
            # prepare clean targets for comparison
            clean_spec = self.compute_feats(batch.clean_sig.data.to(self.device))
            lens = batch.clean_sig.lengths.to(self.device)
            loss = sb.nnet.losses.mse_loss(predictions.permute(0, 2, 1), clean_spec.permute(0, 2, 1), lens)

            # append this batch of losses to the loss metric for easy summarization
            self.loss_metric.append(
                batch.id, predictions, clean_spec, lens, reduction="batch"
            )
            
            return loss

    def compute_feats(self, wavs):
        feats = self.hparams.compute_STFT(wavs)
        feats = sb.processing.features.spectral_magnitude(feats, power=1)
        # power = 1 -> power spec, power = 0.5 -> magnitude

        # feats = torch.log1p(feats).permute(0, 2, 1)
        return feats.permute(0, 2, 1)

    def on_stage_start(self, stage, epoch=None):
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.mse_loss
        )

        # if stage != sb.Stage.TRAIN:
        #     self.stoi_metric = sb.utils.metric_stats.MetricStats(
        #         metric = sb.nnet.loss.stoi_loss.stoi_loss
        #     )

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        
        if stage == sb.Stage.VALID:
            stats = {
                "loss": stage_loss,
            }
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch}, train_stats={"loss": self.train_loss},
                valid_stats=stats
            )
            self.hparams.train_logger_text.log_stats(
                {"Epoch": epoch}, train_stats={"loss": self.train_loss},
                valid_stats=stats
            )

            self.checkpointer.save_and_keep_only(meta=stats, num_to_keep=1, min_keys=["loss"])

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

    # from torchsummary import summary
    # pdb.set_trace()
    # summary(hparams["model"], (2, 513, 40), (2, 513, 40))
    # exit()

    se_brain = SEBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
        run_opts=run_opts
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
            segment_size=hparams["segment_size"]
        ):            
            self.__length = len(examples)
            self.__keys = list(examples[0].keys())
            self.__padded_keys = []
            self.__device_prep_keys = []
            for key in self.__keys:
                values = [example[key] for example in examples]
                if "_sig" in key:
                    len_list = [(len(value) // segment_size) * segment_size for value in values]
                    values = np.concatenate(
                        [values[i][:len_list[i]].reshape(-1, segment_size) 
                            for i in range(self.__length)],
                        axis=0
                    )
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


    hparams["checkpointer"].recover_if_possible(device=se_brain.device)
    hparams["dataloader_options"].update({"collate_fn": MyPaddedBatch})

    # test_stats = se_brain.evaluate(
    #     test_set=datasets["valid"],
    #     min_key="loss",
    #     test_loader_kwargs=hparams["dataloader_options"],
    # )
    # exit()
    pdb.set_trace()

    se_brain.fit(
        epoch_counter=se_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )


# data_folder = "./data"
# output_folder =!ref /n/work3/sekiguch/data_for_paper/ICASSP2022/train_log/<model_name>/<seed>
# save_folder =!ref <output_folder>/save
# train_log =!ref <output_folder>/train_log.txt

# checkpointer: !new:speechbrain.utils.checkpoints.Checkpointer
#     checkpoints_dir: !ref <save_folder>/<model_name>
#     recoverables:
#         model: !ref <model>
#         counter: !ref <epoch_counter>


# # The train logger writes training statistics to a file, as well as stdout.
# train_logger_text: !new:speechbrain.utils.train_logger.FileTrainLogger
#     save_file: !ref <train_log>
# train_logger: !new:speechbrain.utils.train_logger.TensorboardLogger
#     save_dir: !ref <save_folder>/tensorboard

