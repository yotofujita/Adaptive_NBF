import os
import logging

import numpy as np

import torch
import pytorch_lightning as pl

from torchmetrics.audio import (
    SignalNoiseRatio,
    SignalDistortionRatio,
    ScaleInvariantSignalDistortionRatio,
    ShortTimeObjectiveIntelligibility
)
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.srmr import SpeechReverberationModulationEnergyRatio

from speechbrain.inference.ASR import EncoderDecoderASR
from torchmetrics.text import WordErrorRate


class TextLogger(pl.loggers.Logger):
    def __init__(self, log_dir, name="log"):
        super().__init__()
        
        self.log_dir = log_dir
        self.name = name
        
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"{self.name}.log")
        
        # Set up logging to the file
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
    
    @property
    def experiment(self):
        # Return the experiment object associated with this logger
        return self.logger
    
    @property
    def version(self):
        # Return the version of the experiment
        return "1.0"

    @property
    def name(self):
        # Return the name of the experiment
        return self.name

    def log_hyperparams(self, params):
        # Log hyperparameters
        self.logger.info(f"Hyperparameters: {params}")

    def log_metrics(self, metrics, step):
        # Log metrics
        self.logger.info(f"Step {step}: {metrics}")

    def save(self):
        # Any logic you need to save the logger state
        pass

    def finalize(self, status):
        # Any finalization logic when the experiment is done
        self.logger.info(f"Finalizing logger with status: {status}")



class BaseDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super().__init__()
        
        self.batch_size = batch_size 
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = None
            self.valid_dataset = None
        if stage == "test":
            self.test_dataset = None
    
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

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )


class BaseModule(pl.LightningModule):
    def __init__(self, lr=1e-4, skip_nan_grad=True, eval_asr=False, asr_batch_size=10, sr=16000):
        super().__init__()
        
        self.lr = lr
        self.skip_nan_grad = skip_nan_grad
        self.eval_asr = eval_asr
        self.asr_batch_size = asr_batch_size
        self.sr = sr
        
        self.snr = SignalNoiseRatio()
        self.sdr = SignalDistortionRatio()
        self.si_sdr = ScaleInvariantSignalDistortionRatio()
        self.pesq = PerceptualEvaluationSpeechQuality(sr, 'wb')
        self.stoi = ShortTimeObjectiveIntelligibility(sr, False)
        self.srmr = SpeechReverberationModulationEnergyRatio(sr)
        self.metrics = {"loss": [], "SNR": [], "SDR": [], "SI-SDR": [], "PESQ": [], "STOI": [], "SRMR": []}
        
        if eval_asr:
            self.wer = WordErrorRate()
            self.estimates = []
    
    def training_step(self, batch, batch_idx):
        _, src, *_ = batch
        est, *_ = self.step(batch)
        
        loss = - self.sdr(est, src[:, 0])

        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, src, *_ = batch
        est, *_ = self.step(batch)
        
        self.metrics["loss"].append(- self.sdr(est, src[:, 0]))
        self.metrics["SNR"].append(self.snr(est, src[:, 0]))
        self.metrics["SDR"].append(self.sdr(est, src[:, 0]))
        self.metrics["SI-SDR"].append(self.si_sdr(est, src[:, 0]))
        try:
            self.metrics["PESQ"].append(self.pesq(est, src[:, 0]))
        except:
            self.metrics["PESQ"].append(torch.tensor(torch.nan, device=est.device, dtype=est.dtype))
        self.metrics["STOI"].append(self.stoi(est, src[:, 0]))
        self.metrics["SRMR"].append(self.srmr(est))

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        
        for key in self.metrics.keys():
            self.log(
                f"valid_{key}", 
                torch.nanmean(torch.stack(self.metrics[key])), 
                sync_dist=True)

            self.metrics[key].clear()

    def test_step(self, batch, batch_idx):
        if not self.eval_asr:
            _, src, *_ = batch
            est, *_ = self.step(batch)
        
            self.metrics["loss"].append(- self.sdr(est, src[:, 0]))
            self.metrics["SNR"].append(self.snr(est, src[:, 0]))
            self.metrics["SDR"].append(self.sdr(est, src[:, 0]))
            self.metrics["SI-SDR"].append(self.si_sdr(est, src[:, 0]))
            try:
                self.metrics["PESQ"].append(self.pesq(est, src[:, 0]))
            except:
                self.metrics["PESQ"].append(torch.tensor(torch.nan, device=est.device, dtype=est.dtype))
            self.metrics["STOI"].append(self.stoi(est, src[:, 0]))
        
        else:  # ASR target is included at the end for measuring WER
            if not hasattr(self, "asr_target"):
                self.asr_target = batch[-1][0]

            _, src, *_ = batch[:-1]
            est, *_ = self.step(batch[:-1])
    
            self.metrics["loss"].append(- self.sdr(est, src[:, 0]))
            self.metrics["SNR"].append(self.snr(est, src[:, 0]))
            self.metrics["SDR"].append(self.sdr(est, src[:, 0]))
            self.metrics["SI-SDR"].append(self.si_sdr(est, src[:, 0]))
            try:
                self.metrics["PESQ"].append(self.pesq(est, src[:, 0]))
            except:
                self.metrics["PESQ"].append(torch.tensor(torch.nan, device=est.device, dtype=est.dtype))
            self.metrics["STOI"].append(self.stoi(est, src[:, 0]))
            self.metrics["SRMR"].append(self.srmr(est))
            
            self.estimates.append(est)

    def on_test_epoch_end(self):
        super().on_test_epoch_end()
        
        for key in self.metrics.keys():
            self.log(
                f"test_{key}", 
                torch.nanmean(torch.stack(self.metrics[key])), 
                sync_dist=True)

            self.metrics[key].clear()

        if self.eval_asr:
            asr_model = EncoderDecoderASR.from_hparams(
                source="speechbrain/asr-transformer-transformerlm-librispeech",  
                savedir="src/pretrained_models/asr-transformer-transformerlm-librispeech",
                run_opts={"device": "cuda", "dtype": "float32"}
            )

            speech_est = torch.concatenate(self.estimates, dim=0)
            speech_est, speech_est_rest = speech_est[:(speech_est.shape[0] - speech_est.shape[0] % (2*self.asr_batch_size))], speech_est[(speech_est.shape[0] - speech_est.shape[0] % (2*self.asr_batch_size)):]
            speech_est = speech_est.reshape(2, self.asr_batch_size, -1)
            speech_est_rest = speech_est_rest.reshape(1, -1)
            
            transcriptions = []
            print("Transcribing.")
            for se in speech_est:
                transcriptions.extend(asr_model.transcribe_batch(
                    se, 
                    wav_lens=torch.tensor([1. for _ in range(se.shape[0])])
                )[0])
            if speech_est_rest.shape[-1] != 0:
                transcriptions.extend(asr_model.transcribe_batch(
                    speech_est_rest, 
                    wav_lens=torch.tensor([1. for _ in range(speech_est_rest.shape[0])])
                )[0])
                
            transcription = " ".join(transcriptions)

            self.log("test_WER", self.wer(transcription, self.asr_target), sync_dist=True)
            

    def on_after_backward(self):
        super().on_after_backward()

        if hasattr(self, 'skip_nan_grad') and self.skip_nan_grad:
            device = next(self.parameters()).device
            valid_gradients = torch.tensor([1], device=device, dtype=torch.float32)

            for _, param in self.named_parameters():
                if param.grad is not None:
                    is_not_nan_or_inf = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                    if not is_not_nan_or_inf:
                        valid_gradients = valid_gradients * 0
                        break

            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(valid_gradients, op=torch.distributed.ReduceOp.MIN)

            if valid_gradients < 1:
                self.zero_grad()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def __str__(self):
        return f"{self.__class__.__name__}"


class BaseOracleModule(BaseModule):
    def training_step(self, batch, batch_idx):
        _, src, *_ = batch
        est, _, mask = self.step(batch)
        
        loss = - self.sdr(est, src[:, 0])

        self.log("train_loss", loss, sync_dist=True)
        
        return loss + mask.mean()
