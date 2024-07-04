import torch
import pytorch_lightning as pl

from torchmetrics.audio import (
    SignalNoiseRatio,
    SignalDistortionRatio,
    ScaleInvariantSignalDistortionRatio
)
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio import ShortTimeObjectiveIntelligibility


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
    def __init__(self, lr, skip_nan_grad):
        super().__init__()
        
        self.lr = lr
        self.skip_nan_grad = skip_nan_grad
        
        self.snr = SignalNoiseRatio()
        self.sdr = SignalDistortionRatio()
        self.si_sdr = ScaleInvariantSignalDistortionRatio()
        self.pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')
        self.stoi = ShortTimeObjectiveIntelligibility(16000, False)
        
        self.metrics = {"loss": [], "SNR": [], "SDR": [], "SI-SDR": [], "PESQ": [], "STOI": []}
    
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
        self.metrics["PESQ"].append(self.pesq(est, src[:, 0]))
        self.metrics["STOI"].append(self.stoi(est, src[:, 0]))

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        
        for key in self.metrics.keys():
            self.log(
                f"valid_{key}", 
                torch.nanmean(torch.stack(self.metrics[key])), 
                sync_dist=True)

            self.metrics[key].clear()

    def test_step(self, batch, batch_idx):
        _, src, *_ = batch
        est, *_ = self.step(batch)
        
        self.metrics["SNR"].append(self.snr(est, src[:, 0]))
        self.metrics["SDR"].append(self.sdr(est, src[:, 0]))
        self.metrics["SI-SDR"].append(self.si_sdr(est, src[:, 0]))
        self.metrics["PESQ"].append(self.pesq(est, src[:, 0]))
        self.metrics["STOI"].append(self.stoi(est, src[:, 0]))

    def on_test_epoch_end(self):
        super().on_test_epoch_end()
        
        for key in self.metrics.keys():
            self.log(
                f"test_{key}", 
                torch.nanmean(torch.stack(self.metrics[key])), 
                sync_dist=True)

            self.metrics[key].clear()

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


class BaseBSSModule(BaseModule):
    def test_step(self, batch, batch_idx):
        _, src, *_ = batch
        est, *_ = self.step(batch)
        
        B, N, _ = est.shape
        
        sdrs = []
        for e, s in zip(est, src):
            ss = []
            for i in range(self.n_srcs+1):
                ss.append(self.sdr(e[i], s[0]))
            sdrs.append(torch.stack(ss))  # [N]
        sdrs = torch.stack(sdrs, dim=0)  # [B, N]

        target_idx = torch.argmax(sdrs, dim=1)  # [B]
        target_idx = torch.linspace(0, (B-1)*N, B).to(target_idx.device) + target_idx
        target_est = est.flatten(0, 1)[target_idx.to(torch.int)]
        
        self.metrics["SNR"].append(self.snr(target_est, src[:, 0]))
        self.metrics["SDR"].append(self.sdr(target_est, src[:, 0]))
        self.metrics["SI-SDR"].append(self.si_sdr(target_est, src[:, 0]))
        self.metrics["PESQ"].append(self.pesq(target_est, src[:, 0]))
        self.metrics["STOI"].append(self.stoi(target_est, src[:, 0]))
        
        return self.metrics

    def on_test_epoch_end(self):
        super().on_test_epoch_end()
        
        for key in self.metrics.keys():
            self.log(
                f"test_{key}", 
                torch.nanmean(torch.stack(self.metrics[key])), 
                sync_dist=True)

            self.metrics[key].clear()
    

