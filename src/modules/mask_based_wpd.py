import torch
import torchaudio

from opt_einsum import contract

from utils.import_utils import instantiate
from utils.lightning_utils import BaseModule


class MaskBasedWPD(BaseModule):
    def __init__(self, mask_estimator, delay=3, tap=5, threshold=-10, lr=1e-3, skip_nan_grad=False, eval_asr=False, asr_batch_size=10, **kwargs):
        super().__init__(lr=lr, skip_nan_grad=skip_nan_grad, eval_asr=eval_asr)

        self.mask_estimator = instantiate(mask_estimator)
        
        self.delay = delay
        self.tap = tap
        
        self.threshold = threshold
        
        self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256, power=None)
        self.istft = torchaudio.transforms.InverseSpectrogram(n_fft=1024, hop_length=256)

    def step(self, batch):
        mix, src, *_ = batch  # [B, M, S], [B, M, S], [B,]

        stft = self.stft.to(self.device)
        istft = self.istft.to(self.device)
        
        # STFT
        mix_stft = stft(mix)  # [B, M, F, T]

        # Mask estimation 
        mask = self.mask_estimator(mix_stft, batch)  # [B, T, F]

        # WPD
        # sep_stft = self.beamform(mix_stft, stft(src), delay=self.delay, tap=self.tap)  # [B, F, T]
        sep_stft = self.beamform(mix_stft, mask, delay=self.delay, tap=self.tap)  # [B, F, T]

        # Inverse STFT
        sep = istft(sep_stft)  # [B, S]
        
        return sep, sep_stft, mask

    @staticmethod
    def beamform(mix_stft, mask, delay=3, tap=5, ref_channel=0, eps=1e-4):
        mask = mask.permute((0, 2, 1))  # [B, F, T]
        
        B, M, F, T = mix_stft.shape
        D, K, L = delay, tap, delay+tap
        
        speech_est = mix_stft * mask[:, None]  # [B, M, F, T]
        # speech_est = mask
    
        pad = torch.zeros((B, M, F, L-1), device=mix_stft.device)
        mix_padded = torch.concatenate((pad, mix_stft), dim=-1)
        speech_padded = torch.concatenate((pad, speech_est), dim=-1)
    
        mix_tilde = torch.concatenate([mix_padded[..., (L-i-1):T+L-1-i] for i in [0] + list(range(D, L))], dim=1)  # [B, M*(K+1), F, T]
        speech_tilde = torch.concatenate([speech_padded[..., (L-i-1):T+L-1-i] for i in [0] + list(range(D, L))], dim=1)  # [B, M*(K+1), F, T]
    
        sigma = torch.mean(torch.abs(speech_est)**2, dim=1)  # [B, F, T]
        sigma = sigma[:, None].expand((B, M*(K+1), F, T))  # [B, M*(K+1), F, T]
    
        R = contract("bkft,blft->bklf", mix_tilde / (sigma + eps), mix_tilde.conj())  # [B, M*(K+1), M*(K+1), F]
        R_inv = torch.linalg.inv(R.permute((0, 3, 1, 2)))  # [B, F, M*(K+1), M*(K+1)]
    
        P = contract("bkft,blft->bklf", speech_tilde, speech_tilde.conj())  # [B, M*(K+1), M*(K+1), F]
        P = P.permute((0, 3, 1, 2))  # [B, F, M*(K+1), M*(K+1)]
    
        numerator = contract("bfkl,bfl->bfk", R_inv, P[..., ref_channel])  # [B, F, M*(K+1)]
        denominator = torch.sum(torch.diagonal(contract("bfkl,bflm->bfkm", R_inv, P), dim1=-2, dim2=-1), dim=-1)[..., None]  # [B, F, 1]
        w = numerator / denominator  # [B, F, M*(K+1)]
        est = contract("bfk,bkft->bft", w.conj(), mix_tilde)
        
        return est
        

if __name__ == "__main__":
    from omegaconf import OmegaConf 
    cfg1 = OmegaConf.create()
    cfg1._target_ = "src.models.doa_aware_lstm.DOAAwareLSTM"
    cfg1.n_mics = 7
    cfg1.batchsize = 1
    
    device = "cuda"
    module = MaskBasedWPD(cfg1).to(device).to(torch.float64)
    mix = torch.rand((cfg1.batchsize, cfg1.n_mics, 32000), dtype=torch.float64).to(device)
    src = torch.rand((cfg1.batchsize, cfg1.n_mics, 32000), dtype=torch.float64).to(device)
    doa = torch.rand((cfg1.batchsize), dtype=torch.float64).to(device)
    mic_shape = torch.rand((cfg1.batchsize, 2, cfg1.n_mics), dtype=torch.float64).to(device)
    sep, *_ = module.step((mix, src, doa, mic_shape))
    sep.mean().backward()
    