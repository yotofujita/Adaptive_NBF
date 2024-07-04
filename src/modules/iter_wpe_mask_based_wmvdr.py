import torch
import torchaudio

from opt_einsum import contract

from utils.import_utils import instantiate
from utils.lightning_utils import BaseModule


class IterWPEMaskBasedWMVDR(BaseModule):
    def __init__(self, mask_estimator, delay=3, tap=5, threshold=-10, lr=1e-3, skip_nan_grad=False, **kwargs):
        super().__init__(lr, skip_nan_grad)

        self.mask_estimator = instantiate(mask_estimator)
        
        self.delay = delay
        self.tap = tap
        
        self.threshold = threshold
        
        self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256, power=None)
        self.istft = torchaudio.transforms.InverseSpectrogram(n_fft=1024, hop_length=256)

    def step(self, batch):
        mix, *_ = batch  # [B, M, S], [B, M, S], [B,]

        stft = self.stft.to(self.device)
        istft = self.istft.to(self.device)

        # STFT
        mix_stft = stft(mix)  # [B, M, F, T]

        # Dereverberation
        mix_stft_dereverberated = self.dereverberate(mix_stft, delay=self.delay, tap=self.tap)  # [B, F, T]
    
        # Mask estimation
        mask = self.mask_estimator(mix_stft_dereverberated, batch)  # [B, T, F]
        
        # Beamforming
        sep_stft = self.beamform(mix_stft_dereverberated, mask)  # [B, F, T]

        # Inverse STFT
        sep = istft(sep_stft)  # [B, S]
        
        return sep, sep_stft, mask
    
    @staticmethod
    def dereverberate(mix_stft, delay=3, tap=5, eps=1e-4):

        B, M, F, T = mix_stft.shape
        D, K, L = delay, tap, delay+tap
    
        pad = torch.zeros((B, M, F, L-1), device=mix_stft.device)
        mix_padded = torch.concatenate((pad, mix_stft), dim=-1)

        mix_tilde = torch.concatenate([mix_padded[..., (L-i-1):T+L-1-i] for i in range(D, L)], dim=1)  # [B, M*K, F, T]
    
        w = torch.zeros((B, F, M, M*K), device=mix_stft.device, dtype=mix_stft.dtype)

        for _ in range(3):
            est = mix_stft - contract("bfkl, blft -> bkft", w.conj(), mix_tilde)
            sigma = torch.mean(torch.abs(est) ** 2, dim=1)
            sigma = sigma[:, None].expand((B, M*K, F, T))  # [B, M*K, F, T]
    
            R = contract("bkft,blft->bklf", mix_tilde / (sigma + eps), mix_tilde.conj())  # [B, M*K, M*K, F]
            R = R.permute((0, 3, 1, 2))  # [B, F, M*K, M*K]   
    
            P = contract("bkft,blft->bklf", mix_tilde / (sigma + eps), mix_stft.conj())  # [B, M*K, M*K, F]
            P = P.permute((0, 3, 1, 2))  # [B, F, M*K, M]   

            w = torch.linalg.solve(R, P)  # [B, F, M*K, M]
            w = w.permute((0, 1, 3, 2))  # [B, F, M, M*K]

            est = mix_stft - contract("bfkl,blft->bkft", w.conj(), mix_tilde)
        
        return est   

    @staticmethod
    def beamform(mix_stft, mask, ref_channel=0, eps=1e-4):
        mask = mask.permute((0, 2, 1))  # [B, F, T]
        
        B, M, F, T = mix_stft.shape

        speech_stft = mix_stft * mask[:, None]  # [B, M, F, T]
        noise_stft = mix_stft * (1 - mask)[:, None]  # [B, M, F, T]
    
        sigma = torch.mean(torch.abs(speech_stft)**2, dim=1)  # [B, F, T]
        sigma = sigma[:, None].expand((B, M, F, T))  # [B, M, F, T]
    
        R = contract("bkft,blft->bklf", noise_stft / (sigma + eps), noise_stft.conj())  # [B, M, M, F]
        R_inv = torch.linalg.inv(R.permute((0, 3, 1, 2)))  # [B, F, M, M]
    
        P = contract("bkft,blft->bklf", speech_stft, speech_stft.conj())  # [B, M, M, F]
        P = P.permute((0, 3, 1, 2))

        numerator = contract("bfkl,bfl->bfk", R_inv, P[..., ref_channel])
        denominator = torch.sum(torch.diagonal(contract("bfkl,bflm->bfkm", R_inv, P), dim1=-2, dim2=-1), dim=-1)[..., None]
        w = numerator / denominator
        est = contract("bfk,bkft->bft", w.conj(), mix_stft)
        
        return est
        

if __name__ == "__main__":
    from omegaconf import OmegaConf 
    cfg1 = OmegaConf.create()
    cfg1._target_ = "src.models.doa_aware_lstm.DOAAwareLSTM"
    cfg1.n_mics = 7
    cfg1.batchsize = 1
    
    device = "cuda"
    module = IterWPEMaskBasedWMVDR(cfg1).to(device).to(torch.float64)
    mix = torch.rand((cfg1.batchsize, cfg1.n_mics, 32000), dtype=torch.float64).to(device)
    src = torch.rand((cfg1.batchsize, cfg1.n_mics, 32000), dtype=torch.float64).to(device)
    doa = torch.rand((cfg1.batchsize), dtype=torch.float64).to(device)
    mic_shape = torch.rand((cfg1.batchsize, 2, cfg1.n_mics), dtype=torch.float64).to(device)
    sep, *_ = module.step((mix, src, doa, mic_shape))
    sep.mean().backward()
    