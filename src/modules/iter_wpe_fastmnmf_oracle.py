import torch
import torchaudio

from opt_einsum import contract

from utils.lightning_utils import BaseModule
from utils.FastMNMF2 import FastMNMF2
from utils.sp_utils import calc_sv

from torchmetrics.audio import ScaleInvariantSignalDistortionRatio


class FastMNMFOracle(BaseModule):
    def __init__(self, delay=3, tap=5, n_srcs=5, eval_asr=False, asr_batch_size=10, **kwargs):
        super().__init__(eval_asr=eval_asr, asr_batch_size=asr_batch_size)
        
        self.delay = delay
        self.tap = tap
        
        self.n_srcs = n_srcs
        
        self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256, power=None)
        self.istft = torchaudio.transforms.InverseSpectrogram(n_fft=1024, hop_length=256)
        
        self.separater = FastMNMF2(
            n_source=n_srcs,
            n_basis=16,
            device="cuda",
            init_SCM="twostep",
            n_bit=64,
            algo="IP",
            n_iter_init=30,
            g_eps=5e-2,
        )
        
        self.si_sdr = ScaleInvariantSignalDistortionRatio()

    def step(self, batch):
        mix, src, *_ = batch  # [B, M, S], [B, M, S]

        # STFT
        mix_stft = self.stft.to(self.device)(mix)  # [B, M, F, T]
        
        mix_stft_dereverberated = self.dereverberate(mix_stft)

        # Separation
        sep = []
        for mx, s in zip(mix_stft_dereverberated, src):
            est_wave = self.separate(mx, s[0])
            sep.append(est_wave)
        
        return torch.stack(sep), mix_stft_dereverberated, mix_stft

    @staticmethod
    def dereverberate(mix_stft, delay=3, tap=10, eps=1e-4):

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

    def separate(self, mix_stft, src_wave):
        self.separater.load_spectrogram(mix_stft.permute((1, 2, 0)), sample_rate=16000)
        est = self.separater.solve(n_iter=200)  # [N, F, T]

        # Extract speeches by non-intrusive SDR 
        est_wave = self.istft.to(self.device)(est)  # [N, S]
        si_sdr = self.si_sdr.to(self.device)
        ss = []
        for ew in est_wave:
            ss.append(si_sdr(ew, src_wave))
        
        return est_wave[torch.argmax(torch.stack(ss))]
        

if __name__ == "__main__":
    from omegaconf import OmegaConf 
    cfg1 = OmegaConf.create()
    cfg1.n_mics = 7
    cfg1.batchsize = 5
    
    device = "cuda"
    module = FastMNMFDOAEst().to(device).to(torch.float64)
    mix = torch.rand((cfg1.batchsize, cfg1.n_mics, 32000), dtype=torch.float64).to(device)
    # src = torch.rand((cfg1.batchsize, cfg1.n_mics, 32000), dtype=torch.float64).to(device)
    # doa = torch.rand((cfg1.batchsize), dtype=torch.float64).to(device)
    mic_shape = torch.rand((cfg1.batchsize, 2, cfg1.n_mics), dtype=torch.float64).to(device)
    sep, doa = module.step((mix, None, None, mic_shape))
    for s, d in zip(sep, doa):
        print(s.shape, d.shape)
    