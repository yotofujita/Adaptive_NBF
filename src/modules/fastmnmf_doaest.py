import torch
import torchaudio

from opt_einsum import contract

from utils.lightning_utils import BaseModule
from utils.FastMNMF2 import FastMNMF2
from utils.sp_utils import calc_sv

from torchaudio.pipelines import SQUIM_OBJECTIVE


class FastMNMFDOAEst(BaseModule):
    def __init__(self, delay=3, tap=5, n_srcs=5, threshold=-14, lr=1e-3, skip_nan_grad=False, eval_asr=False, asr_batch_size=10, **kwargs):
        super().__init__(lr=lr, skip_nan_grad=skip_nan_grad, eval_asr=eval_asr, asr_batch_size=asr_batch_size)
        
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
        
        self.threshold = threshold
        self.objective_model = SQUIM_OBJECTIVE.get_model()

    def step(self, batch):
        mix, _, _, mic_shape = batch  # [B, M, S], [B, M, S], [B,]

        # STFT
        mix_stft = self.stft.to(self.device)(mix)  # [B, M, F, T]

        # Separation
        sep, doa = [], []
        for mx, mc in zip(mix_stft, mic_shape):
            est_wave, est_doa = self.separate(mx, mc)
            sep.append(est_wave)
            doa.append(est_doa)
        
        return sep, doa

    def separate(self, mix_stft, mic_shape):
        self.separater.load_spectrogram(mix_stft.permute((1, 2, 0)), sample_rate=16000)
        est = self.separater.solve(n_iter=200)  # [N, F, T]

        # Extract speeches by non-intrusive SDR 
        est_wave = self.istft.to(self.device)(est)  # [N, S]
        si_sdr_hyp = []
        for ew in est_wave:
            si_sdr_hyp.append(self.objective_model.to(est_wave.dtype)(ew[None, :])[-1])
        si_sdr_hyp = torch.concatenate(si_sdr_hyp, dim=0)
        est_wave = est_wave[si_sdr_hyp > self.threshold]  # [N', S]

        # Localization by MUSIC 
        G = self.separater.get_SCM()[si_sdr_hyp > self.threshold]  # [N', F, M, M]
        _, v = torch.linalg.eigh(G)  # [N', F, M], [N', F, M, M]
        doas = torch.linspace(0, 360, 360).to(est_wave.device).to(est_wave.dtype)
        svs = calc_sv(doas+180, torch.stack([mic_shape.to(doas.dtype) for _ in range(len(doas))], dim=0), est.shape[1]).to(G.device)  # [360, F, M]
        ss = (contract("bfl,nflm->bnfm", svs, v[:, :, :, :-1]).abs()**2).sum(dim=-1).sum(dim=-1)  # [360, N']
        est_doa = torch.argmin(ss, dim=0)  # [N']
        
        return est_wave, est_doa
        

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
    