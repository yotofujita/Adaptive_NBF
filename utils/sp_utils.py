import torch
from opt_einsum import contract
import pyroomacoustics as pra


def deg2pos(deg, distance=1., center=torch.tensor([0., 0.])):
    _cos = torch.cos(torch.deg2rad(deg))
    _sin = torch.sin(torch.deg2rad(deg))
    _rot_mat = torch.stack([torch.stack([_cos, -_sin], dim=-1), torch.stack([_sin, _cos], dim=-1)], dim=1)
    _front = torch.tensor([distance, 0.], device=_rot_mat.device, dtype=_rot_mat.dtype)
    _center = center.to(_rot_mat.device).to(_rot_mat.dtype)
    return _rot_mat @ _front + _center


def calc_sv(theta, mic_shape, n_freq, sr=16000, ref_mic=0, c=pra.parameters.Physics().get_sound_speed()):
    k1 = deg2pos(theta)
    m = mic_shape - mic_shape[..., ref_mic][..., None]
    freq = torch.linspace(0, sr / 2, n_freq).to(k1.device).to(k1.dtype)
    if k1.dim() == 1:
        return torch.exp(2j*torch.pi*(k1@m/c)*freq[:, None])
    elif k1.dim() == 2:
        return torch.exp(2j*torch.pi*contract("bm,f->bfm", contract("bd,bdm->bm", k1, m/c), freq))
    else:
        assert False


def calc_sv2(theta, mic_shape, n_freq, sr=16000, ref_mic=0, c=pra.parameters.Physics().get_sound_speed()):
    k1 = deg2pos(theta)
    m = mic_shape - mic_shape[..., ref_mic][..., None]
    freq = torch.linspace(0, sr / 2, n_freq).to(k1.device).to(k1.dtype)
    return torch.exp(2j*torch.pi*contract("bm,f->bfm", contract("bd,bdm->bm", k1, m/c), freq))


def SI_SDR(estimate, reference, epsilon=1e-8):
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    reference_pow = reference.pow(2).mean(axis=1, keepdim=True)
    mix_pow = (estimate * reference).mean(axis=1, keepdim=True)
    scale = mix_pow / (reference_pow + epsilon)

    reference = scale * reference
    error = estimate - reference

    reference_pow = reference.pow(2)
    error_pow = error.pow(2)

    reference_pow = reference_pow.mean(axis=1)
    error_pow = error_pow.mean(axis=1)

    sisdr = 10 * torch.log10(reference_pow) - 10 * torch.log10(error_pow)
    return sisdr


def SDR(estimate, reference):
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()

    error = estimate - reference

    reference_pow = reference.pow(2)
    error_pow = error.pow(2)

    reference_pow = reference_pow.mean(axis=1)
    error_pow = error_pow.mean(axis=1)

    sisdr = 10 * torch.log10(reference_pow) - 10 * torch.log10(error_pow)
    return sisdr


def WPD(mix, mask1, mask2, delay=3, tap=5, ref_channel=0, eps=1e-4, return_filter=False):
    if mix.ndim == 3:
        mix = mix[None]  # [B, M, T, F]
        mask1 = mask1[None]  # [B, T, F]
        mask2 = mask2[None]  # [B, T, F]
    
    mix = mix.permute((0, 1, 3, 2))  # [B, M, F, T]
    mask1 = mask1.permute((0, 2, 1))  # [B, F, T]
    mask2 = mask2.permute((0, 2, 1))  # [B, F, T]
        
    B, M, F, T = mix.shape
    D, K, L = delay, tap, delay+tap
        
    speech_est1 = mix * mask1[:, None]  # [B, M, F, T]
    speech_est2 = mix * mask2[:, None]  # [B, M, F, T]
    
    pad = torch.zeros((B, M, F, L-1), device=mix.device)
    mix_padded = torch.concatenate((pad, mix), dim=-1)
    speech_est2_padded = torch.concatenate((pad, speech_est2), dim=-1)
    
    mix_tilde = torch.concatenate([mix_padded[..., (L-i-1):T+L-1-i] for i in [0] + list(range(D, L))], dim=1)  # [B, M*(K+1), F, T]
    speech_est2_tilde = torch.concatenate([speech_est2_padded[..., (L-i-1):T+L-1-i] for i in [0] + list(range(D, L))], dim=1)  # [B, M*(K+1), F, T]
    
    speech_power = torch.mean(torch.abs(speech_est1)**2, dim=1)  # [B, F, T]
    speech_power = speech_power[:, None, None].expand((B, M*(K+1), M*(K+1), F, T))  # [B, M*(K+1), M*(K+1), F, T]
    
    mix_SCM = torch.mean(contract("bkft,blft->bklft", mix_tilde, mix_tilde.conj()) / (speech_power + eps), dim=-1)  # [B, M*(K+1), M*(K+1), F]
    mix_SCM_inv = torch.linalg.inv(mix_SCM.permute((0, 3, 1, 2)))  # [B, F, M*(K+1), M*(K+1)]
    
    speech_SCM = torch.mean(contract("bkft,blft->bklft", speech_est2_tilde, speech_est2_tilde.conj()) / (speech_power + eps), dim=-1)  # [B, M*(K+1), M*(K+1), F]
    speech_SCM = speech_SCM.permute((0, 3, 1, 2))  # [B, F, M*(K+1), M*(K+1)]
    
    numerator = contract("bfkl,bfl->bfk", mix_SCM_inv, speech_SCM[..., ref_channel])  # [B, F, M*(K+1)]
    denominator = torch.sum(torch.diagonal(contract("bfkl,bflm->bfkm", mix_SCM_inv, speech_SCM), dim1=-2, dim2=-1), dim=-1)[..., None]  # [B, F, 1]
    w = numerator / denominator  # [B, F, M*(K+1)]
    enhanced_signal = contract("bfk,bkft->bft", w.conj(), mix_tilde)

    if return_filter:
        return enhanced_signal, w
    else:
        return enhanced_signal


def MVDR(mixture_BMTF, mask_speech_BTF, mask_noise_BTF, ref_channel=0, eps=1e-7, return_filter=False):
    if mixture_BMTF.ndim == 3:
        mixture_BMTF = mixture_BMTF[None]
        mask_speech_BTF = mask_speech_BTF[None]
        mask_noise_BTF = mask_noise_BTF[None]
    n_mic = mixture_BMTF.shape[1]
    speech_est_BMTF = mixture_BMTF * mask_speech_BTF.unsqueeze(1)
    speech_SCM_BFMM = (
        torch.einsum("bitf, bjtf -> bfij", speech_est_BMTF, speech_est_BMTF.conj())
        + eps * torch.eye(n_mic, device=mixture_BMTF.device)[None, None]
    )

    noise_est_BMTF = mixture_BMTF * mask_noise_BTF.unsqueeze(1)
    noise_SCMinv_BFMM = torch.linalg.inv(
        torch.einsum("bitf, bjtf -> bfij", noise_est_BMTF, noise_est_BMTF.conj())
        + eps * torch.eye(n_mic, device=mixture_BMTF.device)[None, None]
    )

    W_BFM = torch.einsum("bfij, bfj -> bfi", noise_SCMinv_BFMM, speech_SCM_BFMM[..., ref_channel]) / torch.sum(
        torch.diagonal(noise_SCMinv_BFMM @ speech_SCM_BFMM, dim1=-2, dim2=-1), dim=-1
    ).unsqueeze(-1)

    if return_filter:
        return torch.einsum("bitf, bfi -> bft", mixture_BMTF, W_BFM.conj()), W_BFM
    else:
        return torch.einsum("bitf, bfi -> bft", mixture_BMTF, W_BFM.conj())


def MA_MVDR(mixture_BMTF, mask_speech_BTF, mask_noise_BTF, weight=0.25, ref_channel=0, eps=1e-7, return_filter=False):
    if mixture_BMTF.ndim == 3:
        raise ValueError("MA_MVDR should be used for sequence data")
    n_batch, n_mic, n_time, n_freq = mixture_BMTF.shape
    speech_est_BMTF = mixture_BMTF * mask_speech_BTF.unsqueeze(1)
    speech_SCM_BFMM = (
        torch.einsum("bitf, bjtf -> bfij", speech_est_BMTF, speech_est_BMTF.conj())
        + eps * torch.eye(n_mic, device=mixture_BMTF.device)[None, None]
    )

    noise_est_BMTF = mixture_BMTF * mask_noise_BTF.unsqueeze(1)
    noise_SCM_BFMM = (
        torch.einsum("bitf, bjtf -> bfij", noise_est_BMTF, noise_est_BMTF.conj())
        + eps * torch.eye(n_mic, device=mixture_BMTF.device)[None, None]
    )

    sep_BFT = torch.zeros([n_batch, n_freq, n_time], dtype=mixture_BMTF.dtype, device=mixture_BMTF.device)
    for b in range(n_batch):
        if b == 0:
            MA_speech_SCM_FMM = speech_SCM_BFMM[b]
            MA_noise_SCM_FMM = noise_SCM_BFMM[b]
        else:
            MA_speech_SCM_FMM = (1 - weight) * MA_speech_SCM_FMM + weight * speech_SCM_BFMM[b]
            MA_noise_SCM_FMM = (1 - weight) * MA_noise_SCM_FMM + weight * noise_SCM_BFMM[b]

        MA_noise_SCMinv_FMM = torch.linalg.inv(MA_noise_SCM_FMM)
        W_FM = torch.einsum("fij, fj -> fi", MA_noise_SCMinv_FMM, MA_speech_SCM_FMM[..., ref_channel]) / torch.sum(
            torch.diagonal(MA_noise_SCMinv_FMM @ MA_speech_SCM_FMM, dim1=-2, dim2=-1), dim=-1
        ).unsqueeze(-1)
        sep_BFT[b] = torch.einsum("itf, fi -> ft", mixture_BMTF[b], W_FM.conj())
    return sep_BFT


def MVDR_SV(mixture_BMTF, mask_speech_BTF, mask_noise_BTF, eps=1e-7, return_filter=False):
    if mixture_BMTF.ndim == 3:
        mixture_BMTF = mixture_BMTF[None]
        mask_speech_BTF = mask_speech_BTF[None]
        mask_noise_BTF = mask_noise_BTF[None]
    n_mic = mixture_BMTF.shape[1]
    speech_est_BMTF = mixture_BMTF * mask_speech_BTF.unsqueeze(1)
    speech_SCM_BFMM = (
        torch.einsum("bitf, bjtf -> bfij", speech_est_BMTF, speech_est_BMTF.conj())
        + eps * torch.eye(n_mic, device=mixture_BMTF.device)[None, None]
    )

    noise_est_BMTF = mixture_BMTF * mask_noise_BTF.unsqueeze(1)
    noise_SCMinv_BFMM = torch.linalg.inv(torch.einsum("bitf, bjtf -> bfij", noise_est_BMTF, noise_est_BMTF.conj()))

    # SV_BFM = speech_SCM_BFMM[..., ref_channel] / torch.linalg.norm(
    #     speech_SCM_BFMM[..., ref_channel], dim=-1, keepdim=True
    # )

    _, eig_vec_BFMM = torch.linalg.eigh(speech_SCM_BFMM)
    W_BFM = torch.einsum("bfij, bfj -> bfi", noise_SCMinv_BFMM, eig_vec_BFMM[..., -1])
    W_BFM = W_BFM / torch.einsum("bfm, bfm -> bf", eig_vec_BFMM[..., -1].conj(), W_BFM)[..., None]

    # W_BFM = torch.einsum("bfij, bfj -> bfi", noise_SCMinv_BFMM, speech_SCM_BFMM[..., ref_channel]) / torch.sum(
    #     torch.diagonal(noise_SCMinv_BFMM @ speech_SCM_BFMM, dim1=-2, dim2=-1), dim=-1
    # ).unsqueeze(-1)

    if return_filter:
        return torch.einsum("bitf, bfi -> bft", mixture_BMTF, W_BFM.conj()), W_BFM
    else:
        return torch.einsum("bitf, bfi -> bft", mixture_BMTF, W_BFM.conj())


def GEV(mixture_BMTF, mask_speech_BTF, mask_noise_BTF, eps=1e-7, return_filter=False):
    if mixture_BMTF.ndim == 3:
        mixture_BMTF = mixture_BMTF[None]
        mask_speech_BTF = mask_speech_BTF[None]
        mask_noise_BTF = mask_noise_BTF[None]
    n_mic = mixture_BMTF.shape[1]
    speech_est_BMTF = mixture_BMTF * mask_speech_BTF.unsqueeze(1)
    speech_SCM_BFMM = (
        torch.einsum("bitf, bjtf -> bfij", speech_est_BMTF, speech_est_BMTF.conj())
        + eps * torch.eye(n_mic, device=mixture_BMTF.device)[None, None]
    )

    noise_est_BMTF = mixture_BMTF * mask_noise_BTF.unsqueeze(1)
    noise_SCMinv_BFMM = torch.linalg.inv(
        torch.einsum("bitf, bjtf -> bfij", noise_est_BMTF, noise_est_BMTF.conj())
        + eps * torch.eye(n_mic, device=mixture_BMTF.device)[None, None]
    )

    tmp = torch.einsum("bfij, bfjk -> bfik", noise_SCMinv_BFMM, speech_SCM_BFMM)
    eig_val_BFM, eig_vec_BFMM = torch.linalg.eig(tmp)
    sorted_idx = torch.real(eig_val_BFM).argsort(axis=-1)
    eig_vec_BFMM = torch.take_along_dim(eig_vec_BFMM, sorted_idx[..., None, :], dim=-1)
    W_BFM = eig_vec_BFMM[:, :, :, -1]

    W_BFM[:, 1:] *= torch.exp(-1j * torch.angle(torch.einsum("bfi, bfi -> bf", W_BFM[:, 1:], W_BFM[:, :-1])))[
        :, :, None
    ]
    if return_filter:
        return torch.einsum("bitf, bfi -> bft", mixture_BMTF, W_BFM.conj()), W_BFM
    else:
        return torch.einsum("bitf, bfi -> bft", mixture_BMTF, W_BFM.conj())


def MUSIC(mixture_BMTF, mask_BT, SV_BFM, start_idx=0, end_idx=None, eps=1e-2):
    if mixture_BMTF.ndim == 3:
        mixture_BMTF = mixture_BMTF[None]
        mask_BT = mask_BT[None]
    n_mic = mixture_BMTF.shape[1]
    masked_mixture_BMTF = mixture_BMTF * mask_BT[:, None, :, None]
    SCM_BFMM = (
        torch.einsum("bitf, bjtf -> bfij", masked_mixture_BMTF, masked_mixture_BMTF.conj())
        + eps * torch.eye(n_mic, device=mixture_BMTF.device)[None, None]
    )

    _, eig_vec_BFMM = torch.linalg.eigh(SCM_BFMM)
    music_spec = torch.einsum("bfi, bfim -> bfm", SV_BFM.conj(), eig_vec_BFMM[..., :-1])
    if end_idx is None:
        music_spec = torch.abs(music_spec[:, start_idx:]).sum(dim=(1, 2))
    else:
        music_spec = torch.abs(music_spec[:, start_idx:end_idx]).sum(dim=(1, 2))

    return music_spec
