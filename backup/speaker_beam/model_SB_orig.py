#! /usr/bin/env python3
# coding:utf-8

import torch
from torch import nn
import torch.nn.functional as F
import speechbrain as sb
from speechbrain.lobes.models.dual_path import SBRNNBlock, Dual_Computation_Block, SBTransformerBlock, select_norm, Dual_Computation_Block
import copy

from utils.utility import segmentation, overlap_add, calculate_SCM
from model_utility import nn_segmentation, nn_over_add, nn_padding

class SpeakerNet(nn.Module):
    def __init__(
        self,
        n_freq = 513,
        hidden_channel = 128,
        output_channel = 30,
        n_layer_tf = 3,
        n_head_tf = 8,
    ):
        super(SpeakerNet, self).__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=n_freq)
        self.conv1d_1 = nn.Conv1d(
                in_channels=n_freq,
                out_channels=hidden_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="replicate"
            )
        self.norm_1 = nn.GroupNorm(num_groups=1, num_channels=hidden_channel)
        self.prelu_1 = nn.LeakyReLU()

        encoder_layer = nn.TransformerEncoderLayer(hidden_channel, n_head_tf)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layer_tf)

        self.speaker_emb_transform = nn.Sequential(
            nn.Linear(output_channel, (output_channel+hidden_channel)//2),
            nn.Softplus(),
            nn.Linear((output_channel+hidden_channel)//2, hidden_channel),
            nn.Softplus(),
        )

        self.linear_output = nn.Linear(hidden_channel, output_channel)


    def forward(self, x, spk_emb_prev):
        x = self.norm(x) # B x F x T
        x = self.prelu_1(self.norm_1(self.conv1d_1(x))) # B x H x T
        x = self.transformer(x.permute(0, 2, 1)) #  # B x T x H

        if spk_emb_prev is not None:
            y = self.speaker_emb_transform(spk_emb_prev) # B x H
            weight_y = torch.einsum("bth, bh -> bt", x, y)
            spk_emb = torch.einsum("bth, bt -> bh", x, weight_y) / weight_y.sum(axis=1, keepdims=True)
            spk_emb = self.linear_output(spk_emb)

            z = self.speaker_emb_transform(spk_emb)
            weight_z = torch.einsum("bth, bh -> bt", x, z)

            return (weight_z.mean(axis=1, keepdims=True) * spk_emb + 
                weight_y.mean(axis=1, keepdims=True) * spk_emb_prev) \
                / (weight_z.mean(axis=1, keepdims=True) + 
                    weight_y.mean(axis=1, keepdims=True))
        else:
            return self.linear_output(torch.mean(x, axis=1))


class MaskNet(nn.Module):
    """The dual path model which is the basis for dualpathrnn, sepformer, dptnet.

    Arguments
    ---------
    in_channels : int
        Number of channels at the output of the encoder.
    out_channels : int
        Number of channels that would be inputted to the intra and inter blocks.
    intra_model : torch.nn.module
        Model to process within the chunks.
    inter_model : torch.nn.module
        model to process across the chunks,
    n_layer_dual : int
        Number of layers of Dual Computation Block.
    norm : str
        Normalization type.
    K : int
        Chunk length.
    num_spks : int
        Number of sources (speakers).
    skip_around_intra : bool
        Skip connection around intra.
    linear_layer_after_inter_intra : bool
        Linear layer after inter and intra.
    use_global_pos_enc : bool
        Global positional encodings.
    max_length : int
        Maximum sequence length.

    Example
    ---------
    >>> intra_block = SBTransformerBlock(1, 64, 8)
    >>> inter_block = SBTransformerBlock(1, 64, 8)
    >>> dual_path_model = Dual_Path_Model(64, 64, intra_block, inter_block, num_spks=2)
    >>> x = torch.randn(10, 64, 2000)
    >>> x = dual_path_model(x)
    >>> x.shape
    torch.Size([2, 10, 64, 2000])
    """

    def __init__(
        self,
        n_freq=513,
        z_dim=30,
        hidden_channel=256,
        n_layer_dual=2,
        n_layer_rnn_tf=2,
        norm_type="ln",
        K=16,
        skip_around_intra=True,
        linear_layer_after_inter_intra=True,
        RNN_or_TF="TF"
    ):
        super(MaskNet, self).__init__()
        self.K = K
        self.n_layer_dual = n_layer_dual
        self.n_layer_rnn_tf = n_layer_rnn_tf
        self.z_dim = z_dim
        self.RNN_or_TF = RNN_or_TF
        self.norm = nn.GroupNorm(num_groups=1, num_channels=n_freq)

        if RNN_or_TF == "RNN":
            intra_model = SBRNNBlock(n_freq, hidden_channel, n_layer_rnn_tf)
            inter_model = SBRNNBlock(n_freq, hidden_channel, n_layer_rnn_tf)
        else:
            intra_model = SBTransformerBlock(
                num_layers=n_layer_rnn_tf,
                d_model=n_freq,
                nhead=3,
                d_ffn=hidden_channel, # 途中のLinearの出力次元
                use_positional_encoding=False,
                norm_before=False,
            )

            inter_model = SBTransformerBlock(
                num_layers=n_layer_rnn_tf,
                d_model=n_freq,
                nhead=3,
                d_ffn=hidden_channel,
                use_positional_encoding=False,
                norm_before=False,
            )

        self.dual_mdl = nn.ModuleList([])
        for i in range(self.n_layer_dual):
            self.dual_mdl.append(
                copy.deepcopy(
                    Dual_Computation_Block(
                        intra_model,
                        inter_model,
                        n_freq,
                        norm_type,
                        skip_around_intra=skip_around_intra,
                        linear_layer_after_inter_intra=linear_layer_after_inter_intra,
                    )
                )
            )
        self.prelu = nn.PReLU()

        self.linear_mdls = nn.ModuleList([])
        for i in range(self.z_dim):
            self.linear_mdls.append(
                nn.Sequential(
                    nn.Linear(n_freq, n_freq),
                    nn.PReLU(),
                )
            )

        self.conv_output = nn.Sequential(
            nn.Conv1d(n_freq, n_freq, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(n_freq, n_freq, kernel_size=1),
            nn.PReLU(),
            nn.Conv1d(n_freq, n_freq, kernel_size=1),
        )


    def forward(self, enhanced, spk_emb):
        """Returns the output tensor.

        Arguments
        ---------
        enhanced : torch.Tensor
            Input tensor of dimension [B, F, T].
        spk_emb : torch.Tensor
            Input tensor of dimension [B, D].

        Returns
        -------
        out : torch.Tensor
            Output tensor of dimension [B, F, T]
               B = Batchsize,
               F = number of filters
               T = the number of time points
        """
        B, F, T = enhanced.shape
        enhanced = self.norm(enhanced)

        x, gap = nn_segmentation(enhanced, self.K) # [B, F, K, S]

        for i in range(self.n_layer_dual): # [B, F, K, S]
            x = self.dual_mdl[i](x)
        x = self.prelu(x)
        x = nn_over_add(x, gap).permute(0, 2, 1).contiguous()

        x = torch.sum(
            torch.stack(
                [self.linear_mdls[i](x) * spk_emb[:, i, None, None] for i in range(self.z_dim)] 
            ), # [C, B, T, F]
            axis=0
        ).permute(0, 2, 1)

        x = torch.sigmoid(self.conv_output(x)) # [B, F, T]

        return x


class SB_orig(nn.Module):
    """

    Arguments
    ---------

    input_length: int,
        The number of channels of the input audio

    Example
    -----
    >>> model = Sepformer()
    >>> inp = torch.rand(1, 160)
    >>> result = model.forward(inp)
    >>> result.shape
    torch.Size([1, 160, 2])
    """

    def __init__(
        self,
        n_freq = 513,
        z_dim = 30,
        hidden_channel = 256,
        RNN_or_TF = "TF",
        n_layer_dual=2,
        n_layer_rnn_tf=2,
    ):

        super(SB_orig, self).__init__()

        self.n_freq = n_freq
        self.z_dim = z_dim
        self.model_name = "SB_orig"

        self.speaker_net = SpeakerNet(
            n_freq = n_freq,
            hidden_channel = hidden_channel,
            output_channel = z_dim,
            n_layer_tf = 3,
            n_head_tf = 4,
        )
        self.mask_net = MaskNet(
            n_freq=n_freq,
            z_dim=z_dim,
            hidden_channel=hidden_channel,
            n_layer_dual=n_layer_dual,
            n_layer_rnn_tf=n_layer_rnn_tf,
            RNN_or_TF=RNN_or_TF,
            norm_type="ln",
            K=16,
            skip_around_intra=True,
            linear_layer_after_inter_intra=True,
        )

        # reinitialize the parameters
        for module in [self.mask_net, self.speaker_net]:
            self.reset_layer_recursively(module)


    def reset_layer_recursively(self, layer):
        """Reinitializes the parameters of the network"""
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)


    def forward(self, noisy_pwr, enhanced_pwr, clean_pwr, speaker_label):
        B, F, T = noisy_pwr.shape
        clean_spk_emb = self.speaker_net(torch.log1p(clean_pwr), None)

        if B > 1:
            clean_spk_emb_list = []
            start_idx = 0
            for i in range(B):
                if (i == B-1) or speaker_label[i] != speaker_label[i+1]:
                    clean_spk_emb_list.append(clean_spk_emb[start_idx])
                    start_idx = i+1
                else:
                    clean_spk_emb_list.append(clean_spk_emb[i+1])
            clean_spk_emb = torch.stack(clean_spk_emb_list, axis=0)

        spk_emb = self.speaker_net(torch.log1p(enhanced_pwr), clean_spk_emb)

        est_mask = self.mask_net(torch.log1p(enhanced_pwr), spk_emb)
        clean_est_mask = self.mask_net(torch.log1p(enhanced_pwr), clean_spk_emb)

        return est_mask, clean_est_mask, spk_emb, clean_spk_emb


    def __str__(self):
        return f"{self.model_name}_{self.RNN_or_TF}_Ld={self.n_layer_dual}"\
            f"_Lr={self.n_layer_rnn_tf}_Z={self.z_dim}_I={self.input_length}_H={self.hidden_channel}"


    def estimate_mask_segment(self, noisy_pwr_FT, enhanced_pwr_FT, spk_emb_prev):
        spk_emb = self.speaker_net(torch.log1p(enhanced_pwr_FT.unsqueeze(0)), spk_emb_prev)

        mask = self.mask_net(
            torch.log1p(enhanced_pwr_FT.unsqueeze(0)),
            spk_emb
        ).squeeze()
        return mask, spk_emb



    # def fine_tuning(
    #     self,
    #     noisy_BFTM,
    #     enhanced_pwr_BFT,
    #     spk_emb_D,
    #     SV_FM,
    #     localization_loss=True,
    #     speaker_loss=True,
    #     flim=[32, 180],
    # ):
    #     B, F, T, M = noisy_BFTM.shape
    #     noisy_pwr_BFT = torch.abs(noisy_BFTM[..., 0]) ** 2
    #     mask_BFT = self.mask_net(torch.log1p(noisy_pwr_BFT), torch.log1p(enhanced_pwr_BFT), spk_emb_D.unsqueeze(0).expand(B, -1))
    #     enhanced_BFTM = noisy_BFTM * mask_BFT.unsqueeze(-1)

    #     loss = 0

    #     if localization_loss:
    #         SCM_BFMM = torch.einsum(
    #             "bfti, bftj -> bfij", enhanced_BFTM, enhanced_BFTM.conj()
    #         )
    #         eig_val_BFM, eig_vec_BFMM = torch.linalg.eigh(SCM_BFMM)
    #         loss +=  (torch.abs(torch.einsum(
    #             "fi, bfij -> bfj", 
    #             SV_FM[flim[0]:flim[1]].conj(), 
    #             eig_vec_BFMM[:, flim[0]:flim[1], :, :-1]
    #         )) ** 2 ).mean()

    #     if speaker_loss:
    #         enhanced_pwr_BFT = torch.abs(enhanced_BFTM[..., 0]) ** 2
    #         new_spk_emb_BD, _ = self.speaker_net(torch.log1p(enhanced_pwr_BFT))
    #         loss += torch.nn.MSELoss()(
    #             new_spk_emb_BD, spk_emb_D.unsqueeze(0).expand(B, -1)
    #         )
    #     return loss



    # def calculate_spk_emb_loss(self, spk_emb, spk_emb_mean, clean_spk_emb, clean_spk_emb_mean, loss_type="type1"):
    #     if loss_type == "type1":
    #         loss1 = sb.nnet.losses.mse_loss(spk_emb, clean_spk_emb)
    #         loss2 = sb.nnet.losses.mse_loss(spk_emb_mean, clean_spk_emb_mean)
    #         return loss1 + loss2
    #     elif loss_type == "type2":
    #         loss1 = sb.nnet.losses.mse_loss(spk_emb, clean_spk_emb)
    #         loss2 = sb.nnet.losses.mse_loss(spk_emb_mean, clean_spk_emb_mean)
    #         return loss1 + loss2
    #     elif loss_type == "type3":
    #         loss = sb.nnet.losses.mse_loss(spk_emb_mean, clean_spk_emb_mean)
    #         return loss


    # def calculate_reliability(self, reliability, SNR):
    #     loss_sep = sb.nnet.losses.mse_loss(predictions.permute(0, 2, 1), clean.permute(0, 2, 1), lens)


if __name__ == "__main__":
    # x = torch.rand(2, 513, 128)
    # speaker_net = SpeakerNet(input_length=128)
    # emb = speaker_net.forward(x, None)
    # emb = speaker_net.forward(x, emb)

    B = 4
    z_dim = 30
    x = torch.rand(B, 513, 128)
    y = torch.rand(B, 513, 128)
    label = torch.Tensor([1, 1, 2, 2])
    s = torch.rand(B, z_dim)

    # mask_net = MaskNet(n_freq=513, z_dim=z_dim, RNN_or_TF="TF")
    # res = mask_net(x, s)

    net = SB_orig()
    est_pwr, clean_est_pwr, spk_emb, clean_spk_emb = net.forward(x, y, y, label)
    # loss = net.calculate_spk_emb_loss(spk_emb, spk_emb_mean)
    # print(est, est.shape, loss)

    from torchinfo import summary
    # summary(
    #     mask_net,
    # )
    summary(
        net,
    )
