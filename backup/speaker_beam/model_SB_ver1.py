#! /usr/bin/env python3
# coding:utf-8

import torch
from torch import nn
import torch.nn.functional as F
import speechbrain as sb
from speechbrain.lobes.models.dual_path import SBRNNBlock, Dual_Computation_Block, SBTransformerBlock, select_norm, Dual_Computation_Block
import copy

from utility import segmentation, overlap_add, calculate_SCM
from model_utility import nn_segmentation, nn_over_add, nn_padding


class SpeakerNet(nn.Module):
    def __init__(
        self,
        input_length = 128,
        n_freq = 513,
        hidden_channel = 128,
        output_channel = 128,
        transformer_n_layer = 3,
        transformer_n_head = 8,
    ):
        super(SpeakerNet, self).__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=n_freq)
        self.conv1d_list = nn.Sequential(
            nn.Conv1d(
                in_channels=n_freq,
                out_channels=hidden_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode="replicate"
            ),
            nn.GroupNorm(num_groups=1, num_channels=hidden_channel),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=hidden_channel,
                out_channels=hidden_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode="replicate"
            ),
            nn.GroupNorm(num_groups=1, num_channels=hidden_channel),
            nn.LeakyReLU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(hidden_channel, transformer_n_head)
        self.transformer = nn.TransformerEncoder(encoder_layer, transformer_n_layer)

        self.conv1d_list_2 = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_channel,
                out_channels=output_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode="replicate"
            ),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=output_channel,
                out_channels=output_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode="replicate"
            ),
            nn.LeakyReLU(),
        )

        self.output_speaker_info = nn.Sequential(
            nn.Linear(output_channel * input_length // 16, output_channel * input_length // 64),
            nn.LeakyReLU(),
            nn.Linear(output_channel * input_length // 64, output_channel)
        )
        self.ouput_reliability = nn.Sequential(
            nn.Linear(output_channel * input_length // 16, output_channel * input_length // 64),
            nn.LeakyReLU(),
            nn.Linear(output_channel * input_length // 64, 1)
        )

    def forward(self, x):
        x = self.norm(x) # B x F x T
        x = self.conv1d_list(x) # B x F x T/4
        x = self.transformer(x.permute(0, 2, 1)) #  # B x T/4 x F
        x = self.conv1d_list_2(x.permute(0, 2, 1))
        speaker_info = self.output_speaker_info(x.flatten(start_dim=1)) # B x FT/4 -> B x D
        reliability = torch.sigmoid(self.ouput_reliability(x.flatten(start_dim=1)))
        return speaker_info, reliability.squeeze()


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
    num_layer_dual : int
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
        z_dim=128,
        hidden_channel=256,
        num_layer_dual=1,
        num_layer_rnn=2,
        norm_num_groups = 2,
        norm_type="ln",
        K=16,
        skip_around_intra=True,
        linear_layer_after_inter_intra=True,
    ):
        super(MaskNet, self).__init__()
        self.K = K
        self.num_layer_dual = num_layer_dual
        self.norm = nn.GroupNorm(num_groups=1, num_channels=n_freq)

        # intra_model = SBRNNBlock(n_freq+z_dim, hidden_channel, num_layer_rnn)
        # inter_model = SBRNNBlock(n_freq+z_dim, hidden_channel, num_layer_rnn)

        # self.dual_mdl = nn.ModuleList([])
        # for i in range(num_layer_dual):
        #     self.dual_mdl.append(
        #         copy.deepcopy(
        #             Dual_Computation_Block(
        #                 intra_model,
        #                 inter_model,
        #                 n_freq+z_dim,
        #                 norm_type,
        #                 skip_around_intra=skip_around_intra,
        #                 linear_layer_after_inter_intra=linear_layer_after_inter_intra,
        #             )
        #         )
        #     )

        # self.prelu = nn.PReLU()
        # self.conv1d = nn.Conv1d(n_freq+z_dim, n_freq, 1)

        # self.conv_list = nn.Sequential(
        #     nn.Conv1d(n_freq * 2, n_freq, kernel_size=3, padding=1),
        #     nn.PReLU(),
        #     nn.Conv1d(n_freq, n_freq, kernel_size=1),
        #     nn.Sigmoid()
        # )

        self.conv_list = nn.Sequential(
            nn.Conv1d(n_freq, n_freq, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(n_freq, hidden_channel, kernel_size=3, padding=1),
            nn.PReLU(),
        )

        intra_model = SBRNNBlock(hidden_channel*2+z_dim, hidden_channel, num_layer_rnn)
        inter_model = SBRNNBlock(hidden_channel*2+z_dim, hidden_channel, num_layer_rnn)

        self.dual_mdl = nn.ModuleList([])
        for i in range(num_layer_dual):
            self.dual_mdl.append(
                copy.deepcopy(
                    Dual_Computation_Block(
                        intra_model,
                        inter_model,
                        hidden_channel*2+z_dim,
                        norm_type,
                        skip_around_intra=skip_around_intra,
                        linear_layer_after_inter_intra=linear_layer_after_inter_intra,
                    )
                )
            )

        self.prelu = nn.PReLU()
        self.conv1d = nn.Conv1d(hidden_channel*2+z_dim, n_freq, 1)


    def forward(self, noisy, enhanced, speaker_vec):
        """Returns the output tensor.

        Arguments
        ---------
        noisy : torch.Tensor
        enhanced : torch.Tensor
            Input tensor of dimension [B, F, T].
        speaker_vec : torch.Tensor
            Input tensor of dimension [B, D].

        Returns
        -------
        out : torch.Tensor
            Output tensor of dimension [B, F, T]
               B = Batchsize,
               F = number of filters
               T = the number of time points
        """
        B, F, T = noisy.shape
        noisy = self.norm(noisy)
        enhanced = self.norm(enhanced)

        # # [B, F+D, T]
        # x = torch.cat([enhanced, speaker_vec.unsqueeze(-1).expand(-1, -1, T)], axis=1)

        # x, gap = nn_segmentation(x, self.K) # [B, F+D, K, S]

        # for i in range(self.num_layer_dual): # [B, F+D, K, S]
        #     x = self.dual_mdl[i](x)
        # x = self.prelu(x)
        # x = nn_over_add(x, gap)
        # x = self.conv1d(x) # [B, F+D, T] -> [B, F, T]

        # x = torch.cat([x, noisy], axis=1) # [B, 2F, T]
        # x = self.conv_list(x) # [B, 2F, T] -> [B, F, T]
        # [B, F+D, T]

        noisy = self.conv_list(noisy)
        enhanced = self.conv_list(enhanced)

        x = torch.cat([enhanced, noisy, speaker_vec.unsqueeze(-1).expand(-1, -1, T)], axis=1)

        x, gap = nn_segmentation(x, self.K) # [B, F+D, K, S]

        for i in range(self.num_layer_dual): # [B, F+D, K, S]
            x = self.dual_mdl[i](x)
        x = self.prelu(x)
        x = nn_over_add(x, gap)
        x = torch.sigmoid(self.conv1d(x)) # [B, F+D, T] -> [B, F, L]

        return x


class SB_ver1(nn.Module):
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
        input_length = 128,
        n_freq = 513,
        z_dim = 128,
        input_obs_enhanced = True,
    ):

        super(SB_ver1, self).__init__()

        # if input_obs_enhanced:
        #     input_dim = n_freq * 2
        # else:
        #     input_dim = n_freq
        self.input_length = input_length
        self.n_freq = n_freq
        self.z_dim = z_dim
        self.input_obs_enhanced = input_obs_enhanced
        self.model_name = "SB_ver1"

        self.speaker_net = SpeakerNet(
            input_length = input_length,
            n_freq = n_freq,
            hidden_channel = 128,
            output_channel = z_dim,
            transformer_n_layer = 3,
            transformer_n_head = 8,
        )
        self.mask_net = MaskNet(
            n_freq=n_freq,
            z_dim=z_dim,
            hidden_channel=256,
            num_layer_dual=1,
            num_layer_rnn=2,
            norm_num_groups = 2,
            norm_type="ln",
            K=16,
            skip_around_intra=True,
            linear_layer_after_inter_intra=True,
        )

        # reinitialize the parameters
        # for module in [self.encoder, self.masknet, self.decoder]:
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
        # speaker_label = B
        B, F, T = noisy_pwr.shape
        # speaker_vec = B x D, reliability = B
        clean_speaker_vec, _ = self.speaker_net(torch.log1p(clean_pwr))
        speaker_vec, reliability = self.speaker_net(torch.log1p(enhanced_pwr))

        speaker_vec_mean_list = []
        clean_speaker_vec_mean_list = []
        for i in set([int(label.item()) for label in speaker_label]):
            valid_indices = (speaker_label == i)
            n_valid = valid_indices.sum()
            valid_reliability_sum = reliability[valid_indices].sum()

            speaker_vec_mean_list.append(((reliability[valid_indices, None] * speaker_vec[valid_indices]).sum(axis=0) / valid_reliability_sum).unsqueeze(0).expand((n_valid, self.z_dim)))
            clean_speaker_vec_mean_list.append(((clean_speaker_vec[valid_indices]).sum(axis=0) / n_valid).unsqueeze(0).expand((n_valid, self.z_dim)))

        speaker_vec_mean = torch.cat(speaker_vec_mean_list, dim=0)
        clean_speaker_vec_mean = torch.cat(clean_speaker_vec_mean_list, dim=0)

        est_mask = self.mask_net(torch.log1p(noisy_pwr), torch.log1p(enhanced_pwr), speaker_vec_mean)
        clean_est_mask = self.mask_net(torch.log1p(noisy_pwr), torch.log1p(enhanced_pwr), clean_speaker_vec_mean)

        return est_mask, clean_est_mask, speaker_vec, speaker_vec_mean, clean_speaker_vec, clean_speaker_vec_mean, reliability


    def __str__(self):
        return f"{self.model_name}_Z={self.z_dim}_I={self.input_length}"


    def estimate_mask_segment(self, noisy_pwr_FT, enhanced_pwr_FT, speaker_vec_prev):
        speaker_vec, reliability = self.speaker_net(torch.log1p(enhanced_pwr_FT.unsqueeze(0)))
        speaker_vec = speaker_vec.squeeze()
        reliability = reliability.squeeze()

        if speaker_vec_prev is not None:
            speaker_vec = (speaker_vec_prev + speaker_vec * reliability) / (1 + reliability)
        mask = self.mask_net(
            torch.log1p(noisy_pwr_FT.unsqueeze(0)),
            torch.log1p(enhanced_pwr_FT.unsqueeze(0)),
            speaker_vec.unsqueeze(0)
        ).squeeze()
        return mask, speaker_vec


    def estimate_mask_segment_w_clean(
        self, noisy_pwr_FT, clean_pwr_FT, enhanced_pwr_FT, speaker_vec_prev
    ):
        speaker_vec, reliability = self.speaker_net(torch.log1p(clean_pwr_FT.unsqueeze(0)))
        speaker_vec = speaker_vec.squeeze()
        reliability = reliability.squeeze()

        if speaker_vec_prev is not None:
            speaker_vec = (speaker_vec_prev + speaker_vec * reliability) / (1 + reliability)

        mask = self.mask_net(
            torch.log1p(noisy_pwr_FT.unsqueeze(0)),
            torch.log1p(enhanced_pwr_FT.unsqueeze(0)),
            speaker_vec.unsqueeze(0)
        ).squeeze()
        return mask, speaker_vec


    def fine_tuning(
        self,
        noisy_BFTM,
        enhanced_pwr_BFT,
        speaker_vec_D,
        SV_FM,
        localization_loss=True,
        speaker_loss=True,
        flim=[32, 180],
    ):
        B, F, T, M = noisy_BFTM.shape
        noisy_pwr_BFT = torch.abs(noisy_BFTM[..., 0]) ** 2
        mask_BFT = self.mask_net(torch.log1p(noisy_pwr_BFT), torch.log1p(enhanced_pwr_BFT), speaker_vec_D.unsqueeze(0).expand(B, -1))
        enhanced_BFTM = noisy_BFTM * mask_BFT.unsqueeze(-1)

        loss = 0

        if localization_loss:
            SCM_BFMM = torch.einsum(
                "bfti, bftj -> bfij", enhanced_BFTM, enhanced_BFTM.conj()
            )
            eig_val_BFM, eig_vec_BFMM = torch.linalg.eigh(SCM_BFMM)
            loss +=  (torch.abs(torch.einsum(
                "fi, bfij -> bfj", 
                SV_FM[flim[0]:flim[1]].conj(), 
                eig_vec_BFMM[:, flim[0]:flim[1], :, :-1]
            )) ** 2 ).mean()

        if speaker_loss:
            enhanced_pwr_BFT = torch.abs(enhanced_BFTM[..., 0]) ** 2
            new_speaker_vec_BD, _ = self.speaker_net(torch.log1p(enhanced_pwr_BFT))
            loss += torch.nn.MSELoss()(
                new_speaker_vec_BD, speaker_vec_D.unsqueeze(0).expand(B, -1)
            )
        return loss



    # def calculate_speaker_vec_loss(self, speaker_vec, speaker_vec_mean, clean_speaker_vec, clean_speaker_vec_mean, loss_type="type1"):
    #     if loss_type == "type1":
    #         loss1 = sb.nnet.losses.mse_loss(speaker_vec, clean_speaker_vec)
    #         loss2 = sb.nnet.losses.mse_loss(speaker_vec_mean, clean_speaker_vec_mean)
    #         return loss1 + loss2
    #     elif loss_type == "type2":
    #         loss1 = sb.nnet.losses.mse_loss(speaker_vec, clean_speaker_vec)
    #         loss2 = sb.nnet.losses.mse_loss(speaker_vec_mean, clean_speaker_vec_mean)
    #         return loss1 + loss2
    #     elif loss_type == "type3":
    #         loss = sb.nnet.losses.mse_loss(speaker_vec_mean, clean_speaker_vec_mean)
    #         return loss


    # def calculate_reliability(self, reliability, SNR):
    #     loss_sep = sb.nnet.losses.mse_loss(predictions.permute(0, 2, 1), clean.permute(0, 2, 1), lens)


if __name__ == "__main__":
    # x = torch.rand(2, 513, 128)
    # speaker_net = SpeakerNet(input_length=128)
    # info, reliability = speaker_net.forward(x)

    B = 4
    x = torch.rand(B, 513, 128)
    y = torch.rand(B, 513, 128)
    label = torch.Tensor([1, 1, 2, 2])
    s = torch.rand(B, 64)
    # mask_net = MaskNet(n_freq=513, z_dim=64)
    net = SB_ver1()
    # est, speaker_vec, speaker_vec_mean, reliability = net.forward(x, y, label)
    # loss = net.calculate_speaker_vec_loss(speaker_vec, speaker_vec_mean)
    # print(est, est.shape, loss)

    from torchinfo import summary
    summary(
        net,
        # input_size=(100, 513, 128)
    )