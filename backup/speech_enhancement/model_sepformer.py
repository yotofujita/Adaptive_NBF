#! /usr/bin/env python3
# coding:utf-8

import torch
from torch import nn
import torch.nn.functional as F
import speechbrain as sb
from speechbrain.lobes.models.dual_path import Encoder, Decoder, SBTransformerBlock, select_norm, Dual_Computation_Block
import copy


class Dual_Path_Model(nn.Module):
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
    num_layers : int
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
        in_channels,
        out_channels,
        intra_model,
        inter_model,
        num_layers=1,
        norm="ln",
        K=200,
        num_spks=2,
        skip_around_intra=True,
        linear_layer_after_inter_intra=True,
        use_global_pos_enc=False,
        max_length=20000,
    ):
        super(Dual_Path_Model, self).__init__()
        self.model_name = "SepFormer"
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = select_norm(norm, in_channels, 3)
        # self.conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.use_global_pos_enc = use_global_pos_enc

        if self.use_global_pos_enc:
            self.pos_enc = PositionalEncoding(max_length)

        self.dual_mdl = nn.ModuleList([])
        for i in range(num_layers):
            self.dual_mdl.append(
                copy.deepcopy(
                    Dual_Computation_Block(
                        intra_model,
                        inter_model,
                        in_channels,
                        norm,
                        skip_around_intra=skip_around_intra,
                        linear_layer_after_inter_intra=linear_layer_after_inter_intra,
                    )
                )
            )

        self.conv2d = nn.Conv2d(
            in_channels, out_channels * num_spks, kernel_size=1
        )
        self.end_conv1x1 = nn.Conv1d(out_channels, out_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        # gated output layer
        self.output = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid()
        )

    def forward(self, x):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, L].

        Returns
        -------
        out : torch.Tensor
            Output tensor of dimension [spks, B, N, L]
            where, spks = Number of speakers
               B = Batchsize,
               N = number of filters
               L = the number of time points
        """

        # before each line we indicate the shape after executing the line

        # [B, N, L]
        x = self.norm(x)

        # [B, N, L]
        # x = self.conv1d(x)
        if self.use_global_pos_enc:
            x = self.pos_enc(x.transpose(1, -1)).transpose(1, -1) + x * (
                x.size(1) ** 0.5
            )

        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)

        # [B, N, K, S]
        for i in range(self.num_layers):
            x = self.dual_mdl[i](x)
        x = self.prelu(x)

        # [B, N*spks, K, S]
        x = self.conv2d(x)
        B, _, K, S = x.shape

        # [B*spks, N, K, S]
        x = x.view(B * self.num_spks, -1, K, S)

        # [B*spks, N, L]
        x = self._over_add(x, gap)
        x = self.output(x) * self.output_gate(x)

        # [B*spks, N, L]
        x = self.end_conv1x1(x)

        # [B, spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)

        # [spks, B, N, L]
        x = x.transpose(0, 1)

        return x

    def _padding(self, input, K):
        """Padding the audio times.

        Arguments
        ---------
        K : int
            Chunks of length.
        P : int
            Hop size.
        input : torch.Tensor
            Tensor of size [B, N, L].
            where, B = Batchsize,
                   N = number of filters
                   L = time points
        """
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        """The segmentation stage splits

        Arguments
        ---------
        K : int
            Length of the chunks.
        input : torch.Tensor
            Tensor with dim [B, N, L].

        Return
        -------
        output : torch.tensor
            Tensor with dim [B, N, K, S].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
               L = the number of time points
        """
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = (
            torch.cat([input1, input2], dim=3).view(B, N, -1, K).transpose(2, 3)
        )

        return input.contiguous(), gap

    def _over_add(self, input, gap):
        """Merge the sequence with the overlap-and-add method.

        Arguments
        ---------
        input : torch.tensor
            Tensor with dim [B, N, K, S].
        gap : int
            Padding length.

        Return
        -------
        output : torch.tensor
            Tensor with dim [B, N, L].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
               L = the number of time points

        """
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input


# old
    # def __init__(
    #     self,
    #     in_channels,
    #     out_channels,
    #     intra_model_1,
    #     inter_model_1,
    #     intra_model_2,
    #     inter_model_2,
    #     num_layers=1,
    #     norm="ln",
    #     K=200,
    #     num_spks=2,
    #     skip_around_intra=True,
    #     linear_layer_after_inter_intra=True,
    #     use_global_pos_enc=False,
    #     max_length=20000,
    # ):
    #     super(Dual_Path_Model, self).__init__()
    #     self.K = K
    #     self.num_spks = num_spks
    #     self.num_layers = num_layers
    #     self.norm = select_norm(norm, in_channels, 3)
    #     self.use_global_pos_enc = use_global_pos_enc

    #     if self.use_global_pos_enc:
    #         self.pos_enc = PositionalEncoding(max_length)

    #     self.dual_mdl = nn.ModuleList([])

    #     self.dual_mdl.append(
    #         copy.deepcopy(
    #             Dual_Computation_Block(
    #                 intra_model_1,
    #                 inter_model_1,
    #                 out_channels,
    #                 norm,
    #                 skip_around_intra=skip_around_intra,
    #                 linear_layer_after_inter_intra=linear_layer_after_inter_intra,
    #             )
    #         )
    #     )

    #     # self.conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    #     self.dual_mdl.append(
    #         copy.deepcopy(
    #             Dual_Computation_Block(
    #                 intra_model_2,
    #                 inter_model_2,
    #                 out_channels,
    #                 norm,
    #                 skip_around_intra=skip_around_intra,
    #                 linear_layer_after_inter_intra=linear_layer_after_inter_intra,
    #             )
    #         )
    #     )

    #     self.conv2d_1 = nn.Conv2d(
    #         in_channels, out_channels, kernel_size=1
    #     )

    #     self.conv2d_2 = nn.Conv2d(
    #         out_channels, out_channels * num_spks, kernel_size=1
    #     )
    #     self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1, bias=False)
    #     self.prelu = nn.PReLU()
    #     self.activation = nn.ReLU()
    #     # gated output layer
    #     self.output = nn.Sequential(
    #         nn.Conv1d(out_channels, out_channels, 1), nn.Tanh()
    #     )
    #     self.output_gate = nn.Sequential(
    #         nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid()
    #     )

    # def forward(self, x):
    #     """Returns the output tensor.

    #     Arguments
    #     ---------
    #     x : torch.Tensor
    #         Input tensor of dimension [B, N, L].

    #     Returns
    #     -------
    #     out : torch.Tensor
    #         Output tensor of dimension [spks, B, N, L]
    #         where, spks = Number of speakers
    #            B = Batchsize,
    #            N = number of filters
    #            L = the number of time points
    #     """

    #     # before each line we indicate the shape after executing the line

    #     # [B, N, L]
    #     x = self.norm(x)

    #     # [B, N, L]
    #     # x = self.conv1d(x)
    #     if self.use_global_pos_enc:
    #         print(x.shape, x.transpose(1, -1).shape, x.transpose(1, -1).transpose(1, -1).shape)
    #         x = self.pos_enc(x.transpose(1, -1)).transpose(1, -1) + x * (
    #             x.size(1) ** 0.5
    #         )

    #     # [B, N, K, S]
    #     x, gap = self._Segmentation(x, self.K)

    #     # [B, N, K, S]
    #     # for i in range(self.num_layers):
    #     x = self.dual_mdl[0](x)
    #     print("before conv2d_1 : ", x.shape, "\n")
    #     x = self.conv2d_1(x)
    #     print("after conv2d_1 : ", x.shape, "\n")
    #     x = self.dual_mdl[1](x)
    #     x = self.prelu(x)

    #     # [B, N*spks, K, S]
    #     x = self.conv2d_2(x)
    #     B, _, K, S = x.shape

    #     # [B*spks, N, K, S]
    #     x = x.view(B * self.num_spks, -1, K, S)

    #     # [B*spks, N, L]
    #     x = self._over_add(x, gap)
    #     x = self.output(x) * self.output_gate(x)

    #     # [B*spks, N, L]
    #     x = self.end_conv1x1(x)

    #     # [B, spks, N, L]
    #     _, N, L = x.shape
    #     x = x.view(B, self.num_spks, N, L)
    #     x = self.activation(x)

    #     # [spks, B, N, L]
    #     x = x.transpose(0, 1)

    #     return x


class Sepformer(nn.Module):
    """The wrapper for the sepformer model which combines the Encoder, Masknet and the decoder
    https://arxiv.org/abs/2010.13154

    Arguments
    ---------

    encoder_kernel_size: int,
        The kernel size used in the encoder
    encoder_in_nchannels: int,
        The number of channels of the input audio
    encoder_out_nchannels: int,
        The number of filters used in the encoder.
        Also, number of channels that would be inputted to the intra and inter blocks.
    masknet_chunksize: int,
        The chunk length that is to be processed by the intra blocks
    masknet_numlayers: int,
        The number of layers of combination of inter and intra blocks
    masknet_norm: str,
        The normalization type to be used in the masknet
        Should be one of 'ln' -- layernorm, 'gln' -- globallayernorm
                         'cln' -- cumulative layernorm, 'bn' -- batchnorm
                         -- see the select_norm function above for more details
    masknet_useextralinearlayer: bool,
        Whether or not to use a linear layer at the output of intra and inter blocks
    masknet_extraskipconnection: bool,
        This introduces extra skip connections around the intra block
    masknet_numspks: int,
        This determines the number of speakers to estimate
    intra_numlayers: int,
        This determines the number of layers in the intra block
    inter_numlayers: int,
        This determines the number of layers in the inter block
    intra_nhead: int,
        This determines the number of parallel attention heads in the intra block
    inter_nhead: int,
        This determines the number of parallel attention heads in the inter block
    intra_dffn: int,
        The number of dimensions in the positional feedforward model in the inter block
    inter_dffn: int,
        The number of dimensions in the positional feedforward model in the intra block
    intra_use_positional: bool,
        Whether or not to use positional encodings in the intra block
    inter_use_positional: bool,
        Whether or not to use positional encodings in the inter block
    intra_norm_before: bool
        Whether or not we use normalization before the transformations in the intra block
    inter_norm_before: bool
        Whether or not we use normalization before the transformations in the inter block

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
        # encoder_kernel_size=16,
        # encoder_in_nchannels=1,
        encoder_out_nchannels=1026,
        masknet_chunksize=250,
        masknet_numlayers=2,
        masknet_norm="ln",
        masknet_useextralinearlayer=False,
        masknet_extraskipconnection=True,
        masknet_numspks=1,
        intra_numlayers=8,
        inter_numlayers=8,
        intra_nhead=9,
        inter_nhead=9,
        intra_dffn=1024,
        inter_dffn=1024,
        intra_use_positional=True,
        inter_use_positional=True,
        intra_norm_before=True,
        inter_norm_before=True,
    ):

        super(Sepformer, self).__init__()

        intra_model = SBTransformerBlock(
            num_layers=intra_numlayers,
            d_model=encoder_out_nchannels,
            nhead=intra_nhead,
            d_ffn=intra_dffn,
            use_positional_encoding=intra_use_positional,
            norm_before=intra_norm_before,
        )

        inter_model = SBTransformerBlock(
            num_layers=inter_numlayers,
            d_model=encoder_out_nchannels,
            nhead=inter_nhead,
            d_ffn=inter_dffn,
            use_positional_encoding=inter_use_positional,
            norm_before=inter_norm_before,
        )

        self.masknet = Dual_Path_Model(
            in_channels=encoder_out_nchannels,
            out_channels=encoder_out_nchannels//2,
            # out_channels=encoder_out_nchannels,
            intra_model=intra_model,
            inter_model=inter_model,
            num_layers=masknet_numlayers,
            norm=masknet_norm,
            K=masknet_chunksize,
            num_spks=masknet_numspks,
            skip_around_intra=masknet_extraskipconnection,
            linear_layer_after_inter_intra=masknet_useextralinearlayer,
        )
        # self.encoder = Encoder(
        #     kernel_size=encoder_kernel_size,
        #     out_channels=encoder_out_nchannels,
        #     in_channels=encoder_in_nchannels,
        # )
        # self.decoder = Decoder(
        #     in_channels=encoder_out_nchannels,
        #     out_channels=encoder_in_nchannels,
        #     kernel_size=encoder_kernel_size,
        #     stride=encoder_kernel_size // 2,
        #     bias=False,
        # )
        self.num_spks = masknet_numspks

        # reinitialize the parameters
        # for module in [self.encoder, self.masknet, self.decoder]:
        for module in [self.masknet]:
            self.reset_layer_recursively(module)

    def reset_layer_recursively(self, layer):
        """Reinitializes the parameters of the network"""
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)


    # def forward(self, y):
    #     noisy_pwr_spec = y[0]
    #     enhanced_pwr_spec = y[1]
    def forward(self, noisy_pwr_spec, enhanced_pwr_spec):
        x = torch.log1p(torch.cat([noisy_pwr_spec, enhanced_pwr_spec], dim=1)) # B x 2F x T
        return self.masknet(x).squeeze(0) * noisy_pwr_spec


    def estimate_mask(self, noisy_pwr_spec, enhanced_pwr_spec):
        x = torch.log1p(torch.cat([noisy_pwr_spec, enhanced_pwr_spec], dim=1))
        return self.masknet(x).squeeze(0)


if __name__ == "__main__":
    mix = torch.rand(1, 256, 20)
    model = Sepformer()
    model.forward(mix)