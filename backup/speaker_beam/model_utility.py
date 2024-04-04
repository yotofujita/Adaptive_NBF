import torch


def nn_padding(input, K):
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
    # gap = K - (P + L % K) % K
    gap = L % P
    if gap > 0:
        pad = torch.zeros(B, N, gap, device=input.device, dtype=input.dtype)
        input = torch.cat([input, pad], dim=2)

    _pad = torch.zeros(B, N, P, device=input.device, dtype=input.dtype)
    input = torch.cat([_pad, input, _pad], dim=2)

    return input, gap


def nn_segmentation(input, K):
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
    input, gap = nn_padding(input, K)
    # [B, N, K, S]
    input1 = input[:, :, :-P].contiguous().view(B, N, -1, P)
    input2 = input[:, :, P:].contiguous().view(B, N, -1, P)
    input = (
        torch.cat([input1, input2], dim=3).transpose(2, 3)
    )

    return input.contiguous(), gap


def nn_over_add(input, gap):
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

    input1 = input[:, :, :P, 1:].permute(0, 1, 3, 2).contiguous().view(B, N, -1)
    input2 = input[:, :, P:, :-1].permute(0, 1, 3, 2).contiguous().view(B, N, -1)
    input = (input1 + input2) / 2

    if gap > 0:
        input = input[:, :, :-gap]

    return input
