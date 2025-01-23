import numpy as np

import torch
from torch.nn.functional import silu

"""
This code is mostly adapted from the paper

    T. Karras, M. Aittala, T. Aila, S. Laine. Elucidating the design space of diffusion-based generative models. NeurIPS 2022 (2022), <https://arxiv.org/abs/2206.00364>

whose source code

    https://github.com/NVlabs/edm

is copyrighted by the NVIDIA CORPORATION & AFFILIATES and published under the
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
licence.
"""

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

class Linear(torch.nn.Module):
    """A linear layer."""
    
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

class GroupNorm(torch.nn.Module):
    """Group normalization."""

    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

class PositionalEmbedding(torch.nn.Module):
    """Positional embedding for noise scales based on sinusoids."""

    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

class Conv2d(torch.nn.Module):
    """A 2D convolution with channelwise bias with an optional downscaling or
    upscaling in front for use in an UNet."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        bias=True,
        up=False,
        down=False,
        resample_filter=[1,1],
        init_mode='kaiming_normal',
        init_weight=1,
        init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        
        # Create the kernels/weights for the convolution. The weights have
        # shape (out_channels, in_channels, kernel, kernel). The bias has shape
        # (out_channels), i.e., one bias is added per channel. The weights are
        # initialized according to the selected initialization scheme
        # implemented in the function weight_init.
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None

        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        # w_pad is the kernel size / 2
        w_pad = w.shape[-1] // 2 if w is not None else 0
        # f_pad is 0 with the default resampling filter
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0
        
        # Here we do the convolution in two steps. If up or down is True, then
        # x is upscaled or downscaled but the number of channels is untouched.
        # The number of channels is then modified by another convolution that
        # does not change input size.
        #
        # In downscaling, a 2x2 kernel based on the resample_filter tensor is
        # applied over the image with stride 2 and suitable padding so that the
        # input size is halved. The default resampling filter merely computes
        # the average of the numbers in the sliding window. The implementation
        # silently assumes that the input size dimensions are divisible by 2;
        # otherwise the leftover part is discarded (e.g., 65x65 -> 32x32).
        #
        # In upscaling, the same 2x2 kernel based on resample_filter is used,
        # but it is multiplied by 4 to cancel out the average. The default
        # resampling filter is the identity filter, so essentially the upscaled
        # tensor gets padded with zeros and the zeros get filled with copies of
        # the one an only actual value in the convolution window. Here it does
        # not matter if the input size dimensions are multiples of 2 (e.g.
        # 65x65 -> 130x130).
        #
        # Notice that the resampling filter parameters are not trainable.

        if self.up:
            # Upsample, but do not change channels.
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
        if self.down:
            # Downsample, but do not change channels. See above.
            x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)

        if w is not None:
            # Uses the actual convolution determined by the weights with shape (out_channels,
            # in_channels, kernel, kernel) to produce a tensor with channels
            # determined by out_channels. Default stride is 1.
            x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))

        return x

class UNetBlock(torch.nn.Module):
    """A UNet block to be used in an UNet encoding or decoding layer. The first
    part of the block passes the input through an optionally downscaling or
    upscaling Conv2d black with positional embedding added via adaptive scaling
    in the middle. There is a residual connection around this first part. The
    second part performs self-attention."""

    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        up=False,
        down=False,
        attention=False,
        num_heads=None,
        channels_per_head=64,
        dropout=0,
        skip_scale=1,
        eps=1e-5,
        resample_filter=[1,1],
        resample_proj=False,
        adaptive_scale=True,
        init=dict(),
        init_zero=dict(init_weight=0),
        init_attn=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        # First convolution block.
        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        # For combining the input with the positional embedding.
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            # If self-attention is enabled with self.num_heads attention heads,
            # define a convolution that is used to find the query, key, and
            # value matrices based on the input.
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)
    
    def forward(self, x, emb):
        # x shape (b, in_channels, k_0, k_0)
        # embedding shape (b, emb_channels)
        orig = x
        # Apply a group normalization, SiLU, and a convolution. The convolution
        # applies a downscaling or an upscaling if the class arguments up or
        # down are true. Otherwise the size is uchanged.
        x = self.conv0(silu(self.norm0(x)))
        # x shape (b, out_channels, k_1, k_1) where k_1 = k_0/2 if downscaled,
        # k_1 = 2k_0 if upscaled, and k_1 = k_0 otherwise.

        # Combine x with the positional embedding.
        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        # params shape (b, out_channels*a) where a = 2 if adaptive scale is
        # used and otherwise a = 1.
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            # After group normalization, scale and shift x according to learned
            # values scale and shift. Then apply SiLU.
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            # Apply a group normalization and SiLU to x + params without
            # scaling and shifting.
            x = silu(self.norm1(x.add_(params)))
        # x shape (b, out_channels, k_1, k_1)

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            # We perform multi-head self-attention. Here each head attends to a
            # channelwise projection, so the projection is not done to the
            # (flattened) image dimension as would be the direct analogue of
            # how it is done in processing word sequences. Presumably this is
            # because the UNet itself already projects the images to smaller
            # dimensions, so we want to do something different here.
            #
            # Each q, k, v will have shape (bh, c/h, k^2) where b is batch, c
            # is channels, k is image dimension, and h is the number of heads.
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            # Next, we compute the term softmax(Q^T K / sqrt(d)) from the "Attention is
            # all you need" paper. This amounts to computing dot product along
            # the dimension that is projected (here: channels) and dividing by
            # the square root of the projected dimension (here: c/h or
            # k.shape[1]). This is what the next line does. The resulting
            # tensor has shape (bh, k^2, k^2).
            w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
            # Then we compute the weighted sum of the value matrix elements
            # (k^2 vectors of length k^2) with the obtained weights and reorder
            # to shape (bh, c/h, k^2).
            a = torch.einsum('nqk,nck->ncq', w, v)
            # We then reshape to the original shape (b, c, k, k), apply the
            # output linear transformation, and add this self-attention vector
            # to x. x is then multiplied by skip_scale.
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x

class DhariwalUNet(torch.nn.Module):
    """An UNet architecture that consists of a fixed number of encoding and
    decoding layers. The encode layers downscale the image and pass a residual
    connection to a decoding block on the same level. An encode block
    downscales the input via a UNetBlock and passes it through a specified
    number of UNetBlocks that preserve the input size. The decoding part is
    analogous but has one more UNetBlock to accomodate for all residual
    connections."""

    def __init__(
        self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        model_channels      = 192,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [32,16,8],    # List of resolutions with self-attention.
        dropout             = 0.10,         # List of resolutions with self-attention.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.
    ):
        super().__init__()
        self.label_dim = label_dim
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)
        block_kwargs = dict(emb_channels=emb_channels, channels_per_head=64, dropout=dropout, init=init, init_zero=init_zero)

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_layer0 = Linear(in_features=model_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        self.map_label = Linear(in_features=label_dim, out_features=emb_channels, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(label_dim)) if self.label_dim else None

        # Encoder, i.e., the left part of the UNet where we downscale the image
        # at each level but increase the number of channels.
        #
        # The downscaling halves the image size (implemented as a bit shift).
        # The channels are multiplied as specified by the sequence
        # channel_mult.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        skips = [block.out_channels for block in self.enc.values()]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, class_labels=None):
        # Expected shapes:
        # x: (b, c, k, k)
        # noise_labels: (b)
        # class_labels: (b, class_label_dim)
        assert len(x.shape) == 4
        assert len(noise_labels.shape) == 1
        assert class_labels is None or len(class_labels.shape) == 2
        assert x.shape[0] == class_labels.shape[0]
        # Embed the noise scales (noise_labels) to a vector via a positional embedding.
        emb = self.map_noise(noise_labels)
        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)

        # Transform the class labels to be added to the positional embedding (with dropout).
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp)
        emb = silu(emb)

        # Encoder. Apply UNetBlock modules with inputs x and the embedding.
        skips = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)
        
        # Decoder. Apply UNetBlock modules with inputs x, the embedding, and
        # residual connections from the encoder layers.
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)

        x = self.out_conv(silu(self.out_norm(x)))
        return x
    
class EDMPrecond(torch.nn.Module):
    """Denoising model that preconditions an existing model.
    
    The aim is to denoise a noised input y + n to y where n is noise from
    N(0, \sigma^2*I) for a known \sigma. It is natural to train the model F to
    estimate the noise instead of y, i.e., to compute x - F(x, \sigma). It is
    easier to train neural networks with normalized inputs and outsputs, so it
    is natural to compute x - \sigma*F(x, \sigma). This form still has the
    problem that the input can vary a lot in magnitude when \sigma is large.
    This leads to the idea of computing
    x - \sigma*F(1/\sigma*x, \sigma). For large \sigma, any errors in F get
    amplified by a factor of \sigma, and directly precicting y can be a better
    choice. This in turn leads to the idea that sometimes, depending on \sigma,
    the role of x should be removed. This is expressed in

    c_skip(\sigma)*x + c_out(\sigma)*F(c_in(\sigma)*x, c_noise(\sigma))
    
    where c_skip(\sigma), c_out(\sigma), c_in(\sigma), and c_noise(\sigma) are
    numbers. These numbers could be learned by a neural network, but Karras et
    al. give fixed choices for these numbers in the Appendix B.6 of [0]. These
    are

    c_skip(\sigma) = \sigma_data^2 / (\sigma^2 + \sigma_data^2),
    c_out(\sigma) = \sigma * \sigma_data/\sqrt{\sigma_data^2 + \sigma^2}
    c_in(\sigma) = 1/\sqrt{\sigma^2 + \sigma_data^2}
    c_noise(\sigma) = \log(\sigma)/4
    
    where \sigma_data is the standard deviation of the target distribution
    (given as a parameter, default 0.5).

    So simply put: given the model F as the argument model to __init__, this
    model computes the above formula for the above scaling choices."""

    def __init__(
        self,
        model,                              # The model F to be preconditioned.
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
    ):
        super().__init__()
        self.model = model
        self.label_dim = model.label_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

    def forward(self, x, sigma, class_labels=None, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4
        
        F_x = self.model((c_in * x).to(torch.float32), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x
