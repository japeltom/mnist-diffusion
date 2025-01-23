import torch

"""
This code is mostly adapted from the paper

    T. Karras, M. Aittala, T. Aila, S. Laine. Elucidating the design space of diffusion-based generative models. NeurIPS 2022 (2022), <https://arxiv.org/abs/2206.00364>

whose source code

    https://github.com/NVlabs/edm

is copyrighted by the NVIDIA CORPORATION & AFFILIATES and published under the
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
licence.
"""

class EDMLoss:
    """
    Computes a denoising loss |F(y + n) - y|^2 for a sample y and noise n
    averaged over samples y, noise n from N(0, \sigma*I), and \sigma from a
    lognormal distribution (designed to sample noise levels of medium
    magnitude). The loss is weighted by multiplying by (\sigma^2 +
    \sigma_data^2)/(\sigma*\sigma_data)^2 where where \sigma_data is the
    standard deviation of the sample distribution (given as a parameter,
    default 0.5); see [0] for justification."""

    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, model, images, labels=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal*self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma*self.sigma_data)**2
        y = images
        n = torch.randn_like(y) * sigma
        D_yn = model(y + n, sigma, labels)
        loss = weight * ((D_yn - y)**2)
        return loss.sum() / images.shape[0]

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
