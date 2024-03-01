import torch
import torch.nn as nn


class MMDLoss(nn.Module):

    def __init__(self):
        super(MMDLoss, self).__init__()

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        Kxx = self.gaussian_kernel(f_s, f_s).mean()
        Kyy = self.gaussian_kernel(f_t, f_t).mean()
        Kxy = self.gaussian_kernel(f_s, f_t).mean()
        return Kxx + Kyy - 2 * Kxy
