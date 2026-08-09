"""Losses for Dan'sPaperGan CRB training (local copies)."""

import torch
import torch.nn.functional as F

from utilities.warp import warp


class DVFLoss:
    def loss(self, target_dvf, predict_dvf, mask):
        m = mask.expand_as(target_dvf)
        error = torch.abs(target_dvf - predict_dvf) * m
        return error.sum() / m.sum().clamp_min(1.0)


class ImageSimilarityLoss:
    def __init__(self, metric='ncc'):
        self.metric = metric

    def loss(self, reference_ct, target_ct, predict_dvf, mask):
        warped = warp(reference_ct, predict_dvf)
        if self.metric == 'mse':
            error = (warped - target_ct) ** 2 * mask
            return error.sum() / mask.sum().clamp_min(1.0)
        return self._ncc_masked(warped, target_ct, mask)

    @staticmethod
    def _ncc_masked(a, b, mask, eps=1e-5):
        m = mask
        dims = tuple(range(1, a.ndim))
        w = m.sum(dim=dims, keepdim=True).clamp_min(1.0)
        a_mean = (a * m).sum(dim=dims, keepdim=True) / w
        b_mean = (b * m).sum(dim=dims, keepdim=True) / w
        a = (a - a_mean) * m
        b = (b - b_mean) * m
        num = (a * b).sum(dim=dims)
        den = torch.sqrt((a * a).sum(dim=dims) * (b * b).sum(dim=dims) + eps)
        return torch.mean(1.0 - num / den)


class SmoothnessLoss:
    def loss(self, predict_dvf):
        dy = predict_dvf[:, :, 1:, :, :] - predict_dvf[:, :, :-1, :, :]
        dx = predict_dvf[:, :, :, 1:, :] - predict_dvf[:, :, :, :-1, :]
        dz = predict_dvf[:, :, :, :, 1:] - predict_dvf[:, :, :, :, :-1]
        return (torch.mean(dx ** 2) + torch.mean(dy ** 2) + torch.mean(dz ** 2)) / 3.0


class GANLoss:
    def __init__(self, real_label=0.9):
        self.real_label = real_label

    def discriminator_loss(self, discriminator, real_a, real_b, fake_a, fake_b):
        pred_real = discriminator(real_a, real_b)
        pred_fake = discriminator(fake_a, fake_b)
        loss_real = F.mse_loss(pred_real, torch.full_like(pred_real, self.real_label))
        loss_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))
        return 0.5 * (loss_real + loss_fake)

    def generator_loss(self, discriminator, fake_a, fake_b):
        pred = discriminator(fake_a, fake_b)
        return F.mse_loss(pred, torch.ones_like(pred))
