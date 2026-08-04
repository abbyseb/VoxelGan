"""Losses for phase-conditioned DVF GAN training.

Four objects, each with a .loss(...) method taking raw tensors — same shape
as the sample's flow_mask / image classes. GANLoss exposes discriminator_loss
and generator_loss so the train loop can keep .detach() visible at the call site.

DVFLoss and ImageSimilarityLoss are lung-masked so Elastix rib/chest-wall
errors do not dominate the supervised signal.
"""

import torch
import torch.nn.functional as F

from utilities.svf import warp


class DVFLoss:
    """L1 between generated DVF and Elastix target DVF (lung-masked)."""

    def loss(self, target_dvf, predict_dvf, mask):
        # mask: (B, 1, D, H, W) → broadcast over 3 DVF channels
        m = mask.expand_as(target_dvf)
        error = torch.abs(target_dvf - predict_dvf) * m
        return error.sum() / m.sum().clamp_min(1.0)


class ImageSimilarityLoss:
    """Warp reference CT by predicted DVF, compare to target CT inside lung."""

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
        # lung-masked global NCC; return 1 - NCC so lower is better
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
    """Diffusion regularizer: mean squared finite differences of the DVF."""

    def loss(self, predict_dvf):
        dy = predict_dvf[:, :, 1:, :, :] - predict_dvf[:, :, :-1, :, :]
        dx = predict_dvf[:, :, :, 1:, :] - predict_dvf[:, :, :, :-1, :]
        dz = predict_dvf[:, :, :, :, 1:] - predict_dvf[:, :, :, :, :-1]
        return (torch.mean(dx ** 2) + torch.mean(dy ** 2) + torch.mean(dz ** 2)) / 3.0


class GANLoss:
    """LSGAN on PatchGAN logits. Real labels smoothed to 0.9."""

    def __init__(self, real_label=0.9):
        self.real_label = real_label

    def discriminator_loss(self, discriminator, reference_ct, real_dvf, fake_dvf):
        # fake_dvf must already be .detach()'d at the call site
        pred_real = discriminator(reference_ct, real_dvf)
        pred_fake = discriminator(reference_ct, fake_dvf)
        loss_real = F.mse_loss(pred_real, torch.full_like(pred_real, self.real_label))
        loss_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))
        return 0.5 * (loss_real + loss_fake)

    def generator_loss(self, discriminator, reference_ct, fake_dvf):
        pred = discriminator(reference_ct, fake_dvf)
        return F.mse_loss(pred, torch.ones_like(pred))
