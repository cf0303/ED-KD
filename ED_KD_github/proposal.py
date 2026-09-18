"""Corrected ED-KD manuscript modules, bidirectional SR loss and mode control.

This is an SR adaptation, not a reproduction of classification SwitOKD.
All image losses use raw RGB predictions; standardized features are not probabilities.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
from losses import mse_each


def check_feature(x: Tensor) -> None:
    if x.ndim != 4 or min(x.shape) < 1:
        raise ValueError('Expected nonempty [B,C,H,W] tensor')


def channel_standardize(x: Tensor) -> Tensor:
    check_feature(x)
    value = x.float()  # [B,C,H,W]; FP32 statistics even under autocast
    variance, mean = torch.var_mean(value, dim=(2, 3), unbiased=False, keepdim=True)
    return (value - mean) / (variance + 1e-6).sqrt()  # [B,C,H,W]


def mae_each(x: Tensor, y: Tensor) -> Tensor:
    if x.shape != y.shape or x.ndim != 4:
        raise ValueError('MAE requires equal [B,C,H,W]; broadcasting is forbidden')
    return (x.float() - y.float()).abs().flatten(1).mean(1)  # [B]


class ProposalFAM(nn.Module):
    """Channel-spatial mask with a signed residual content path.

    GroupNorm(1,C) is independent of the batch and has no running statistics.
    This is local cross-stream modulation, not geometric or semantic registration.
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels < 2:
            raise ValueError('Proposal FAM requires channels >= 2')
        self.content = nn.Conv2d(2 * channels, channels, 1)
        self.mask = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1), nn.GroupNorm(1, channels),
            nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        check_feature(left)
        if left.shape != right.shape:
            raise ValueError('FAM requires equal feature shapes')
        pair = torch.cat((left.float(), right.float()), dim=1)  # [B,2C,H,W]
        content = self.content(pair).float()  # [B,C,H,W]
        mask = self.mask(pair).float()  # [B,C,H,W], mask in [0,1]
        return content * (1.0 + mask)  # [B,C,H,W], signed feature, not a mask


class ProposalQA(nn.Module):
    """Smooth contrast gate with detached relative SR-error conditioning."""
    def __init__(self, channels: int, threshold: float, temperature: float, floor: float) -> None:
        super().__init__()
        if not (math.isfinite(threshold) and math.isfinite(temperature) and temperature > 0 and 0 < floor < 1):
            raise ValueError('QA requires finite threshold, temperature>0 and 0<floor<1')
        self.threshold = nn.Parameter(torch.full((1, channels, 1, 1), float(threshold)))
        self.temperature, self.floor = temperature, floor

    def forward(self, feature: Tensor, errors: Tensor) -> Tensor:
        check_feature(feature)
        if errors.shape != (feature.shape[0], 2):
            raise ValueError('QA errors must be [B,2]')
        value = feature.float()  # raw features: standardizing here erases contrast differences
        variance, mean = torch.var_mean(value, dim=(2, 3), unbiased=False, keepdim=True)
        contrast = (variance + 1e-6).sqrt()  # [B,C,1,1]
        own, peer = errors.detach().float().unbind(1)  # [B], raw SR MSE
        quality = ((peer - own) / (peer + own + 1e-6)).reshape(-1, 1, 1, 1)
        score = (mean + contrast - self.threshold) / self.temperature + quality
        return self.floor + (1.0 - self.floor) * score.sigmoid()  # [B,C,1,1]


class ProposalEGA(nn.Module):
    """Teacher-guided spatial mask; output is identity-centered C-channel modulation."""
    def __init__(self) -> None:
        super().__init__()
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)
        self.strength_logit = nn.Parameter(torch.zeros(()))

    def forward(self, guide: Tensor, feature: Tensor) -> Tensor:
        check_feature(guide)
        if guide.shape != feature.shape:
            raise ValueError('EGA inputs must have identical shapes')
        value = guide.float()  # [B,C,H,W]
        descriptor = torch.cat((value.amax(1, keepdim=True), value.mean(1, keepdim=True)), dim=1)
        attention = self.spatial(descriptor).float().sigmoid()  # [B,1,H,W]
        strength = 0.5 * self.strength_logit.sigmoid()  # scalar; initially 0.25
        # No RMS after QA: renormalizing would cancel uniform gate suppression.
        residual = feature.float().tanh() * attention.expand_as(feature)
        return 1.0 + strength * residual  # [B,C,H,W], in (0.5,1.5)


class ProposalPair(nn.Module):
    def __init__(self, channels: int, threshold: float, temperature: float, floor: float,
                 guide_right: bool = False) -> None:
        super().__init__()
        self.guide_right = guide_right
        self.fam = ProposalFAM(channels)
        self.left_gate = ProposalQA(channels, threshold, temperature, floor)
        self.right_gate = ProposalQA(channels, threshold, temperature, floor)
        self.ega = ProposalEGA()

    def forward(self, left: Tensor, right: Tensor, errors: Tensor) -> Tensor:
        content = self.fam(left, right)  # [B,C,H,W]
        gated = content * self.left_gate(left, errors).expand_as(content)
        gated = gated * self.right_gate(right, errors.flip(1)).expand_as(content)
        return self.ega(right if self.guide_right else left, gated)  # [B,C,H,W]


class ProposalFusionStage(nn.Module):
    def __init__(self, channels: int, scale: int, threshold: float = 0.0,
                 temperature: float = 1.0, floor: float = 0.05) -> None:
        super().__init__()
        self.pair_a = ProposalPair(channels, threshold, temperature, floor)
        self.pair_b = ProposalPair(channels, threshold, temperature, floor, guide_right=True)
        self.head = nn.Sequential(nn.Conv2d(channels, 3 * scale ** 2, 3, padding=1), nn.PixelShuffle(scale))

    def forward(self, a: Tensor, b: Tensor, c: Tensor, d: Tensor,
                errors_a: Tensor, errors_b: Tensor) -> tuple[Tensor, Tensor]:
        first = self.pair_a(a, b, errors_a)  # [B,C,H,W]
        second = self.pair_b(c, d, errors_b)  # [B,C,H,W]
        feature = first * second - 1.0  # [B,C,H,W]; central Figure product preserved
        return feature, self.head(feature)  # feature, [B,3,rH,rW]


class ProposalController(nn.Module):
    """MAE-based switch with raw SR outputs and a teacher-quality safeguard.

    Decide on the first microbatch of an accumulation group and hold the flags
    for its entire backward/update. Loss and optimizer use the identical flags.
    Optional EMA is updated once per group; momentum=0 uses current observations.
    """
    def __init__(self, momentum: float = 0.0) -> None:
        super().__init__()
        if not 0 <= momentum < 1:
            raise ValueError('switch_ema must lie in [0,1)')
        self.momentum = momentum
        self.register_buffer('ema', torch.zeros(2, 3))
        self.register_buffer('initialized', torch.tensor(False))

    @torch.no_grad()
    def decide(self, sr: dict[str, Tensor], hr: Tensor, enabled: bool) -> tuple[bool, bool]:
        observation = torch.stack([
            torch.stack((mae_each(sr[u], hr).mean(), mae_each(sr[v], hr).mean(),
                         mae_each(sr[u], sr[v]).mean()))
            for u, v in (('S', 'AT'), ('AT', 'T'))
        ])  # [2,3]: teacher MAE, student MAE, mutual MAE
        if not torch.isfinite(observation).all():
            raise FloatingPointError('Nonfinite switch observation')
        if bool(self.initialized):
            self.ema.mul_(self.momentum).add_(observation, alpha=1 - self.momentum)
        else:
            self.ema.copy_(observation)
            self.initialized.fill_(True)
        if not enabled:
            return False, False
        teacher_error, student_error, gap = self.ema.unbind(1)  # [2]
        mixing = torch.exp(-teacher_error / student_error.clamp_min(1e-8))
        threshold = student_error - mixing * teacher_error
        expert = (gap > threshold) & (teacher_error < student_error)
        flags = expert.cpu().tolist()
        return bool(flags[0]), bool(flags[1])


class ProposalLoss(nn.Module):
    """Raw RGB MSE + normalized feature KD, mutual only in learning mode."""
    def __init__(self, kd_weight: float = 0.1, auxiliary_weight: float = 0.25,
                 reverse_weight: float = 0.1, feature_weight: float = 0.1,
                 alpha1: float = 0.1, beta1: float = 0.2,
                 alpha2: float = 0.1, beta2: float = 0.2) -> None:
        super().__init__()
        weights = (kd_weight, auxiliary_weight, reverse_weight, feature_weight, alpha1, beta1, alpha2, beta2)
        if any(not math.isfinite(w) or w < 0 for w in weights) or beta1 < alpha1 or beta2 < alpha2:
            raise ValueError('Loss weights must be finite, nonnegative and beta_i >= alpha_i')
        self.kd_weight, self.auxiliary_weight = kd_weight, auxiliary_weight
        self.reverse_weight, self.feature_weight = reverse_weight, feature_weight
        self.alphas, self.betas = (alpha1, alpha2), (beta1, beta2)

    def forward(self, features: dict[str, Tensor], sr: dict[str, Tensor], hr: Tensor,
                ramp: float = 1.0, expert: tuple[bool, bool] = (False, False)
                ) -> tuple[Tensor, dict[str, Tensor]]:
        if len(expert) != 2 or any(type(flag) is not bool for flag in expert):
            raise ValueError('expert must contain two Python bool flags')
        main = torch.stack([mse_each(sr[k], hr).mean() for k in ('L', 'M', 'S', 'AT', 'T')]).mean()
        auxiliary = torch.stack([mse_each(sr[k], hr).mean() for k in ('TF', 'AF')]).mean()
        normalized = {key: channel_standardize(value) for key, value in features.items()}

        def distill(source: str, destination: str) -> Tensor:
            image = mse_each(sr[destination], sr[source].detach()).mean()
            feature = mse_each(normalized[destination], normalized[source].detach()).mean()
            return image + self.feature_weight * feature

        stage_terms: list[Tensor] = []
        for i, (teacher, student) in enumerate((('S', 'AT'), ('AT', 'T'))):
            weight = self.betas[i] if expert[i] else self.alphas[i]
            term = weight * distill(teacher, student)
            if not expert[i]:
                term = term + self.reverse_weight * self.alphas[i] * distill(student, teacher)
            stage_terms.append(term)
        stage_kd = torch.stack(stage_terms).mean()
        support_kd = torch.stack([distill(u, v) for u, v in
                                 (('L', 'M'), ('M', 'S'), ('TF', 'S'), ('AF', 'AT'))]).mean()
        kd = stage_kd + self.kd_weight * support_kd
        total = main + self.auxiliary_weight * auxiliary + ramp * kd
        return total, {'loss': total.detach(), 'main_mse': main.detach(), 'aux_mse': auxiliary.detach(),
                       'kd': kd.detach(), 'expert_fraction': total.new_tensor(sum(expert) / 2)}
