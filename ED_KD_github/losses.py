"""MSE SR objective and explicitly adapted (not original) SwitOKD-SR-online."""
from __future__ import annotations
import torch
from torch import Tensor, nn
from models import norm_feature


def mse_each(x: Tensor, y: Tensor) -> Tensor:
    if x.shape != y.shape:
        raise ValueError(f'MSE refuses broadcasting: {x.shape} versus {y.shape}')
    return (x.float()-y.float()).square().flatten(1).mean(1)  # [B]


@torch.no_grad()
def switch_statistics(teacher: Tensor, student: Tensor, hr: Tensor) -> tuple[Tensor, Tensor]:
    # Original classification L1 distances are replaced by per-image RGB RMSE.
    dt = mse_each(teacher, hr).add(1e-12).sqrt()
    ds = mse_each(student, hr).add(1e-12).sqrt()
    gap = mse_each(teacher, student).add(1e-12).sqrt()
    epsilon = torch.exp(-dt / (dt+ds).clamp_min(1e-8))
    threshold = ds - epsilon*dt
    expert = (gap > threshold) & (dt < ds)  # [B]
    quality = torch.sigmoid(4.0*(ds-dt)/(ds+dt).clamp_min(1e-8))  # [B]
    return expert, quality


class DistillationLoss(nn.Module):
    def __init__(self, kd_weight: float = 0.1, auxiliary_weight: float = 0.25,
                 reverse_weight: float = 0.1) -> None:
        super().__init__()
        self.kd_weight, self.auxiliary_weight, self.reverse_weight = kd_weight, auxiliary_weight, reverse_weight

    def forward(self, features: dict[str, Tensor], sr: dict[str, Tensor], hr: Tensor,
                ramp: float = 1.0) -> tuple[Tensor, dict[str, Tensor]]:
        main = torch.stack([mse_each(sr[k], hr).mean() for k in ('L','M','S','AT','T')]).mean()
        auxiliary = torch.stack([mse_each(sr[k], hr).mean() for k in ('TF','AF')]).mean()
        losses, modes = [], []
        # Pairwise KD implements the named pairs; fused routes compress into
        # standalone S, assistant T, and finally a separate student T.
        for teacher, student in (('L','M'), ('M','S'), ('S','L'), ('TF','S'), ('AF','AT'), ('AT','T')):
            expert, quality = switch_statistics(sr[teacher], sr[student], hr)
            ft, fs = norm_feature(features[teacher]), norm_feature(features[student])
            forward = mse_each(fs, ft.detach()) + mse_each(sr[student], sr[teacher].detach())
            reverse = mse_each(ft, fs.detach()) + mse_each(sr[teacher], sr[student].detach())
            # All networks retain supervised HR updates. Expert mode only shuts
            # off this edge's reverse KD: it does NOT freeze the original teacher.
            losses.append((quality*forward).mean() + self.reverse_weight*((~expert).float()*(1-quality)*reverse).mean())
            modes.append(expert.float().mean())
        kd = torch.stack(losses).mean()
        total = main + self.auxiliary_weight*auxiliary + self.kd_weight*ramp*kd
        return total, {'loss': total.detach(), 'main_mse': main.detach(), 'aux_mse': auxiliary.detach(),
                       'kd': kd.detach(), 'expert_fraction': torch.stack(modes).mean().detach()}


class FigureDistillationLoss(nn.Module):
    """Forward quality-weighted KD with bounded spatial texture weighting.

    Keeps SR MSE as the main objective. Eliminates direct small-to-large S->L
    KD while retaining the S/L feature fusion pair. Reverse KD is omitted.
    """
    def __init__(self, kd_weight: float = 0.1, auxiliary_weight: float = 0.25, texture_weight: float = 1.0) -> None:
        super().__init__()
        if texture_weight < 0:
            raise ValueError('texture_weight must be nonnegative')
        self.kd_weight,self.auxiliary_weight=kd_weight,auxiliary_weight
        self.texture_weight=texture_weight

    def forward(self, features: dict[str,Tensor], sr: dict[str,Tensor], hr: Tensor,
                ramp: float = 1.0) -> tuple[Tensor,dict[str,Tensor]]:
        from figure_modules import highpass
        main = torch.stack([mse_each(sr[k],hr).mean() for k in ('L','M','S','AT','T')]).mean()
        auxiliary = torch.stack([mse_each(sr[k],hr).mean() for k in ('TF','AF')]).mean()
        terms,modes=[],[]
        for teacher,student in (('L','M'),('M','S'),('TF','S'),('AF','AT'),('S','AT'),('AT','T')):
            expert,quality=switch_statistics(sr[teacher],sr[student],hr)
            target=norm_feature(features[teacher]).detach()  # [B,C,H,W]
            current=norm_feature(features[student])
            if current.shape != target.shape:
                raise ValueError('KD feature shapes must match')
            with torch.no_grad():
                edge=highpass(target.abs().mean(1,keepdim=True)).abs()  # [B,1,H,W]
                weight=(1+self.texture_weight*edge/edge.mean((2,3),keepdim=True).add(1e-6)).clamp_max(4)
                weight=weight/weight.mean((2,3),keepdim=True)
            feature_loss=((current-target).square()*weight.expand_as(current)).flatten(1).mean(1)
            terms.append((quality*(feature_loss+mse_each(sr[student],sr[teacher].detach()))).mean())
            modes.append(expert.float().mean())
        kd=torch.stack(terms).mean()
        total=main+self.auxiliary_weight*auxiliary+self.kd_weight*ramp*kd
        return total,{'loss':total.detach(),'main_mse':main.detach(),'aux_mse':auxiliary.detach(),
                      'kd':kd.detach(),'expert_fraction':torch.stack(modes).mean().detach()}
