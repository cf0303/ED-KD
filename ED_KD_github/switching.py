"""Actual optimizer freeze with explicit ownership for shared backbones."""
from __future__ import annotations
import torch
from torch import Tensor, nn
from losses import mse_each


@torch.no_grad()
def switch_observations(sr: dict[str,Tensor], hr: Tensor) -> Tensor:
    rows = []
    # Stage outputs are the exported compressed Teacher S and Assistant AT.
    for teacher,student in (('S','AT'),('AT','T')):
        dt = mse_each(sr[teacher].detach(),hr).add(1e-12).sqrt()
        ds = mse_each(sr[student].detach(),hr).add(1e-12).sqrt()
        gap = mse_each(sr[teacher].detach(),sr[student].detach()).add(1e-12).sqrt()
        rows.append(torch.stack((dt,ds,gap),1).sum(0))
    return torch.stack(rows)  # [2,3], SUM over images; caller divides by group sample count


class FreezeController(nn.Module):
    def __init__(self, momentum: float = 0.9) -> None:
        super().__init__()
        if not 0 <= momentum < 1:
            raise ValueError('switch_ema must be in [0,1)')
        self.momentum = momentum
        self.register_buffer('ema',torch.zeros(2,3))
        self.register_buffer('initialized',torch.tensor(False))

    @torch.no_grad()
    def decide(self, observation: Tensor, enabled: bool) -> tuple[bool,bool]:
        if observation.shape != (2,3) or not torch.isfinite(observation).all():
            raise ValueError('Expected finite switch observations [2,3]')
        if bool(self.initialized):
            self.ema.mul_(self.momentum).add_(observation,alpha=1-self.momentum)
        else:
            self.ema.copy_(observation)
            self.initialized.fill_(True)
        if not enabled:
            return False,False
        dt,ds,gap = self.ema.unbind(1)
        threshold = ds-torch.exp(-dt/(dt+ds).clamp_min(1e-8))*dt
        expert = (gap>threshold)&(dt<ds)
        flags=expert.cpu().tolist()
        return bool(flags[0]),bool(flags[1])


@torch.no_grad()
def apply_freeze(system: nn.Module, teacher: bool, assistant: bool) -> None:
    """None gradients are skipped by Adam, including its moment updates.

    Shared L/M/S: either freeze request wins. TF: Teacher ownership.
    AF+AT: Assistant ownership. Final T: never frozen by these switches.
    No requires_grad toggles or optimizer recreation; unfreeze keeps Adam state.
    """
    modules: list[nn.Module] = []
    if teacher or assistant:
        modules.extend(system.models[name] for name in ('L','M','S'))
    if teacher:
        modules.append(system.teacher_fusion)
    if assistant:
        modules.extend((system.assistant_fusion,system.models['AT']))
    for module in modules:
        for parameter in module.parameters():
            parameter.grad = None
