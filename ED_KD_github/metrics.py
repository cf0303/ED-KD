"""RGB SR metrics and atomic, epoch-indexed CSV logging."""
from __future__ import annotations
import csv
import os
from pathlib import Path
from typing import Any
import torch
from torch import Tensor
import torch.nn.functional as F

MODEL_LABELS = {'L':'CNCAN_L', 'M':'CNCAN_M', 'S':'Teacher_output',
                'AT':'AssistantTeacher_output', 'T':'Student_output'}
CSV_FIELDS = ['epoch','split','model','loss_mse','optimization_mse','psnr_rgb','ssim_rgb',
              'total_loss','main_loss','auxiliary_loss','kd_loss','expert_fraction','kd_ramp','learning_rate',
              'teacher_freeze_fraction','assistant_freeze_fraction','shared_freeze_fraction']


@torch.no_grad()
def ssim_each(x: Tensor, y: Tensor) -> Tensor:
    """Per-image RGB SSIM: Gaussian sigma=1.5, K1=.01, K2=.03, range=1.

    Valid 11x11 windows and population covariance; average across channels and
    spatial locations. Tiny inputs use the largest fitting odd window (>=1).
    Inputs must already be clipped to [0,1] and border-cropped by the caller.
    """
    if x.shape != y.shape or x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 1:
        raise ValueError('SSIM requires matching nonempty [B,3,H,W] tensors')
    x, y = x.float(), y.float()
    size = min(11, x.shape[-2], x.shape[-1])
    size -= 1 - size % 2
    coords = torch.arange(size, device=x.device, dtype=torch.float32) - size//2
    weights = torch.exp(-coords.square()/(2*1.5**2))
    weights = weights/weights.sum()
    kernel = (weights[:,None]*weights[None,:]).view(1,1,size,size).expand(3,1,size,size).contiguous()
    def smooth(z: Tensor) -> Tensor:
        return F.conv2d(z, kernel, groups=3)  # [B,3,H-size+1,W-size+1]
    mx, my = smooth(x), smooth(y)
    vx = (smooth(x*x)-mx*mx).clamp_min(0)
    vy = (smooth(y*y)-my*my).clamp_min(0)
    cov = smooth(x*y)-mx*my
    numerator = (2*mx*my+0.01**2)*(2*cov+0.03**2)
    denominator = (mx.square()+my.square()+0.01**2)*(vx+vy+0.03**2)
    return (numerator/denominator).flatten(1).mean(1)  # [B]


def update_epoch_csv(path: Path, epoch: int, rows: list[dict[str, Any]]) -> None:
    """Replace current/future epoch rows on resume; preserve earlier epoch rows."""
    if any(int(row['epoch']) != epoch for row in rows):
        raise ValueError('Every row must belong to the requested epoch')
    previous: list[dict[str, Any]] = []
    if path.exists():
        with path.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != CSV_FIELDS:
                raise ValueError(f'CSV schema differs: {path}')
            previous = [row for row in reader if int(row['epoch']) < epoch]
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(previous)
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
