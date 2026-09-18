"""Figure-preserving feature modulation and stable multiplicative fusion.

Both pairwise QA multiplications and the top/bottom product are explicit.
Performance is a hypothesis to test, not an established result.
"""
from __future__ import annotations
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def rms(x: Tensor) -> Tensor:
    x = x.float()
    return x/x.square().mean((1,2,3),keepdim=True).add(1e-6).sqrt()


def highpass(x: Tensor) -> Tensor:
    return x-F.avg_pool2d(F.pad(x,(1,1,1,1),mode='replicate'),3,1)


class ResidualMultiScaleFAM(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(2*channels,channels,1)
        self.branches = nn.ModuleList(nn.Conv2d(channels,channels,3,padding=d,dilation=d,groups=channels) for d in (1,2,3))
        self.select = nn.Conv2d(2*channels,3,1)
        self.refine = nn.Conv2d(channels,channels,1)
        self.residual_logit = nn.Parameter(torch.tensor(-2.197224577))  # sigmoid=.1

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        if left.ndim != 4 or left.shape != right.shape:
            raise ValueError('FAM requires matching [B,C,H,W]')
        paired = torch.cat((rms(left),rms(right)),1)  # [B,2C,H,W]
        base = self.project(paired)  # [B,C,H,W], signed content path
        weights = self.select(paired).float().softmax(1)  # [B,3,H,W], scale axis
        detail = sum(branch(F.gelu(base)).float()*weights[:,i:i+1].expand_as(base)
                     for i,branch in enumerate(self.branches))
        return base.float()+self.residual_logit.sigmoid()*self.refine(F.gelu(detail)).float()


class QualityChannelGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(2*channels+2,max(8,channels//4),1),nn.GELU(),
                                 nn.Conv2d(max(8,channels//4),channels,1))
        # Mostly open at initialization; still channel-selective and trainable.
        nn.init.normal_(self.net[-1].weight,std=0.01)
        nn.init.constant_(self.net[-1].bias,2.0)

    def forward(self, feature: Tensor, errors: Tensor) -> Tensor:
        if errors.shape != (feature.shape[0],2):
            raise ValueError('QA errors must be [B,2]')
        x = rms(feature)
        mean = x.mean((2,3),keepdim=True)
        std = x.var((2,3),unbiased=False,keepdim=True).add(1e-8).sqrt()
        err = errors.detach().float().log1p().reshape(x.shape[0],2,1,1)
        desc = torch.cat((mean,std,err),1)  # [B,2C+2,1,1]
        return 0.05+0.95*self.net(desc).float().sigmoid()  # [B,C,1,1]


class IdentityModulationEGA(nn.Module):
    """E=1+rho*tanh(RMS(U))*attention. E is not a probability mask."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3,8,3,padding=1),nn.GELU(),nn.Conv2d(8,1,3,padding=1))
        self.modulation_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, guide: Tensor, fused: Tensor) -> Tensor:
        if guide.shape != fused.shape:
            raise ValueError('EGA guide and fused feature must have matching shape')
        g = rms(guide)  # [B,C,H,W]
        energy = g.abs().mean(1,keepdim=True)
        descriptor = torch.cat((g.mean(1,keepdim=True),g.abs().amax(1,keepdim=True),
                                highpass(energy).abs()),1)  # [B,3,H,W]
        attention = 0.05+0.95*self.net(descriptor).float().sigmoid()  # [B,1,H,W]
        rho = 0.5*self.modulation_logit.sigmoid()  # scalar in (0,.5), initially .25
        residual = rms(fused).tanh()*attention.expand_as(fused)
        return 1.0+rho*residual  # [B,C,H,W], bounded in (.5,1.5)


class FigurePair(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fam = ResidualMultiScaleFAM(channels)
        self.left_gate, self.right_gate = QualityChannelGate(channels),QualityChannelGate(channels)
        self.ega = IdentityModulationEGA()

    def forward(self, left: Tensor, right: Tensor, errors: Tensor) -> Tensor:
        refined = self.fam(left,right)
        product = refined*self.left_gate(left,errors).expand_as(refined)
        product = product*self.right_gate(right,errors.flip(1)).expand_as(refined)
        return self.ega(left,product)  # [B,C,H,W]


class FigureFusionStage(nn.Module):
    def __init__(self, channels: int, scale: int) -> None:
        super().__init__()
        self.pair_a, self.pair_b = FigurePair(channels),FigurePair(channels)
        self.head = nn.Sequential(nn.Conv2d(channels,3*scale**2,3,padding=1),nn.PixelShuffle(scale))

    def forward(self, a: Tensor, b: Tensor, c: Tensor, d: Tensor,
                errors_a: Tensor, errors_b: Tensor) -> tuple[Tensor,Tensor]:
        first,second = self.pair_a(a,b,errors_a),self.pair_b(c,d,errors_b)
        # Same central multiplication as the Figure. Centering is part of the head input.
        feature = first*second-1.0  # [B,C,H,W]
        return feature,self.head(feature)  # [B,C,H,W], [B,3,rH,rW]
