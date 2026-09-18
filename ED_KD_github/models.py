"""CNCAN backbone and training-only adaptive distillation modules."""
from __future__ import annotations
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def channel_std(x: Tensor) -> Tensor:
    # FP32 variance and epsilon prevent sqrt(0) backward singularities.
    return (x.float().var(dim=(-2, -1), unbiased=False, keepdim=True) + 1e-8).sqrt().to(x.dtype)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, contrast: bool, gelu: bool = True) -> None:
        super().__init__()
        self.contrast = contrast
        self.net = nn.Sequential(nn.Conv2d(channels, max(1, channels // 16), 1),
                                 nn.GELU() if gelu else nn.ReLU(),
                                 nn.Conv2d(max(1, channels // 16), channels, 1), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        descriptor = x.float().mean((-2, -1), keepdim=True).to(x.dtype)  # [B,C,1,1]
        if self.contrast:
            descriptor = descriptor + channel_std(x)
        return x * self.net(descriptor).expand_as(x)  # [B,C,H,W]


class ESA(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        f = max(1, channels // 4)
        self.conv1 = nn.Conv2d(channels, f, 1)
        self.conv_f = nn.Conv2d(f, f, 1)
        self.conv2 = nn.Conv2d(f, f, 3, 2, 0)
        self.conv_max = nn.Conv2d(f, f, 3)
        self.conv3 = nn.Conv2d(f, f, 3)
        self.conv3_ = nn.Conv2d(f, f, 3)
        self.conv4 = nn.Conv2d(f, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        c1 = self.conv1(x)  # [B,C//4,H,W]
        # Original valid convolutions require >=51 LR pixels. Replicate padding
        # extends only this ESA branch on smaller inputs; backbone size is intact.
        small = F.pad(c1, (0, max(0, 3-c1.shape[-1]), 0, max(0, 3-c1.shape[-2])), mode='replicate')
        c2 = self.conv2(small)
        c2 = F.pad(c2, (0, max(0, 25-c2.shape[-1]), 0, max(0, 25-c2.shape[-2])), mode='replicate')
        z = F.max_pool2d(c2, 7, 3)  # spatial dimensions >=7
        z = F.gelu(self.conv_max(z))
        z = F.gelu(self.conv3(z))
        z = self.conv3_(z)
        z = F.interpolate(z, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return x * torch.sigmoid(self.conv4(z + self.conv_f(c1)))


class CNCAB(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c, c, 1)
        self.gca = ChannelAttention(c, False)
        self.gcca = ChannelAttention(c, True)

    def forward(self, x: Tensor) -> Tensor:
        z = F.gelu(self.conv(x))
        return self.gca(z) + self.gcca(z)


class CNCARB(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.gca = ChannelAttention(c, False)
        self.gcca = ChannelAttention(c, True)

    def forward(self, x: Tensor) -> Tensor:
        z = self.conv(x)
        return F.gelu(self.gca(z) + self.gcca(z) + x)


class CNCAM(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.left, self.right = CNCAB(c), CNCARB(c)
        self.c4, self.c5 = nn.Conv2d(c, c, 3, padding=1), nn.Conv2d(4*c, c, 1)
        self.esa = ESA(c)
        self.cca = ChannelAttention(c, True, False)
        self.gca = ChannelAttention(c, False)

    def forward(self, x: Tensor) -> Tensor:
        # The SAME left/right parameters are reused three times, as in user code.
        l1, r1 = self.left(x), self.right(x)
        l2, r2 = self.left(r1), self.right(r1)
        l3, r3 = self.left(r2), self.right(r2)
        fused = self.c5(torch.cat((l1, l2, l3, F.gelu(self.c4(r3))), 1))  # [B,C,H,W]
        return self.cca(self.esa(fused)) + x + self.gca(x)


class CNCAN(nn.Module):
    def __init__(self, num_block: int = 8, num_feat: int = 64, upscale: int = 4,
                 activation_checkpoint: bool = False) -> None:
        super().__init__()
        if num_block not in (1, 2, 4, 8) or upscale not in (2, 4, 8) or num_feat < 4:
            raise ValueError('blocks: 1/2/4/8; scale: 2/4/8; channels >=4')
        self.config = dict(num_block=num_block, num_feat=num_feat, upscale=upscale)
        self.activation_checkpoint = activation_checkpoint
        self.fea_conv = nn.Conv2d(12, num_feat, 3, padding=1)
        self.blocks = nn.ModuleList(CNCAM(num_feat) for _ in range(num_block))
        self.c1 = nn.Conv2d(num_feat*num_block, num_feat, 1)
        self.c2 = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.upsampler = nn.Sequential(nn.Conv2d(num_feat, 3*upscale**2, 3, padding=1), nn.PixelShuffle(upscale))

    def features(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 1:
            raise ValueError('Expected nonempty RGB [B,3,H,W]')
        base = self.fea_conv(torch.cat((x, x, x, x), 1))  # [B,C,H,W]
        z, stages = base, []
        for block in self.blocks:
            z = checkpoint(block, z, use_reentrant=False) if self.activation_checkpoint and self.training else block(z)
            stages.append(z)
        return self.c2(F.gelu(self.c1(torch.cat(stages, 1)))) + base  # [B,C,H,W]

    def forward(self, x: Tensor) -> Tensor:
        return self.upsampler(self.features(x))  # [B,3,scale*H,scale*W]


def norm_feature(x: Tensor) -> Tensor:
    x = x.float()
    return x / x.square().mean((1, 2, 3), keepdim=True).add(1e-6).sqrt()


def edge_map(x: Tensor) -> Tensor:
    # Local high-pass energy, same spatial shape, including 1x1 inputs.
    mono = x.float().abs().mean(1, keepdim=True)
    smooth = F.avg_pool2d(F.pad(mono, (1, 1, 1, 1), mode='replicate'), 3, 1)
    return (mono - smooth).abs()  # [B,1,H,W]


class FAM(nn.Module):
    """Cross-conditioned, channel-concatenated multi-scale local alignment.

    No claim of geometric registration: paired inputs must already be registered.
    """
    def __init__(self, c: int) -> None:
        super().__init__()
        self.mix = nn.Conv2d(2*c, c, 1)
        self.scales = nn.ModuleList(nn.Conv2d(c, c, 3, padding=d, dilation=d, groups=c) for d in (1, 2, 3))
        self.selector = nn.Conv2d(2*c, 3, 1)
        self.out = nn.Conv2d(c, c, 1)

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        if left.shape != right.shape:
            raise ValueError('FAM features must have identical shapes')
        both = torch.cat((norm_feature(left), norm_feature(right)), 1)  # [B,2C,H,W]
        z = F.gelu(self.mix(both))  # [B,C,H,W]
        weights = self.selector(both).softmax(1)  # [B,3,H,W]
        aligned = sum(layer(z) * weights[:, i:i+1].expand_as(z) for i, layer in enumerate(self.scales))
        return self.out(F.gelu(aligned))


class QASwitch(nn.Module):
    """Learned soft channel gate conditioned on structure and detached SR errors."""
    def __init__(self, c: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(2*c+2, max(8, c//4), 1), nn.GELU(),
                                 nn.Conv2d(max(8, c//4), c, 1))

    def forward(self, feature: Tensor, errors: Tensor) -> Tensor:
        x = norm_feature(feature)
        descriptor = torch.cat((x.mean((2, 3), keepdim=True), channel_std(x),
                                errors.detach().float().log1p().view(x.shape[0], 2, 1, 1)), 1)
        return 0.05 + 0.95 * self.net(descriptor).sigmoid()  # [B,C,1,1]; anti-collapse floor


class EGA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.GELU(), nn.Conv2d(8, 1, 3, padding=1))

    def forward(self, guide: Tensor, student_side: Tensor) -> Tensor:
        g = norm_feature(guide)
        descriptor = torch.cat((g.mean(1, keepdim=True), g.abs().amax(1, keepdim=True), edge_map(g)), 1)
        attention = 0.05 + 0.95*self.attention(descriptor).sigmoid()  # [B,1,H,W]
        return student_side * attention.expand_as(student_side)


class PairTransfer(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.fam, self.qa_left, self.qa_right, self.ega = FAM(c), QASwitch(c), QASwitch(c), EGA()

    def forward(self, left: Tensor, right: Tensor, errors: Tensor) -> Tensor:
        refined = self.fam(left, right)  # [B,C,H,W]
        product = refined * self.qa_left(left, errors).expand_as(refined)
        product = product * self.qa_right(right, errors.flip(1)).expand_as(refined)
        return self.ega(left, product)  # [B,C,H,W]; first member is EGA guide


class FusionStage(nn.Module):
    def __init__(self, c: int, scale: int) -> None:
        super().__init__()
        self.pair_a, self.pair_b = PairTransfer(c), PairTransfer(c)
        self.fuse = nn.Conv2d(2*c, c, 1)
        self.head = nn.Sequential(nn.Conv2d(c, 3*scale**2, 3, padding=1), nn.PixelShuffle(scale))

    def forward(self, a: Tensor, b: Tensor, c: Tensor, d: Tensor,
                errors_a: Tensor, errors_b: Tensor) -> tuple[Tensor, Tensor]:
        first, second = self.pair_a(a, b, errors_a), self.pair_b(c, d, errors_b)
        feature = self.fuse(torch.cat((first, second), 1))  # [B,C,H,W]
        return feature, self.head(feature)


class KDSystem(nn.Module):
    def __init__(self, channels: int = 64, scale: int = 4, activation_checkpoint: bool = True,
                 transfer_design: str = 'adaptive', qa_threshold_teacher: float = 0.0,
                 qa_threshold_assistant: float = 0.0, qa_temperature: float = 1.0,
                 qa_floor: float = 0.05) -> None:
        super().__init__()
        # L/M/S shared across Teacher and Assistant. AT and T are independent tiny models.
        self.models = nn.ModuleDict({name: CNCAN(blocks, channels, scale, activation_checkpoint)
                                     for name, blocks in dict(L=8, M=4, S=2, AT=1, T=1).items()})
        if transfer_design not in ('adaptive','paper_corrected','figure','proposal'):
            raise ValueError('Unknown transfer_design')
        if transfer_design == 'proposal':
            from proposal import ProposalFusionStage
            self.teacher_fusion = ProposalFusionStage(channels, scale, qa_threshold_teacher, qa_temperature, qa_floor)
            self.assistant_fusion = ProposalFusionStage(channels, scale, qa_threshold_assistant, qa_temperature, qa_floor)
            return
        if transfer_design == 'figure':
            from figure_modules import FigureFusionStage
            stage = FigureFusionStage
        else:
            stage = FusionStage if transfer_design == 'adaptive' else PaperFusionStage
        self.teacher_fusion = stage(channels, scale)
        self.assistant_fusion = stage(channels, scale)

    def forward(self, lr: Tensor, hr: Tensor) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        features = {name: model.features(lr) for name, model in self.models.items()}
        sr = {name: model.upsampler(features[name]) for name, model in self.models.items()}
        errors = {name: (value.detach().float()-hr.float()).square().mean((1, 2, 3)) for name, value in sr.items()}
        # Auxiliary heads train fusion modules against HR. Detaching here prevents
        # a backbone from learning to manufacture its own easier KD target.
        f = {name: value.detach() for name, value in features.items()}
        def pair_error(a: str, b: str) -> Tensor:
            return torch.stack((errors[a], errors[b]), 1)
        features['TF'], sr['TF'] = self.teacher_fusion(f['L'], f['M'], f['M'], f['S'], pair_error('L','M'), pair_error('M','S'))
        features['AF'], sr['AF'] = self.assistant_fusion(f['M'], f['S'], f['S'], f['L'], pair_error('M','S'), pair_error('S','L'))
        return features, sr


class PaperFAM(nn.Module):
    """Corrected Eq.45-48: output is a channel-spatial mask, not an SR feature."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        # GroupNorm replaces batch-dependent BN for batch-size-one training.
        self.net = nn.Sequential(nn.Conv2d(2*channels, channels, 1), nn.GroupNorm(1, channels),
                                 nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1), nn.Sigmoid())

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        if left.ndim != 4 or left.shape != right.shape:
            raise ValueError('PaperFAM requires identical [B,C,H,W] features')
        return self.net(torch.cat((left, right), dim=1))  # [B,C,H,W] in [0,1]


class ContrastGate(nn.Module):
    """Smooth version of Eq.32-38; a contrast proxy, not a quality guarantee."""
    def __init__(self, channels: int, floor: float = 0.05, temperature: float = 1.0) -> None:
        super().__init__()
        self.threshold = nn.Parameter(torch.zeros(1,channels,1,1))
        self.floor, self.temperature = floor, temperature

    def forward(self, feature: Tensor) -> Tensor:
        x = feature.float()
        mean = x.mean((-2,-1),keepdim=True)  # [B,C,1,1]
        std = (x.var((-2,-1),unbiased=False,keepdim=True)+1e-8).sqrt()
        gate = self.floor+(1-self.floor)*torch.sigmoid((mean+std-self.threshold)/self.temperature)
        return gate.to(feature.dtype)  # [B,C,1,1]


class PaperEGA(nn.Module):
    """A(U) is the spatial mask; E(U)=U*A(U) is the attended feature."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2,1,7,padding=3)

    def forward(self, feature: Tensor) -> Tensor:
        if feature.ndim != 4:
            raise ValueError('PaperEGA requires [B,C,H,W]')
        descriptor = torch.cat((feature.amax(1,keepdim=True),
                                feature.float().mean(1,keepdim=True).to(feature.dtype)),1)  # [B,2,H,W]
        mask = torch.sigmoid(self.conv(descriptor))  # [B,1,H,W]
        return feature*mask.expand_as(feature)  # [B,C,H,W]


class PaperPairTransfer(nn.Module):
    """Corrected Eq.22/23/26/27: concat produces 3C, then project to C."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fam = PaperFAM(channels)
        self.left_gate, self.right_gate = ContrastGate(channels), ContrastGate(channels)
        self.project = nn.Conv2d(3*channels,channels,1)
        self.ega = PaperEGA()

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        mask = self.fam(left,right)  # [B,C,H,W]
        mixed = torch.cat((left*self.left_gate(left).expand_as(left),
                           right*self.right_gate(right).expand_as(right),mask),1)  # [B,3C,H,W]
        return self.ega(self.project(mixed))  # [B,C,H,W]


class PaperFusionStage(nn.Module):
    """Corrected Eq.24/28: multiply attended FEATURES, not 1-channel masks.

    This implements the corrected fusion equations only. The online loss and
    five-backbone compression graph remain the existing explicitly documented design.
    """
    def __init__(self, channels: int, scale: int) -> None:
        super().__init__()
        self.pair_a, self.pair_b = PaperPairTransfer(channels), PaperPairTransfer(channels)
        self.head = nn.Sequential(nn.Conv2d(channels,3*scale**2,3,padding=1),nn.PixelShuffle(scale))

    def forward(self, a: Tensor, b: Tensor, c: Tensor, d: Tensor,
                errors_a: Tensor, errors_b: Tensor) -> tuple[Tensor, Tensor]:
        # Error descriptors are accepted for API compatibility; contrast QA does
        # not use SR quality. It must not be described as quality-aware gating.
        first,second = self.pair_a(a,b),self.pair_b(c,d)
        if first.shape != second.shape:
            raise ValueError('Attended branches must have identical shapes')
        feature = first*second  # [B,C,H,W]
        return feature,self.head(feature)  # feature, [B,3,sH,sW]
