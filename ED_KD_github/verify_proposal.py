"""Meaningful CPU checks for the corrected manuscript implementation."""
from __future__ import annotations

import torch
from models import KDSystem
from proposal import ProposalLoss, ProposalController, ProposalEGA, ProposalQA, channel_standardize
from switching import apply_freeze


def main() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(42)
    for scale in (2, 4, 8):
        system = KDSystem(8, scale, True, 'proposal')
        lr, hr = torch.rand(1, 3, 9, 11), torch.rand(1, 3, 9 * scale, 11 * scale)
        features, sr = system(lr, hr)
        assert not system.teacher_fusion.pair_a.guide_right
        assert system.teacher_fusion.pair_b.guide_right
        assert not system.assistant_fusion.pair_a.guide_right
        assert system.assistant_fusion.pair_b.guide_right
        loss, _ = ProposalLoss()(features, sr, hr)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in system.parameters())
        assert features['TF'].shape == (1, 8, 9, 11) and sr['T'].shape == hr.shape
        print(f'PASS x{scale}: all parameters have finite gradients')
    constant = torch.ones(1, 8, 1, 1, requires_grad=True)
    channel_standardize(constant).square().sum().backward()
    assert torch.isfinite(constant.grad).all()
    assert torch.equal(channel_standardize(constant), torch.zeros_like(constant))
    x = torch.randn(2, 8, 5, 7)
    torch.testing.assert_close(channel_standardize(x).mean((2, 3)), torch.zeros(2, 8), atol=1e-6, rtol=0)
    print('PASS channel normalization: axes and constant 1x1 gradient')
    ega = ProposalEGA()
    guide, feature = torch.randn(1, 8, 3, 5), torch.ones(1, 8, 3, 5) * 0.1
    full, suppressed = ega(guide, feature), ega(guide, feature * 0.01)
    assert (suppressed - 1).abs().mean() < (full - 1).abs().mean() * 0.02
    assert torch.equal(ega(guide, torch.zeros_like(feature)), torch.ones_like(feature))
    gate = ProposalQA(8, 0.0, 1.0, 0.05)
    good = gate(guide, torch.tensor([[0.01, 0.5]]))
    bad = gate(guide, torch.tensor([[0.5, 0.01]]))
    assert (good > bad).all() and (bad > 0.05).all() and (good < 1).all()
    print('PASS QA floor, error ordering and EGA preserves gate attenuation')
    hr = torch.zeros(1, 3, 4, 4)
    # Opposite errors: gap .3 > .2-exp(-.1/.2)*.1; better teacher becomes expert.
    sr = {'S': torch.full_like(hr, 0.1), 'AT': torch.full_like(hr, -0.2), 'T': torch.full_like(hr, 0.3)}
    controller = ProposalController(0.0)
    assert controller.decide(sr, hr, False) == (False, False)
    assert controller.decide(sr, hr, True) == (True, True)
    worse = {'S': torch.full_like(hr, 0.4), 'AT': torch.full_like(hr, 0.2), 'T': torch.full_like(hr, 0.1)}
    assert controller.decide(worse, hr, True) == (False, False)
    equal = {k: torch.zeros_like(hr) for k in ('S', 'AT', 'T')}
    assert controller.decide(equal, hr, True) == (False, False)
    print('PASS MAE controller: expert, worse teacher safeguard, equality, warmup')
    # Isolate KD gradients from supervised terms: teacher feature has reverse
    # gradient in learning, exactly zero in expert (targets always detached).
    for expert in (False, True):
        fs = {k: torch.randn(1, 8, 4, 4, requires_grad=True) for k in ('L', 'M', 'S', 'AT', 'T', 'TF', 'AF')}
        zs = {k: torch.randn_like(hr, requires_grad=True) for k in fs}
        objective = ProposalLoss(kd_weight=0, auxiliary_weight=0, reverse_weight=1)
        loss, _ = objective(fs, zs, hr, expert=(expert, expert))
        loss.backward()
        assert fs['S'].grad is not None
        if expert:
            assert torch.count_nonzero(fs['S'].grad) == 0
        else:
            assert fs['S'].grad.abs().sum() > 0
    print('PASS reverse KD exists only in learning; detached targets do not receive gradients')
    system = KDSystem(8, 2, False, 'proposal')
    optimizer = torch.optim.Adam(system.parameters(), lr=1e-3)
    lr, hr = torch.rand(1, 3, 8, 8), torch.rand(1, 3, 16, 16)
    def backward(flags: tuple[bool, bool]) -> None:
        optimizer.zero_grad(set_to_none=True)
        features, sr = system(lr, hr)
        ProposalLoss()(features, sr, hr, expert=flags)[0].backward()
    backward((False, False))
    optimizer.step()
    modules = {**dict(system.models.items()), 'TF': system.teacher_fusion, 'AF': system.assistant_fusion}
    for flags in ((True, False), (False, True), (True, True), (False, False)):
        snapshots = {key: [(p.detach().clone(), {n: v.clone() for n, v in optimizer.state[p].items()})
                           for p in module.parameters()] for key, module in modules.items()}
        backward(flags)
        apply_freeze(system, *flags)
        optimizer.step()
        frozen = set(('L', 'M', 'S')) if any(flags) else set()
        if flags[0]: frozen.add('TF')
        if flags[1]: frozen.update(('AF', 'AT'))
        for key, module in modules.items():
            changed = False
            for parameter, (before, state) in zip(module.parameters(), snapshots[key]):
                changed |= not torch.equal(parameter, before)
                if key in frozen:
                    assert torch.equal(parameter, before)
                    assert all(torch.equal(optimizer.state[parameter][n], value) for n, value in state.items())
            if key not in frozen: assert changed, key
        print(f'PASS freeze {flags}: exact parameter and Adam moment preservation, active groups update')
    print('All proposal CPU checks passed.')


if __name__ == '__main__':
    main()
