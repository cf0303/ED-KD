"""Figure topology, gradients, real Adam freeze, and controller checks (CPU)."""
from __future__ import annotations
import torch
from models import KDSystem
from losses import FigureDistillationLoss
from figure_modules import IdentityModulationEGA
from switching import FreezeController, apply_freeze


def main() -> None:
    torch.set_num_threads(2)
    for scale in (2,4,8):
        torch.manual_seed(42)
        model=KDSystem(8,scale,True,'figure')
        lr,hr=torch.rand(1,3,9,11),torch.rand(1,3,9*scale,11*scale)
        f,sr=model(lr,hr)
        loss,_=FigureDistillationLoss()(f,sr,hr)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert f['TF'].shape==(1,8,9,11) and sr['TF'].shape==hr.shape
        print(f'PASS: x{scale} shapes and all finite gradients; loss={loss.item():.6f}')
    ega=IdentityModulationEGA()
    z=torch.zeros(1,8,3,5)
    assert torch.equal(ega(z,z),torch.ones_like(z))
    e=ega(torch.randn_like(z),torch.randn_like(z))
    assert e.min()>.5 and e.max()<1.5
    residual_a,residual_b=torch.randn_like(z)*.1,torch.randn_like(z)*.1
    torch.testing.assert_close((1+residual_a)*(1+residual_b)-1,
                               residual_a+residual_b+residual_a*residual_b)
    print('PASS: neutral EGA, bounded modulation, product expansion')
    model=KDSystem(8,2,False,'figure')
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    lr,hr=torch.rand(1,3,8,8),torch.rand(1,3,16,16)
    def backward() -> None:
        optimizer.zero_grad(set_to_none=True)
        f,sr=model(lr,hr)
        FigureDistillationLoss()(f,sr,hr)[0].backward()
    backward(); optimizer.step()
    modules={**dict(model.models.items()),'TF':model.teacher_fusion,'AF':model.assistant_fusion}
    for teacher,assistant in ((True,False),(False,True),(True,True),(False,False)):
        before={name:[p.detach().clone() for p in module.parameters()] for name,module in modules.items()}
        steps={name:[optimizer.state[p]['step'].item() for p in module.parameters()] for name,module in modules.items()}
        backward(); apply_freeze(model,teacher,assistant); optimizer.step()
        frozen=set(('L','M','S')) if teacher or assistant else set()
        if teacher: frozen.add('TF')
        if assistant: frozen.update(('AF','AT'))
        for name,module in modules.items():
            equal=[torch.equal(p,b) for p,b in zip(module.parameters(),before[name])]
            if name in frozen:
                assert all(equal),name
                assert all(optimizer.state[p]['step'].item()==t for p,t in zip(module.parameters(),steps[name])),name
            else:
                assert not all(equal),name
        print('PASS: true Adam freeze/unfreeze',teacher,assistant)
    controller=FreezeController(0.0)
    assert controller.decide(torch.tensor([[.1,1.,1.],[1.,.1,1.]]),True)==(True,False)
    assert controller.decide(torch.tensor([[1.,.1,1.],[.1,1.,1.]]),True)==(False,True)
    assert controller.decide(torch.tensor([[.1,1.,1.],[.1,1.,1.]]),False)==(False,False)
    restored=FreezeController(0.0); restored.load_state_dict(controller.state_dict())
    assert torch.equal(restored.ema,controller.ema)
    print('PASS: mode decisions, warmup, controller state round trip')


if __name__=='__main__':
    main()
