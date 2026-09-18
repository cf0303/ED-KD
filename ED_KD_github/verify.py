"""CPU integration checks; synthetic data, not an SR quality benchmark."""
from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from models import CNCAN, KDSystem
from losses import DistillationLoss, mse_each, switch_statistics


def main() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(7)
    results: dict[str, object] = {}
    for scale in (2,4,8):
        system = KDSystem(8,scale,True).train()
        lr,hr = torch.rand(1,3,9,11),torch.rand(1,3,9*scale,11*scale)
        features,sr = system(lr,hr)
        assert all(x.shape == hr.shape for x in sr.values())
        loss,_ = DistillationLoss()(features,sr,hr)
        loss.backward()
        for name,module in [('system',system),*system.models.items(),('teacher_fusion',system.teacher_fusion),('assistant_fusion',system.assistant_fusion)]:
            grads = [p.grad for p in module.parameters()]
            assert all(g is not None and torch.isfinite(g).all() for g in grads), name
            assert sum(g.abs().sum().item() for g in grads) > 0, name
        results[f'x{scale}_forward_backward_all_modules'] = 'passed'
    for depth in (1,2,4,8):
        model=CNCAN(depth,8,2)
        x=torch.zeros(1,3,1,1,requires_grad=True)
        model(x).square().mean().backward()
        assert torch.isfinite(x.grad).all()
    results['constant_1x1_backward'] = 'passed'
    try:
        mse_each(torch.ones(1,3,2,2),torch.ones(1,1,2,2))
        raise AssertionError('Broadcast was not rejected')
    except ValueError:
        results['broadcast_rejection']='passed'
    target=torch.zeros(1,3,2,2)
    expert,_=switch_statistics(target,torch.ones_like(target),target)
    assert expert.item()
    expert,_=switch_statistics(torch.ones_like(target),target,target)
    assert not expert.item()
    results['switch_expert_and_learning']='passed'
    system=KDSystem(8,2,False)
    f,sr=system(torch.rand(1,3,8,8),torch.rand(1,3,16,16))
    mse_each(f['S'],f['TF'].detach()).mean().backward()
    assert all(p.grad is None for p in system.teacher_fusion.parameters())
    assert all(p.grad is None for p in system.models['L'].parameters())
    results['kd_target_gradient_isolation']='passed'
    profile={}
    for label,depth in dict(L=8,M=4,S=2,T=1).items():
        model=CNCAN(depth,64,4).eval()
        counter=[0]
        def hook(module: torch.nn.Conv2d, inputs: tuple[torch.Tensor,...], output: torch.Tensor) -> None:
            counter[0] += output.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
        handles=[m.register_forward_hook(hook) for m in model.modules() if isinstance(m,torch.nn.Conv2d)]
        with torch.no_grad():
            model(torch.rand(1,3,64,64))
        for handle in handles:
            handle.remove()
        profile[label]={'parameters':sum(p.numel() for p in model.parameters()),'conv_MACs_LR64_x4':counter[0]}
    results['profile']=profile
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        (root/'lr').mkdir()
        (root/'hr').mkdir()
        rng=np.random.default_rng(2)
        for i in range(4):
            a=rng.integers(0,256,(12,14,3),dtype=np.uint8)
            Image.fromarray(a).save(root/'lr'/f'{i}.png')
            Image.fromarray(a).resize((28,24),Image.Resampling.BICUBIC).save(root/'hr'/f'{i}.png')
        for split in ('val','val2'):
            (root/f'{split}_lr').mkdir()
            (root/f'{split}_hr').mkdir()
            image=Image.fromarray(rng.integers(0,256,(12,14,3),dtype=np.uint8))
            image.save(root/f'{split}_lr'/'sample.png')
            image.resize((28,24)).save(root/f'{split}_hr'/'sample.png')
        command=[sys.executable,'train.py','--hr-root',str(root/'hr'),'--lr-root',str(root/'lr'),
                 '--val-hr-root',str(root/'val_hr'),'--val-lr-root',str(root/'val_lr'),
                 '--val2-hr-root',str(root/'val2_hr'),'--val2-lr-root',str(root/'val2_lr'),
                 '--output-root',str(root/'out'),'--run-name','check','--device','cpu','--channels','8',
                 '--scale','2','--patch','8','--epochs','1','--warmup-epochs','0','--ramp-epochs','1',
                 '--workers','0','--accumulation','2','--save-every','1']
        environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        subprocess.run(command,check=True,env=environment)
        folder=root/'out'/'IJRR2017'/'x2'/'check'
        assert len(list(folder.glob('*_epoch_0001.pt')))==6
        for split,count in (('train',4),('val',1),('val2',1)):
            for label in ('Teacher_output','AssistantTeacher_output','Student_output'):
                assert len(list((folder/'sr_images'/'epoch_0001'/split/label).glob('*.png')))==count

        state=torch.load(folder/'Student_output_epoch_0001.pt',weights_only=True)
        standalone=CNCAN(**state['config'])
        standalone.load_state_dict(state['state_dict'],strict=True)
        subprocess.run(command+['--resume',str(folder/'last.pt')],check=True,env=environment)
        subprocess.run([sys.executable,'infer.py','--checkpoint',str(folder/'Student_output_epoch_0001.pt'),
                        '--input',str(root/'lr'),'--output',str(root/'sr'),'--device','cpu'],check=True,env=environment)
        assert len(list((root/'sr').glob('*.png')))==4
        results['train_validate_export_resume_load_infer']='passed'
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
