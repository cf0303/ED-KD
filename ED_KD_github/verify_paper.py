import torch
from models import KDSystem, PaperFAM, PaperEGA
from losses import DistillationLoss

torch.set_num_threads(2)
for scale in (2,4,8):
    torch.manual_seed(42)
    model=KDSystem(8,scale,False,'paper_corrected')
    lr=torch.rand(1,3,9,11); hr=torch.rand(1,3,9*scale,11*scale)
    features,sr=model(lr,hr)
    assert features['TF'].shape==(1,8,9,11)
    assert sr['TF'].shape==hr.shape and sr['AF'].shape==hr.shape
    loss,_=DistillationLoss()(features,sr,hr); loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(sum(p.grad.abs().sum().item() for p in stage.parameters())>0 for stage in (model.teacher_fusion,model.assistant_fusion))
    print('PASS: paper_corrected x',scale,'loss',loss.item())
x=torch.rand(1,8,9,11)
m=PaperFAM(8)(x,x); assert m.shape==x.shape and m.min()>=0 and m.max()<=1
assert PaperEGA()(x).shape==x.shape
print('PASS: mask and feature dimensional contracts')
