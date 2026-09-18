from __future__ import annotations
import argparse
import csv
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from data import PairedImages, pair_images, seed_worker
from models import KDSystem
from losses import DistillationLoss, FigureDistillationLoss, mse_each
from switching import FreezeController, switch_observations, apply_freeze
from proposal import ProposalController, ProposalLoss
from metrics import MODEL_LABELS, ssim_each, update_epoch_csv


def save_atomic(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def save_rgb(x: Tensor, path: Path) -> None:
    from PIL import Image
    a = (x.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()*255).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path)


def export_models(system: KDSystem, folder: Path, suffix: str) -> None:
    for key, label in MODEL_LABELS.items():
        model = system.models[key]
        save_atomic({'config': model.config, 'state_dict': {k:v.detach().cpu() for k,v in model.state_dict().items()},
                     'role': key}, folder / f'{label}_{suffix}.pt')


@torch.inference_mode()
def validate(system: KDSystem, loader: DataLoader, device: torch.device, scale: int,
             preview: Path | None = None) -> dict[str, dict[str, float]]:
    system.eval()
    totals = {key: torch.zeros(3, device=device, dtype=torch.float64) for key in system.models}
    count = 0
    image_manifest: list[dict[str, str]] = []
    for index, (lr, hr, names) in enumerate(loader):
        lr, hr = lr.to(device), hr.to(device)
        if lr.shape[0] != 1:
            raise ValueError('Full-image validation uses batch size 1')
        # Index prefix prevents collisions for duplicate basenames in subfolders.
        image_name = f'{index:06d}_{Path(names[0]).stem}.png'
        for key, model in system.models.items():
            raw = model(lr)  # [1,3,sH,sW], full-image FP32 inference
            val_mse = mse_each(raw, hr).mean()  # raw, full-frame reconstruction loss
            pred = raw.clamp(0, 1)
            p, y = pred, hr
            if min(hr.shape[-2:]) > 2*scale:
                p, y = pred[:, :, scale:-scale, scale:-scale], hr[:, :, scale:-scale, scale:-scale]
            metric_mse = mse_each(p, y)
            psnr = -10*torch.log10(metric_mse.clamp_min(1e-12))
            ssim = ssim_each(p, y)
            totals[key] += torch.stack((val_mse, psnr.mean(), ssim.mean())).double()
            if preview is not None and key in ('S', 'AT', 'T'):
                save_rgb(pred[0], preview / MODEL_LABELS[key] / image_name)
        if preview is not None:
            save_rgb(hr[0], preview / 'HR' / image_name)
            save_rgb(lr[0], preview / 'LR' / image_name)
            low, high = loader.dataset.pairs[index]
            image_manifest.append({'saved_name':image_name, 'lr_source':str(low), 'hr_source':str(high)})
        count += 1
    if not count:
        raise ValueError('Validation set is empty')
    if preview is not None:
        with (preview/'image_manifest.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream,fieldnames=['saved_name','lr_source','hr_source'])
            writer.writeheader()
            writer.writerows(image_manifest)
    return {key: dict(zip(('loss_mse','psnr_rgb','ssim_rgb'), (values/count).cpu().tolist()))
            for key,values in totals.items()}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Online CNCAN SR KD; RGB [0,1], paired LR/HR')
    p.add_argument('--dataset', choices=['IJRR2017','CWFID','rice_seedling'], default='IJRR2017')
    p.add_argument('--scale', type=int, choices=[2,4,8], default=4)
    p.add_argument('--hr-root', type=Path)
    p.add_argument('--lr-root', type=Path)
    p.add_argument('--val-hr-root', type=Path)
    p.add_argument('--val-lr-root', type=Path)
    p.add_argument('--val2-hr-root', type=Path)
    p.add_argument('--val2-lr-root', type=Path)
    p.add_argument('--output-root', type=Path, default=Path(r'E:\LAB_all_data\result_files'))
    p.add_argument('--run-name', default=None)
    p.add_argument('--resume', type=Path)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--patch', type=int, default=64, help='LR patch size')
    p.add_argument('--channels', type=int, default=64)
    p.add_argument('--transfer-design', choices=['adaptive','paper_corrected','figure','proposal'], default='adaptive')
    p.add_argument('--switch-mode', choices=['direction','freeze'], default='direction')
    p.add_argument('--switch-ema', type=float, default=0.9)
    p.add_argument('--qa-threshold-teacher', type=float, default=0.0)
    p.add_argument('--qa-threshold-assistant', type=float, default=0.0)
    p.add_argument('--qa-temperature', type=float, default=1.0)
    p.add_argument('--qa-floor', type=float, default=0.05)
    p.add_argument('--feature-weight', type=float, default=0.1)
    p.add_argument('--alpha1', type=float, default=0.1)
    p.add_argument('--beta1', type=float, default=0.2)
    p.add_argument('--alpha2', type=float, default=0.1)
    p.add_argument('--beta2', type=float, default=0.2)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--warmup-epochs', type=int, default=5)
    p.add_argument('--ramp-epochs', type=int, default=10)
    p.add_argument('--kd-weight', type=float, default=0.1)
    p.add_argument('--texture-weight', type=float, default=1.0)
    p.add_argument('--auxiliary-weight', type=float, default=0.25)
    p.add_argument('--reverse-weight', type=float, default=0.1)
    p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--no-checkpoint', action='store_true')
    p.add_argument('--compile', action='store_true', help='Optional; test backend support on your OS first')
    p.add_argument('--deterministic', action='store_true')
    return p


def arguments() -> argparse.Namespace:
    return build_parser().parse_args()


def run(args: argparse.Namespace) -> None:
    for key in ('epochs','batch_size','accumulation','patch','save_every','ramp_epochs'):
        if getattr(args, key) < 1:
            raise ValueError(f'{key} must be positive')
    if args.workers < 0 or args.warmup_epochs < 0:
        raise ValueError('Invalid workers/warmup/validation fraction')
    if args.lr <= 0 or min(args.kd_weight,args.auxiliary_weight,args.reverse_weight,args.texture_weight) < 0:
        raise ValueError('Learning rate must be positive; loss weights nonnegative')
    if args.deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = not args.deterministic
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Install a CUDA PyTorch build.')
    roots = {'train':(args.hr_root,args.lr_root),
             'val':(args.val_hr_root,args.val_lr_root),
             'val2':(args.val2_hr_root,args.val2_lr_root)}
    missing = [name for name in ('hr_root','lr_root','val_hr_root','val_lr_root','val2_hr_root','val2_lr_root')
               if getattr(args,name) is None]
    if missing:
        raise ValueError(f'Set all six train/val/val2 LR/HR paths in config.yaml. Missing: {missing}. '
                         'Automatic splitting is disabled.')
    splits = {name:pair_images(high,low,args.scale) for name,(high,low) in roots.items()}
    # HR roots define the split. A common LR pool (and matching LR file) may
    # be reused across splits, as explicitly requested by the user.
    for first,second in (('train','val'),('train','val2'),('val','val2')):
        left = {str(high.resolve()).casefold() for _,high in splits[first]}
        right = {str(high.resolve()).casefold() for _,high in splits[second]}
        if left & right:
            raise ValueError(f'{first}/{second} overlap in HR files. Use distinct HR split roots.')
    train_pairs = splits['train']
    manifest = {name:[[str(l),str(h)] for l,h in entries] for name,entries in splits.items()}
    output = args.resume.parent if args.resume else args.output_root / args.dataset / f'x{args.scale}' / (args.run_name or datetime.now().strftime('%Y%m%d_%H%M%S'))
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f'Run directory is not empty: {output}. Use a new --run-name or --resume.')
    output.mkdir(parents=True, exist_ok=True)
    train_set = PairedImages(train_pairs,args.scale,args.patch,True,args.seed)
    eval_sets = {name:PairedImages(entries,args.scale,args.patch,False,args.seed) for name,entries in splits.items()}
    generator = torch.Generator()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=device.type=='cuda', worker_init_fn=seed_worker, generator=generator,
                              persistent_workers=False)
    eval_loaders = {name:DataLoader(dataset,batch_size=1,num_workers=args.workers,
                                   pin_memory=device.type=='cuda',shuffle=False) for name,dataset in eval_sets.items()}
    is_proposal = args.transfer_design == 'proposal'
    system = KDSystem(args.channels,args.scale,not args.no_checkpoint,args.transfer_design,
                      args.qa_threshold_teacher,args.qa_threshold_assistant,args.qa_temperature,args.qa_floor).to(device)
    criterion = (FigureDistillationLoss(args.kd_weight,args.auxiliary_weight,args.texture_weight) if args.transfer_design=='figure'
                 else DistillationLoss(args.kd_weight,args.auxiliary_weight,args.reverse_weight))
    if is_proposal:
        criterion = ProposalLoss(args.kd_weight,args.auxiliary_weight,args.reverse_weight,args.feature_weight,
                                 args.alpha1,args.beta1,args.alpha2,args.beta2)
    controller = (ProposalController(args.switch_ema) if is_proposal else FreezeController(args.switch_ema)).to(device)
    optimizer = torch.optim.Adam(system.parameters(),lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=args.lr*.01)
    use_amp = device.type=='cuda' and not args.no_amp
    scaler = torch.amp.GradScaler('cuda',enabled=use_amp)
    start, best = 0, -float('inf')
    if args.resume:
        state = torch.load(args.resume,map_location='cpu',weights_only=True)
        if state['args'].get('transfer_design','adaptive') != args.transfer_design:
            raise ValueError('Cannot resume with a different transfer_design; start a new run.')
        if state['args'].get('switch_mode','direction') != args.switch_mode or state['args'].get('switch_ema',0.9) != args.switch_ema:
            raise ValueError('Cannot resume with a different switch policy.')
        if 'switch_controller' in state:
            controller.load_state_dict(state['switch_controller'])
        if is_proposal:
            for key in ('qa_threshold_teacher','qa_threshold_assistant','qa_temperature','qa_floor',
                        'feature_weight','alpha1','beta1','alpha2','beta2'):
                if state['args'].get(key) != getattr(args,key):
                    raise ValueError(f'Resume proposal configuration mismatch: {key}')
        if state['args'].get('texture_weight',1.0) != args.texture_weight:
            raise ValueError('Cannot resume with a different texture_weight.')
        # Exact epoch-boundary resume requires identical data and optimization configuration.
        for key in ('channels','scale','seed','epochs','lr','batch_size','accumulation','patch',
                    'warmup_epochs','ramp_epochs','kd_weight','auxiliary_weight','reverse_weight'):
            if state['args'][key] != getattr(args,key):
                raise ValueError(f'Resume configuration mismatch: {key}')
        if state['manifest'] != manifest:
            raise ValueError('Resume train/val/val2 manifest differs. For an older two-split checkpoint, start a new run; old best scores are not comparable.')
        system.load_state_dict(state['system'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        start, best = state['epoch']+1, state['best_psnr']
        torch.set_rng_state(state['torch_rng'])
        if device.type=='cuda' and state['cuda_rng']:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    (output/'config.json').write_text(json.dumps(vars(args),default=str,indent=2),encoding='utf-8')
    (output/'split.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Parameters:', {k:sum(p.numel() for p in m.parameters()) for k,m in system.models.items()}, flush=True)
    print(f'Splits: { {name:len(dataset) for name,dataset in eval_sets.items()} }, output={output}',flush=True)
    forward_model = torch.compile(system) if args.compile else system
    for epoch in range(start,args.epochs):
        system.train()
        train_set.epoch = epoch
        generator.manual_seed(args.seed+epoch)
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(5,device=device)
        per_model_mse = torch.zeros(len(MODEL_LABELS),device=device)
        epoch_lr = optimizer.param_groups[0]['lr']
        observations = torch.zeros(2,3,device=device)
        freeze_counts = [0,0,0]
        optimizer_groups = 0
        group_expert = (False,False)
        ramp = max(0.,min(1.,(epoch-args.warmup_epochs+1)/args.ramp_epochs))
        for step,(lr,hr,_) in enumerate(train_loader):
            lr,hr = lr.to(device,non_blocking=True),hr.to(device,non_blocking=True)
            group_start = (step//args.accumulation)*args.accumulation
            group_samples = min(args.accumulation*args.batch_size, len(train_set)-group_start*args.batch_size)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                features,sr = forward_model(lr,hr)
                if is_proposal:
                    # One decision governs BOTH the loss and optimizer for this group.
                    if step % args.accumulation == 0:
                        group_expert = controller.decide(sr,hr,enabled=(args.switch_mode=='freeze' and epoch>=args.warmup_epochs))
                    loss,log = criterion(features,sr,hr,ramp,group_expert)
                else:
                    loss,log = criterion(features,sr,hr,ramp)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f'Nonfinite loss at epoch={epoch}, step={step}; try --no-amp')
            scaler.scale(loss*lr.shape[0]/group_samples).backward()
            totals += torch.stack(list(log.values()))*lr.shape[0]
            with torch.no_grad():
                if args.switch_mode=='freeze' and not is_proposal:
                    observations += switch_observations(sr,hr)
                per_model_mse += torch.stack([mse_each(sr[k].detach(),hr).mean() for k in MODEL_LABELS])*lr.shape[0]
            if (step+1)%args.accumulation==0 or step+1==len(train_loader):
                teacher_frozen,assistant_frozen=False,False
                if args.switch_mode=='freeze':
                    if is_proposal:
                        teacher_frozen,assistant_frozen=group_expert
                    else:
                        teacher_frozen,assistant_frozen=controller.decide(observations/group_samples, enabled=epoch>=args.warmup_epochs)
                    apply_freeze(system,teacher_frozen,assistant_frozen)
                    observations.zero_()
                freeze_counts[0]+=int(teacher_frozen)
                freeze_counts[1]+=int(assistant_frozen)
                freeze_counts[2]+=int(teacher_frozen or assistant_frozen)
                optimizer_groups+=1
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(system.parameters(),1.0,error_if_nonfinite=not use_amp)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            del loss,features,sr,log
        scheduler.step()
        save_due = (epoch+1) % args.save_every == 0
        evaluation = {
            name:validate(system,eval_loaders[name],device,args.scale,
                          output/'sr_images'/f'epoch_{epoch+1:04d}'/name if save_due else None)
            for name in ('train','val','val2')}
        score = evaluation['val2']['T']['psnr_rgb']
        improved = math.isfinite(score) and score > best
        if improved:
            best = score
        train_values = (totals/len(train_set)).cpu().tolist()
        model_losses = dict(zip(MODEL_LABELS,(per_model_mse/len(train_set)).cpu().tolist()))
        rows = []
        for split in ('train','val','val2'):
            for key,label in MODEL_LABELS.items():
                rows.append({'epoch':epoch+1,'split':split,'model':label, **evaluation[split][key],
                             'optimization_mse':model_losses[key] if split=='train' else '',
                             'total_loss':train_values[0],'main_loss':train_values[1],
                             'auxiliary_loss':train_values[2],'kd_loss':train_values[3],
                             'expert_fraction':train_values[4],'kd_ramp':ramp,'learning_rate':epoch_lr,
                             'teacher_freeze_fraction':freeze_counts[0]/optimizer_groups,
                             'assistant_freeze_fraction':freeze_counts[1]/optimizer_groups,
                             'shared_freeze_fraction':freeze_counts[2]/optimizer_groups})
        update_epoch_csv(output/'metrics.csv',epoch+1,rows)
        record = {'epoch':epoch+1,'ramp':ramp,'train_optimization':train_values,
                  'evaluation':evaluation,'lr':epoch_lr,'saved':save_due,'best_updated':improved,
                  'freeze_fractions':[n/optimizer_groups for n in freeze_counts]}
        if save_due or improved:
            state = {'system':system.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                     'scaler':scaler.state_dict(),'switch_controller':controller.state_dict(),'epoch':epoch,'best_psnr':best,
                     'best_metric':'val2/Student_output/psnr_rgb',
                     'args':json.loads(json.dumps(vars(args),default=str)), 'manifest':manifest,
                     'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
            if save_due:
                export_models(system,output,f'epoch_{epoch+1:04d}')
                save_atomic(state, output/f'training_epoch_{epoch+1:04d}.pt')
                save_atomic(state, output/'last.pt')
            if improved:
                export_models(system,output,'best_epoch')
                save_atomic(state,output/'best_epoch.pt')
                (output/'best_epoch.json').write_text(json.dumps(
                    {'epoch':epoch+1,'selection':'val2/Student_output/psnr_rgb',
                     'score':score,'val2':evaluation['val2']},indent=2),encoding='utf-8')
                # Only best val2 images are re-evaluated, avoiding GPU image caches.
                validate(system,eval_loaders['val2'],device,args.scale,output/'sr_images'/'best_epoch'/'val2')
        print(json.dumps(record),flush=True)


if __name__=='__main__':
    run(arguments())
