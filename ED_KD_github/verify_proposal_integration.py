import sys,tempfile,csv,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch,numpy as np,yaml
from PIL import Image
from main import main

torch.set_num_threads(2)
with tempfile.TemporaryDirectory() as temp:
    root=Path(temp)
    (root/'lr').mkdir()
    for split,count in (('train',5),('val',1),('val2',1)):
        (root/split).mkdir()
        for i in range(count):
            name=f'{split}_{i}.png'
            im=Image.fromarray(np.random.default_rng(i+len(split)).integers(0,256,(10,12,3),dtype=np.uint8))
            im.save(root/'lr'/name); im.resize((24,20)).save(root/split/name)
    config=yaml.safe_load((Path(__file__).resolve().parent/'config.yaml').read_text())
    config['train'].update(hr_root=str(root/'train'),lr_root=str(root/'lr'),val_hr_root=str(root/'val'),val_lr_root=str(root/'lr'),val2_hr_root=str(root/'val2'),val2_lr_root=str(root/'lr'),output_root=str(root/'out'),run_name='integration',epochs=3,save_every=2,channels=8,patch=8,scale=2,workers=0,device='cpu',warmup_epochs=0,ramp_epochs=1,accumulation=2)
    p=root/'config.yaml'; p.write_text(yaml.safe_dump(config)); main(p)
    out=root/'out'/'IJRR2017'/'x2'/'integration'
    with (out/'metrics.csv').open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    assert len(rows)==45
    for split,count in (('train',5),('val',1),('val2',1)):
        for model in ('Teacher_output','AssistantTeacher_output','Student_output'):
            assert len(list((out/'sr_images'/'epoch_0002'/split/model).glob('*.png')))==count
    state=torch.load(out/'last.pt',weights_only=True)
    assert state['epoch']==1 and 'switch_controller' in state
    from infer import run as infer_run
    import argparse
    infer_run(argparse.Namespace(checkpoint=out/'Student_output_epoch_0002.pt',input=root/'lr',output=root/'inferred',device='cpu'))
    assert len(list((root/'inferred').glob('*.png')))==7
    config['train']['resume']=str(out/'last.pt'); p.write_text(yaml.safe_dump(config)); main(p)
    with (out/'metrics.csv').open(encoding='utf-8-sig') as f: replay=list(csv.DictReader(f))
    assert len(replay)==45
    for before,after in zip(rows,replay):
        for key in ('loss_mse','psnr_rgb','ssim_rgb','teacher_freeze_fraction','assistant_freeze_fraction','shared_freeze_fraction'):
            assert abs(float(before[key])-float(after[key]))<1e-6,(key,before,after)
    print('PASS: proposal YAML training, full split images, standalone inference, CSV, incomplete accumulation group, exact CPU resume replay')
