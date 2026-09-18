from __future__ import annotations
import argparse
from pathlib import Path
import torch
from data import image_index, read_rgb
from models import CNCAN
from train import save_rgb


def main() -> None:
    p = argparse.ArgumentParser(description='Standalone S/AT/T/L/M SR inference; no teacher or HR required')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    run(args)


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    state = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model = CNCAN(**state['config']).to(args.device).eval()
    model.load_state_dict(state['state_dict'])
    paths = list(image_index(args.input).values()) if args.input.is_dir() else [args.input]
    for path in paths:
        lr = read_rgb(path).unsqueeze(0).to(args.device)  # [1,3,H,W]
        sr = model(lr)  # [1,3,sH,sW]
        relative = path.relative_to(args.input) if args.input.is_dir() else Path(path.name)
        destination = args.output/relative.with_suffix('.png')
        save_rgb(sr[0],destination)
        print(destination)


if __name__=='__main__':
    main()
