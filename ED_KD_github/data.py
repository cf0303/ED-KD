from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

EXTENSIONS = {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}


def image_index(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f'Image directory does not exist: {root}')
    result: dict[str, Path] = {}
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            key = path.relative_to(root).with_suffix('').as_posix().casefold()
            if key in result:
                raise ValueError(f'Duplicate image stem: {result[key]} / {path}')
            result[key] = path
    if not result:
        raise ValueError(f'No RGB image files in {root}')
    return result


def pair_images(hr_root: Path, lr_root: Path, scale: int) -> list[tuple[Path, Path]]:
    hr, lr = image_index(hr_root), image_index(lr_root)
    # The HR root defines the selected split. The LR root may contain additional
    # images from other splits; never add these to the selected training pairs.
    # Prefer exact relative stem. If folder layouts differ, use a unique
    # filename stem (extension ignored); never guess among duplicate LR names.
    by_name: dict[str, list[Path]] = {}
    for candidate in lr.values():
        by_name.setdefault(candidate.stem.casefold(), []).append(candidate)
    pairs: list[tuple[Path, Path]] = []
    missing: list[str] = []
    for key, high in sorted(hr.items()):
        low = lr.get(key)
        if low is None:
            candidates = by_name.get(high.stem.casefold(), [])
            if len(candidates) > 1:
                raise ValueError(f'Ambiguous LR filename for {high}: {candidates}. '
                                 'Use matching relative folders or unique LR filenames.')
            if not candidates:
                missing.append(key)
                continue
            low = candidates[0]
        pairs.append((low, high))
    if missing:
        raise ValueError(f'{len(missing)} HR images have no matching LR. '
                         f'Missing LR stems: {missing[:8]}. HR images are not silently dropped.')
    selected = {low for low,_ in pairs}
    extra = [path for path in lr.values() if path not in selected]
    print(f'[Pairing] HR={hr_root}: pairs={len(pairs)}, ignored extra LR={len(extra)}', flush=True)
    for low, high in pairs:
        with Image.open(low) as l, Image.open(high) as h:
            if h.size != (l.width*scale, l.height*scale):
                raise ValueError(f'x{scale} mismatch: {low} {l.size}, {high} {h.size}. '
                                 'Supply LR generated at the requested scale; no silent resizing is applied.')
    return pairs


def read_rgb(path: Path) -> Tensor:
    with Image.open(path) as image:
        a = np.array(image.convert('RGB'), dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).contiguous()  # [3,H,W]


class PairedImages(Dataset[tuple[Tensor, Tensor, str]]):
    def __init__(self, pairs: list[tuple[Path, Path]], scale: int, patch: int = 64,
                 training: bool = True, seed: int = 42) -> None:
        self.pairs, self.scale, self.patch, self.training, self.seed = pairs, scale, patch, training, seed
        self.epoch = 0
        if training:
            for low, _ in pairs:
                with Image.open(low) as im:
                    if min(im.size) < patch:
                        raise ValueError(f'{low}: smaller than LR patch {patch}; reduce --patch')

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str]:
        low, high = self.pairs[index]
        lr, hr = read_rgb(low), read_rgb(high)
        if self.training:
            rng = random.Random(self.seed + self.epoch*1_000_003 + index)
            p, s = self.patch, self.scale
            y, x = rng.randrange(lr.shape[1]-p+1), rng.randrange(lr.shape[2]-p+1)
            lr, hr = lr[:, y:y+p, x:x+p], hr[:, y*s:(y+p)*s, x*s:(x+p)*s]
            for axis in (1, 2):
                if rng.random() < .5:
                    lr, hr = lr.flip(axis), hr.flip(axis)
            if rng.random() < .5:
                lr, hr = lr.transpose(1, 2), hr.transpose(1, 2)
        return lr.contiguous(), hr.contiguous(), low.name


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
