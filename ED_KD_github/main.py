"""Open this file in an IDE and press Run. Edit config.yaml for all settings."""
from __future__ import annotations
from pathlib import Path
from multiprocessing import freeze_support
from configuration import load_config

CONFIG_PATH = Path(__file__).resolve().with_name('config.yaml')


def main(config_path: Path = CONFIG_PATH) -> None:
    mode, args = load_config(config_path)
    print(f'Configuration: {config_path.resolve()} | mode={mode}', flush=True)
    if mode == 'train':
        from train import run
    else:
        from infer import run
    run(args)


if __name__ == '__main__':
    freeze_support()
    main()
