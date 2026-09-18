"""Strict YAML configuration loader; relative paths use the YAML directory."""
from __future__ import annotations
import argparse
import math
from pathlib import Path, PureWindowsPath
from typing import Any
import torch
import yaml
from train import build_parser


class UniqueSafeLoader(yaml.SafeLoader):
    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError('YAML keys must be strings')
            if key in result:
                raise ValueError(f'Duplicate YAML key: {key} (line {key_node.start_mark.line + 1}). '
                                 f'Keep only one {key}: entry in this section; for save_every use 50.')
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def resolve_path(value: str, parent: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() or PureWindowsPath(value).is_absolute():
        return path
    return parent / path


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be a YAML mapping')
    return value


def load_config(path: Path) -> tuple[str, argparse.Namespace]:
    path = path.resolve()
    with path.open(encoding='utf-8-sig') as stream:
        config = require_mapping(yaml.load(stream, Loader=UniqueSafeLoader), 'config')
    unknown = set(config) - {'mode', 'train', 'inference'}
    if unknown:
        raise ValueError(f'Unknown configuration sections: {sorted(unknown)}')
    mode = config.get('mode', 'train')
    if mode not in ('train', 'inference'):
        raise ValueError('mode must be train or inference')
    if mode == 'inference':
        values = require_mapping(config.get('inference'), 'inference').copy()
        if set(values) - {'checkpoint', 'input', 'output', 'device'}:
            raise ValueError('Unknown inference option')
        for key in ('checkpoint', 'input', 'output'):
            value = values.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'inference.{key} must be a nonempty path')
            values[key] = resolve_path(value, path.parent)
        device = values.get('device', 'auto')
        if not isinstance(device, str):
            raise ValueError('inference.device must be a string')
        values['device'] = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        return mode, argparse.Namespace(**values)
    parser = build_parser()
    values = vars(parser.parse_args([]))
    supplied = require_mapping(config.get('train'), 'train')
    unknown = set(supplied) - set(values)
    if unknown:
        raise ValueError(f'Unknown train options: {sorted(unknown)}')
    # Use the training parser's declared types and choices; no sys.argv parsing.
    actions = {action.dest: action for action in parser._actions}
    for key, value in supplied.items():
        action = actions[key]
        if value is None:
            if action.default is not None:
                raise ValueError(f'train.{key} cannot be null')
        elif isinstance(action.default, bool):
            if type(value) is not bool:
                raise ValueError(f'train.{key} must be true or false, without quotes')
        elif action.type in (int, float):
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise ValueError(f'train.{key} must be numeric')
            if action.type is int and isinstance(value, float) and not value.is_integer():
                raise ValueError(f'train.{key} must be an integer')
            value = action.type(value)
            if not math.isfinite(value):
                raise ValueError(f'train.{key} must be finite')
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f'train.{key} must be a nonempty string')
        if action.choices is not None and value not in action.choices:
            raise ValueError(f'train.{key}: choose from {action.choices}')
        values[key] = value
    if values['device'] == 'auto':
        values['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    for key in ('hr_root', 'lr_root', 'val_hr_root', 'val_lr_root', 'val2_hr_root', 'val2_lr_root', 'output_root', 'resume'):
        if values[key] is not None:
            values[key] = resolve_path(str(values[key]), path.parent)
    return mode, argparse.Namespace(**values)
