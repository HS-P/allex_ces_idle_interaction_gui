from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

def _atomic_write_text(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)

class ConfigError(RuntimeError):
    pass

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ConfigError("config root must be a JSON object")
    if cfg.get("version") != 1:
        raise ConfigError(f"Unsupported config version: {cfg.get('version')}")
    # minimal required fields
    for k in ("device", "ros", "layers", "behavior", "layer"):
        if k not in cfg:
            raise ConfigError(f"Missing required field: {k}")
    if not isinstance(cfg["layers"], list) or len(cfg["layers"]) != 9:
        raise ConfigError("cfg.layers must be a list of 9 layer objects")

    # ----------------
    # Backward/robust defaults
    # ----------------
    # NOTE: Users may already have a mapping.json created by an older version.
    # We fill in safe defaults rather than forcing them to delete their config.

    # device.grab default = True
    if isinstance(cfg.get("device"), dict):
        cfg["device"].setdefault("grab", True)
        cfg["device"].setdefault("reconnect_period_ms", 500)

    # layer indicator keys: F1~F9 -> layer 0~8
    if isinstance(cfg.get("layer"), dict):
        cfg["layer"].setdefault("initial", 0)
        cfg["layer"].setdefault("suppress_base_action_on_indicator", True)
        _default_ind = {
            "KEY_F1": 0,
            "KEY_F2": 1,
            "KEY_F3": 2,
            "KEY_F4": 3,
            "KEY_F5": 4,
            "KEY_F6": 5,
            "KEY_F7": 6,
            "KEY_F8": 7,
            "KEY_F9": 8,
        }
        ind = cfg["layer"].setdefault("indicator_keys", {})
        if not isinstance(ind, dict):
            ind = {}
            cfg["layer"]["indicator_keys"] = ind
        for k, v in _default_ind.items():
            ind.setdefault(k, v)

    # robot selection defaults (optional)
    # NOTE: not exclusive by default (do NOT suppress other actions unless user opts in).
    if isinstance(cfg.get("robot_selection"), dict):
        cfg["robot_selection"].setdefault("enabled", False)
        cfg["robot_selection"].setdefault("suppress_base_action", False)
        cfg["robot_selection"].setdefault("keys", {})
    else:
        cfg["robot_selection"] = {"enabled": False, "suppress_base_action": False, "keys": {}}


    # behavior defaults
    if isinstance(cfg.get("behavior"), dict):
        cfg["behavior"].setdefault("suppress_base_action_on_chord", True)
    return cfg

def save_config(path: str, cfg: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _atomic_write_text(path, json.dumps(cfg, indent=2, ensure_ascii=False))
