# app_config.py — 設定の保存/読み込み
import json
from pathlib import Path

CONFIG_PATH = Path("config.json")

DEFAULT_CONFIG = {
    "camera_index": 2,           # あなたの元コード既定値
    "swap_sec": 15.0,            # 二人モードの入れ替え秒
    "max_scale": 4.5,            # 表示上の最大倍率クランプ
    "debug_overlay": 1           # 0/1
}

def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return {**DEFAULT_CONFIG, **(data or {})}
        except Exception:
            return dict(DEFAULT_CONFIG)
    return dict(DEFAULT_CONFIG)

def save_config(cfg: dict):
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
