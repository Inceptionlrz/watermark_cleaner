"""IO 工具函数"""

import os
import shutil
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image
import numpy as np

SUPPORTED_IMAGE = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
SUPPORTED_VIDEO = {".mp4", ".mov", ".webm", ".avi"}


def is_image(path: str) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_IMAGE


def is_video(path: str) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_VIDEO


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def save_image(array: np.ndarray, path: str, quality: int = 95):
    img = Image.fromarray(array.astype(np.uint8))
    img.save(path, quality=quality)


def backup_original(path: str, backup_dir: str) -> str:
    os.makedirs(backup_dir, exist_ok=True)
    fname = Path(path).name
    stem, ext = os.path.splitext(fname)
    backup_path = os.path.join(backup_dir, f"{stem}_original{ext}")
    shutil.copy2(path, backup_path)
    return backup_path


def ensure_output_path(input_path: str, output_dir: str, suffix: str = "_cleaned") -> str:
    fname = Path(input_path).name
    stem, ext = os.path.splitext(fname)
    out_name = f"{stem}{suffix}{ext}"
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, out_name)
    return os.path.join(os.path.dirname(input_path), out_name)


def collect_files(paths: list, recursive: bool = False) -> list:
    files = []
    for p in paths:
        p = os.path.abspath(p)
        if os.path.isfile(p):
            if is_image(p):
                files.append(p)
        elif os.path.isdir(p):
            for root, _, filenames in os.walk(p):
                for fn in filenames:
                    fp = os.path.join(root, fn)
                    if is_image(fp):
                        files.append(fp)
                if not recursive:
                    break
    return sorted(set(files))