"""暗水印清洗工具 — 配置管理"""

import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DetectorConfig:
    synthid_enabled: bool = True

    synthid_confidence_threshold: float = 0.35
    c2pa_enabled: bool = True
    stablesig_enabled: bool = False
    treering_enabled: bool = False

@dataclass
class RemoverConfig:
    method: str = "diffusion"          # diffusion / spectral / frequency_subtract
    strength: float = 0.5              # 0.0 ~ 1.0
    diffusion_steps: int = 50
    diffusion_guidance_scale: float = 7.5
    preserve_face: bool = True
    preserve_text: bool = True

@dataclass
class MetadataConfig:
    strip_c2pa: bool = True
    strip_exif_ai: bool = True
    strip_xmp: bool = True
    strip_png_chunks: bool = True
    keep_camera_exif: bool = True     # 保留拍摄参数（非AI图片）

@dataclass
class QualityConfig:
    compute_psnr: bool = True
    compute_ssim: bool = True
    compute_lpips: bool = False

@dataclass
class AppConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    remover: RemoverConfig = field(default_factory=RemoverConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    output_dir: str = ""
    keep_backup: bool = True
    verbose: bool = False
    gpu_device: str = "cuda:0"

    def merge_cli(self, **kwargs):
        for k, v in kwargs.items():
            if v is not None and hasattr(self, k):
                setattr(self, k, v)