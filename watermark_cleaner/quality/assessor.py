"""质量评估模块

计算处理后图像的质量指标：
- PSNR（峰值信噪比）
- SSIM（结构相似性）
- 生成对比报告
"""

import numpy as np
from typing import Dict, Any
from PIL import Image


def _rgb_to_grayscale(img: np.ndarray) -> np.ndarray:
    """RGB 转灰度"""
    return 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]


def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """计算 PSNR"""
    mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def compute_ssim(original: np.ndarray, processed: np.ndarray) -> float:
    """计算 SSIM
    
    基于 Wang et al. 2004 的简化实现
    """
    if original.ndim == 3:
        original = _rgb_to_grayscale(original)
    if processed.ndim == 3:
        processed = _rgb_to_grayscale(processed)

    orig = original.astype(np.float64)
    proc = processed.astype(np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = _gaussian_blur(orig, sigma=1.5)
    mu2 = _gaussian_blur(proc, sigma=1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _gaussian_blur(orig ** 2, sigma=1.5) - mu1_sq
    sigma2_sq = _gaussian_blur(proc ** 2, sigma=1.5) - mu2_sq
    sigma12 = _gaussian_blur(orig * proc, sigma=1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(np.mean(ssim_map))


def _gaussian_blur(img: np.ndarray, sigma: float, kernel_size: int = 11) -> np.ndarray:
    """简单高斯模糊"""
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    kernel /= kernel.sum()

    h, w = img.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")

    result = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            result[i, j] = np.sum(padded[i:i + kh, j:j + kw] * kernel)
    return result


def assess(original_path: str, processed_path: str) -> Dict[str, Any]:
    """评估处理前后的图像质量"""
    orig = np.array(Image.open(original_path).convert("RGB"))
    proc = np.array(Image.open(processed_path).convert("RGB"))

    if orig.shape != proc.shape:
        proc = np.array(Image.open(processed_path).resize(
            (orig.shape[1], orig.shape[0]), Image.LANCZOS
        ))

    psnr = compute_psnr(orig, proc)
    ssim = compute_ssim(orig, proc)

    grade = "A" if psnr >= 38 and ssim >= 0.97 else \
            "B" if psnr >= 35 and ssim >= 0.95 else \
            "C" if psnr >= 30 and ssim >= 0.90 else "D"

    return {
        "psnr": round(psnr, 2),
        "ssim": round(ssim, 4),
        "grade": grade,
        "pass": psnr >= 35 and ssim >= 0.95,
    }


def format_report(report: Dict[str, Any]) -> str:
    """格式化质量报告"""
    lines = [
        "=" * 50,
        "质量评估报告",
        "=" * 50,
        f"PSNR:   {report['psnr']:.2f} dB",
        f"SSIM:   {report['ssim']:.4f}",
        f"评级:   {report['grade']}",
        f"通过:   {'是' if report['pass'] else '否'}",
    ]
    if "note" in report:
        lines.append(f"说明:   {report['note']}")
    lines.append("=" * 50)
    return "\n".join(lines)