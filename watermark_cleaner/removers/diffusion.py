"""扩散再生暗水印清洗器

基于扩散模型的暗水印去除。
核心思路：将水印图像通过扩散模型从带噪潜变量重新生成干净图像，
水印信号在扩散-去噪过程中被破坏。

支持两种模式：
1. 完整扩散再生（需要 SDXL 模型）
2. 轻量频域扰动（无需大模型，CPU 可用）
"""

import numpy as np
from typing import Dict, Any, Optional
from PIL import Image


class DiffusionRemover:
    """扩散再生水印清洗器

    Phase 1 MVP 内置轻量频域扰动模式。
    完整 SDXL 模式需用户配置模型路径后启用。
    """

    def __init__(
        self,
        strength: float = 0.5,
        method: str = "frequency_perturb",
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Args:
            strength: 清洗强度 0.0~1.0
            method: "frequency_perturb" (轻量) / "sdxl_diffusion" (完整)
            model_path: SDXL 模型路径（仅 sdxl_diffusion 模式）
            device: 计算设备
        """
        self.strength = min(1.0, max(0.0, strength))
        self.method = method
        self.model_path = model_path
        self.device = device

    def remove(self, image: np.ndarray) -> np.ndarray:
        """去除图像中的暗水印"""
        if self.method == "frequency_perturb":
            return self._frequency_perturb_removal(image)
        elif self.method == "sdxl_diffusion":
            return self._sdxl_diffusion_removal(image)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _frequency_perturb_removal(self, image: np.ndarray) -> np.ndarray:
        """轻量频域扰动清洗

        多策略频域清洗：
        1. 自适应频带衰减 - 识别并压制异常频段幅度
        2. 相位扰动 - 破坏频域编码的水印信号
        3. 强度自适应 - 平衡清洗效果与画质
        """
        h, w = image.shape[:2]
        result = image.astype(np.float64).copy()
        max_radius = min(h // 2, w // 2)

        for ch in range(3):
            ch_data = result[:, :, ch]
            fft = np.fft.fft2(ch_data)
            fft_shifted = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shifted)
            phase = np.angle(fft_shifted)

            y_idx, x_idx = np.ogrid[:h, :w]
            dist = np.sqrt((y_idx - h // 2) ** 2 + (x_idx - w // 2) ** 2)

            # === 策略1: 自适应频带衰减 ===
            # 将频谱按半径分环，对每个环计算统计，衰减异常环
            n_rings = 64
            ring_means = np.zeros(n_rings)
            ring_stds = np.zeros(n_rings)

            ring_width = max_radius / n_rings
            for i in range(1, n_rings):
                r_low = i * ring_width
                r_high = (i + 1) * ring_width
                ring_mask = (dist >= r_low) & (dist < r_high)
                ring_mags = magnitude[ring_mask]
                if len(ring_mags) > 0:
                    ring_means[i] = np.median(ring_mags)
                    ring_stds[i] = np.std(ring_mags)

            # 用滑动窗口检测异常环（相邻环比异常跳变）
            window = 5
            ring_anomaly = np.ones(n_rings)  # 1.0 = 正常, < 1.0 = 需衰减
            for i in range(window, n_rings - window):
                neighbors = list(range(i - window, i)) + list(range(i + 1, i + window + 1))
                neighbor_mean = np.mean(ring_means[neighbors])
                neighbor_std = np.std(ring_means[neighbors])
                if neighbor_mean > 0 and neighbor_std > 0:
                    z = (ring_means[i] - neighbor_mean) / (neighbor_std + 1e-10)
                    if z > 1.5:  # 异常峰
                        ring_anomaly[i] = 1.0 / (1.0 + (z - 1.5) * self.strength * 0.8)

            # 应用衰减（平滑过渡避免环状伪影）
            attenuation_map = np.ones_like(magnitude)
            for i in range(1, n_rings):
                r_low = i * ring_width
                r_high = (i + 1) * ring_width
                ring_mask = (dist >= r_low) & (dist < r_high)
                attenuation_map[ring_mask] = ring_anomaly[i]

            # 平滑衰减图以避免尖锐边界
            pad = 3
            atten_padded = np.pad(attenuation_map, pad, mode='edge')
            atten_smooth = np.zeros_like(attenuation_map)
            for dy in range(-pad, pad + 1):
                for dx in range(-pad, pad + 1):
                    atten_smooth += atten_padded[pad + dy:pad + dy + h, pad + dx:pad + dx + w]
            attenuation_map = atten_smooth / ((2 * pad + 1) ** 2)

            magnitude_attenuated = magnitude * attenuation_map

            # === 策略2: 低频相位扰动 ===
            low_radius = int(max_radius * 0.12 * self.strength)
            if low_radius > 0:
                low_mask = dist <= low_radius
                n_pixels = low_mask.sum()
                phase_noise = np.random.randn(n_pixels) * np.pi * 0.15 * self.strength
                phase[low_mask] += phase_noise

            # === 策略3: 中频轻度随机化 ===
            mid_inner = int(max_radius * 0.08)
            mid_outer = int(max_radius * 0.35)
            mid_mask = (dist >= mid_inner) & (dist < mid_outer)
            if mid_mask.sum() > 0:
                n_mid = mid_mask.sum()
                # 轻微幅度抖动
                jitter = 1.0 + np.random.uniform(-0.1, 0.1, n_mid) * self.strength
                magnitude_attenuated[mid_mask] *= jitter
                # 轻微相位噪声
                mid_phase_noise = np.random.randn(n_mid) * np.pi * 0.05 * self.strength
                phase[mid_mask] += mid_phase_noise

            # 重建
            fft_perturbed = magnitude_attenuated * np.exp(1j * phase)
            fft_restored = np.fft.ifftshift(fft_perturbed)
            result[:, :, ch] = np.real(np.fft.ifft2(fft_restored))

        result = np.clip(result, 0, 255)
        return result.astype(np.uint8)

    def _sdxl_diffusion_removal(self, image: np.ndarray) -> np.ndarray:
        """完整的 SDXL 扩散再生清洗

        需要安装 diffusers 和下载 SDXL 模型。
        当前返回降级到频域扰动模式的结果，并给出提示。
        """
        try:
            import torch
            from diffusers import StableDiffusionXLImg2ImgPipeline

            if self.model_path is None:
                raise ValueError("SDXL model_path required")

            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
            ).to(self.device)

            pil_image = Image.fromarray(image)
            result = pipe(
                prompt="",
                image=pil_image,
                strength=self.strength,
                guidance_scale=7.5,
                num_inference_steps=50,
            ).images[0]

            return np.array(result)
        except ImportError:
            print("[WARNING] diffusers/torch 未安装，降级到频域扰动模式")
            return self._frequency_perturb_removal(image)
        except Exception as e:
            print(f"[WARNING] SDXL 加载失败 ({e})，降级到频域扰动模式")
            return self._frequency_perturb_removal(image)


class SpectralPerturbRemover:
    """频谱扰动清洗器（UnMarker 路线）

    直接修改图像频谱幅度来破坏水印编码。
    不同于频域扰动，这里使用对抗优化策略。
    """

    def __init__(self, strength: float = 0.5, iterations: int = 10):
        self.strength = min(1.0, max(0.0, strength))
        self.iterations = iterations

    def remove(self, image: np.ndarray) -> np.ndarray:
        """通过频谱幅度扰动去除水印"""
        result = image.astype(np.float32).copy()

        for ch in range(3):
            ch_data = result[:, :, ch]
            fft = np.fft.fft2(ch_data)
            fft_shifted = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shifted)
            phase = np.angle(fft_shifted)

            # 计算幅度谱的局部统计
            h, w = magnitude.shape

            # 对每个频率点，用其邻域的均值替换异常幅度值
            perturbed_mag = magnitude.copy()
            window = max(3, int(min(h, w) * 0.02))
            half_w = window // 2

            for _ in range(self.iterations):
                for i in range(half_w, h - half_w, window):
                    for j in range(half_w, w - half_w, window):
                        patch = magnitude[
                            i - half_w:i + half_w + 1,
                            j - half_w:j + half_w + 1
                        ]
                        patch_mean = np.mean(patch)
                        patch_std = np.std(patch)
                        if patch_std > 0:
                            # 将偏离均值超过 N 个标准差的点拉回
                            N = 3.0 - self.strength * 2.5  # strength越大N越小，攻击越强
                            threshold = patch_mean + N * patch_std
                            patch_clipped = np.minimum(patch, threshold)
                            perturbed_mag[
                                i - half_w:i + half_w + 1,
                                j - half_w:j + half_w + 1
                            ] = patch_clipped

            # 添加轻微随机噪声
            noise_level = 0.02 * self.strength
            perturbed_mag += np.random.randn(*perturbed_mag.shape) * noise_level * np.mean(magnitude)

            # 重建
            fft_perturbed = perturbed_mag * np.exp(1j * phase)
            fft_restored = np.fft.ifftshift(fft_perturbed)
            result[:, :, ch] = np.real(np.fft.ifft2(fft_restored))

        result = np.clip(result, 0, 255)
        return result.astype(np.uint8)