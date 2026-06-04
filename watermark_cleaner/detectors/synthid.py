"""SynthID 暗水印检测器

基于全图频域分析的 SynthID 水印检测。
SynthID 在图像频域（主要是低频区域）嵌入水印信号，
通过分析全图频谱中的异常模式来检测水印存在。
"""

import numpy as np
from typing import Tuple, Dict, Any


class SynthIDDetector:
    """SynthID 水印检测器

    检测原理：
    1. 对全图做 FFT 得到频谱
    2. 径向分环分析各频段的幅度统计特征
    3. 自然图像频谱遵循 1/f 幂律分布，水印嵌入会破坏这一规律
    4. 通过幂律拟合残差 + 高频尾部分析判定水印存在
    5. 多通道加权（G > R > B，匹配 SynthID 嵌入强度分布）
    """

    CHANNEL_WEIGHTS = {"G": 1.0, "R": 0.85, "B": 0.70}

    # 径向分析：将频谱分成 N 个同心环
    N_RINGS = 32

    def __init__(self, confidence_threshold: float = 0.35):
        self.threshold = confidence_threshold

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        检测图像中的 SynthID 水印

        Args:
            image: RGB 图像数组 (H, W, 3)，值范围 0-255

        Returns:
            dict: {
                "has_watermark": bool,
                "confidence": float (0-1),
                "channel_scores": {"R": float, "G": float, "B": float},
                "spectral_anomaly_map": np.ndarray
            }
        """
        h, w = image.shape[:2]

        channel_scores = {}
        anomaly_maps = []

        for ch_idx, ch_name in enumerate(["R", "G", "B"]):
            ch_data = image[:, :, ch_idx].astype(np.float64)
            score, anomaly = self._analyze_full_spectrum(ch_data)
            channel_scores[ch_name] = score
            anomaly_maps.append(anomaly)

        # 多通道加权融合
        weighted_score = sum(
            channel_scores[ch] * self.CHANNEL_WEIGHTS[ch]
            for ch in ["R", "G", "B"]
        ) / sum(self.CHANNEL_WEIGHTS.values())

        return {
            "has_watermark": weighted_score >= self.threshold,
            "confidence": float(weighted_score),
            "channel_scores": channel_scores,
            "spectral_anomaly_map": anomaly_maps[1],  # G 通道异常图
        }

    def _analyze_full_spectrum(self, data: np.ndarray) -> Tuple[float, np.ndarray]:
        """全图频谱分析：径向分环 + 幂律拟合残差"""
        fft = np.fft.fft2(data)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)
        h, w = magnitude.shape
        ch, cw = h // 2, w // 2
        max_radius = min(ch, cw) - 1

        # 径向分环：计算每个环的平均幅度
        ring_means = np.zeros(self.N_RINGS)
        ring_stds = np.zeros(self.N_RINGS)
        ring_radii = np.zeros(self.N_RINGS)

        # 预计算距离矩阵（只算四分之一，利用对称性）
        y_indices, x_indices = np.ogrid[:h, :w]
        dist = np.sqrt((y_indices - ch) ** 2 + (x_indices - cw) ** 2)

        ring_width = max_radius / self.N_RINGS
        for i in range(self.N_RINGS):
            r_low = i * ring_width
            r_high = (i + 1) * ring_width
            mask = (dist >= r_low) & (dist < r_high)
            if mask.sum() > 0:
                ring_means[i] = magnitude[mask].mean()
                ring_stds[i] = magnitude[mask].std()
                ring_radii[i] = (r_low + r_high) / 2

        # 跳过 DC 分量（r=0）和无效环
        valid = ring_means > 1e-10
        if valid.sum() < 5:
            return 0.0, np.zeros((h, w), dtype=np.float32)

        # 幂律拟合：log(magnitude) = alpha * log(radius) + beta
        log_r = np.log(ring_radii[valid])
        log_m = np.log(ring_means[valid])

        # 跳过第一个环（含 DC 附近），从第2环开始
        start = max(1, valid.argmax()) if valid.any() else 0
        fit_r = log_r[start:]
        fit_m = log_m[start:]

        if len(fit_r) < 3:
            return 0.0, np.zeros((h, w), dtype=np.float32)

        # 线性回归拟合 alpha (斜率)
        A = np.vstack([fit_r, np.ones_like(fit_r)]).T
        alpha, beta = np.linalg.lstsq(A, fit_m, rcond=None)[0]

        # 计算拟合残差（在 log 空间）
        predicted = alpha * log_r + beta
        residuals = log_m - predicted
        residual_rms = np.sqrt(np.mean(residuals[start:] ** 2))

        # 特征1：幂律拟合残差 RMS（自然图像拟合好，水印图像差）
        score_residual = min(1.0, residual_rms / 0.8)

        # 特征2：低频段（前 25% 环）的局部频谱方差
        low_band_end = max(1, self.N_RINGS // 4)
        low_means = ring_means[1:low_band_end]
        if len(low_means) > 1:
            low_cv = np.std(low_means) / (np.mean(low_means) + 1e-10)
            score_low_cv = min(1.0, low_cv / 0.5)
        else:
            score_low_cv = 0.0

        # 特征3：高频尾部的异常陡峭度
        # 水印清洗或嵌入会导致高频尾部形态异常
        high_start = max(self.N_RINGS * 3 // 4, 2)
        high_vals = ring_means[high_start:]
        if len(high_vals) > 2:
            # 自然图像高频遵循平滑衰减，水印操作会引入不自然的拐点
            high_diffs = np.diff(high_vals)
            high_roughness = np.std(high_diffs) / (np.abs(np.mean(high_diffs)) + 1e-10)
            score_high = min(1.0, high_roughness / 3.0)
        else:
            score_high = 0.0

        # 特征4：频域峰度异常
        # 水印嵌入会在某些频段引入尖锐峰值
        all_mag = magnitude[dist > 1].flatten()  # 排除 DC
        if len(all_mag) > 0:
            kurtosis = self._kurtosis(all_mag)
            score_kurt = min(1.0, max(0.0, (kurtosis - 3.0) / 15.0))
        else:
            score_kurt = 0.0

        # 综合评分：加权融合四个特征
        score = (
            0.35 * score_residual +
            0.25 * score_low_cv +
            0.20 * score_high +
            0.20 * score_kurt
        )

        # 生成异常图（log 幅度残差的 2D 可视化）
        anomaly_map = np.zeros((h, w), dtype=np.float32)
        for i in range(self.N_RINGS):
            r_low = i * ring_width
            r_high = (i + 1) * ring_width
            mask = (dist >= r_low) & (dist < r_high)
            if valid[i]:
                residual_fraction = abs(log_m[i] - predicted[i]) / max(abs(log_m[i]), 1e-10)
                anomaly_map[mask] = float(min(1.0, residual_fraction))

        return float(score), anomaly_map

    @staticmethod
    def _kurtosis(x: np.ndarray) -> float:
        """计算峰度 (Fisher definition, normal=0)"""
        n = len(x)
        if n < 4:
            return 0.0
        mean = np.mean(x)
        m2 = np.mean((x - mean) ** 2)
        m4 = np.mean((x - mean) ** 4)
        if m2 < 1e-15:
            return 0.0
        return m4 / (m2 ** 2)