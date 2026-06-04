# Watermark Cleaner — AI 暗水印检测与清洗工具

> Phase 1 MVP | v0.1.0

## 功能

- **暗水印检测**：SynthID 频谱分析、C2PA 清单检测
- **暗水印清洗**：频域扰动（轻量）/ 扩散再生（需 SDXL 模型）
- **元数据剥离**：C2PA、EXIF AI 标签、XMP、PNG 文本块
- **质量评估**：PSNR / SSIM 指标报告

## 安装

```bash
pip install -r requirements.txt
pip install -e .
```

## 使用

### 检测暗水印

```bash
watermark-cleaner detect image.jpg
watermark-cleaner detect ./images/ -r --json
```

### 清洗暗水印

```bash
watermark-cleaner clean image.jpg -s 0.6
watermark-cleaner clean ./images/ -r -o ./cleaned/
```

### 剥离 AI 元数据

```bash
watermark-cleaner meta image.jpg
watermark-cleaner meta ./images/ -r -o ./nometa/
```

### 一键全链清理

```bash
watermark-cleaner all image.jpg -s 0.5
```

### 质量评估

```bash
watermark-cleaner quality original.jpg cleaned.jpg
```

## 项目结构

```
watermark_cleaner/
├── watermark_cleaner/
│   ├── __init__.py
│   ├── cli.py              # CLI 入口
│   ├── config.py           # 配置管理
│   ├── detectors/
│   │   ├── synthid.py      # SynthID 检测
│   │   └── c2pa.py         # C2PA 检测
│   ├── removers/
│   │   └── diffusion.py    # 扩散再生 + 频谱扰动清洗
│   ├── metadata/
│   │   └── cleaner.py      # 元数据剥离
│   ├── quality/
│   │   └── assessor.py     # 质量评估 (PSNR/SSIM)
│   └── utils/
│       └── io.py           # 文件 IO 工具
├── setup.py
├── requirements.txt
└── README.md
```

## 技术路线

| 清洗方法 | 依赖 | 适用目标 | GPU 要求 |
|---------|------|---------|---------|
| `frequency_perturb` | numpy | SynthID 等频域水印 | 无 |
| `sdxl_diffusion` | torch, diffusers | SynthID, StableSignature, TreeRing | RTX 3060+ |

## 合规声明

本工具仅用于安全研究、水印鲁棒性测试及合法隐私保护场景。
请遵守当地法律法规，不得用于欺诈、造假或其他非法用途。