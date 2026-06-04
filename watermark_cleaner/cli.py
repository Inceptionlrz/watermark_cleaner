"""CLI 主入口

Phase 1 MVP 命令：
  watermark-cleaner detect <file>        # 检测水印
  watermark-cleaner clean <file>         # 清洗暗水印
  watermark-cleaner meta <file>          # 剥离元数据
  watermark-cleaner all <file>           # 一键全链清理
  watermark-cleaner quality <orig> <proc> # 质量评估
"""

import argparse
import sys
import os
import time
import json
from pathlib import Path

# 确保包可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from watermark_cleaner.config import AppConfig
from watermark_cleaner.detectors.synthid import SynthIDDetector
from watermark_cleaner.detectors.c2pa import C2PADetector
from watermark_cleaner.removers.diffusion import DiffusionRemover
from watermark_cleaner.metadata.cleaner import MetadataCleaner
from watermark_cleaner.quality.assessor import assess, format_report
from watermark_cleaner.utils.io import (
    load_image, save_image, backup_original,
    ensure_output_path, collect_files, is_image,
)


def cmd_detect(args, config: AppConfig):
    """检测暗水印"""
    files = collect_files([args.input], recursive=args.recursive)
    if not files:
        print("未找到支持的图片文件")
        return

    print(f"检测 {len(files)} 个文件...\n")

    synthid_detector = SynthIDDetector(config.detector.synthid_confidence_threshold)
    c2pa_detector = C2PADetector()

    results = []
    for fp in files:
        r = {"file": fp, "detections": []}

        # SynthID 检测
        if config.detector.synthid_enabled:
            img = load_image(fp)
            sd = synthid_detector.detect(img)
            if sd["has_watermark"]:
                r["detections"].append({
                    "type": "SynthID",
                    "confidence": round(sd["confidence"], 3),
                    "channel_scores": {k: round(v, 3) for k, v in sd["channel_scores"].items()},
                })

        # C2PA 检测
        if config.detector.c2pa_enabled:
            cd = c2pa_detector.detect(fp)
            if cd["has_c2pa"]:
                r["detections"].append({
                    "type": "C2PA",
                    "format": cd.get("format", ""),
                    "raw_size": cd.get("raw_size", 0),
                    "manifest_preview": cd.get("raw_preview"),
                })

        results.append(r)

        status = "FOUND" if r["detections"] else "CLEAN"
        types = ", ".join(d["type"] for d in r["detections"])
        conf = ", ".join(
            f"{d.get('confidence', 'N/A')}" for d in r["detections"] if "confidence" in d
        )
        print(f"  [{status}] {fp}")
        if r["detections"]:
            print(f"         水印类型: {types}  置信度: {conf}")

    # 汇总
    detected = sum(1 for r in results if r["detections"])
    print(f"\n检测完成: {detected}/{len(results)} 个文件含水印")

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))


def cmd_clean(args, config: AppConfig):
    """清洗暗水印"""
    files = collect_files([args.input], recursive=args.recursive)
    if not files:
        print("未找到支持的图片文件")
        return

    strength = args.strength if args.strength is not None else config.remover.strength
    method = args.method or config.remover.method

    print(f"清洗模式: {method}  强度: {strength}")
    print(f"处理 {len(files)} 个文件...\n")

    remover = DiffusionRemover(
        strength=strength,
        method="frequency_perturb" if method == "diffusion" else method,
        model_path=args.model_path,
        device=config.gpu_device,
    )

    output_dir = args.output or config.output_dir or ""
    backup_dir = os.path.join(output_dir, ".backups") if output_dir else ""

    cleaned = []
    for i, fp in enumerate(files):
        print(f"  [{i+1}/{len(files)}] {fp}")
        t0 = time.time()

        # 备份
        if config.keep_backup and backup_dir:
            backup_original(fp, backup_dir)

        # 清洗
        img = load_image(fp)
        cleaned_img = remover.remove(img)

        # 保存
        out_path = ensure_output_path(fp, output_dir, "_cleaned")
        save_image(cleaned_img, out_path)

        elapsed = time.time() - t0
        print(f"          -> {out_path} ({elapsed:.1f}s)")
        cleaned.append(out_path)

    print(f"\n清洗完成: {len(cleaned)} 个文件")


def cmd_meta(args, config: AppConfig):
    """剥离元数据"""
    files = collect_files([args.input], recursive=args.recursive)
    if not files:
        print("未找到支持的图片文件")
        return

    print(f"剥离 {len(files)} 个文件的元数据...\n")

    cleaner = MetadataCleaner(
        strip_c2pa=config.metadata.strip_c2pa,
        strip_exif_ai=config.metadata.strip_exif_ai,
        strip_xmp=config.metadata.strip_xmp,
        strip_png_chunks=config.metadata.strip_png_chunks,
        keep_camera_exif=config.metadata.keep_camera_exif,
    )

    output_dir = args.output or config.output_dir or ""
    cleaned = []

    for i, fp in enumerate(files):
        out_path = ensure_output_path(fp, output_dir, "_nometa")
        result = cleaner.clean(fp, out_path)
        actions = ", ".join(result.get("actions", []))
        print(f"  [{i+1}/{len(files)}] {fp} -> {out_path}")
        if actions:
            print(f"          操作: {actions}")
        cleaned.append(out_path)

    print(f"\n元数据剥离完成: {len(cleaned)} 个文件")


def cmd_all(args, config: AppConfig):
    """一键全链清理"""
    files = collect_files([args.input], recursive=args.recursive)
    if not files:
        print("未找到支持的图片文件")
        return

    print("一键全链清理: 检测 -> 暗水印清洗 -> 元数据剥离 -> 质量评估\n")
    print(f"处理 {len(files)} 个文件...\n")

    synthid = SynthIDDetector(config.detector.synthid_confidence_threshold)
    c2pa = C2PADetector()
    remover = DiffusionRemover(strength=args.strength or config.remover.strength)
    meta_cleaner = MetadataCleaner()

    output_dir = args.output or config.output_dir or ""

    for i, fp in enumerate(files):
        print(f"--- [{i+1}/{len(files)}] {Path(fp).name} ---")

        # Step 1: 检测
        img = load_image(fp)
        sd = synthid.detect(img)
        cd = c2pa.detect(fp)
        has_synthid = sd["has_watermark"]
        has_c2pa = cd["has_c2pa"]
        print(f"  检测: SynthID={'Y' if has_synthid else 'N'} "
              f"(置信度:{sd['confidence']:.2f})  C2PA={'Y' if has_c2pa else 'N'}")

        # Step 2: 清洗暗水印
        if has_synthid:
            cleaned_img = remover.remove(img)
            temp_path = ensure_output_path(fp, output_dir, "_temp_cleaned")
            save_image(cleaned_img, temp_path)
            print(f"  暗水印清洗: 完成")
        else:
            temp_path = fp
            print(f"  暗水印清洗: 跳过（未检测到）")

        # Step 3: 剥离元数据
        final_path = ensure_output_path(fp, output_dir, "_fully_cleaned")
        meta_result = meta_cleaner.clean(temp_path, final_path)
        print(f"  元数据剥离: {', '.join(meta_result.get('actions', ['无']))}")

        # Step 4: 质量评估
        quality = assess(fp, final_path)
        print(f"  质量: PSNR={quality['psnr']}dB  SSIM={quality['ssim']}  评级={quality['grade']}")

        # 清理临时文件
        if has_synthid and temp_path != fp and os.path.exists(temp_path):
            os.remove(temp_path)

        print()

    print("全链清理完成")


def cmd_quality(args, config: AppConfig):
    """质量评估"""
    report = assess(args.original, args.processed)
    print(format_report(report))


def main():
    parser = argparse.ArgumentParser(
        prog="watermark-cleaner",
        description="AI 暗水印检测与清洗工具 v0.1.0 (Phase 1 MVP)",
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # detect
    p_detect = subparsers.add_parser("detect", help="检测暗水印")
    p_detect.add_argument("input", help="输入文件或目录")
    p_detect.add_argument("-r", "--recursive", action="store_true", help="递归处理目录")
    p_detect.add_argument("--json", action="store_true", help="JSON 格式输出")

    # clean
    p_clean = subparsers.add_parser("clean", help="清洗暗水印")
    p_clean.add_argument("input", help="输入文件或目录")
    p_clean.add_argument("-o", "--output", help="输出目录")
    p_clean.add_argument("-s", "--strength", type=float, help="清洗强度 (0.0-1.0)")
    p_clean.add_argument("-m", "--method", help="清洗方法 (frequency_perturb/sdxl_diffusion)")
    p_clean.add_argument("--model-path", help="SDXL 模型路径")
    p_clean.add_argument("-r", "--recursive", action="store_true")

    # meta
    p_meta = subparsers.add_parser("meta", help="剥离 AI 元数据")
    p_meta.add_argument("input", help="输入文件或目录")
    p_meta.add_argument("-o", "--output", help="输出目录")
    p_meta.add_argument("-r", "--recursive", action="store_true")

    # all
    p_all = subparsers.add_parser("all", help="一键全链清理")
    p_all.add_argument("input", help="输入文件或目录")
    p_all.add_argument("-o", "--output", help="输出目录")
    p_all.add_argument("-s", "--strength", type=float, help="清洗强度")
    p_all.add_argument("-r", "--recursive", action="store_true")

    # quality
    p_quality = subparsers.add_parser("quality", help="质量评估")
    p_quality.add_argument("original", help="原始图片路径")
    p_quality.add_argument("processed", help="处理后图片路径")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    config = AppConfig()
    dispatcher = {
        "detect": cmd_detect,
        "clean": cmd_clean,
        "meta": cmd_meta,
        "all": cmd_all,
        "quality": cmd_quality,
    }

    dispatcher[args.command](args, config)


if __name__ == "__main__":
    main()