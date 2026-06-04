"""元数据清洗器

剥离图像中的 AI 相关元数据：
- C2PA 内容来源清单
- EXIF AI 标签
- XMP DigitalSourceType
- PNG 文本块
"""

import struct
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List


class MetadataCleaner:
    """AI 元数据清洗器"""

    def __init__(
        self,
        strip_c2pa: bool = True,
        strip_exif_ai: bool = True,
        strip_xmp: bool = True,
        strip_png_chunks: bool = True,
        keep_camera_exif: bool = True,
    ):
        self.strip_c2pa = strip_c2pa
        self.strip_exif_ai = strip_exif_ai
        self.strip_xmp = strip_xmp
        self.strip_png_chunks = strip_png_chunks
        self.keep_camera_exif = keep_camera_exif

    def clean(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """清洗图像元数据"""
        ext = Path(input_path).suffix.lower()

        if ext in (".jpg", ".jpeg"):
            return self._clean_jpeg(input_path, output_path)
        elif ext == ".png":
            return self._clean_png(input_path, output_path)
        else:
            # 其他格式直接复制
            shutil.copy2(input_path, output_path)
            return {
                "format": ext,
                "actions": ["copied_as_is"],
                "note": f"格式 {ext} 暂不支持深度元数据清洗，文件已复制"
            }

    def _clean_jpeg(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """清洗 JPEG 元数据"""
        actions = []

        with open(input_path, "rb") as f:
            data = f.read()

        if data[:2] != b"\xff\xd8":
            return {"error": "Invalid JPEG"}

        # 解析 JPEG 段
        segments = self._parse_jpeg_segments(data)
        cleaned_segments = []
        ai_segments_removed = []

        for seg_type, seg_data, seg_raw in segments:
            if seg_type == 0xE1:  # APP1 (EXIF/XMP)
                if self.strip_xmp and b"http://ns.adobe.com/xap/" in seg_data:
                    actions.append("removed:XMP")
                    ai_segments_removed.append("XMP")
                    continue
                if (self.strip_exif_ai or self.strip_c2pa) and b"Exif" in seg_data[:4]:
                    if self.keep_camera_exif and not self._has_ai_exif(seg_data):
                        cleaned_segments.append(seg_raw)
                    else:
                        cleaned_segments.append(self._strip_ai_exif(seg_data))
                        actions.append("cleaned:EXIF_AI_tags")
                    continue
            elif seg_type == 0xE2 and self.strip_c2pa:  # APP2 (ICC/其他)
                actions.append("removed:APP2")
                ai_segments_removed.append("APP2")
                continue

            cleaned_segments.append(seg_raw)

        # 重建 JPEG
        result = bytearray()
        result.extend(data[:2])  # SOI
        for seg in cleaned_segments:
            result.extend(seg)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(result)

        return {
            "format": "jpeg",
            "actions": actions,
            "segments_removed": ai_segments_removed,
        }

    def _parse_jpeg_segments(self, data: bytes) -> List:
        """解析 JPEG 段"""
        segments = []
        pos = 2  # 跳过 SOI

        while pos < len(data) - 2:
            marker = data[pos]
            if marker != 0xFF:
                pos += 1
                continue

            seg_type = data[pos + 1]
            if seg_type in (0xD8, 0xD9):  # SOI/EOI
                pos += 2
                continue
            if seg_type == 0xDA:  # SOS (图像数据开始)
                segments.append((seg_type, data[pos + 2:], data[pos:]))
                break

            if pos + 4 > len(data):
                break

            seg_len = struct.unpack(">H", data[pos + 2:pos + 4])[0]
            seg_end = pos + 2 + seg_len

            if seg_end > len(data):
                break

            seg_data = data[pos + 4:seg_end]
            seg_raw = data[pos:seg_end]
            segments.append((seg_type, seg_data, seg_raw))
            pos = seg_end

        return segments

    def _has_ai_exif(self, exif_data: bytes) -> bool:
        """检查 EXIF 是否包含 AI 相关标签"""
        ai_markers = [
            b"GeneratedBy", b"AI", b"Stable Diffusion",
            b"Midjourney", b"OpenAI", b"DALL-E", b"Firefly",
            b"c2pa", b"C2PA", b"DigitalSourceType",
        ]
        return any(m in exif_data for m in ai_markers)

    def _strip_ai_exif(self, exif_data: bytes) -> bytes:
        """移除 EXIF 中的 AI 标签，保留相机参数"""
        # 简化处理：移除已知 AI 标签的 IFD 条目
        cleaned = bytearray(exif_data)
        ai_tags = [
            b"UserComment", b"ImageDescription", b"Software",
            b"Artist", b"Copyright", b"Make", b"Model"  # 对于 AI 图，这些可能是假的
        ]
        # MVP: 替换可疑文本为空白
        for tag in ai_tags:
            idx = cleaned.find(tag)
            while idx >= 0:
                # 简单清零该 tag 的值域
                end = cleaned.find(b"\x00\x00", idx)
                if end > idx:
                    for i in range(idx, min(end + 2, len(cleaned))):
                        cleaned[i] = 0
                idx = cleaned.find(tag, idx + 1)
        return bytes(cleaned)

    def _clean_png(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """清洗 PNG 元数据"""
        actions = []

        with open(input_path, "rb") as f:
            data = f.read()

        if data[:8] != b"\x89PNG\r\n\x1a\n":
            return {"error": "Invalid PNG"}

        result = bytearray(data[:8])  # PNG signature
        pos = 8
        ai_chunks_removed = []

        while pos < len(data):
            if pos + 8 > len(data):
                break
            chunk_len = struct.unpack(">I", data[pos:pos + 4])[0]
            chunk_type = data[pos + 4:pos + 8]
            chunk_end = pos + 12 + chunk_len

            if chunk_end > len(data):
                break

            chunk_type_str = chunk_type.decode("ascii", errors="ignore")
            keep = True

            # 移除 AI 相关的辅助块
            if self.strip_png_chunks and chunk_type_str in ("iTXt", "tEXt", "zTXt"):
                chunk_data = data[pos + 8:pos + 8 + chunk_len]
                ai_keywords = [b"c2pa", b"C2PA", b"AI", b"Stable", b"Midjourney",
                              b"parameters", b"prompt", b"Negative prompt"]
                if any(kw in chunk_data for kw in ai_keywords):
                    keep = False
                    actions.append(f"removed:PNG_{chunk_type_str}")
                    ai_chunks_removed.append(chunk_type_str)

            if keep:
                result.extend(data[pos:chunk_end])

            if chunk_type == b"IEND":
                result.extend(data[pos:chunk_end])
                break

            pos = chunk_end

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(result)

        return {
            "format": "png",
            "actions": actions,
            "chunks_removed": ai_chunks_removed,
        }