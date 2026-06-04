"""C2PA 内容来源检测器

检测图像中的 C2PA (Coalition for Content Provenance and Authenticity) 清单。
C2PA 数据嵌入在 JPEG/PNG/AVIF/HEIF 等格式的元数据中。
"""

import struct
import json
from typing import Dict, Any, Optional


class C2PADetector:
    """C2PA 清单检测器
    
    支持的格式：
    - JPEG: XMP / JUMBF box
    - PNG: iTXt / tEXt chunk (c2pa 标识)
    - AVIF/HEIF: ISO BMFF box
    """

    C2PA_MARKERS = {
        "jpeg": [b"http://ns.adobe.com/c2pa/", b"c2pa", b"jumbf"],
        "png": [b"c2pa", b"http://ns.adobe.com/c2pa/"],
        "webp": [b"c2pa", b"http://ns.adobe.com/c2pa/"],
    }

    def detect(self, image_path: str) -> Dict[str, Any]:
        """检测图像中的 C2PA 数据"""
        result = {
            "has_c2pa": False,
            "format": None,
            "manifest": None,
            "raw_size": 0,
        }

        ext = image_path.lower().split(".")[-1] if "." in image_path else ""

        if ext in ("jpg", "jpeg"):
            result.update(self._detect_jpeg(image_path))
        elif ext == "png":
            result.update(self._detect_png(image_path))
        elif ext == "webp":
            result.update(self._detect_webp(image_path))
        else:
            result["format"] = ext
            result["note"] = f"格式 {ext} 暂不支持 C2PA 自动检测"

        return result

    def _detect_jpeg(self, path: str) -> Dict[str, Any]:
        result = {"format": "jpeg", "has_c2pa": False, "manifest": None, "raw_size": 0}
        try:
            with open(path, "rb") as f:
                data = f.read()

            pos = 0
            while pos < len(data) - 4:
                if data[pos] == 0xFF and data[pos + 1] == 0xE1:
                    seg_len = struct.unpack(">H", data[pos + 2:pos + 4])[0]
                    seg_data = data[pos + 4:pos + 2 + seg_len]
                    for marker in self.C2PA_MARKERS["jpeg"]:
                        if marker in seg_data:
                            result["has_c2pa"] = True
                            result["raw_size"] = len(seg_data)
                            result["raw_preview"] = self._extract_manifest_preview(seg_data)
                            break
                    if result["has_c2pa"]:
                        break
                    pos += 2 + seg_len
                else:
                    pos += 1
        except Exception as e:
            result["error"] = str(e)
        return result

    def _detect_png(self, path: str) -> Dict[str, Any]:
        result = {"format": "png", "has_c2pa": False, "manifest": None, "raw_size": 0}
        try:
            with open(path, "rb") as f:
                sig = f.read(8)
                if sig != b"\x89PNG\r\n\x1a\n":
                    result["error"] = "Invalid PNG signature"
                    return result

                while True:
                    chunk_len_bytes = f.read(4)
                    if len(chunk_len_bytes) < 4:
                        break
                    chunk_len = struct.unpack(">I", chunk_len_bytes)[0]
                    chunk_type = f.read(4)
                    chunk_data = f.read(chunk_len)
                    f.read(4)  # CRC

                    chunk_type_str = chunk_type.decode("ascii", errors="ignore")
                    if chunk_type_str in ("iTXt", "tEXt", "zTXt"):
                        for marker in self.C2PA_MARKERS["png"]:
                            if marker in chunk_data:
                                result["has_c2pa"] = True
                                result["raw_size"] = len(chunk_data)
                                result["chunk_type"] = chunk_type_str
                                result["raw_preview"] = self._extract_manifest_preview(chunk_data)
                                break
                    if result["has_c2pa"]:
                        break
                    if chunk_type == b"IEND":
                        break
        except Exception as e:
            result["error"] = str(e)
        return result

    def _detect_webp(self, path: str) -> Dict[str, Any]:
        result = {"format": "webp", "has_c2pa": False, "manifest": None, "raw_size": 0}
        try:
            with open(path, "rb") as f:
                data = f.read()
            for marker in self.C2PA_MARKERS["webp"]:
                idx = data.find(marker)
                if idx >= 0:
                    result["has_c2pa"] = True
                    result["raw_size"] = min(4096, len(data) - idx)
                    result["raw_preview"] = self._extract_manifest_preview(
                        data[idx:idx + 4096]
                    )
                    break
        except Exception as e:
            result["error"] = str(e)
        return result

    def _extract_manifest_preview(self, data: bytes) -> Optional[Dict]:
        """从二进制数据中提取可读的 JSON manifest 预览"""
        try:
            text = data.decode("utf-8", errors="ignore")
            brace_depth = 0
            json_start = -1
            for i, ch in enumerate(text):
                if ch == "{":
                    if brace_depth == 0:
                        json_start = i
                    brace_depth += 1
                elif ch == "}":
                    brace_depth -= 1
                    if brace_depth == 0 and json_start >= 0:
                        candidate = text[json_start:i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            json_start = -1
        except Exception:
            pass
        return None