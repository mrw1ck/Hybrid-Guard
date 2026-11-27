"""QR Code metadata extraction for Hybrid-Guard (Approach 1: Combined QR + URL).

Extracts structural and protocol-level metadata from QR code images without network calls:
- Version (1-40)
- Grid Dimension (21-177)
- Total Modules, Black Modules, White Modules, Dot Density
- Error Correction Level (L, M, Q, H mapped to numeric 0, 1, 2, 3)
- Mask Pattern (0-7)
- Unused Error Correction Capacity (UEC)
- Payload Length (number of characters)
- Payload Capacity Ratio
- Module Pixel Size and Image Aspect Ratio
"""

import io
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image
import zxingcpp

# Error correction mapping to numeric codes matching Dataset.csv
ECC_MAP: Dict[str, int] = {
    "L": 0,
    "M": 1,
    "Q": 2,
    "H": 3,
}

QR_FEATURE_NAMES = [
    "QRVersion",
    "QRDimension",
    "QRTotalModules",
    "QRBlackModules",
    "QRWhiteModules",
    "QRDotDensity",
    "QRECCLevel",
    "QRDataMask",
    "QRUnusedECC",
    "QRPayloadLength",
    "QRCapacityRatio",
    "QRModulePixelSize",
    "QRImageAspectRatio",
    "HasValidQR",
]

DEFAULT_QR_METADATA: Dict[str, float] = {
    "QRVersion": 3.0,
    "QRDimension": 29.0,
    "QRTotalModules": 841.0,
    "QRBlackModules": 424.0,
    "QRWhiteModules": 406.0,
    "QRDotDensity": 0.513,
    "QRECCLevel": 0.0,
    "QRDataMask": 3.0,
    "QRUnusedECC": 1.0,
    "QRPayloadLength": 0.0,
    "QRCapacityRatio": 0.048,
    "QRModulePixelSize": 6.0,
    "QRImageAspectRatio": 1.0,
    "HasValidQR": 0.0,
}


def extract_qr_metadata(
    image_input: Union[str, Path, bytes, io.BytesIO, Image.Image]
) -> Dict[str, Any]:
    """Parse QR code image and return structural metadata and decoded payload.

    Args:
        image_input: File path, bytes, BytesIO, or PIL Image instance.

    Returns:
        dict containing:
          - All 14 numeric QR features matching Dataset.csv
          - 'url': decoded string (or empty if failed)
          - 'valid': bool
          - 'error': error message if any
    """
    result_dict: Dict[str, Any] = dict(DEFAULT_QR_METADATA)
    result_dict["url"] = ""
    result_dict["valid"] = False
    result_dict["error"] = None

    try:
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input)
        elif isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, io.BytesIO):
            img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            result_dict["error"] = f"Unsupported image input type: {type(image_input)}"
            return result_dict

        img_gray = img.convert("L")
        width, height = img.width, img.height
        aspect_ratio = float(width) / float(height) if height else 1.0
        result_dict["QRImageAspectRatio"] = round(aspect_ratio, 4)

        # Decode QR code
        barcode = zxingcpp.read_barcode(img, formats=zxingcpp.BarcodeFormat.QRCode)
        if not barcode or not barcode.valid:
            # Fallback retry on grayscale image
            barcode = zxingcpp.read_barcode(img_gray, formats=zxingcpp.BarcodeFormat.QRCode)

        if not barcode or not barcode.valid:
            result_dict["error"] = "Không tìm thấy hoặc không thể giải mã mã QR hợp lệ trong ảnh."
            return result_dict

        decoded_text = barcode.text or ""
        payload_len = len(decoded_text)
        extra = barcode.extra or {}

        # 1. Version & Dimension
        version_val = int(extra.get("Version", 1)) if "Version" in extra else 1
        dim_val = version_val * 4 + 17
        total_mods = dim_val * dim_val

        # 2. ECC Level & Mask
        ec_str = str(extra.get("ECLevel", "L")).upper()
        ecc_level = ECC_MAP.get(ec_str, 0)
        data_mask = int(extra.get("DataMask", 0)) if "DataMask" in extra else 0
        uec_val = float(extra.get("UEC", 1.0)) if "UEC" in extra else 1.0

        # 3. Bounding box & Module sampling
        pos = barcode.position
        pts = [(p.x, p.y) for p in [pos.top_left, pos.top_right, pos.bottom_right, pos.bottom_left]]
        min_x = max(0, min(p[0] for p in pts))
        max_x = min(width, max(p[0] for p in pts))
        min_y = max(0, min(p[1] for p in pts))
        max_y = min(height, max(p[1] for p in pts))

        box_w = max(1, max_x - min_x)
        box_h = max(1, max_y - min_y)
        module_pixel_size = float(box_w) / float(dim_val)

        # Count black and white modules across the grid
        arr = np.array(img_gray)
        black_cnt = 0
        for r in range(dim_val):
            for c in range(dim_val):
                py = int(min_y + (r + 0.5) * (box_h / dim_val))
                px = int(min_x + (c + 0.5) * (box_w / dim_val))
                if 0 <= py < arr.shape[0] and 0 <= px < arr.shape[1]:
                    if arr[py, px] < 128:
                        black_cnt += 1

        white_cnt = max(0, total_mods - black_cnt)
        dot_density = black_cnt / total_mods if total_mods else 0.5
        capacity_ratio = payload_len / total_mods if total_mods else 0.0

        result_dict.update(
            {
                "QRVersion": float(version_val),
                "QRDimension": float(dim_val),
                "QRTotalModules": float(total_mods),
                "QRBlackModules": float(black_cnt),
                "QRWhiteModules": float(white_cnt),
                "QRDotDensity": float(dot_density),
                "QRECCLevel": float(ecc_level),
                "QRDataMask": float(data_mask),
                "QRUnusedECC": float(uec_val),
                "QRPayloadLength": float(payload_len),
                "QRCapacityRatio": float(capacity_ratio),
                "QRModulePixelSize": float(module_pixel_size),
                "QRImageAspectRatio": float(aspect_ratio),
                "HasValidQR": 1.0,
                "url": decoded_text,
                "valid": True,
                "error": None,
                "ec_level_str": ec_str,
            }
        )
        return result_dict

    except Exception as exc:
        result_dict["error"] = f"Lỗi khi phân tích ảnh QR: {exc}"
        return result_dict

