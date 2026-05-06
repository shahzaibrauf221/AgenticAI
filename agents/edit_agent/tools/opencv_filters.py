from __future__ import annotations

from pathlib import Path

FILTER_PRESETS: dict[str, dict] = {
    "darker": {"brightness": 0.5, "contrast": 1.0},
    "brighter": {"brightness": 1.6, "contrast": 1.1},
    "sepia": {"sepia": True},
    "grayscale": {"grayscale": True},
    "warm": {"hue_shift": 15},
    "cool": {"hue_shift": -20},
    "vintage": {"brightness": 0.85, "sepia": True},
    "vivid": {"saturation": 1.5, "contrast": 1.2},
}


def apply_filter_to_image(image_path: Path, filter_name: str, output_path: Path) -> bool:
    preset = FILTER_PRESETS.get(filter_name, {})
    try:
        import cv2
        import numpy as np

        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError("cv2 could not read image")

        if preset.get("grayscale"):
            img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

        if preset.get("sepia"):
            k = np.array(
                [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]
            )
            img = np.clip(cv2.transform(img, k), 0, 255).astype(np.uint8)

        if brightness := preset.get("brightness"):
            img = cv2.convertScaleAbs(img, alpha=brightness, beta=0)

        if contrast := preset.get("contrast"):
            if contrast != 1.0:
                img = cv2.convertScaleAbs(img, alpha=contrast, beta=128 * (1 - contrast))

        if saturation := preset.get("saturation"):
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if hue_shift := preset.get("hue_shift"):
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img)
        return True
    except Exception:
        try:
            from PIL import Image, ImageEnhance

            img = Image.open(image_path).convert("RGB")
            if brightness := preset.get("brightness"):
                img = ImageEnhance.Brightness(img).enhance(brightness)
            if contrast := preset.get("contrast"):
                img = ImageEnhance.Contrast(img).enhance(contrast)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path)
            return True
        except Exception:
            return False

