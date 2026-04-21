from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _normalize_foreground(binary_img: np.ndarray) -> np.ndarray:
    """Ensure foreground digit is white on black background."""
    white_pixels = np.count_nonzero(binary_img > 0)
    if white_pixels > binary_img.size // 2:
        return cv2.bitwise_not(binary_img)
    return binary_img


def _crop_to_content(binary_img: np.ndarray, min_content_pixels: int = 20) -> np.ndarray:
    """Crop to digit bounding box and reject near-empty images."""
    points = cv2.findNonZero(binary_img)
    if points is None:
        raise ValueError("No digit detected in the image.")
    x, y, w, h = cv2.boundingRect(points)
    cropped = binary_img[y : y + h, x : x + w]
    if np.count_nonzero(cropped) < min_content_pixels:
        raise ValueError("Image appears too empty/noisy to contain a digit.")
    return cropped


def preprocess_digit_image(image_path: str | Path) -> np.ndarray:
    """
    Convert a user image to MNIST-like float tensor input.
    Returns ndarray with shape (1, 1, 28, 28), values in [0, 1].
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = _normalize_foreground(binary)
    cropped = _crop_to_content(binary)

    h, w = cropped.shape
    target_side = 20
    scale = target_side / max(h, w)
    resized = cv2.resize(cropped, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_offset = (28 - resized.shape[0]) // 2
    x_offset = (28 - resized.shape[1]) // 2
    canvas[y_offset : y_offset + resized.shape[0], x_offset : x_offset + resized.shape[1]] = resized

    normalized = canvas.astype(np.float32) / 255.0
    return normalized[None, None, :, :]
