"""Synthetic damage augmentation for training the damage-detection U-Net.

Each function takes a uint8 BGR ndarray and a uint8 single-channel mask
(0 = clean, 255 = damaged) and returns the modified (image, mask) pair.
Callers compose them randomly to produce varied training examples.
"""

from __future__ import annotations

import random

import cv2
import numpy as np


def add_scratches(
    img: np.ndarray,
    mask: np.ndarray,
    n_min: int = 3,
    n_max: int = 25,
    thickness_range: tuple[int, int] = (1, 4),
) -> tuple[np.ndarray, np.ndarray]:
    """Straight and curved line scratches simulating physical damage."""
    h, w = img.shape[:2]
    for _ in range(random.randint(n_min, n_max)):
        x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
        x2, y2 = random.randint(0, w - 1), random.randint(0, h - 1)
        t = random.randint(*thickness_range)
        brightness = random.randint(190, 255)
        color = (brightness, brightness, brightness)
        cv2.line(img, (x1, y1), (x2, y2), color, t)
        cv2.line(mask, (x1, y1), (x2, y2), 255, t + 2)
    return img, mask


def add_dust_noise(
    img: np.ndarray,
    mask: np.ndarray,
    density: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """Salt-and-pepper noise representing dust particles."""
    h, w = img.shape[:2]
    n = int(h * w * density)
    for _ in range(n):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        val = 255 if random.random() > 0.5 else 0
        img[y, x] = (val, val, val)
        mask[y, x] = 255
    return img, mask


def add_stains(
    img: np.ndarray,
    mask: np.ndarray,
    n_min: int = 1,
    n_max: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Elliptical water / age stains with semi-transparent blending."""
    h, w = img.shape[:2]
    for _ in range(random.randint(n_min, n_max)):
        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)
        rx = random.randint(15, 90)
        ry = random.randint(15, 90)
        angle = random.randint(0, 180)
        color = tuple(random.randint(140, 210) for _ in range(3))
        alpha = random.uniform(0.15, 0.45)
        overlay = img.copy()
        cv2.ellipse(overlay, (cx, cy), (rx, ry), angle, 0, 360, color, -1)
        img[:] = cv2.addWeighted(img, 1.0 - alpha, overlay, alpha, 0)
        cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 255, -1)
    return img, mask


def add_fading(
    img: np.ndarray,
    mask: np.ndarray,
    alpha_range: tuple[float, float] = (0.4, 0.8),
) -> tuple[np.ndarray, np.ndarray]:
    """Global fading: desaturates and washes out the image."""
    alpha = random.uniform(*alpha_range)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img[:] = cv2.addWeighted(img, alpha, gray_bgr, 1.0 - alpha, 50)
    # Fading is global — mark whole image as damaged with soft value
    mask[:] = np.maximum(mask, 80)
    return img, mask


def add_crack_texture(
    img: np.ndarray,
    mask: np.ndarray,
    n_min: int = 2,
    n_max: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-segment crack patterns (polylines) for old photo emulsion cracks."""
    h, w = img.shape[:2]
    for _ in range(random.randint(n_min, n_max)):
        n_pts = random.randint(3, 7)
        pts = [(random.randint(0, w - 1), random.randint(0, h - 1)) for _ in range(n_pts)]
        t = random.randint(1, 3)
        brightness = random.randint(30, 100)
        color = (brightness, brightness, brightness)
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, color, t)
            cv2.line(mask, a, b, 255, t + 2)
    return img, mask


def add_gaussian_blur_patch(
    img: np.ndarray,
    mask: np.ndarray,
    n_min: int = 1,
    n_max: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Blurred rectangular patches simulating lens smudges or partial focus loss."""
    h, w = img.shape[:2]
    for _ in range(random.randint(n_min, n_max)):
        x1 = random.randint(0, w - 20)
        y1 = random.randint(0, h - 20)
        x2 = min(x1 + random.randint(20, w // 3), w)
        y2 = min(y1 + random.randint(20, h // 3), h)
        ksize = random.choice([11, 21, 31])
        img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (ksize, ksize), 0)
        mask[y1:y2, x1:x2] = 255
    return img, mask


AUGMENTATIONS = [
    add_scratches,
    add_dust_noise,
    add_stains,
    add_crack_texture,
    add_gaussian_blur_patch,
]


def apply_random_damage(
    img_rgb: np.ndarray,
    min_ops: int = 2,
    max_ops: int = 4,
    fading_prob: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a random subset of damage functions to a uint8 RGB image.

    Returns
    -------
    damaged_rgb : uint8 ndarray (H, W, 3)
    mask        : uint8 ndarray (H, W)  — 255 = damaged, 0 = clean
    """
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    ops = random.sample(AUGMENTATIONS, k=random.randint(min_ops, max_ops))
    for op in ops:
        img, mask = op(img, mask)

    if random.random() < fading_prob:
        img, mask = add_fading(img, mask)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mask
