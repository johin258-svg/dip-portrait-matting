# -*- coding: utf-8 -*-
"""
Step 1 — 图像预处理 (Part A)
  - 缩放: 限制最大尺寸 1024px
  - Gamma 校正: 调整整体明暗
  - CLAHE: 自适应直方图均衡化 (LAB空间, 仅L通道)
  - 高斯/中值滤波: 去噪
"""

import cv2
import numpy as np


def resize_image(image, max_size=1024):
    """限制最大边长，保持纵横比"""
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image
    s = max_size / max(h, w)
    return cv2.resize(image, (int(w * s), int(h * s)), cv2.INTER_AREA)


def gamma_correction(image, gamma=1.0):
    """Gamma 校正: >1 变暗, <1 变亮"""
    t = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, t)


def clahe_enhancement(image, clip=2.0, grid=(8, 8)):
    """CLAHE 对比度增强 (LAB空间 L通道)"""
    if len(image.shape) == 2:
        return cv2.createCLAHE(clipLimit=clip, tileGridSize=grid).apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def gaussian_denoise(image, ksize=5):
    """高斯滤波去噪"""
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def median_denoise(image, ksize=5):
    """中值滤波去噪"""
    return cv2.medianBlur(image, ksize)
