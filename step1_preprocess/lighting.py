# -*- coding: utf-8 -*-
"""
Step 1 — 光照校正 (Part A)
  - Homomorphic Filter: 频域同态滤波, 增强暗部同时压缩亮部
  - Illumination Correction: 形态学背景估计 + 除法校正
  - Retinex SSR: 单尺度 Retinex, 分离光照和反射
"""

import cv2
import numpy as np


def homomorphic_filter(image, r=40, hg=2.0, lg=0.5):
    """
    同态滤波 — 频域增强暗部、压缩亮部。
    r:  截止频率, hg: 高频增益, lg: 低频增益
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    g = gray.astype(np.float64) + 1
    lf = np.fft.fft2(np.log(g))
    fs = np.fft.fftshift(lf)
    rows, cols = gray.shape

    y, x = np.ogrid[:rows, :cols]
    d = np.sqrt((x - cols // 2) ** 2 + (y - rows // 2) ** 2)
    H = (hg - lg) * (1 - np.exp(-d ** 2 / (2 * r ** 2))) + lg

    r_ = np.real(np.fft.ifft2(np.fft.ifftshift(fs * H)))
    res = np.clip(np.exp(r_) - 1, 0, 255).astype(np.uint8)

    if len(image.shape) == 3:
        yc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        yc[:, :, 0] = res
        return cv2.cvtColor(yc, cv2.COLOR_YCrCb2BGR)
    return res


def illumination_correction(image, ks=31):
    """
    形态学光照校正:
    用大核闭运算估计背景光照 → 原图/背景 → 均匀光照
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    bg = cv2.GaussianBlur(cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k), (0, 0), ks // 3)
    c_ = cv2.divide(gray.astype(np.float64), bg.astype(np.float64) + 1, scale=255)
    c_ = np.clip(c_, 0, 255).astype(np.uint8)

    if len(image.shape) == 3:
        yc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        yc[:, :, 0] = c_
        return cv2.cvtColor(yc, cv2.COLOR_YCrCb2BGR)
    return c_


def retinex_ssr(image, sigma=50):
    """
    单尺度 Retinex (SSR):
    log(R) = log(I) - log(I * Gauss)
    """
    r = np.zeros_like(image, dtype=np.float64)
    for i in range(3):
        c = image[:, :, i].astype(np.float64) + 1
        r[:, :, i] = np.log(c) - np.log(cv2.GaussianBlur(c, (0, 0), sigma))
    return cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
