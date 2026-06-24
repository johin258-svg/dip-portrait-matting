# -*- coding: utf-8 -*-
"""
Step 5 — 抠图与背景替换 (Part C-3)
  Alpha 混合公式: result = fg × α + bg × (1-α)

输出类型:
  - 透明 PNG (RGBA): 带 alpha 通道, 可直接叠加到任意背景
  - 纯色背景: 白/红/蓝/绿四种常用背景
  - 自定义背景: 读取 background.jpg 替换背景
"""

import cv2
import numpy as np


def alpha_blend(fg, mask, bg, blur_size=1):
    """
    Alpha 混合: fg × α + bg × (1-α)
    blur_size=1: morphology 已输出软 alpha, 不再额外羽化
    """
    if blur_size <= 1:
        feather = mask
    else:
        k = blur_size if blur_size % 2 == 1 else blur_size + 1
        feather = cv2.GaussianBlur(mask, (k, k), 0)

    alpha = feather.astype(np.float32) / 255.0
    alpha_3 = np.stack([alpha] * 3, axis=-1)
    result = fg.astype(np.float32) * alpha_3 + bg.astype(np.float32) * (1 - alpha_3)
    return np.clip(result, 0, 255).astype(np.uint8)


def cutout_transparent(img, mask, blur_size=1):
    """透明背景 RGBA — PNG 格式保留 alpha 通道"""
    if blur_size <= 1:
        feather = mask
    else:
        k = blur_size if blur_size % 2 == 1 else blur_size + 1
        feather = cv2.GaussianBlur(mask, (k, k), 0)
    b, g, r = cv2.split(img)
    return cv2.merge([b, g, r, feather])


def cutout_solid(img, mask, color_name, blur_size=1):
    """纯色背景替换: white / red / blue / green"""
    colors = {
        "white": (255, 255, 255), "red": (0, 0, 255),
        "blue": (255, 0, 0), "green": (0, 255, 0),
    }
    bg = np.ones_like(img, dtype=np.float32) * np.array(colors.get(color_name, (255, 255, 255)),
                                                         dtype=np.float32)
    return alpha_blend(img, mask, bg, blur_size)


def cutout_custom_bg(img, mask, bg_img, blur_size=1):
    """
    自定义背景替换。
    如果背景尺寸不匹配, 自动缩放 (Lanczos4 插值)。
    """
    fg_h, fg_w = img.shape[:2]
    bg_h, bg_w = bg_img.shape[:2]
    if (bg_h, bg_w) != (fg_h, fg_w):
        bg_resized = cv2.resize(bg_img, (fg_w, fg_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        bg_resized = bg_img
    return alpha_blend(img, mask, bg_resized.astype(np.float32), blur_size)
