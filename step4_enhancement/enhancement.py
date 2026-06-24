# -*- coding: utf-8 -*-
"""
Step 4 — 人像增强 (Part C-2)
  只对人像区域 (mask > 0) 做处理, 保持背景不变。

增强类型:
  - 美白:   通道分离调整 (B×1.0 G×0.88 R×0.85) + 提亮 +22
  - 降噪:   双边滤波 (保边)
  - 锐化:   Unsharp Mask (Gaussian 残差放大)
  - 复古:   暖琥珀色调 + 降饱和 (全图)
  - Dream:  增对比 + 偏蓝冷调 (全图)
  - 胶片:   提阴影 + 暖高光 + 颗粒噪点 (全图)
  - 粉色:   粉紫调 + 柔光 (全图)
  - 综合:   降噪 → 美白 → 锐化 → 提亮
"""

import cv2
import numpy as np


def apply_on_person(img, mask, func):
    """只对 mask 内像素做增强, 背景不变"""
    result = img.copy()
    result[mask > 0] = func(img)[mask > 0]
    return result


# ==================== 人像区域增强 ====================

def whiten(img, mask):
    """
    美白: 通道分离调整 + 提亮
      B×1.00 → 中性
      G×0.88 → 减黄
      R×0.85 → 减暖红 (陶瓷肌)
      整体 +22 亮度
    """
    def _channel_whiten(roi):
        b, g, r = cv2.split(roi.astype(np.float32))
        b = np.clip(b * 1.00, 0, 255)
        g = np.clip(g * 0.88, 0, 255)
        r = np.clip(r * 0.85, 0, 255)
        return cv2.convertScaleAbs(cv2.merge([b, g, r]), alpha=1.0, beta=22)
    return apply_on_person(img, mask, _channel_whiten)


def denoise_bilateral(img, mask):
    """双边滤波降噪 (保边缘)"""
    return apply_on_person(img, mask,
        lambda roi: cv2.bilateralFilter(roi, 7, 60, 60))


def sharpen_unsharp(img, mask):
    """Unsharp Mask 锐化: sharp = img × 2.2 + blur × (-1.2)"""
    def _sharp(roi):
        blurred = cv2.GaussianBlur(roi, (0, 0), 1.5)
        return cv2.addWeighted(roi, 2.2, blurred, -1.2, 0)
    return apply_on_person(img, mask, _sharp)


def comprehensive_enhance(img, mask):
    """综合增强: 降噪 → 美白 → 锐化 → 提亮"""
    result = denoise_bilateral(img, mask)
    result = whiten(result, mask)
    result = sharpen_unsharp(result, mask)
    return apply_on_person(result, mask,
        lambda roi: cv2.convertScaleAbs(roi, alpha=1.0, beta=10))


# ==================== 全图滤镜 ====================

def vintage(img):
    """
    复古滤镜: 暖琥珀色调 + 降饱和 65%
      B×0.80 G×1.03 R×1.08 → 暖黄琥珀
    """
    result = img.astype(np.float32)
    result[:, :, 0] *= 0.80
    result[:, :, 1] *= 1.03
    result[:, :, 2] *= 1.08
    np.clip(result, 0, 255, out=result)
    hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 0.65
    np.clip(hsv, 0, 255, out=hsv)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def dream(img):
    """
    Dream 滤镜: 增对比度 + 偏蓝冷调
      B×1.06 G×0.96 R×0.94 → 微冷
      alpha=1.12 对比度增强
    """
    result = img.astype(np.float32)
    result[:, :, 0] *= 1.06
    result[:, :, 1] *= 0.96
    result[:, :, 2] *= 0.94
    np.clip(result, 0, 255, out=result)
    return cv2.convertScaleAbs(result.astype(np.uint8), alpha=1.12, beta=0)


def film(img):
    """
    胶片质感: 提阴影 + 暖高光 + 高斯颗粒噪点
      暗部 +15, 高光 R+微暖 B-微冷
      饱和度 80%, 颗粒 std=4
    """
    result = img.astype(np.float32)
    h, w = result.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    shadow_lift = np.dstack([(1.0 - gray) * 15] * 3)
    result += shadow_lift
    result[:, :, 2] += gray * 0.08 * 255  # R+ 暖高光
    result[:, :, 0] -= gray * 0.08 * 200  # B- 冷减
    np.clip(result, 0, 255, out=result)
    hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 0.80
    np.clip(hsv, 0, 255, out=hsv)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    noise = np.random.randn(h, w, 3).astype(np.float32) * 4.0
    result += noise
    np.clip(result, 0, 255, out=result)
    return result.astype(np.uint8)


def pink(img):
    """
    粉色氛围: 粉紫调 + 柔光
      B×1.06 R×1.04 G×0.92 → 粉紫
      高斯模糊 15% 叠加 → 柔焦
    """
    result = img.astype(np.float32)
    result[:, :, 0] *= 1.06
    result[:, :, 1] *= 0.92
    result[:, :, 2] *= 1.04
    np.clip(result, 0, 255, out=result)
    blurred = cv2.GaussianBlur(result, (0, 0), 3)
    result = cv2.addWeighted(result, 0.85, blurred, 0.15, 0)
    np.clip(result, 0, 255, out=result)
    return result.astype(np.uint8)
