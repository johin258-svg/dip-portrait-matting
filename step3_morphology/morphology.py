# -*- coding: utf-8 -*-
"""
Step 3 — 形态学优化 (保守清理 + 头发精修 + 引导滤波平滑)

流水线:
  ① 二值化 (thresh=10)
  ② 保守形态学清理: 开运算 → 去小连通域 → 保留前3大域 → 填孔洞
  ③ 头发精修: Frangi + Gabor + 各向异性扩展 (仅头部区域)
  ④ 边缘引导平滑: 引导滤波 (r=5) 将硬边界转为软 alpha
  ⑤ 硬阈值消除微弱背景 (alpha < 30 → 0)
  ⑥ 输出软 alpha PNG (0-255)
"""

import cv2
import numpy as np
from step3_morphology.guided_filter import guided_filter
from step3_morphology.hair_refine import hair_refine


# ==================== 工具函数 ====================

def threshold_to_binary(gray, thresh=10):
    """确保纯二值 (0/255)"""
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return binary


def remove_small_components(mask, min_area=500):
    """连通域分析: 去掉面积 < min_area 的小噪点"""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def keep_top_n_components(mask, top_n=3):
    """保留面积最大的前 N 个连通域"""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return mask
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_labels)]
    areas.sort(key=lambda x: x[1], reverse=True)
    keep = {i for i, _ in areas[:top_n]}
    result = np.zeros_like(mask)
    for lb in keep:
        result[labels == lb] = 255
    return result


def fill_inner_holes(mask, max_hole_area_ratio=0.05):
    """填充内部孔洞 (< 图像 5% 面积), 使用层级关系识别真正孔洞"""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return mask

    filled = mask.copy()
    limit = mask.shape[0] * mask.shape[1] * max_hole_area_ratio
    hierarchy = hierarchy[0]

    for i, (cnt, h) in enumerate(zip(contours, hierarchy)):
        if h[3] != -1:  # 有父级 → 孔洞
            if cv2.contourArea(cnt) < limit:
                cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)
    return filled


# ==================== 保守清理 ====================

def conservative_clean(mask):
    """
    保守形态学清理: 只做向内操作，绝不向外膨胀。
    1. 轻开运算 (3×3, 1轮) → 去噪
    2. 去除 < 500px 小连通域
    3. 保留前 3 大连通域
    4. 填充内部孔洞
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = remove_small_components(mask, min_area=500)
    mask = keep_top_n_components(mask, top_n=3)
    mask = fill_inner_holes(mask, max_hole_area_ratio=0.05)
    return mask


# ==================== 边缘平滑 ====================

def edge_aware_smooth(mask, original_gray, radius=5, eps=1e-4):
    """
    引导滤波边缘平滑 — 将二值 mask 转为软 alpha。
    过渡带宽度 ~radius px, 边缘跟随原图特征。
    """
    mask_f = mask.astype(np.float32) / 255.0
    guide_f = original_gray.astype(np.float32) / 255.0
    alpha_f = guided_filter(guide_f, mask_f, r=radius, eps=eps)
    return (alpha_f * 255).astype(np.uint8)
