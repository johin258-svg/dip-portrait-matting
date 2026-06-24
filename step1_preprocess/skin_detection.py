# -*- coding: utf-8 -*-
"""
Step 1 — 肤色检测 (Part A)
  - HSV 肤色掩膜: H∈[0,20]∪[170,180], S∈[30,255], V∈[60,255]
  - YCrCb 肤色掩膜: Cr∈[133,173], Cb∈[77,127]
  - 融合: OR 策略 (任一空间判定为肤色即采纳)
  - 形态学精炼: 开运算 + 闭运算
"""

import cv2
import numpy as np


def skin_mask_hsv(image):
    """HSV 空间肤色检测 — 覆盖黄/白/黑肤色"""
    h = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(h, np.array([0, 30, 60]), np.array([20, 255, 255]))
    m2 = cv2.inRange(h, np.array([170, 30, 60]), np.array([180, 255, 255]))
    return cv2.bitwise_or(m1, m2)


def skin_mask_ycrcb(image):
    """YCrCb 空间肤色检测 — 对亚洲肤色更精确"""
    y = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    return cv2.inRange(y, np.array([0, 133, 77]), np.array([255, 173, 127]))


def combine_skin_masks(m_hsv, m_ycrcb, mode="or"):
    """融合 HSV + YCrCb 肤色掩膜: 'or' 或 'and'"""
    if mode == "or":
        return cv2.bitwise_or(m_hsv, m_ycrcb)
    return cv2.bitwise_and(m_hsv, m_ycrcb)


def apply_mask(image, mask):
    """应用掩膜 (保留肤色区域, 其余变黑)"""
    return cv2.bitwise_and(image, image, mask=mask)


def morphological_refine(mask, ks=5, iterations=1):
    """形态学精炼: 开运算(去噪) → 闭运算(填孔洞)"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=iterations)
    return mask


def detect_skin(image, mode="or", refine=True):
    """
    完整肤色检测流水线:
    image → HSV + YCrCb → 融合 → 形态学精炼 → 肤色掩膜 (0/255)
    """
    m_hsv = skin_mask_hsv(image)
    m_ycrcb = skin_mask_ycrcb(image)
    combined = combine_skin_masks(m_hsv, m_ycrcb, mode)
    if refine:
        combined = morphological_refine(combined)
    return combined
