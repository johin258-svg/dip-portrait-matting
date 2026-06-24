# -*- coding: utf-8 -*-
"""
Step 2 — GrabCut 人像分割 (Part B 核心)
  基于 GrabCut 图割算法, 用 xbw 肤色掩膜作为种子, 分割出完整人物。

核心算法:
  GrabCut (Rother et al. 2004):
    使用 BGR 3通道 GMM 建模前景/背景颜色分布 + 图割空间约束。
    即使前景/背景颜色相近 (如白色衣服+白色背景), 也能通过空间上下文分离。

初始化策略:
  GC_FGD (确信前景): xbw 面部皮肤像素 (侵蚀后)
  GC_BGD (确信背景): 图像边界条带
  GC_PR_FGD (可能前景): 身体+头发估计椭圆区域
  GC_PR_BGD (可能背景): 其余区域 (默认)

分割流程:
  第1轮 GrabCut (5次迭代, MASK初始化) → 第2轮 (3次迭代精修)
  → 形态学后处理 → 凸包约束扩展 → 头发HSV+Canny精修

为什么用 original_images 而非 light_fixed:
  GrabCut 的 GMM 颜色建模需要完整的 3 通道色彩信息。
  light_fixed 经过了光照校正、可能丢失色彩细节。
  original_images 保留了最自然的色彩分布。
"""

import cv2
import numpy as np
import os


# ==================== 身体/头发区域估计 ====================

def estimate_body_hair_regions(xbw_mask):
    """
    从 xbw 面部包围盒估计身体和头发区域 → GrabCut PR_FGD 种子。

    返回: (body_mask, hair_mask) — 0/255 uint8 椭圆区域
    """
    contours, _ = cv2.findContours(xbw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    H, W = xbw_mask.shape

    # 身体区域: 面部下方 → 图像底部
    body_cx = x + w // 2
    body_hw = max(int(w * 0.9), 20)
    body_top = y + int(h * 0.7)
    body_bottom = H
    body_rh = max((body_bottom - body_top) // 2, 20)

    body_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(body_mask, (body_cx, (body_top + body_bottom) // 2),
                (body_hw, body_rh), 0, 0, 360, 255, -1)

    # 头发区域: 面部上方
    hair_top = max(0, y - int(h * 0.6))
    hair_bottom = y + int(h * 0.2)
    hair_hw = max(int(w * 0.7), 10)

    hair_mask = np.zeros((H, W), dtype=np.uint8)
    rh = max((hair_bottom - hair_top) // 2, 10)
    cv2.ellipse(hair_mask, (x + w // 2, (hair_top + hair_bottom) // 2),
                (hair_hw, rh), 0, 0, 360, 255, -1)

    return body_mask, hair_mask


# ==================== GrabCut 初始掩膜 ====================

def create_grabcut_init_mask(H, W, xbw_mask):
    """
    创建 GrabCut 初始掩膜:
      GC_BGD    = 0 — 图像边界条带 (确信背景)
      GC_FGD    = 1 — xbw 皮肤像素 (确信前景)
      GC_PR_BGD = 2 — 默认区域 (可能背景)
      GC_PR_FGD = 3 — 身体+头发估计 (可能前景)

    Returns: (mask, use_rect_fallback)
      use_rect_fallback: xbw 覆盖率 > 80% 时回退到矩形初始化
    """
    mask = np.full((H, W), cv2.GC_PR_BGD, dtype=np.uint8)

    # 1. GC_BGD: 边界条带 + 四角
    border = int(min(H, W) * 0.08)
    corner = int(min(H, W) * 0.15)
    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD
    mask[:corner, :corner] = cv2.GC_BGD
    mask[:corner, -corner:] = cv2.GC_BGD
    mask[-corner:, :corner] = cv2.GC_BGD
    mask[-corner:, -corner:] = cv2.GC_BGD

    # 2. GC_FGD: xbw 皮肤 (侵蚀后更可靠)
    use_rect_fallback = False
    if xbw_mask is not None and np.sum(xbw_mask > 0) > 0:
        xbw_coverage = np.sum(xbw_mask > 0) / xbw_mask.size
        if xbw_coverage < 0.80:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            xbw_eroded = cv2.erode(xbw_mask, k, iterations=2)
            mask[xbw_eroded > 0] = cv2.GC_FGD
        else:
            use_rect_fallback = True

    # 3. GC_PR_FGD: 身体+头发估计区域
    if xbw_mask is not None and np.sum(xbw_mask > 0) > 0 and not use_rect_fallback:
        body_mask, hair_mask = estimate_body_hair_regions(xbw_mask)
        pr_fgd = np.zeros((H, W), dtype=np.uint8)
        if body_mask is not None:
            pr_fgd = cv2.bitwise_or(pr_fgd, body_mask)
        if hair_mask is not None:
            pr_fgd = cv2.bitwise_or(pr_fgd, hair_mask)

        # 面部下方宽列
        contours, _ = cv2.findContours(xbw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            lx, ly, lw, lh = cv2.boundingRect(max(contours, key=cv2.contourArea))
            cl = max(0, lx + lw // 2 - int(lw * 0.75))
            cr = min(W, lx + lw // 2 + int(lw * 0.75))
            ct = ly + int(lh * 0.5)
            if cr > cl and H > ct:
                pr_fgd[ct:H, cl:cr] = 255

        pr_fgd[mask != cv2.GC_PR_BGD] = 0
        mask[pr_fgd > 0] = cv2.GC_PR_FGD

    return mask, use_rect_fallback


# ==================== GrabCut 分割 ====================

def grabcut_segment(original, xbw_mask):
    """
    运行 GrabCut 分割。

    当 xbw 覆盖率 > 80% 时使用矩形初始化, 否则使用掩膜初始化 + 两轮迭代。
    Returns: 前景二值掩膜 (0/255 uint8)
    """
    H, W = original.shape[:2]
    init_mask, use_rect = create_grabcut_init_mask(H, W, xbw_mask)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    mask = init_mask.copy()

    if use_rect:
        rect = (int(W * 0.05), int(H * 0.02), int(W * 0.90), int(H * 0.95))
        cv2.grabCut(original, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    else:
        cv2.grabCut(original, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
        cv2.grabCut(original, mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)

    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


# ==================== 后处理 ====================

def post_process_mask(fg_mask, xbw_mask=None):
    """
    形态学后处理:
      1. 闭运算填补缺口
      2. 保留主要连通域 (≥ 5% 最大域)
      3. 凸包约束扩展 (利用"人物是凸的"先验)
      4. 孔洞填充 (< 10% 图像面积)
      5. 确保覆盖 xbw
      6. 安全检查 (覆盖率 > 90% 时回退)
    """
    H, W = fg_mask.shape

    # 1. 闭运算
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k, iterations=1)

    # 2. 保留主要连通域
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels > 1:
        areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_labels)]
        areas.sort(key=lambda x: x[1], reverse=True)
        if areas:
            largest_area = areas[0][1]
            keep = {i for i, a in areas if a >= max(500, largest_area * 0.05)}
            mask = np.where(np.isin(labels, list(keep)), 255, 0).astype(np.uint8)

    # 3. 凸包约束扩展
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        all_pts = np.vstack([c.reshape(-1, 2) for c in contours])
        if len(all_pts) >= 3:
            hull = cv2.convexHull(all_pts)
            hull_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(hull_mask, [hull], -1, 255, -1)
            k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            safe_expand = cv2.bitwise_and(hull_mask, cv2.dilate(mask, k_d, iterations=1))
            mask = cv2.bitwise_or(mask, safe_expand)

    # 4. 孔洞填充
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    inv = cv2.bitwise_not(filled)
    inv_contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hole_limit = H * W * 0.10
    for cnt in inv_contours:
        if cv2.contourArea(cnt) < hole_limit:
            cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)
    mask = filled

    # 5. 确保覆盖 xbw
    if xbw_mask is not None:
        mask = cv2.bitwise_or(mask, xbw_mask)

    # 6. 安全检查
    coverage = np.sum(mask > 0) / mask.size
    if coverage > 0.90:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            conservative = np.zeros_like(mask)
            cv2.drawContours(conservative, [largest], -1, 255, thickness=cv2.FILLED)
            if np.sum(conservative > 0) / conservative.size <= 0.90:
                mask = conservative
            elif xbw_mask is not None:
                k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                mask = cv2.dilate(xbw_mask, k_d, iterations=2)

    return mask


# ==================== 头发精修 ====================

def refine_hair(original, mask, xbw_mask):
    """
    GrabCut 在头发区域容易漏检 — 用 HSV+纹理+边缘 检测头发像素,
    只保留与现有掩膜连通的区域。
    """
    if xbw_mask is None or np.sum(xbw_mask > 0) == 0:
        return mask

    H, W = original.shape[:2]
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(xbw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    fx, fy, fw, fh = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # ROI: 面部上方 + 两侧
    t, b = 0, fy + int(fh * 0.25)
    l, r = max(0, fx - int(fw * 0.25)), min(W, fx + fw + int(fw * 0.25))
    if b <= t or r <= l:
        return mask

    roi = original[t:b, l:r]
    roi_gray = gray[t:b, l:r]
    roi_hsv = hsv[t:b, l:r]
    roi_mask = mask[t:b, l:r]
    rh, rw = roi_gray.shape

    v = roi_hsv[:, :, 2].astype(np.float32)
    s = roi_hsv[:, :, 1].astype(np.float32)

    # 纹理 (局部标准差)
    gf = roi_gray.astype(np.float32)
    m = cv2.blur(gf, (7, 7))
    m2 = cv2.blur(gf ** 2, (7, 7))
    texture = np.sqrt(np.maximum(m2 - m ** 2, 0))

    # 三个头发检测通道
    dark_hair = ((v < 70) & (s < 60) & (texture > 3)).astype(np.uint8) * 255
    mid_hair = ((v >= 70) & (v < 140) & (texture > 5)).astype(np.uint8) * 255
    hair_edges = cv2.morphologyEx(cv2.Canny(roi_gray, 10, 35),
                                   cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                                   iterations=1)

    hair_pixels = cv2.bitwise_or(cv2.bitwise_or(dark_hair, mid_hair), hair_edges)
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    hair_pixels = cv2.morphologyEx(hair_pixels, cv2.MORPH_CLOSE, kc, iterations=1)
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hair_pixels = cv2.morphologyEx(hair_pixels, cv2.MORPH_OPEN, ko, iterations=1)

    if np.sum(hair_pixels > 0) < 100:
        return mask

    # 只保留与现有掩膜连通的头发区域
    ka = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    roi_mask_dilated = cv2.dilate(roi_mask, ka, iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hair_pixels, connectivity=8)
    hair_connected = np.zeros_like(hair_pixels)
    for i in range(1, n_labels):
        comp = (labels == i).astype(np.uint8)
        if np.sum(cv2.bitwise_and(comp, roi_mask_dilated)) > 0:
            hair_connected[comp > 0] = 255

    if np.sum(hair_connected > 0) < 50:
        return mask

    full_hair = np.zeros((H, W), dtype=np.uint8)
    full_hair[t:b, l:r] = hair_connected
    return cv2.bitwise_or(mask, full_hair)
