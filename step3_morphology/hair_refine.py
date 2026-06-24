# -*- coding: utf-8 -*-
"""
Step 3 — 头发精修算法 (Frangi + Gabor + 各向异性扩展)

核心思想:
  传统形态学膨胀会无差别扩展所有方向, 导致白边。
  本算法只在"确实有头发"的位置做有距离上限的定向扩展。

三通道检测:
  1. Frangi Vesselness (Frangi 1998):
     Hessian 特征值识别管状/线状结构 — 发丝 = λ1≈0(沿发丝) + λ2>0(横跨发丝)
     与斑点/角点有本质区别。

  2. Gabor 滤波器组:
     多方向纹理验证。头发有强方向性(max >> mean), 非头发纹理均匀。

  3. 软颜色似然:
     深色头发: darkness² + desaturation
     浅色/灰白头发: Lab 中性灰 + 高明度

融合策略:
  confidence = frangi × (0.3 × directionality + 0.7 × dark_color)
  → 乘性门控: 只有几何+纹理+颜色三者同时满足的才是头发

各向异性扩展:
  强头发(confidence≈1) → 扩展最多 12px
  弱头发(confidence≈0.05) → 扩展约 3px
  非头发(confidence≈0) → 不扩展
"""

import cv2
import numpy as np
from step3_morphology.guided_filter import guided_filter


# ==================== Frangi Vesselness ====================

def _compute_hessian_eigenvalues(gray_f32, sigma):
    """多尺度 σ 下计算 Hessian 矩阵的解析特征值 (λ1, λ2)"""
    ksize = int(2 * 3 * sigma + 1) | 1
    ksize = max(ksize, 3)

    smoothed = cv2.GaussianBlur(gray_f32, (ksize, ksize), sigma)
    Ix = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    Ixx = cv2.GaussianBlur(cv2.Sobel(Ix, cv2.CV_32F, 1, 0, ksize=3),
                           (ksize, ksize), sigma)
    Ixy = cv2.GaussianBlur(cv2.Sobel(Ix, cv2.CV_32F, 0, 1, ksize=3),
                           (ksize, ksize), sigma)
    Iyy = cv2.GaussianBlur(cv2.Sobel(Iy, cv2.CV_32F, 0, 1, ksize=3),
                           (ksize, ksize), sigma)

    trace = Ixx + Iyy
    det = Ixx * Iyy - Ixy * Ixy
    sqrt_disc = np.sqrt(np.maximum(trace * trace - 4.0 * det, 0.0))

    return 0.5 * (trace - sqrt_disc), 0.5 * (trace + sqrt_disc)


def frangi_vesselness(gray, sigmas=None, beta=0.5, c=None):
    """
    多尺度 Frangi 管状结构检测。

    Rb = |λ1|/|λ2| → 0=线状, 1=斑点
    S  = √(λ1²+λ2²) → 结构强度
    V  = exp(-Rb²/2β²) × (1-exp(-S²/2c²))  仅当 λ2>0

    Args:
        gray:   灰度图 (uint8, 0-255)
        sigmas: 尺度列表, 默认 [0.5, 1.0, 1.5, 2.0]
        beta:   线状选择性 (越小越严)
        c:      噪声抑制阈值 (自适应)

    Returns: vesselness (float32, 0~1)
    """
    if sigmas is None:
        sigmas = [0.5, 1.0, 1.5, 2.0]

    gray_f = gray.astype(np.float32)
    H, W = gray_f.shape
    vesselness = np.zeros((H, W), dtype=np.float32)

    for sigma in sigmas:
        lambda1, lambda2 = _compute_hessian_eigenvalues(gray_f, sigma)

        eps = 1e-8
        Rb = np.abs(lambda1) / (np.abs(lambda2) + eps)
        S = np.sqrt(lambda1 * lambda1 + lambda2 * lambda2)

        c_val = 0.5 * np.max(S) if c is None else c
        c_val = max(c_val, 1.0)

        v = np.where(lambda2 > 0,
                     np.exp(-Rb * Rb / (2.0 * beta * beta)) *
                     (1.0 - np.exp(-S * S / (2.0 * c_val * c_val))),
                     0.0)
        vesselness = np.maximum(vesselness, v)

    vmax = vesselness.max()
    if vmax > 0:
        vesselness /= vmax
    return vesselness.astype(np.float32)


# ==================== Gabor 滤波器组 ====================

def gabor_texture_response(gray, orientations=6, sigma=2.5, lambd=3.0, gamma=0.5):
    """
    Gabor 多方向纹理 — 方向性度量。

    核心改进: 用 (max-mean)/(max+eps) 度量方向性, 而非全局归一化。
    头发: max >> mean → 接近 1; 均匀纹理: max ≈ mean → 接近 0。

    Returns: 方向性图 (float32, 0~1)
    """
    H, W = gray.shape
    gray_f = gray.astype(np.float32)
    ksize = 11

    all_resp = np.zeros((H, W, orientations), dtype=np.float32)
    for i in range(orientations):
        theta = i * np.pi / orientations
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0,
                                    ktype=cv2.CV_32F)
        all_resp[:, :, i] = np.abs(cv2.filter2D(gray_f, cv2.CV_32F, kernel))

    resp_max = all_resp.max(axis=2)
    resp_mean = all_resp.mean(axis=2)
    eps = 1e-6
    directionality = (resp_max - resp_mean) / (resp_max + eps)
    max_norm = resp_max / (resp_max.max() + eps)

    gabor_hair = directionality * max_norm
    gmax = gabor_hair.max()
    if gmax > 0:
        gabor_hair /= gmax
    return gabor_hair.astype(np.float32)


# ==================== 颜色似然 ====================

def hair_color_likelihood(bgr):
    """
    软颜色似然 — 深色 + 浅色头发覆盖。

    深色: darkness² (平方增强鉴别力) × 0.7 + desaturation × 0.3
    浅色: Lab a,b 中性灰 × L 明度 × 0.3 (降权避免误检)
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32)

    V, S = hsv[:, :, 2] / 255.0, hsv[:, :, 1] / 255.0
    L_norm = lab[:, :, 0] / 255.0
    A_norm, B_norm = lab[:, :, 1] / 255.0, lab[:, :, 2] / 255.0

    darkness2 = (1.0 - V) ** 2
    dark_likelihood = darkness2 * 0.7 + (1.0 - S) * 0.3

    neutrality = np.clip((1.0 - np.abs(A_norm - 0.5) * 3.0) *
                         (1.0 - np.abs(B_norm - 0.5) * 3.0), 0, 1)
    light_likelihood = neutrality * L_norm * 0.3

    return np.clip(np.maximum(dark_likelihood, light_likelihood), 0, 1).astype(np.float32)


# ==================== 融合 ====================

def compute_hair_confidence(bgr, gray):
    """
    三通道融合 → 头发置信度。

    confidence = frangi × (0.3 × directionality + 0.7 × dark_color)
    乘性门控: 三者同时满足 → 头发, 任一不满足 → 抑制。
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)

    frangi = frangi_vesselness(gray_enhanced)
    directionality = gabor_texture_response(gray_enhanced)
    dark_color = hair_color_likelihood(bgr)

    soft_gate = 0.3 * directionality + 0.7 * dark_color
    confidence = frangi * soft_gate
    return np.clip(confidence, 0, 1).astype(np.float32)


# ==================== 各向异性扩展 ====================

def expand_mask_along_hair(mask, hair_confidence, min_expand=3, max_expand=12):
    """
    沿头发方向的可变距离扩展 (替代固定椭圆膨胀)。

    - 强头发 (conf≈1) → 扩展 12px, 捕获飞丝
    - 弱头发 (conf≈0.05) → 扩展 ~3px, 保守
    - 非头发 (conf≈0) → 不扩展

    Args:
        mask:            二值 mask (0/255)
        hair_confidence: 头发置信度 (0~1)
        min_expand:      最小扩展距离 (px)
        max_expand:      最大扩展距离 (px)

    Returns: 扩展后二值 mask (0/255)
    """
    # 距离变换
    dist_in = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
    signed_dist = dist_in - dist_out  # 内部>0, 边界=0, 外部<0

    expand_dist = min_expand + hair_confidence * (max_expand - min_expand)

    # 扩展带: mask 外部, 在扩展距离内, 置信度 > 0.03
    in_band = (signed_dist > -expand_dist) & (signed_dist <= 0)
    is_hair = hair_confidence > 0.03
    expansion = (in_band & is_hair).astype(np.uint8) * 255

    # 去噪: 闭运算连接碎片 + 去除 < 12px 孤立连通域
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    expansion = cv2.morphologyEx(expansion, cv2.MORPH_CLOSE, k, iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(expansion, connectivity=8)
    cleaned = np.zeros_like(expansion)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 12:
            cleaned[labels == i] = 255

    return cv2.bitwise_or(mask, cleaned)


# ==================== 统一入口 ====================

def hair_refine(mask_binary, original_gray, original_bgr):
    """
    头发精修主入口 (v7: Frangi + Gabor + 各向异性扩展)

    流程:
      1. 定位头部 ROI (mask 上 55% 区域)
      2. 计算头发置信度
      3. 各向异性扩展 (3-12px)
      4. 引导滤波平滑过渡 (r=6)
      5. 硬裁剪 (≤ 12px 安全上限)
      6. 融合回全局 mask

    Returns: 精修后软 alpha mask (0-255)
    """
    H, W = mask_binary.shape

    # 1. 定位头部
    upper = mask_binary.copy()
    upper[int(H * 0.55):, :] = 0
    contours, _ = cv2.findContours(upper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask_binary

    fx, fy, fw, fh = cv2.boundingRect(max(contours, key=cv2.contourArea))

    ht, hb = max(0, fy - int(fh * 0.9)), min(H, fy + int(fh * 0.35))
    hl, hr = max(0, fx - int(fw * 0.35)), min(W, fx + fw + int(fw * 0.35))
    if hb <= ht or hr <= hl:
        return mask_binary

    # 2. ROI
    roi_mask = mask_binary[ht:hb, hl:hr]
    roi_gray = original_gray[ht:hb, hl:hr]
    roi_bgr = original_bgr[ht:hb, hl:hr]

    # 3. 置信度
    hair_conf = compute_hair_confidence(roi_bgr, roi_gray)
    if np.sum(hair_conf > 0.03) < 50:
        return mask_binary

    # 4. 各向异性扩展
    roi_expanded = expand_mask_along_hair(roi_mask, hair_conf, min_expand=3, max_expand=12)

    # 5. 引导滤波平滑
    roi_f = roi_expanded.astype(np.float32) / 255.0
    guide_f = roi_gray.astype(np.float32) / 255.0
    roi_smooth = guided_filter(guide_f, roi_f, r=6, eps=1e-5)
    roi_smooth = (roi_smooth * 255).astype(np.uint8)

    # 6. 硬裁剪
    k_clip = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    roi_clip = cv2.dilate(roi_mask, k_clip, iterations=1)
    roi_smooth = np.minimum(roi_smooth, roi_clip)

    # 7. 融合
    result = mask_binary.astype(np.float32)
    result[ht:hb, hl:hr] = np.maximum(result[ht:hb, hl:hr], roi_smooth.astype(np.float32))
    return np.clip(result, 0, 255).astype(np.uint8)
