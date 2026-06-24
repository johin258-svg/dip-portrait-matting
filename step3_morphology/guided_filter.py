# -*- coding: utf-8 -*-
"""
Step 3 — 引导滤波 (Guided Filter)
  He et al. 2013 — 边缘保持平滑, 计算量 O(N)。

原理:
  输出 q_i = a_k * I_i + b_k, ∀i ∈ window_k
  局部线性: a_k = cov(I,p)_k / (var(I)_k + ε), b_k = mean(p)_k - a_k * mean(I)_k
  窗口平均: q = mean(a)_k * I + mean(b)_k

用途:
  - 二值mask → 软alpha过渡 (跟随原图边缘, 自然羽化)
  - 头发精修后平滑过渡

参数:
  r:   窗口半径 (px) — 控制过渡带宽度。r=5 约 10px 过渡
  eps: 正则化 — 越小越贴合原图边缘, 默认 1e-4
"""

import cv2
import numpy as np


def guided_filter(I, p, r, eps):
    """
    引导滤波 — 边缘保持平滑。

    Args:
        I:   引导图像 (float32, 0~1), 通常用原图灰度
        p:   待滤波图像 (float32, 0~1), 通常用二值 mask
        r:   窗口半径 (px)
        eps: 正则化参数

    Returns: 滤波后图像 (float32, 0~1)
    """
    ksize = (2 * r + 1, 2 * r + 1)

    mean_I = cv2.boxFilter(I, -1, ksize, normalize=True)
    mean_p = cv2.boxFilter(p, -1, ksize, normalize=True)
    corr_I = cv2.boxFilter(I * I, -1, ksize, normalize=True)
    corr_Ip = cv2.boxFilter(I * p, -1, ksize, normalize=True)

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, ksize, normalize=True)
    mean_b = cv2.boxFilter(b, -1, ksize, normalize=True)

    q = mean_a * I + mean_b
    return np.clip(q, 0.0, 1.0)
