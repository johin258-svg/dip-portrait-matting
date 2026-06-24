# -*- coding: utf-8 -*-
"""
共享可视化工具
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def save_comparison_grid(images, titles, save_path, cols=4, figsize=(20, 10),
                         cmap_gray=True, dpi=150):
    """保存对比网格图"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for ax, img, title in zip(axes, images, titles):
        is_gray = cmap_gray and (len(img.shape) == 2 or img.shape[2] == 1)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if not is_gray else img,
                  cmap="gray" if is_gray else None)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    for ax in axes[len(images):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def draw_overlay(image, mask, color=(0, 255, 0)):
    """在原图上绘制掩膜边界 (绿色)"""
    overlay = image.copy()
    boundary = cv2.Canny(mask, 0, 255)
    overlay[boundary > 0] = color
    return overlay
