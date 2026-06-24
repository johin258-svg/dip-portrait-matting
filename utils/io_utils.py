# -*- coding: utf-8 -*-
"""
共享 I/O 工具 — 中文路径安全读写 + 掩膜加载
"""

import cv2
import numpy as np
import os


def imread(path):
    """中文路径安全读取图像 (BGR)"""
    if not os.path.exists(path):
        return None
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite(path, img):
    """中文路径安全写入图像"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jpg":
        cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(path)
    else:
        cv2.imencode(".png", img)[1].tofile(path)


def load_mask_binary(path, threshold=30):
    """加载掩膜为二值图 (0/255)"""
    img = imread(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


def load_mask_gray(path):
    """加载掩膜为灰度图 (保留软alpha 0-255)"""
    img = imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img


def find_original_image(image_id, img_dir):
    """在目录中按多种扩展名查找原图"""
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        p = os.path.join(img_dir, f"{image_id}{ext}")
        if os.path.exists(p):
            return imread(p)
    return None
