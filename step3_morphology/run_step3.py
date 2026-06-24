# -*- coding: utf-8 -*-
"""
Step 3 入口 — 形态学优化与 mask 修复 (Part C-1)
  输入: outputs/step2/fusion/weighted/imageX_0.5yfy_0.5xbw_fusion_mask.jpg
        data/original_images/imageX.jpg (用于头发精修和引导滤波)
  输出: outputs/step3/imageX_cleaned.png  (软 alpha, 0-255)
"""

import os, sys, re
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io_utils import imread, imwrite, load_mask_binary, find_original_image
from step3_morphology.morphology import (
    threshold_to_binary, conservative_clean, edge_aware_smooth
)
from step3_morphology.hair_refine import hair_refine


def process_one(image_id, mask_dir, img_dir, output_dir):
    """处理单张 mask: 清理 → 头发精修 → 引导滤波 → 软 alpha"""
    # 1. 加载融合 mask
    mask_path = os.path.join(mask_dir, f"{image_id}_0.5yfy_0.5xbw_fusion_mask.jpg")
    mask_img = imread(mask_path)
    if mask_img is None:
        print(f"  [SKIP] 找不到 {mask_path}")
        return None

    if len(mask_img.shape) == 3:
        gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask_img
    mask = threshold_to_binary(gray, thresh=10)

    # 2. 加载原图 (用于头发精修和引导滤波)
    original = find_original_image(image_id, img_dir)
    if original is not None:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        if mask.shape[:2] != original_gray.shape[:2]:
            mask = cv2.resize(mask, (original_gray.shape[1], original_gray.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    else:
        original_gray = None

    # 3. 保守形态学清理
    mask = conservative_clean(mask)

    # 4. 头发精修 (Frangi + Gabor + 各向异性扩展)
    if original is not None:
        mask = hair_refine(mask, original_gray, original)

    # 5. 边缘引导平滑 (引导滤波)
    if original_gray is not None:
        mask = edge_aware_smooth(mask, original_gray, radius=5, eps=1e-4)
    else:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)

    # 6. 硬阈值消除微弱背景残留 (alpha < 30 → 0)
    mask[mask < 30] = 0

    # 7. 输出
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{image_id}_cleaned.png")
    imwrite(out_path, mask)

    tag = "guided" if original is not None else "fallback"
    print(f"  [OK] {image_id} → Frangi+Gabor hair + guided-filter(r=5) [{tag}]")
    return mask


def run_step3(data_dir, step2_output_dir, output_dir):
    """批量运行 Step 3"""
    print("=" * 60)
    print("  Step 3: 形态学优化 + 头发精修 (Part C-1)")
    print("=" * 60)

    mask_dir = os.path.join(step2_output_dir, "fusion", "weighted")
    img_dir = os.path.join(data_dir, "original_images")
    step3_out = os.path.join(output_dir, "step3")
    os.makedirs(step3_out, exist_ok=True)

    # 扫描 mask
    pattern = re.compile(r"(image\d+)_0\.5yfy_0\.5xbw_fusion_mask\.jpg")
    image_ids = sorted(
        {m.group(1) for f in os.listdir(mask_dir) if (m := pattern.match(f))},
        key=lambda x: int(x.replace("image", ""))
    )

    if not image_ids:
        print(f"[ERROR] 在 {mask_dir} 中找不到融合 mask")
        return

    print(f"  Mask 数: {len(image_ids)}")
    print(f"  算法: 保守清理 → Frangi+Gabor 头发精修 → 引导滤波(r=5) → 硬阈值30")
    print(f"  输入: {mask_dir}")
    print(f"  输出: {step3_out}\n")

    for img_id in image_ids:
        print(f"  [{img_id}]")
        process_one(img_id, mask_dir, img_dir, step3_out)

    print(f"\n  ✓ Step 3 完成!")
    return step3_out


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE, "data")
    STEP2_OUT = os.path.join(BASE, "outputs", "step2")
    OUTPUT_DIR = os.path.join(BASE, "outputs")
    run_step3(DATA_DIR, STEP2_OUT, OUTPUT_DIR)
