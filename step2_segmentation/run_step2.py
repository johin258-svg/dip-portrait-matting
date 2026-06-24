# -*- coding: utf-8 -*-
"""
Step 2 入口 — GrabCut 分割 + 融合 (Part B)
  输入: data/original_images/imageX.jpg (原图, 用于 GrabCut 颜色建模)
        outputs/step1/xbw_masks/imageX_xbw_mask.jpg (肤色种子)
        outputs/step1/light_fixed/imageX_xbw_light2.jpg (获取目标尺寸)
  输出: outputs/step2/masks/yfy_masks/imageX_yfy_mask.jpg
        outputs/step2/fusion/weighted/imageX_0.5yfy_0.5xbw_fusion_mask.jpg
"""

import os, sys, re
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io_utils import imread, imwrite, load_mask_binary
from step2_segmentation.grabcut_segment import (
    grabcut_segment, post_process_mask, refine_hair
)
from step2_segmentation.fusion import batch_fusion


def process_one(image_id, original_dir, light_fixed_dir, xbw_dir, output_dir):
    """处理单张: GrabCut 分割 → 后处理 → 头发精修 → 保存 yfy_mask"""
    # 1. 加载原图 (用于 GrabCut 颜色建模)
    original = None
    for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
        p = os.path.join(original_dir, f"{image_id}{ext}")
        if os.path.exists(p):
            original = imread(p)
            break
    if original is None:
        print(f"  [SKIP] 无原图: {image_id} — 跳过 (仅处理有原始照片的图像)")
        return None

    orig_h, orig_w = original.shape[:2]

    # 2. 获取目标尺寸 (light_fixed)
    lf = imread(os.path.join(light_fixed_dir, f"{image_id}_xbw_light2.jpg"))
    if lf is None:
        print(f"  [ERROR] 找不到 light_fixed: {image_id}")
        return None
    lf_h, lf_w = lf.shape[:2]

    # 3. 加载 xbw (缩放到原图尺寸作种子)
    xbw_lf = load_mask_binary(os.path.join(xbw_dir, f"{image_id}_xbw_mask.jpg"))
    if xbw_lf is not None:
        xbw_gc = cv2.resize(xbw_lf, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    else:
        print(f"  [WARN] 无 xbw, 使用中心椭圆")
        xbw_gc = np.zeros((orig_h, orig_w), dtype=np.uint8)
        cv2.ellipse(xbw_gc, (orig_w // 2, orig_h // 3),
                    (orig_w // 4, orig_h // 3), 0, 0, 360, 255, -1)

    # 4. GrabCut 分割 (原图分辨率)
    fg = grabcut_segment(original, xbw_gc)
    final_orig = post_process_mask(fg, xbw_gc)
    final_orig = refine_hair(original, final_orig, xbw_gc)

    # 5. 缩放到 light_fixed 尺寸保存
    if (orig_h, orig_w) != (lf_h, lf_w):
        final = cv2.resize(final_orig, (lf_w, lf_h), interpolation=cv2.INTER_NEAREST)
    else:
        final = final_orig

    os.makedirs(output_dir, exist_ok=True)
    mask_path = os.path.join(output_dir, f"{image_id}_yfy_mask.jpg")
    imwrite(mask_path, final)

    yfy_px = int(np.sum(final > 0))
    img_area = lf_h * lf_w
    print(f"  [OK] {image_id} | yfy={100*yfy_px/img_area:.1f}% "
          f"| {orig_w}x{orig_h}→{lf_w}x{lf_h}")

    return final


def run_step2(data_dir, step1_output_dir, output_dir):
    """批量运行 Step 2"""
    print("=" * 60)
    print("  Step 2: GrabCut 人像分割 + 掩膜融合 (Part B)")
    print("=" * 60)

    original_dir = os.path.join(data_dir, "original_images")
    light_fixed_dir = os.path.join(step1_output_dir, "light_fixed")
    xbw_dir = os.path.join(step1_output_dir, "xbw_masks")
    yfy_output_dir = os.path.join(output_dir, "step2", "masks", "yfy_masks")
    fusion_output_dir = os.path.join(output_dir, "step2", "fusion")
    os.makedirs(yfy_output_dir, exist_ok=True)

    # 扫描图像
    image_ids = set()
    for f in os.listdir(original_dir):
        m = re.match(r"(image\d+)", os.path.splitext(f)[0])
        if m:
            image_ids.add(m.group(1))
    image_ids = sorted(image_ids, key=lambda x: int(x.replace("image", "")))

    if not image_ids:
        print(f"[ERROR] 在 {original_dir} 中找不到图片")
        return

    print(f"  图片数: {len(image_ids)}")
    print(f"  算法: GrabCut (5+3 iter) + 凸包约束 + 头发精修")
    print(f"  原图: {original_dir}")
    print(f"  种子: {xbw_dir}\n")

    for img_id in image_ids:
        print(f"  [{img_id}]")
        process_one(img_id, original_dir, light_fixed_dir, xbw_dir, yfy_output_dir)

    # 融合
    print(f"\n  --- 融合 yfy + xbw ---")
    batch_fusion(yfy_output_dir, xbw_dir, fusion_output_dir)

    print(f"\n  ✓ Step 2 完成!")
    return yfy_output_dir, fusion_output_dir


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE, "data")
    STEP1_OUT = os.path.join(BASE, "outputs", "step1")
    OUTPUT_DIR = os.path.join(BASE, "outputs")
    run_step2(DATA_DIR, STEP1_OUT, OUTPUT_DIR)
