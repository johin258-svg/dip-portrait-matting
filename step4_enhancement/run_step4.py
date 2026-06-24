# -*- coding: utf-8 -*-
"""
Step 4 入口 — 人像增强 (Part C-2)
  输入: data/original_images/imageX.jpg (原图)
        outputs/step3/imageX_cleaned.png (软 alpha mask)
  输出: outputs/step4/imageX_whiten.jpg / denoise.jpg / sharpen.jpg
        / vintage.jpg / dream.jpg / film.jpg / pink.jpg / enhanced.jpg
"""

import os, sys, re
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io_utils import imread, imwrite, load_mask_gray, find_original_image
from step4_enhancement.enhancement import (
    whiten, denoise_bilateral, sharpen_unsharp, comprehensive_enhance,
    vintage, dream, film, pink
)


def process_one(image_id, img_dir, mask_dir, output_dir):
    """处理单张: 各种增强 + 滤镜"""
    img = find_original_image(image_id, img_dir)
    if img is None:
        print(f"  [SKIP] 找不到原图 {image_id}")
        return

    mask = load_mask_gray(os.path.join(mask_dir, f"{image_id}_cleaned.png"))
    if mask is None:
        print(f"  [SKIP] 找不到 mask {image_id}")
        return

    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    os.makedirs(output_dir, exist_ok=True)

    # 人像区域增强
    imwrite(os.path.join(output_dir, f"{image_id}_whiten.jpg"), whiten(img, mask))
    imwrite(os.path.join(output_dir, f"{image_id}_denoise.jpg"), denoise_bilateral(img, mask))
    imwrite(os.path.join(output_dir, f"{image_id}_sharpen.jpg"), sharpen_unsharp(img, mask))
    imwrite(os.path.join(output_dir, f"{image_id}_enhanced.jpg"), comprehensive_enhance(img, mask))

    # 全图滤镜
    imwrite(os.path.join(output_dir, f"{image_id}_vintage.jpg"), vintage(img))
    imwrite(os.path.join(output_dir, f"{image_id}_dream.jpg"), dream(img))
    imwrite(os.path.join(output_dir, f"{image_id}_film.jpg"), film(img))
    imwrite(os.path.join(output_dir, f"{image_id}_pink.jpg"), pink(img))

    print(f"  [OK] {image_id} → 美白/降噪/锐化/综合/复古/Dream/胶片/粉色")


def run_step4(data_dir, step3_output_dir, output_dir):
    """批量运行 Step 4"""
    print("=" * 60)
    print("  Step 4: 人像增强 (Part C-2)")
    print("=" * 60)

    img_dir = os.path.join(data_dir, "original_images")
    mask_dir = step3_output_dir
    step4_out = os.path.join(output_dir, "step4")
    os.makedirs(step4_out, exist_ok=True)

    pattern = re.compile(r"(image\d+)_cleaned\.png")
    image_ids = sorted(
        {m.group(1) for f in os.listdir(mask_dir) if (m := pattern.match(f))},
        key=lambda x: int(x.replace("image", ""))
    )

    if not image_ids:
        print(f"[ERROR] 在 {mask_dir} 中找不到 cleaned mask")
        return

    print(f"  图片数: {len(image_ids)}")
    print(f"  增强: 美白(B×1.0 G×0.88 R×0.85 + β22)")
    print(f"        双边降噪(7,60,60) / Unsharp锐化(2.2)")
    print(f"  全图滤镜: 复古(暖琥珀+降饱和) / Dream(冷调+增对比)")
    print(f"            胶片(提阴影+颗粒) / 粉色(粉紫+柔光)")
    print(f"  输出: {step4_out}\n")

    for img_id in image_ids:
        print(f"  [{img_id}]")
        process_one(img_id, img_dir, mask_dir, step4_out)

    print(f"\n  ✓ Step 4 完成!")
    return step4_out


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE, "data")
    STEP3_OUT = os.path.join(BASE, "outputs", "step3")
    OUTPUT_DIR = os.path.join(BASE, "outputs")
    run_step4(DATA_DIR, STEP3_OUT, OUTPUT_DIR)
