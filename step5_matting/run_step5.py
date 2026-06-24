# -*- coding: utf-8 -*-
"""
Step 5 入口 — 抠图输出与背景替换 (Part C-3)
  输入: data/original_images/imageX.jpg (前景原图)
        outputs/step3/imageX_cleaned.png (软 alpha mask)
  输出: outputs/step5/imageX_transparent.png  (RGBA 透明背景)
                         /solid_white.jpg 等 (纯色背景)
                         /custom_bg.jpg    (自定义背景)
"""

import os, sys, re
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io_utils import imread, imwrite, load_mask_gray, find_original_image
from step5_matting.matting import cutout_transparent, cutout_solid, cutout_custom_bg


def process_one(image_id, img_dir, mask_dir, output_dir, custom_bg=None):
    """处理单张: 透明 PNG + 纯色背景 + 自定义背景"""
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

    # 透明 PNG
    imwrite(os.path.join(output_dir, f"{image_id}_transparent.png"),
            cutout_transparent(img, mask))

    # 纯色背景
    for color in ["white", "red", "blue", "green"]:
        imwrite(os.path.join(output_dir, f"{image_id}_solid_{color}.jpg"),
                cutout_solid(img, mask, color))

    # 自定义背景
    if custom_bg is not None:
        imwrite(os.path.join(output_dir, f"{image_id}_custom_bg.jpg"),
                cutout_custom_bg(img, mask, custom_bg))

    tag = " +custom_bg" if custom_bg is not None else ""
    print(f"  [OK] {image_id} → transparent + solid×4{tag}")


def run_step5(data_dir, step3_output_dir, output_dir):
    """批量运行 Step 5"""
    print("=" * 60)
    print("  Step 5: 抠图输出与背景替换 (Part C-3)")
    print("=" * 60)

    img_dir = os.path.join(data_dir, "original_images")
    mask_dir = step3_output_dir
    step5_out = os.path.join(output_dir, "step5")
    os.makedirs(step5_out, exist_ok=True)

    # 加载自定义背景
    custom_bg = None
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bg_path = os.path.join(base_dir, "background.jpg")
    if os.path.exists(bg_path):
        custom_bg = imread(bg_path)
        if custom_bg is not None:
            print(f"  自定义背景: {bg_path} ({custom_bg.shape[1]}x{custom_bg.shape[0]})")
    if custom_bg is None:
        print(f"  [INFO] 未找到 background.jpg, 跳过自定义背景")

    pattern = re.compile(r"(image\d+)_cleaned\.png")
    image_ids = sorted(
        {m.group(1) for f in os.listdir(mask_dir) if (m := pattern.match(f))},
        key=lambda x: int(x.replace("image", ""))
    )

    if not image_ids:
        print(f"[ERROR] 在 {mask_dir} 中找不到 cleaned mask")
        return

    print(f"  图片数: {len(image_ids)}")
    print(f"  输出: 透明PNG(RGBA) + 纯色×4(白/红/蓝/绿) + 自定义背景")
    print(f"  输出目录: {step5_out}\n")

    for img_id in image_ids:
        print(f"  [{img_id}]")
        process_one(img_id, img_dir, mask_dir, step5_out, custom_bg)

    print(f"\n  ✓ Step 5 完成!")
    return step5_out


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE, "data")
    STEP3_OUT = os.path.join(BASE, "outputs", "step3")
    OUTPUT_DIR = os.path.join(BASE, "outputs")
    run_step5(DATA_DIR, STEP3_OUT, OUTPUT_DIR)
