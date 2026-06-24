# -*- coding: utf-8 -*-
"""
Step 1 入口 — 图像预处理 + 肤色检测 + 光照校正 (Part A)
  输入: data/original_images/imageX.jpg
  输出: outputs/step1/xbw_masks/imageX_xbw_mask.jpg
        outputs/step1/light_fixed/imageX_xbw_light2.jpg
"""

import os, sys, re
import cv2
import numpy as np

# 添加 pipeline 根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io_utils import imread, imwrite, load_mask_binary
from step1_preprocess.preprocess import (
    resize_image, gamma_correction, clahe_enhancement, median_denoise
)
from step1_preprocess.skin_detection import detect_skin
from step1_preprocess.lighting import (
    homomorphic_filter, illumination_correction, retinex_ssr
)


def process_one(image_id, img_dir, output_xbw_dir, output_light_dir, max_size=1024):
    """处理单张: 预处理 → 肤色检测 → 光照校正"""
    # 1. 读取原图
    original = None
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        p = os.path.join(img_dir, f"{image_id}{ext}")
        if os.path.exists(p):
            original = imread(p)
            break
    if original is None:
        print(f"  [SKIP] 找不到原图: {image_id}")
        return None

    # 2. 缩放到最大尺寸以内
    img = resize_image(original, max_size)

    # 3. 肤色检测
    skin_mask = detect_skin(img, mode="or", refine=True)
    xbw_path = os.path.join(output_xbw_dir, f"{image_id}_xbw_mask.jpg")
    imwrite(xbw_path, skin_mask)

    # 4. 光照校正 (homomorphic 效果最好，作为 light2)
    light2 = resize_image(homomorphic_filter(img, r=40), max_size)
    light2_path = os.path.join(output_light_dir, f"{image_id}_xbw_light2.jpg")
    imwrite(light2_path, light2)

    # 5. 其他光照校正 (light1 = illumination, light3 = retinex)
    light1 = resize_image(illumination_correction(img, ks=31), max_size)
    imwrite(os.path.join(output_light_dir, f"{image_id}_xbw_light1.jpg"), light1)

    light3 = resize_image(retinex_ssr(img, sigma=50), max_size)
    imwrite(os.path.join(output_light_dir, f"{image_id}_xbw_light3.jpg"), light3)

    skin_px = int(np.sum(skin_mask > 0))
    skin_ratio = skin_px / skin_mask.size
    print(f"  [OK] {image_id} | skin={100*skin_ratio:.1f}% ({skin_px}px)")
    return skin_mask


def run_step1(data_dir, output_dir):
    """批量运行 Step 1"""
    print("=" * 60)
    print("  Step 1: 图像预处理 + 肤色检测 + 光照校正 (Part A)")
    print("=" * 60)

    img_dir = os.path.join(data_dir, "original_images")
    output_xbw = os.path.join(output_dir, "step1", "xbw_masks")
    output_light = os.path.join(output_dir, "step1", "light_fixed")
    os.makedirs(output_xbw, exist_ok=True)
    os.makedirs(output_light, exist_ok=True)

    # 扫描图像
    image_ids = set()
    for f in os.listdir(img_dir):
        m = re.match(r"(image\d+)", os.path.splitext(f)[0])
        if m:
            image_ids.add(m.group(1))
    image_ids = sorted(image_ids, key=lambda x: int(x.replace("image", "")))

    if not image_ids:
        print(f"[ERROR] 在 {img_dir} 中找不到图片 (imageX 格式)")
        return

    print(f"  图片数: {len(image_ids)}")
    print(f"  肤色检测: HSV + YCrCb (OR融合)")
    print(f"  光照校正: Homomorphic / Illumination / Retinex")
    print(f"  输出: {output_xbw}, {output_light}\n")

    for img_id in image_ids:
        print(f"  [{img_id}]")
        process_one(img_id, img_dir, output_xbw, output_light)

    print(f"\n  ✓ Step 1 完成!")
    return output_xbw, output_light


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE, "data")
    OUTPUT_DIR = os.path.join(BASE, "outputs")
    run_step1(DATA_DIR, OUTPUT_DIR)
