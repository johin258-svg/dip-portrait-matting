# -*- coding: utf-8 -*-
"""
AI / 商用抠图对照组

提供多种 AI 抠图方法的统一接口，作为传统 DIP 方法的对照组:
  - rembg (u2net / u2net_human_seg / isnet-general-use)
  - MODNet (onnx 推理)
  - MediaPipe 自拍分割
  - 商用工具结果加载 (remove.bg 等)

用法:
    from ai_baseline import run_rembg_baseline, run_u2net_baseline

    mask, result = run_rembg_baseline("data/input/portrait.jpg", "outputs/ai/")
    mask, result = run_u2net_baseline("data/input/portrait.jpg", "outputs/ai/")
    commercial = load_commercial_result("portrait1", "data/commercial/")
"""

import os
import sys
import time
import numpy as np
import cv2

# 添加 pipeline 根目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.io_utils import imread, imwrite


# ============================================================
#  内部工具
# ============================================================

def _ensure_output_dir(output_dir):
    """确保输出目录存在"""
    os.makedirs(output_dir, exist_ok=True)


def _image_name(image_path):
    """从路径提取纯文件名 (无扩展名)"""
    base = os.path.splitext(os.path.basename(image_path))[0]
    return base


def _load_image(image_path, max_size=1024):
    """加载图像，可选缩放到最大边长"""
    img = imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def _mask_from_alpha(rgba):
    """从 RGBA 图像提取 alpha 通道作为 mask (0-255)"""
    if rgba.shape[2] == 4:
        return rgba[:, :, 3]
    return np.ones(rgba.shape[:2], dtype=np.uint8) * 255


def _compose_rgba(image_bgr, mask):
    """将 BGR 图像与 mask 合成 RGBA"""
    if mask.shape[:2] != image_bgr.shape[:2]:
        mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]))
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    rgba = np.dstack([image_bgr, mask])
    return rgba


# ============================================================
#  rembg 基线
# ============================================================

def run_rembg_baseline(image_path, output_dir, model_name="u2net"):
    """
    使用 rembg 进行抠图

    参数:
        image_path:  输入图像路径
        output_dir:  输出目录
        model_name:  rembg 模型名
                     - "u2net" (默认, 通用)
                     - "u2net_human_seg" (人像专用)
                     - "isnet-general-use" (ISNet, 更精细)
                     - "sam" (SAM, 若已安装)

    返回:
        mask:   二值/软 mask (H, W) uint8 0-255
        result: RGBA 透明背景图 (H, W, 4)
    """
    _ensure_output_dir(output_dir)
    name = _image_name(image_path)

    try:
        from rembg import remove, new_session
    except ImportError:
        raise ImportError("请先安装 rembg: pip install rembg")

    img = _load_image(image_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]  # 去掉已有 alpha

    t0 = time.time()
    session = new_session(model_name)
    output = remove(img, session=session, alpha_matting=True,
                    alpha_matting_foreground_threshold=240,
                    alpha_matting_background_threshold=10,
                    alpha_matting_erode_size=10)
    elapsed = time.time() - t0

    mask = _mask_from_alpha(output)
    result = _compose_rgba(img, output[:, :, 3] if output.shape[2] == 4 else mask)

    # 保存
    mask_path = os.path.join(output_dir, f"{name}_rembg_{model_name}_mask.png")
    result_path = os.path.join(output_dir, f"{name}_rembg_{model_name}_cutout.png")
    imwrite(mask_path, mask)
    imwrite(result_path, result)

    print(f"  [rembg] {name} | model={model_name} | {elapsed:.2f}s")
    return mask, result


def run_u2net_baseline(image_path, output_dir):
    """
    U²-Net 人像抠图 (rembg u2net_human_seg 后端)

    这是 run_rembg_baseline 的便捷封装，使用人像专用模型。
    """
    return run_rembg_baseline(image_path, output_dir, model_name="u2net_human_seg")


# ============================================================
#  MODNet 基线 (ONNX 推理, 不依赖 rembg)
# ============================================================

_MODNET_SESSION = None
_MODNET_URL = ("https://github.com/ZHKKKe/MODNet/releases/download/"
               "v1.0.0/modnet_photographic_portrait_matting.onnx")


def _get_modnet_session():
    """懒加载 MODNet ONNX 会话"""
    global _MODNET_SESSION
    if _MODNET_SESSION is not None:
        return _MODNET_SESSION

    import urllib.request

    # 模型缓存路径
    cache_dir = os.path.join(os.path.dirname(__file__), ".model_cache")
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, "modnet.onnx")

    if not os.path.exists(model_path):
        print(f"  [MODNet] 下载模型到 {model_path} ...")
        urllib.request.urlretrieve(_MODNET_URL, model_path)
        print(f"  [MODNet] 下载完成")

    import onnxruntime
    _MODNET_SESSION = onnxruntime.InferenceSession(
        model_path, providers=["CPUExecutionProvider"]
    )
    return _MODNET_SESSION


def run_modnet_baseline(image_path, output_dir):
    """
    MODNet 人像抠图 (ONNX 直接推理, 无需 rembg)

    返回:
        mask:   软 alpha mask (H, W) uint8 0-255
        result: RGBA 透明背景图
    """
    _ensure_output_dir(output_dir)
    name = _image_name(image_path)

    img_orig = _load_image(image_path)
    h, w = img_orig.shape[:2]

    # MODNet 需要 512x512 输入
    ref_size = 512
    im_h, im_w = h, w
    if max(h, w) > ref_size:
        scale = ref_size / max(h, w)
        im_h, im_w = int(h * scale), int(w * scale)

    img = cv2.resize(img_orig, (im_w, im_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    # 填充到 32 的倍数
    pad_h = (32 - im_h % 32) % 32
    pad_w = (32 - im_w % 32) % 32
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    # NHWC → NCHW → normalize
    blob = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
    blob = (blob - 0.5) / 0.5

    t0 = time.time()
    session = _get_modnet_session()
    outputs = session.run(None, {"input": blob.astype(np.float32)})
    elapsed = time.time() - t0

    matte = outputs[0][0, 0]  # (H, W)
    matte = matte[:im_h, :im_w]  # 去掉 padding
    matte = np.clip(matte * 255, 0, 255).astype(np.uint8)

    # 缩放回原始尺寸
    if (im_h, im_w) != (h, w):
        matte = cv2.resize(matte, (w, h), interpolation=cv2.INTER_LINEAR)

    result = _compose_rgba(img_orig[:, :, :3], matte)

    mask_path = os.path.join(output_dir, f"{name}_modnet_mask.png")
    result_path = os.path.join(output_dir, f"{name}_modnet_cutout.png")
    imwrite(mask_path, matte)
    imwrite(result_path, result)

    print(f"  [MODNet] {name} | {elapsed:.2f}s")
    return matte, result


# ============================================================
#  MediaPipe 自拍分割
# ============================================================

def run_mediapipe_baseline(image_path, output_dir):
    """
    MediaPipe Selfie Segmentation

    返回:
        mask:   软 mask (H, W) uint8 0-255
        result: RGBA 透明背景图
    """
    _ensure_output_dir(output_dir)
    name = _image_name(image_path)

    try:
        import mediapipe as mp
    except ImportError:
        raise ImportError("请先安装 mediapipe: pip install mediapipe")

    img = _load_image(image_path)
    h, w = img.shape[:2]

    mp_selfie = mp.solutions.selfie_segmentation
    with mp_selfie.SelfieSegmentation(model_selection=1) as segmenter:
        t0 = time.time()
        rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        results = segmenter.process(rgb)
        elapsed = time.time() - t0

    mask = (results.segmentation_mask * 255).astype(np.uint8)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h))
    result = _compose_rgba(img[:, :, :3], mask)

    mask_path = os.path.join(output_dir, f"{name}_mediapipe_mask.png")
    result_path = os.path.join(output_dir, f"{name}_mediapipe_cutout.png")
    imwrite(mask_path, mask)
    imwrite(result_path, result)

    print(f"  [MediaPipe] {name} | {elapsed:.2f}s")
    return mask, result


# ============================================================
#  商用结果加载
# ============================================================

def load_commercial_result(image_name, commercial_dir):
    """
    加载商用抠图工具的预计算输出

    约定: 文件放在 commercial_dir/ 下，以 image_name 开头:
        - {image_name}_commercial.png / .jpg   (透明背景抠图)
        - {image_name}_commercial_mask.png     (mask)

    参数:
        image_name:      图像标识 (如 "image3")
        commercial_dir:  商用结果目录

    返回:
        mask:   二值/灰度 mask 或 None
        result: 抠图结果 (BGR/RGBA) 或 None
    """
    result = None
    mask = None

    for ext in [".png", ".jpg", ".jpeg"]:
        # mask
        mask_path = os.path.join(commercial_dir, f"{image_name}_commercial_mask{ext}")
        if os.path.exists(mask_path):
            m = imread(mask_path)
            if m is not None:
                mask = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY) if len(m.shape) == 3 else m
            break

    for ext in [".png", ".jpg", ".jpeg"]:
        result_path = os.path.join(commercial_dir, f"{image_name}_commercial{ext}")
        if os.path.exists(result_path):
            result = imread(result_path)
            break

    if result is None and mask is None:
        print(f"  [commercial] 未找到 {image_name} 的商用结果 (在 {commercial_dir}/)")
    else:
        print(f"  [commercial] 已加载 {image_name}")

    return mask, result


# ============================================================
#  批量运行所有 AI 基线
# ============================================================

def run_all_baselines(image_path, output_dir,
                      methods=("rembg", "rembg_human", "modnet", "mediapipe")):
    """
    对单张图片运行所有可用的 AI 基线方法

    返回:
        dict: {method_name: {"mask": ..., "result": ..., "time": ...}}
    """
    results = {}

    if "rembg" in methods:
        try:
            t0 = time.time()
            mask, result = run_rembg_baseline(image_path, output_dir, model_name="u2net")
            results["rembg_u2net"] = {"mask": mask, "result": result,
                                       "time": time.time() - t0}
        except Exception as e:
            print(f"  [SKIP] rembg_u2net: {e}")

    if "rembg_human" in methods:
        try:
            t0 = time.time()
            mask, result = run_rembg_baseline(image_path, output_dir,
                                              model_name="u2net_human_seg")
            results["rembg_human"] = {"mask": mask, "result": result,
                                       "time": time.time() - t0}
        except Exception as e:
            print(f"  [SKIP] rembg_human: {e}")

    if "modnet" in methods:
        try:
            t0 = time.time()
            mask, result = run_modnet_baseline(image_path, output_dir)
            results["modnet"] = {"mask": mask, "result": result,
                                  "time": time.time() - t0}
        except Exception as e:
            print(f"  [SKIP] modnet: {e}")

    if "mediapipe" in methods:
        try:
            t0 = time.time()
            mask, result = run_mediapipe_baseline(image_path, output_dir)
            results["mediapipe"] = {"mask": mask, "result": result,
                                     "time": time.time() - t0}
        except Exception as e:
            print(f"  [SKIP] mediapipe: {e}")

    return results


# ============================================================
#  CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI 抠图基线")
    parser.add_argument("image", help="输入图像路径")
    parser.add_argument("-o", "--output", default="outputs/ai_baselines",
                        help="输出目录")
    parser.add_argument("-m", "--method",
                        choices=["rembg", "rembg_human", "modnet", "mediapipe", "all"],
                        default="all", help="方法选择")
    args = parser.parse_args()

    if args.method == "all":
        run_all_baselines(args.image, args.output)
    elif args.method == "rembg":
        run_rembg_baseline(args.image, args.output)
    elif args.method == "rembg_human":
        run_rembg_baseline(args.image, args.output, model_name="u2net_human_seg")
    elif args.method == "modnet":
        run_modnet_baseline(args.image, args.output)
    elif args.method == "mediapipe":
        run_mediapipe_baseline(args.image, args.output)
