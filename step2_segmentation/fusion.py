# -*- coding: utf-8 -*-
"""
Step 2 — 掩膜融合 (Part B)
  融合 yfy (GrabCut 人物掩膜) + xbw (肤色掩膜), 互补优势。

融合策略:
  OR  (并集): yfy ∪ xbw — 最大覆盖范围
  AND (交集): yfy ∩ xbw — 最保守, 分析一致性
  Weighted (加权): 0.5*yfy + 0.5*xbw → 阈值100 → OR-like 融合
    保留 yfy 的衣服+头发 + xbw 的皮肤覆盖
"""

import cv2
import numpy as np
import os, re


def load_mask_binary(path, threshold=30):
    """加载掩膜为二值图 (0/255)"""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


def imwrite_chinese(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jpg":
        cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(path)
    else:
        cv2.imencode(".png", img)[1].tofile(path)


def fuse_pair(yfy_path, xbw_path, img_id, output_dir, w_yfy=0.5, w_xbw=0.5):
    """融合一对 yfy + xbw 掩膜"""
    yfy = load_mask_binary(yfy_path)
    xbw = load_mask_binary(xbw_path)
    if yfy is None or xbw is None:
        return None, None, None

    if yfy.shape != xbw.shape:
        xbw = cv2.resize(xbw, (yfy.shape[1], yfy.shape[0]))

    or_mask = cv2.bitwise_or(yfy, xbw)
    and_mask = cv2.bitwise_and(yfy, xbw)
    weighted = cv2.addWeighted(yfy, w_yfy, xbw, w_xbw, 0)
    _, weighted_bin = cv2.threshold(weighted, 100, 255, cv2.THRESH_BINARY)

    or_dir = os.path.join(output_dir, "OR")
    and_dir = os.path.join(output_dir, "AND")
    weighted_dir = os.path.join(output_dir, "weighted")
    for d in [or_dir, and_dir, weighted_dir]:
        os.makedirs(d, exist_ok=True)

    imwrite_chinese(os.path.join(or_dir, f"{img_id}_or_fusion_mask.jpg"), or_mask)
    imwrite_chinese(os.path.join(and_dir, f"{img_id}_and_fusion_mask.jpg"), and_mask)
    # 使用固定文件名, 兼容后续步骤
    imwrite_chinese(os.path.join(weighted_dir,
        f"{img_id}_0.5yfy_0.5xbw_fusion_mask.jpg"), weighted_bin)

    return or_mask, and_mask, weighted_bin


def batch_fusion(yfy_dir, xbw_dir, output_dir, w_yfy=0.5, w_xbw=0.5):
    """批量融合所有匹配的掩膜对"""
    yfy_files = {}
    for f in os.listdir(yfy_dir):
        m = re.match(r"(image\d+)_yfy_mask\.jpg", f)
        if m:
            yfy_files[m.group(1)] = os.path.join(yfy_dir, f)

    xbw_files = {}
    for f in os.listdir(xbw_dir):
        m = re.match(r"(image\d+)_xbw_mask\.jpg", f)
        if m:
            xbw_files[m.group(1)] = os.path.join(xbw_dir, f)

    common_ids = sorted(set(yfy_files.keys()) & set(xbw_files.keys()),
                        key=lambda x: int(x.replace("image", "")))

    if not common_ids:
        print("[ERROR] 没有找到配对掩膜")
        return {}

    print(f"  配对掩膜: {len(common_ids)} 对")
    print(f"  融合权重: yfy={w_yfy}, xbw={w_xbw}\n")

    results = {}
    for img_id in common_ids:
        print(f"  [{img_id}]")
        _, _, weighted = fuse_pair(yfy_files[img_id], xbw_files[img_id],
                                    img_id, output_dir, w_yfy, w_xbw)
        results[img_id] = weighted

    return results
