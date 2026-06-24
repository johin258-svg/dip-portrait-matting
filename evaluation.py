# -*- coding: utf-8 -*-
"""
实验评估模块

提供:
  - 分割评价指标: IoU, Dice, Precision, Recall
  - 边缘质量评估
  - 毛发保留度 / 背景残留度 启发式估计
  - 运行时间测量
  - 对比网格可视化
  - 指标 CSV 导出

用法:
    from evaluation import iou_score, dice_score, create_comparison_grid

    iou = iou_score(pred_mask, gt_mask)
    dice = dice_score(pred_mask, gt_mask)
"""

import os
import sys
import time
import csv
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.io_utils import imread, imwrite, load_mask_binary, load_mask_gray
from utils.vis_utils import save_comparison_grid


# ============================================================
#  基础指标
# ============================================================

def _to_binary(mask, threshold=127):
    """将 mask 统一转为二值 uint8 0/1 (或 0/255)"""
    if mask is None:
        return None
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.dtype == bool:
        return mask.astype(np.uint8) * 255
    if mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    _, binary = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    return binary // 255  # → 0/1


def _align_masks(pred, gt):
    """对齐尺寸，返回 (pred, gt) tuple"""
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    return pred, gt


def iou_score(pred_mask, gt_mask):
    """
    Intersection over Union (Jaccard Index)

    参数:
        pred_mask: 预测 mask (H, W), 灰度/二值
        gt_mask:   ground truth mask (H, W)

    返回:
        float: IoU ∈ [0, 1]
    """
    pred = _to_binary(pred_mask)
    gt = _to_binary(gt_mask)
    if pred is None or gt is None:
        return 0.0
    pred, gt = _align_masks(pred, gt)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0  # 两张空 mask 视为完美一致
    return float(intersection / union)


def dice_score(pred_mask, gt_mask):
    """
    Dice Coefficient (F1 Score)

    Dice = 2 * |P ∩ G| / (|P| + |G|)
    """
    pred = _to_binary(pred_mask)
    gt = _to_binary(gt_mask)
    if pred is None or gt is None:
        return 0.0
    pred, gt = _align_masks(pred, gt)

    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0
    return float(2 * intersection / total)


def precision_recall(pred_mask, gt_mask):
    """
    计算 Precision 和 Recall

    返回:
        (precision, recall): float tuple
    """
    pred = _to_binary(pred_mask)
    gt = _to_binary(gt_mask)
    if pred is None or gt is None:
        return 0.0, 0.0
    pred, gt = _align_masks(pred, gt)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(precision), float(recall)


def f1_score(pred_mask, gt_mask):
    """F1 = 2 * P * R / (P + R) — 等价于 dice_score"""
    return dice_score(pred_mask, gt_mask)


# ============================================================
#  边缘质量
# ============================================================

def estimate_edge_quality(mask):
    """
    启发式边缘自然度评估

    思路:
      1. 提取 mask 边缘
      2. 对边缘像素沿法线方向采样，计算梯度分布
      3. 边缘过渡越平滑 → 分数越高 (软 alpha 优于硬二值)

    参数:
        mask: (H, W) uint8 软 alpha 0-255

    返回:
        dict: {
            "edge_smoothness":  边缘平滑度 (0-1, 越高越平滑)
            "edge_complexity":  边缘复杂度 (归一化)
            "softness_ratio":   软过渡像素占比
        }
    """
    if mask is None:
        return {"edge_smoothness": 0.0, "edge_complexity": 0.0, "softness_ratio": 0.0}

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = mask.astype(np.float32) / 255.0

    # 使用 Canny 检测边缘
    mask_u8 = (mask * 255).astype(np.uint8)
    edges = cv2.Canny(mask_u8, 30, 100).astype(np.float32) / 255.0

    edge_px = edges.sum()
    if edge_px < 10:
        return {"edge_smoothness": 1.0, "edge_complexity": 0.0, "softness_ratio": 0.0}

    # 边缘复杂度: 边缘像素占总像素比 (归一化, 复杂度适中更好)
    total_px = mask.shape[0] * mask.shape[1]
    edge_complexity = min(edge_px / (total_px * 0.02), 1.0)

    # 软过渡像素占比: alpha 在 (0.1, 0.9) 之间的像素
    soft_px = np.sum((mask > 0.05) & (mask < 0.95))
    softness_ratio = soft_px / max(edge_px, 1)

    # 边缘平滑度: 软过渡越多越平滑
    edge_smoothness = min(softness_ratio * 2.0, 1.0)

    return {
        "edge_smoothness":  round(float(edge_smoothness), 4),
        "edge_complexity":  round(float(edge_complexity), 4),
        "softness_ratio":   round(float(softness_ratio), 4),
    }


# ============================================================
#  毛发保留度 / 背景残留度 (启发式)
# ============================================================

def estimate_hair_preservation(mask, edge_band_width=15):
    """
    启发式毛发保留度估计

    思路: 边缘带内的高频细节量。高频越多 → 毛发保留越好。

    返回:
        float: 毛发保留度分数 (0-1)
    """
    if mask is None:
        return 0.0
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask_u8 = mask if mask.dtype == np.uint8 else (mask * 255).astype(np.uint8)

    # 取边缘带
    edges = cv2.Canny(mask_u8, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (edge_band_width, edge_band_width))
    edge_band = cv2.dilate(edges, kernel, iterations=1)

    if edge_band.sum() == 0:
        return 1.0

    # 边缘带内 Laplacian 方差 → 高频细节量
    lap = cv2.Laplacian(mask_u8.astype(np.float32), cv2.CV_32F)
    band_detail = np.std(lap[edge_band > 0])

    # 全局 Laplacian 方差作参照
    global_detail = np.std(lap) + 1e-8

    score = min(band_detail / global_detail, 2.0) / 2.0
    return round(float(score), 4)


def estimate_background_residue(mask, border_width=5):
    """
    启发式背景残留度估计

    思路: 图像边缘区域的非零 mask 占比。边缘靠近背景 → 残留越多。

    返回:
        float: 背景残留分数 (0-1, 越低越好)
    """
    if mask is None:
        return 1.0
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_u8 = mask if mask.dtype == np.uint8 else (mask * 255).astype(np.uint8)

    h, w = mask_u8.shape
    border_mask = np.zeros((h, w), dtype=np.uint8)
    border_mask[:border_width, :] = 1
    border_mask[-border_width:, :] = 1
    border_mask[:, :border_width] = 1
    border_mask[:, -border_width:] = 1

    border_px = border_mask.sum()
    fg_in_border = (mask_u8[border_mask > 0] > 30).sum()

    residue = fg_in_border / max(border_px, 1)
    return round(float(residue), 4)


# ============================================================
#  运行时间测量
# ============================================================

def measure_runtime(func, *args, **kwargs):
    """
    测量函数运行时间

    用法:
        result, elapsed = measure_runtime(run_rembg_baseline, img_path, out_dir)

    返回:
        (func_return_value, elapsed_seconds)
    """
    t0 = time.perf_counter()
    ret = func(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return ret, elapsed


# ============================================================
#  对比网格可视化
# ============================================================

def create_comparison_grid(results, output_path, cols=4, figsize=(20, 15)):
    """
    生成多方法对比网格图

    参数:
        results: dict, {
            "方法名": {"image": BGR图像 或 None,
                      "mask": 灰度/二值 mask 或 None,
                      "result": RGBA/BGR 抠图结果 或 None},
            ...
        }
        output_path: 保存路径
        cols:        每行列数
        figsize:     matplotlib figure 尺寸

    输出:
        保存一张包含所有方法结果和 mask 的对比图

    示例:
        results = {
            "原图":       {"image": img},
            "肤色检测":    {"mask": skin_mask},
            "GrabCut":    {"mask": gc_mask, "result": gc_result},
            "传统融合":    {"mask": fusion_mask, "result": fusion_result},
            "rembg":      {"mask": ai_mask, "result": ai_result},
        }
        create_comparison_grid(results, "outputs/comparison.png")
    """
    images = []
    titles = []

    for method_name, data in results.items():
        if "image" in data and data["image"] is not None:
            images.append(data["image"])
            titles.append(f"{method_name}")
        if "mask" in data and data["mask"] is not None:
            mask = data["mask"]
            if len(mask.shape) == 2:
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            images.append(mask)
            titles.append(f"{method_name} - Mask")
        if "result" in data and data["result"] is not None:
            images.append(data["result"])
            titles.append(f"{method_name} - 结果")

    if not images:
        print("[evaluation] 无图像可绘制")
        return

    save_comparison_grid(images, titles, output_path, cols=cols, figsize=figsize)
    print(f"  [evaluation] 对比图已保存: {output_path}")


# ============================================================
#  完整评估流程
# ============================================================

def evaluate_one(pred_mask, gt_mask, method_name="unknown",
                 pred_result=None, runtime=None):
    """
    对单个预测 mask 进行全面评估

    返回:
        dict: 包含所有指标的字典
    """
    metrics = {"method": method_name}

    # 基础指标 (需 GT)
    if gt_mask is not None:
        metrics["iou"]       = round(iou_score(pred_mask, gt_mask), 4)
        metrics["dice"]      = round(dice_score(pred_mask, gt_mask), 4)
        precision, recall    = precision_recall(pred_mask, gt_mask)
        metrics["precision"] = round(precision, 4)
        metrics["recall"]    = round(recall, 4)
    else:
        metrics["iou"] = metrics["dice"] = None
        metrics["precision"] = metrics["recall"] = None

    # 边缘质量
    edge_q = estimate_edge_quality(pred_mask)
    metrics.update(edge_q)

    # 毛发保留度
    metrics["hair_preservation"] = estimate_hair_preservation(pred_mask)

    # 背景残留度
    metrics["bg_residue"] = estimate_background_residue(pred_mask)

    # 运行时间
    if runtime is not None:
        metrics["runtime_s"] = round(runtime, 3)

    return metrics


def evaluate_all_methods(predictions, gt_mask=None):
    """
    对多个方法的预测进行批量评估

    参数:
        predictions: dict, {
            "方法名": {"mask": ..., "result": ..., "time": ...},
            ...
        }
        gt_mask: 共同 ground truth

    返回:
        list[dict]: 每个方法的评估结果
    """
    all_metrics = []
    for method, data in predictions.items():
        m = evaluate_one(
            pred_mask=data.get("mask"),
            gt_mask=gt_mask,
            method_name=method,
            pred_result=data.get("result"),
            runtime=data.get("time"),
        )
        all_metrics.append(m)
    return all_metrics


# ============================================================
#  CSV 导出
# ============================================================

_METRIC_FIELDS = [
    "method", "iou", "dice", "precision", "recall",
    "edge_smoothness", "edge_complexity", "softness_ratio",
    "hair_preservation", "bg_residue", "runtime_s"
]


def export_metrics_csv(metrics_list, csv_path):
    """
    将评估指标导出为 CSV 文件

    参数:
        metrics_list: list[dict], evaluate_all_methods 的返回值
        csv_path:     CSV 输出路径
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # 收集所有出现的字段
    all_fields = set()
    for m in metrics_list:
        all_fields.update(m.keys())
    fields = [f for f in _METRIC_FIELDS if f in all_fields]
    fields += sorted(all_fields - set(_METRIC_FIELDS))

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in metrics_list:
            writer.writerow(row)

    print(f"  [evaluation] CSV 已导出: {csv_path}")


# ============================================================
#  综合实验报告生成
# ============================================================

def generate_report(all_results, output_path):
    """
    生成 Markdown 格式实验报告摘要

    参数:
        all_results: dict, {
            "image_name": {
                "method_A": {"mask": ..., "metrics": {...}},
                "method_B": {...},
            },
            ...
        }
        output_path: 报告保存路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = []
    lines.append("# DIP 人像抠图 — 实验对比报告\n")
    lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 按图像汇总
    for img_name, methods in all_results.items():
        lines.append(f"## {img_name}\n")
        lines.append("| 方法 | IoU | Dice | Precision | Recall | "
                     "边缘平滑 | 毛发保留 | 背景残留 | 耗时(s) |")
        lines.append("|------|-----|------|-----------|--------|"
                     "----------|----------|----------|---------|")

        for method, data in methods.items():
            m = data.get("metrics", {})
            row = (f"| {method} "
                   f"| {m.get('iou', '-')} "
                   f"| {m.get('dice', '-')} "
                   f"| {m.get('precision', '-')} "
                   f"| {m.get('recall', '-')} "
                   f"| {m.get('edge_smoothness', '-')} "
                   f"| {m.get('hair_preservation', '-')} "
                   f"| {m.get('bg_residue', '-')} "
                   f"| {m.get('runtime_s', '-')} |")
            lines.append(row)

        lines.append("")

    # 整体总结
    lines.append("## 总结\n")
    lines.append("### 传统方法优势\n")
    lines.append("- **可解释性强**: 每一步 (肤色检测→GrabCut→形态学→引导滤波) 都有明确的物理意义\n")
    lines.append("- **依赖少**: 仅需 OpenCV + NumPy，无需 GPU 和大型模型文件\n")
    lines.append("- **流程可控**: 每步参数可调，中间结果可检查，适合课程教学\n")
    lines.append("- **对简单场景效果好**: 在简单背景、良好光照下，传统方法与 AI 方法差距不大\n")

    lines.append("### 传统方法不足\n")
    lines.append("- **复杂背景**: 人物与背景颜色接近时，GrabCut 易误分割\n")
    lines.append("- **毛发细节**: 虽然 Frangi+Gabor 能检测发丝，但精细度仍不及 AI 方法\n")
    lines.append("- **光照变化**: 暗光/高噪声条件下肤色检测准确率下降\n")
    lines.append("- **泛化能力**: 参数需针对不同场景调整，不如 AI 方法鲁棒\n")

    lines.append("### 课程价值\n")
    lines.append("通过本次实验，系统学习了数字图像处理的核心技术链: "
                 "颜色空间转换、形态学操作、图割分割、频域滤波、引导滤波等。"
                 "这些方法构成了理解现代计算机视觉的基础。\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  [evaluation] 报告已保存: {output_path}")


# ============================================================
#  CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="评估模块 CLI")
    sub = parser.add_subparsers(dest="cmd")

    # evaluate 子命令
    p_eval = sub.add_parser("evaluate", help="评估两个 mask")
    p_eval.add_argument("pred", help="预测 mask 路径")
    p_eval.add_argument("gt", help="ground truth mask 路径")
    p_eval.add_argument("--method", default="unknown", help="方法名")

    # compare 子命令
    p_comp = sub.add_parser("compare", help="生成对比图")
    p_comp.add_argument("masks", nargs="+", help="mask 路径列表")
    p_comp.add_argument("--names", nargs="+", help="对应的名称列表")
    p_comp.add_argument("-o", "--output", default="outputs/comparison.png",
                         help="输出路径")

    args = parser.parse_args()

    if args.cmd == "evaluate":
        pred = load_mask_gray(args.pred)
        gt = load_mask_binary(args.gt)
        metrics = evaluate_one(pred, gt, args.method)
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        export_metrics_csv([metrics], "outputs/metrics/eval_result.csv")

    elif args.cmd == "compare":
        results = {}
        for i, mp in enumerate(args.masks):
            name = args.names[i] if args.names and i < len(args.names) else f"mask_{i}"
            mask = load_mask_gray(mp)
            results[name] = {"mask": mask}
        create_comparison_grid(results, args.output)
