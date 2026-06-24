# -*- coding: utf-8 -*-
"""
DIP 人像抠图系统 — 面向用户的三界面设计
  界面1: 抠图导出 — 传统方法结果预览 + 导出透明PNG
  界面2: AI对比分析 — 传统 vs AI mask 对比 + 差异图 + 报告导出
  界面3: 背景与滤镜 — 换背景/纯色 + 艺术滤镜 + 导出

启动: streamlit run app.py
"""

import os, sys, time, io, csv
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import streamlit as st

from utils.io_utils import imread, imwrite
from config import OUTPUT_DIR, ORIGINAL_IMAGES

from step1_preprocess.preprocess import resize_image
from step1_preprocess.skin_detection import detect_skin
from step1_preprocess.lighting import homomorphic_filter

from step2_segmentation.grabcut_segment import (
    grabcut_segment, post_process_mask, refine_hair
)

from step3_morphology.morphology import (
    threshold_to_binary, conservative_clean, edge_aware_smooth
)
from step3_morphology.hair_refine import hair_refine

from step4_enhancement.enhancement import (
    whiten, denoise_bilateral, sharpen_unsharp, comprehensive_enhance,
    vintage, dream, film, pink
)

from step5_matting.matting import cutout_transparent, cutout_solid, cutout_custom_bg

from ai_baseline import run_all_baselines
from evaluation import evaluate_one

# ============================================================
#  page config
# ============================================================
st.set_page_config(page_title="人像抠图系统", page_icon="🎨", layout="wide")

# ============================================================
#  helpers
# ============================================================

def rgba_to_white(rgba):
    """RGBA → white-background BGR"""
    if rgba is None or rgba.shape[2] == 3:
        return rgba
    a = rgba[:, :, 3:4].astype(np.float32) / 255.0
    fg = rgba[:, :, :3].astype(np.float32)
    bg = np.ones_like(fg) * 255
    return np.clip(fg * a + bg * (1 - a), 0, 255).astype(np.uint8)


def bgr_to_pil_bytes(bgr):
    """BGR → PNG bytes (for download)"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    buf.seek(0)
    return buf


def rgba_to_pil_bytes(rgba):
    """RGBA → PNG bytes preserving alpha"""
    rgba2 = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)
    buf = io.BytesIO()
    Image.fromarray(rgba2).save(buf, format="PNG")
    buf.seek(0)
    return buf


def mask_diff_viz(mask_a, mask_b, name_a="传统", name_b="AI"):
    """生成 mask 差异可视化: 一致/仅A/仅B 三色图"""
    a = mask_a.copy()
    b = mask_b.copy()
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
    if len(a.shape) == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if len(b.shape) == 3:
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    _, a = cv2.threshold(a, 127, 1, cv2.THRESH_BINARY)
    _, b = cv2.threshold(b, 127, 1, cv2.THRESH_BINARY)

    h, w = a.shape
    viz = np.zeros((h, w, 3), dtype=np.uint8)
    both = (a == 1) & (b == 1)      # 一致 → 白
    only_a = (a == 1) & (b == 0)    # 仅传统 → 绿
    only_b = (a == 0) & (b == 1)    # 仅AI → 红
    viz[both] = (255, 255, 255)
    viz[only_a] = (0, 255, 0)
    viz[only_b] = (255, 0, 0)

    total_fg = both.sum() + only_a.sum() + only_b.sum() + 1e-8
    legend = {
        f"{name_a}∩{name_b} (一致)": both.sum(),
        f"仅{name_a}": only_a.sum(),
        f"仅{name_b}": only_b.sum(),
        f"{name_a}占比": f"{only_a.sum()/total_fg*100:.1f}%",
        f"{name_b}占比": f"{only_b.sum()/total_fg*100:.1f}%",
    }
    return viz, legend


@st.cache_resource
def get_test_images():
    if not os.path.isdir(ORIGINAL_IMAGES):
        return []
    return sorted([os.path.join(ORIGINAL_IMAGES, f)
                   for f in os.listdir(ORIGINAL_IMAGES)
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])


def get_active_matting():
    """根据侧边栏选择返回当前生效的抠图结果
    Returns: {"proc": BGR, "mask": gray, "white_bgr": BGR, "transparent": RGBA, "name": str}
    """
    choice = st.session_state.get("pipeline_choice", "传统方法")
    proc = st.session_state.proc_img

    if choice == "传统方法" or not st.session_state.ai_results:
        return {
            "proc": proc,
            "mask": st.session_state.cleaned_mask,
            "white_bgr": st.session_state.white_bgr,
            "transparent": st.session_state.transparent_rgba,
            "name": "传统方法",
        }
    else:
        ai_data = st.session_state.ai_results.get(choice, {})
        mask = ai_data.get("aligned_mask")
        if mask is None:
            mask = ai_data.get("mask")
        white = ai_data.get("white_bgr")
        transparent = ai_data.get("aligned_transparent")
        if transparent is None:
            transparent = ai_data.get("result")

        # 确保 mask 与 proc 对齐
        if mask is not None and proc is not None and mask.shape[:2] != proc.shape[:2]:
            mask = cv2.resize(mask, (proc.shape[1], proc.shape[0]))

        # 如果没有 white_bgr，用 mask 合成
        if white is None and mask is not None and proc is not None:
            white = cutout_solid(proc, mask, "white")

        return {
            "proc": proc,
            "mask": mask,
            "white_bgr": white,
            "transparent": transparent,
            "name": choice,
        }


# ============================================================
#  session state init
# ============================================================
def init_state():
    defaults = {
        "img_bgr": None, "img_name": "", "img_path": None,
        "proc_img": None, "skin_mask": None, "light_fixed": None,
        "grabcut_mask": None, "fusion_mask": None, "cleaned_mask": None,
        "transparent_rgba": None, "white_bgr": None, "trad_time": 0.0,
        "step4_enhanced": None, "step5_solid": None, "step5_custom": None,
        "ai_results": None, "ai_time": 0.0,
        "gt_mask": None,
        "edit_bg_img": None, "edit_bg_result": None, "edit_filter_result": None,
        "adjust_result": None, "adjust_src": None,
        "pipeline_choice": "传统方法",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# ============================================================
#  pipeline runner (kept lean)
# ============================================================
def run_traditional_pipeline():
    img = st.session_state.img_bgr
    max_sz = 1024
    proc = resize_image(img, max_sz)
    st.session_state.proc_img = proc

    # Step 1
    skin = detect_skin(proc, mode="or", refine=True)
    light = homomorphic_filter(proc, r=40)
    light = resize_image(light, max_sz)
    st.session_state.skin_mask = skin
    st.session_state.light_fixed = light

    # Step 2
    fg = grabcut_segment(proc, skin)
    yfy = post_process_mask(fg, skin)
    yfy = refine_hair(proc, yfy, skin)
    st.session_state.grabcut_mask = yfy
    w = cv2.addWeighted(yfy.astype(np.float32), 0.5,
                         skin.astype(np.float32), 0.5, 0)
    _, fusion = cv2.threshold(w, 100, 255, cv2.THRESH_BINARY)
    fusion = fusion.astype(np.uint8)
    st.session_state.fusion_mask = fusion

    # Step 3
    gray_proc = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    mask_gray = cv2.cvtColor(fusion, cv2.COLOR_BGR2GRAY) if len(fusion.shape) == 3 else fusion
    mask_gray = mask_gray.astype(np.uint8)
    mask_bin = threshold_to_binary(mask_gray, 10)
    mask_bin = mask_bin.astype(np.uint8)
    mask_bin = conservative_clean(mask_bin)
    mask_bin = hair_refine(mask_bin, gray_proc, proc)
    mask_soft = edge_aware_smooth(mask_bin, gray_proc, 5, 1e-4)
    mask_soft[mask_soft < 30] = 0
    st.session_state.cleaned_mask = mask_soft

    # resize mask to proc size
    if mask_soft.shape[:2] != proc.shape[:2]:
        mask_soft = cv2.resize(mask_soft, (proc.shape[1], proc.shape[0]))

    # Step 4
    mask_bin4 = (mask_soft > 30).astype(np.uint8) * 255
    enhanced = {
        "美白": whiten(proc, mask_bin4),
        "降噪(双边)": denoise_bilateral(proc, mask_bin4),
        "锐化(USM)": sharpen_unsharp(proc, mask_bin4),
        "综合增强": comprehensive_enhance(proc, mask_bin4),
        "复古滤镜": vintage(proc),
        "Dream冷调": dream(proc),
        "胶片质感": film(proc),
        "粉色氛围": pink(proc),
    }
    st.session_state.step4_enhanced = enhanced

    # Step 5
    st.session_state.transparent_rgba = cutout_transparent(proc, mask_soft)
    st.session_state.white_bgr = cutout_solid(proc, mask_soft, "white")
    solid = {c: cutout_solid(proc, mask_soft, c) for c in ["white", "red", "blue", "green"]}
    st.session_state.step5_solid = solid

    bgp = os.path.join(os.path.dirname(__file__), "background.jpg")
    if os.path.exists(bgp):
        cbg = imread(bgp)
        if cbg is not None:
            st.session_state.step5_custom = cutout_custom_bg(proc, mask_soft, cbg)


def run_ai_baselines():
    img = st.session_state.img_bgr
    tmpdir = os.path.join(OUTPUT_DIR, "ai_baselines")
    os.makedirs(tmpdir, exist_ok=True)
    tmp = os.path.join(tmpdir, f"{st.session_state.img_name}_input.png")
    imwrite(tmp, img)
    results = run_all_baselines(tmp, tmpdir, methods=["rembg_human", "mediapipe"])
    # 为每个 AI 结果生成白底版 + 对齐后的 mask
    proc = st.session_state.proc_img
    for name, data in results.items():
        ai_mask = data.get("mask")
        ai_result = data.get("result")
        if ai_mask is not None and proc is not None:
            if ai_mask.shape[:2] != proc.shape[:2]:
                ai_mask = cv2.resize(ai_mask, (proc.shape[1], proc.shape[0]))
            data["aligned_mask"] = ai_mask
        if ai_result is not None:
            data["white_bgr"] = rgba_to_white(ai_result)
            # 生成对齐 proc 的透明 RGBA
            if ai_result.shape[:2] != proc.shape[:2]:
                data["aligned_transparent"] = cv2.resize(ai_result, (proc.shape[1], proc.shape[0]))
            else:
                data["aligned_transparent"] = ai_result
    st.session_state.ai_results = results


# ============================================================
#  SIDEBAR
# ============================================================
with st.sidebar:
    st.title("🎨 人像抠图系统")
    st.caption("数字图像处理 · 方案A")

    st.divider()
    st.subheader("📷 图片来源")
    src = st.radio("来源", ["上传图片", "测试图片集"], label_visibility="collapsed")

    if src == "上传图片":
        uf = st.file_uploader("选择照片", type=["jpg", "jpeg", "png", "bmp"])
        if uf is not None:
            arr = np.frombuffer(uf.read(), np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is not None:
                h, w = bgr.shape[:2]
                if h > 4000 or w > 4000:
                    st.warning(f"图片过大 ({w}×{h})，已自动缩放至 4000px")
                    scale = 4000 / max(h, w)
                    bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))
                st.session_state.img_bgr = bgr
                st.session_state.img_name = os.path.splitext(uf.name)[0]
                st.session_state.img_path = None
                for k in ["proc_img", "skin_mask", "light_fixed", "grabcut_mask",
                          "fusion_mask", "cleaned_mask", "transparent_rgba",
                          "white_bgr", "trad_time", "step4_enhanced",
                          "step5_solid", "step5_custom",
                          "ai_results", "ai_time", "gt_mask",
                          "edit_bg_result", "edit_filter_result",
                          "adjust_result", "adjust_src"]:
                    st.session_state[k] = None
                st.session_state.pipeline_choice = "传统方法"
                st.success(f"✓ {uf.name}")
    else:
        tests = get_test_images()
        if tests:
            sel = st.selectbox("选择图片", [os.path.basename(p) for p in tests])
            if st.button("📥 加载"):
                idx = [os.path.basename(p) for p in tests].index(sel)
                bgr = imread(tests[idx])
                if bgr is not None:
                    h, w = bgr.shape[:2]
                    if h > 4000 or w > 4000:
                        st.warning(f"图片过大 ({w}×{h})，已自动缩放至 4000px")
                        scale = 4000 / max(h, w)
                        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))
                st.session_state.img_bgr = bgr
                st.session_state.img_name = os.path.splitext(sel)[0]
                st.session_state.img_path = tests[idx]
                for k in ["proc_img", "skin_mask", "light_fixed", "grabcut_mask",
                          "fusion_mask", "cleaned_mask", "transparent_rgba",
                          "white_bgr", "trad_time", "step4_enhanced",
                          "step5_solid", "step5_custom",
                          "ai_results", "ai_time", "gt_mask",
                          "edit_bg_result", "edit_filter_result",
                          "adjust_result", "adjust_src"]:
                    st.session_state[k] = None
                st.success(f"✓ {sel}")

    st.divider()
    st.subheader("⚙️ 运行")
    btn_run = st.button("🚀 一键运行 (传统 + AI)", use_container_width=True, type="primary")
    st.caption("运行后即可使用全部界面")

    st.divider()
    st.subheader("🔀 后续处理数据源")
    # 可选方法列表
    choice_options = ["传统方法"]
    if st.session_state.ai_results:
        choice_options += list(st.session_state.ai_results.keys())
    if "pipeline_choice" not in st.session_state:
        st.session_state.pipeline_choice = "传统方法"
    # 确保默认值在选项中
    if st.session_state.pipeline_choice not in choice_options:
        st.session_state.pipeline_choice = "传统方法"
    st.radio("选择抠图结果用于换背景/滤镜/调节",
             choice_options, key="pipeline_choice")
    st.caption("界面3、4将使用此处选择的结果")

    st.divider()
    st.subheader("📏 GT Mask (可选)")
    gf = st.file_uploader("上传标准答案", type=["png", "jpg", "jpeg"], key="gt_up")
    if gf is not None:
        gb = np.frombuffer(gf.read(), np.uint8)
        gt = cv2.imdecode(gb, cv2.IMREAD_GRAYSCALE)
        if gt is not None:
            st.session_state.gt_mask = gt
            st.success("GT 已加载")
    if st.button("清除 GT"):
        st.session_state.gt_mask = None


# ============================================================
#  run buttons
# ============================================================
if btn_run:
    if st.session_state.img_bgr is None:
        st.error("请先上传图片")
    else:
        with st.spinner("传统方法 + AI 对照组运行中，请耐心等待..."):
            t0 = time.time()
            run_traditional_pipeline()
            st.session_state.trad_time = time.time() - t0
            run_ai_baselines()
            st.session_state.ai_time = time.time() - t0
        st.rerun()


# ============================================================
#  MAIN
# ============================================================
if st.session_state.img_bgr is None:
    st.info("👈 请先在左侧上传或选择一张照片")
    st.stop()

st.title("🎨 人像抠图系统")
img = st.session_state.img_bgr
st.caption(f"当前图片: **{st.session_state.img_name}**  |  {img.shape[1]}×{img.shape[0]}")

t1, t2, t3, t4 = st.tabs(["📤 抠图导出", "🔬 AI对比分析", "🖼️ 背景与滤镜", "🎚️ 图像调节"])


# ============================================================
#  界面1: 抠图导出
# ============================================================
with t1:
    active = get_active_matting()

    if active["white_bgr"] is None:
        st.warning("请先在左侧点击「🚀 一键运行」")
    else:
        col_orig, col_result = st.columns(2)

        with col_orig:
            st.subheader("原图")
            st.image(cv2.cvtColor(active["proc"], cv2.COLOR_BGR2RGB),
                     use_container_width=True)

        with col_result:
            st.subheader(f"抠图结果 ({active['name']})")
            st.image(cv2.cvtColor(active["white_bgr"], cv2.COLOR_BGR2RGB),
                     use_container_width=True)
            if active["name"] == "传统方法":
                st.caption(f"⏱ 耗时 {st.session_state.trad_time:.1f}s")

        st.divider()

        # 中间过程 — 仅传统方法时展示
        if active["name"] == "传统方法" and st.session_state.skin_mask is not None:
            with st.expander("🔍 查看传统方法中间处理过程", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.caption("肤色检测")
                    st.image(st.session_state.skin_mask, clamp=True, use_container_width=True)
                with c2:
                    st.caption("GrabCut 分割")
                    st.image(st.session_state.grabcut_mask, clamp=True, use_container_width=True)
                with c3:
                    st.caption("融合 Mask")
                    st.image(st.session_state.fusion_mask, clamp=True, use_container_width=True)
                with c4:
                    st.caption("精修软 Alpha")
                    st.image(st.session_state.cleaned_mask, clamp=True, use_container_width=True)

        st.divider()

        # 导出区
        st.subheader("💾 导出结果")
        ec1, ec2, ec3 = st.columns(3)

        with ec1:
            tp = active["transparent"]
            if tp is not None:
                st.download_button(
                    "📥 下载透明PNG (RGBA)",
                    data=rgba_to_pil_bytes(tp),
                    file_name=f"{st.session_state.img_name}_{active['name']}_cutout.png",
                    mime="image/png", use_container_width=True
                )
        with ec2:
            st.download_button(
                "📥 下载白底JPG",
                data=bgr_to_pil_bytes(active["white_bgr"]),
                file_name=f"{st.session_state.img_name}_{active['name']}_white.jpg",
                mime="image/jpeg", use_container_width=True
            )
        with ec3:
            cm = active["mask"]
            if cm is not None:
                mask_png = io.BytesIO()
                Image.fromarray(cm).save(mask_png, format="PNG")
                mask_png.seek(0)
                st.download_button(
                    "📥 下载软Alpha Mask",
                    data=mask_png,
                    file_name=f"{st.session_state.img_name}_{active['name']}_mask.png",
                    mime="image/png", use_container_width=True
                )

        # 其他背景色快览 (仅传统方法有预生成)
        if active["name"] == "传统方法":
            st.caption("其他背景色预览")
            solid = st.session_state.step5_solid or {}
            if solid:
                scols = st.columns(len(solid))
                for col, (color, bgr) in zip(scols, solid.items()):
                    with col:
                        st.caption(color)
                        st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_container_width=True)


# ============================================================
#  界面2: AI对比分析
# ============================================================
with t2:
    trad_ready = st.session_state.cleaned_mask is not None
    ai_ready = st.session_state.ai_results is not None

    if not trad_ready or not ai_ready:
        st.warning("请先在左侧运行「🚀 传统抠图」和「🤖 AI抠图」")
    else:
        # 选择要对比的 AI 方法
        ai_methods = list(st.session_state.ai_results.keys())
        selected_ai = st.selectbox("选择 AI 方法进行对比", ai_methods)

        ai_data = st.session_state.ai_results[selected_ai]
        ai_mask = ai_data.get("mask")
        ai_result = ai_data.get("result")
        trad_mask = st.session_state.cleaned_mask

        if ai_mask is not None:
            # 对齐尺寸
            if trad_mask.shape[:2] != ai_mask.shape[:2]:
                ai_mask_aligned = cv2.resize(ai_mask,
                                             (trad_mask.shape[1], trad_mask.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)
            else:
                ai_mask_aligned = ai_mask

            # Mask 对比行
            st.subheader("📊 Mask 对比")
            mc1, mc2, mc3 = st.columns(3)

            with mc1:
                st.caption("传统方法 Mask")
                st.image(trad_mask, clamp=True, use_container_width=True)
            with mc2:
                st.caption(f"{selected_ai} Mask")
                ai_disp = ai_mask_aligned if len(ai_mask_aligned.shape) == 2 \
                    else cv2.cvtColor(ai_mask_aligned, cv2.COLOR_BGR2GRAY)
                st.image(ai_disp, clamp=True, use_container_width=True)
            with mc3:
                st.caption("差异图 (白=一致 绿=仅传统 红=仅AI)")
                diff_viz, diff_legend = mask_diff_viz(trad_mask, ai_mask_aligned)
                st.image(cv2.cvtColor(diff_viz, cv2.COLOR_BGR2RGB),
                         use_container_width=True)

            # 差异统计
            st.caption("差异像素统计")
            dcols = st.columns(len(diff_legend))
            for col, (key, val) in zip(dcols, diff_legend.items()):
                with col:
                    st.metric(key, val)

            # 抠图结果对比
            st.divider()
            st.subheader("🖼️ 抠图结果对比")
            rc1, rc2 = st.columns(2)
            with rc1:
                st.caption("传统方法 (白底)")
                st.image(cv2.cvtColor(st.session_state.white_bgr, cv2.COLOR_BGR2RGB),
                         use_container_width=True)
                st.download_button("📥 传统透明PNG",
                                   data=rgba_to_pil_bytes(st.session_state.transparent_rgba),
                                   file_name=f"{st.session_state.img_name}_传统_透明.png",
                                   mime="image/png", use_container_width=True)
            with rc2:
                st.caption(f"{selected_ai} (白底)")
                if ai_result is not None:
                    ai_w = rgba_to_white(ai_result)
                    st.image(cv2.cvtColor(ai_w, cv2.COLOR_BGR2RGB),
                             use_container_width=True)
                    if ai_result.shape[2] == 4:
                        st.download_button(f"📥 {selected_ai} 透明PNG",
                                           data=rgba_to_pil_bytes(ai_result),
                                           file_name=f"{st.session_state.img_name}_{selected_ai}_透明.png",
                                           mime="image/png", use_container_width=True)
                else:
                    st.caption("无结果")

            # 指标对比
            st.divider()
            st.subheader("📈 定量指标")
            gt = st.session_state.gt_mask

            trad_m = evaluate_one(trad_mask, gt, "传统方法",
                                  runtime=st.session_state.trad_time)
            ai_m = evaluate_one(ai_mask_aligned, gt, selected_ai,
                                runtime=ai_data.get("time", 0))
            all_m = [trad_m, ai_m]

            # 指标表
            rows = []
            for m in all_m:
                rows.append({
                    "方法": m["method"],
                    "IoU": f"{m['iou']:.4f}" if m['iou'] is not None else "—",
                    "Dice": f"{m['dice']:.4f}" if m['dice'] is not None else "—",
                    "Precision": f"{m['precision']:.4f}" if m['precision'] is not None else "—",
                    "Recall": f"{m['recall']:.4f}" if m['recall'] is not None else "—",
                    "边缘平滑": f"{m['edge_smoothness']:.4f}",
                    "毛发保留": f"{m['hair_preservation']:.4f}",
                    "背景残留": f"{m['bg_residue']:.4f}",
                    "耗时(s)": f"{m.get('runtime_s', 0):.2f}",
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

            # 柱状图
            if gt is not None:
                names = [m["method"] for m in all_m]
                ious = [m["iou"] or 0 for m in all_m]
                dices = [m["dice"] or 0 for m in all_m]

                fig, ax = plt.subplots(figsize=(6, 3))
                x = np.arange(len(names))
                w = 0.3
                ax.bar(x - w / 2, ious, w, label="IoU", color="#4C72B0")
                ax.bar(x + w / 2, dices, w, label="Dice", color="#DD8452")
                ax.set_xticks(x)
                ax.set_xticklabels(names)
                ax.set_ylabel("Score")
                ax.set_ylim(0, 1.05)
                ax.legend()
                ax.grid(axis="y", alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)

            # 一键导出报告
            st.divider()
            st.subheader("📋 导出对比报告")
            if st.button("📄 生成并下载报告", use_container_width=True):
                # CSV
                csv_buf = io.StringIO()
                fields = ["method", "iou", "dice", "precision", "recall",
                          "edge_smoothness", "hair_preservation", "bg_residue", "runtime_s"]
                w_csv = csv.DictWriter(csv_buf, fieldnames=fields, extrasaction="ignore")
                w_csv.writeheader()
                for m in all_m:
                    w_csv.writerow(m)

                # Markdown 报告
                md_lines = [
                    f"# 人像抠图对比报告",
                    f"",
                    f"**图片**: {st.session_state.img_name}",
                    f"**传统方法耗时**: {st.session_state.trad_time:.1f}s",
                    f"**AI方法**: {selected_ai}",
                    f"",
                    f"## Mask 差异统计",
                ]
                for k, v in diff_legend.items():
                    md_lines.append(f"- {k}: {v}")
                md_lines.append("")
                md_lines.append("## 定量指标")
                md_lines.append("| 方法 | IoU | Dice | Precision | Recall | 边缘平滑 | 毛发保留 | 背景残留 | 耗时 |")
                md_lines.append("|------|-----|------|-----------|--------|----------|----------|----------|------|")
                for m in all_m:
                    md_lines.append(
                        f"| {m['method']} "
                        f"| {m.get('iou', '-')} "
                        f"| {m.get('dice', '-')} "
                        f"| {m.get('precision', '-')} "
                        f"| {m.get('recall', '-')} "
                        f"| {m.get('edge_smoothness', '-')} "
                        f"| {m.get('hair_preservation', '-')} "
                        f"| {m.get('bg_residue', '-')} "
                        f"| {m.get('runtime_s', '-')} |"
                    )
                md_lines.append("")
                md_lines.append("## 总结")
                md_lines.append("传统方法优势: 可解释、无GPU依赖、流程可控。")
                md_lines.append("AI方法优势: 边缘更精细、泛化能力强。")

                md = "\n".join(md_lines)

                # 打包成 zip-like: 分别下载 CSV + MD
                st.download_button(
                    "📥 下载 CSV 指标",
                    csv_buf.getvalue().encode("utf-8-sig"),
                    file_name=f"{st.session_state.img_name}_comparison.csv",
                    mime="text/csv",
                )
                st.download_button(
                    "📥 下载 Markdown 报告",
                    md.encode("utf-8"),
                    file_name=f"{st.session_state.img_name}_report.md",
                    mime="text/markdown",
                )

                # 差异图下载
                diff_buf = io.BytesIO()
                Image.fromarray(cv2.cvtColor(diff_viz, cv2.COLOR_BGR2RGB)).save(diff_buf, format="PNG")
                diff_buf.seek(0)
                st.download_button(
                    "📥 下载差异可视化图",
                    diff_buf,
                    file_name=f"{st.session_state.img_name}_diff.png",
                    mime="image/png",
                )

                st.success("报告已生成，请点击上方按钮下载")


# ============================================================
#  界面3: 背景与滤镜
# ============================================================
with t3:
    active = get_active_matting()

    if active["mask"] is None:
        st.warning("请先在左侧运行「🚀 一键运行」")
    else:
        proc = active["proc"]
        cm = active["mask"]
        if cm.shape[:2] != proc.shape[:2]:
            cm = cv2.resize(cm, (proc.shape[1], proc.shape[0]))

        st.markdown(f"### 🎨 更换背景 (当前: {active['name']})")

        bg_left, bg_right = st.columns([1, 2])

        with bg_left:
            bg_mode = st.radio("选择方式", ["上传自定义背景", "纯色背景"],
                               label_visibility="collapsed")

            if bg_mode == "上传自定义背景":
                bf = st.file_uploader("上传背景图", type=["jpg", "jpeg", "png", "bmp"],
                                      key="edit_bg")
                if bf is not None:
                    bb = np.frombuffer(bf.read(), np.uint8)
                    bg_img = cv2.imdecode(bb, cv2.IMREAD_COLOR)
                    if bg_img is not None:
                        st.session_state.edit_bg_img = bg_img
                        st.success(f"已加载 ({bg_img.shape[1]}×{bg_img.shape[0]})")
                if st.button("🔄 换背景", use_container_width=True):
                    bg_img = st.session_state.get("edit_bg_img")
                    if bg_img is None:
                        st.error("请先上传背景图")
                    else:
                        st.session_state.edit_bg_result = cutout_custom_bg(
                            proc, cm, bg_img)
            else:
                bg_color = st.selectbox("选择颜色",
                                        ["white", "red", "blue", "green",
                                         "gray", "black", "yellow", "pink"],
                                        format_func=lambda x: {
                                            "white": "白色", "red": "红色",
                                            "blue": "蓝色", "green": "绿色",
                                            "gray": "灰色", "black": "黑色",
                                            "yellow": "黄色", "pink": "粉色",
                                        }.get(x, x))
                if st.button("🔄 换背景", use_container_width=True):
                    st.session_state.edit_bg_result = cutout_solid(proc, cm, bg_color)

        with bg_right:
            result_bg = st.session_state.get("edit_bg_result")
            if result_bg is not None:
                st.image(cv2.cvtColor(result_bg, cv2.COLOR_BGR2RGB),
                         use_container_width=True)
                st.download_button("📥 导出", bgr_to_pil_bytes(result_bg),
                                   file_name=f"{st.session_state.img_name}_newbg.png",
                                   mime="image/png")
            else:
                st.caption("点击「换背景」预览")

        st.divider()
        st.markdown("### 🌈 选择滤镜")

        FILTERS = {
            "原图": None,
            "美白": whiten,
            "降噪(双边)": denoise_bilateral,
            "锐化(USM)": sharpen_unsharp,
            "综合增强": comprehensive_enhance,
            "复古滤镜": vintage,
            "Dream冷调": dream,
            "胶片质感": film,
            "粉色氛围": pink,
        }

        fl, fr = st.columns([1, 2])

        with fl:
            sel_f = st.selectbox("滤镜", list(FILTERS.keys()))
            base = st.session_state.get("edit_bg_result")
            if base is None:
                base = cutout_solid(proc, cm, "white")

            if st.button("✨ 应用滤镜", use_container_width=True):
                if FILTERS[sel_f] is not None:
                    mask_bin = (cm > 30).astype(np.uint8) * 255
                    if sel_f in ["美白", "降噪(双边)", "锐化(USM)", "综合增强"]:
                        st.session_state.edit_filter_result = FILTERS[sel_f](base, mask_bin)
                    else:
                        st.session_state.edit_filter_result = FILTERS[sel_f](base)
                else:
                    st.session_state.edit_filter_result = base

        with fr:
            fr_res = st.session_state.get("edit_filter_result")
            if fr_res is not None:
                st.image(cv2.cvtColor(fr_res, cv2.COLOR_BGR2RGB),
                         use_container_width=True)
                st.download_button("📥 导出", bgr_to_pil_bytes(fr_res),
                                   file_name=f"{st.session_state.img_name}_filtered.png",
                                   mime="image/png")
            else:
                st.caption("点击「应用滤镜」预览")


# ============================================================
#  界面4: 图像调节 (实时预览 · 工作流: 抠图→换背景→滤镜→调节)
# ============================================================
with t4:
    active = get_active_matting()

    if active["mask"] is None:
        st.warning("请先在左侧运行「🚀 一键运行」")
    else:
        proc = active["proc"]
        cm = active["mask"]
        if cm.shape[:2] != proc.shape[:2]:
            cm = cv2.resize(cm, (proc.shape[1], proc.shape[0]))

        # 工作流串联: 滤镜结果 → 换背景结果 → active 白底抠图
        def get_pipeline_src():
            """沿工作流链找最新的源图"""
            f = st.session_state.get("edit_filter_result")
            if f is not None:
                return f
            b = st.session_state.get("edit_bg_result")
            if b is not None:
                return b
            w = active["white_bgr"]
            if w is not None:
                return w
            return cutout_solid(proc, cm, "white")

        src = get_pipeline_src()

        # 如果界面3产出了新结果，自动更新 src
        if "adjust_src" not in st.session_state:
            st.session_state.adjust_src = src
        else:
            new_src = get_pipeline_src()
            if new_src is not None and new_src is not st.session_state.adjust_src:
                st.session_state.adjust_src = new_src
                src = new_src

        st.markdown("### 🎚️ 图像参数调节")
        st.caption("工作流: 抠图 → 换背景 → 滤镜 → **调节** (拖动滑块实时预览)")

        adj_left, adj_right = st.columns([1, 2])

        with adj_left:
            exposure = st.slider("曝光", -100, 100, 0, 1,
                                 help="正值提亮，负值压暗", key="adj_exp")
            contrast = st.slider("对比度", 0.5, 2.0, 1.0, 0.05,
                                 help=">1 增强对比，<1 降低对比", key="adj_con")
            saturation = st.slider("饱和度", 0.0, 2.0, 1.0, 0.05,
                                   help=">1 增饱和，<1 降饱和，0=黑白", key="adj_sat")
            sharpen_amt = st.slider("锐化强度", 0.0, 3.0, 0.0, 0.1,
                                    help="0=不锐化，越大越锐", key="adj_shp")
            apply_mode = st.radio("应用范围", ["仅人物", "全图"],
                                  label_visibility="collapsed", key="adj_mode")

            if st.button("↩ 重置参数", use_container_width=True):
                st.session_state.adj_exp = 0
                st.session_state.adj_con = 1.0
                st.session_state.adj_sat = 1.0
                st.session_state.adj_shp = 0.0
                st.session_state.adjust_src = None
                st.rerun()

        # --- 实时计算调整结果 ---
        result = src.copy().astype(np.float32)

        # 曝光
        result += exposure
        np.clip(result, 0, 255, out=result)

        # 对比度
        mean = result.mean(axis=(0, 1), keepdims=True)
        result = (result - mean) * contrast + mean
        np.clip(result, 0, 255, out=result)

        # 饱和度
        hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= saturation
        np.clip(hsv[:, :, 1], 0, 255, out=hsv[:, :, 1])
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

        # 锐化
        if sharpen_amt > 0.01:
            blurred = cv2.GaussianBlur(result, (0, 0), 1.5)
            result = cv2.addWeighted(result, 1.0 + sharpen_amt,
                                     blurred, -sharpen_amt, 0)
            np.clip(result, 0, 255, out=result)

        result = result.astype(np.uint8)

        # 仅人物
        if apply_mode == "仅人物":
            mask_bin = (cm > 30).astype(np.float32)
            mask_3 = np.stack([mask_bin] * 3, axis=-1)
            result = (result * mask_3 + src * (1 - mask_3)).astype(np.uint8)

        with adj_right:
            st.caption("实时预览")
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                     use_container_width=True)
            st.download_button("📥 导出结果",
                               data=bgr_to_pil_bytes(result),
                               file_name=f"{st.session_state.img_name}_adjusted.png",
                               mime="image/png")


# ============================================================
st.divider()
st.caption("DIP 数字图像处理课程大作业 · 方案A · 传统方法主线 · AI对照组")
