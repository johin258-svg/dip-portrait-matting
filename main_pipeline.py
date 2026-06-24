# -*- coding: utf-8 -*-
"""
DIP 人像分割 — 主流水线入口

一键运行完整管线: Step 1 → Step 2 → Step 3 → Step 4 → Step 5

用法:
    python main_pipeline.py              # 运行所有步骤
    python main_pipeline.py --step 1     # 只运行 Step 1
    python main_pipeline.py --step 2     # 只运行 Step 2 (需要 Step 1 输出)
    python main_pipeline.py --from 3     # 从 Step 3 开始 (需要 Step 1-2 输出)
    python main_pipeline.py --help       # 查看帮助

数据准备:
    data/original_images/imageX.jpg      # 原始人像照片 (必须)
    background.jpg                       # 自定义背景图 (可选)

管线流程:
    Step 1 — 预处理 + 肤色检测 + 光照校正
        → outputs/step1/xbw_masks/       (肤色掩膜)
        → outputs/step1/light_fixed/     (光照校正图)

    Step 2 — GrabCut 人像分割 + 掩膜融合
        → outputs/step2/masks/yfy_masks/ (人物掩膜)
        → outputs/step2/fusion/weighted/ (融合掩膜 ★ 关键产物)

    Step 3 — 形态学优化 + Frangi+Gabor 头发精修 + 引导滤波
        → outputs/step3/                 (精修软alpha掩膜 ★ 关键产物)

    Step 4 — 人像增强 (美白/降噪/锐化/滤镜)
        → outputs/step4/                 (增强后人像)

    Step 5 — 抠图输出 (透明PNG/纯色背景/自定义背景)
        → outputs/step5/                 (最终交付产物)
"""

import os, sys, argparse, time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import DATA_DIR, OUTPUT_DIR


def run_all():
    """运行完整管线"""
    print("\n" + "=" * 65)
    print("  DIP 人像分割 — 完整管线")
    print("  基于 GrabCut + Frangi/Gabor 发丝检测 + 引导滤波")
    print("=" * 65)

    t0 = time.time()

    # Step 1
    print("\n" + "─" * 65)
    t1 = time.time()
    from step1_preprocess.run_step1 import run_step1
    run_step1(DATA_DIR, OUTPUT_DIR)
    print(f"  ⏱  Step 1 耗时: {time.time() - t1:.1f}s")

    # Step 2
    print("\n" + "─" * 65)
    t2 = time.time()
    step1_out = os.path.join(OUTPUT_DIR, "step1")
    step2_out = os.path.join(OUTPUT_DIR, "step2")
    from step2_segmentation.run_step2 import run_step2
    run_step2(DATA_DIR, step1_out, OUTPUT_DIR)
    print(f"  ⏱  Step 2 耗时: {time.time() - t2:.1f}s")

    # Step 3
    print("\n" + "─" * 65)
    t3 = time.time()
    from step3_morphology.run_step3 import run_step3
    run_step3(DATA_DIR, step2_out, OUTPUT_DIR)
    print(f"  ⏱  Step 3 耗时: {time.time() - t3:.1f}s")

    # Step 4
    print("\n" + "─" * 65)
    t4 = time.time()
    step3_out = os.path.join(OUTPUT_DIR, "step3")
    from step4_enhancement.run_step4 import run_step4
    run_step4(DATA_DIR, step3_out, OUTPUT_DIR)
    print(f"  ⏱  Step 4 耗时: {time.time() - t4:.1f}s")

    # Step 5
    print("\n" + "─" * 65)
    t5 = time.time()
    from step5_matting.run_step5 import run_step5
    run_step5(DATA_DIR, step3_out, OUTPUT_DIR)
    print(f"  ⏱  Step 5 耗时: {time.time() - t5:.1f}s")

    total = time.time() - t0
    print("\n" + "=" * 65)
    print(f"  ✓ 全部完成! 总耗时: {total:.1f}s ({total/60:.1f}min)")
    print(f"  最终输出: {os.path.join(OUTPUT_DIR, 'step5')}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="DIP 人像分割管线 — 一键运行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main_pipeline.py              # 运行完整管线
  python main_pipeline.py --step 1     # 仅 Step 1: 预处理+肤色检测
  python main_pipeline.py --step 3     # 仅 Step 3: 形态学优化
  python main_pipeline.py --from 2     # 从 Step 2 开始到结束
        """
    )
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4, 5],
                        help="只运行指定步骤")
    parser.add_argument("--from", dest="from_step", type=int, choices=[1, 2, 3, 4, 5],
                        help="从指定步骤开始运行到 Step 5")
    args = parser.parse_args()

    if args.step:
        step = args.step
        step1_out = os.path.join(OUTPUT_DIR, "step1")
        step2_out = os.path.join(OUTPUT_DIR, "step2")
        step3_out = os.path.join(OUTPUT_DIR, "step3")

        if step == 1:
            from step1_preprocess.run_step1 import run_step1
            run_step1(DATA_DIR, OUTPUT_DIR)
        elif step == 2:
            from step2_segmentation.run_step2 import run_step2
            run_step2(DATA_DIR, step1_out, OUTPUT_DIR)
        elif step == 3:
            from step3_morphology.run_step3 import run_step3
            run_step3(DATA_DIR, step2_out, OUTPUT_DIR)
        elif step == 4:
            from step4_enhancement.run_step4 import run_step4
            run_step4(DATA_DIR, step3_out, OUTPUT_DIR)
        elif step == 5:
            from step5_matting.run_step5 import run_step5
            run_step5(DATA_DIR, step3_out, OUTPUT_DIR)
    elif args.from_step:
        start = args.from_step
        step1_out = os.path.join(OUTPUT_DIR, "step1")
        step2_out = os.path.join(OUTPUT_DIR, "step2")
        step3_out = os.path.join(OUTPUT_DIR, "step3")

        if start <= 1:
            from step1_preprocess.run_step1 import run_step1
            run_step1(DATA_DIR, OUTPUT_DIR)
        if start <= 2:
            from step2_segmentation.run_step2 import run_step2
            run_step2(DATA_DIR, step1_out, OUTPUT_DIR)
        if start <= 3:
            from step3_morphology.run_step3 import run_step3
            run_step3(DATA_DIR, step2_out, OUTPUT_DIR)
        if start <= 4:
            from step4_enhancement.run_step4 import run_step4
            run_step4(DATA_DIR, step3_out, OUTPUT_DIR)
        if start <= 5:
            from step5_matting.run_step5 import run_step5
            run_step5(DATA_DIR, step3_out, OUTPUT_DIR)
    else:
        run_all()


if __name__ == "__main__":
    main()
