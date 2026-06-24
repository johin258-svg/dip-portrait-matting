# -*- coding: utf-8 -*-
"""
DIP 人像分割管线 — 全局配置
"""

import os

# ==================== 路径 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 输入数据
DATA_DIR = os.path.join(BASE_DIR, "data")
ORIGINAL_IMAGES = os.path.join(DATA_DIR, "original_images")

# 输出根目录 (每步输出到 outputs/stepN/)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# 自定义背景图
CUSTOM_BG_PATH = os.path.join(BASE_DIR, "background.jpg")

# ==================== Step 1: 预处理参数 ====================
STEP1_MAX_SIZE = 1024       # 图像最大边长
SKIN_MODE = "or"            # 肤色融合: "or" 或 "and"
SKIN_REFINE = True          # 是否形态学精炼
SKIN_REFINE_KSIZE = 5       # 精炼核大小

# ==================== Step 2: GrabCut 参数 ====================
GC_ITER1 = 5                # 第1轮迭代次数
GC_ITER2 = 3                # 第2轮精修迭代次数
XBW_COVERAGE_THRESHOLD = 0.80  # xbw 覆盖率 > 此值则回退到矩形初始化

# 后处理参数
PP_CLOSE_KSIZE = (7, 7)     # 闭运算核
PP_DILATE_KSIZE = (9, 9)    # 凸包扩展核
PP_HOLE_MAX_RATIO = 0.10    # 孔洞填充最大面积比
PP_COVERAGE_MAX = 0.90      # 覆盖率安全上限

# 头发精修 (PartB 阶段)
HAIR_CANNY_LOW = 10         # Canny 低阈值
HAIR_CANNY_HIGH = 35        # Canny 高阈值
HAIR_MIN_PIXELS = 100       # 最少头发像素数
HAIR_COMPONENT_MIN = 50     # 连通域最少像素

# 融合权重
FUSION_W_YFY = 0.5
FUSION_W_XBW = 0.5
FUSION_THRESHOLD = 100      # 加权融合二值化阈值

# ==================== Step 3: 形态学参数 ====================
# 保守清理
MORPH_OPEN_KSIZE = (3, 3)   # 开运算核
MORPH_MIN_AREA = 500        # 最小连通域面积
MORPH_TOP_N = 3             # 保留前N大连通域
MORPH_HOLE_MAX_RATIO = 0.05 # 孔洞填充 最大面积比

# 头发精修 (Frangi + Gabor)
FRANGI_SIGMAS = [0.5, 1.0, 1.5, 2.0]  # 多尺度
FRANGI_BETA = 0.5           # 线状选择性
GABOR_ORIENTATIONS = 6      # Gabor 方向数
GABOR_SIGMA = 2.5           # 高斯包络
GABOR_LAMBDA = 3.0          # 正弦波长
GABOR_GAMMA = 0.5           # 长宽比
HAIR_CONFIDENCE_MIN = 0.03  # 最低头发置信度
HAIR_EXPAND_MIN = 3         # 最小扩展距离 (px)
HAIR_EXPAND_MAX = 12        # 最大扩展距离 (px)
HAIR_COMPONENT_MIN_AREA = 12  # 连通域最小面积

# 引导滤波
GF_RADIUS_HAIR = 6          # 头发区域过渡带半径
GF_RADIUS_EDGE = 5          # 边缘过渡带半径
GF_EPS = 1e-5               # 头发区域正则化
GF_EPS_EDGE = 1e-4          # 边缘正则化

# 硬阈值
MASK_HARD_THRESHOLD = 30    # alpha < 此值 → 0

# ==================== Step 4: 增强参数 ====================
WHITEN_B = 1.00             # 美白: B通道系数
WHITEN_G = 0.88             # 美白: G通道系数 (减黄)
WHITEN_R = 0.85             # 美白: R通道系数 (减暖红)
WHITEN_BETA = 22            # 美白: 整体提亮
BILATERAL_D = 7             # 双边滤波直径
BILATERAL_SIGMA = 60        # 双边滤波 sigma
UNSHARP_AMOUNT = 2.2        # Unsharp Mask 强度
VINTAGE_SATURATION = 0.65   # 复古滤镜: 饱和度系数

# ==================== Step 5: 抠图参数 ====================
MATTING_BLUR = 1            # 羽化半径 (1=不额外羽化, morphology已输出软alpha)
