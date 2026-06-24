# DIP 人像抠图系统 (Digital Image Processing — Portrait Matting System)

> 数字图像处理课程大作业 · 方案A · 传统方法主线 + AI对照组

## 🌐 Web 应用

```bash
pip install -r requirements.txt
streamlit run app.py
```

**四大功能界面:**

| 界面 | 功能 |
|------|------|
| 📤 抠图导出 | 原图 vs 结果对比，下载透明PNG/白底JPG/Mask |
| 🔬 AI对比分析 | 传统方法 vs AI mask 差异图，一键导出对比报告 |
| 🖼️ 背景与滤镜 | 上传自定义背景/纯色 + 9种艺术滤镜 |
| 🎚️ 图像调节 | 实时曝光/对比度/饱和度/锐化，仅人物/全图切换 |

**工作流:** 一键运行传统+AI抠图 → 对比分析 → 换背景+滤镜 → 实时调节

---

## 命令行管线

也可以单独运行传统 DIP 管线（不启动 Web 界面）。

---

## 管线概览

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: 预处理 + 肤色检测 (Part A)                              │
│  HSV/YCrCb肤色掩膜 + Homomorphic/Illumination/Retinex光照校正    │
│  → outputs/step1/xbw_masks/ + light_fixed/                      │
├─────────────────────────────────────────────────────────────────┤
│  Step 2: GrabCut 人像分割 + 掩膜融合 (Part B)                     │
│  两轮GrabCut(5+3 iter) + 凸包约束 + 头发HSV精修 + yfy/xbw融合    │
│  → outputs/step2/masks/yfy_masks/ + fusion/weighted/ ★          │
├─────────────────────────────────────────────────────────────────┤
│  Step 3: 形态学优化 + 头发精修 (Part C-1)                        │
│  保守清理 → Frangi发丝检测 → Gabor纹理验证 → 各向异性扩展         │
│  → 引导滤波平滑 → 硬阈值30                                       │
│  → outputs/step3/imageX_cleaned.png ★ (软alpha, 最终掩膜)       │
├─────────────────────────────────────────────────────────────────┤
│  Step 4: 人像增强 (Part C-2)                                     │
│  美白(通道调整) + 双边降噪 + Unsharp锐化 + 4种全图滤镜            │
│  → outputs/step4/                                               │
├─────────────────────────────────────────────────────────────────┤
│  Step 5: 抠图输出 (Part C-3)                                     │
│  透明PNG(RGBA) + 纯色背景×4 + 自定义背景替换                      │
│  → outputs/step5/ ★ 最终交付                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 项目结构

```
pipeline/
├── main_pipeline.py            # 主入口 — 一键运行完整管线
├── config.py                   # 全局配置参数
├── README.md                   # 本文档
├── background.jpg              # 自定义背景图 (可选)
│
├── data/
│   └── original_images/        # 原始人像照片 (image1~13.jpg/png)
│
├── outputs/                    # 所有输出 (自动生成)
│   ├── step1/
│   │   ├── xbw_masks/          # 肤色掩膜 (imageX_xbw_mask.jpg)
│   │   └── light_fixed/        # 光照校正图 (light1/2/3)
│   ├── step2/
│   │   ├── masks/yfy_masks/    # GrabCut 人物掩膜
│   │   └── fusion/
│   │       ├── weighted/       # ★ 加权融合掩膜 (后续步骤输入)
│   │       ├── OR/             # 并集 (参考)
│   │       └── AND/            # 交集 (参考)
│   ├── step3/                  # ★ 精修软alpha掩膜
│   ├── step4/                  # 增强后人像
│   └── step5/                  # ★ 最终抠图输出
│
├── utils/                      # 共享工具模块
│   ├── io_utils.py             # 中文路径读写、掩膜加载
│   └── vis_utils.py            # 可视化工具
│
├── step1_preprocess/           # Part A: 预处理
│   ├── preprocess.py           # 缩放/Gamma/CLAHE/滤波
│   ├── lighting.py             # Homomorphic/Illumination/Retinex
│   ├── skin_detection.py       # HSV/YCrCb 肤色检测
│   └── run_step1.py            # Step 1 入口
│
├── step2_segmentation/         # Part B: 分割
│   ├── grabcut_segment.py      # GrabCut + 后处理 + 头发精修
│   ├── fusion.py               # yfy/xbw 掩膜融合
│   └── run_step2.py            # Step 2 入口
│
├── step3_morphology/           # Part C-1: 形态学
│   ├── guided_filter.py        # 引导滤波 (He et al. 2013)
│   ├── hair_refine.py          # Frangi+Gabor+各向异性头发精修
│   ├── morphology.py           # 保守清理 + 边缘平滑
│   └── run_step3.py            # Step 3 入口
│
├── step4_enhancement/          # Part C-2: 增强
│   ├── enhancement.py          # 美白/降噪/锐化/4种滤镜
│   └── run_step4.py            # Step 4 入口
│
└── step5_matting/              # Part C-3: 抠图
    ├── matting.py              # Alpha混合 + 背景替换
    └── run_step5.py            # Step 5 入口
```

---

## 快速开始

### 1. 环境要求

```bash
pip install opencv-python numpy matplotlib
```

### 2. 准备数据

```bash
# 将原始人像照片放入 data/original_images/
# 命名格式: image2.jpg, image3.jpg, ... image13.jpg
# 管线自动扫描目录，缺少原图的编号会被跳过
mkdir -p data/original_images
```

> **注意**: 管线只处理 `data/original_images/` 中存在的图像。如果某张图没有原始照片（如 image1），该图会被自动跳过，不会报错。

### 3. 运行

```bash
# 一键运行完整管线
python main_pipeline.py

# 只运行某一步
python main_pipeline.py --step 1    # 仅预处理+肤色检测
python main_pipeline.py --step 3    # 仅形态学优化

# 从某步开始运行到结束
python main_pipeline.py --from 3    # 从 Step 3 开始

# 也可以单独运行每步
python step1_preprocess/run_step1.py
python step2_segmentation/run_step2.py
python step3_morphology/run_step3.py
python step4_enhancement/run_step4.py
python step5_matting/run_step5.py
```

---

## 核心技术原理

### Step 1: 肤色检测

| 方法 | 颜色空间 | 阈值范围 |
|------|---------|---------|
| HSV | H∈[0,20]∪[170,180], S∈[30,255], V∈[60,255] | 覆盖黄/白/黑肤色 |
| YCrCb | Cr∈[133,173], Cb∈[77,127] | 亚洲肤色更精确 |
| 融合 | OR (任一空间判定为肤色) | 最大化覆盖 |

### Step 2: GrabCut 图割

- **GMM 颜色建模**: BGR 3通道高斯混合模型 → 前景/背景概率
- **Graph Cut**: min-cut 全局优化分割边界
- **种子初始化**: xbw 皮肤 = FGD, 边界条带 = BGD, 身体/头发椭圆 = PR_FGD
- **两轮迭代**: 5轮掩盖初始化 + 3轮精修

### Step 3: Frangi + Gabor 头发精修

- **Frangi Vesselness** (Frangi 1998): Hessian 特征值检测管状结构
  - 发丝: λ1≈0 (沿发丝方向), λ2>0 (横跨发丝)
  - V = exp(-Rb²/2β²) × (1-exp(-S²/2c²)), 仅当 λ2>0

- **Gabor 纹理验证**: 多方向 (每30°) 滤波, (max-mean)/(max+eps) 度量方向性

- **软颜色似然**: darkness² + desaturation (深色) + Lab中性灰 (浅色/灰白)

- **各向异性扩展**: 强头发→12px, 弱头发→3px, 非头发→不扩展

- **引导滤波** (He et al. 2013): 边缘保持平滑, O(N) 复杂度

### Step 4: 人像增强

- **美白**: B×1.0 G×0.88 R×0.85 + β=22 (通道分离调整)
- **降噪**: 双边滤波 (保边缘, d=7, σ=60)
- **锐化**: Unsharp Mask (sharp = img×2.2 + blur×(-1.2))
- **全图滤镜**: 复古(暖琥珀) / Dream(冷调) / 胶片(颗粒) / 粉色(柔光)

### Step 5: Alpha 抠图

- result = fg × α + bg × (1-α)
- α 来自 Step 3 的软 alpha 掩膜 (引导滤波已生成自然过渡)
- 支持透明 PNG(RGBA) + 纯色背景 + 自定义背景

---

## 关键产物

| 步骤 | 关键输出 | 格式 | 用途 |
|------|---------|------|------|
| Step 1 | `xbw_masks/imageX_xbw_mask.jpg` | 二值 0/255 | GrabCut 种子 |
| Step 2 | `fusion/weighted/imageX_0.5yfy_0.5xbw_fusion_mask.jpg` | 二值 0/255 | 融合掩膜 |
| **Step 3** | **`step3/imageX_cleaned.png`** | **软alpha 0-255** | **★ 最终精修掩膜** |
| Step 4 | `step4/imageX_enhanced.jpg` | BGR | 增强后人像 |
| **Step 5** | **`step5/imageX_transparent.png`** | **RGBA** | **★ 最终抠图** |
| Step 5 | `step5/imageX_solid_white.jpg` | BGR | 白色背景合成 |
| Step 5 | `step5/imageX_custom_bg.jpg` | BGR | 自定义背景合成 |

---

## 已知局限性

1. **白边**: 引导滤波 r=5 仍有 ~5px 过渡带, 极端背光场景可见白边
2. **浅色衣服 + 浅色背景**: GrabCut GMM 难以区分, 部分衣服边缘可能缺失
3. **极细发丝**: confidence<0.03 的发丝被舍弃, 避免噪声
4. **依赖 xbw 面部检测**: 面部检测失败时回退到中心椭圆, 效果下降

---

## 参数调优

所有可调参数集中在 `config.py`, 主要调优入口:

| 参数 | 默认值 | 效果 |
|------|-------|------|
| `GF_RADIUS_EDGE` | 5 | 边缘过渡带宽度 (越小越锐利, 但可能不平滑) |
| `MASK_HARD_THRESHOLD` | 30 | 背景消除强度 (越大越干净, 但可能丢失边缘) |
| `HAIR_EXPAND_MAX` | 12 | 头发最大扩展距离 (越大捕获越多发丝, 但可能产生白边) |
| `FRANGI_BETA` | 0.5 | 发丝选择性 (越小越严格, 减少误检) |
| `HAIR_CONFIDENCE_MIN` | 0.03 | 头发信号最低阈值 (越小越敏感) |
| `FUSION_W_YFY` | 0.5 | yfy 掩膜权重 (越大 GrabCut 贡献越多) |
