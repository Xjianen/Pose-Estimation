# segmentation/ — 高质量前景 mask 重生成

针对 PTAD_plus 查询图 `images/*.png` 的二值前景 mask。原始 `mask/*.png` 是**阈值分割**产物，质量不稳定；本模块用**免训练的开源模型 + 确定性兜底**重新生成，输出到 `mask_clean/`，文件名与原图一一对应。

---

## 1. 旧 mask 到底差在哪

`mask/` 实际上是把相机空间法向缓冲 `test/` 按「背景常量色」阈值化得到的：

```
mask = (test_pixel != purple_background)
```

法向图里**物体、地板、地板阴影都不是背景色**，所以三者一起被判成前景。实测两个 shape 共 45 张查询图：

| 缺陷 | 表现 |
|---|---|
| 地板阴影并入轮廓 | 平均每张多出 ~3.3k px（最多 10k px），底部被填成实心凸包 |
| 拱洞 / 椅腿间隙被填平 | 棕色椅、蓝色椅底部的凹形开口消失 |
| 椒盐噪声 | 平均每张 24.3 个孤立小连通块（阴影区的阈值抖动） |
| 浅灰物体边界 | 阈值把浅灰地板与浅灰椅面混在一起 |

**注意两个容易踩的坑：**

1. `render/` 和 `reference/` 不是查询视角真值。它们由另一套内参渲染（参考 FOV 39.66° vs 查询 50°），边界与查询 RGB 的强梯度对齐率只有 **0.86**，而旧 mask 有 **0.98**。拿 `render/` 当 GT 会得出完全错误的结论。
2. 法向图里地板颜色与物体高光色有重叠（都在 `[180,250,250]` 附近），靠颜色阈值分离地板不可靠。

---

## 2. 试过但不采用的方案（本地 45 张实测）

| 方案 | 结果 |
|---|---|
| 固定亮度阈值 190 / Otsu / GrabCut | 深/浅色物体都能保住，但**会填平拱洞**，且阈值 235（≈现状）最差 |
| Canny 边缘闭合 + 漫填 | 能抓到拱洞轮廓，但深色椅子内部纹理造成泄漏，8 张里 2 张崩掉 |
| RMBG-1.4 (ONNX) 直接用 | 深色椅极好（边缘平滑、拱洞正确），但**浅灰椅整块座面被当背景**：45 张里 9 张 IoU<0.6，最差 0.10 |
| RMBG-1.4 + 换背景/反色/CLAHE 集成 | 换背景、反色更差；CLAHE 只把失败数从 9 降到 5，仍不稳 |
| 直接用 `render/` 剪影 | FOV 不同视角，不是真值（见上） |

结论：**纯 CV 保得住颜色但保不住形状；显著性模型保得住形状但保不住浅色物体**。所以用 **prompt 驱动的 SAM**——它按提示点分割，与物体颜色无关。

---

## 3. 最终方案

```
RGB ──> coarse_foreground()      白底粗前景（故意过 inclusive，保证浅色物体不丢）
          │
          ├─> sample_prompts()  正样本：eroded 内部 3×3 网格最深处
          │                    负样本：四角/底边 + 质心以下的平滑灰色阴影区
          │
          ├─> SAM (SlimSAM ONNX, 14 MB, 零样本) ─> 3 个候选 mask
          │        选与 coarse 一致性最高 + 面积合理(0.55~1.45×) 的那个
          │        （不用模型自己的 IoU head：它有时把"背景"打成高分）
          │
          └─> cv_refine() 兜底   模型缺失/结果不合理时，按"低色度 + 低纹理 +
                                 质心以下 + 面积 <50%" 剔除地板阴影块

        ──> largest_components + fill_small_holes
            （只补小洞，绝不填大开口 —— 这正是旧 mask 的 bug）
```

- **模型**：`Xenova/slimsam-77-uniform` 量化 ONNX（encoder 8.9 MB + decoder 4.9 MB），无需训练、无需 token。
- **依赖**：`numpy`, `opencv-python`, `onnxruntime`（项目已有）。
- **速度**：CPU 约 2.2 s/张；`--mode cv` 约 0.05 s/张。
- 图/面板里所有文字均为 ASCII，服务器无中文字体也能正常渲染。

---

## 4. 实测效果（两个 shape，45 张查询图）

| 指标 | 旧 mask | 新 mask | 变化 |
|---|---|---|---|
| `edge_align`（边界落在 RGB 强梯度上，越高越好） | 0.9694 | **0.9968** | +0.0274 |
| `shadow_residue`（下半部低色度平滑像素占比，越低越好） | 0.1392 | **0.1084** | −0.0309 |
| `speckles`（孤立小连通块数，越低越好） | 24.27 | **0.00** | −24.27 |
| 平均剔除像素 / 新增像素 | — | 3758 / 60 | 主要删的是地板阴影 |
| IoU(old, new) | — | 0.9298 | 约 7% 像素被修正 |

45/45 张全部由 SAM 精修成功，**0 次兜底、0 失败**，总耗时 98.5 s（单进程 CPU）。
可视化见 `--compare` 生成的 `compare.png`：棕椅拱洞恢复、橙椅阴影斑点消失、浅灰椅保持完整。

> 评估用的是无监督质量代理，不是像素级 GT——因为查询视角没有干净 GT（原因见第 1 节）。这些指标直接对应位姿估计最在意的性质：轮廓边界正确、无阴影污染、无散点噪声。

---

## 5. 用法

```bash
cd ~/project/Third_Work/PTAD_plus

# ① 一次性下载权重（14 MB，自动走 hf-mirror，失败回退 huggingface）
python segmentation/download_models.py

# ② 先跑两个 shape 看效果（生成 compare.png + 报告）
python segmentation/run_segmentation.py \
    --root /data/xje/datasets/brokenchairs180k/shapes \
    --limit 2 --compare --report seg_report.json

# ③ 全量（8000+ shape），8 个 shape 级并行进程，可断点续跑
python segmentation/run_segmentation.py \
    --root /data/xje/datasets/brokenchairs180k/shapes \
    --workers 8 --report /data/xje/datasets/brokenchairs180k/runs/seg_report.json

# ④ 对比评估（终端表格 + CSV + 对照图）
python segmentation/evaluate.py \
    --root /data/xje/datasets/brokenchairs180k/shapes \
    --csv seg_eval.csv --panel seg_eval.png
```

**输出位置**：默认写 `<shape>/mask_clean/<原文件名>.png`（0/255 二值 PNG），不覆盖原 `mask/`。
已有文件默认跳过；`--overwrite` 强制重算。也可用 `--mask-root DIR` 写到独立目录 `DIR/<shape>/`。

**常用参数**：

| 参数 | 说明 |
|---|---|
| `--mode auto\|sam\|cv` | `auto`=有模型用 SAM、否则 CV（默认）；`cv` 纯确定性、最快 |
| `--workers N` | shape 级多进程（每个进程独立 ONNX session）；GPU 模式建议 1 |
| `--model-dir` | 权重目录，默认 `segmentation/models/` |
| `--images-subdir` / `--out-subdir` | 默认 `images` / `mask_clean` |
| `--shape shape_1095` / `--limit N` | 只跑指定 shape / 前 N 个 |
| `--compare` | 每个 shape 额外输出 `compare.png`（RGB \| old \| clean） |
| `--report PATH` | JSON：每张图的 method + 质量指标 + 总计 |

`download_models.py` 支持 `--mirror URL` 自定义源（例如内网镜像）。

---

## 6. 接入位姿估计

`train.py` 新增 `--mask-subdir`（默认 `mask`，**不改变原有行为**）：

```bash
python train.py --root /data/xje/datasets/brokenchairs180k/shapes/shape_1095 \
    --codebook-mode refs --mask-subdir mask_clean
```

`ShadObject` / `ShadDataset` 会把查询 mask 路径指向 `mask_clean/`，`remove_floor_shadow()`
仍保留（对新 mask 是空操作，不影响）。批处理脚本 `run_batch.sh` 要换 mask 时加一行
`--mask-subdir mask_clean` 即可。

---

## 7. 文件

| 文件 | 作用 |
|---|---|
| `core.py` | 核心库：粗前景、阴影模型、prompt 生成、SAM 封装、CV 兜底、质量指标 |
| `download_models.py` | 下载 SlimSAM ONNX 权重（带镜像回退、大小校验） |
| `run_segmentation.py` | 批量 CLI：多进程、断点续跑、进度、JSON 报告、对照图 |
| `evaluate.py` | 新旧 mask 对比：终端表格 + CSV + 对照图 |
| `models/` | 下载的权重（已 gitignore，不入库） |
