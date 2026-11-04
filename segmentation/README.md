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

> 本节是无监督质量代理，不是像素级 GT。后来确认数据集的 `annotations/**/seg_new_*.png`
> 就是查询视角的官方干净剪影（见第 8 节），**像素级 mIoU 请用 `eval_miou.py`**。
> 这些代理指标依然有价值：它们直接对应位姿估计最在意的性质——轮廓边界正确、
> 无阴影污染、无散点噪声，且不依赖标注。

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
| `--dilate N` | 输出 mask 的边界补偿（`+1`=膨胀 1px），默认 `0` 不改动；仅在需要对齐硬边 GT 的 mIoU 时开启，见第 8 节 |
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
| `evaluate.py` | 新旧 mask 无监督对比：终端表格 + CSV + 对照图 |
| `eval_miou.py` | 官方 `seg_new_*` 真值的像素级 mIoU 评估（见第 8 节） |
| `models/` | 下载的权重（已 gitignore，不入库） |

---

## 8. 官方标注的像素级 mIoU 评估

### 8.1 先搞清楚标注里到底是什么

`annotations/<bucket>/` 下每个 anomaly query 有 5 个文件，用途完全不同：

| 文件 | 内容 | 能否当前景 GT |
|---|---|---|
| `seg_new_<key>.png` | 部件语义标签，`0`=背景、`38`=异常部件、`42`=正常部件 | **能，`>0` 就是异常后查询视角的完整剪影** |
| `seg_old_<key>.png` | 异常**前**的部件标签 | 不能（形状已变，与 `seg_new` IoU 仅 0.886） |
| `mask_new_<key>.png` | 异常部件的小块区域（≈7k px） | 不能，只是局部缺陷 mask |
| `mask_old_<key>.png` | 异常前该部件的位置 | 不能 |
| `info_<key>.json` | `2d_bbox` / `cam_params` / 异常配置元数据 | — |

`seg_new` 的 `38` 类面积与 `mask_new` 完全一致（实测 7126 vs 7127 px），
且 100% 落在 `seg_new` 非零区内，印证了上面的语义。

**命名映射**：查询图 `render_<key>.png` ↔ 标注 `seg_new_<key>.png`，
`key = <shape>_<examplar>_<radius>_<azim>_<elev>_<idx>_<type>`。
同一个 shape 的不同视角会散落在**不同的 bucket 目录**里，所以 `eval_miou.py`
按 key 建全局索引，不假设两者在同一目录。

**覆盖率**：全库 177,779 张查询图里只有 99,785 个 `seg_new` 标注（**56.1%**）——
因为 `*_normal` 查询没有标注。评估只在交集上进行，脚本会把覆盖率打在表头。

**视角一致性验证**：GT 与查询图同内参。150 张上 bbox 宽高比 `old/GT = 0.987/0.991`、
质心差仅 `(0.2, 4.9)` px、面积比 0.997，**无需配准**就能直接算 IoU。

### 8.2 为什么不能只报一个 raw mIoU

GT 是渲染引擎输出的**硬边标签（无抗锯齿）**，查询图是**抗锯齿 RGB**，两者对
"边界在哪"的定义天然差 1–3 px：

| 观察（120 / 40 张抽样） | 数值 |
|---|---|
| 分歧像素落在 GT 边界 ±2.5 px 内的比例 | **67.9%** |
| 旧 mask：FN / FP 落在边界带内 | 88.3% / 65.8% |
| SAM mask：FN / FP 落在边界带内 | 84.2% / 78.9% |
| 剔除 2 px 边界带后旧 mask IoU | 0.9078 → **0.9457** |

也就是说 raw mIoU 的大头在量**标注量化噪声**，而不是"阴影块 / 拱洞填平 /
斑点"这类真实结构错误。因此协议给出一组互补指标，而不是一个数字：

| 指标 | 含义 | 用途 |
|---|---|---|
| `mIoU` | 逐图 IoU 宏平均，`--dilate 0` | 标准、严格、可与文献比 |
| `mIoU@trim2` | 剔除 GT 边界 ±2px 带后再算 | **主判据**：只看结构性错误 |
| `micro IoU` | 总交 / 总并 | 防止小物体被宏平均稀释 |
| `shape-macro` | 每个 shape 先平均再平均 | 防止 query 多的 shape 主导 |
| `p10 / median` | 分位数 | 暴露长尾失败 |
| `FP / FN`、`fp_blobs`、`fp_biggest` | 多扰/漏检像素与大连通块 | 直接量化阴影块与斑点 |
| `fp/fn_band_frac` | 误差中边界带占比 | 判断 raw 分数是否被量化主导 |
| `by_structure` | 按 GT 细结构程度分 solid/medium/thin 三档 | 细框架物体上 trim 会失效，单独看 |
| `win/tie/lose` | 与 baseline 逐图配对比较 | 比两个独立均值敏感得多 |

`--dilate 1` 是**诊断参数**，不是可以拿去报的成绩：从抗锯齿 RGB 得到的 mask
边界通常比硬边 GT 内缩约 1 px，膨胀 1 px 能看出这个偏移贡献了多少。

### 8.3 实测结论（全库均匀抽样 200 张 `*_anomaly`，跨度覆盖各 bucket）

| 指标 | 旧 `mask/` | SAM `mask_clean/` | 变化 |
|---|---|---|---|
| **raw mIoU**（严格，`--dilate 0`） | **0.9103** | 0.8977 | −0.0127 |
| **mIoU@trim2**（主判据） | 0.9522 | **0.9607** | **+0.0085** |
| micro IoU | 0.9070 | 0.8985 | −0.0085 |
| shape-macro | 0.9103 | 0.8977 | −0.0126 |
| median / p10 | 0.9299 / 0.8353 | 0.9242 / 0.8029 | — |
| 平均 FP（多扰像素） | 2689 | **1111** | **−58.7%** |
| 平均 FN（漏检像素） | 2252 | 4118 | +82.9% |
| 最大 FP 连通块（阴影块） | 1944 px | **629 px** | **−67.6%** |
| FP 连通块数（斑点） | 1.98 | **1.32** | −33.3% |
| 误差落在边界带内 FP / FN | 65.6% / 90.7% | 80.8% / 85.7% | — |
| 配对 win / tie / lose | — | 63 / 0 / 137 | 31.5% win |

**按 GT 细结构程度分层**（这是最关键的一张表）：

| GT 结构 | n | 旧 `mask/` | SAM | 差 |
|---|---|---|---|---|
| solid（实心大块） | 66 | 0.9028 | **0.9065** | **+0.0037** |
| medium | 68 | **0.9183** | 0.9086 | −0.0097 |
| thin（细框架 / 薄板） | 66 | **0.9096** | 0.8776 | **−0.0320** |

**边界补偿诊断**（同一 200 张，`eval_miou.py --dilate N`）：

| | dilate 0 | dilate 1 | dilate 2 |
|---|---|---|---|
| 旧 `mask/` | 0.9103 | 0.9077 | 0.8765 |
| SAM `mask_clean/` | 0.8977 | **0.9158** | 0.9042 |
| Δ(SAM − 旧) | −0.0127 | **+0.0080** | +0.0276 |

**怎么解读（结论是混合的，不是"全面变好"）**：

1. **单一 raw mIoU 会误导**。85–91% 的 FN、66–81% 的 FP 都在 GT 边界 ±2px 内；
   GT 是硬边渲染、查询图有抗锯齿，从 RGB 得到的 mask 边界系统性内缩约 1 px。
   给预测补 1 px 后 SAM 就从落后 0.0127 变成领先 0.0080。
2. **结构上 SAM 确实更好**：trim IoU 提升、FP 减 59%、最大阴影块减 68%、
   斑点连通块减 33%。这些正是位姿估计 chamfer 粗匹配最敏感的东西
   （大块阴影会把粗匹配拉到错误视角），而 1px 边界差异对 chamfer 几乎无影响。
3. **退化集中在细框架物体**（thin 档 −0.032）：SAM 会把细弯管、薄椅腿、
   窄板断裂漏掉；实心物体上反而略好。这是当前方案已知的短板。
4. 因此**推荐主判据是 `mIoU@trim2` + FP 结构指标 + 分层**，而不是 raw mIoU；
   `--dilate 1` 只作为边界偏移的诊断，不要当成常规成绩上报。

**可操作建议**：

- 若目标是位姿估计精度：保持 `run_segmentation.py` 默认（`--dilate 0`），
  别把 1px 边界加回去，那会略微复活阴影。
- 若目标是在官方 GT 上刷 raw mIoU：生成时加 `--dilate 1`
  （或评估时用 `eval_miou.py --dilate 1`），但要如实说明做了边界补偿。
- 后续改进方向：对被 SAM 删掉的**细长连通块**做保护（或与旧 `mask/` 的
  细结构部分融合），专门补 thin 档的 0.032 缺口。

### 8.4 用法（服务器 shape 布局）

```bash
cd ~/project/Third_Work/PTAD_plus

# 已有 mask_clean/（推荐，纯 numpy/cv2，秒级）
python segmentation/eval_miou.py \
    --root /data/xje/datasets/brokenchairs180k/shapes \
    --gt-root /data/xje/datasets/brokenchairs180k/annotations \
    --pred-subdir mask --pred-subdir mask_clean \
    --csv /data/xje/datasets/brokenchairs180k/runs/miou.csv \
    --json /data/xje/datasets/brokenchairs180k/runs/miou.json \
    --panel /data/xje/datasets/brokenchairs180k/runs/miou.png

# 还没生成 mask_clean：即时跑分割并评估（需要 onnxruntime，约 2 s/张）
python segmentation/eval_miou.py --root ... --gt-root ... \
    --pred-subdir mask --predict --limit 200 --csv miou.csv

# 生成"边界补偿版" mask 后再评估（见 8.3）
python segmentation/run_segmentation.py \
    --root /data/xje/datasets/brokenchairs180k/shapes \
    --out-subdir mask_clean_d1 --dilate 1 --workers 8
python segmentation/eval_miou.py \
    --root /data/xje/datasets/brokenchairs180k/shapes \
    --gt-root /data/xje/datasets/brokenchairs180k/annotations \
    --pred-subdir mask --pred-subdir mask_clean_d1 --csv miou_d1.csv

# 诊断：边界偏移贡献了多少 / 换 trim 半径 / 只看某几个 shape
python segmentation/eval_miou.py --root ... --gt-root ... \
    --pred-subdir mask --pred-subdir mask_clean --dilate 1
python segmentation/eval_miou.py --root ... --gt-root ... --trim-radius 0
python segmentation/eval_miou.py --root ... --gt-root ... --shape shape_1095
```

`--gt-root` 用递归扫描 `seg_new_*.png`，所以服务器上的标注无论是在
`annotations/<bucket>/` 分桶、按 shape 拆开，还是平铺，都能自动索引。
`--layout bucket` 只在评估**本地扁平发布版**（`<root>/images/<bucket>/`、
`<root>/masks/<bucket>/`）时才需要，代码与文档的默认路径一律按服务器
shape 布局。

**输出**：终端表格、逐图逐方法 CSV（含 `iou` / `iou_trim` / `fp` / `fn` /
边界带占比 / 连通块）、汇总 JSON、`miou.png`（挑 dIoU 最大/最小的样本，
RGB \| GT \| baseline \| best，全英文标注）。
