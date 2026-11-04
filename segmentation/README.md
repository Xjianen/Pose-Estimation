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
| `--mode auto\|sam\|cv\|refine\|sam2` | `auto`=有模型用 SAM、否则 CV（默认）；`cv` 纯确定性；`refine`=无模型去地板 + 边界校准（**raw mIoU 最优**，见第 9 节，输出 `mask_refined/`）；`sam2`=可选 SAM2.1 ONNX |
| `--source-subdir` | `refine` 模式读取的旧 mask 子目录，默认 `mask` |
| `--scale / --origin / --dx / --dy` | `refine` 的全局边界校准，默认 `1.01 / centre / 0 / 0`；`--scale 1.0` 关闭 |
| `--prior-threshold` | `refine` 放宽颜色先验（默认 200 = 不改旧 mask；205/210 在部分抽样上 +0.001） |
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

---

## 9. 能不能把 mIoU 提到 0.95？——换更大的 SAM 不是答案

### 9.1 先把误差拆开（400 张官方 GT）

| 旧 `mask/` 的误差 | 数值 | 含义 |
|---|---|---|
| 平均 FP / FN | 2681 / 2562 px | — |
| FP 落在 GT 边界 ±2px 内 | **29.7%** | 约 70% 是**结构性**多扰（地板/阴影） |
| FN 落在边界带内 | **55.3%** | 约 45% 是**结构性**漏检（浅色/细薄部件） |
| `mIoU@trim2` | 0.9505 | 只看内部一致性其实已经 ≈0.95 |

另外，GT 是**硬边标签渲染**、query 是**抗锯齿 RGB 阈值**，两者是同一视角的
**两次不同渲染**：实测 bbox 宽高比 `old/GT = 0.987 / 0.991`。也就是说存在约
1% 的系统性几何错位（FOV / 光栅化差异），这类误差会稳定地产生 1~2px 的
边界环。**任何分割网络都学不到这个错位**——它不是语义错误，换 SAM2/SAM3/
ViT-H 都不会凭空把预测边界挪到另一次渲染的光栅化位置上。

反过来，更大的模型还有副作用：SAM 的 1/16 分辨率 mask decoder 会**漏掉细弯管、
薄椅腿、窄板**，§8.3 里 thin 档 SAM 比旧 mask 低 0.032 就是这个原因。

### 9.2 真的试了更大的模型（同一批 60 张，严格 raw mIoU）

| 方法 | 权重 | raw mIoU |
|---|---|---|
| 旧 `mask/` | — | 0.8887 |
| SlimSAM（当前方案，ViT-B 蒸馏） | 14 MB | 0.8794 |
| SlimSAM + 1px 边界补偿 | 14 MB | 0.8971 |
| SlimSAM ∪ 旧 mask | 14 MB | 0.9052 |
| **SAM 2.1 Hiera-B+ ONNX int8**（box + 点提示） | 108 MB | **0.8606** |
| SAM 2.1 + 1.01 几何校准 | 108 MB | 0.8816 |
| 旧 mask + 1.01 几何校准（**无模型**） | 0 | **0.9039** |

SAM2 的 prompt 必须放到 1024 坐标空间并带 box（不带 box 只有 0.8120，
不带坐标换算几乎输出空 mask），上面的数字已经是调好的公平版本。
即便如此，参数量更大、架构更新的 SAM2.1 仍**低于** 14 MB 的 SlimSAM，
更低于"什么都不学、只做几何校准"的旧 mask。

**SAM 3 / 3.1 为什么没实测**：官方权重在 `facebook/sam3` 上是 **gated**
（3.4 GB `sam3.pt` + 3.4 GB safetensors，需申请访问 + `hf auth login`），
要求 Python ≥ 3.12、torch ≥ 2.7、CUDA ≥ 12.6，且没有官方完整 ONNX 导出
（社区 ONNX 主要是 tracker / text-prompt 分支）。本机是 Windows + Python 3.13、
**无 torch、无 CUDA**，无法复现；它的提升集中在开放词汇概念分割与视频跟踪，
对"单物体白底硬边标签"的亚像素边界没有针对性。**结论是"这个任务上没必要"，
不是"SAM2/SAM3 全面更差"**；服务器上有 48G 卡想复现，可用
`download_sam2.py --variant large` 和 `--mode sam2`（fp16/Large 仍未测）。

### 9.3 真正有效的方案：`refine.py`（无模型、无训练、约 5 ms/张）

```
shipped mask ──> remove_floor_plane()   只删「与图像外部连通的背景色/阴影」
                                        且位于质心以下的部分；
                                        绝不删被物体包围的浅灰座面/细腿
             ──> calibrate_boundary()  以图像中心为原点各向同性缩放 s=1.01
                                        （FOV 差异本来就是关于主点的径向缩放）
```

关键点：**从旧 mask 出发，而不是从 SAM 输出出发**。旧 mask 虽然脏，但它
保留了全部细结构；我们只做减法（去地板）和一个全局几何变换。因此 FP 大降
而 FN 不涨，正好补上 §8.3 里 SAM 的短板。

`s=1.01` 是**一个固定常数**，由 `calibrate_refine.py` 在 200 张开发集上拟合，
然后原封不动用在另外 200 张留出集上；不是逐图调参，也不是训练网络。
`--scale 1.0` 可以完全关掉它。

### 9.4 实测（官方 `seg_new_*` GT，多个不相交划分）

**(a) 校准/留出协议**（常数只在校准集上拟合，留出集完全不参与）

| 划分 | 旧 `mask/` | 去地板 | **+ 1.01 校准** | trim2（留出） |
|---|---|---|---|---|
| 400 张（200/200） | 0.9110 | 0.9235 | **0.9434**（校准集 0.9450） | 0.9639 |
| 1200 张（600/600） | 0.9116 | 0.9254 | **0.9449**（校准集 0.9446） | 0.9651 |

600 张校准集上重新网格搜索，最优仍是 **s=1.01 / centre**（dx=dy=0），
说明这个常数不是小样本过拟合。p10 0.8451 → **0.8896**。
尺度敏感性（400 张留出）：1.0075→0.9413、1.010→**0.9434**、1.0125→0.9419；
加 1px 条件膨胀、dx/dy 平移、放宽颜色先验都**没有**稳定收益。

**(b) 大样本复核**（固定常数 1.01，不再拟合；这一步用于消除抽样偏差）

| 抽样方式 | n | 旧 `mask/` | 未校准 | **refine** | **trim2** | micro |
|---|---|---|---|---|---|---|
| 固定种子随机（seed=42，推荐引用） | 2000 | 0.9002 | 0.9150 | **0.9344** | **0.9554** | 0.9337 |
| 等距（step=49） | 2000 | 0.9008 | 0.9171 | 0.9368 | 0.9581 | 0.9360 |

随机样本上 median 0.9719、p10 0.8594（旧 mask 0.8195）；
分层 solid / medium / thin = 0.9337 / 0.9368 / 0.9327，
trim2 = 0.9490 / 0.9561 / 0.9610（三档都涨，与 SAM 在 thin 档退化相反）。

> 等距抽样与不同 size 的划分之间 raw mIoU 有 ±0.005 的波动（0.934~0.945），
> 因为按 key 排序等距取样会与数据集的分桶结构产生混叠。**报告时请用随机
> 大样本的 0.9344，不要挑好看的小样本。**

### 9.5 「0.95」到底达到了没有（如实回答）

- **严格 raw mIoU：没有到 0.95**。无偏随机大样本 **0.9344**，公平的
  校准/留出划分最好 0.9449，都差 0.005~0.016。校准后剩余误差里 FP 约 50%、
  FN 约 73% 落在边界带**之外**，主要是浅色/细薄部件在 RGB 阈值与硬边标签
  之间的真实分歧，以及少量残余地板；**这部分不是换分割模型能解决的**。
- **`mIoU@trim2 = 0.9554~0.9651，稳定超过 0.95`**。按 §8.2 的协议，trim2
  才是排除标注量化噪声后的主判据。如果论文允许报 trim / boundary-tolerant
  IoU，0.95 已经达到；如果必须报未经任何容差的 raw mIoU，就报 **0.9344**
  （大样本）或 **0.9449**（1200 张留出），**不要写成 0.95**。
- 想再往上只有两条路：① 逐图几何配准（`info_*.json` 里有 `2d_bbox` /
  相机参数，但那是标注元数据，用了等于标签泄漏，不建议）；② HQ-SAM /
  亚像素 alpha matting 专门对齐硬边 GT——收益预计也只有千分之几，
  却要引入 GPU 依赖。对位姿估计而言，1px 边界差异对 chamfer 粗匹配几乎无影响。

### 9.6 用法（默认服务器 shape 布局）

```bash
cd ~/project/Third_Work/PTAD_plus

# ① 可选：在 disjoint dev split 上重新拟合校准常数，并打印留出集成绩
python segmentation/calibrate_refine.py \
    --root /data/xje/datasets/brokenchairs180k/shapes \
    --gt-root /data/xje/datasets/brokenchairs180k/annotations \
    --limit 400 --json refine_calib.json

# ② 全库生成 mask_refined/（无权重、无 GPU，8 个 shape 并行）
python segmentation/run_segmentation.py \
    --root /data/xje/datasets/brokenchairs180k/shapes \
    --mode refine --workers 8

# ③ 用同一套 mIoU 协议评估
python segmentation/eval_miou.py \
    --root /data/xje/datasets/brokenchairs180k/shapes \
    --gt-root /data/xje/datasets/brokenchairs180k/annotations \
    --pred-subdir mask --pred-subdir mask_refined \
    --csv miou_refined.csv --json miou_refined.json

# ④ 接到位姿估计
python train.py --root .../shape_1095 --codebook-mode refs \
    --mask-subdir mask_refined
```

`--mode refine` 默认读 `mask/`、写 `mask_refined/`（`--source-subdir` /
`--out-subdir` 可改）；`--scale 1.0` 关闭校准（大样本 0.9150）。
`--prior-threshold 205` 只在小样本上有 +0.001，随机大样本无收益（0.9367
vs 0.9368），默认关闭。本地扁平发布版用 `--layout bucket --mask-subdir masks`。

### 9.7 必须说清楚的边界

1. `refine.py` 使用了数据集自带的 `mask/`（阈值分割产物）和 RGB，等价于
   利用了渲染缓冲信息。它**不是纯 RGB 零样本分割**，不能与 SAM 类方法的
   "零样本分割"成绩直接比较；它的定位是**给位姿估计提供更准的查询 mask**。
2. `s=1.01` 是用官方 GT 的一个**不相交** split 拟合的全局边界补偿常数，
   必须在论文里注明；`--scale 1.0` 的结果（0.9235）才是完全未校准的成绩。
3. SAM2 的对比是 **int8 ONNX + CPU**；fp16 / SAM2-Large / 服务器 GPU 未测。
   SAM3 因权重 gated 与环境限制未实测。

### 9.8 新增文件

| 文件 | 作用 |
|---|---|
| `refine.py` | 去地板 + 全局边界校准，无模型 |
| `calibrate_refine.py` | 在 disjoint split 上拟合校准常数并报告留出集 mIoU |
| `sam2_onnx.py` | 可选 SAM 2.1 ONNX 后端（复现 §9.2 对比用） |
| `download_sam2.py` | 下载 SAM 2.1 ONNX 权重（tiny/small/base-plus/large） |
