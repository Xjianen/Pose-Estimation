# Pose-Estimation
# 查询图像到模型的对齐（用于 3D 异常检测）

估计**带纹理查询渲染图**相对**无纹理 3D 模型**的相机位姿，并导出对齐后的参考
**掩码（mask）**和**深度（depth）**，供下游异常检测阶段与查询图做比对。

两阶段，都很快：

| 阶段 | 作用 | 单张耗时 |
|---|---|---|
| **粗估计** | 将查询图剪影与模板码本匹配 → 取 top-K 位姿假设 | ~15 ms |
| **精修** | 对每个假设做可微 gsplat SE(3) 剪影对齐，保留最终 loss 最低者 | ~1.74 s（3 假设 × 80 步 + 早停） |

在 `shape_8430` 上测得（16 张查询，**只使用数据集自带的 20 个参考视角**，与 PTAD
同预算）：

| 指标 | 中位 | 均值 | 最大 | 目标 | 通过 |
|---|---|---|---|---|---|
| 旋转 | 0.571° | 0.532° | 0.923° | < 10° | **16/16** |
| 平移 | 0.0402 | 0.0384 | 0.0628 | < 0.1 | **16/16** |

单张总耗时 ~1.77 s（粗估计 15 ms + 精修 1.74 s + 导出 12 ms）；一次性初始化 0.5 s。
另外尝试过六种加速方案均被否决——见第 8 节。

---

## 1. 安装

```bash
conda activate <env>          # 需要带 CUDA 的 PyTorch
pip install gsplat opencv-python numpy
```

已在以下环境验证：Python 3.10、torch 2.7.1+cu118、gsplat 1.5.3、opencv 5.0、RTX 3090。

快速自检（用和管线**完全相同**的签名渲染几个高斯，并确认梯度能传到高斯 `means`
以及 `viewmats`）：

```bash
python check_env.py
```

---

## 2. 数据布局

```
/data/xje/datasets/brokenchairs180k/shapes/        <- --data-root
    shape_1095/
        1095.obj                 网格         （只用包围盒中心和物体尺寸）
        point_cloud/1095.ply      3DGS 模型    （纯几何白模：所有高斯为白色）
        mv_images/               参考视角：<id>_<r>_<azim>_<elev>.{png,json,npy}
        images/                  查询渲染：render_<id>_<r>_<azim>_<elev>_<type>.png
        camera/                  查询相机 json（仅部分 shape 有）
        annotations/             异常 mask / 物体 mask
    shape_8430/
    ...
```

视角参数取自文件名中**最后三个数字字段**，所以 `1095_3.0_108_20` 和
`render_1095_3.8_31_29_anomaly` 都能正确解析。**注意**：某些 shape（如 `shape_1334`）
文件名里夹了额外的 id 字段（如 `render_1334_16105_2.5_20_25_1_normal`），
此时「最后三个」是错的，正确的取法是「第一个带小数点的字段 + 紧随其后的两个字段」。
**用 `check_convention.py`（见第 4 节）能一次确认解析规则**。

缺 mesh / `.ply` / 参考 / 查询的 shape 会被报告并跳过。

---

## 3. 用法

```bash
# 单个 shape
python train.py --data-root /data/xje/datasets/brokenchairs180k/shapes \
    --shapes shape_1095 --out-dir runs/fair

# 子集（glob）或整个数据集
python train.py --data-root .../shapes --shapes "shape_10*" --out-dir runs/batch
python train.py --data-root .../shapes --max-shapes 50    --out-dir runs/batch

# 直接给单个文件夹（覆盖 --data-root）
python train.py --root /data/.../shapes/shape_1095 --out-dir runs/one

# 密集码本（而非数据集自带的参考视角；与 PTAD 不同预算，不可直接对比）
python train.py --data-root .../shapes --shapes shape_1095 \
    --codebook-mode dense --codebook-cache-dir cache --out-dir runs/dense
```

### 输出

```
runs/fair/
    shape_1095/
        mask/  render_..._mask.png     uint8 0/255   对齐后的参考剪影
        depth/ render_..._depth.npy    float32       对齐后的参考深度（物体外为 0）
    results.json                       配置 + 逐张误差 + 计时
```

终端同时打印：逐张表格、精度汇总、假设选择统计、各阶段计时。

### 主要选项

| 选项 | 含义 |
|---|---|
| `--codebook-mode refs\|dense` | 用数据集自带参考视角（公平对比 PTAD）或渲染的 azim×elev 网格 |
| `--metric chamfer\|iou` | PTAD 的对称区域倒角距离，或 mask IoU |
| `--hyps K` | 每张查询精修的假设数（`1` 关闭多假设） |
| `--iters`, `--lr` | 精修调度 |
| `--patience`, `--min-delta` | 早停：N 步内 loss 改善不足 `--min-delta` 则停（默认开启） |
| `--w-sil`, `--w-dt`, `--w-edge` | loss 权重（`--w-dt 0` 关掉距离变换项） |
| `--param auto\|camera\|object` | 位姿参数化（见第 4 节）；`auto` 优先用更快的 `camera` |
| `--batched` | 一次性把所有假设渲染进单次 gsplat 调用（实测无净收益，默认关） |
| `--sym-azim P` | 物体方位对称周期，用于评分；`0`=无（默认），`180` 用于 `shape_8430` |
| `--azim-sign` | `-1` 用于 `render_*` 查询文件名（见第 5 节） |
| `--rot-target`, `--trans-target` | 通过阈值（默认 10°、0.1） |
| `--save-rgb`, `--max-queries`, `--device`, `--verbose` | 同时导出对齐 RGB / 每 shape 查询数上限 / 设备 / 逐步日志 |
| `--azim-step`, `--elev-min/max/step`, `--cb-radius/res/focal`, `--mask-res` | 密集码本的网格和模板分辨率（仅 `--codebook-mode dense` 用） |

`python train.py --help` 列出全部选项。

---

## 4. 方法

### 粗估计：剪影码本

模板和查询图走**同一套**归一化——白底阈值取前景、包围盒方形裁剪、resize——
让匹配对物体尺度和图像位置不变。整张码本的打分是一次二值掩码矩阵乘（无网络前向）。

因为裁剪消除了尺度，**半径不是码本维度**；它由掩码大小解析求得：

```
r ≈ r_codebook · (bbox_px_codebook / bbox_px_query) · (f_query / f_codebook)
```

通常已落在真值 ~0.05–0.08 范围内。

### 精修：可微剪影对齐

优化一个 6-DoF SE(3) 增量 `ξ`；有效位姿恒为 `V = V₀ · exp(ξ)`。通过 `--param` 提供
两种等价参数化，区别只在梯度走哪条路：

| `--param` | 做法 | 代价 |
|---|---|---|
| `camera` | 把 `V₀ · exp(ξ)` 当 viewmat 传给 gsplat，高斯不动 | 梯度只有 16 个数；gsplat 不需要产出逐高斯梯度 |
| `object` | 把 `exp(ξ)` 作用到高斯 means/quats，viewmat 固定为 `V₀` | 每步一次 4×N 变换 **加** 逐高斯反向（N ≈ 414k） |
| `auto`（默认） | 探测一次 gsplat，优先 `camera` | — |

`object` 存在是因为 viewmat 梯度是版本相关特性，而 means/quats 梯度永远可用。两者
代数上完全等价（同一个有效 viewmat），应给出同一位姿——`bench_param.py` 验证并测速：

```bash
python bench_param.py --root /data/.../shape_1095 --iters 80 --repeat 3
```

**内参不优化**：`K` 由码本焦距按查询分辨率缩放得到。只精修 6-DoF 外参（平移、
也即半径，包含在内）。

Loss 是多尺度（1, ½, ¼）剪影一致性：

* **Dice + L1**：渲染 alpha 对查询 mask——重叠项；
* **距离变换**：`mean(α · d_到物体) + mean((1−α) · d_到背景)`。
  这是让精修能用的关键：它在**全图**都有非零梯度，所以能闭合大的初始视角差
  （实测能闭合 17° 俯仰差），而 Dice/IoU 在剪影不重叠时梯度为零；
* 可选的 **边缘项**，只对带明暗的模型有用（本数据集 `.ply` 是纯白，默认关）。

没有光度项——见第 5 节关于纯几何 `.ply`。

每个假设都精修，最终保留 loss 最低者。实测：粗估计 top-1 在 50% 的查询里
**不是**最优假设，所以这一步是真正起作用的。

---

## 5. 约定与陷阱

这些都是花过真时间调出来的；代码里要么断言要么直接处理掉。

**相机位姿。** 相机在以世界原点为中心的球面上，看向**网格包围盒中心**（用原点会留
~0.5° 残差）：

```
C      = r · (cos(el)cos(az), sin(el), cos(el)sin(az))      # 世界是 y-up
z = normalise(C − target);  x = normalise(up × z);  y = z × x
R_w2c  = [x y z]ᵀ ;  t = −R_w2c · C                          # OpenGL w2c（= 数据集 `RT`）
```

`train.py` 启动时对每个 shape 用所有参考的 `(r, az, el)` 重建 `RT`，**误差大于 1e-5
就拒绝运行**（实测 8e-8）。

**OpenGL 与 OpenCV。** 数据集 `RT` 是 OpenGL 世界到相机（相机看 −z，y 朝上）。
gsplat/OpenCV 看 +z。两者差 `S = diag(1,−1,−1)`，是个对合。用已知参考位姿渲染 `.ply`
得到剪影 IoU **0.975**，同时验证了约定转换和「`.ply`、网格、位姿共享同一世界系」。

**方位符号。** `mv_images/` 文件名的方位方向与存储的 `RT` 一致；`render_*` 查询
文件名的方位方向**相反**，所以标注 azim 245 的查询实际在 azim 115。由
`QuerySample.gt_pose()` 经 `--azim-sign -1` 应用。**忽略它会让一个工作的管线显示
成 ~45° 误差而非 ~1°**。

**物体对称性是逐物体的。** `shape_8430`（一个包）在**外观上**近似 180° 方位对称：
θ 与 θ+180 渲染图灰度差 **1.9/255**，而 18° 邻居差 **21.8**；剪影 IoU 对面 **0.980**
对比 18° 邻居 **0.739**——即剪影损失**主动偏好**翻转位姿。对这种物体，任何单图方法
都分不出正反面，翻转必须按等价解评分（`--sym-azim 180`）而非错误。大多数椅子
不对称，默认 `--sym-azim 0`。

**GT 精度。** 文件名的方位/俯仰是整数、半径一位小数，所以推导出的 GT 只有
~0.5° / ~0.05 的精度。低于此的误差是标签量化噪声，不是测量——**不要据此声称亚度级
精度**。

**`.ply` 是纯几何。** 每个高斯是纯白（基色 `[1,1,1]`，std 0），高阶 SH 全为 0；它
是从网格采样的，不是从图像训练的。所以渲染里唯一能和带纹理查询比较的通道是
**剪影**；RGB 没有内部结构。深度仍然有意义，是导出的内容之一。

---

## 6. 上新数据集之前：用 `check_convention.py`

在跑 `train.py` 之前，先用这个只读脚本确认新 shape（或整个新数据集）的渲染/标定
约定是否与本管线一致。它不需要网格、也不信任文件名——因为每个 shape 都同时
带文件名参数和真值 `RT`，可以让它自己跟自己对账。

```bash
# 单个 shape
python check_convention.py --root shape_1334

# 批量扫整个数据集（这是它真正的用途）
python check_convention.py --data-root /data/xje/datasets/brokenchairs180k/shapes \
    --shapes "shape_*" --max-shapes 30
```

它会检查五件事：

1. **相机中心约定**——`C = r·(cos el cos az, sin el, cos el sin az)` 能否复现存储
   `RT` 推出的中心？这一步同时验证世界轴、方位/俯仰定义、半径单位，且无需网格。
2. **look-at 目标**——仅从旋转最小二乘拟合出所有相机看向的那一点；和 `.ply`
   包围盒中心对比。残差小也能证明相机确实汇聚于一点。
3. **完整 RT 重建**——用拟合目标从 `(r, az, el)` 重建每个参考 `RT`，报告最大元素
   误差。
4. **文件名解析**——对比「最后三个数字」和「首个小数字段 + 后两个」两条规则，
   并标出「最后三个」会出错的目录（静默错误，会让所有误差数字失去意义）。
5. **资源**——`train.py` 会找到 / 缺失什么。

末尾给 `convention identical: N/M` 和需要注意的 shape 列表。

---

## 7. 仓库结构

```
train.py                 主驱动：粗估计 → 精修 → 导出 mask/depth → 指标
bench_param.py           camera vs object 参数化的等价性验证与测速
check_env.py             环境 / gsplat API 自检（含 viewmat 梯度是否可用）
check_convention.py      上新数据集前确认标定约定一致
dataset/
    shad.py              ShadDataset（多 shape_*）、ShadObject、QuerySample
utils/
    geometry.py          位姿构造、OpenGL↔OpenCV、SE(3)、误差度量、对称等价误差
    masks.py             前景、包围盒归一化、距离场
    gaussians.py         3DGS `.ply` 解析、gsplat 渲染（rgb/alpha/depth）
    pose_init.py         Codebook：from_references / render / match
    pose_refine.py       SE(3) 精修、多假设选择
    stats.py             StageTimer、PoseMetrics
```

`Codebook.match` 和 `refine_pose` 是匹配与精修的**唯一实现**——刻意不复制，因为
之前两条评估路径分叉时，曾把 1° 的结果显示成 46°。

---

## 8. 消融实验

下列命令均假设 `$D` 是 shapes 根目录，并复用码本缓存（不重测粗估计阶段）。对
本地单个 shape，把 `--data-root $D --shapes shape_8430` 换成 `--root shape_8430`。

```bash
D=/data/xje/datasets/brokenchairs180k/shapes
COMMON="--data-root $D --shapes shape_8430 --codebook-cache-dir cache"
```

### 基准（默认配置）

```bash
python train.py $COMMON --out-dir runs/ref
```

| 旋转 中位/最大 | 平移 中位/最大 | 通过 | 精修中位 |
|---|---|---|---|
| **0.571° / 0.923°** | **0.0402 / 0.0628** | **16/16** | **1.74 s** |

### 已测消融

```bash
# A1  关闭早停（跑满 80 步上限）
python train.py $COMMON --out-dir runs/a1_noearly   --patience 0

# A2  激进早停
python train.py $COMMON --out-dir runs/a2_aggr      --patience 5 --min-delta 3e-4

# A3  所有假设一次性渲染进单次 gsplat 调用
python train.py $COMMON --out-dir runs/a3_batched   --batched

# A4  变换高斯而非相机（不依赖 viewmat 梯度）
python train.py $COMMON --out-dir runs/a4_objparam  --param object

# A5  密集渲染码本（替代自带 20 个参考视角）
python train.py $COMMON --out-dir runs/a5_dense     --codebook-mode dense
```

| # | 改动 | 旋转 中位/最大 | 平移 中位/最大 | 通过 | 精修中位 |
|---|---|---|---|---|---|
| — | **默认** | 0.571° / 0.923° | 0.0402 / 0.0628 | **16/16** | 1.74 s |
| A1 | `--patience 0` | 0.614° / 1.078° | 0.0435 / 0.0614 | 16/16 | 2.22 s |
| A2 | `--patience 5 --min-delta 3e-4` | 1.160° / 179.4° | 0.0692 / 6.06 | **10/16** | 0.89 s |
| A3 | `--batched` | 0.564° / 1.078° | 0.0417 / 0.0614 | 16/16 | 1.79 s |
| A4 | `--param object` | 与 `camera` 同位姿（1e-4 内） | — | 16/16 | 13.21 vs 11.72 ms/步 |
| A5 | `--codebook-mode dense` | 粗估计 azim 1.0° / 2.0°（vs 5.0° / 12°） | 粗估计半径 0.05 / 0.10 | — | 一次性渲染 +4.3 s |

读法：**A1** 表明早停免费提速 1.28× 且不损精度，所以默认开。**A2** 表明它必须保守——
否则假设在不收敛的 loss 上排序，三张翻 180°。**A3** 持平（见第 9 节）。**A5** 让
粗估计好 5×，但最终数字几乎不变，**说明精度来自精修而非模板密度**——这也正是公平
的 20 视角设置作为默认的原因。

### 值得再跑的消融（尚未测）

```bash
# A6  关闭多假设搜索——只精修粗估计 top-1
python train.py $COMMON --out-dir runs/a6_hyps1     --hyps 1

# A7  关掉距离变换项（仅 Dice + L1 剪影损失）
python train.py $COMMON --out-dir runs/a7_nodt      --w-dt 0

# A8  粗估计用 mask IoU 替代 PTAD 的倒角距离
python train.py $COMMON --out-dir runs/a8_iou       --metric iou

# A9  减/增精修步数
python train.py $COMMON --out-dir runs/a9_iters40   --iters 40
python train.py $COMMON --out-dir runs/a9_iters160  --iters 160

# A10 用物体的 180° 对称等价评分
#     （只对对称物体如 shape_8430 有意义）
python train.py $COMMON --out-dir runs/a10_sym      --sym-azim 180
```

预期（需确认，不可假设）：

* **A6** 应该是破坏力最强的消融：粗估计 top-1 在 50% 查询里不是最优假设
  （`sel` 计数 `#0:8 #1:5 #2:3`），这一项量化多假设的贡献。
* **A7** 应对俯仰差最大的那张（`render_8430_3.9_344_37_anomaly`，与所有参考差 17°）
  破坏最大，因为距离变换项正是剪影不重叠时的梯度来源。
* **A10** 不改变当前默认数字——每个精修后的位姿都落在 GT 那一侧——但会让
  **粗估计**误差大变（旋转中位 92.8° → 8.95°），所以对称物体翻转时用它。

### 粗估计阶段单独对比

未暴露为开关，但开发时测过（见第 5 节方位符号）：

| 粗估计匹配器 | 方位 中位/最大 | 正确视角在 top-3 内 |
|---|---|---|
| 倒角距离 + 包围盒归一化（**默认**） | 5.0° / 12° | 16/16 |
| 倒角距离 + 原始 resize（不归一化） | 11.0° / 28° | 14/16 |
| 密集码本，1920 模板 | 1.0° / 2.0° | — |

---

## 9. 速度：测过的和否决的

单张查询里精修占 ~99% 成本（粗估计 15 ms、精修 1.74 s、导出 12 ms）。测过六种
加速方案均**否决**；其中四种已从代码中移除。无新证据前不要再试：

| 尝试 | 结果 | 原因 |
|---|---|---|
| camera vs object 参数化 | **仅 1.13×**（13.21 → 11.72 ms/步），位姿一致到 1e-4 | 逐高斯变换只占一步的 11% |
| 256² 渲染替代 512² | **几乎不快**，平移中位 0.0435 → 0.1016（阈值是 0.1），旋转 0.614° → 1.061° | 减半分辨率只是减半可达精度 |
| 级联：3 假设在 256²/20 步粗筛 → 只精修赢家 | 2.22 → 1.53 s 但 **13/16 通过**，3 张翻 180° | 对称物体正反两面 loss 差小于短低分辨率的噪声，粗筛选错侧 |
| 高斯抽稀到 36%（按 1/√frac 放大面片补偿） | 1.74 → 1.43 s，但平移最大 0.0629 → **0.1065**（15/16） | 2.8× 少的高斯只换 11% 提速；稀疏剪影偏置位姿 |
| 激进早停（patience 5, min-delta 3e-4） | 1.74 → 0.89 s 但 **10/16 通过** | 假设在未收敛的 loss 上比较，选错假设 |
| 所有假设批进一次 gsplat 调用 | **无净收益**：1.79 s 对 1.74 s | 单相机渲染便宜 1.33×（9.94 → 7.46 ms），但早停从「逐假设」变「整批」，多跑 1.37× 渲染——两者抵消 |

几轮才看清的两条规律：

1. **gsplat 的成本随「渲染的相机数」缩放，对像素数和高斯数都不敏感。** 减半分辨率
   或砍 2.8× 高斯，各只让每步快 ~10%；批 3 相机让单相机成本降 1.33×。没有大块固定
   开销可摊薄，所以「减少渲染工作量」这条路整体是死的。
2. **假设选择必须基于已收敛的 loss。** 所有提前排序的方案（级联、激进早停）都选错
   假设并丢失查询到 180° 翻转。初始 loss 甚至是**反向指标**——某次查询三个假设
   起 0.1939 / 0.2010 / 0.2054，终 0.01294 / 0.00778 / 0.00755，**完全倒序**。

因此保留配置就是默认：所有假设顺序精修、全分辨率、完整模型，配保守早停
（`--patience 10 --min-delta 1e-4`），它同时是测过最快和最准的变体。

---

## 10. 局限

* 只在一个物体（`shape_8430`，16 张查询）上验证过。在 `brokenchairs180k` 多个
  椅子上做广验证是显然的下一步，这也是 `train.py` 多 shape 循环的用途。
* 报告的误差在 GT 量化地板附近（第 5 节），所以更细的对比需要未取整的相机参数
  （`camera/*.json` 带浮点 `azimuth_deg`，有就用）。
* 对称物体位姿只确定到对称群。当对齐渲染用于 2D 异常比对时无害（两种位姿渲染几乎
  一样），但若要把异常在 3D 中定位到 mesh 上则会错（映射到镜像位置）。
