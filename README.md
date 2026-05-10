# RFDP

本仓库是一个面向视觉重排序（visual re-ranking）的研究型代码库，核心目标是复现和运行 Zheng et al., 2021《Hybrid Regularization of Diffusion Process for Visual Re-Ranking》中的扩散重排序方法。

当前推荐主线是 `RFDP/MODEL/Paper_RFDP/`。仓库中仍保留旧版通用 RFDP 框架 `RFDP/MODEL/Frmwk_RFDP/`，但它主要用于历史兼容和早期实验，不建议作为论文复现或 ReID 实验的主入口。

## 1. 项目定位

仓库不负责 CNN 训练，也不直接从图像抽特征。它默认上游已经给出样本间关系，或者给出可转成关系矩阵的特征，例如：

- ReID 网络导出的 `query_f / gallery_f`。
- 全量欧氏距离矩阵 `all_dists`。
- 稠密 affinity 矩阵 `aff_mat`。
- 已稀疏化的 KNN 图 `knn_aff`。

整体流程是：

```text
features or distance matrix
-> affinity matrix
-> sparse KNN graph
-> query-specific local problem
-> HyRDP / GMFPT solver
-> reranked sample ids
-> retrieval metrics
```

## 2. 推荐代码主线

论文复现路径的核心执行链路：

```text
RFDP/Test_RFDP/reid_paper_experiment.py     # ReID 特征文件入口
RFDP/Test_RFDP/paper_experiment.py          # 普通矩阵入口
RFDP/MODEL/Aff_RFDP/Dist2Aff.py             # distance -> affinity
RFDP/MODEL/Aff_RFDP/KnnRstct.py             # affinity -> KNN graph
RFDP/MODEL/Iput_RFDP/IputWhl.py             # query -> connected component subset
RFDP/MODEL/Paper_RFDP/PaperRanker.py        # single-shot HyRDP / GMFPT ranker
RFDP/MODEL/Paper_RFDP/PaperIterative.py     # iterative I-HyRDP / I-GMFPT
RFDP/MODEL/Paper_RFDP/PaperUtils.py         # L/A/B split, local subgraph, temporary edges
RFDP/MODEL/Paper_RFDP/PaperSolvers.py       # Eq. 13/14/15/20 solvers
RFDP/EVAL/ReIDEval.py                       # ReID mAP / CMC evaluator
```

旧版通用框架路径：

```text
RFDP/MODEL/Frmwk_RFDP/
RFDP/MODEL/Smth_RFDP/
RFDP/MODEL/Fit_RFDP/
RFDP/MODEL/Sol_RFDP/
```

这条旧路径把方法抽象成“平滑约束 + 拟合约束 + 求解器”，但没有直接表达论文中的 `L/A/B` 局部划分、HyRDP 闭式解、GMFPT 解和迭代临时连边机制。因此理解或使用 Zheng 2021 方法时，应优先看 `Paper_RFDP`。

## 3. 数据结构与输出

### 3.1 样本编号

仓库内部使用全局样本 id。对于 ReID 入口，默认拼接顺序是：

```text
all_features = [query_f; gallery_f]
```

因此：

- query 全局 id 范围是 `[0, query_num)`。
- gallery 全局 id 范围是 `[query_num, query_num + gallery_num)`。

### 3.2 Query 形式

query 是样本 id 列表，而不是文本或图片：

```python
[[0], [1], [20]]
```

多样本 query 也可以表示为：

```python
[[0, 5], [10, 12]]
```

### 3.3 Rank 输出

排序输出是：

```python
list[list[int]]
```

每个子列表是一条 query 的全局样本 id 排序，例如：

```python
[[0, 7, 3, 10], [1, 4, 9, 8]]
```

ReID 评测时会再把全局排序转换成 gallery 排序，并过滤无效 gallery。

## 4. Paper_RFDP 如何对应论文

### 4.1 图构建

如果输入是距离矩阵，会先执行：

```text
distance -> Dist2Aff -> affinity -> KnnRstct -> sparse graph
```

对应论文中的相似度构造和 reciprocal KNN 图构建。

### 4.2 局部问题构造

`PaperUtils.build_paper_problem()` 会把局部子图切成：

- `L`：已标注集合，也就是当前 query / labeled samples。
- `A`：与 `L` 直接相连的未标注点。
- `B`：不与 `L` 直接相连、但仍在局部子图里的未标注点。

它同时构造 `W_UU`、`W_UL`、`h_u`、`d_u`、`e_u`、`P_UU` 等 solver 所需变量。

### 4.3 求解器

`PaperSolvers.py` 当前实现：

- `HyRdpClosedForm`：HyRDP 闭式解，对应论文 Eq. 13/14。
- `HyRdpIterative`：HyRDP 迭代解，对应论文 Eq. 15/16。
- `GmfptClosedForm`：GMFPT 闭式解，对应论文 Eq. 20。

### 4.4 迭代重排序

`PaperIterative.py` 实现：

- `IHyRdpPaper`
- `IGmfptPaper`

每轮从当前 labeled 集合出发，构造 shell subset，求解局部问题，然后把 top `lambda_step` 个样本加入 labeled 集合。对于非连通图，会用 `full_w` 添加少量临时连接。

## 5. ReID 应用流程

ReID 专用入口：

```bash
python3 RFDP/Test_RFDP/reid_paper_experiment.py
```

默认 `.mat` 字段：

- `query_f`
- `gallery_f`
- `query_label`
- `gallery_label`
- `query_cam`
- `gallery_cam`

该入口会自动完成：

1. 读取 ReID 特征、标签和 camera id。
2. 拼接 query/gallery 特征。
3. 对特征做 L2 normalization。
4. 构造全量 pairwise L2 distance。
5. 构造 `aff_mat / knn_aff`。
6. 调用 `Paper_RFDP` 重排序。
7. 按 ReID 协议计算 `mAP` 和 CMC。

### 5.1 小规模试跑

```bash
python3 RFDP/Test_RFDP/reid_paper_experiment.py \
  --mat-path RFDP/pytorch_result.mat \
  --method hyrdp \
  --hyrdp-solver iterative \
  --k-num 20 \
  --topk 2000 \
  --local-topk 500 \
  --query-limit 50
```

### 5.2 全量运行并保存缓存

```bash
python3 RFDP/Test_RFDP/reid_paper_experiment.py \
  --mat-path RFDP/pytorch_result.mat \
  --method ihyrdp \
  --k-num 20 \
  --topk 2000 \
  --local-topk 500 \
  --save-cache RFDP/reid_graph_cache.npz \
  --save-json RFDP/reid_metrics.json \
  --save-rankings RFDP/reid_rankings.npy
```

### 5.3 复用缓存调参

```bash
python3 RFDP/Test_RFDP/reid_paper_experiment.py \
  --mat-path RFDP/pytorch_result.mat \
  --cache-path RFDP/reid_graph_cache.npz \
  --method ihyrdp \
  --topk 2000 \
  --local-topk 500 \
  --save-json RFDP/reid_metrics.json
```

## 6. ReID 参数速查

数据字段：

- `--mat-path`：ReID 特征 `.mat` 文件路径，默认 `RFDP/pytorch_result.mat`。
- `--query-feature-key`：query 特征字段名，默认 `query_f`。
- `--gallery-feature-key`：gallery 特征字段名，默认 `gallery_f`。
- `--query-label-key`：query 身份标签字段名，默认 `query_label`。
- `--gallery-label-key`：gallery 身份标签字段名，默认 `gallery_label`。
- `--query-cam-key`：query camera id 字段名，默认 `query_cam`。
- `--gallery-cam-key`：gallery camera id 字段名，默认 `gallery_cam`。

图构建与缓存：

- `--k-num`：`Dist2Aff` 和 `KnnRstct` 使用的 K，默认 `20`。
- `--gamma`：距离转 affinity 的缩放参数，默认 `0.4`。
- `--batch-size`：构造全量距离矩阵时的 feature batch 大小，默认 `512`。
- `--dtype`：矩阵数据类型，可选 `float32` / `float64`，默认 `float32`。
- `--save-cache`：保存本次构造出的 `aff_mat / knn_aff`。
- `--cache-path`：加载已有 `.npz` 图缓存。
- `--cache-aff-key`：缓存中 full affinity 字段名，默认 `aff_mat`。
- `--cache-knn-key`：缓存中 KNN affinity 字段名，默认 `knn_aff`。

算法：

- `--method`：可选 `hyrdp`、`gmfpt`、`ihyrdp`、`igmfpt`，默认 `ihyrdp`。
- `--hyrdp-solver`：可选 `iterative`、`closed`，默认 `iterative`。
- `--alpha`：HyRDP alpha，默认 `5.0`。
- `--beta`：HyRDP beta，默认 `25.0`。
- `--topk`：每个 query 保留的 RFDP 输出长度，默认全部样本。设得过小会截断 gallery 排序，可能影响 mAP。
- `--lambda-step`：迭代方法每轮加入 labeled 集合的样本数，默认 `5`。
- `--local-topk`：每次 solver 调用的局部候选池大小，默认不开启；大规模 ReID 建议先试 `300`、`500` 或 `1000`。

评测与输出：

- `--query-limit`：只评估前 N 个 query，默认全部。
- `--eval-max-rank`：CMC 最大输出深度，默认 `50`。
- `--save-json`：保存指标、配置和 CMC 曲线到 JSON。
- `--save-rankings`：保存每个 query 的全局排序列表到 `.npy`。
- `--preview`：终端预览前几个 query 的排序结果，默认 `3`。

## 7. ReID 评测协议

ReID 指标不是由 `P@K` 转换得到。`reid_paper_experiment.py` 会从重排序后的 gallery 排名直接计算：

- `mAP`：每个有效 query 的 average precision 均值。
- `CMC / rank-k`：第一个有效正样本是否出现在前 k 名。

过滤规则：

- 只评估 gallery 样本。
- 过滤 `gallery_label == -1`。
- 过滤 `gallery_label == query_label 且 gallery_cam == query_cam` 的同 camera 样本。

## 8. 普通矩阵实验

如果已经有 `N x N` 距离矩阵、affinity 矩阵或 KNN affinity 图，可以用：

```bash
python3 RFDP/Test_RFDP/paper_experiment.py
```

常用参数：

- `--data-path`
- `--data-key`
- `--input-kind distance|affinity|knn_affinity`
- `--label-path`
- `--label-key`
- `--method hyrdp|gmfpt|ihyrdp|igmfpt`
- `--topk`
- `--query-limit`
- `--k-num`
- `--local-topk`
- `--cache-path`
- `--save-json`

最小 demo：

```bash
python3 RFDP/Test_RFDP/paper_demo.py
```

## 9. 内存与性能

`reid_paper_experiment.py` 会把 `query + gallery` 拼成总样本数 `N`，并构造 `N x N` 矩阵。单个 `float32` 方阵约占：

```text
N * N * 4 bytes
```

Market1501 常见规模约 `3368 query + 19732 gallery = 23100`，单个 `float32` 方阵约 `2.1GB`。实际运行时可能同时存在距离矩阵、`aff_mat`、`knn_aff` 和局部临时变量，所以服务器内存应明显高于单矩阵大小。

建议：

- 使用 `--dtype float32`。
- 首次全量运行使用 `--save-cache`。
- 后续调参使用 `--cache-path`。
- 大规模实验设置 `--local-topk`。
- 先用 `--query-limit` 小规模验证。

## 10. 测试

论文路径测试：

```bash
python3 -m unittest RFDP.Test_RFDP.test_paper
```

ReID evaluator 测试：

```bash
python3 -m unittest RFDP.Test_RFDP.test_reid_eval
```

## 11. 导入注意事项

历史代码大量使用：

```python
from MODEL.xxx import yyy
```

因此建议从仓库根目录运行脚本。`paper_demo.py`、`paper_experiment.py` 和 `reid_paper_experiment.py` 已经把 `RFDP` 目录加入 `sys.path`，正常从仓库根目录启动即可。
