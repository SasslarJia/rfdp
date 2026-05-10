# RFDP

本仓库用于复现和运行 Zheng et al., 2021《Hybrid Regularization of Diffusion Process for Visual Re-Ranking》中的视觉重排序方法。当前推荐使用 `RFDP/MODEL/Paper_RFDP/` 这条论文复现路径；旧版通用 RFDP 框架仍保留在 `RFDP/MODEL/Frmwk_RFDP/`，主要用于兼容历史代码，不建议作为 ReID 实验主入口。

## 1. 项目做什么

仓库不训练 CNN，也不直接从图像抽特征。它接收样本间关系矩阵或 ReID 特征文件，完成：

1. 距离/特征转 affinity。
2. affinity 转 KNN 稀疏图。
3. 针对每个 query 构造局部图问题。
4. 运行 HyRDP / GMFPT / I-HyRDP / I-GMFPT。
5. 输出重排序结果。
6. 在 ReID 入口中直接计算 `mAP`、`rank-1`、`rank-5`、`rank-10` 等指标。

## 2. 推荐代码路径

论文复现主线：

```text
RFDP/Test_RFDP/reid_paper_experiment.py
-> RFDP/MODEL/Aff_RFDP/Dist2Aff.py
-> RFDP/MODEL/Aff_RFDP/KnnRstct.py
-> RFDP/MODEL/Iput_RFDP/IputWhl.py
-> RFDP/MODEL/Paper_RFDP/PaperRanker.py
-> RFDP/MODEL/Paper_RFDP/PaperIterative.py
-> RFDP/MODEL/Paper_RFDP/PaperUtils.py
-> RFDP/MODEL/Paper_RFDP/PaperSolvers.py
-> RFDP/EVAL/ReIDEval.py
```

关键概念：

- `aff_mat`：稠密 affinity，用于补全排序和局部候选选择。
- `knn_aff`：KNN 稀疏 affinity 图，用于扩散过程。
- `query`：样本全局索引列表，例如 `[[0], [1], [2]]`。
- `rankings`：每个 query 对应的全局样本 id 排序。

## 3. ReID 云服务器运行流程

假设你已经完成：

1. 拉取代码。
2. 安装依赖。
3. 用 ReID 网络导出 `.mat` 特征文件。

默认 `.mat` 需要包含：

- `query_f`
- `gallery_f`
- `query_label`
- `gallery_label`
- `query_cam`
- `gallery_cam`

### 3.1 小规模试跑

先限制 query 数量，确认字段、导入、构图、重排序和 ReID 指标都正常：

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

### 3.2 首次全量运行并保存缓存

首次正式运行建议保存 `aff_mat / knn_aff`，避免后续重复构图：

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

### 3.3 后续复用缓存调参

```bash
python3 RFDP/Test_RFDP/reid_paper_experiment.py \
  --mat-path RFDP/pytorch_result.mat \
  --cache-path RFDP/reid_graph_cache.npz \
  --method ihyrdp \
  --topk 2000 \
  --local-topk 500 \
  --save-json RFDP/reid_metrics.json
```

## 4. ReID 入口参数

入口脚本：

```bash
python3 RFDP/Test_RFDP/reid_paper_experiment.py
```

### 4.1 数据输入

- `--mat-path`：ReID 特征 `.mat` 文件路径，默认 `RFDP/pytorch_result.mat`。
- `--query-feature-key`：query 特征字段名，默认 `query_f`。
- `--gallery-feature-key`：gallery 特征字段名，默认 `gallery_f`。
- `--query-label-key`：query 身份标签字段名，默认 `query_label`。
- `--gallery-label-key`：gallery 身份标签字段名，默认 `gallery_label`。
- `--query-cam-key`：query camera id 字段名，默认 `query_cam`。
- `--gallery-cam-key`：gallery camera id 字段名，默认 `gallery_cam`。

### 4.2 图构建与缓存

- `--k-num`：`Dist2Aff` 和 `KnnRstct` 使用的 K，默认 `20`。
- `--gamma`：距离转 affinity 的缩放参数，默认 `0.4`。
- `--batch-size`：构造全量距离矩阵时的 feature batch 大小，默认 `512`。
- `--dtype`：矩阵数据类型，可选 `float32` / `float64`，默认 `float32`。
- `--save-cache`：保存本次构造出的 `aff_mat / knn_aff` 到 `.npz`。
- `--cache-path`：加载已有 `.npz` 图缓存，跳过距离矩阵和 KNN 图构造。
- `--cache-aff-key`：缓存中 full affinity 字段名，默认 `aff_mat`。
- `--cache-knn-key`：缓存中 KNN affinity 字段名，默认 `knn_aff`。

### 4.3 算法参数

- `--method`：重排序方法，可选 `hyrdp`、`gmfpt`、`ihyrdp`、`igmfpt`，默认 `ihyrdp`。
- `--hyrdp-solver`：HyRDP 求解器，可选 `iterative`、`closed`，默认 `iterative`。
- `--alpha`：HyRDP alpha，默认 `5.0`。
- `--beta`：HyRDP beta，默认 `25.0`。
- `--topk`：每个 query 保留的 RFDP 输出长度，默认全部样本。设得过小会截断 gallery 排序，可能影响 mAP。
- `--lambda-step`：迭代方法每轮加入 labeled 集合的样本数，默认 `5`，只对 `ihyrdp / igmfpt` 有效。
- `--local-topk`：每次 solver 调用的局部候选池大小，默认不开启。大规模 ReID 建议设置为 `300`、`500` 或 `1000` 先试。

### 4.4 评测与输出

- `--query-limit`：只评估前 N 个 query，默认全部 query。
- `--eval-max-rank`：CMC 最大输出深度，默认 `50`。
- `--save-json`：保存指标、配置和 CMC 曲线到 JSON。
- `--save-rankings`：保存每个 query 的全局排序列表到 `.npy`。
- `--preview`：终端预览前几个 query 的排序结果，默认 `3`。

## 5. ReID 评测协议

ReID 指标不是由 `P@K` 转换得到。`reid_paper_experiment.py` 会从重排序后的完整 gallery 排名直接计算：

- `mAP`：每个有效 query 的 average precision 均值。
- `CMC / rank-k`：第一个有效正样本是否出现在前 k 名。

过滤规则：

- 只评估 gallery 样本。
- 过滤 `gallery_label == -1`。
- 过滤 `gallery_label == query_label 且 gallery_cam == query_cam` 的同 camera 样本。

## 6. 普通矩阵实验入口

如果你已经有 `N x N` 距离矩阵、affinity 矩阵或 KNN affinity 图，可以用：

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

## 7. 内存与性能注意事项

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

## 8. 测试

论文路径测试：

```bash
python3 -m unittest RFDP.Test_RFDP.test_paper
```

ReID evaluator 测试：

```bash
python3 -m unittest RFDP.Test_RFDP.test_reid_eval
```

## 9. 导入注意事项

历史代码大量使用：

```python
from MODEL.xxx import yyy
```

因此建议从仓库根目录运行脚本。`paper_demo.py`、`paper_experiment.py` 和 `reid_paper_experiment.py` 已经把 `RFDP` 目录加入 `sys.path`，正常从仓库根目录启动即可。
