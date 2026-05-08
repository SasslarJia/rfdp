# RFDP

本仓库用于实现和验证 RFDP 相关的视觉重排序方法。当前代码库已经补入了 Zheng 等人在 2021 年论文 *Hybrid Regularization of Diffusion Process for Visual Re-Ranking* 中核心方法的代码级实现，并以一条独立的、面向论文语义的 NumPy 路径提供，不会改写旧版 RFDP 代码行为。

## 当前结论

可以认为：代码库现在已经按论文提出的方法完成了核心算法层面的实现。

这里的“已实现”指：

- 已实现论文中的局部问题构造方式，包括 `L / A / B` 划分。
- 已实现 HyRDP 的闭式求解与迭代求解。
- 已实现 GMFPT 的闭式求解。
- 已实现论文中的迭代式重排序流程，包括在非连通图上补临时连接的策略。
- 已提供可运行的 synthetic demo 和单元测试用于方法级验证。

这里的“未完全覆盖”指：

- 当前仓库还没有把论文中的整套真实数据实验、评测脚本、`.mat` 数据准备流程完整复刻到可直接复现实验结果的程度。
- 仓库中旧的 RFDP 路径仍然保留，作为兼容代码存在；论文精确实现应使用新增的 `MODEL/Paper_RFDP/` 路径。

## 依赖环境

- `python3`
- `numpy`

安装示例：

```bash
python3 -m pip install --user numpy
```

## 快速开始

项目根目录为当前仓库根目录，示例命令如下：

```bash
python3 RFDP/Test_RFDP/paper_demo.py
```

预期输出示例：

```text
Single-shot HyRDP: [0, 1, 3, 4]
Single-shot GMFPT: [0, 1, 3, 4]
Iterative HyRDP: [0, 1, 2, 3]
Iterative GMFPT: [0, 1, 2, 3]
```

其中：

- `Single-shot` 表示单次重排序。
- `Iterative` 表示论文中的迭代重排序流程。
- 示例里 `Iterative` 能把原本不在同一连通分量中的隐藏相关样本逐步找回来。

如果希望运行论文实现路径的端到端实验脚本，可以使用：

```bash
python3 RFDP/Test_RFDP/paper_experiment.py
```

这个脚本默认运行 synthetic 示例，也支持真实数据矩阵输入，例如：

```bash
python3 RFDP/Test_RFDP/paper_experiment.py \
  --data-path /path/to/IDSC_1000.mat \
  --data-key all_dists \
  --input-kind distance \
  --cls-num 50 \
  --ele-num 20 \
  --method ihyrdp \
  --topk 20 \
  --k-num 20
```

如果输入已经是相似度矩阵，可改为：

```bash
python3 RFDP/Test_RFDP/paper_experiment.py \
  --data-path /path/to/affinity.npy \
  --input-kind affinity \
  --method hyrdp \
  --topk 20
```

## 运行测试

运行论文实现对应的单元测试：

```bash
python3 -m unittest RFDP.Test_RFDP.test_paper
```

当前测试覆盖的重点包括：

- `A / B` 集合划分是否正确。
- HyRDP 闭式解与迭代解是否一致。
- GMFPT 方程是否满足论文定义。
- 临时连接策略是否按预期工作。
- 迭代式 HyRDP 是否能恢复跨连通分量的隐藏相关样本。

## 使用方法

### 1. 构造输入模型

论文路径的基础输入类是 `IputWhl`，位于：

- [RFDP/MODEL/Iput_RFDP/IputWhl.py](/Users/jqk/PycharmProjects/rfdp/RFDP/MODEL/Iput_RFDP/IputWhl.py)

其构造方式：

```python
from MODEL.Asst_RFDP.OperAsstNumpy import OperAsstNumpy
from MODEL.Iput_RFDP.IputWhl import IputWhl

asst = OperAsstNumpy()
mdl_iput = IputWhl(
    smth_w=base_w,
    fit_w=base_w,
    full_w=full_w,
    asst_rela=asst,
)
```

参数说明：

- `smth_w`：平滑约束图的权重矩阵，形状为 `n x n`。
- `fit_w`：拟合项对应矩阵。当前论文路径通常可直接传 `base_w`。
- `full_w`：完整相似度矩阵，形状为 `n x n`。迭代补边和尾部补全会用到它。
- `asst_rela`：矩阵操作助手。论文路径建议使用 `OperAsstNumpy()`。

### 2. 单次重排序

HyRDP：

```python
from MODEL.Paper_RFDP.PaperRanker import HyRdpPaper
from MODEL.Paper_RFDP.PaperSolvers import HyRdpClosedForm

ranker = HyRdpPaper(
    mdl_iput,
    solver=HyRdpClosedForm(),
    alpha=5.0,
    beta=25.0,
    incl_qry=False,
)
result = ranker([[0]], 4)
```

GMFPT：

```python
from MODEL.Paper_RFDP.PaperRanker import GmfptPaper

ranker = GmfptPaper(
    mdl_iput,
    incl_qry=False,
)
result = ranker([[0]], 4)
```

入参说明：

- `iput_qries`：查询列表，类型为 `list[list[int]]`。每个子列表是一条 query 对应的已标注样本索引集合。
- `iput_len`：希望返回的重排序长度。

出参说明：

- 返回类型为 `list[list[int]]`。
- 每个子列表对应一条 query 的重排序结果，是样本索引按相关性重排后的列表。

### 3. 迭代式重排序

I-HyRDP：

```python
from MODEL.Paper_RFDP.PaperIterative import IHyRdpPaper

iter_ranker = IHyRdpPaper(
    base_ranker=ranker,
    nq=4,
    lambda_step=1,
)
result = iter_ranker([[0]])
```

I-GMFPT：

```python
from MODEL.Paper_RFDP.PaperIterative import IGmfptPaper

iter_ranker = IGmfptPaper(
    base_ranker=ranker,
    nq=4,
    lambda_step=1,
)
result = iter_ranker([[0]])
```

入参说明：

- `base_ranker`：基础排序器，通常是 `HyRdpPaper` 或 `GmfptPaper` 实例。
- `nq`：最终希望累计获得的 query 扩展规模。
- `lambda_step`：每轮迭代最多新加入的样本数，也对应临时连接的桥接强度控制。

出参说明：

- 返回类型为 `list[list[int]]`。
- 每个子列表是迭代扩展后的最终样本索引序列，长度最多为 `nq`。

## 关键接口

### 单次排序入口

- `HyRdpPaper.__call__(iput_qries: list, iput_len: int) -> list`
- `GmfptPaper.__call__(iput_qries: list, iput_len: int) -> list`

代码文件：

- [RFDP/MODEL/Paper_RFDP/PaperRanker.py](/Users/jqk/PycharmProjects/rfdp/RFDP/MODEL/Paper_RFDP/PaperRanker.py)

### 迭代排序入口

- `IHyRdpPaper.__call__(iput_qries: list) -> list`
- `IGmfptPaper.__call__(iput_qries: list) -> list`

代码文件：

- [RFDP/MODEL/Paper_RFDP/PaperIterative.py](/Users/jqk/PycharmProjects/rfdp/RFDP/MODEL/Paper_RFDP/PaperIterative.py)

### 关键求解器

- `HyRdpClosedForm.solve(problem, alpha, beta)`
- `HyRdpIterative.solve(problem, alpha, beta)`
- `GmfptClosedForm.solve(problem)`

代码文件：

- [RFDP/MODEL/Paper_RFDP/PaperSolvers.py](/Users/jqk/PycharmProjects/rfdp/RFDP/MODEL/Paper_RFDP/PaperSolvers.py)

### 论文局部问题建模

- `build_paper_problem(weight_mat, sub_ids, query_ids)`
- `PaperProblem`

代码文件：

- [RFDP/MODEL/Paper_RFDP/PaperUtils.py](/Users/jqk/PycharmProjects/rfdp/RFDP/MODEL/Paper_RFDP/PaperUtils.py)
- [RFDP/MODEL/Paper_RFDP/PaperProblem.py](/Users/jqk/PycharmProjects/rfdp/RFDP/MODEL/Paper_RFDP/PaperProblem.py)

### 临时连接与候选扩展

- `add_temporary_connections(base_w, full_w, active_ids, bridge_num)`
- `build_shell_subset(weight_mat, labeled_ids)`
- `rank_by_similarity(full_w, query_ids, excl_ids, top_num=None)`

代码文件：

- [RFDP/MODEL/Paper_RFDP/PaperUtils.py](/Users/jqk/PycharmProjects/rfdp/RFDP/MODEL/Paper_RFDP/PaperUtils.py)

## 关键文件映射

- [RFDP/MODEL/Paper_RFDP/PaperRanker.py](/Users/jqk/PycharmProjects/rfdp/RFDP/MODEL/Paper_RFDP/PaperRanker.py)：单次 HyRDP / GMFPT 排序入口。
- [RFDP/MODEL/Paper_RFDP/PaperIterative.py](/Users/jqk/PycharmProjects/rfdp/RFDP/MODEL/Paper_RFDP/PaperIterative.py)：I-HyRDP / I-GMFPT 迭代流程。
- [RFDP/MODEL/Paper_RFDP/PaperSolvers.py](/Users/jqk/PycharmProjects/rfdp/RFDP/MODEL/Paper_RFDP/PaperSolvers.py)：论文公式对应求解器。
- [RFDP/MODEL/Paper_RFDP/PaperProblem.py](/Users/jqk/PycharmProjects/rfdp/RFDP/MODEL/Paper_RFDP/PaperProblem.py)：局部问题数据结构。
- [RFDP/MODEL/Paper_RFDP/PaperUtils.py](/Users/jqk/PycharmProjects/rfdp/RFDP/MODEL/Paper_RFDP/PaperUtils.py)：问题构建、补边、排序辅助函数。
- [RFDP/MODEL/Iput_RFDP/IputWhl.py](/Users/jqk/PycharmProjects/rfdp/RFDP/MODEL/Iput_RFDP/IputWhl.py)：输入图与连通分量组织。
- [RFDP/Test_RFDP/paper_demo.py](/Users/jqk/PycharmProjects/rfdp/RFDP/Test_RFDP/paper_demo.py)：最小可运行示例。
- [RFDP/Test_RFDP/paper_experiment.py](/Users/jqk/PycharmProjects/rfdp/RFDP/Test_RFDP/paper_experiment.py)：论文路径端到端实验脚本，支持 synthetic / `.mat` / `.npy` / `.npz` 输入。
- [RFDP/Test_RFDP/test_paper.py](/Users/jqk/PycharmProjects/rfdp/RFDP/Test_RFDP/test_paper.py)：论文路径单元测试。
- [RFDP/PAPER_IMPL_NOTES.md](/Users/jqk/PycharmProjects/rfdp/RFDP/PAPER_IMPL_NOTES.md)：实现范围说明。

## 与旧实现的关系

- 旧代码路径主要位于 `RFDP/MODEL/Frmwk_RFDP/`、`RFDP/MODEL/Fit_RFDP/`、`RFDP/MODEL/Sol_RFDP/`。
- 这些旧模块没有被删除，目的是保持兼容性。
- 如果目标是使用与论文方法语义对齐的代码，应优先使用 `RFDP/MODEL/Paper_RFDP/` 下的新接口。
- 如果目标是运行仓库中早期 demo 或历史脚本，需要注意其中部分路径依赖外部数据文件，当前并不保证开箱即用。

## 当前建议

如果后续要继续补全到“论文实验可复现”层级，下一步通常是：

- 接入真实数据集读取与 `.mat` 适配。
- 补充论文实验配置。
- 对齐论文评测指标与结果产出格式。
- 增加端到端实验脚本。
