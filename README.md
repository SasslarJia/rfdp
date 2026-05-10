# RFDP

本仓库是一个面向视觉重排序（visual re-ranking）的研究型代码库，核心主题是基于图上的扩散过程（diffusion process）做检索结果重排。它当前同时保留了两条实现主线：

- 一条是仓库原有的通用 RFDP 框架，用“平滑约束 + 拟合约束 + 求解器”的模块化方式组合不同方法。
- 一条是后来补入的、尽量贴近 Zheng et al., 2021《Hybrid Regularization of Diffusion Process for Visual Re-Ranking》论文语义的实现路径。

因此，这个仓库现在不是单一算法的脚本集合，而是一个“图构建 + 子问题组织 + 扩散求解 + 重排序输出”的实验平台。

## 1. 当前代码库到底在做什么

如果用一句话概括，本仓库做的是：

1. 接收一个样本间的距离矩阵或相似度矩阵。
2. 把它变成图上的权重矩阵。
3. 针对每个 query，在图上构建一个局部或连通子问题。
4. 用 RFDP / HyRDP / GMFPT 这类扩散或正则化方法求得新的相关性分数。
5. 输出每个 query 对应的重排序结果。

它不负责 CNN 训练，也不直接抽特征。更准确地说，它默认假设上游已经给出了样本间关系，例如：

- 欧氏距离矩阵 `all_dists`
- 相似度/亲和度矩阵 `aff_mat`
- 已经做过 KNN 稀疏化的图 `knn_aff`

在这个前提下，仓库主要解决“如何从样本关系图出发进行 rerank”。

## 2. 当前仓库的两条主线

### 2.1 旧版 RFDP 框架路径

这条路径是仓库原生设计，特点是组件化强。它把一个方法拆成：

- 输入组织模块
- 平滑约束模块
- 拟合约束模块
- 求解器模块
- 排序输出模块

主入口在：

- `RFDP/MODEL/Frmwk_RFDP/Rfdp.py`
- `RFDP/MODEL/Frmwk_RFDP/RfdpCtofOtcm.py`

这条路径更像“通用框架”，可以替换不同约束和求解策略。

### 2.2 论文精确实现路径

这条路径主要位于：

- `RFDP/MODEL/Paper_RFDP/`

它不是在旧框架上简单调参，而是新增了一套更贴近论文定义的实现，重点补了：

- 论文里的 `L / A / B` 局部集合划分
- HyRDP 闭式求解
- HyRDP 迭代求解
- GMFPT 闭式求解
- 论文式迭代重排序
- 非连通图下的临时连接策略

因此，现在仓库内部实际是“并行存在两套实现思路”：

- `Frmwk_RFDP/` 偏原框架
- `Paper_RFDP/` 偏论文复现

## 3. 仓库里数据是怎样流动的

这是理解整个项目最重要的一部分。

### 3.1 输入形式

当前代码支持的核心输入不是图像，而是样本关系矩阵。常见有三种：

1. 距离矩阵
   例如 `all_dists`，通常来自 CNN 特征后的欧氏距离。
2. 稠密相似度矩阵
   例如 `aff_mat`。
3. 已稀疏化的 KNN 图
   例如 `knn_aff`。

在 `RFDP/Test_RFDP/paper_experiment.py` 中，对应参数是：

- `--input-kind distance`
- `--input-kind affinity`
- `--input-kind knn_affinity`

如果输入是距离矩阵，通常会走：

`distance -> Dist2Aff -> affinity -> KnnRstct -> sparse graph`

对应文件：

- `RFDP/MODEL/Aff_RFDP/Dist2Aff.py`
- `RFDP/MODEL/Aff_RFDP/KnnRstct.py`

### 3.2 Query 形式

仓库内部的 query 不是文本或图片，而是“索引列表”：

```python
[[0], [1], [20]]
```

或者多样本 query：

```python
[[0, 5], [10, 12]]
```

也就是说，每条 query 本质上是“当前已知相关的样本 id 集合”。

### 3.3 输出形式

输出是一个 `list[list[int]]`，每条 query 对应一个重排序后的样本索引列表。例如：

```python
[[0, 7, 3, 10], [1, 4, 9, 8]]
```

如果通过 `paper_experiment.py` 保存结果，则还可以输出：

- 排名结果 JSON
- Precision / Recall 指标
- PR 曲线

## 4. 整体运算流程

从整个仓库视角看，内部运算可以理解为下面 5 个阶段。

### 4.1 阶段一：把输入矩阵变成图

相关文件：

- `RFDP/MODEL/Aff_RFDP/Dist2Aff.py`
- `RFDP/MODEL/Aff_RFDP/KnnRstct.py`

作用：

- `Dist2Aff`：把距离矩阵变成亲和度矩阵。
- `KnnRstct`：只保留每个节点的 KNN 邻居，构建更稀疏的图。

对大规模数据来说，这一步非常关键，因为后续的连通分量、局部建模、迭代求解都依赖图结构。如果把稠密距离矩阵直接当图传进去，后续会非常慢。

### 4.2 阶段二：输入组织器决定每条 query 该在哪个子图上求解

相关文件：

- `RFDP/MODEL/Iput_RFDP/MdlIput.py`
- `RFDP/MODEL/Iput_RFDP/IputWhl.py`
- `RFDP/MODEL/Iput_RFDP/IputLyr.py`

作用：

- `MdlIput`：输入组织层的抽象基类，负责保存 `smth_w / fit_w / full_w`，并计算连通分量。
- `IputWhl`：按整个连通分量组织 query 子问题。
- `IputLyr`：按 query 邻域层数组织子问题。

当前默认更常用的是 `IputWhl`，即：

- 先找出 query 落在哪些连通分量里
- 再把这些连通分量里的样本作为子问题候选集

`MdlIput.py` 现在还包含两个很重要的性能点：

- 连通分量优先用 `scipy.sparse.csgraph.connected_components`
- 边列表按需延迟构建，不再一开始就强制把整图转成 Python 邻接表

### 4.3 阶段三：对每个 query 构造局部优化问题

这一步分成旧路径和论文路径两种逻辑。

#### 旧路径

在 `RFDP/MODEL/Frmwk_RFDP/Rfdp.py` 中：

1. `self._mdl_iput(qry_lst, self._incl_qry)`  
   先把 query 分到对应子块。
2. `self._mdl_iput.get_eles(q2b.rela_ids)`  
   取出当前块里的样本。
3. `self._smth_cstr(...)`  
   生成平滑项。
4. `self._fit_cstr(...)`  
   生成拟合项。
5. `self._sol_proc(...)`  
   求解。

#### 论文路径

在 `RFDP/MODEL/Paper_RFDP/PaperRanker.py` 中：

1. 先通过 `MdlIput` 找到 query 的默认候选子集。
2. 如果设置了 `local_topk`，进一步把候选集截断成更小的局部子图。
3. 调用 `build_paper_problem(...)` 构造论文定义的局部问题。

这个局部问题的核心在：

- `L`：已标注的 query 集
- `A`：与 `L` 直接相连的未标注点
- `B`：未与 `L` 直接相连，但仍在局部子图内的点

对应文件：

- `RFDP/MODEL/Paper_RFDP/PaperUtils.py`
- `RFDP/MODEL/Paper_RFDP/PaperProblem.py`

### 4.4 阶段四：求解扩散/正则化方程

#### 旧路径的求解

旧路径把求解过程抽象成通用接口：

- `RFDP/MODEL/Smth_RFDP/SmthCstr.py`
- `RFDP/MODEL/Fit_RFDP/FitCstr.py`
- `RFDP/MODEL/Sol_RFDP/SolProc.py`

默认组合常见于：

- `RFDP/MODEL/Frmwk_RFDP/RfdpCtofOtcm.py`

例如默认搭配是：

- 平滑项：`PairwiseSC`
- 拟合项：`HyDiffFC`
- 求解器：`ItrtLinGrw` 或 `ItrtExpGrw`

它本质上是先形成一个图正则化问题，再用迭代方式求一个排序分数。

#### 论文路径的求解

对应文件：

- `RFDP/MODEL/Paper_RFDP/PaperSolvers.py`

当前主要实现了：

- `HyRdpClosedForm`
- `HyRdpIterative`
- `GmfptClosedForm`

这些 solver 接收的是 `PaperProblem`，也就是已经切分好的局部问题，而不是旧框架里那种“平滑项 + 拟合项”的通用接口。

### 4.5 阶段五：把解转成最终排序结果

无论是旧路径还是论文路径，最后都要做两件事：

1. 把局部求解得到的分数转成样本排序。
2. 如果局部子图里的候选不够，再从 `full_w` 里补全剩余结果。

旧路径中，这一步主要在：

- `RFDP/MODEL/Frmwk_RFDP/Rfdp.py`

相关方法：

- `_sol2rank()`
- `_cplt_oput()`

论文路径中，这一步主要在：

- `RFDP/MODEL/Paper_RFDP/PaperRanker.py`

相关方法：

- `rank_subset()`
- `_complete_output()`

## 5. 论文路径的内部运算过程

如果你当前主要关心论文复现，那么最值得看的主线是下面这一条：

`paper_experiment.py -> IputWhl -> HyRdpPaper / GmfptPaper -> PaperProblem -> PaperSolvers`

### 5.1 单次排序

入口文件：

- `RFDP/Test_RFDP/paper_experiment.py`

入口类：

- `HyRdpPaper`
- `GmfptPaper`

运算逻辑：

1. 加载矩阵。
2. 如果是距离矩阵，先转 affinity，再 KNN 稀疏化。
3. 用 `IputWhl` 构造输入模型。
4. 对每个 query 确定默认子集。
5. 若指定 `local_topk`，把默认子集压缩成一个更小的局部候选池。
6. 调用 `build_paper_problem()` 生成 `PaperProblem`。
7. 调用 solver 解出未标注节点的分数。
8. 将未标注节点按分数排序，拼回 query，形成输出。
9. 若数量不足，从 `full_w` 中补尾部结果。

### 5.2 迭代排序

入口类：

- `IHyRdpPaper`
- `IGmfptPaper`

对应文件：

- `RFDP/MODEL/Paper_RFDP/PaperIterative.py`

运算逻辑：

1. 从初始 query 开始，把它当成当前标注集合。
2. 在当前图上构造 shell 子集，即当前标注点周围的一圈候选。
3. 若图不连通，可根据 `full_w` 添加少量临时连接。
4. 调用基础排序器重新求解。
5. 每轮新增若干个最值得加入的点。
6. 重复直到达到 `nq` 或没有可扩展样本。

当前实现里还有一个很重要的工程性优化：

- 如果设置了 `local_topk`，迭代法会先为每条 query 固定一个局部候选池，后续每一轮迭代都只在这个小池子里做运算，而不是反复在全图上求解。

这点对大规模数据集非常关键。

## 6. 旧版 RFDP 路径的内部运算过程

如果你想理解仓库最早的设计思想，建议按下面路径看：

`RfdpCtofOtcm -> Rfdp -> SmthCstr / FitCstr / SolProc`

### 6.1 主入口

文件：

- `RFDP/MODEL/Frmwk_RFDP/Rfdp.py`
- `RFDP/MODEL/Frmwk_RFDP/RfdpCtofOtcm.py`

其中：

- `Rfdp` 是抽象主体，定义了完整 rerank 流程。
- `RfdpCtofOtcm` 是一个具体实现，表示“把 RFDP 的结果截断为最终输出”。

### 6.2 这条路径怎样工作

`Rfdp.__call__()` 的典型流程是：

1. `ProcAsst.find_idcl(...)`
   合并重复 query，避免重复计算。
2. `_mdl_iput(...)`
   把 query 映射到对应块。
3. `_smth_cstr(...)`
   生成平滑项的 `D` 和 `W`。
4. `_fit_cstr(...)`
   生成拟合项的系数和常数。
5. `_sol_proc(...)`
   求解并返回排序。
6. `_sol2rank(...)`
   从解里取前若干个样本。
7. `_cplt_oput(...)`
   用 `full_w` 把结果补满。

也就是说，这条路径的设计目标是“把很多扩散类方法写成统一接口”，而不是只服务某一篇论文。

## 7. 关键目录与文件索引

下面按目录给出当前最关键的文件说明。

### 7.1 图构建与预处理

- `RFDP/MODEL/Aff_RFDP/Dist2Aff.py`
  把距离矩阵转成 affinity 矩阵。
- `RFDP/MODEL/Aff_RFDP/KnnRstct.py`
  对 affinity 做 KNN 稀疏化。
- `RFDP/MODEL/Aff_RFDP/CnctCpnt.py`
  连通分量辅助逻辑；当前更多作为 `scipy` 不可用时的备用实现。

### 7.2 输入组织层

- `RFDP/MODEL/Iput_RFDP/MdlIput.py`
  输入模型抽象基类；维护 `smth_w / fit_w / full_w`，并负责连通分量、边延迟构建等底层逻辑。
- `RFDP/MODEL/Iput_RFDP/IputWhl.py`
  以“整个连通分量”为求解范围组织 query。
- `RFDP/MODEL/Iput_RFDP/IputLyr.py`
  以“query 周围若干层邻域”为求解范围组织 query。
- `RFDP/MODEL/Iput_RFDP/Qry2Blk.py`
  query 到 block 的映射数据结构。

### 7.3 旧版 RFDP 框架

- `RFDP/MODEL/Frmwk_RFDP/Rfdp.py`
  旧框架总调度器。
- `RFDP/MODEL/Frmwk_RFDP/RfdpCtofOtcm.py`
  一个常用具体实现，负责截断输出长度。
- `RFDP/MODEL/Frmwk_RFDP/RfdpSuplExtn.py`
  旧路径中的扩展型框架代码。

### 7.4 旧路径的约束与求解器

- `RFDP/MODEL/Smth_RFDP/SmthCstr.py`
  平滑项抽象基类。
- `RFDP/MODEL/Smth_RFDP/PairwiseSC.py`
  最常见的 pairwise smoothness 实现。
- `RFDP/MODEL/Fit_RFDP/FitCstr.py`
  拟合项抽象基类。
- `RFDP/MODEL/Fit_RFDP/HyDiffFC.py`
  常用的 hybrid fitting constraint。
- `RFDP/MODEL/Sol_RFDP/SolProc.py`
  求解器抽象基类。
- `RFDP/MODEL/Sol_RFDP/ItrtLinGrw.py`
  线性增长式迭代求解器。
- `RFDP/MODEL/Sol_RFDP/ItrtExpGrw.py`
  指数增长式迭代求解器。

### 7.5 论文实现路径

- `RFDP/MODEL/Paper_RFDP/PaperRanker.py`
  论文单次排序主入口，包含 `HyRdpPaper` 和 `GmfptPaper`。
- `RFDP/MODEL/Paper_RFDP/PaperIterative.py`
  论文迭代排序主入口，包含 `IHyRdpPaper` 和 `IGmfptPaper`。
- `RFDP/MODEL/Paper_RFDP/PaperUtils.py`
  论文路径最核心的工具文件，包含局部子图选择、`L/A/B` 划分、临时连接、相似度补全等关键逻辑。
- `RFDP/MODEL/Paper_RFDP/PaperSolvers.py`
  论文求解器实现。
- `RFDP/MODEL/Paper_RFDP/PaperProblem.py`
  论文局部问题的数据容器。

### 7.6 工具层与公共辅助

- `RFDP/MODEL/Asst_RFDP/OperAsst.py`
  张量/矩阵操作抽象接口。
- `RFDP/MODEL/Asst_RFDP/OperAsstNumpy.py`
  NumPy 后端实现。
- `RFDP/MODEL/Asst_RFDP/OperAsstTorch.py`
  Torch 后端实现。
- `RFDP/MODEL/Asst_RFDP/ProcAsst.py`
  query 去重、索引恢复等流程辅助函数。

### 7.7 实验、评测与演示

- `RFDP/Test_RFDP/paper_demo.py`
  最小论文路径示例，适合理解方法行为。
- `RFDP/Test_RFDP/paper_experiment.py`
  当前最实用的论文路径 CLI 入口。
- `RFDP/Test_RFDP/test_paper.py`
  论文路径单元测试。
- `RFDP/Test_RFDP/rfdp_demo.py`
  旧版 RFDP 路径 demo。
- `RFDP/EVAL/RefRslt.py`
  参考结果封装。
- `RFDP/EVAL/EvalPrcsRcal.py`
  Precision/Recall 评测与绘图辅助。

### 7.8 其他说明文件

- `RFDP/PAPER_IMPL_NOTES.md`
  论文实现补充说明。

## 8. 当前最推荐的阅读顺序

如果你的目标是“先把项目看懂”，建议按下面顺序读：

1. `README.md`
2. `RFDP/Test_RFDP/paper_demo.py`
3. `RFDP/MODEL/Paper_RFDP/PaperRanker.py`
4. `RFDP/MODEL/Paper_RFDP/PaperUtils.py`
5. `RFDP/MODEL/Paper_RFDP/PaperSolvers.py`
6. `RFDP/MODEL/Iput_RFDP/MdlIput.py`
7. `RFDP/Test_RFDP/paper_experiment.py`

如果你的目标是“理解旧版架构”，则建议读：

1. `RFDP/MODEL/Frmwk_RFDP/Rfdp.py`
2. `RFDP/MODEL/Frmwk_RFDP/RfdpCtofOtcm.py`
3. `RFDP/MODEL/Smth_RFDP/`
4. `RFDP/MODEL/Fit_RFDP/`
5. `RFDP/MODEL/Sol_RFDP/`

## 9. 运行入口

### 9.1 最小论文示例

```bash
python3 RFDP/Test_RFDP/paper_demo.py
```

这个脚本使用一个非常小的 synthetic 图，适合快速确认：

- HyRDP / GMFPT 是否能运行
- 单次排序与迭代排序行为是否符合预期

### 9.2 论文路径实验入口

```bash
python3 RFDP/Test_RFDP/paper_experiment.py
```

这个脚本支持三种模式：

- 不传数据路径：跑内置 synthetic case
- 传距离矩阵：脚本内部构图
- 传缓存图：直接加载 `aff_mat / knn_aff`

常用参数：

- `--data-path`
- `--data-key`
- `--input-kind`
- `--label-path`
- `--label-key`
- `--method`
- `--topk`
- `--query-limit`
- `--k-num`
- `--local-topk`
- `--cache-path`
- `--save-json`

### 9.3 单元测试

```bash
python3 -m unittest RFDP.Test_RFDP.test_paper
```

## 10. 大规模数据上的工程注意事项

这部分对实际跑 Market-1501、ReID 风格数据非常重要。

### 10.1 不要把稠密距离矩阵直接当 `smth_w`

正确做法通常是：

1. 距离矩阵先转成 `aff_mat`
2. 再对 `aff_mat` 做 `KnnRstct`
3. 用 `knn_aff` 作为 `smth_w`
4. 用 `aff_mat` 作为 `full_w`

也就是：

- `smth_w = knn_aff`
- `fit_w = knn_aff`
- `full_w = aff_mat`

### 10.2 大数据应优先使用缓存图

`paper_experiment.py` 支持：

- `--cache-path`
- `--cache-aff-key`
- `--cache-knn-key`

这样可以避免每次重复从 `all_dists` 重新计算 `aff_mat` 和 `knn_aff`。

### 10.3 大数据应尽量开启 `local_topk`

对大图而言，如果每次都在整个连通分量上求解，CPU 会很慢。当前实现已经支持：

- `--local-topk N`

它会把每个 query 的求解候选池限制在一个局部子图中，从而显著降低矩阵规模。

### 10.4 当前实现仍主要是 CPU / NumPy 实验路径

论文路径 `Paper_RFDP` 目前主要基于 NumPy 做局部建模与求解，适合方法验证和中等规模实验。对特别大的全量数据，如果不做缓存和局部化，运行时间会明显增加。

## 11. 现阶段应该怎样理解“代码实现完成度”

当前可以比较明确地说：

- 论文提出的核心方法已经完成了代码级实现。
- 仓库中已经存在可运行的论文路径入口、demo 与测试。
- 旧版 RFDP 框架仍然保留，作为另一条并行实现路径。

但也要准确理解边界：

- 当前仓库更接近“方法实现 + 实验入口”，不是一份完整打包好的论文复现实验仓库。
- 数据准备、特征提取、真实 benchmark 的全套复现实验流程，并没有在仓库里被完全封装成一键流水线。

## 12. 运行环境与导入注意事项

这个仓库保留了较多历史代码风格，模块导入大量使用：

```python
from MODEL.xxx import yyy
```

因此实际运行时通常需要：

- 从仓库根目录启动脚本
- 或者像 `paper_demo.py` / `paper_experiment.py` 一样，先把 `RFDP` 根目录加入 `sys.path`

如果是在 Kaggle、Notebook 或远程环境中运行，导入失败时优先检查的不是算法，而是：

- 当前工作目录是否正确
- `sys.path` 是否包含仓库中的 `RFDP` 目录

---

如果你接下来要继续补全项目，最建议先抓住下面这条主线：

`paper_experiment.py -> IputWhl -> PaperRanker / PaperIterative -> PaperUtils -> PaperSolvers`

这是当前仓库里最清晰、也最接近论文语义的一条执行链路。
