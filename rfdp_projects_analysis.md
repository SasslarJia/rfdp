# `rfdp` 仓库两个子项目分析报告

## 1. 结论先行

这个仓库里表面上有两个项目:

- `RFDP`
- `ItrtExtnRFDP`

但从代码结构和导入关系看，它们并不是两个彼此独立的产品级项目，而是:

1. `RFDP` 是核心框架，负责在相似图上做基于正则化的扩散式重排。
2. `ItrtExtnRFDP` 是建立在 `RFDP` 之上的一个薄扩展，核心目标是做迭代式查询扩展。

你的判断“看起来像和扩散过程有关的 demo”是**部分正确**的，但要加一个重要限定:

- **是**“图上的扩散过程 / diffusion process / manifold-ranking 风格重排”。
- **不是**“生成式扩散模型”那条路线上的 demo。

更准确地说，这个仓库是一个**面向检索重排与查询扩展的图扩散研究代码**，不是类似 DDPM、Stable Diffusion 那种图像生成扩散模型工程。

---

## 2. 仓库整体结构判断

### 2.1 真实入口不在 README 和 `main.py`

仓库根目录的 `README.md` 还是 GitLab 默认模板，没有提供任何项目说明。  
`RFDP/main.py` 和 `ItrtExtnRFDP/main.py` 也都是 PyCharm 自动生成的占位脚本，没有业务逻辑。

因此，这个仓库的真实信息主要来自:

- `RFDP/Test_RFDP/rfdp_demo.py`
- `ItrtExtnRFDP/Test_ItrtExtn/ItrtExtn_Demo.py`
- `RFDP/MODEL/**`

这本身就是一个很强的信号: 代码更像研究/实验原型，而不是面向发布的工程化项目。

### 2.2 两个目录的关系不是“并列产品”，而是“核心框架 + 扩展实验”

`ItrtExtnRFDP` 目录中并没有自己的 `MODEL`、`EVAL`、`RfdpLog` 等实现，但它的 demo 和扩展类大量直接导入:

- `from MODEL...`
- `from EVAL...`

而这些模块实际上都存在于 `RFDP` 目录下。说明 `ItrtExtnRFDP` 的运行依赖 `RFDP` 的源码路径。

这意味着:

- `RFDP` 是底座。
- `ItrtExtnRFDP` 不是完整重写，而是基于底座做功能追加。

---

## 3. `RFDP` 项目在做什么

## 3.1 项目定位

`RFDP/MODEL/Frmwk_RFDP/Rfdp.py:17-20` 直接给出了名称解释:

> `Regularization Intg_Frmwk of Diffusion Process (RFDP)`

从核心类 `Rfdp` 的实现看，`RFDP` 的输入是:

- 样本间距离矩阵或相似矩阵
- 一组 query
- 若干约束与求解策略

输出是:

- 每个 query 对应的一组重排结果索引

也就是一个**检索重排框架**。

## 3.2 从 demo 看它的最小工作流

`RFDP/Test_RFDP/rfdp_demo.py` 的流程很完整，基本可以视为官方示范:

1. 从 `.mat` 文件读取成对距离矩阵  
   证据: `RFDP/Test_RFDP/rfdp_demo.py:23-47`
2. 用 `Dist2Aff` 把距离转成亲和度/相似度矩阵  
   证据: `RFDP/Test_RFDP/rfdp_demo.py:50-58`
3. 用 `KnnRstct` 把亲和图限制成 KNN 图  
   证据: `RFDP/Test_RFDP/rfdp_demo.py:55-58`
4. 用 `IputWhl` 构造模型输入  
   证据: `RFDP/Test_RFDP/rfdp_demo.py:58`
5. 选择 `RfdpCtofOtcm + HyDiffFC + ItrtExpGrw` 做扩散式重排  
   证据: `RFDP/Test_RFDP/rfdp_demo.py:82-83`
6. 用 `EvalPrcsRcal` 画 Precision-Recall 曲线  
   证据: `RFDP/Test_RFDP/rfdp_demo.py:98-99`

这条链路说明 `RFDP` 不是训练模型，而是**对已有距离矩阵进行图构建、扩散传播和重排求解**。

## 3.3 它处理的不是“原始图像”，而是“样本之间的距离关系”

demo 里加载的是 `.mat` 中的 `all_dists`:

- `RFDP/Test_RFDP/rfdp_demo.py:25-26`

这说明输入并不是原始数据样本本身，而是**预先计算好的 pairwise distance matrix**。  
这非常像传统检索/re-ranking 研究代码的做法:

- 先在外部提特征
- 再计算样本间距离
- 最后在距离图或相似图上做后处理重排

## 3.4 `RFDP` 的核心算法链路

从源码可还原出下面这条主流程:

```text
距离矩阵
  -> Dist2Aff: 距离转亲和度
  -> KnnRstct: 亲和图做 KNN 限制
  -> IputWhl/MdlIput: 构图、找连通分量、分块组织 query
  -> PairwiseSC: 构造平滑项 / 图拉普拉斯相关项
  -> HyDiffFC: 构造拟合项
  -> ItrtExpGrw 或 ItrtLinGrw: 迭代求解
  -> RfdpCtofOtcm 或 RfdpSuplExtn: 输出重排结果
  -> EvalPrcsRcal: 评估 P-R 曲线
```

下面拆开解释。

### 3.4.1 `Dist2Aff`: 距离转亲和度

`RFDP/MODEL/Aff_RFDP/Dist2Aff.py:21-35` 显示:

- 会先取每个样本的局部近邻距离
- 再构造一个自适应的缩放项 `delta_mat`
- 最后通过指数函数把距离映射为相似度

这本质上是**局部尺度自适应的相似度构图**，而不是神经网络。

### 3.4.2 `KnnRstct`: 把全连接相似图裁成 KNN 图

`RFDP/MODEL/Aff_RFDP/KnnRstct.py:24-40` 表明:

- 它先找每列 top-k
- 再生成 KNN 掩码
- 最终返回限制后的图权重矩阵

也就是说，`RFDP` 的传播不是在任意稠密图上做，而是先做邻域约束。

### 3.4.3 `MdlIput` / `IputWhl`: 把 query 映射到连通块上

`RFDP/MODEL/Iput_RFDP/MdlIput.py:40-43` 会:

- 从图里抽边
- 计算 connected components

`RFDP/MODEL/Iput_RFDP/IputWhl.py:10-12` 的类注释写得很直接:

> `Diffusion process on the whole connected component(s)`

`IputWhl._stl_qries` 进一步会把落在相同连通分量集合上的 query 分组:

- `RFDP/MODEL/Iput_RFDP/IputWhl.py:25-49`

这说明 `RFDP` 的求解不是“全数据一次性硬算”，而是**按 query 相关的连通子图分块求解**。

### 3.4.4 `Rfdp`: 主框架在做“带平滑项和拟合项的图传播”

`RFDP/MODEL/Frmwk_RFDP/Rfdp.py:61-88` 是最关键的一段:

- 先根据 query 找到其对应的图子集
- 再构造 smoothness constraint
- 再构造 fitting constraint
- 最后调用 solver 求解
- 之后再把解转为排序结果

其中 `Rfdp.py:76-83` 明确出现:

- `L = D - W` 风格的平滑项
- `coef_cols`, `cnst_cols` 形式的拟合项
- 统一 solver

这就是典型的**图正则化 / diffusion-based ranking**求解范式。

### 3.4.5 `PairwiseSC`: 平滑约束

`RFDP/MODEL/Smth_RFDP/PairwiseSC.py:6-8` 注释为:

> `Smoothness constraint reflects pairwise relationship`

它负责把图结构转换成平滑项输入。  
结合 `SmthCstr` 基类注释:

- `RFDP/MODEL/Smth_RFDP/SmthCstr.py` 把 smoothness 写成和拉普拉斯矩阵相关的形式

因此可以判断它是在编码:

- 图上相邻样本应该有相近响应

### 3.4.6 `HyDiffFC`: 拟合约束

`RFDP/MODEL/Fit_RFDP/HyDiffFC.py:6-8` 注释为:

> `L2 terms with different weights in Hybrid fitting constraint`

其实现中:

- `RFDP/MODEL/Fit_RFDP/HyDiffFC.py:26-34` 根据 query 邻域响应生成系数
- `RFDP/MODEL/Fit_RFDP/HyDiffFC.py:45-49` 生成常数项

这说明 query 对最终传播解不是简单“一票投票”，而是有一套带权拟合项在控制扩散方向。

### 3.4.7 `ItrtSol` / `ItrtExpGrw` / `ItrtLinGrw`: 迭代式求解

`RFDP/MODEL/Sol_RFDP/ItrtSol.py:31-44` 先把问题改写为:

- `Lambda = alpha * Coef + D`
- `Pwr = inv(Lambda) * W`
- `Post = inv(Lambda) * y`

再交给具体迭代器去逼近解。

`RFDP/MODEL/Sol_RFDP/ItrtExpGrw.py:8-12` 写得更明确:

> 指数增长式迭代  
> `X(1) = I + Pwr, Y(1) = Pwr^2; X(t+1) = [Y(t) + I]×X(t), Y(t+1) = Y(t)^2`

这再一次证明这里的“diffusion”是**矩阵传播/图扩散**意义上的 diffusion。

## 3.5 `RFDP` 最终产出的其实是“重排列表”

`RFDP/MODEL/Frmwk_RFDP/RfdpCtofOtcm.py:13-16` 的类注释是:

> `Cutoff the outcome of RFDP as the re-ranking result`

这句话几乎直接给项目定性了:

- 这个框架的最终用途是 `re-ranking`
- 不是分类训练
- 不是生成采样

---

## 4. `ItrtExtnRFDP` 项目在做什么

## 4.1 项目定位

`ItrtExtnRFDP/Itrt_Extn/ItrtQryExtn.py:8-12` 注释非常直接:

> `Iterative Query Extension Based on RFDP`  
> `Re-ranking by iteratively extend each query`

所以这个项目的目标不是另起炉灶，而是在 `RFDP` 的结果基础上继续做:

- 查询扩展
- 迭代补充
- 最终得到更完整的 rerank 结果

## 4.2 它新增的核心代码其实很少

从目录结构看，`ItrtExtnRFDP` 里真正新增的核心模块基本只有:

- `ItrtExtnRFDP/Itrt_Extn/ItrtQryExtn.py`

其余大量依赖仍然来自 `RFDP` 的:

- `MODEL`
- `EVAL`

所以 `ItrtExtnRFDP` 更像:

- 一个算法扩展实验
- 或者一个论文中“在 RFDP 上继续做 query expansion”的后续版本

而不是一套全新框架。

## 4.3 它的工作方式

`ItrtQryExtn.__call__` 的逻辑非常清晰:

- `ItrtExtnRFDP/Itrt_Extn/ItrtQryExtn.py:31-32` 先检查 query 长度是否已经达到目标重排长度
- `ItrtExtnRFDP/Itrt_Extn/ItrtQryExtn.py:38-42` 对当前 query 集合调用一次 `RFDP`
- `ItrtExtnRFDP/Itrt_Extn/ItrtQryExtn.py:44-50` 如果某个 query 的结果还不够长，就把当前输出继续作为下一轮输入
- 持续迭代直到达到 `rerank_num`

这就是标准的**迭代式查询扩展**:

- 先用少量 query seed 启动
- 让 RFDP 找到一些高相关样本
- 把这些样本并回 query
- 再继续扩展

## 4.4 其 demo 也证实了“扩展版 RFDP”的角色

`ItrtExtnRFDP/Test_ItrtExtn/ItrtExtn_Demo.py:96-98` 的核心调用是:

```python
ItrtQryExtn(
    RfdpSuplExtn(...),
    20
)([[1], [3], [5], [6], [7], [8]], 3)
```

这里的语义非常清楚:

- 内核是 `RfdpSuplExtn`
- 外层是 `ItrtQryExtn`
- `20` 是目标重排长度
- `3` 是每轮扩展步长

也就是:

- 先用 `RfdpSuplExtn` 每次补几个样本
- 再通过 `ItrtQryExtn` 做多轮扩展

## 4.5 `RfdpSuplExtn` 本身就是“补 query”的适配层

`RFDP/MODEL/Frmwk_RFDP/RfdpSuplExtn.py:13-16` 的类注释是:

> `Supplement the query with several elements from the outcome of RFDP`

这就把层次关系解释完整了:

- `RfdpCtofOtcm` 是“截断输出，直接拿来做 rerank”
- `RfdpSuplExtn` 是“先拿 RFDP 输出补 query”
- `ItrtQryExtn` 是“多轮重复做 query 补充”

所以 `ItrtExtnRFDP` 并不是第二套 diffusion 框架，而是**RFDP 的 query expansion 版本**。

---

## 5. 这是不是“扩散过程 demo”?

## 5.1 可以说“是”，但要说完整

如果只问“是不是和扩散过程有关”，答案是:

- **是**

因为源码里多处直接写了:

- `Diffusion Process`
- `Diffusion process on the whole connected component(s)`
- `Iterative Query Extension Based on RFDP`

## 5.2 但不是当下常说的“扩散模型 demo”

如果你说的“扩散过程代码 demo”是指类似:

- DDPM
- score-based model
- latent diffusion
- 文生图 / 图生图

那答案是:

- **不是**

理由很明确:

1. 仓库里没有神经网络模型定义，如 U-Net、Transformer diffusion backbone。
2. 没有噪声调度、前向加噪、反向去噪采样流程。
3. 没有训练循环、loss 回传、checkpoint 管理。
4. 输入不是图像张量，而是样本间距离矩阵。
5. 输出不是生成样本，而是 query 对应的排序索引列表。

所以更准确的叫法应该是:

- **图扩散重排框架**
- **基于图扩散的检索后处理代码**
- **带 query expansion 的 diffusion-based re-ranking 研究代码**

---

## 6. 两个项目的关系图

```text
RFDP
  ├─ 提供图构建、约束构造、迭代求解、重排输出
  ├─ 可以直接输出截断后的 rerank 结果
  └─ 也可以输出“补充后的 query”

ItrtExtnRFDP
  ├─ 复用 RFDP 的 MODEL / EVAL
  ├─ 新增 ItrtQryExtn
  └─ 通过多轮调用 RfdpSuplExtn 实现 iterative query expansion
```

一句话概括:

- `RFDP` 是底层扩散重排引擎
- `ItrtExtnRFDP` 是基于它做迭代查询扩展的上层策略

---

## 7. 模块职责拆解

| 模块 | 作用 | 结论 |
| --- | --- | --- |
| `RFDP/MODEL/Aff_RFDP` | 距离转相似图、KNN 图裁剪 | 图构建层 |
| `RFDP/MODEL/Iput_RFDP` | 组织 query、连通块、子图 | 输入编排层 |
| `RFDP/MODEL/Smth_RFDP` | 平滑约束 | 图正则项 |
| `RFDP/MODEL/Fit_RFDP` | 拟合约束 | query 注入项 |
| `RFDP/MODEL/Sol_RFDP` | 迭代求解 | 传播求解层 |
| `RFDP/MODEL/Frmwk_RFDP` | `Rfdp` 主框架及输出策略 | 总控层 |
| `RFDP/EVAL` | Precision-Recall 评估 | 实验评估层 |
| `ItrtExtnRFDP/Itrt_Extn` | 迭代式 query extension | 扩展策略层 |

---

## 8. 数据与评估方式说明

## 8.1 数据格式

demo 默认加载的是外部 `.mat` 文件:

- `IDSC_1000.mat`
- 也留了 `SC_ANIM`、`SC_SWDLEAF` 等替代数据注释

说明项目默认假设:

- 数据集已经被预处理为距离矩阵
- 样本按类别顺序排列

## 8.2 评估方式是检索评估，不是生成评估

`RFDP/EVAL/EvalPrcsRcal.py:6-9` 明确是:

> `Plot Precision-Recall curve`

`RFDP/EVAL/RefRslt.py:29-45` 的 `easy_cstr` 则按:

- 每 `ele_num` 个样本作为一个类别块
- 用 query 所在类别块构造 ground truth

这意味着项目在评估的是:

- 给定 query，排出来的结果里有多少同类样本

这完全符合检索/重排任务，而不是生成任务。

---

## 9. 工程状态与可运行性分析

## 9.1 工程成熟度不高，更像论文/实验代码

主要表现:

1. README 没写项目说明。
2. 两个 `main.py` 都是 IDE 占位文件。
3. 没有 `requirements.txt`、`pyproject.toml`、`setup.py`、`environment.yml`。
4. 数据集不随仓库提供。
5. 代码命名大量使用缩写，如 `Itrt`、`Qry`、`Extn`、`CtofOtcm`、`SmthCstr`，明显偏研究原型风格。

## 9.2 当前环境下无法直接运行完整实验

本地非侵入式检查结果:

- `python3 -m unittest RFDP.Test_RFDP.test_asst RFDP.Test_RFDP.test_xfrm`
- `python3 -m unittest RFDP.Test_RFDP.test_rfdp`

都因为缺少依赖失败，报错包括:

- `ModuleNotFoundError: No module named 'numpy'`
- `ModuleNotFoundError: No module named 'scipy'`

因此可以确定:

- 仓库不是开箱即用状态
- 运行前至少需要自行补装依赖
- 还需要自行准备 `DataSet/*.mat`

## 9.3 `ItrtExtnRFDP` 的 demo 存在一个明显风险点

`ItrtExtnRFDP/Test_ItrtExtn/ItrtExtn_Demo.py:100` 写的是:

```python
EvalPrcsRcal.plot(oput_rslts, RefRslt.easy_cstr(mdl_iput, cls_num, ele_num), 'bx-')
```

但 `RefRslt.easy_cstr` 的签名是:

- `RFDP/EVAL/RefRslt.py:29`

```python
def easy_cstr(qry_lst: list, cls_num: int, ele_num: int) -> list:
```

它显然期望第一个参数是 `qry_lst`，不是 `mdl_iput`。  
从函数内部 `len(qry_lst)` 与 `qry_lst[idx][0]` 的用法看，这里大概率是 demo 里写错了参数。

这进一步说明:

- `ItrtExtnRFDP` 更像实验性扩展代码
- demo 质量没有完全打磨

---

## 10. 我对两个项目的最终定位

### 10.1 `RFDP`

可以把它理解成:

- 一个**基于图扩散的检索重排框架**
- 输入是距离/相似图和 query
- 输出是重排结果
- 支持不同的平滑项、拟合项和迭代解法

它最接近的语境不是“深度生成”，而是:

- manifold ranking
- graph diffusion
- re-ranking
- query-dependent graph propagation

### 10.2 `ItrtExtnRFDP`

可以把它理解成:

- `RFDP` 的**查询扩展版**
- 用 RFDP 的输出结果来补 query
- 再多轮迭代，从而提升最终重排结果完整度

---

## 11. 简明回答你的原始问题

如果把问题压缩成一句话:

> `rfdp` 下的两个项目都是干什么的?

我的结论是:

- `RFDP` 是一个基于图扩散和正则化约束的检索重排框架。
- `ItrtExtnRFDP` 是它的迭代式查询扩展版本。

如果继续压缩成一句更直白的话:

- **这不是“生成式扩散模型 demo”，而是“图扩散式检索重排/查询扩展 demo”。**

---

## 12. 后续如果你要继续深挖，最值得看的文件

按阅读优先级建议:

1. `RFDP/Test_RFDP/rfdp_demo.py`
2. `RFDP/MODEL/Frmwk_RFDP/Rfdp.py`
3. `RFDP/MODEL/Iput_RFDP/MdlIput.py`
4. `RFDP/MODEL/Iput_RFDP/IputWhl.py`
5. `RFDP/MODEL/Fit_RFDP/HyDiffFC.py`
6. `RFDP/MODEL/Sol_RFDP/ItrtSol.py`
7. `ItrtExtnRFDP/Itrt_Extn/ItrtQryExtn.py`
8. `ItrtExtnRFDP/Test_ItrtExtn/ItrtExtn_Demo.py`

如果你愿意，我下一步可以继续帮你做两件事里的任意一种:

1. 把这套 `RFDP` 的数学流程再翻成更容易读的中文伪代码。
2. 帮你梳理一份“如何把这个仓库真正跑起来”的运行说明和依赖清单。
