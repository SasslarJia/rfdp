# 代码与实验数据索引

整理日期：2026-10-30。当前只进行本地资料与理论整理。定时任务已由用户关闭（本地配置为PAUSED），服务器由用户告知暂时关闭。本次未访问服务器或运行实验。

## 阅读入口

- [当前研究总结：问题、贡献、稳定条件与后续验证](../RFDP/experiments/stationary_measure/research-summary.md)
- [当前模型与实现](../RFDP/experiments/stationary_measure/README.md)
- [原完整ReID复现入口](FULL_REID_REPRODUCTION.md)
- [既有CSW实现](../RFDP/experiments/reliability/README.md)

## 代码职责与保留依据

| 入口 | 职责及状态 |
|---|---|
| `RFDP/MODEL/Paper_RFDP/PaperFullIterative.py` | 原动态I-HyRDP的前沿、扩种子、桥接与排名逻辑 |
| `RFDP/experiments/continuous_fitting/model.py`、`certified.py` | 当前目标装配和实际分数证书；该目录有活跃共享代码 |
| `RFDP/experiments/continuous_fitting/full.py`、`verify.py` | 正式查询执行，保存分数及全流程读回 |
| `RFDP/experiments/continuous_fitting/launch.py` | compute→analysis→acceptance；analysis实际调用下行模块 |
| `RFDP/experiments/internal_nonlinear/runtime/analyze_route.py` | 当前仍使用的原IHy配对分析 |
| `RFDP/experiments/stationary_measure/` | 当前推导、完整结果、复盘和固定协议 |
| `RFDP/experiments/full_reid/`、`RFDP/EVAL/` | 特征/完整排序与标准ReID评价 |
| `RFDP/experiments/reliability/` | 保留的CSW与相关有效实现 |
| `RFDP/Test_RFDP/` | 正确性检查，不能替代正式结果；本次未执行 |

本轮没有移动共享模块或删除冻结源。目录名中的历史模型名不等于代码已经闲置。后续如统一包名，应作为单独重构提交处理，不能用最新HEAD替代旧结果的执行提交。

## 主比较矩阵：已有28个单元

下表由已有验收主表转录，mAP / Rank-1均为百分比；没有重新评价排名。Market查询/图库为3368/19732，Duke为2228/17661。原始距离、自身基线和五种外部方法分开识别。

| 方法 | Market OSNet | Duke OSNet | Market TransReID | Duke TransReID |
|---|---:|---:|---:|---:|
| raw-cosine | 83.565969 / 94.239905 | 72.686019 / 87.028725 | 88.989069 / 95.219715 | 82.110766 / 90.709156 |
| k-reciprocal | 92.454390 / 95.397862 | 85.894921 / 90.260323 | 94.056011 / 95.605701 | 91.327403 / 92.684022 |
| ecn-rankdist | 92.647124 / 95.397862 | 86.528042 / 90.529623 | 94.046875 / 95.843230 | 91.475870 / 93.132855 |
| caj | 92.565170 / 95.694774 | 85.040133 / 90.754039 | 94.127063 / 95.724466 | 90.906476 / 93.447038 |
| cheb-gr | 90.074025 / 94.032067 | 83.127026 / 88.285458 | 91.993598 / 95.516627 | 87.504278 / 92.010772 |
| mr-fixed-kernel | 92.751630 / 95.783848 | 86.088396 / 90.933573 | 94.418075 / 96.199525 | 90.402566 / 92.684022 |
| I-HyRDP | 93.134902 / 95.160333 | 87.264883 / 89.991023 | 94.530890 / 95.279097 | 91.446321 / 92.414722 |

来源：research目录下 `experiments/graph-routes-20261026/four-cell-main-comparison-20261027.json`，SHA256 `865162d1a1020408964278f1ac0f4eccee44e2c813ae0084f61b7e922d1125f9`。原表未改。固定比较口径：CAJ采用CLI3/3；Cheb-GR采用lambda6参考配置；MR采用研究固定核及alpha=.99。这些是同输入的固定配置比较，不代表各方法搜索后的最优结果，也不能单用此精度表比较速度。

## 当前改进新增的两个单元

| 输入 | mAP / Rank-1（%） | 相对同特征原IHy（pp） | 状态 |
|---|---:|---:|---|
| Market OSNet | 93.174274 / 95.249406 | +0.039372 / +0.089074 | 四阶段exit0，完整读回通过 |
| Market TransReID | 94.537030 / 95.190024 | +0.006139 / -0.089074 | 四阶段exit0，完整读回通过 |
| Duke OSNet | — | — | 当前改进未运行，用户延后 |
| Duke TransReID | — | — | 当前改进未运行，用户延后 |

两组均k20/gamma.4、alpha5/beta25、step5/depth500、16workers×BLAS1；无新增组件超参。所有query/identity配对95%区间均跨零。Market两种表征不能计作两个独立数据集。新结果不覆盖冻结主表，应作为新方法独立一行和对应证据使用。

### 当前结果与原始证据位置

| 输入 | Git完整统计 | 正式执行冻结 | S1原始run |
|---|---|---|---|
| OSNet | [result.json](../RFDP/experiments/stationary_measure/result.json) | `3ddf0520db18852401c636620147bd837754db0f` | `/root/autodl-tmp/rfdp-graph-routes-20261026/market-stationary-measure-fitting-v1` |
| TransReID | [transfer-result.json](../RFDP/experiments/stationary_measure/transfer-result.json) | `72d42e24ba15ea38094af4130a271419c1198a75` | `/root/autodl-tmp/rfdp-graph-routes-20261026/market-transreid-stationary-measure-v1` |

两份JSON包含各自科学签名、输入SHA、全排名SHA、FULL_ACCEPTANCE SHA、全部退出码、耗时、资源与配对区间。以下为本次本地读取的文件SHA，便于与该索引对照；不是新服务器读回：

- `result.json`：`3811f4ab78cd4af232de6e65e585f4ddfc821faf9d30626a8456b229439844f9`
- `transfer-result.json`：`5cbb11ebbcdb39a242a9b07977c7ca2e5726751d0d675167a6215d28ca413be7`
- `diagnosis.json`：`48fa4ae2e4006b84da3a637e3e6137437390dbaa4b14297b6bf7f48fea263152`
- `transfer-diagnosis.json`：`a49efb5c9f066a89ba07195daeb8d85a7beb60ad911b96051e200666e2706b23`

OSNet有界诊断执行冻结 `090a046266c5fe32d94f56c5db3e4d6ebcd98c12`；TransReID诊断冻结 `64c0b7edae42aece0695e2895f3784554cd5e41a`。两者均为全查询、最多前5个共同批次、首次有序分歧停止，不能解释成原完整轨迹全部重建。原始逐查询结果保存在各S1诊断目录，摘要在以上Git JSON。

### 其他已有材料，不与当前两格混用

- 原IHy另有Market/Duke × ResNet50历史结果。ResNet50不是当前两主前端矩阵中的TransReID。
- CSW+IHy及早期反馈的Market/Duke × ResNet50/OSNet已有独立记录。它们是不同组件结果，不能当作CSW+当前度加权组件的组合结果；该组合四格尚未做。
- 旧四输入比较与中期表保存在research的 `experiments/recovery-20261024/results/comparison-four-inputs.json`、`writing/report-revision-20261024/source/verified-results.json`；精确来源按 `experiments/ihyrdp-result-registry.json` 查询。冻结主表和中期Word保持不变。
- 连续源项、边界权重、两步重置固定形式已停止；只查对应 `result.json` 和方法集停止标记，不再重新排队。它们已清理的逐批分数轨迹不能声称仍可从剩余文件重新做全分数验收。
- 非线性固定形式因计算代价停在626/3368，无完整精度结论。执行源 `5e5ec194ef362c9bf179d73f8bd6ea0a10d71601` 与已完成产物保留；服务器关机后不能把旧SIGSTOP记录当作当前可继续的进程。

## 数据管理边界

Git管理代码、推导、协议、精简结果及此导航；大特征、排名、实际分数和运行原件保持在既有服务器/归档位置。当前两组完整度加权证据保留。本次没有重新校验服务器文件或进行删除/迁移，不将服务器关机写成数据已归档或数据丢失。恢复工作时从已有摘要中的输入/执行/验收参照核对，避免重跑已完成单元。
