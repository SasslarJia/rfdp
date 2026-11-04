# RFDP 实验代码

`test` 分支仅管理实验实现、测试、依赖清单和配置模板。推导、论文、实验协议、结果、运行记录和数据保存在仓库外的本地研究目录或服务器；不提交到此分支。

| 目录 | 用途 |
|---|---|
| `RFDP/MODEL/` | 原 HyRDP / I-HyRDP、图构建与数值求解 |
| `RFDP/EVAL/` | 统一 ReID 评价 |
| `RFDP/experiments/full_reid/` | 特征缓存、基线及固定对比方法入口 |
| `RFDP/experiments/reliability/` | CSW 实现与相关工具 |
| `RFDP/experiments/seed_support/` | 原有轨迹的只读来源支持与首批诊断，不产生新检索结果 |
| `RFDP/experiments/continuous_fitting/` | 多种二次目标共用的求解、执行、读回和诊断代码 |
| `RFDP/experiments/stationary_measure/` | 度加权方法的辅助工具 |
| `RFDP/experiments/internal_nonlinear/` | 非线性实现与共用配对分析模块 |
| `RFDP/Test_RFDP/` | 正确性测试 |
| `configs/` | 不包含实际运行产物的配置模板 |

目录名保留历史兼容性；共用模块不能按旧方法名称删除。

## 安装与测试

在独立 Python 环境中安装核心 CPU 依赖：

```bash
python -m pip install -r RFDP/experiments/full_reid/requirements-cpu.txt
python -m pip install mpmath==1.3.0
PYTHONPATH=RFDP python -m unittest discover -s RFDP/Test_RFDP -p 'test_camera_admission.py'
```

特征提取与 GPU 对比入口还需对应前端的 PyTorch 等依赖；运行环境应与具体配置匹配。

## 运行配置

将 `configs/stationary-measure.example.json` 复制到仓库外，填写实际输入路径、输入 SHA256 和执行提交。正式入口为：

```bash
PYTHONPATH=RFDP python -m experiments.continuous_fitting.launch --config /path/to/run-config.json
```

运行配置必须绑定干净的执行提交；历史结果使用其原冻结版本。输出目录、特征缓存与原始证据均放在仓库外。

CSW组合使用 `configs/csw-stationary.example.json` 和同一启动入口。`relation_model="csw"` 复用已有相机/共有邻域支持实现；`source="stationary_measure"` 为组合，`source="original"` 为CSW单项。支持矩阵通过 `RFDP/experiments/reliability/prepare_support.py` 从冻结缓存生成，绑定原亲和矩阵与相机顺序。`comparisons` 可列出多个同特征完整基线。

配置中的正值 `alpha`、`beta` 同时用于生产求解与保存分数的回读验收；完整标准query/gallery、step=5和depth=500约束不变。`test_csw_stationary.py` 覆盖非默认参数在原图和CSW图上的完整轨迹回读，并确认错误参数会被拒绝。

可选种子接纳使用 `configs/camera-admission.example.json` 和同一 `launch --config` 入口；模板不会自动执行。默认 `admission_policy="original"` 保留原行为；`camera_gate` 审核原提名的旧种子相机支持，`camera_gate_prefix` 使用同状态门槛得到的数量接纳原排名前缀。两者限原目标、原关系和5/25参数；相机文件为按query后gallery排列的一维整数 `.npy`，其SHA与策略版本写入运行绑定。`test_camera_admission.py` 只用小图检验接纳、桥接与数值回读。

接纳规则可能增加外层次数和轨迹量；正式运行前应按实际输入与留存方式填写资源预算，模板数值不是容量测量。达到资源线时中止，不通过缩减查询、前沿、depth或数值精度继续。

已有原轨迹的诊断使用 `PYTHONPATH=RFDP python -m experiments.seed_support.audit_original_support --index /path/to/input-archive-index.json --evidence /path/to/original-state-evidence.json --work /path/to/temporary-analysis`。脚本逐成员核对归档SHA，只计算原状态的支持统计，不调用扩散求解器。完成后可用 `PYTHONPATH=RFDP python -m experiments.seed_support.audit_initial_batch --work /path/to/temporary-analysis --output /path/to/initial-batch-summary.json` 检查原首批；输出及输入索引都保存在仓库外。

`experiments.stationary_measure.analyze_saved_rankings` 接受仓库外的分析配置和输出目录，复算既有完整排名中的逐查询AP／Rank-1，分解正例位置变化。其可选前沿诊断在已保存SC前沿上计算原目标的反事实选择；`--frontier-stages` 只限制诊断窗口，不改变任何检索运行或正式指标。缺少轨迹、窗口外差异及反事实范围在输出中明确记录。

`trace_retention="sampled_after_readback"` 对每个查询的完整轨迹逐分数读回，再只保留配置中预先固定的32个查询轨迹。所有查询均计算和验收，`step`、`depth`、候选构造、完整图库排名及数值阈值不变。保留完整排名、全部逐查询统计、前缀和验收摘要；失败轨迹保留。原有 `trace_retention="all"` 模式仍保存全体轨迹。不同留存方式下实际分数、前沿、桥接、前缀和最终排名的一致性由 `test_csw_stationary.py` 检查。

对比方法采用一套合理固定配置（原论文推荐或官方代码默认），已有正确全量结果直接复用，不因两者差异自动补跑；调优集中于自身方法。各方法共享冻结特征和评价协议，并保留各自规定的输入处理。

## 外部部署材料

非线性历史部署入口读取外部清单和协议：

```bash
python RFDP/experiments/internal_nonlinear/prepare_deployment.py \
  --materials-root /path/to/research/materials/rfdp \
  --runtime-root /path/to/new/runtime \
  --repo-root /path/to/frozen-code-checkout \
  --cache-dir /path/to/feature-graph-cache \
  --mat-file /path/to/features.mat \
  --python /path/to/python
```

`--materials-root` 内沿用 `RFDP/experiments/internal_nonlinear/model/` 相对结构。导出器保留冻结源码核对，只生成部署目录，不启动实验。外部方法的源码版本/下载校验清单属于可执行依赖，随代码维护。
