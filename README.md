# RFDP 实验代码

`test` 分支仅管理实验实现、测试、依赖清单和配置模板。推导、论文、实验协议、结果、运行记录和数据保存在仓库外的本地研究目录或服务器；不提交到此分支。

| 目录 | 用途 |
|---|---|
| `RFDP/MODEL/` | 原 HyRDP / I-HyRDP、图构建与数值求解 |
| `RFDP/EVAL/` | 统一 ReID 评价 |
| `RFDP/experiments/full_reid/` | 特征缓存、基线及固定对比方法入口 |
| `RFDP/experiments/reliability/` | CSW 实现与相关工具 |
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
PYTHONPATH=RFDP python -m unittest discover -s RFDP/Test_RFDP -p 'test_stationary_measure.py'
```

特征提取与 GPU 对比入口还需对应前端的 PyTorch 等依赖；运行环境应与具体配置匹配。

## 运行配置

将 `configs/stationary-measure.example.json` 复制到仓库外，填写实际输入路径、输入 SHA256 和执行提交。正式入口为：

```bash
PYTHONPATH=RFDP python -m experiments.continuous_fitting.launch --config /path/to/run-config.json
```

运行配置必须绑定干净的执行提交；历史结果使用其原冻结版本。输出目录、特征缓存与原始证据均放在仓库外。

CSW组合使用 `configs/csw-stationary.example.json` 和同一启动入口。`relation_model="csw"` 复用已有相机/共有邻域支持实现；`source="stationary_measure"` 为组合，`source="original"` 为CSW单项。支持矩阵通过 `RFDP/experiments/reliability/prepare_support.py` 从冻结缓存生成，绑定原亲和矩阵与相机顺序。`comparisons` 可列出多个同特征完整基线。

`trace_retention="sampled_after_readback"` 对每个查询的完整轨迹逐分数读回，再只保留配置中预先固定的32个查询轨迹。所有查询均计算和验收，`step`、`depth`、候选构造、完整图库排名及数值阈值不变。保留完整排名、全部逐查询统计、前缀和验收摘要；失败轨迹保留。原有 `trace_retention="all"` 模式仍保存全体轨迹。不同留存方式下实际分数、前沿、桥接、前缀和最终排名的一致性由 `test_csw_stationary.py` 检查。

对比方法采用原论文推荐参数，不针对当前输入额外搜索；超参调优集中于自身方法。各方法共享冻结特征和评价协议，并保留各自规定的输入处理。

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
