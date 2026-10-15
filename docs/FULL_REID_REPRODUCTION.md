# 动态 I-HyRDP 与 CSW 的复现入口

2026-10-15 将已运行的服务器实验实现整理回仓库。论文中的 CSW 是 Camera Structure Weighting（相机结构加权），不是特征网络权重，也不是排名融合。`rho` 在模型代码中沿用 `eta` 变量名：同相机边乘以该值，跨相机边保留，候选内部度、边界权重及跨分量临时连接均据此计算。`rho=1` 对应 I-HyRDP，报告配置为 `rho=0.5`。

## 实现与版本来源

- `RFDP/MODEL/Paper_RFDP/PaperFullIterative.py`：实际使用的动态扩散，逐字节恢复；不设置固定候选池，包含跨连通分量桥接及求解残差检查。
- `PaperCamera.py`：相机图先验；移植自服务器 `camera_model.py`，调整导入、格式及输入检查，保留加权计算。
- `PaperPrecision.py`：必要时对同一方程进行迭代精化；不是更换目标函数。
- `RFDP/experiments/full_reid/`：可指定路径的建图、运行、断点续跑、KR/ECN 对照、特征提取和独立验证入口。
- `docs/experiments/source-provenance.json`：原实验源文件 SHA、归档 SHA、移植映射。
- `docs/experiments/duke-20261015-results.csv`：已经完成并验证的真实指标，非本次代码集成的合成测试结果。

CPU 负责建图与扩散，GPU 只用于特征提取。本次没有将扩散改为 GPU 算法，也没有重新调参或重跑全量基准。旧 `Test_RFDP/reid_paper_experiment.py` 保留用于历史静态/局部候选实验；复现当前报告应使用下面的新入口。

## 环境与数据

推荐 Linux Python 3.10/3.11；集成检查在 Python 3.9、NumPy 2.0.2、SciPy 1.13.1 上完成。建图保留全量稠密 affinity，再以 CSR 存储互近邻图，仍需要足够主机内存和磁盘。不要把 CSR 误认为彻底消除了全矩阵内存需求。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r RFDP/experiments/full_reid/requirements-cpu.txt
python RFDP/experiments/full_reid/fetch_controls.py
```

`fetch_controls.py` 仅下载固定 commit 的 KR/ECN 作者源码并核对 SHA。作者来源及版本在 `upstream_sources.json`；源码版权和使用条款仍属于上游。第三方代码不混入本项目原创实现。ECN 是扩展交叉邻域方法，不是 RRF。

特征 MAT 须包含 `query_f / gallery_f / query_label / gallery_label / query_cam / gallery_cam`，行数对应完整测试划分；特征维度必须一致。支持 Market1501 和 Duke 相同格式的导出。身份标签只用于评价，CSW 使用相机元信息。作者 checkpoint 必须与数据集训练划分对应。

## 全量运行及恢复

从仓库根目录执行。将示例数据目录换成实际位置；输出目录不可混用不同特征或协议。

```bash
python RFDP/experiments/full_reid/run_full.py \
  --mat-path /data/features/resnet50.mat \
  --cache-dir /data/cache/resnet50 \
  --output-dir /data/results/resnet50 \
  --depth 500 --rho 0.5 --workers 8

python RFDP/experiments/full_reid/run_full.py \
  --mat-path /data/features/osnet.mat \
  --cache-dir /data/cache/osnet \
  --output-dir /data/results/osnet \
  --depth 500 --rho 0.5 --workers 8
```

默认运行 Original、I-HyRDP、CSW、k-reciprocal、ECN，覆盖 MAT 的全部查询和全部图库，无默认查询截断。`depth=500` 表示动态种子集合最终包含 500 个节点（含查询），不表示只评价 500 张图片；未进入前缀的图库按原始余弦相似度补齐，评价始终使用完整排名。动态候选来自全图两跳邻域。默认 `k=20,gamma=.4,alpha=5,beta=25,step=5`。

断开 SSH 后可用 `screen` 或 `tmux` 保持任务运行。中断后使用**完全相同的命令**恢复：每查询的前缀、排名和轨迹有配置签名与 SHA；变更数据、代码或参数会拒绝沿用旧输出。不得并发写入同一输出目录。损坏/不完整 cache 应改用新目录重建，不能直接视为可复用。

多个进程使用 POSIX fork 共享只读矩阵；单进程使用 `--workers 1`。实际实验均在 Linux 上完成，其他平台的数值库和长双精度实现可能存在差别。排序边界处近似并列需要复核，不能保证不同环境逐位一致。

## 验收和文件

`COMPUTE_COMPLETE.json` 仅表示计算结束。`COMPLETE.json` 的 `status=verified` 才表示独立验证通过，包括完整图库排列、前缀/轨迹一致性、SHA、mAP/mINP/CMC 一致性。

```bash
python RFDP/experiments/full_reid/verify_full.py \
  --mat-path /data/features/resnet50.mat --output-dir /data/results/resnet50
```

保存 `*.rankings.npy`、逐查询 `prefix/trace/diagnostics`、`run-manifest.json`、`verified-results.json` 和特征/权重/源码。图像、权重、特征、全量排名与生成日志不提交 Git；保存在实验归档中。可重建 cache 也不提交。报告中的图表继续使用原先独立复算通过的证据。

## Duke 特征恢复

已经完成的 Duke 实验：2228 查询、17661 图库；作者训练划分的 ResNet50/OSNet-x1.0 checkpoint，256×128 RGB、ImageNet 标准化、无翻转、L2 特征。原 GPU 环境为 PyTorch 2.5.1+cu124（CUDA 12.4）；对应 torchvision 0.20.1。使用 GPU 环境时另装匹配的 PyTorch/torchvision 与 Pillow，不要用 ImageNet 通用权重替代。

`duke-download-manifest.json` 保存数据和权重来源与 SHA；`duke-model-sources.json` 保存已核实的固定作者模型源码链接及 SHA。准备 `DukeMTMC-reID.zip`、`resnet50-duke.pth.tar`、`osnet-duke.pth`、`resnet.py` 后：

```bash
python RFDP/experiments/full_reid/extract_duke.py \
  --assets /data/duke-assets --osnet-source /data/model-source/osnet.py \
  --output-dir /data/duke-extract
```

入口严格校验数据划分、模型源码与权重，导出两份 `features/*.mat`。本次本机无 CUDA，移植后的 GPU 提取入口仅检查语法；其原始服务器版本已经完成实际特征提取并归档。

## 快速回归

```bash
python -m unittest \
  RFDP/Test_RFDP/test_full_ihyrdp.py RFDP/Test_RFDP/test_camera_ihyrdp.py \
  RFDP/Test_RFDP/test_paper.py RFDP/Test_RFDP/test_paper_safety.py \
  RFDP/Test_RFDP/test_paper_ranking_safety.py RFDP/Test_RFDP/test_reid_eval.py
python -m unittest RFDP/Test_RFDP/test_full_pipeline.py
```

覆盖独立稠密方程、跨分量桥接、相机加权等价性、`rho=1` 退化、精化、尾部排列、评价过滤，以及五方法合成输入端到端和恢复保护。合成测试只验证实现，不用作论文指标。
