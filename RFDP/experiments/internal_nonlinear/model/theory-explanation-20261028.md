# I-HyRDP 内部非线性平滑：推导、实现与实验口径

2026-10-29。此文解释正式实验使用的固定目标；运行源码、源锁和参数保持不变。

## 1. 从原目标出发

在一次外层扩种子步骤内，U 是当前候选集，W 是其原局部对称非负权重矩阵。令 h_i=Σ_{j∈U}w_ij（保留原自环），e_i 为原种子边界权重（含原算法需要的临时桥接）；H=diag(h)，b_i=α（e_i>0）或β（e_i=0）。原归一化线性系统写为

\[
\left[I-H^{-1/2}WH^{-1/2}+
\operatorname{diag}\left(\frac{\alpha e_i}{h_i+e_i}\right)\right]\psi=b.
\]

取 z=H^{-1/2}ψ、d_i=αh_ie_i/(h_i+e_i)、c_i=√h_i b_i，即得 A_0z=c，其中 A_0=L(W)+diag(d)，L(W)=diag(h)−W。自环在拉普拉斯中抵消，但仍进入原 h。等价能量（省略与 z 无关的常数）为

\[
E_0(z)=\frac12\sum_{\{i,j\}\in E}w_{ij}(z_i-z_j)^2
+\frac12\sum_i d_i z_i^2-c^\top z.
\]

每条无向非自环边只计一次。这里没有把原间接拟合改成新的二次锚项。

## 2. 只替换平滑势函数

先求该阶段原解 z_0=A_0^{-1}c，再定义

\[
\delta^2=\frac{\sum_{\{i,j\}\in E}w_{ij}(z_{0i}-z_{0j})^2}
{\sum_{\{i,j\}\in E}w_{ij}}.
\]

δ 在本次非线性求解中固定；无边或精确 δ=0 时用原二次分支。它是整个阶段共享的尺度，因此不同非平凡分量可以通过 δ 耦合。它不由身份、相机标签或 AP 选择。

\[
\phi_{\theta,\delta}(t)=\frac{1-\theta}{2}t^2+
\theta\delta^2\left(\sqrt{1+(t/\delta)^2}-1\right),
\quad
E_\theta(z)=\sum_Ew_{ij}\phi_{\theta,\delta}(z_i-z_j)
+\frac12\sum_i d_i z_i^2-c^\top z.
\]

固定候选 θ=0.5。

对 t 求导：

\[
\phi'(t)=a(t)t,\quad
a(t)=(1-\theta)+\frac{\theta}{\sqrt{1+(t/\delta)^2}},
\qquad
\phi''(t)=(1-\theta)+\frac{\theta}{(1+(t/\delta)^2)^{3/2}}.
\]

故一阶条件是

\[
\sum_j w_{ij}a(z_i-z_j)(z_i-z_j)+d_i z_i=c_i.
\]

传播通量依赖当前待求解分数差；原 W、h、e、拟合项不随内层迭代被重定义。θ=0.5 时 a 在 (0.5,1]，小差异接近原平滑，大差异相对减弱，边不删除。这里仍可等价解释为状态相关有效传导权重；它与扩散前一次性校准固定 W 的位置不同，不能据此预设组合互补。

## 3. 数学性质与证明条件

Hessian 为 L(wφ''(t))+diag(d)。0≤θ<1 且 d≥0 时，

\[
\nabla^2E_\theta(z)\succeq(1-\theta)L(W)+\operatorname{diag}(d)
\succeq(1-\theta)A_0.
\]

每个连通分量至少有一个 e_i>0 的锚，且 h_i>0、α>0，保证 A_0 正定：若 x^TA_0x=0，各分量上 x 为常数，锚项迫使该常数为0。因此新能量强凸、有下界且唯一极小点。未锚定输入被拒绝。

若某个极小分数 z_i≤0，则在全局最小节点上 Σ_j w_ij a_ij(z_i−z_j)≤0 且 d_i z_i≤0，与 c_i>0 矛盾，故解严格正。θ=0 时一阶条件恢复 A_0z=c，实际代码直接委托原准确求解链。

保留二次部分是模型需要：原间接拟合含负线性项；纯线性尾部鲁棒平滑可能不足以限制它。例如 sqrt(1+t²)−1−25t 随 t→+∞ 无下界。混合目标的二次部分不等于数值求解器额外加 ridge。

这些结论证明目标和求解的性质，不证明 mAP 提升。大分数差也可能来自同身份跨相机变化；是否有益需全量实验。整体平滑减弱与选择性传播的贡献，也需后续控制才能作机制归因。

## 4. 对应实现与认证

- sparse-certificate-v1/sparse_solver.py：稀疏目标、梯度、Hessian、Newton/Armijo；候选求解相对梯度目标1e−11、最多100次Newton、最多60次回溯。
- scalable-certificate-v1/scalable_certificate.py：精确二进制 W/e 对应的有理系数，原解及 δ 区间，稀疏LU与MP残差修正，80/120位向外认证；不走大规模稠密MP。
- runner-interface-v1/internal_adapter.py：仅替换进程内 solve_hybrid_sparse。保持 FullIHyRDP.rank；θ=0返回原求解对象，θ=.5返回认证后的 ψ=√h z。
- retrieval-qualification-v1/run_fixed.py、verify_artifact.py、accept_fixed.py：保存实际MP分数与全部轨迹，再从原图和保存分数重建前沿、边界、选点、桥接和全排名；不调用生产求解器修改存储分数。

利用锚生成树给 μ>0 的保守下界，以 ||∇E||/μ 包围解误差，换回 ψ 后严格认证实际前五。非并列必须区间分离；真并列仅接受精确交换对称与唯一性的证明。原式归一化残差上界≤1e−8；不能用旧线性残差验收新非线性目标。独立读回与生产共享部分数学包围工具，未声称证明实现完全独立。

必要数学与正确性检查已完成；正式全量已启动，精度尚待完整结果。

## 5. 当前正式实验

2026-10-29 01:05:22已启动Market正式全量：3368查询/19732图库，原图k20/gamma.4、alpha5/beta25、step5/depth500，16workers×BLAS1。source为internal-nonlinear-smoothing-v1-20261028/population-v1，run为market-internal-nonlinear-theta0p5-v1。进程和阶段状态只读heartbeat-latest.json及现场日志，不把历史PID当活动。

完整链为compute→原IHy配对分析→全部实际分数及轨迹读回验收。query/identity各10000次bootstrap、seed260626；mAP点提升与稳定增益分别报告，同时报告Rank-1。实际返回分数的新目标残差上界≤1e-8且严格认证前五，真并列仅凭精确对称证明。禁止ridge、剪桥、数值补尾、放宽阈值或改深度；8192局部上限是停止条件。独立读回共享部分数学包围工具，不声称证明实现完全独立。

当前目标16workers×BLAS1，已启动；不再追加性能资格或重跑已完成固定查询。资源和进度见运行记录与heartbeat-latest.json。

## 6. 文献依据与项目适配

已核对PMLR官方页：
1. Yang et al., Graph Neural Networks Inspired by Classical Iterative Algorithms, ICML 2021, 139:11773–11783. https://proceedings.mlr.press/v139/yang21g.html 。依据是能量驱动的非线性图平滑及IRLS式状态相关传播。
2. Liu et al., Elastic Graph Neural Networks, ICML 2021, 139:6837–6849. https://proceedings.mlr.press/v139/liu21k.html 。依据是结合二次与保边界平滑，增强局部适应性。
3. Fu, Zhao, Bian, p-Laplacian Based Graph Neural Networks, ICML 2022, 162:6878–6917. https://proceedings.mlr.press/v162/fu22e.html 。依据是由非二次图正则化得到非线性传播。

本项目的Charbonnier混合、保留I-HyRDP混合拟合、阶段δ规则、θ=.5选择是本项目适配；三篇论文未提出这一完整公式，也未证明它改善ReID。没有把作者GNN训练或硕士论文成对矩阵模型作为已复现结果。更近的HiD-Net（AAAI2024）、GREAD（ICML2023）等审读范围另见sources.json，未作为已实施组件。

