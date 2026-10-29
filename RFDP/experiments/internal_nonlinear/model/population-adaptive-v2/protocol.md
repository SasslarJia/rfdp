# 原目标的自适应精度续跑

用户于2026-10-29要求落实保持效果的自适应精度。科学模型、原图、theta=.5、阶段共享的精确原解RMS delta、alpha/beta、step/depth均保持。计算精度依次24/40/80/120十进制位；不足时提高精度，绝不放宽新目标残差1e-8、正性和实际前五严格排序。原解delta区间仍向外包围；所有修正只是候选，只有最终区间证明可以接纳。低精度原解修正停止目标仅用于生成候选，不充当验收条件。真并列仍须精确交换对称证明。

复用候选产生的浮点基线模型，不信任其delta作为真实目标。缓存每个精度下的向外常量，减少重复转换。读回保持原逐行计算、原前沿/边界/桥接/完整排名重建；精度依次24/40/100/160/240，不改动保存分数。原100位读回另核已有真实轨迹的3个阶段，未重跑查询或计算AP。

原population-v1源码与锁保持。用户主动切换前保留完成查询及逐文件SHA、原退出和进度，未完成片段移入同run的execution-history。新source为population-adaptive-v2，OUT仍是market-internal-nonlinear-theta0p5-v1，继承原run-manifest签名并另记execution-adaptive-v2.json和新源码锁。保留原签名表示同一科学运行；它不是新执行源码锁。仅跳过经SHA核对的完成查询，不补尾、不根据部分AP选择。最终验收覆盖新旧全部查询。

16workers×BLAS1、每worker4GiB AS、40GiB总输出预算、RAM70%/GPU75%/每盘空闲max25%和10GiB保持。GPU旧模块针对ProjectedState.linear，当前Newton调用SciPy sparse_solver，不存在可直接启用的同接口开关。本修订仍是CPU数值后端，未声称GPU启用。
