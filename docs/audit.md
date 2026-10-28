# 原 ZIP 审计与修复映射

原始 ZIP 和解压目录均留在项目父目录。核对发现六个核心文件在 ZIP 内与解压后逐字节一致：LLM_main.py、environment_llm.py、llm_algorithm.py、ppo_discrete.py、replaybuffer_discrete.py、normalization_discrete.py。

新实现位于 EdgeInferenceLab，原件不覆盖。没有创建文件校验摘要；上述对照直接比较文件内容。

| 原位置/问题 | 修复位置 | 验证 |
|---|---|---|
| 14B 权重 28 GB，节点 16/20/24 GB | config.py 默认采用论文 7B/24 GB | test_config_7b_fits_and_old_14b_does_not |
| request_arrival 重复消费历史到达 | workload.py 消费后将 next_arrival 设为无待到达；完成后再启动 thinking | 无重复、边界到达测试 |
| KV 未包含输出 | physics.py `4hLb(s+o)` | 增长差值、手工数值测试 |
| 同批按各自输出计算不同推理时间 | physics.py 批次最大输入、最大输出 | 同批 GPU 时延一致测试 |
| 原时延采用 2N/HBM 近似 | physics.py 实现论文明示 FLOPs 式 | 手工计算验证；明确不称其为实测 |
| 上下行共用信道和噪声 | physics.py 两套衰落 RNG 与噪声配置 | 上下行不同、同 seed 重现 |
| 用户/节点数、维数写死 | Config + EdgeEnv 动态推导 | 1×1、10×2、20×6 场景 |
| 节点提前释放/显存状态不一致风险 | environment.py 使用绝对完成时间 | 多时隙忙碌与释放边界测试 |
| 零下行带宽仍提前计入 Throughput | environment.py 按真正反馈统计 | 丢弃、超时反馈及 inference_completed 分列 |
| 用户无输出时没有清楚的结束语义 | model_contract.md 的失败通知规则 | 不产生无限忙碌状态 |
| 环境构造早于种子设置 | reset(seed)、每用户/信道独立 RNG | 完整轨迹逐字段比较 |
| Normalization 初始 std=x | ppo.py Welford，零方差单位尺度 | 零方差测试 |
| rollout/训练/evaluation 耦合 | ppo.py、experiment.py、environment.py | 训练更新、checkpoint、独立 seed 测试 |
| 评估只保留 reward/ratio | experiment.py 保存配置、原始曲线、测试请求、批次、功率区间及模型 | 完整 suite 输出 |

## 保留的算法核心

两层 128 单元 Tanh Actor/Critic、正交初始化、Actor 输出增益 0.01、独立 Adam、分类动作、每用户 mask、联合 log probability 求和、GAE、clipped PPO、熵奖励与梯度裁剪均来自导师核心方案。

测试目录保留独立的教师 Actor 参考快照，用固定输入、mask 和种子核对重建策略的初始概率，测试不依赖父目录文件才能运行。

actor 仍只学习卸载；上行 P2 和下行 P3 属于 TP-MPPO 的第二阶段。不是论文中“所有变量都由 MPPO 学习”的对照方法。

## 必须披露的差异

1. 新增独立下行信道使观测维度从原 ZIP 的 96 变为默认 138；不兼容旧权重，这是状态定义对齐带来的必要变化。
2. 采用正确的 Gymnasium time-limit truncation：价值 bootstrap，但 GAE 不穿过 reset。
3. 原文未给出所有数值、未明确失败通知及原始训练超参数，因此不承诺曲线逐点相同。
4. 本次 G1–G3 不伪造论文的 MPPO、TP-PPO、Rewardless、Heuristic 结果；目录中的旧 npy/CSV 未作为新结果使用。

## 参考仓库的使用范围

Gymnasium 用于接口；SB3-Contrib 用于核对多离散分类、联合概率和 mask 评估语义；InferSim 用于理解理想计算量与性能画像校准的区别。本版没有把 InferSim/Vidur 的 GPU 模型混入论文兼容配置。

固定参考版本、许可证及模块见 `references.md`。
