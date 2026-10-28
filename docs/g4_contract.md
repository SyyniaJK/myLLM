# G4：统一比较契约

## 方法身份与可替换接口

本次比较 TP-MPPO、无掩码 TP-PPO、合法随机、最强上行信道、批次均衡和 cap 显存贪心。后四者为本平台明确实现的基础对照，不对应原论文中实现细节未公开的 Rewardless 或加权 Heuristic。

所有方法实现 `act(observation, mask, deterministic=True) -> np.ndarray[N]`；外部 factory 的签名为 `factory(config, replicate_seed) -> (Policy, training_steps)`。通过 `--custom name=module:factory --methods ... name` 接入。factory 在每个场景重新构建策略状态；不依赖方法运行顺序。启发式显式返回预算 0；新学习方法按 replicate_seed 选择自己的 checkpoint 并返回实际训练步数，该声明参与同预算比较，不隐式把未知训练成本填零。

本轮六种内置方法共用物理计算、P2/P3 资源分配、显存、时延、失败语义、奖励与能耗，均使用同一 EdgeEnv。没有改变 G1–G3 的物理参数。TP-PPO 仅关闭策略采样和 PPO likelihood 更新中的 mask，二者同时关闭。其他网络、优化器、奖励、预算相同。

新增方法也可通过 `--allocator custom_name=module:factory` 接入现有 BandwidthAllocator 接口（factory(config) 返回 allocator）。此时改变的是该自定义方法的资源分配算法，物理模型和资源约束保持相同；配置记录在 benchmark.json，不能用此选项悄悄改写内置方法的身份。

## 输入与两类负载

保存场景的 format=1，包含完整配置、原始外部 seed、episode seed、上下行信道数组、固定外生请求列表。JSON/NPZ 内容直接读取和核对，不使用摘要校验。

闭环模式使用 G1–G3 原有的每用户随机流和完成后思考机制；不同方法得到的后续到达时刻可能不同。

固定模式每用户每时隙独立采样一次指数偏移，偏移小于时隙长度时产生一个请求；因此提供概率为 1−exp(−τ/mean_think_s)，不是声称“所有 Poisson 到达都被保留”。同一用户的不同请求允许在途重叠，但每时隙至多一个。此模式没有 AP 跨时隙队列；节点忙碌的处理仍由共同 action/环境规则决定。最小模型没有新增共享用户功率预算，发射能耗仍逐传输计。

实际输出只供环境在推理完成后的物理结算使用；策略观测仍只有输入、声明 cap、SLO、当前信道和 idle。外部策略不得通过文件/环境内部属性读取未来信息。

当前两种学习方法在论文闭环机制下训练。fixed 评估是同一冻结策略向外生负载的迁移诊断，分别报告，不描述为已经在 fixed 分布训练。

## 主指标与边界

主 Goodput、Throughput 与能耗都在 [0,H] 统计。窗口末尾未完成结果不预先计入服务量。cohort Goodput、尾部能耗和 pending 另存。

能效是 mean(E_window)/mean(Goodput_window)，同时报告两项原值。零 Goodput 时比值未定义，没有小常数替换。时延仅对窗口内完成输出计算，并记录对应完成请求数；不因时延较低直接判断服务总体更好。

策略决策耗时只计 act()，不含环境推进、带宽分配和硬件 GPU 推理；是并行评估时的 CPU 墙钟诊断量。物理指标、动作和 reward 的精确重放不要求墙钟耗时相等。

## 重复、配对与区间

每个 learned case 有 3 个独立训练 seed，预算均为 65536 步。重用的 TP-MPPO checkpoint 与新训练 TP-PPO 的配置及交互预算逐个核对。

每个策略重复与相同的 10 个测试场景交叉评估。固定模式是严格同请求；闭环模式是相同外生规则与信道的配对。确定性启发式在策略重复维度没有训练随机性，重复行不当作新增独立训练样本。

区间用 crossed/pigeonhole bootstrap：策略重复行与场景列分别有放回抽样。全部方法使用同一批索引，方法差保持配对；不把每个请求当成独立重复。95% 是名义百分位水平，2000 次抽样，固定统计 seed=171。三训练 seed 的结果属于有限样本探索，不提供显著性星号或全家族同时覆盖承诺。

无效/缺失的计划单元不被静默丢弃；完整格点用显式契约核对。完成时延未定义的 episode 不填零，其完成样本数与缺失计数另存。统计始终分 case、分负载模式。

来源：[Owen (2007)](https://arxiv.org/abs/0712.1111) 的交叉随机因子重采样；[Agarwal et al. (2021)](https://arxiv.org/abs/2108.13264) 的强化学习区间与评估协议原则。此处算法实现独立于 rliable，没有引入额外统计依赖。

## 图表契约

Python/matplotlib，使用实际 episodes.json/summary.json，不使用旧历史数组。所有方法、所有已计划格点均纳入；无挑选最好 seed、无平滑删点。

- `performance`：默认场景的两种模式分别展示窗口合规服务量的均值和名义 95% 区间，以及 Goodput–模拟能耗权衡。
- `scaling`：用户/节点数量变化下的窗口 Goodput 比例，方法颜色/标记全图一致。
- `paired`：TP-MPPO 减其他方法的窗口 Goodput 差及配对区间；正值偏向 TP-MPPO，不标显著性。
- `learning_comparison`：相同验证场景上的两个 learned 方法，均值±训练 seed 间 SD。此图不与主测试区间混用。

白底、可编辑 SVG/PDF、600 dpi PNG，标签说明误差范围。图形解释服务数量与能耗的共同变化，不单按最低总能耗宣称优越。
