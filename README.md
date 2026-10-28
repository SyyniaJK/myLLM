# EdgeInferenceLab

无线边缘 LLM 推理科研基础平台。首个基线为 **TP-MPPO**；环境、物理公式、随机过程、指标与能耗观察器独立于策略。

已实现 G1 系统修复、G2 真实训练/独立评估、G3 被动能耗核算、G4 统一算法比较及 G5 研究问题诊断。原 ZIP 与解压目录保留在父目录，未覆盖。没有加入新研究算法。

当前版本 **0.2.0**。仓库：[SyyniaJK/myLLM](https://github.com/SyyniaJK/myLLM)。新聊天可先阅读 [项目交接说明](CHAT_CONTEXT.md)；提交、版本标签和讨论 ZIP 的生成方法见 [版本管理说明](docs/versioning.md)。

## 从哪里看

- [模型和编程契约](docs/model_contract.md)：所有公式、单位、信息可见性、失败和统计边界。
- [原代码审计](docs/audit.md)：论文—代码—修复—测试映射。
- [参考仓库记录](docs/references.md)：固定参考版本、许可证与采用范围。
- [本次正式实验报告](results/g123_final/report.md)、[汇总原始数据](results/g123_final/summary.csv)。
- [G1 测试结果](results/g1_tests.txt)。
- [G4 对比契约](docs/g4_contract.md)、[G4 实验报告](results/g4/benchmark/report.md)、[G4 完成与验证记录](docs/g4_verification.md)、[G4 测试结果](results/g4_tests.txt)。
- [G5 诊断契约](docs/g5_contract.md)、[G5 研究问题诊断结果](results/g5/report.md)。

## 安装

Python 3.11+，以下命令在本目录的 PowerShell 中执行。

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install -e . --no-deps
```

当前交付目录已有可用的 `.venv`。训练只需要 CPU；模拟 GPU 的参数不要求本机安装相应 GPU。

从 GitHub 或讨论 ZIP 获取的副本不包含 `.venv`，按上面的命令重新安装即可。正式策略、场景、评估表和动作轨迹随版本保存；逐请求、批次与功率区间的完整明细由评估/重放命令重新生成。

## 测试

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
.venv/Scripts/python.exe -m pytest -q
```

关闭 pytest 的第三方插件自动加载，是为了使测试只使用本项目的 pytest 功能；本机全局环境的 xonsh 插件会在无交互终端中报错。

## 复现本次 G1–G3 实验

```powershell
.venv/Scripts/python.exe -m edgeinfer suite --steps 65536 --seeds 1 2 3 --eval-episodes 5 --eval-every 4096 --workers 3 --out results/new_run
```

默认 15 用户/3 节点，另评估 10/30 用户、2/6 节点。每个规模均重新训练 3 个策略，总计 15 次训练、983040 个训练交互步。

每次训练保存：

- `run.json`：全部配置、seed、运行库版本与交互预算。
- `learning_curve.csv`、`training_episodes.csv`：检查点和逐 episode 奖励。
- `policy.pt`：Actor/Critic 与观测归一化状态，用于评估。
- `test_episodes.json`：独立测试指标及能耗分项。
- `test_traces/`：逐请求、逐批次、功率区间原始记录。
- `g3_invariance.json`：额外环境 seed 上能耗开关前后完全一致的检查结果。

suite 汇总为 `summary.csv`、`report.md` 和 PNG/SVG/PDF 图。报告只使用这次实际运行的数据，不加载旧目录的历史结果。

## 单次训练与评估

```powershell
.venv/Scripts/python.exe -m edgeinfer train --config configs/paper_aligned.toml --seed 1 --steps 65536 --out results/my_policy
.venv/Scripts/python.exe -m edgeinfer evaluate results/my_policy/policy.pt --seeds 21000 21001 21002 --out results/my_evaluation
.venv/Scripts/python.exe -m edgeinfer energy-check results/my_policy/policy.pt --out results/my_policy/passive_energy.json
.venv/Scripts/python.exe -m edgeinfer plot results/g123_final
```

修改场景时编辑 TOML；用户和节点维度从配置推导，不再手动同步多个文件。保存的策略对应固定 N/J，规模改变后重新训练。

## 接入其他方法

实现 `experiment.Policy.act(observation, mask, deterministic=True)`，返回 N 个整数卸载动作；使用同一个 EdgeEnv 和 evaluate 即可评价。资源分配方法实现 `allocation.BandwidthAllocator`，作为 EdgeEnv 的 allocator 传入。新方法不能修改 physics、workload 或最终指标。

`EdgeEnv.action_masks()` 提供 SB3-Contrib 的扁平 MultiDiscrete 掩码形式；`info['action_mask']` 提供二维形式。标准 MaskablePPO 应使用其专门的 masked evaluation 接口，不能遗漏评估时的 mask。SB3-Contrib 仅为可选依赖，本次基线仍保留导师 PPO 核心结构。

## 能耗

通信按功率乘实际传输时间积分；GPU 按空闲基线加 prefill/decode 增量核算。能耗模块不改变 reward、动作、随机数或物理状态。

**默认相位功率是演示估算值。当前结果是模拟能耗，不是 NVIDIA GPU 实测。** 待取得目标 GPU 的功率轨迹后，替换 TOML 中的相位功率；或用以下命令积分测量文件：

```powershell
.venv/Scripts/python.exe -m edgeinfer integrate-power examples/power_trace.csv
```

CSV 契约为 `time_s,power_w`，时间严格递增、功率非负。本例为已知人工积分样例，期望 55 J，不是硬件采样。

## G4：统一算法比较

G4 内置六种方法：TP-MPPO、无掩码 TP-PPO、合法随机、最强信道、批次均衡和 cap 显存贪心。所有方法共用物理环境、P2/P3 带宽分配、能耗与最终指标。

一次完整运行分成三个可单独重用的步骤：

```powershell
# 从 G1–G3 读取相同配置、seed 与预算，训练 TP-PPO。
.venv/Scripts/python.exe -m edgeinfer benchmark-train --reference-suite results/g123_final --out results/g4_training --workers 3

# 固定 5 个规模 × 10 个外部场景，保存请求和上下行信道。
.venv/Scripts/python.exe -m edgeinfer benchmark-freeze --reference-suite results/g123_final --out results/g4/scenarios

# 复用两个学习方法的策略，运行所有方法、两种负载和统计汇总。
.venv/Scripts/python.exe -m edgeinfer benchmark --out results/g4/new_comparison
```

当前交付已保存训练、场景和正式结果，查看 `results/g4/benchmark`。再次比较时可以跳过前两步，直接使用保存的输入与策略。

主指标使用同一固定窗口的 Goodput、Throughput 和模拟能耗。报告有两种模式：`closed_loop` 保留原论文的完成后思考机制；`fixed` 让各方法接收完全相同的外生请求。学习策略均在闭环模式训练，因此 fixed 同时是迁移测试。

统计包含每策略重复/场景单元、均值、重复间标准差、名义 95% crossed-bootstrap 区间和配对方法差。对时延单列完成样本数；零服务的 J/合规请求未定义，不填零或小常数。

重新生成报告、重放任意一个测试单元：

```powershell
.venv/Scripts/python.exe -m edgeinfer benchmark-report results/g4/benchmark
.venv/Scripts/python.exe -m edgeinfer benchmark-replay results/g4/benchmark --case default --mode fixed --method tp_mppo --replicate 1 --scenario 40001 --out results/g4/replay_example
```

`benchmark-replay` 精确比较保存的动作、reward 和物理指标，排除非确定性的 CPU 墙钟时间。每个测试单元均有 trajectory.npz；每种方法/重复/模式/case 的首场景另有完整请求、批次和功率明细及自动重放记录。

新增方法只需实现原有 Policy 接口，并提供显式预算的 factory：

```python
def build(config, replicate_seed):
    policy = YourPolicy(config, replicate_seed)
    return policy, 0  # 启发式预算为 0；学习方法返回实际训练步数。
```

实际示例在 [examples/custom_policy.py](examples/custom_policy.py)：

```powershell
.venv/Scripts/python.exe -m edgeinfer benchmark --methods tp_mppo cyclic --custom cyclic=examples.custom_policy:build --out results/g4/custom_comparison
```

学习方法的 factory 按 replicate_seed 加载自己的策略，并返回对应训练步数。该正预算参与与内置学习方法的同预算核对。factory 不在测试时训练或读取保存场景中的未来输出。

如果新方法还要更换带宽分配器，实现现有 `BandwidthAllocator` 接口即可；通过 `--allocator cyclic=examples.custom_policy:equal_resources` 指定。该示例返回等带宽分配器。此选项只作用于明确命名的自定义方法，所有方法仍使用同一物理模型和资源约束。

## 当前复现程度

实现的是论文兼容公式和 TP-MPPO 核心，不能把本版称为论文所有图表的精确复现。原文未公开的数值和下行失败语义已在模型契约中列明。TP-PPO 已作为同预算无掩码消融实现；全变量 MPPO、Rewardless、文献中的加权 Heuristic、真实 GPU 校准及后续创新机制尚未加入。

`results/smoke`、`results/seed2_diagnostic` 是短程调试记录；`results/g123` 是浮点归一化契约修正前的失败尝试。正式报告只使用 `results/g123_final`。

## G5：研究问题诊断

G5 审计能耗来源与固定项，扫描无线条件 × 提供负载，并执行 SLO/批次组成、长度估计误差和小规模精确放置对照。不新增神经网络、不重新训练策略、不覆盖 G1–G4。

```powershell
.venv/Scripts/python.exe -m edgeinfer g5-run --plan configs/g5.json --out results/g5_new --workers 3
.venv/Scripts/python.exe -m edgeinfer g5-report results/g5
.venv/Scripts/python.exe -m edgeinfer g5-replay results/g5 --case think_0.5_atten_40 --method tp_mppo --allocator equal_uplink --replicate 1 --scenario 50001 --out results/g5_replayed
# 在 Git checkout 中重建全部小规模控制与前沿，重放 24 个主扫描单元，再核对汇总。
.venv/Scripts/python.exe tools/verify_g5.py results/g5
```

当前保存的是 `results/g5`；重新运行必须选用新目录。主扫描有 1620 次完整评估，PPO 为冻结策略的迁移诊断。离线 oracle 明确标注未来信息；精确前沿仅针对单次放置和固定 P2/P3，不代表全时域全局最优。原始表、成对差、图、输入和动作轨迹均保存，能耗仍是未做硬件校准的估算。

先读 [G5 主要发现与去留判断](docs/g5_findings.md)，再看完整报告与原始表。Git checkout 以当前提交记录来源；讨论 ZIP 使用包内 `BUNDLE_INFO.json` 的提交记录，运行 G5 不需要伪造一个 Git 仓库。
