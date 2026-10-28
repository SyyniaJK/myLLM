# 实现参考与版本记录

核对日期：2026-10-28。下列 commit 是核对时的参考快照，不要求运行平台时下载整套仓库。

| 仓库 | 参考 commit | 许可证 | 本版使用范围 | 是否改变 TP-MPPO 算法语义 |
|---|---|---|---|---|
| [Gymnasium](https://github.com/Farama-Foundation/Gymnasium/tree/db8591eb9f5695dc34074b69fe39856d847db15a) | db8591eb9f5695dc34074b69fe39856d847db15a | MIT | Env reset/step、spaces、seed、check_env；实际运行版本 1.4.0 | 规范环境接口，正确区分 time-limit truncation；不替换卸载策略 |
| [SB3-Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/tree/a8af63682070df0bda9707400b2e971055600910) | a8af63682070df0bda9707400b2e971055600910 | MIT | 核对 MaskableMultiCategorical 的逐维 mask、联合 log probability 和 entropy 求和；未复制其算法实现 | 保留导师自己的 PPO；只提供兼容掩码接口 |
| [InferSim](https://github.com/alibaba/InferSim/tree/d12ec47f83d5896efd77007ad5d5f2ad044c2876) | d12ec47f83d5896efd77007ad5d5f2ad044c2876 | Apache-2.0 | 阅读其 FLOPs/FLOPS、显存带宽、MFU 和 benchmark 校准的适用边界；未引入代码 | 不变；论文兼容模型仍使用原文公式 |

实际检查了 Gymnasium `core.py`、SB3-Contrib `common/maskable/distributions.py` 和 InferSim README。许可证按上述固定快照的 LICENSE 确认。

接口依据：[Gymnasium 自定义环境](https://gymnasium.farama.org/introduction/create_custom_env/)、[MaskablePPO 官方文档](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)。特别注意，普通 env checker 不使用 action mask；因此环境对范围内的越掩码动作定义为失败转移，masked policy 自身仍不得采样这些动作。

能耗边界参考：[Zeus 测量窗口](https://ml.energy/zeus/measure/)、[TokenPowerBench](https://github.com/chenxuniu/TokenPowerBench)。本版只实现模拟能耗和测量 CSV 积分，未运行这些工具、未声称用它们测得了目标 GPU 功率。

Vidur、SimPy、Sionna、OR-Tools、vLLM 不作为当前运行依赖。G1 的小规模下行最优性由六用户的全部子集枚举验证，无须额外整数规划求解器。

教师原 PPO 代码是本项目提供的研究材料；重建版本保留其核心结构和出处说明。公开发布前应由代码权利人确定适用许可证。
