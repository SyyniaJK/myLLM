# 实验图契约

绘图后端为 Python/matplotlib。此处交付的是基线核验图，不作为原论文图的重新发表版本。

1. 核心问题：训练是否改变策略表现；规模变化下如何变化；能耗核算由哪些分量构成。不得预设改善、收敛或节能结论。
2. `learning` 三个定量面板分别显示验证 reward、Goodput 比例和 OOM。所有已完成训练 seed、全部检查点均使用。阴影是训练 seed 间样本 SD，不是 CI。
3. `scaling` 显示不同用户数、节点数的独立测试均值与训练 seed 间 SD。每个训练策略先对测试环境取均值。所有场景使用相同训练预算。
4. `energy` 显示默认场景全部独立测试 episodes 的固定窗口相位能耗均值。GPU 与无线使用分开的纵轴尺度，避免很小的无线分量无法辨认；各柱标出原始 J 值。相位功率为演示参数，图标题明确非实测硬件能耗。
5. 布局为 quantitative grid，白底、固定方法颜色、可编辑 SVG/PDF、600 dpi PNG，图宽约 160–179 mm。无数据删点或平滑隐藏。
6. 数据来源为 suite summary.csv、每策略 learning_curve.csv 与 test_episodes.json；不使用历史数组、图片数字化数据或虚构结果。
7. 训练/验证/测试 split、环境 seed 和训练 seed 区分记录在 suite.json。暂不进行显著性检验或主张原论文的性能优势。

交付前执行静态绘图检查并检查 PNG。标注和误差线定义与 report.md 保持一致。

静态检查接受 PNG 预览而未导出 TIFF 的提示：本次交付 PNG 和可编辑 PDF/SVG，尚无目标期刊的 TIFF 提交要求。
