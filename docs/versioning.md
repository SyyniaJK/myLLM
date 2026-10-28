# 版本管理与讨论包

Git 仓库根目录是 `EdgeInferenceLab`，远程仓库是 https://github.com/SyyniaJK/myLLM 。首个快照标记为 `v0.1.0`。项目父目录的论文、教师原 ZIP、录音和其他材料不会因本仓库的操作自动上传。

## 纳入版本管理的内容

- 源代码、测试、配置、依赖版本、示例与研究契约文档。
- G1–G3 正式结果、两种学习方法的最终策略和训练曲线。
- G4 固定测试场景、1800 次逐场景指标和动作/reward、统计表、图与重放检查。

`.gitignore` 排除环境、缓存、失败实验、新的临时运行和可由重放生成的完整明细。已有本地明细仍然保留。需要纳入新的正式实验时，先在 `.gitignore` 中明确增加该结果目录，再提交；不要把全部调试输出加入版本库。

## 日常保存

以下命令均在仓库根目录执行。提交之前查看实际改动，再选择要保存的文件。

```powershell
git status
git diff
git add edgeinfer tests configs docs README.md CHANGELOG.md
git diff --cached --stat
git commit -m "Describe the research or implementation change"
git push
```

每完成一个可解释的改动提交一次。发布里程碑时同步修改 `pyproject.toml` 的版本号与 `CHANGELOG.md`，完成验证后创建相同版本的标签：

```powershell
git tag -a v0.2.0 -m "Describe this milestone"
git push origin main
git push origin v0.2.0
```

标签用于固定论文中的代码/实验版本；以后新增方法不会改写 `v0.1.0` 的结果。

## 重新生成讨论 ZIP

打包工具仅导出当前 **已提交且带版本标签** 的 Git 快照，要求版本号与标签对应。它不会把未提交修改、虚拟环境或忽略目录混入 ZIP。额外参考文件由命令明确给出。

```powershell
.venv/Scripts/python.exe tools/build_chat_bundle.py --out dist/EdgeInferenceLab-v0.1.0-chat.zip --references ../TP-MPPO.md ../TPO-PPO.md ../TP-MPPO.pdf ../TPO-PPO.pdf
```

ZIP 内有单一 `EdgeInferenceLab/` 根目录，先阅读 `CHAT_CONTEXT.md`。`BUNDLE_INFO.json` 记录版本标签、Git 提交编号、远程仓库与额外参考文件；`BUNDLE_CONTENTS.txt` 列出完整包内容。包内没有另行计算内容摘要；文件由 Git 快照直接导出，并用 ZIP 自带的完整性检查验证。

把生成的 ZIP 直接附到新的 ChatGPT 对话即可。本工具不自动向外部聊天发送文件。ZIP 存在本地 `dist/`，不作为重复的大文件提交到 Git；GitHub 已提供同一标签下的平台源代码下载。

## 从仓库重建

```powershell
git clone https://github.com/SyyniaJK/myLLM.git
cd myLLM
git checkout v0.1.0
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install -e . --no-deps
.venv/Scripts/python.exe -m edgeinfer benchmark-report results/g4/benchmark
.venv/Scripts/python.exe -m edgeinfer benchmark-replay results/g4/benchmark --case default --mode fixed --method tp_mppo --replicate 1 --scenario 40001 --out results/replayed_from_checkout
```

精确数值重放应使用记录的 Python/依赖版本和 CPU 设置。训练本身有随机种子，测试期策略与场景均已保存；墙钟耗时单独统计。
