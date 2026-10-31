# 约束后训练代码暂停记录

2026-10-31，用户要求暂停实验与开发，提交并推送现有代码，以重新打磨研究设计。此记录不启动训练、不发布数据、不激活新的约束。

研究目标是先从参考轨迹建立行为先验，再用低成本、训练时不依赖仿真器的 action 成功代理接替后训练目标。当前几何约束只是候选实现；不能把低约束损失等同于任务成功。

## 当前实现

- RoboTwin EEF 原始动作是双臂 20D（XYZ、旋转 6D、名义夹爪开度），映射到统一 80D 槽位；本训练路径使用绝对世界 EEF 动作。
- 训练从 HDF5 的 `trajectory_constraints/json` 读取标签，采用固定参考激活 mask，在 scheduler 恢复的可微干净动作估计上计算离线几何 loss。
- 保留原有 video/action flow-matching 目标。已完成训练采用 `lambda_video=1`、`lambda_action=1`、约束权重 0.1；不以生成视频作为成功判据。
- 已完成的 5 epoch、全局 batch 256、四卡模型使用旧数据 schema 1.2 / `full_motion_v1`。当前 RAX 双瓶 V3 候选和后续开发回放没有进入该数据或 checkpoint。
- 当前主开发代码包含局部 EEF 点、双夹爪组合和源绑定元数据等能力；不能据此声称旧模型已接受这些新标签训练。

## 保留分支

`feat/constraint-supervision` 是现有离线约束训练主实现；`feat/angle-center-hint` 保留独立角度中心梯度修订，与 RAX 同名分支配套。后者尚未合并或激活。

RAX 侧实现和实验状态见 [RAX 暂停记录](https://github.com/KraHsu/rax/blob/feat/robotwin-constraint-supervision/docs/constraint_research_pause_20261031.md)。独立实验源码另保存在其 `archive/constraint-research-20261031` 分支。

后加的严格统计验证协议不属于 loss 本身，也未在旧冻结自动训练流程中启用。暂停期间不更改既有数据发布/训练 hold，不增加门槛，不自动恢复任何训练、benchmark 或周期巡检。
