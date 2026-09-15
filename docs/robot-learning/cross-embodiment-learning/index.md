---
title: "Cross-Embodiment Learning"
kind: "canonical"
domain: "Robot Learning"
parent: "Robot Learning"
canonical: "/robot-learning/cross-embodiment-learning/"
prerequisites:
  - "/robot-learning/imitation-learning/"
related:
  - "/robot-learning/vision-language-action-model/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Cross-Embodiment Learning

Cross-Embodiment Learning 研究如何利用来自不同 robot embodiments 的数据学习可以共享、迁移或适配的 policy representations。

这里的 embodiment 包含与控制相关的 physical interface，例如 kinematics、joint count、morphology、gripper、camera placement、proprioceptive state、control frequency 与 action space。

因此核心问题是：哪些知识可以跨机器人共享，哪些表示必须保留 embodiment-specific structure。

## Formal Setting

设不同机器人为：

\[
e\in\mathcal E.
\]

每个 embodiment 可以具有自己的 observation space：

\[
\mathcal O_e
\]

与 action space：

\[
\mathcal A_e.
\]

数据来自：

\[
\mathcal D_e
=
\{(o_t^{(e)},a_t^{(e)},\ldots)\}.
\]

cross-embodiment model 希望利用：

\[
\bigcup_{e\in\mathcal E}\mathcal D_e
\]

学习共享 parameters、shared representations 或 transferable policy，同时正确处理各 embodiment 不同的 interfaces。

## Shared 与 Embodiment-Specific Structure

不同机器人虽然低层控制接口不同，但仍可能共享高层规律，例如：

- object semantics；
- visual features；
- language grounding；
- manipulation phases；
- contact patterns；
- task temporal structure；
- affordances；
- goal relationships。

同一 task 的语义可以跨 embodiments 保持一致，而具体 joint trajectory 与 actuator command 通常不会一致。

因此模型需要同时表示：

\[
\text{shared task / semantic structure}
\]

与：

\[
\text{embodiment-specific control structure}.
\]

## Action-Space Heterogeneity

不同机器人 action spaces 可能具有不同维度和物理含义：

\[
a^{(A)}\in\mathbb R^{7},
\qquad
a^{(B)}\in\mathbb R^{14}.
\]

即使维度相同，也可能分别表示 joint position、joint velocity、end-effector delta pose、torque 或 normalized actuator command。

常见对齐策略包括：

- common Cartesian representation；
- action normalization；
- padding 与 masks；
- embodiment-specific output heads；
- shared backbone + adapters；
- action tokenization；
- conditioning on embodiment identity；
- learned embodiment-specific mappings。

这些方案在共享程度和保留物理差异之间做不同取舍。

## Observation-Space Heterogeneity

视觉输入通常可以使用相似 encoder，但 proprioception 更明显地依赖 embodiment。

不同 robot state 可能包含不同数量的 joints、joint ordering、gripper state、base pose、force/torque 或 end-effector pose。

因此需要定义：

\[
o^{(e)}\rightarrow h^{(e)}
\]

如何映射到共享 representation space，并保留各字段的物理语义。

## Positive 与 Negative Transfer

跨机器人联合训练希望获得 positive transfer：

\[
\text{data from robot A}
\rightarrow
\text{better performance on robot B}.
\]

但若 sensor semantics、control interface、task distribution 或 morphology 差异过大，共享 parameters 也可能产生 negative transfer。

因此 cross-embodiment scaling 不等于简单拼接 datasets。

## Data Standardization

大规模 cross-embodiment training 需要统一 dataset representation，包括：

- observation schema；
- action schema；
- timestamps；
- camera calibration；
- language/task labels；
- episode boundaries；
- metadata；
- normalization statistics。

如果不同 datasets 中相同数值字段具有不同物理含义，统一 tensor shape 并不等于统一语义。

## Open X-Embodiment

Open X-Embodiment 将多个机构、多个 robot platforms 的 datasets 组织成标准化 collection，并训练 RT-X models 研究跨机器人联合训练。

该工作展示了多机器人数据可以统一用于高容量 policy training，并观察到跨平台 positive transfer。它把 cross-embodiment learning 从单个 robot 的 transfer problem 推向大规模 heterogeneous data problem。

## Multi-Task 与 Cross-Embodiment

Multi-task learning 与 cross-embodiment learning 对应两个不同变化轴：

\[
\text{task diversity}
\times
\text{embodiment diversity}.
\]

同一 robot 可以执行很多 tasks；同一 task 也可以由很多 robots 完成。generalist robot model 往往需要同时处理两个维度。

## Generalization Settings

Cross-embodiment evaluation 可以包括：

1. 多个 seen robots 的联合训练；
2. seen robot 上的新 task；
3. 少量数据下对新 robot adaptation；
4. unseen embodiment 上的 zero-shot 或 near-zero-shot transfer。

这些 setting 的难度与含义不同，不应使用同一个“cross-embodiment generalization”指标笼统概括。

## Relation to VLA Models

大型 [Vision-Language-Action Model](/robot-learning/vision-language-action-model/) 经常同时使用多任务、多环境与多机器人 data。

VLA 描述 vision、language 与 action 的 policy model family；cross-embodiment learning 描述 robot-interface diversity 下的共享与迁移问题。两者相关，但不是同一个概念。

## Sources

- Open X-Embodiment Collaboration et al. *Open X-Embodiment: Robotic Learning Datasets and RT-X Models*. 2023/2024.
