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
---

# Cross-Embodiment Learning

Cross-Embodiment Learning 研究如何让一个 learning system 利用来自**不同机器人 embodiment**的数据，并把跨平台的共同经验转化成可迁移能力。

不同 embodiment 可能拥有：

- 不同 kinematics；
- 不同 joint counts；
- 不同 grippers；
- 不同 camera placement；
- 不同 control frequencies；
- 不同 action representations；
- 不同 workspace 与 morphology。

所以“把所有 robot datasets concat 在一起训练”并不能自动解决 cross-embodiment problem。

## Embodiment 的含义

Embodiment 不只是 robot 外壳长得不同。

对 policy 而言，它决定了 observation 和 action interface。

例如同一个 high-level task：

> 把杯子放进盒子。

Robot A 可能输出 7-DoF joint velocity；Robot B 输出 end-effector delta pose；Robot C 还多一个 gripper scalar。

任务语义可以共享，但 motor realization 不同。

## 可以共享什么

跨机器人数据可能共享：

- object semantics；
- task language；
- visual concepts；
- contact / manipulation patterns；
- high-level temporal structure；
- “抓取 → 移动 → 放置”等 abstract skill regularities。

而更低层：

- joint-specific control；
- torque / velocity ranges；
- kinematic constraints；

通常更 embodiment-specific。

Cross-embodiment model 的关键是让共享 information 与专属 interface 共存。

## Action Space 是最大的 Alignment 问题之一

如果 action dimensions 不同：

\[
a^{(A)}\in\mathbb R^7,
\qquad
 a^{(B)}\in\mathbb R^{14},
\]

不能直接要求一个固定 output head 同时解释两者。

常见策略包括：

- 统一到 common action representation；
- padding + mask；
- embodiment-specific action heads；
- tokenized action spaces；
- shared backbone + robot-specific adapters；
- conditioning on embodiment identity / state schema。

没有一种表示天然适合全部机器人。

## Observation Alignment

不同 robots 的 cameras、proprioception 也不同。

视觉部分相对容易共享，因为 images 具有相似 data type；proprioception 则需要处理不同 state dimensions 与 physical meaning。

因此 cross-embodiment learning 不只是 action normalization，也涉及 multimodal input alignment。

## Positive Transfer 与 Negative Transfer

共享训练只有在不同 embodiments 存在可复用 structure 时才有价值。

如果共享 representation 帮助某 robot 学得更好，叫 positive transfer。

如果不相关或冲突数据让 performance 下降，就是 negative transfer。

因此“data 越多越好”不能作为 cross-embodiment learning 的无条件结论。

## Open X-Embodiment

Open X-Embodiment collaboration 将多个机构、多个 robot platforms 的 datasets 标准化汇总，并训练 RT-X models。

其重要意义在于把问题从“一个 robot 一个 dataset”推向：

```text
many robots
many tasks
many environments
      ↓
shared generalist training
```

论文报告跨平台 training 可以为多个 robots 带来 positive transfer，展示了 cross-embodiment scaling 的可行性。

## 与 Multi-Task Learning 的区别

Multi-task learning 强调多个 tasks 共享模型。

Cross-embodiment learning 强调：

> 即使 task 相似，执行这个 task 的 physical agent interface 也发生了变化。

两者可以同时存在，但 axis 不同：

```text
task diversity
×
embodiment diversity
```

## 与 VLA 的关系

大型 [Vision-Language-Action Model](/robot-learning/vision-language-action-model/) 常希望同时覆盖：

- many language tasks；
- many visual environments；
- many robot embodiments。

因此 cross-embodiment data 是 generalist VLA training 的重要组成部分，但 Cross-Embodiment Learning 不是 π0 或 VLA 才出现的概念。

## Sources

- Open X-Embodiment Collaboration et al. *Open X-Embodiment: Robotic Learning Datasets and RT-X Models*. 2023/2024.
