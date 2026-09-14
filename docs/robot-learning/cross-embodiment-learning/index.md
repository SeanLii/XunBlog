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
  - "/robot-learning/pi0/pretraining-and-posttraining/"
---

# Cross-Embodiment Learning

Cross-Embodiment Learning 指的是：**把来自不同机器人形态、不同控制空间的数据放进同一个学习系统，让模型跨机器人共享可迁移的知识。**

这里的 embodiment 不只是“机器人长得不一样”，还包括动作空间、关节数量、相机配置、控制频率等差异。

例如：

```text
single-arm robot ───┐
dual-arm robot ─────┤
mobile manipulator ─┼──→ shared policy / foundation model
other robot ────────┘
```

## 为什么不同机器人数据仍然可能共享信息

两个机器人即使关节结构完全不同，也可能面对相似的物理问题：

- 哪个物体是杯子；
- 抓取前要先对准；
- 衣服被折叠后视觉形态怎样变化；
- 打开抽屉需要先接触把手再施加运动；
- 失败以后怎样重新定位目标。

这些知识不一定绑定某一个具体关节编号。

因此 cross-embodiment training 的目标不是让所有机器人拥有完全相同的低层动作，而是尽可能共享更高层的视觉、语义和操作经验，再保留各 embodiment 自己的 action representation。

## 最大困难在动作空间

视觉和语言比较容易共享统一表示，但 robot action 往往完全不同。

例如：

```text
Robot A: 7 DoF arm
Robot B: 14 DoF bimanual
Robot C: mobile base + two arms
```

它们的 action vector dimension、每一维物理意义、控制范围都可能不同。

因此跨 embodiment 模型通常需要：

- 对 state / action 做统一的数据接口；
- padding 或 masking 不存在的 action dimensions；
- normalization；
- embodiment-specific transforms；
- 让共享 backbone 与不同 action spaces 能兼容。

这些是工程实现的重要部分，而不是“把所有轨迹文件直接 concat”这么简单。

## 与多任务学习的区别

Multi-task learning 强调任务不同；cross-embodiment learning 强调执行这些任务的**机器人身体和控制空间也不同**。

两者可以同时存在：π0 的 pre-training mixture 既跨 tasks，也跨 embodiments。

所以可以把它看成二维的数据扩展：

```text
              Task 1   Task 2   Task 3 ...
Robot A         ✓        ✓
Robot B         ✓                 ✓
Robot C                  ✓        ✓
...
```

模型希望从整个矩阵里学习共享结构。

## π0 中的 cross-embodiment training

π0 论文把多个 single-arm、dual-arm 与 mobile manipulator 的数据放进同一个 pre-training mixture，并希望训练出一个 generalist robot policy。

这件事与 π0 的 VLM backbone 配合：

- VLM 提供跨场景、跨物体的通用视觉语言表示；
- cross-embodiment robot data 提供真实 physical interaction experience；
- post-training 再把 base model 专门适配到高质量 downstream task。

因此 π0 的 generality 不只来自“模型很大”，也来自训练数据覆盖了多个机器人与多个任务。

## Sources

- Black et al., **π0: A Vision-Language-Action Flow Model for General Robot Control**, 2024. https://arxiv.org/abs/2410.24164
- Open X-Embodiment Collaboration et al., **Open X-Embodiment: Robotic Learning Datasets and RT-X Models**, 2023. https://arxiv.org/abs/2310.08864
