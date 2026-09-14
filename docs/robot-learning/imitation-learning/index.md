---
title: "Imitation Learning"
kind: "canonical"
domain: "Robot Learning"
parent: "Robot Learning"
canonical: "/robot-learning/imitation-learning/"
prerequisites:
  []
related:
  - "/robot-learning/behavior-cloning/"
  - "/robot-learning/act/"
---

# Imitation Learning

Imitation Learning 是让 agent 通过专家 demonstrations 学习行为的一类方法。

在机器人场景里，demonstration 通常是一系列“机器人看到了什么、专家接下来做了什么”：

\[
\tau=(o_1,a_1,o_2,a_2,\ldots,o_T,a_T).
\]

目标不是直接手写控制规则，而是从这些 demonstration 中得到一个 policy：

\[
\pi(a\mid o).
\]

也就是：给定当前 observation，应该执行什么 action。

## Demonstration 的数据结构

以机械臂抓取为例：

```text
时刻 t

camera image + joint state
            │
            ↓
expert chooses action
            │
            ↓
new robot state
```

记录多次以后，就得到 trajectories。它们可以来自人工 teleoperation、专家 controller、motion planner 或其他可靠 policy。

Imitation learning 的信息来源与 reinforcement learning 不同。RL 通常依赖 reward 告诉 agent “结果好不好”；imitation learning 直接获得“专家在这里做了什么”。

## 最直接的方法：Behavior Cloning

[Behavior Cloning](/robot-learning/behavior-cloning/) 把 demonstration 拆成 supervised pairs：

\[
(o_t,a_t).
\]

然后训练

\[
\pi_\theta(o_t)\approx a_t.
\]

它简单、容易扩展，也特别适合已有大量 demonstration data 的机器人任务。

但顺序决策和普通 supervised learning 有一个根本区别：policy 的预测会影响自己下一步看到的输入。因此单步模仿准确，并不自动保证完整 rollout 成功。

## Imitation Learning 的主要困难

### Distribution shift

训练数据来自 expert 常访问的状态，部署数据却来自 learned policy 自己的行为。如果 learner 偏离 expert trajectory，它可能进入没有足够训练数据的区域。

### Multimodality

同一个观测下可能存在多个合理动作。例如抓杯子可以从左侧接近，也可以从右侧接近。简单 deterministic regression 可能把不同策略平均成一个并不合理的中间动作。

### Long horizon

任务越长，policy 需要把更多局部决策正确衔接起来。早期误差可能影响很久以后的状态。

这些问题分别推动了 data aggregation、generative policies、action chunking 等不同方向。

## ACT 在这个体系中的位置

ACT 仍然是 imitation learning。它使用 demonstrations 学习从 observation 到 action sequence 的 policy，而不依赖在线 reward optimization。

它重点处理两个现实问题：

- 通过 [Action Chunking](/robot-learning/act/action-chunking/) 一次预测多步动作，降低长序列逐步决策的负担；
- 通过 [CVAE in ACT](/robot-learning/act/cvae-in-act/) 在训练中表示 human demonstration 的变化。

因此 ACT 不是 imitation learning 的替代物，而是 imitation learning 中一种具体 policy 设计。

## Sources

- Ross, Gordon & Bagnell, **A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning**, 2011. https://proceedings.mlr.press/v15/ross11a.html
- Zhao et al., **Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware**, 2023. https://arxiv.org/abs/2304.13705
