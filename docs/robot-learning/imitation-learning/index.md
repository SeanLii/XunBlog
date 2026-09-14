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

Imitation Learning 让 policy 从 expert demonstrations 中学习行为。它与强化学习的核心区别不在网络结构，而在监督信号来源：imitation learning 直接利用 expert behavior，而不是只依靠环境 reward 通过试错寻找策略。

## 基本对象

设环境状态或观测为 $s_t$，动作为 $a_t$，policy 为

\[
\pi(a_t|s_t).
\]

Expert policy 记为

\[
\pi^*(a_t|s_t).
\]

Demonstration dataset 通常由 expert trajectories 构成：

\[
\mathcal D
=
\{(s_0,a_0),(s_1,a_1),\ldots\}.
\]

在视觉机器人中，$s_t$ 不一定是完整物理 state，也可以是 cameras、proprioception 等 observation。此时更准确的符号可以写成 $o_t$。

## 学习目标的不同形式

Imitation Learning 是上位概念。不同方法对“怎样使用 demonstration”有不同答案。

[Behavior Cloning](/robot-learning/behavior-cloning/) 直接把 demonstration 当成 supervised dataset，学习 observation-to-action mapping。DAgger 一类方法则让 learned policy 进入它自己会访问的 states，并在这些 states 上继续获得 expert labels，从而处理 deployment distribution 与 demonstration distribution 不一致的问题。

此外，还存在 inverse reinforcement learning 等路线，但它们不是理解 ACT 的必要前提，因此当前知识范围不继续展开。

## Sequential Deployment

机器人 policy 的预测会改变后续状态：

\[
s_{t+1}\sim P(s_{t+1}|s_t,a_t).
\]

因此动作误差不是独立样本上的一次性分类错误。一个错误 action 可能把系统带到 demonstration 中少见的 state；下一步 policy 又必须在这个新 state 上继续决策。

这种 closed-loop property 是理解 imitation learning 的核心，也是 ACT 论文将 compounding error 作为主要问题背景的原因。

## ACT 的位置

ACT 属于 imitation learning。它仍然从 demonstrations 学习，而不是在环境中通过 reward optimization 重新发现动作。ACT 的创新重点在于：把未来动作按 chunk 预测、用 temporal ensemble 融合重叠预测，并把 policy 训练成 CVAE 以适应 human demonstrations 的变化。

## Sources

- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware — Zhao et al., 2023](https://arxiv.org/abs/2304.13705)
- [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning — Ross, Gordon, Bagnell, 2011](https://proceedings.mlr.press/v15/ross11a.html)
