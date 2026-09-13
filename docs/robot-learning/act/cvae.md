---
title: CVAE 与 latent z
description: 为什么 ACT 用条件潜变量表示示范动作的风格与多样性。
status: learning
pageType: application
canonical: /robot-learning/act/cvae
difficulty: intermediate
updated: "2026-09"
---

# 为什么需要 CVAE 与 latent z？

<NoteMeta />

## 一种观测，多种合理动作

面对相同观测，人类示范可能以不同速度、轨迹或姿态完成任务。如果只用均方误差拟合，模型可能把多种模式平均成一个并不自然的动作。

CVAE 引入 latent variable $z$，把“当前条件下示范动作的变化因素”编码进连续空间：

$$
q_\phi(z\mid A, o), \qquad p_\theta(A\mid o,z)
$$

其中 $o$ 是观测，$A$ 是动作 chunk。

## Reparameterization

Encoder 输出 $\mu$ 与 $\sigma$，再写成：

$$
z = \mu + \sigma \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0,I)
$$

这样随机性被移到 $\epsilon$，梯度仍能通过 $\mu$ 和 $\sigma$ 传播。

## 为什么推理时常用 z = 0

训练时，$z$ 帮助模型解释示范中的变化；KL loss 同时把 posterior 约束到标准正态先验附近。推理时取先验均值 $z=0$，相当于选择一个确定、典型的行为模式，避免每次采样给控制引入额外随机性。

::: info 当前理解
“$z=0$”不是说 latent 没有用。正是训练阶段使用 latent 建模多样性，并通过 KL 让空间结构靠近先验，才让先验均值在推理时成为可用选择。
:::
