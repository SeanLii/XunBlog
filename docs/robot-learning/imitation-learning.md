---
title: Imitation Learning
description: 从专家示范中学习机器人策略。
---

# Imitation Learning

<NoteMeta status="Seed" difficulty="Foundation" updated="2026-09" />

## 问题定义

给定专家示范数据 $\mathcal{D}=\{(o_t,a_t)\}$，学习策略 $\pi_\theta(a_t\mid o_t)$，让机器人在观测 $o_t$ 下生成合理动作 $a_t$。

最直接的方法是 Behavior Cloning：把策略学习看成监督学习。

## 核心困难

训练数据来自专家访问过的状态；部署时，模型的小错误会把机器人带到专家数据之外。新的状态导致更不可靠的预测，误差因此不断累积。这正是理解 ACT 的起点。

