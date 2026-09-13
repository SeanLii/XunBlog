---
title: Temporal Ensemble
description: 合并多个重叠 action chunks，得到稳定的当前动作。
---

# Temporal Ensemble

<NoteMeta status="Draft" difficulty="Intermediate" updated="2026-09" />

## 为什么会有多个当前动作

如果策略在每个时刻都预测一个长度为 $k$ 的 chunk，那么执行时刻 $t$ 会同时被多个历史 chunk 覆盖：

```text
at t-2:  [a(t-2), a(t-1), a(t),   ...]
at t-1:          [a(t-1), a(t),   ...]
at t:                    [a(t),   ...]
```

这些预测都对当前动作 $a_t$ 提供一个估计。

## 加权组合

Temporal ensemble 对这些候选预测加权平均，通常让更新的预测拥有更高权重：

$$
a_t = \frac{\sum_i w_i\hat{a}^{(i)}_t}{\sum_i w_i}
$$

直觉上，它像一个滚动的短期共识：旧预测提供连续性，新预测及时吸收最新观测。

## 它和 Action Chunking 的关系

- Chunking 让模型预测连贯的动作片段。
- Receding prediction 让策略持续看见新观测。
- Temporal ensemble 平滑重叠片段之间的差异。

