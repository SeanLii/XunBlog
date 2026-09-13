---
title: 为什么需要 Action Chunking
description: 从误差累积与动作连贯性理解 ACT 的出发点。
---

# 为什么机器人需要 Action Chunking？

<NoteMeta status="Learning" difficulty="Foundation" updated="2026-09" />

## 1. 从单步预测的问题开始

最直接的 Behavior Cloning 策略在每个时刻预测一个动作：

$$
a_t = \pi_\theta(o_t)
$$

这看起来合理，但策略执行上一步预测后，下一步观测已经受它自己的误差影响。一个很小的偏差可能让机器人进入训练数据中很少出现的状态；模型在那里更容易继续犯错。

```text
小预测误差 → 状态偏移 → 遇到陌生观测 → 更大预测误差
```

这通常被称为 **compounding error（误差累积）**。

## 2. Action Chunking 的直觉

Action Chunking 不只预测 $a_t$，而是一次预测长度为 $k$ 的动作序列：

$$
\hat{A}_t = (\hat{a}_{t}, \hat{a}_{t+1}, \ldots, \hat{a}_{t+k-1})
$$

模型因此学习的不是一个孤立控制点，而是一小段具有内部结构的动作。比如“抓住杯子”并不是几十个毫无关系的关节命令，而是一段相互依赖的运动。

## 3. 它改变了什么

### 更长的决策视野

单步策略只对下一刻负责；chunk policy 需要让一段动作整体与当前任务状态相符。

### 更连贯的动作

同一个 chunk 内的动作由一次前向计算共同产生，模型可以表达它们之间的时间关系。

### 更少的有效决策次数

如果每隔若干步才重新规划，长任务中的独立预测次数减少，逐步漂移的机会也随之降低。

::: warning 需要修正的直觉
“预测一段动作”不代表机器人盲目执行整段、完全不再看环境。ACT 可以在每个时刻持续产生重叠的 chunks，再通过 temporal ensemble 组合当前动作。
:::

## 4. 一个极简接口

```python [act.py] {7-8}
import torch
from torch import nn

class ActionChunkPolicy(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int, chunk_size: int):
        super().__init__()
        self.chunk_size = chunk_size
        self.head = nn.Linear(hidden_dim, action_dim * chunk_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        actions = self.head(features)
        return actions.view(features.shape[0], self.chunk_size, -1)
```

这个例子只展示输出形状：`[batch, chunk_size, action_dim]`。真正的 ACT 还需要视觉特征、机器人状态、Transformer 和 CVAE。

## 5. 还没有解决的问题

Action Chunking 引出三个新问题：

1. 如何让模型同时理解观测和动作序列？→ [Transformer](/robot-learning/act/transformer)
2. 同一观测存在多种合理示范时怎么办？→ [CVAE](/robot-learning/act/cvae)
3. 不同时刻预测出的重叠 chunks 如何合并？→ [Temporal Ensemble](/robot-learning/act/temporal-ensemble)

