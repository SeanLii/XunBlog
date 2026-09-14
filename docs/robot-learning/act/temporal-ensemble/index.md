---
title: "Temporal Ensemble"
kind: "canonical"
domain: "Robot Learning / ACT"
parent: "ACT"
canonical: "/robot-learning/act/temporal-ensemble/"
prerequisites:
  - "/robot-learning/act/action-chunking/"
related:
  - "/robot-learning/act/inference/"
---

# Temporal Ensemble

Temporal Ensemble 是 ACT 在推理时融合 overlapping action chunks 的方法。

它出现的前提是：ACT 虽然每次预测未来 $k$ 步，但仍然可以在**每个 timestep 都重新查询 policy**。于是同一个执行时刻会收到来自不同历史 query 的多个预测。

## Overlapping Chunks 产生重复预测

假设 chunk size 为 4。

在 $t=0$：

```text
P0 = [a_0^(0), a_1^(0), a_2^(0), a_3^(0)]
```

在 $t=1$：

```text
P1 = [a_1^(1), a_2^(1), a_3^(1), a_4^(1)]
```

在 $t=2$：

```text
P2 = [a_2^(2), a_3^(2), a_4^(2), a_5^(2)]
```

于是机器人真正要在 $t=2$ 执行动作时，至少已经有三份候选：

\[
a_2^{(0)},\quad a_2^{(1)},\quad a_2^{(2)}.
\]

上标表示“这份预测是在什么时候产生的”。

## 最简单的选择并不理想

可以永远只用最新预测 $a_t^{(t)}$。这样响应最新视觉最直接，但相邻 timestep 的输出可能发生明显跳动。

也可以只执行最早 chunk 中的计划，直到它结束。这样动作连续，但会失去高频 closed-loop correction。

Temporal Ensemble 试图同时保留两者：

- 继续每一步看新 observation；
- 不让执行动作完全被一次最新预测突然替换。

## 指数加权融合

ACT 对同一执行时刻的多个候选动作做加权平均。论文写成指数权重形式。若有 $m$ 个候选预测，可以抽象为

\[
\bar a_t
=
\frac{\sum_i w_i a_t^{(i)}}{\sum_i w_i},
\]

其中

\[
w_i=\exp(-\lambda i)
\]

一类指数衰减权重。

具体“哪个候选对应较大权重”要结合实现中候选的排列顺序理解。官方代码把已填充的历史 predictions 取出后按数组顺序使用

\[
\exp(-0.01\cdot[0,1,2,\ldots])
\]

归一化，再加权求和。

因此阅读论文公式与代码时，不应只看一句“exponential weighting”，还要确认索引的时间方向。

## 连续 Joint Targets 的加权融合

在 ACT 的 ALOHA 设置中，action 是连续 joint-position target：

\[
a_t\in\mathbb R^{14}.
\]

多个预测位于同一个连续向量空间中，因此可以逐维做 weighted average。

如果 action 是离散符号或具有特殊几何约束，简单平均未必有意义。Temporal Ensemble 的形式与 action representation 有关。

## 它和普通 ensemble 不一样

普通 model ensemble 往往是“多个不同模型对同一个输入预测，再平均”。

ACT 的 temporal ensemble 可以只有**同一个模型**。差异来自不同时间的 observation：

```text
o_{t-2} → prediction for a_t
o_{t-1} → prediction for a_t
o_t     → prediction for a_t
```

所以这里 ensemble 的维度是时间，而不是模型数量。

## 官方实现的数据结构

官方 evaluation code 在启用 `temporal_agg` 时把 policy query frequency 设为 1，也就是每一个 control timestep 都生成一个新的 chunk。随后把第 $t$ 次 query 的输出写入 `all_time_actions[t, t:t+k]`，执行时再读取目标时刻这一列的所有已填充预测。

官方 evaluation code 用一个大 tensor 保存所有历史 chunk 对所有未来时刻的预测：

```text
query time ↓       target time →

0   a0 a1 a2 a3 ...
1      a1 a2 a3 ...
2         a2 a3 ...
3            a3 ...
```

执行时刻 $t$ 到来后，就取这一列所有已经存在的预测并融合。

这种二维视图非常适合理解 temporal ensemble：**每一行是一整个 chunk，每一列是同一个物理时刻的多次预测。**

## Temporal Ensemble 的能力边界

它可以平滑相邻预测并利用持续反馈，但它并不保证 policy 在 distribution shift 下恢复，也不等于 trajectory optimization。

如果所有历史 chunks 都因为视觉误判而预测错，平均它们不会自动得到正确动作。

它解决的是 overlapping chunk execution 的具体问题，而不是所有 long-horizon robot-control 问题。

## Sources

- Zhao et al., **Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware**, 2023. https://arxiv.org/abs/2304.13705
- Official evaluation implementation: https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py
