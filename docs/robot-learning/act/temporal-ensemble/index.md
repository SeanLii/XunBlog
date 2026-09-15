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
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Temporal Ensemble

> **知识边界**：本文的 canonical 对象是 **Temporal Ensemble**。依赖机制由 [Action Chunking](/robot-learning/act/action-chunking/) 的 canonical page 定义；本文只在当前语境中调用其接口。


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

因此在执行时刻 $t=2$，至少已经存在三份候选预测：

\[
a_2^{(0)},\quad a_2^{(1)},\quad a_2^{(2)}.
\]

上标表示“这份预测是在什么时候产生的”。

## Direct Execution Strategies

可以永远只用最新预测 $a_t^{(t)}$。这样响应最新视觉最直接，但相邻 timestep 的输出可能发生明显跳动。

也可以只执行最早 chunk 中的计划，直到它结束。这样动作连续，但会失去高频 closed-loop correction。

Temporal Ensemble 试图同时保留两者：

- 继续每一步看新 observation；
- 不让执行动作完全被一次最新预测突然替换。

## 指数加权融合

ACT 对同一执行时刻的多个候选动作做加权平均。对当前执行时刻 $t$，把所有可用预测按**从最旧到最新**排列为

\[
\hat a_t^{[0]},
\hat a_t^{[1]},
\ldots,
\hat a_t^{[n-1]},
\]

其中 $\hat a_t^{[0]}$ 是最早历史 query 对 $a_t$ 的预测，$\hat a_t^{[n-1]}$ 是最新预测。ACT 论文定义

\[
w_i=\exp(-m i),\qquad m>0,
\]

并明确规定 $w_0$ 对应**最旧预测**。归一化后

\[
\tilde w_i=\frac{w_i}{\sum_{j=0}^{n-1}w_j},
\]

最终执行

\[
\bar a_t=\sum_{i=0}^{n-1}\tilde w_i\hat a_t^{[i]}.
\]

因为指数权重随 $i$ 增大而减小，所以在这个索引约定下，**越旧的预测权重越大，越新的预测权重越小**。这并不是代码排列造成的偶然结果，而是论文中明确给出的 temporal-ensemble convention。

参数 $m$ 控制新 observation 被纳入执行动作的速度。若 $m$ 较小，各个历史预测的权重更接近，新预测能够更快地对加权结果产生明显影响；若 $m$ 较大，权重更集中在旧预测上，执行更平滑但对最新 observation 的响应更慢。

官方 released evaluation code 与论文这一时间方向一致：`all_time_actions[:, t]` 按 query time 从早到晚取出当前时刻的已填充预测，再使用

\[
\exp(-0.01\cdot[0,1,2,\ldots])
\]

归一化并加权求和，因此代码中的第一个 populated row 就是权重最大的最旧预测。

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
