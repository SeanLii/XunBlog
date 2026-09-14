---
title: "Action Chunking"
kind: "canonical"
domain: "Robot Learning / ACT"
parent: "ACT"
canonical: "/robot-learning/act/action-chunking/"
prerequisites:
  - "/robot-learning/behavior-cloning/"
related:
  - "/robot-learning/act/temporal-ensemble/"
  - "/robot-learning/act/inference/"
---

# Action Chunking

Action Chunking 把 policy 的输出从单个下一步动作改成一段未来动作序列。ACT 使用它降低高频 manipulation task 在 policy prediction 层面的 effective horizon，并直接建模连续动作之间的局部时间结构。

## Definition

single-step policy 预测

\[
\hat a_t=\pi_\theta(o_t).
\]

chunked policy 一次输出

\[
\hat A_t
=
(\hat a_{t|t},\hat a_{t+1|t},\ldots,
\hat a_{t+k-1|t}),
\]

其中下标 $u|t$ 表示“在 timestep $t$ 根据 $o_t$ 对 timestep $u$ 做出的预测”。因此

\[
\hat A_t\in\mathbb R^{k\times d_a}.
\]

ACT 原论文中 $d_a=14$，默认 $k=100$，所以单次 policy output 是

\[
100\times14.
\]

## Effective Horizon

若一个 episode 有 $T$ 个 low-level timesteps，single-step policy 需要做大约 $T$ 次独立的 next-action prediction。最朴素的 chunk execution 每次预测 $k$ 个动作并连续执行，可以把 chunk-level decision count 降到大约

\[
\frac{T}{k}.
\]

论文把这称为 $k$-fold reduction in effective horizon。

这里的“effective horizon”不是说物理任务真的只剩 $T/k$ 个 timestep，也不是说低层控制频率下降。它描述的是高层 learned policy 一次承担更长的 action sequence prediction，从而减少反复做单步行为选择的次数。

## Compounding Error

在 [Behavior Cloning](/robot-learning/behavior-cloning/) 中，单步错误会改变后续 observation distribution。Action chunking 的思路不是收集新的 recovery labels，而是减少连续任务被切成多少次 policy-level predictions。

如果一个局部操作需要多步协调，模型可以直接学习这一整段结构，而不是每一步都重新决定下一动作。论文实验中，$k=1$ 对应没有 chunking；在其四个 ablation settings 的平均结果里，success rate 从 $k=1$ 的约 1% 提升到 $k=100$ 的约 44%，之后更大的 $k$ 略有下降。

这个结果是 ACT 论文特定实验上的经验结果，不是对所有任务的单调定律。

## Temporal Structure

Human demonstrations 可能包含 temporally correlated behavior。例如同一个局部状态附近，operator 可能停顿一段时间。只看单步 observation 的 Markovian policy 很难知道“现在是停顿的第几步”。

Chunked output 可以直接表示

\[
(a_t,a_{t+1},\ldots,a_{t+k-1})
\]

之间的时间关系，因此当这类变化发生在一个 chunk 内时，模型可以把它作为 sequence pattern 学习。

## Chunk Size Trade-off

$k$ 太小时，模型接近 single-step BC，effective horizon reduction 很弱。

$k$ 太大时，一次预测需要覆盖很长未来。越远的动作越依赖尚未发生的环境变化，因此 prediction difficulty 增加，执行也更容易变成 open-loop。论文把 $k=\text{episode length}$ 描述为极端 fully open-loop 情况：只根据初始 observation 输出整集动作。

因此 $k$ 控制的是两件互相影响的事：

- 更大的 $k$ 减少 policy-level horizon，并提供更长 temporal context；
- 更大的 $k$ 也要求预测更远未来，降低对新 observation 的即时依赖。

## From Chunking to Temporal Ensemble

最朴素方式是每 $k$ 步才重新 query policy，但这样新的 observation 只在 chunk boundary 被使用，两个 chunks 的交界也可能产生动作突变。

ACT 最终执行方式因此不是简单的“预测 100 步然后闭眼执行 100 步”。启用 [Temporal Ensemble](/robot-learning/act/temporal-ensemble/) 时，policy 每一步都重新预测一个 chunk，再把多个 chunks 对当前 timestep 的预测融合。这使 action chunking 与 closed-loop feedback 同时存在。

## Sources

- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware — Zhao et al., 2023](https://arxiv.org/abs/2304.13705)
