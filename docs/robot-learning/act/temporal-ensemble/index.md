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

Temporal Ensemble 是 ACT 在推理阶段融合重叠 action chunks 的方法。它不是对相邻时间的已执行动作做普通平滑，而是把**不同 query 时刻对同一个目标 timestep 的预测**加权平均。

## Overlapping Predictions

设 chunk size 为 $k$。在 timestep $s$ query policy，会得到

\[
\hat A_s
=
(\hat a_{s|s},\hat a_{s+1|s},\ldots,
\hat a_{s+k-1|s}).
\]

如果每个 timestep 都 query，那么目标时刻 $t$ 可能同时拥有：

\[
\hat a_{t|t},
\hat a_{t|t-1},
\hat a_{t|t-2},
\ldots
\]

最多约 $k$ 个有效预测。它们预测的是同一个 $a_t$，但依据的是不同时间取得的 observations。

## Weighted Ensemble

把当前 timestep 的有效 predictions 按时间顺序记为

\[
A_t[0],A_t[1],\ldots,A_t[n_t-1],
\]

论文规定 $A_t[0]$ 对应最老的预测，并使用

\[
w_i=\exp(-mi).
\]

最终执行动作是

\[
a_t
=
\frac{\sum_i w_iA_t[i]}
{\sum_i w_i}.
\]

$m>0$ 时，索引越大的较新 prediction 权重越小，因此**较老 prediction 权重更高**。论文说明较小的 $m$ 会更快纳入新 observation；从公式看，$m$ 越小，权重衰减越慢，新 predictions 与旧 predictions 的权重差距越小。

## 与普通 Smoothing 的区别

普通 temporal smoothing 可能把

\[
a_{t-1},a_t,a_{t+1}
\]

这些不同目标时刻的动作混合。这样可能改变轨迹时序，引入 bias。

ACT 的 ensemble 只混合

\[
\hat a_{t|s_1},\hat a_{t|s_2},\ldots
\]

也就是所有“目标都是 timestep $t$”的预测。目标时刻不变，变化的只是 prediction 所依据的 observation time。

## Feedback 与 Consistency

较老 chunk 提供跨时间的一致计划；较新的 chunk 包含更新后的 visual feedback。Temporal ensemble 把两者连续混合，而不是在 chunk boundary 突然从旧计划跳到新计划。

因此它同时处理两个实际问题：

- action chunking 带来的 chunk-boundary discontinuity；
- 只按 chunk 执行时 observation update 太慢的问题。

## Released Implementation

官方 `imitate_episodes.py` 在启用 `--temporal_agg` 后把 `query_frequency` 设为 1，也就是每个 timestep 都 query policy。代码为每个 query time 保存完整 action chunk，再抽取当前 timestep 的所有已填充 predictions。

当前代码把指数系数固定为

\[
m=0.01,
\]

对应 `np.exp(-0.01 * arange(...))`，随后归一化并求加权和。

这个 $0.01$ 是 released implementation 的具体数值，不属于 Temporal Ensemble 的定义。

## Computational Cost

Temporal ensemble 不改变 training objective，也不需要额外训练分支。代价主要出现在 inference：如果不做 temporal ensemble，可以每 $k$ 步 query 一次 policy；启用后则每一步都运行 policy，因此计算量上升。

## Sources

- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware — Zhao et al., 2023](https://arxiv.org/abs/2304.13705)
- [ACT official implementation — tonyzhaozh/act](https://github.com/tonyzhaozh/act)
