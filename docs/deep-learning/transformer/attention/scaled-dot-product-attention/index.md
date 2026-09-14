---
title: "Scaled Dot-Product Attention"
kind: "canonical"
domain: "Deep Learning / Transformer / Attention"
parent: "Attention"
canonical: "/deep-learning/transformer/attention/scaled-dot-product-attention/"
prerequisites:
  - "/deep-learning/transformer/attention/qkv/"
  - "/mathematics/linear-algebra/dot-product/"
  - "/deep-learning/core/softmax/"
related:
  - "/deep-learning/transformer/attention/multi-head-attention/"
---

# Scaled Dot-Product Attention

Scaled Dot-Product Attention 是原始 Transformer 使用的基础 attention 运算。它用 query-key dot product 产生分数，除以 $\sqrt{d_k}$ 控制数值尺度，再用 softmax 得到 value 的聚合权重。

## 公式

设

\[
Q\in\mathbb R^{n_q\times d_k},
\qquad
K\in\mathbb R^{n_k\times d_k},
\qquad
V\in\mathbb R^{n_k\times d_v}.
\]

attention 定义为

\[
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V.
\]

计算可以拆成四步。

### Compatibility scores

\[
S=QK^\top
\in\mathbb R^{n_q\times n_k}.
\]

$S_{ij}=q_i\cdot k_j$，表示第 $i$ 个 query 与第 $j$ 个 key 的匹配分数。

### Scaling

\[
\tilde S=\frac{S}{\sqrt{d_k}}.
\]

假设 query 和 key 各分量独立、均值为 0、方差为 1，则点积

\[
q\cdot k=\sum_{r=1}^{d_k}q_rk_r
\]

的方差会随 $d_k$ 增长到约 $d_k$。除以 $\sqrt{d_k}$ 后，尺度被拉回到更稳定的量级。

原论文给出的动机是：当 $d_k$ 较大时，未缩放点积可能变得很大，使 softmax 进入梯度很小的区域。Scaling 的目的不是改变排序，而是控制送入 softmax 的数值尺度。

### Normalization

沿 key dimension 做 softmax：

\[
A=\operatorname{softmax}(\tilde S).
\]

于是

\[
A\in\mathbb R^{n_q\times n_k},
\]

每一行是一组针对某个 query 的归一化权重。

### Value aggregation

\[
O=AV
\in\mathbb R^{n_q\times d_v}.
\]

对第 $i$ 个 query，输出是

\[
o_i=\sum_{j=1}^{n_k}A_{ij}v_j.
\]

所以 score matrix 决定“从哪里取”，$V$ 决定“取到什么”。

## Mask 的位置

如果某些 key 不允许被关注，通常在 softmax 前把对应 score 设成一个非常大的负数。这样 softmax 后对应权重接近 0。Padding mask 与 causal mask 都可以利用这一步实现，但它们屏蔽位置的原因不同。

## Sources

- [Attention Is All You Need — Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
