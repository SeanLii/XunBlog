---
title: "Dot Product"
kind: "canonical"
domain: "Mathematics / Linear Algebra"
parent: "Linear Algebra"
canonical: "/mathematics/linear-algebra/dot-product/"
prerequisites:
  - "/mathematics/linear-algebra/vector/"
related:
  - "/deep-learning/transformer/attention/scaled-dot-product-attention/"
---

# Dot Product

Dot Product 把两个同维向量变成一个 scalar：

\[
x\cdot y
=x^\top y
=
\sum_{i=1}^{d}x_i y_i.
\]

例如

\[
x=[1,2],\qquad y=[3,4],
\]

则

\[
x\cdot y=1\times3+2\times4=11.
\]

## 几何意义

Dot product 也可以写成

\[
x\cdot y
=\|x\|\|y\|\cos\theta.
\]

其中 $\theta$ 是两个 vectors 的夹角。

所以它同时受两件事影响：

- vectors 的长度；
- vectors 的方向是否一致。

若两个 unit vectors 完全同方向，dot product 为 1；垂直时为 0；反方向时为 -1。

## Dot Product 不是纯粹的“相似度”

因为长度也会影响结果，一个 norm 很大的向量即使角度没有特别接近，也可能产生较大 dot product。

如果只想比较方向，常使用 cosine similarity：

\[
\cos\theta
=\frac{x\cdot y}{\|x\|\|y\|}.
\]

Transformer attention 使用的是 learned Q/K vectors 的 dot product，而不是自动归一化后的 cosine similarity。

## QKᵀ 作为并行 Pairwise Dot Products

把 queries 按行组成矩阵 $Q$，keys 按行组成 $K$：

\[
QK^\top.
\]

其中第 $(i,j)$ 个元素正好是

\[
q_i\cdot k_j.
\]

因此一个 matrix multiplication 就得到所有 query-key pairs 的 scores。

## 高维下的尺度问题

如果 vector 各维独立、均值 0、方差约 1，那么 $d$ 个乘积相加后，dot-product variance 会随 $d$ 增大。

这就是 Transformer [Scaled Dot-Product Attention](/deep-learning/transformer/attention/scaled-dot-product-attention/) 再除以

\[
\sqrt{d_k}
\]

的原因之一：控制 logits 的尺度，避免 softmax 过早进入非常尖锐的区域。
