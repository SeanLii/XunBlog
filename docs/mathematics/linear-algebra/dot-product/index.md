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

点积（dot product，也称 inner product 在欧氏空间中的标准形式）把两个同维向量映射成一个标量。Transformer 用它比较 query 与 key；几何上，它同时受两个向量的长度和夹角影响。

## 定义

对于

\[
\mathbf{x},\mathbf{y}\in\mathbb{R}^d,
\]

点积定义为

\[
\mathbf{x}\cdot\mathbf{y}
=\mathbf{x}^\top\mathbf{y}
=\sum_{i=1}^{d}x_i y_i.
\]

输入是两个 $d$ 维向量，输出是一个实数。

例如

\[
\mathbf{x}=(1,2),\qquad \mathbf{y}=(3,4),
\]

则

\[
\mathbf{x}\cdot\mathbf{y}=1\times3+2\times4=11.
\]

## 几何意义

点积也可以写成

\[
\mathbf{x}\cdot\mathbf{y}
=\lVert\mathbf{x}\rVert_2\lVert\mathbf{y}\rVert_2\cos\theta,
\]

其中 $\theta$ 是两个向量的夹角。因此，当向量长度固定时，方向越接近，点积越大；方向相反时点积可以为负；正交时点积为零。

这给 attention 提供了一个重要直觉：query 和 key 经过学习得到的投影后，较大的点积可以表示它们在模型学到的特征空间中更匹配。但“点积大”不是天然语义相似，而是由训练出来的表示和投影决定的。

## 从一个点积到一张分数矩阵

如果把多个 query 和 key 分别堆成矩阵

\[
Q\in\mathbb{R}^{n_q\times d_k},\qquad
K\in\mathbb{R}^{n_k\times d_k},
\]

那么

\[
QK^\top\in\mathbb{R}^{n_q\times n_k}.
\]

其中

\[
(QK^\top)_{ij}=q_i\cdot k_j.
\]

这一步同时计算所有 query-key pair 的点积。后续的 [Scaled Dot-Product Attention](/deep-learning/transformer/attention/scaled-dot-product-attention/) 会再除以 $\sqrt{d_k}$，并通过 softmax 把这些分数变成权重。
