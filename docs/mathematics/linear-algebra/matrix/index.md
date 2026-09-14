---
title: "Matrix"
kind: "canonical"
domain: "Mathematics / Linear Algebra"
parent: "Linear Algebra"
canonical: "/mathematics/linear-algebra/matrix/"
prerequisites:
  - "/mathematics/linear-algebra/vector/"
related:
  - "/deep-learning/core/linear-layer/"
  - "/deep-learning/transformer/attention/scaled-dot-product-attention/"
---

# Matrix

矩阵（matrix）是按行和列排列的数值数组。在线性代数中，它可以表示线性变换；在神经网络中，权重、一个 batch 的特征以及 attention 中的 $Q,K,V$ 都经常写成矩阵。

## 定义

一个 $m\times n$ 的实矩阵写作

\[
A\in\mathbb{R}^{m\times n}.
\]

$m$ 是行数，$n$ 是列数。元素 $A_{ij}$ 位于第 $i$ 行、第 $j$ 列。

如果

\[
\mathbf{x}\in\mathbb{R}^{n},\qquad A\in\mathbb{R}^{m\times n},
\]

那么矩阵向量乘法得到

\[
A\mathbf{x}\in\mathbb{R}^{m}.
\]

第 $i$ 个输出分量是

\[
(A\mathbf{x})_i=\sum_{j=1}^{n}A_{ij}x_j.
\]

也就是说，每个输出分量都是输入向量各分量的加权和。

## 矩阵乘法

若

\[
A\in\mathbb{R}^{m\times n},\qquad B\in\mathbb{R}^{n\times p},
\]

则

\[
C=AB\in\mathbb{R}^{m\times p},
\]

并且

\[
C_{ij}=\sum_{k=1}^{n}A_{ik}B_{kj}.
\]

中间维度必须一致，因为 $A$ 的每一行要与 $B$ 的每一列做一次 [Dot Product](/mathematics/linear-algebra/dot-product/)。

## 转置

矩阵转置 $A^\top$ 会交换行和列：

\[
(A^\top)_{ij}=A_{ji}.
\]

Transformer 中的 $QK^\top$ 正是利用转置，让每个 query 与每个 key 两两计算内积，从而一次得到完整的 attention score matrix。

## Shape 是公式的一部分

在深度学习里，只看符号而不看 shape 很容易误解计算。例如

\[
Q\in\mathbb{R}^{n_q\times d_k},\qquad
K\in\mathbb{R}^{n_k\times d_k}
\]

意味着

\[
QK^\top\in\mathbb{R}^{n_q\times n_k}.
\]

输出的第 $(i,j)$ 个元素表示第 $i$ 个 query 与第 $j$ 个 key 的匹配分数。这个 shape 关系是理解 attention 的关键部分，而不是实现细节。
