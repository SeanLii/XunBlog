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

Matrix 是按行和列排列的一组数：

\[
A=
\begin{bmatrix}
a_{11}&a_{12}&\cdots\\
a_{21}&a_{22}&\cdots\\
\vdots&\vdots&
\end{bmatrix}
\in\mathbb R^{m\times n}.
\]

在深度学习里，Matrix 最常见的两个角色是：**装一批 vectors**，以及**表示 linear transformation**。

## 一批 vectors

如果有 $m$ 个 $n$ 维 vectors，把它们按行排列：

\[
X\in\mathbb R^{m\times n}.
\]

Transformer 的 token matrix 就是这种形式：

```text
row 1 = token 1 hidden vector
row 2 = token 2 hidden vector
...
row m = token m hidden vector
```

## Matrix 作为线性变换

若

\[
x\in\mathbb R^n,
\qquad
W\in\mathbb R^{m\times n},
\]

则

\[
y=Wx\in\mathbb R^m.
\]

矩阵 $W$ 把 $n$ 维向量映射到 $m$ 维空间。

神经网络中的 Linear Layer、Q/K/V projections 都使用这类矩阵乘法。

## Matrix Multiplication

若

\[
A\in\mathbb R^{m\times n},
\qquad
B\in\mathbb R^{n\times p},
\]

则

\[
C=AB\in\mathbb R^{m\times p}.
\]

元素

\[
C_{ij}
=
\sum_{k=1}^{n}A_{ik}B_{kj}.
\]

内侧 dimension $n$ 必须相同。

## Attention 中的 QKᵀ

如果

\[
Q\in\mathbb R^{n_q\times d_k},
\qquad
K\in\mathbb R^{n_k\times d_k},
\]

则

\[
K^\top\in\mathbb R^{d_k\times n_k},
\]

于是

\[
QK^\top
\in\mathbb R^{n_q\times n_k}.
\]

结果的每一个元素都是一个 query vector 与一个 key vector 的 dot product。

因此 Matrix Multiplication 可以一次并行完成所有 query-key pairs 的相似度计算。

## Shape 是理解 Matrix 的核心工具

在 AI 模型中，看懂矩阵往往不是先问“每个数字是多少”，而是先问：

```text
这个 axis 表示 batch？
这个 axis 表示 token？
这个 axis 表示 hidden dimension？
```

只要 shape 与每个 axis 的语义清楚，很多复杂 tensor 运算就能还原成普通 matrix/vector operations。
