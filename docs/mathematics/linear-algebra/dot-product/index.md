---
title: "Dot Product"
kind: "canonical"
domain: "Mathematics / Linear Algebra"
parent: "Linear Algebra"
canonical: "/mathematics/linear-algebra/dot-product/"
prerequisites:
  - "/mathematics/linear-algebra/vector/"
  - "/mathematics/linear-algebra/vector-norm/"
related:
  - "/deep-learning/transformer/scaled-dot-product-attention/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Dot Product

> **知识边界**：本文的 canonical 对象是 **Dot Product**。依赖机制由 [Vector](/mathematics/linear-algebra/vector/)、[Vector Norm](/mathematics/linear-algebra/vector-norm/) 的 canonical page 定义；本文只在当前语境中调用其接口。


Dot Product 是 Euclidean vector space 中最基本的 inner product。对：

\[
x,y\in\mathbb R^n,
\]

定义：

\[
x^\top y
=
\sum_{i=1}^{n}x_i y_i.
\]

结果是一个 scalar。

## Algebraic Properties

Dot product 满足：

### Symmetry

\[
x^\top y=y^\top x.
\]

### Linearity

\[
x^\top(\alpha y+\beta z)
=
\alpha x^\top y+
\beta x^\top z.
\]

### Positive definiteness

\[
x^\top x\ge0,
\]

且：

\[
x^\top x=0\iff x=0.
\]

这些性质使 dot product 定义了 Euclidean geometry。

## Norm

Euclidean norm 可以由 dot product 得到：

\[
\|x\|_2
=
\sqrt{x^\top x}.
\]

因此：

\[
x^\top x=\|x\|_2^2.
\]

## Angle

对非零 vectors：

\[
x^\top y
=
\|x\|_2\|y\|_2\cos\theta.
\]

所以：

\[
\cos\theta
=
\frac{x^\top y}{\|x\|_2\|y\|_2}.
\]

这说明 dot product 同时受 magnitude 与 direction alignment 影响。

## Orthogonality

如果：

\[
x^\top y=0,
\]

则 $x$ 与 $y$ orthogonal。

在 Euclidean geometry 中，这对应夹角：

\[
\theta=90^\circ.
\]

Orthogonality 是 basis construction、projection、least squares 与 many decompositions 的基础。

## Projection

把 $x$ 投影到非零 vector $u$ 的 direction 上：

\[
\operatorname{proj}_u(x)
=
\frac{x^\top u}{u^\top u}u.
\]

若 $u$ 已 normalized：

\[
\|u\|_2=1,
\]

则：

\[
\operatorname{proj}_u(x)
=(x^\top u)u.
\]

这里 dot product 给出 $x$ 沿 $u$ direction 的 signed component。

## Cauchy–Schwarz Inequality

Dot product 满足：

\[
|x^\top y|
\le
\|x\|_2\|y\|_2.
\]

因此 cosine similarity 一定落在：

\[
[-1,1].
\]

等号在两个 vectors linearly dependent 时成立。

## Cosine Similarity

对 nonzero vectors：

\[
\operatorname{cosine}(x,y)
=
\frac{x^\top y}{\|x\|_2\|y\|_2}.
\]

它移除了整体 magnitude，只比较 direction。

Dot-product similarity 与 cosine similarity 因此不是同一个 quantity；前者同时保留 vector norms。

## Matrix Multiplication 中的 Dot Product

若：

\[
A\in\mathbb R^{m\times n},
\qquad
B\in\mathbb R^{n\times p},
\]

则：

\[
(AB)_{ij}
=
A_{i,:}^\top B_{:,j}.
\]

Matrix multiplication 的每个 output entry 都是对应 row 与 column 的 dot product。

## High-Dimensional Scale

若 random vectors 的 coordinates 独立、均值为 0、variance 为 1，则：

\[
x^\top y
=
\sum_{i=1}^{d}x_i y_i
\]

的 variance 通常随 dimension $d$ 增长。

这也是 [Scaled Dot-Product Attention](/deep-learning/transformer/scaled-dot-product-attention/) 中除以 $\sqrt{d_k}$ 的数学背景。

## Connections

- [Vector Norm](/mathematics/linear-algebra/vector-norm/)：Euclidean norm 可由 dot product 定义。
- [Matrix](/mathematics/linear-algebra/matrix/)：matrix multiplication 的基本局部操作。
- [Scaled Dot-Product Attention](/deep-learning/transformer/scaled-dot-product-attention/)：使用 dot products 作为 query-key scores。
