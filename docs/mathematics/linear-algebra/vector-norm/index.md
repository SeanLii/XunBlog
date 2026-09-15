---
title: "Vector Norm"
kind: "canonical"
domain: "Mathematics / Linear Algebra"
parent: "Linear Algebra"
canonical: "/mathematics/linear-algebra/vector-norm/"
prerequisites:
  - "/mathematics/linear-algebra/vector/"
related:
  - "/mathematics/linear-algebra/dot-product/"
---

# Vector Norm

Vector Norm 是衡量 vector 大小的函数。对 vector space 中的 $x$，norm 记作：

\[
\|x\|.
\]

一个函数 $\|\cdot\|$ 要成为 norm，需要满足三条性质：

1. **positive definiteness**

\[
\|x\|\ge0,
\qquad
\|x\|=0\iff x=0;
\]

2. **absolute homogeneity**

\[
\|\alpha x\|=|\alpha|\|x\|;
\]

3. **triangle inequality**

\[
\|x+y\|\le\|x\|+\|y\|.
\]

## $L_p$ Norm

对：

\[
x=(x_1,\ldots,x_n),
\]

当 $p\ge1$ 时：

\[
\|x\|_p
=
\left(
\sum_{i=1}^{n}|x_i|^p
\right)^{1/p}.
\]

常见情况包括：

### $L_1$ norm

\[
\|x\|_1=\sum_i|x_i|.
\]

### $L_2$ norm

\[
\|x\|_2
=
\sqrt{\sum_i x_i^2}.
\]

这是 Euclidean length。

### $L_\infty$ norm

\[
\|x\|_\infty
=
\max_i|x_i|.
\]

不同 norms 对 coordinates 的变化有不同敏感度，因此会定义不同 geometry。

## Norm 与 Distance

任意 norm 都可以诱导 distance：

\[
d(x,y)=\|x-y\|.
\]

例如 Euclidean distance：

\[
d_2(x,y)=\|x-y\|_2.
\]

因此很多“误差大小”实际上是在某个 norm 所定义的 geometry 中衡量两个 vectors 的差异。

## Unit Vector 与 Normalization

对非零 vector：

\[
\hat x=\frac{x}{\|x\|_2},
\]

则：

\[
\|\hat x\|_2=1.
\]

Normalization 删除 vector 的整体 magnitude，保留其 direction。它常用于 cosine similarity、feature matching 与 representation comparison。

## 与 Dot Product 的关系

在 Euclidean space：

\[
\|x\|_2^2=x^\top x.
\]

进一步：

\[
x^\top y
=
\|x\|_2\|y\|_2\cos\theta.
\]

因此 [Dot Product](/mathematics/linear-algebra/dot-product/) 同时给出 vector length 与 angular relationship。

## Norm Equivalence in Finite Dimensions

有限维 vector space 中，不同 norms 在拓扑意义上等价：对任意两种 norms $\|\cdot\|_a$ 与 $\|\cdot\|_b$，存在只依赖于这两种 norms 和空间维数的常数 $c,C>0$，使所有 $x$ 都满足

\[
c\|x\|_a
\le
\|x\|_b
\le
C\|x\|_a.
\]

这个不等式意味着两种 norms 不会在有限维空间中对“趋近于 0”给出矛盾判断。例如若

\[
\|x_n-x\|_a\to0,
\]

则由上界

\[
\|x_n-x\|_b\le C\|x_n-x\|_a\to0,
\]

所以同一个 sequence 也会在 $\|\cdot\|_b$ 下收敛到 $x$。反方向由另一侧不等式同理成立。

因此 finite-dimensional norm equivalence 说的是：这些 norms 诱导相同的 convergence / continuity topology；它**不**意味着数值相同，也不意味着 optimization 中可以任意互换。不同 norms 仍然定义不同的 unit balls、loss geometry、gradient behavior 与 regularization bias。

## Machine Learning 中的使用

常见例子包括：

- $L_1$ / $L_2$ prediction error；
- parameter regularization；
- gradient clipping；
- embedding normalization；
- distance-based retrieval；
- robustness constraints。

具体选择哪种 norm 应由模型目标与数据结构决定，而不是由“哪个公式更常见”决定。
