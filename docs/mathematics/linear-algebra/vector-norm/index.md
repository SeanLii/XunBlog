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

Vector Norm 用一个非负 scalar 描述 vector 的“大小”。最熟悉的是二维或三维空间里的几何长度，但 norm 是一个更一般的概念。

对于 $x\in\mathbb R^d$，Euclidean norm，也就是 $L_2$ norm：

\[
\|x\|_2=\sqrt{\sum_{i=1}^{d}x_i^2}.
\]

例如

\[
x=[3,4],
\]

则

\[
\|x\|_2=5.
\]

## Norm 必须满足的性质

一个函数 $\|\cdot\|$ 要成为 norm，需要满足三类核心性质。

第一，非负并且只有零向量长度为零：

\[
\|x\|\ge 0,
\qquad
\|x\|=0\iff x=0.
\]

第二，数乘会按绝对值缩放长度：

\[
\|cx\|=|c|\|x\|.
\]

第三，满足 triangle inequality：

\[
\|x+y\|\le \|x\|+\|y\|.
\]

最后这条就是“走两段路不会比直接从起点到终点更短”的抽象形式。

## L1、L2 与一般 Lp Norm

更一般地，$p\ge1$ 时：

\[
\|x\|_p=
\left(\sum_i |x_i|^p\right)^{1/p}.
\]

其中：

\[
\|x\|_1=\sum_i |x_i|,
\]

\[
\|x\|_2=\sqrt{\sum_i x_i^2}.
\]

它们都在衡量“大小”，但几何形状与优化性质不同。L1 对每个分量的绝对值累加；L2 对大分量的惩罚增长更快。

## Normalization

如果 $x\neq0$，可以构造：

\[
\hat x=\frac{x}{\|x\|_2}.
\]

于是：

\[
\|\hat x\|_2=1.
\]

这叫 unit normalization。它不是 [Layer Normalization](/deep-learning/core/layer-normalization/)；前者把整个 vector 的长度缩放到 1，后者根据 feature mean 与 variance 做标准化，并带可学习的 affine parameters。

## Norm 与 Dot Product

在 Euclidean space 中：

\[
\|x\|_2^2=x^\top x.
\]

因此 [Dot Product](/mathematics/linear-algebra/dot-product/) 同时连接了长度与角度：

\[
x^\top y=\|x\|_2\|y\|_2\cos\theta.
\]

这说明 dot product 较大可能来自两个原因：方向接近，或者 vectors 本身很长。

## 在机器学习中的常见位置

Norm 会出现在很多互不相同的地方：

- weight decay 与参数大小；
- distance / similarity；
- gradient clipping；
- embedding normalization；
- error metrics；
- optimization constraints。

因此 norm 是独立的线性代数概念，而不是 attention 或 embedding 的附属操作。
