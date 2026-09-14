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
---

# Dot Product

Dot Product 接收两个同维 vectors，输出一个 scalar：

\[
x\cdot y
=x^\top y
=\sum_{i=1}^{d}x_i y_i.
\]

例如：

\[
x=[1,2],\qquad y=[3,4],
\]

则：

\[
x\cdot y=1\times3+2\times4=11.
\]

只看这个公式，它像是“对应分量相乘再求和”。但 dot product 真正重要的地方在于，它把**代数计算**和**几何关系**连接了起来。

## 几何形式

在 Euclidean space 中：

\[
x\cdot y
=\|x\|_2\|y\|_2\cos\theta,
\]

其中 $\theta$ 是两个 vectors 的夹角。

因此 dot product 同时包含：

- 两个 vectors 的长度；
- 两个 vectors 的方向关系。

如果两个 vectors 都是 unit vectors：

\[
\|x\|_2=\|y\|_2=1,
\]

则：

\[
x\cdot y=\cos\theta.
\]

这时 dot product 才直接只表示方向接近程度。

## Orthogonality

如果：

\[
x\cdot y=0,
\]

则在 Euclidean geometry 中，两个 vectors orthogonal，也就是垂直。

Orthogonality 在高维空间里仍然成立，不需要我们能画出这些 vectors。

例如：

\[
[1,0,0]\cdot[0,1,0]=0.
\]

更一般地，一组两两 orthogonal 的 vectors 可以形成非常方便的 coordinate directions。

## Projection

Dot product 还能回答：一个 vector 在另一个方向上有多少分量。

若 $u$ 是 unit vector，则 $x$ 在 $u$ 方向上的 scalar component 为：

\[
x\cdot u.
\]

对应的 projected vector 是：

\[
\operatorname{proj}_u(x)=(x\cdot u)u.
\]

如果 $u$ 不是 unit vector：

\[
\operatorname{proj}_u(x)
=\frac{x\cdot u}{u\cdot u}u.
\]

这说明 dot product 并不是专门为了“相似度”设计的，它本质上是 Euclidean geometry 中测量方向关系和分解向量的重要工具。

## Cosine Similarity

因为 raw dot product 同时受长度影响，所以如果只想比较方向，常归一化：

\[
\operatorname{cosine}(x,y)
=\frac{x\cdot y}{\|x\|_2\|y\|_2}.
\]

因此：

- dot product 大，不一定意味着方向非常接近；
- vector 很长，也会放大 dot product；
- cosine similarity 去掉了长度这一因素。

## Matrix multiplication 中的 Dot Product

如果

\[
A\in\mathbb R^{m\times d},
\qquad
B\in\mathbb R^{d\times n},
\]

那么：

\[
(AB)_{ij}
=A_{i,:}\cdot B_{:,j}.
\]

也就是说，matrix multiplication 的每一个输出元素都是一对 row/column 的 dot product。

在 attention 中：

\[
QK^\top
\]

会一次计算所有 query-key pairs 的 dot products。这是 dot product 的一个重要应用，但不是这个概念本身的定义范围。

## 高维随机 vectors 的尺度

假设 $x_i,y_i$ 大致独立、均值为 0、方差有限，那么

\[
x\cdot y=\sum_i x_i y_i
\]

是很多随机项的和。维度增大时，这个和的典型尺度也会增大。

这就是 [Scaled Dot-Product Attention](/deep-learning/transformer/scaled-dot-product-attention/) 需要考虑 $d_k$ 的背景：attention 不是因为 dot product “不稳定”才缩放，而是因为高维 dot-product logits 的统计尺度会影响后续 softmax。
