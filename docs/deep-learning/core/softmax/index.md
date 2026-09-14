---
title: "Softmax"
kind: "canonical"
domain: "Deep Learning / Core"
parent: "Deep Learning"
canonical: "/deep-learning/core/softmax/"
prerequisites:
  - "/mathematics/linear-algebra/vector/"
related:
  - "/deep-learning/transformer/attention/scaled-dot-product-attention/"
---

# Softmax

Softmax 把一组任意实数转换成非负且总和为 1 的数。它常被用来把一组相对分数变成归一化权重。

## 定义

给定

\[
\mathbf s=(s_1,\ldots,s_n),
\]

softmax 的第 $i$ 个输出为

\[
\operatorname{softmax}(\mathbf s)_i
=\frac{e^{s_i}}{\sum_{j=1}^{n}e^{s_j}}.
\]

因此

\[
0<\operatorname{softmax}(\mathbf s)_i<1,
\qquad
\sum_i\operatorname{softmax}(\mathbf s)_i=1.
\]

## 相对差异

Softmax 对所有输入同时加上同一个常数不敏感：

\[
\operatorname{softmax}(\mathbf s+c)=
\operatorname{softmax}(\mathbf s).
\]

所以它真正利用的是分数之间的相对差异，而不是绝对零点。

指数函数会放大分数差异。若一个 score 比另一个大很多，它获得的权重会迅速接近 1；反过来，较小 score 的权重会接近 0。

## 数值稳定性

直接计算 $e^{s_i}$ 可能溢出。实现中通常先减去最大值 $m=\max_i s_i$：

\[
\frac{e^{s_i-m}}{\sum_j e^{s_j-m}}.
\]

由于所有分数都减去同一个常数，结果不变，但指数输入不会出现不必要的大正数。

## 在 Attention 中的作用

[Scaled Dot-Product Attention](/deep-learning/transformer/attention/scaled-dot-product-attention/) 先用 query 与 key 得到 compatibility scores，再沿 key 维度使用 softmax：

\[
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right).
\]

矩阵 $A$ 的每一行总和为 1。随后用 $AV$ 对 value 做加权和。Softmax 在这里不是判断“对或错”，而是在多个 value 之间分配相对权重。
