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

Softmax 把一组任意实数 scores 转换成一组非负、总和为 1 的数：

\[
\operatorname{softmax}(z_i)
=
\frac{e^{z_i}}{\sum_j e^{z_j}}.
\]

因此输出可以被当作 categorical probabilities，也可以被当作一组 normalized weights。

## 一个数值例子

设 scores 为

\[
[1,2,3].
\]

指数后约为

\[
[e^1,e^2,e^3]
\approx[2.72,7.39,20.09].
\]

除以总和约 30.20：

\[
[0.09,0.24,0.67].
\]

最大的 score 得到最大权重，但其他位置仍保留非零值。

## Softmax 看的是相对差异

如果所有 logits 同时加同一个常数 $c$：

\[
\operatorname{softmax}(z+c)=\operatorname{softmax}(z).
\]

因为分子分母都会多出同一个因子 $e^c$。

所以 Softmax 关心的是 logits 之间的差，而不是它们的绝对基准。

## 数值稳定形式

直接计算很大的 $e^{z_i}$ 可能 overflow。实现通常先减去最大值：

\[
\operatorname{softmax}(z_i)
=
\frac{e^{z_i-m}}{\sum_j e^{z_j-m}},
\qquad
m=\max_j z_j.
\]

由于所有 logits 同时减去 $m$，结果不变，但数值更稳定。

## 在 Attention 中的作用

Attention 先计算 query-key scores：

\[
s_{ij}=\frac{q_i^\top k_j}{\sqrt{d_k}}.
\]

然后对固定 query 的所有 keys 做 softmax：

\[
\alpha_{ij}=\operatorname{softmax}_j(s_{ij}).
\]

于是 $\alpha_{ij}$ 成为一组总和为 1 的读取权重，再用它们加权 Values。

所以 Softmax 在 Attention 中不是“分类器”，而是把任意匹配 scores 归一化为可用于 weighted sum 的权重。
