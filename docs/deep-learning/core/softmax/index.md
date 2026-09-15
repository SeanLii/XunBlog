---
title: "Softmax"
kind: "canonical"
domain: "Deep Learning / Core"
parent: "Deep Learning"
canonical: "/deep-learning/core/softmax/"
prerequisites:
  - "/mathematics/linear-algebra/vector/"
related:
  - "/mathematics/information-theory/cross-entropy/"
  - "/deep-learning/transformer/scaled-dot-product-attention/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Softmax

Softmax 把一组 arbitrary real-valued scores 转成一组正数，并且总和为 1：

\[
p_i=
\frac{e^{z_i}}
{\sum_{j=1}^{K}e^{z_j}}.
\]

输入 $z\in\mathbb R^K$ 常被叫 logits，输出 $p$ 位于 probability simplex：

\[
p_i>0,
\qquad
\sum_i p_i=1.
\]

因此输出可以解释成 categorical probabilities，也可以解释成 normalized positive weights。

## 一次完整计算

设：

\[
z=[1,2,3].
\]

指数：

\[
e^z\approx[2.72,7.39,20.09].
\]

归一化：

\[
\operatorname{softmax}(z)
\approx[0.09,0.24,0.67].
\]

最大的 logit 得到最大 probability，但其他位置仍然保留非零 mass。

## Softmax 关心相对差异

如果所有 logits 同时加常数 $c$：

\[
\operatorname{softmax}(z+c\mathbf 1)
=
\operatorname{softmax}(z).
\]

因为：

\[
\frac{e^{z_i+c}}
{\sum_j e^{z_j+c}}
=
\frac{e^c e^{z_i}}
{e^c\sum_j e^{z_j}}.
\]

公共 factor 被约掉。

因此 Softmax 没有绝对零点，它只关心 logits 之间的相对差。

## Temperature

可以加入 temperature $T>0$：

\[
p_i=
\frac{e^{z_i/T}}
{\sum_j e^{z_j/T}}.
\]

当 $T<1$，differences 被放大，distribution 更尖锐；当 $T>1$，differences 被压小，distribution 更平坦。

极限上：

- 当最大 logit 唯一时，$T\to0^+$ 会把全部 probability mass 集中到该最大位置，因此趋向对应的 one-hot distribution；
- 若存在多个并列最大 logits，$T\to0^+$ 时 probability mass 会只保留在这些最大位置上，并在它们之间平均分配，而不是收敛到唯一 one-hot vector；
- $T\to\infty$ 时，所有有限 logits 之间的相对差异被压到 0，distribution 趋向 uniform distribution。

因此 temperature 控制的是 logits 差异在归一化前被放大的程度。低 temperature 并不无条件等于“取 argmax”；只有唯一最大值时才得到唯一 one-hot 极限。

## Numerical Stability

直接计算 $e^{z_i}$ 可能 overflow。

利用 translation invariance，可以先减最大 logit：

\[
m=\max_j z_j,
\]

\[
p_i=
\frac{e^{z_i-m}}
{\sum_j e^{z_j-m}}.
\]

最大 exponent 变成 $e^0=1$，其余不大于 1，结果完全不变但数值稳定得多。

## Gradient Structure

Softmax 的 Jacobian：

\[
\frac{\partial p_i}{\partial z_j}
=p_i(\delta_{ij}-p_j).
\]

因此某一个 logit 的变化不仅改变自己的 probability，也会通过 normalization 改变所有其他 positions。

这就是 Softmax 与逐元素 sigmoid 的根本差别：Softmax outputs 之间是耦合的，因为它们共享同一个 denominator。

## 与 Cross-Entropy

分类模型常将 logits 经过 Softmax 得到：

\[
p(y=k\mid x).
\]

再用 [Cross-Entropy](/mathematics/information-theory/cross-entropy/) 比较 predicted distribution 与 target distribution。

实际库通常把 `softmax + log + NLL` 合并成稳定的 cross-entropy implementation，不需要手动先算 Softmax。

## 在 Attention 中

[Scaled Dot-Product Attention](/deep-learning/transformer/scaled-dot-product-attention/) 先得到一组 query-key scores：

\[
s_j=\frac{q^\top k_j}{\sqrt{d_k}},
\]

再：

\[
\alpha_j=\operatorname{softmax}(s)_j.
\]

这里 Softmax 不代表“分类”，只是把多个 matching scores 转成总和为 1 的 reading weights。

因此 Softmax 的核心身份是：**把 relative logits 映射到 probability simplex**。分类和 attention 都是这个机制的应用。
