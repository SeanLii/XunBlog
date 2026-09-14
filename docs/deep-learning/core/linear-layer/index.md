---
title: "Linear Layer"
kind: "canonical"
domain: "Deep Learning / Core"
parent: "Deep Learning"
canonical: "/deep-learning/core/linear-layer/"
prerequisites:
  - "/mathematics/linear-algebra/vector/"
  - "/mathematics/linear-algebra/matrix/"
related:
  - "/deep-learning/core/embedding/"
  - "/robot-learning/act/architecture/"
---

# Linear Layer

Linear Layer 把一个输入向量通过可学习的矩阵和偏置映射到另一个向量空间：

\[
y=Wx+b.
\]

这是神经网络中最基础的可学习变换之一。

## 输入和输出

若

\[
x\in\mathbb R^{d_{in}},
\]

希望输出

\[
y\in\mathbb R^{d_{out}},
\]

则参数 shape 为

\[
W\in\mathbb R^{d_{out}\times d_{in}},
\qquad
b\in\mathbb R^{d_{out}}.
\]

每个输出维度都是输入所有维度的加权和再加偏置。

## 线性变换的作用

矩阵 $W$ 可以改变表示的维度、旋转/拉伸表示空间，并重新组合已有 features。

例如一个 14 维机器人 joint vector 可以被映射成 512 维 hidden representation：

\[
\mathbb R^{14}\rightarrow\mathbb R^{512}.
\]

这不意味着凭空增加了 498 个独立信息源，而是把原来的信息重新编码到一个更适合后续网络使用的 learned feature space 中。

## Batch 形式

如果有 batch：

\[
X\in\mathbb R^{B\times d_{in}},
\]

则输出

\[
Y=XW^\top+b
\in\mathbb R^{B\times d_{out}}.
\]

深度学习框架中的 `Linear(in_features, out_features)` 本质就是这类运算。

## Linear Layer 本身不能表示任意非线性关系

连续堆很多纯 linear layers：

\[
W_3(W_2(W_1x))
\]

仍可以合并成一个大矩阵乘法，因此仍是线性变换。

神经网络通常在 linear layers 之间加入 ReLU、GELU 等非线性 activation，才能表示更复杂函数。

## 在 Transformer 与 ACT 中的位置

Transformer 中 Q/K/V 都通过 linear projections 得到：

\[
Q=XW_Q,
K=XW_K,
V=XW_V.
\]

ACT 也大量使用 linear layers：把 qpos 投影到 hidden dimension、把 latent $z$ 投影到 hidden dimension、把 decoder hidden states 映射为 14 维 actions。

所以很多“模块连接”本质上就是在不同 representation spaces 之间做 learned linear projection。
