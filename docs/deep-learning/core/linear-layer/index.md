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

Linear Layer 是神经网络中最基本的可学习变换之一。它把输入向量乘以权重矩阵，再加上偏置，从一个特征空间映射到另一个特征空间。

## 定义

对输入

\[
\mathbf x\in\mathbb R^{d_{in}},
\]

线性层通常写成

\[
\mathbf y=W\mathbf x+\mathbf b,
\]

其中

\[
W\in\mathbb R^{d_{out}\times d_{in}},
\qquad
\mathbf b\in\mathbb R^{d_{out}}.
\]

因此输出

\[
\mathbf y\in\mathbb R^{d_{out}}.
\]

严格来说，只要包含非零偏置 $\mathbf b$，这个变换在数学上是 affine transformation；深度学习库仍普遍把它命名为 Linear layer。

## 可学习参数

$W$ 和 $\mathbf b$ 都由训练数据学习。网络不是提前规定“第 17 个输出维度应该代表什么”，而是通过损失函数调整整个映射，使输出表示对任务有用。

如果一批 token 写成矩阵

\[
X\in\mathbb R^{B\times d_{in}},
\]

常见实现形式是

\[
Y=XW^\top+b,
\]

得到 $B\times d_{out}$ 的输出。

## 在 Transformer 与 ACT 中的作用

Transformer 用不同的 linear projection 生成 $Q,K,V$。ACT 还用 linear layer 把 14 维 joint position、32 维 latent $z$ 等不同来源的数据投影到统一的 hidden dimension，再交给 Transformer 处理。

“投影到同一维度”并不意味着这些输入变成同一种数据；它只是让它们能够在同一个向量空间里参与后续 attention 和 feature fusion。
