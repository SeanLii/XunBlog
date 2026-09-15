---
title: "Linear Layer"
kind: "canonical"
domain: "Deep Learning / Core"
parent: "Deep Learning"
canonical: "/deep-learning/core/linear-layer/"
prerequisites:
  - "/mathematics/linear-algebra/linear-transformation/"
  - "/mathematics/linear-algebra/matrix/"
related:
  - "/deep-learning/core/activation-function/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Linear Layer

Linear Layer 是神经网络中最基本的可学习 affine mapping。对输入

\[
x\in\mathbb R^{d_{in}},
\]

常见定义为

\[
y=Wx+b,
\]

其中

\[
W\in\mathbb R^{d_{out}\times d_{in}},
\qquad
b\in\mathbb R^{d_{out}},
\qquad
y\in\mathbb R^{d_{out}}.
\]

深度学习框架通常把这一层命名为 `Linear`，但当 $b\neq0$ 时，从严格数学定义看它是 affine transformation，而不是纯 [Linear Transformation](/mathematics/linear-algebra/linear-transformation/)。

## Coordinate Form

第 $j$ 个输出分量为

\[
y_j=\sum_{i=1}^{d_{in}}W_{ji}x_i+b_j.
\]

因此 $W$ 的第 $j$ 行定义了第 $j$ 个输出坐标如何组合全部输入 features；$b_j$ 则允许该坐标整体平移。

训练过程通过任务 loss 调整这些 parameters，而不是预先规定每个 row 对应某个人工语义。

## Input and Output Spaces

Linear Layer 可以改变 representation dimension。例如

\[
d_{in}=4,
\qquad
d_{out}=3
\]

时，

\[
W\in\mathbb R^{3\times4}.
\]

它把 $\mathbb R^4$ 中的表示映射到 $\mathbb R^3$。因此 projection、dimension expansion、dimension reduction 与 output head 都可以由同一数学结构实现。

## Parameter Count

标准 Linear Layer 的参数量为

\[
d_{out}d_{in}+d_{out},
\]

其中第一项来自 weight matrix，第二项来自 bias。若关闭 bias，则只剩

\[
d_{out}d_{in}.
\]

当输入和输出维度很大时，Linear Layers 往往占据模型参数量的重要部分。

## Batch and Sequence Inputs

对 batch

\[
X\in\mathbb R^{B\times d_{in}},
\]

可以写成

\[
Y=XW^\top+b,
\]

得到

\[
Y\in\mathbb R^{B\times d_{out}}.
\]

对 sequence tensor

\[
X\in\mathbb R^{B\times N\times d_{in}},
\]

Linear Layer 通常独立作用于最后一个 feature dimension，并在所有 batch items 与 sequence positions 共享同一组 $W,b$：

\[
Y\in\mathbb R^{B\times N\times d_{out}}.
\]

因此 Linear Layer 本身不会在不同 sequence positions 之间交换信息。

## Rank and Information

若忽略 bias，$W$ 的 rank 决定线性映射最多能保留多少独立方向。

当

\[
d_{out}<d_{in},
\]

或 $W$ 本身 rank-deficient 时，映射可能丢失输入空间中的部分信息。反过来，把维度扩展到更高空间也不会凭空增加输入所包含的信息，只是提供新的坐标表示供后续 nonlinear layers 使用。

## Composition of Linear Layers

连续两层若没有 nonlinear activation：

\[
y=W_1x+b_1,
\]

\[
z=W_2y+b_2,
\]

则

\[
z=W_2W_1x+(W_2b_1+b_2),
\]

仍然只是一个 affine transformation。因此增加纯 Linear Layers 并不会形成一般 nonlinear function。

[Activation Function](/deep-learning/core/activation-function/) 使多层网络不再能折叠为单个 affine mapping。

## Common Roles

同一个 Linear Layer 结构可以承担不同 architecture roles：

- feature projection；
- dimensionality change；
- classifier / regression head；
- Attention 的 Q/K/V projections；
- MLP / FFN 中的 affine transformations；
- multimodal feature alignment；
- latent parameter prediction。

这些用途的差别来自上游表示与下游 objective，而不是 Linear Layer 的数学定义发生变化。
