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
---

# Linear Layer

Linear Layer 是神经网络里最基本的可学习映射之一。它接收一个 feature vector，把它映射到另一个 feature space。

常见形式：

\[
y=Wx+b,
\]

其中：

\[
x\in\mathbb R^{d_{in}},
\qquad
W\in\mathbb R^{d_{out}\times d_{in}},
\qquad
b\in\mathbb R^{d_{out}},
\qquad
y\in\mathbb R^{d_{out}}.
\]

很多深度学习框架把它叫 `Linear`，但只要 $b\neq0$，从严格数学定义看它其实是 affine transformation，而不是纯 [Linear Transformation](/mathematics/linear-algebra/linear-transformation/)。

## 一层到底学了什么

展开第 $j$ 个输出：

\[
y_j=
\sum_{i=1}^{d_{in}}W_{ji}x_i+b_j.
\]

所以每个 output feature 都是所有 input features 的一个 learned weighted sum，再加 bias。

可以把 $W$ 的第 $j$ 行理解为：第 $j$ 个输出 feature“怎样读取输入空间”。

训练的过程就是调整这些 weights 与 biases，使映射更适合任务目标。

## Shape 表达输入空间与输出空间

如果：

\[
d_{in}=4,
\qquad d_{out}=3,
\]

那么：

\[
W\in\mathbb R^{3\times4}.
\]

输入一个 4-dimensional vector，得到 3-dimensional vector。

Linear Layer 因此经常承担 projection 的角色：

```text
old representation space
        ↓
   Linear Layer
        ↓
new representation space
```

它不要求输入输出拥有相同 dimension。

## Batch 与 Sequence

实际模型通常不是一次只处理一个 vector。

如果：

\[
X\in\mathbb R^{B\times d_{in}},
\]

则对 batch 中每个 row 使用同一组 parameters：

\[
Y=XW^\top+b.
\]

得到：

\[
Y\in\mathbb R^{B\times d_{out}}.
\]

对于 sequence：

\[
X\in\mathbb R^{B\times N\times d_{in}},
\]

Linear Layer 通常只作用在最后一个 feature dimension，每个 token 共享同一组 $W,b$：

\[
Y\in\mathbb R^{B\times N\times d_{out}}.
\]

因此它不会自动在 tokens 之间交换信息；它对每个位置执行同一个 feature transformation。

## 多层 Linear Mapping 与 Nonlinearity

假设连续两层都没有 nonlinear activation：

\[
y=W_1x+b_1,
\]

\[
z=W_2y+b_2.
\]

代入：

\[
z=W_2W_1x+(W_2b_1+b_2).
\]

仍然只是一个 affine transformation。

所以无论叠多少个纯 Linear Layers，都可以折叠成一层。

神经网络真正获得 nonlinear function approximation ability，需要在 layers 之间加入 [Activation Function](/deep-learning/core/activation-function/)：

\[
y=\phi(Wx+b).
\]

## 常见角色

Linear Layer 可以承担很多不同角色：

- feature projection；
- dimensionality change；
- classifier output head；
- regression head；
- Transformer Q/K/V projections；
- MLP / FFN 中的 learned transformation；
- multimodal feature alignment。

这些用途共享的是同一个基本机制：learned affine mapping。Linear Layer 不是 Transformer 专属组件，它是现代 neural network 的基础 building block。
