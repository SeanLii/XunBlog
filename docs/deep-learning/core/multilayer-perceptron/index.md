---
title: "Multilayer Perceptron"
kind: "canonical"
domain: "Deep Learning / Core"
parent: "Deep Learning"
canonical: "/deep-learning/core/multilayer-perceptron/"
prerequisites:
  - "/deep-learning/core/linear-layer/"
  - "/deep-learning/core/activation-function/"
related:
  - "/deep-learning/transformer/position-wise-feed-forward-network/"
---

# Multilayer Perceptron

Multilayer Perceptron（MLP）是把多个 affine transformation 与 nonlinear activation 串联起来的前馈神经网络。

最简单的两层形式可以写成：

\[
h=\phi(W_1x+b_1),
\]

\[
y=W_2h+b_2.
\]

这里 $x$ 是输入，$h$ 是 hidden representation，$\phi$ 是 activation function。它和单个 Linear Layer 的关键区别不是“层数更多”，而是中间加入了 nonlinearity。

## 从 Linear Mapping 到 Nonlinear Function

如果没有 activation：

\[
y=W_2(W_1x+b_1)+b_2,
\]

两层 affine mapping 仍然可以合并成一层 affine mapping：

\[
y=W'x+b'.
\]

因此单纯堆很多 linear layers 并不会得到更复杂的函数族。MLP 的表达能力来自：

```text
Linear
  ↓
Nonlinearity
  ↓
Linear
  ↓
Nonlinearity
  ↓
...
```

每一层先改变坐标与 feature combination，再由非线性打破“所有层最终仍等价于一个线性映射”的限制。

## Width 与 Depth

MLP 的两个基本尺度是：

- **width**：某层有多少 hidden units；
- **depth**：有多少连续的 learned layers / nonlinear stages。

例如：

\[
x\in\mathbb R^{128}
\rightarrow
h_1\in\mathbb R^{512}
\rightarrow
h_2\in\mathbb R^{512}
\rightarrow
y\in\mathbb R^{10}.
\]

更宽通常提供更大的同层表示容量；更深则允许函数被分解成更多级组合。两者都会影响参数量、优化难度和可表达的函数结构。

## Hidden Representation

MLP 中间层不是必须对应人工定义的语义。

训练时，loss 只约束最终任务目标；hidden units 会形成对任务有用的中间表示：

\[
h=\phi(Wx+b).
\]

因此“hidden layer 学到了什么”取决于数据、objective、architecture 与 optimization，而不是由某个 neuron 的名字预先决定。

## Batch 与高维输入

对 batch：

\[
X\in\mathbb R^{B\times d_{in}},
\]

Linear Layer 可以写成：

\[
H=XW^\top+b.
\]

MLP 只是继续沿最后一个 feature dimension 做变换，所以也可以直接作用于：

\[
X\in\mathbb R^{B\times N\times d}.
\]

此时相同 MLP 被应用到每个 $(B,N)$ 位置。

这正是 Transformer 中 position-wise feed-forward sublayer 的基础，但 MLP 本身远早于 Transformer，也广泛存在于分类器、autoencoder、policy network 与各种 prediction head 中。

## Output Layer 由任务决定

MLP 最后一层是否使用 activation 取决于输出语义。

例如分类 logits 可以直接输出实数：

\[
z=W_oh+b_o,
\]

再由 Softmax 转成概率。

回归任务可能直接输出：

\[
\hat y=W_oh+b_o.
\]

所以“MLP 必须以某个 activation 结束”并不是定义的一部分。

## 与 Transformer FFN 的关系

Transformer 的 [Position-Wise Feed-Forward Network](/deep-learning/transformer/position-wise-feed-forward-network/) 是 MLP 的一个具体使用方式：同一个小型 MLP 独立作用于 sequence 中每个 token position。

MLP 是更一般的神经网络结构；Transformer 中的 Position-Wise Feed-Forward Network 则是把同一个 MLP 独立应用到每个 sequence position 的具体 architecture 子层。

## Sources

- Goodfellow, Bengio, Courville. *Deep Learning*, Chapter 6.
- Cybenko. *Approximation by Superpositions of a Sigmoidal Function*. 1989.
