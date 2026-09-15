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
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Multilayer Perceptron

> **知识边界**：本文的 canonical 对象是 **Multilayer Perceptron**。依赖机制由 [Linear Layer](/deep-learning/core/linear-layer/)、[Activation Function](/deep-learning/core/activation-function/) 的 canonical page 定义；本文只在当前语境中调用其接口。


Multilayer Perceptron（MLP）是由多层 affine transformations 与 nonlinear activation functions 组成的前馈神经网络。对两层 MLP，常写成

\[
h=\phi(W_1x+b_1),
\]

\[
y=W_2h+b_2.
\]

其中 $h$ 是 hidden representation，$\phi$ 提供非线性。

## Feed-Forward Structure

MLP 的计算图没有循环反馈：信息从输入依次经过 hidden layers 到达输出。

一个深度为 $L$ 的 MLP 可写为

\[
h^{(0)}=x,
\]

\[
h^{(l)}=\phi_l(W_lh^{(l-1)}+b_l),
\qquad l=1,\ldots,L-1,
\]

最后一层根据任务需要产生 output。

如果删除所有 nonlinear activations，多层 affine mappings 会合并成单个 affine mapping，因此 nonlinearity 是 MLP 表达复杂函数的关键。

## Width and Depth

**Width** 描述 hidden layer 的 feature dimension，**depth** 描述连续 learned transformations 的层数。

例如

\[
\mathbb R^{128}
\rightarrow
\mathbb R^{512}
\rightarrow
\mathbb R^{512}
\rightarrow
\mathbb R^{10}
\]

表示两个 512-dimensional hidden stages 与一个 10-dimensional output。

增加 width 与增加 depth 都可能提升 representation capacity，但会以不同方式改变参数量、optimization difficulty 与 function composition structure。

## Hidden Representations

Hidden units 不需要对应人工预先定义的概念。网络只受到 training objective、data distribution 与 architecture constraints 的约束，因此 hidden representation 的结构是学习结果。

对

\[
h=\phi(Wx+b),
\]

weight matrix 先组合输入 features，activation 再施加非线性。多层重复后，后层可以基于前层构造的 features 形成更复杂的函数表示。

## Universal Approximation

经典 universal approximation results 表明，在一定 activation 与 regularity 条件下，具有足够 hidden units 的前馈网络可以在紧致域上逼近广泛的连续函数。

这些结果说明 MLP 具有很强的表示能力，但不保证：

- 需要的宽度很小；
- gradient-based optimization 一定容易找到目标函数；
- 有限数据下具有良好 generalization；
- 某个具体 MLP 是给定任务最有效的 architecture。

表达能力与可训练性、sample efficiency 是不同问题。

## Output Layer

MLP 的最后一层由建模对象决定。例如：

- categorical classification：输出 logits，再接 Softmax / cross-entropy；
- binary classification：输出 scalar logit；
- regression：直接输出 real-valued vector；
- positive parameter：使用合适的 positive transformation。

因此“MLP 必须在输出层使用某种固定 activation”不是定义的一部分。

## High-Dimensional Inputs

Linear Layer 通常作用于 tensor 的最后一个 feature dimension，因此 MLP 也可以直接应用于

\[
X\in\mathbb R^{B\times N\times d}.
\]

只要每个 $(b,n)$ 位置共享同一组 MLP parameters，就会得到 position-wise transformation，而不在不同 positions 之间交换信息。

## Optimization and Regularization

MLP 的训练行为取决于 activation、initialization、normalization、optimizer 与 regularization。常见 regularization 包括 weight decay、dropout、early stopping 等。

深层 MLP 也可能遇到 vanishing / exploding gradients，因此 residual connections、normalization 与现代 initialization methods 常用于更大网络。

## Relation to Transformer FFN

Transformer 的 [Position-Wise Feed-Forward Network](/deep-learning/transformer/position-wise-feed-forward-network/) 是 MLP 的具体应用：同一组 MLP parameters 独立作用于 sequence 中每个 token position。

MLP 是更一般的 feed-forward neural network；Position-Wise FFN 则规定了它在 Transformer layer 中的输入组织、共享方式与 dimension expansion pattern。

## Sources

- Goodfellow, Bengio, Courville. *Deep Learning*, Chapter 6.
- Cybenko. *Approximation by Superpositions of a Sigmoidal Function*. 1989.
- Hornik, Stinchcombe, White. *Multilayer Feedforward Networks Are Universal Approximators*. 1989.
