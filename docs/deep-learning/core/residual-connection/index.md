---
title: "Residual Connection"
kind: "canonical"
domain: "Deep Learning / Core"
parent: "Deep Learning"
canonical: "/deep-learning/core/residual-connection/"
prerequisites:
  - "/mathematics/linear-algebra/vector/"
related:
  - "/deep-learning/cnn/resnet/"
  - "/deep-learning/transformer/transformer-encoder/"
---

# Residual Connection

Residual Connection 把某个模块的输入直接加到模块输出上。它让网络学习“在原表示上应该增加怎样的修正”，而不必让每一层从头重建完整表示。

## 定义

若一个子网络为 $F$，输入为 $x$，最基本的 residual form 是

\[
y=F(x)+x.
\]

如果 $F(x)$ 与 $x$ 的 shape 不一致，就不能直接相加，需要先通过投影或其他变换让维度匹配。

## Residual Learning

ResNet 原论文把期望映射记作 $H(x)$，并令网络学习

\[
F(x)=H(x)-x.
\]

于是最终输出为

\[
H(x)=F(x)+x.
\]

这不是说网络一定只学很小的修正，而是改变了参数化方式。论文的核心经验结论是，这种 residual formulation 能让更深网络更容易优化。

## Transformer 中的使用

原始 Transformer 在 attention 子层和 feed-forward 子层周围都使用 residual connection，再结合 Layer Normalization。这样每个子层处理的是当前表示的增量，同时原输入保留一条直接路径。

Residual connection 只负责信息与梯度的直接通路；它不等同于 normalization，也不替代 attention 或 convolution。

## Sources

- [Deep Residual Learning for Image Recognition — He et al., 2015](https://arxiv.org/abs/1512.03385)
- [Attention Is All You Need — Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
