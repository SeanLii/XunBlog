---
title: "Residual Connection"
kind: "canonical"
domain: "Deep Learning / Convolutional Neural Networks / ResNet"
parent: "ResNet"
canonical: "/deep-learning/cnn/resnet/residual-connection/"
prerequisites:
  - "/deep-learning/cnn/resnet/"
related:
  - "/deep-learning/transformer/"
---

# Residual Connection

Residual Connection 是 ResNet 中最核心的结构之一：

\[
y=x+F(x).
\]

与直接学习完整 mapping：

\[
y=H(x)
\]

相比，ResNet 把 learned branch 写成 residual：

\[
F(x)=H(x)-x.
\]

于是 block 学习的是“相对 identity mapping 需要补充什么”。

## Residual Learning

如果理想 mapping 接近 identity：

\[
H(x)\approx x,
\]

普通 stack 必须通过多层参数主动逼近 identity。

Residual block 则只需要：

\[
F(x)\approx0.
\]

因为：

\[
H(x)=x+F(x).
\]

这让 identity mapping 成为 architecture 中天然存在的 reference path。

## Identity Shortcut

最简单的 residual block：

```text
x ────────────────────┐
│                     │
└→ learned branch F(x)│
                      ↓
                  x + F(x)
                      ↓
                      y
```

上方 shortcut 不需要 learned parameters。

这条 identity path 是 ResNet 与普通深层 CNN 的关键差别。

## Gradient Path

对：

\[
y=x+F(x),
\]

有：

\[
\frac{\partial y}{\partial x}
=
I+
\frac{\partial F}{\partial x}.
\]

因此反向传播包含显式 identity term：

\[
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial y}
\left(
I+
\frac{\partial F}{\partial x}
\right).
\]

这不意味着深层网络从此“绝对不会梯度消失”，但 identity route 明显改变了 optimization path，使极深网络更容易训练。

## Shape 改变时的 Projection Shortcut

直接 addition 要求：

\[
\operatorname{shape}(x)
=
\operatorname{shape}(F(x)).
\]

如果 block 同时改变 channel 数或 spatial resolution，可以加入 projection：

\[
y=P(x)+F(x).
\]

ResNet 常用 $1\times1$ convolution 完成这个 projection。

因此 shortcut 有两类：

- identity shortcut；
- projection shortcut。

更早的网络已经出现过 skip connections，但 ResNet 系统化了 $H(x)=F(x)+x$ 的 residual learning formulation，后来 Transformer 等架构继续复用了这种连接方式。

## Transformer 中的复用

Transformer 的每个 attention / FFN sublayer 周围也使用 residual connection。

例如 post-norm 形式：

\[
y=\operatorname{LN}(x+F(x)),
\]

pre-norm 形式：

\[
y=x+F(\operatorname{LN}(x)).
\]

这是 ResNet residual design 的跨架构复用。

所以 Transformer 页面应链接到这里，但不会重新拥有这套理论。

## Sources

- He et al. *Deep Residual Learning for Image Recognition*. 2015/2016.
- Vaswani et al. *Attention Is All You Need*. 2017.
