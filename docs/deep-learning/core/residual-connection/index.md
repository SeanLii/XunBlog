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

Residual Connection 把一个子网络的输入直接加回它的输出：

\[
y=x+F(x).
\]

图上就是一条绕过中间变换的 shortcut：

```text
x ──────────────┐
│               │
↓               │
F(x)            │
│               │
└────── + ←─────┘
        │
        ↓
        y
```

## 它改变了学习目标

如果目标映射是 $H(x)$，普通 block 直接学习

\[
H(x).
\]

Residual block 等价于学习

\[
F(x)=H(x)-x.
\]

如果最合适的变换接近 identity，那么只需要让 $F(x)$ 接近 0，就可以得到

\[
y\approx x.
\]

这是 ResNet 论文提出 residual learning 时的重要动机。

## Gradient 也有直接路径

对

\[
y=x+F(x)
\]

求导：

\[
\frac{\partial y}{\partial x}
=I+\frac{\partial F}{\partial x}.
\]

identity 路径意味着 gradient 不必完全穿过复杂的 $F$ 才能回到前面。这有助于非常深的网络优化。

## Shape 必须兼容

直接相加要求两边 shape 一致。如果 $F(x)$ 改变了 channel/dimension，就需要 projection shortcut 先把 $x$ 变到兼容 shape。

## 在 Transformer 中

Transformer 每个 attention/FFN sublayer 周围都使用 residual path。抽象写成：

\[
x' = x + \operatorname{Attention}(x),
\]

\[
y = x' + \operatorname{FFN}(x').
\]

具体 LayerNorm 是放在相加前还是相加后取决于 pre-norm / post-norm 架构。

## 与 ResNet 的关系

Residual connection 是通用机制；ResNet 是大量使用这种连接构建的 CNN architecture。Transformer 也使用 residual connection，但不因此成为 ResNet。
