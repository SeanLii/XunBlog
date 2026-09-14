---
title: "Feed-Forward Network"
kind: "canonical"
domain: "Deep Learning / Transformer"
parent: "Transformer"
canonical: "/deep-learning/transformer/feed-forward-network/"
prerequisites:
  - "/deep-learning/core/linear-layer/"
related:
  - "/deep-learning/transformer/transformer-encoder/"
  - "/deep-learning/transformer/transformer-decoder/"
---

# Feed-Forward Network

Transformer 中的 Feed-Forward Network（FFN）是在每个位置上独立应用的两层非线性网络。

原始 Transformer 写成

\[
\operatorname{FFN}(x)
=
\max(0,xW_1+b_1)W_2+b_2.
\]

也就是：

```text
hidden vector d_model
      │
      ↓ Linear
larger intermediate dimension
      │
      ↓ activation
      │
      ↓ Linear
hidden vector d_model
```

## Attention 与 FFN 的职责不同

Attention 允许位置之间交换信息：

\[
y_i\text{ can depend on }x_j.
\]

FFN 对每个位置使用同一套网络，但位置之间不互相混合：

\[
y_i=\operatorname{FFN}(x_i).
\]

因此可以把一个 Transformer layer 的主节奏记成：

```text
Attention: 从别人那里读信息
FFN:       对读完后的自己做变换
```

## Position-wise 的含义

“position-wise”不是说每个位置有独立参数。恰恰相反，同一个 FFN 参数会被共享到所有 positions。

如果 $X\in\mathbb R^{n\times d}$，FFN 相当于对每一行使用同一个函数 $f$：

\[
Y_i=f(X_i).
\]

## Intermediate Dimension

原始 Transformer Base 使用

\[
d_{model}=512,
\qquad
d_{ff}=2048.
\]

所以第一层先扩维，再压回 model dimension。

这些数值是具体架构配置，不是 FFN 的定义。ACT 论文使用自己的 hidden / feed-forward dimensions。

## Sources

- Vaswani et al., **Attention Is All You Need**, 2017. https://arxiv.org/abs/1706.03762
