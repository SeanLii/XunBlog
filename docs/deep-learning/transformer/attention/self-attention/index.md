---
title: "Self-Attention"
kind: "canonical"
domain: "Deep Learning / Transformer / Attention"
parent: "Attention"
canonical: "/deep-learning/transformer/attention/self-attention/"
prerequisites:
  - "/deep-learning/transformer/attention/scaled-dot-product-attention/"
related:
  - "/deep-learning/transformer/transformer-encoder/"
  - "/deep-learning/transformer/attention/cross-attention/"
---

# Self-Attention

Self-Attention 指 query、key 和 value 都来自同一组输入表示。它让序列中的每个位置根据当前内容动态读取同一序列中的其他位置。

## 计算

给定

\[
X\in\mathbb R^{n\times d_{model}},
\]

先计算

\[
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
\]

然后

\[
O=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V.
\]

因为 $Q$ 与 $K$ 都有 $n$ 个位置，score matrix 的 shape 是

\[
n\times n.
\]

第 $i$ 行说明第 $i$ 个位置如何从整个输入序列读取信息。

## “Self” 的含义

Self 并不意味着每个位置只看自己。它表示查询者与被查询的信息来自同一个 sequence。是否可以访问未来位置，则由 mask 决定。

Encoder self-attention 通常允许任意位置互相读取。自回归 decoder 的 masked self-attention 会用 [Causal Mask](/deep-learning/transformer/causal-mask/) 阻止当前位置读取未来 token。

## 顺序信息

如果不额外加入位置信息，纯 self-attention 对输入位置的排列没有足够的顺序感知能力。Transformer 因此结合 [Positional Encoding](/deep-learning/transformer/positional-encoding/) 或其他位置表示，让模型知道各 token 的位置关系。

## ACT 中的使用

ACT 的 CVAE encoder 和 observation Transformer encoder 都使用 self-attention 来融合各自输入序列中的信息。但 ACT 的 action generation 不是标准语言模型式的逐 token 自回归生成；其 decoder queries 同时代表多个 action slots，这一点属于 ACT 的具体架构，而不是 Self-Attention 的定义。

## Sources

- [Attention Is All You Need — Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
