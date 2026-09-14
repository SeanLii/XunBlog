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

Transformer 的 Feed-Forward Network（FFN）是对每个序列位置独立应用的两层非线性网络。Attention 在位置之间交换信息，FFN 则在每个位置内部重新组合特征维度。

## 原始 Transformer 形式

对单个 hidden vector $x\in\mathbb R^{d_{model}}$，原论文写为

\[
\operatorname{FFN}(x)
=\max(0,xW_1+b_1)W_2+b_2.
\]

也就是：

1. 从 $d_{model}$ 投影到更大的中间维度 $d_{ff}$；
2. 使用 ReLU；
3. 再投影回 $d_{model}$。

同一层中的所有 token 使用相同的 $W_1,W_2,b_1,b_2$，但每个 token 独立计算。

## Position-wise 的含义

如果输入 shape 为

\[
(B,N,d_{model}),
\]

FFN 不在 $N$ 个 positions 之间做混合；它只对最后的 feature dimension 做变换，输出 shape 仍是

\[
(B,N,d_{model}).
\]

位置之间的信息交换已经由 attention 完成。

## 与普通 MLP 的关系

从单个 token 看，它就是一个小型 MLP。称为 Feed-Forward Network 是 Transformer 文献中的惯用名称。后续架构可以替换 activation、增加 gating 或改变中间维度，但这些变体不改变原始 FFN 的基本角色。

ACT 使用的 Transformer 继承了这一结构。具体 `dim_feedforward` 是实现超参数，不应写进 FFN 的概念定义。

## Sources

- [Attention Is All You Need — Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
