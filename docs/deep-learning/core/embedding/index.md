---
title: "Embedding"
kind: "canonical"
domain: "Deep Learning / Core"
parent: "Deep Learning"
canonical: "/deep-learning/core/embedding/"
prerequisites:
  - "/mathematics/linear-algebra/vector/"
related:
  - "/deep-learning/transformer/cls-token/"
  - "/deep-learning/transformer/learnable-query-embedding/"
---

# Embedding

Embedding 是把离散 ID 或离散位置映射成可学习向量的机制。

最典型的形式是一张参数表：

\[
E\in\mathbb R^{N\times d}.
\]

输入一个 index $i$，就取第 $i$ 行：

\[
e_i=E[i]\in\mathbb R^d.
\]

## 从 ID 到连续向量

离散 ID 本身没有适合神经网络计算的几何结构。例如 token id 17 与 18 数字上相邻，并不表示两个词语语义接近。

Embedding 把每个 ID 变成 learned vector：

```text
ID 17 → [ ... d values ... ]
ID 18 → [ ... d values ... ]
```

向量内容通过训练目标学习，而不是由 ID 数字大小决定。

## Embedding 与 Linear Layer 的关系

如果离散 ID 先写成 one-hot vector $e_i$，再乘 embedding matrix，本质上也能得到同一行参数。

但实际实现直接做 table lookup 更高效，不需要显式构造巨大 one-hot vector。

## Positional / Query Embedding

Embedding 不只用于词。

Transformer 中常见：

- learned positional embeddings；
- segment/type embeddings；
- CLS token embedding；
- DETR / ACT 的 learnable query embeddings。

例如 ACT 有 $k$ 个 action query embeddings：

\[
E_q\in\mathbb R^{k\times d}.
\]

它们不是从输入查询表里取出的 token meanings，而是直接训练的一组 model parameters。

## Embedding 表示的是可学习身份

可以把 embedding 理解为：给一个离散身份配一个可训练向量接口，让后续 neural network 能在连续向量空间中使用这个身份。
