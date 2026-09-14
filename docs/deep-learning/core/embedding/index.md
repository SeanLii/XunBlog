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

Embedding 是把离散索引或可学习槽位映射成连续向量的机制。它让模型可以通过梯度下降学习每个离散对象对应的向量表示。

## 查表形式

设 embedding table 为

\[
E\in\mathbb R^{N\times d},
\]

其中 $N$ 是可用索引数，$d$ 是 embedding dimension。给定索引 $i$，输出就是第 $i$ 行：

\[
\mathbf e_i=E[i]\in\mathbb R^d.
\]

从前向计算看，它像查表；从训练角度看，$E$ 是可学习参数，使用到的行会接收梯度更新。

## Embedding 与 Linear Layer

对 one-hot 向量 $\mathbf x\in\mathbb R^N$，embedding lookup 等价于某种矩阵乘法。直接使用索引查表更高效，不需要显式构造绝大多数位置为 0 的 one-hot vector。

## 不只有“词向量”

Embedding 并不局限于自然语言。只要有一组离散身份或一组需要学习的固定槽位，都可以用 embedding 参数化。

BERT 的 `[CLS]` token 有可学习 embedding；DETR 和 ACT 的 decoder query 也使用 `nn.Embedding` 创建一组可学习 query slots。它们的语义不同，但数学上都属于“学习一组固定向量参数”。
