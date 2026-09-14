---
title: "Query / Key / Value"
kind: "canonical"
domain: "Deep Learning / Transformer / Attention"
parent: "Attention"
canonical: "/deep-learning/transformer/attention/qkv/"
prerequisites:
  - "/deep-learning/transformer/attention/"
  - "/deep-learning/core/linear-layer/"
related:
  - "/deep-learning/transformer/attention/self-attention/"
  - "/deep-learning/transformer/attention/cross-attention/"
---

# Query / Key / Value

Query、Key、Value 不是三种神秘的数据结构，而是同一批表示经过不同 linear projections 后承担的三种计算角色。

先看一次“检索”：

```text
我要找什么？        → Query
你适不适合被我读取？ → Key
如果读取你，拿走什么？→ Value
```

这个类比只帮助区分角色；正式计算仍然是向量与矩阵运算。

## 从输入向量得到 Q、K、V

设一组输入表示为

\[
X\in\mathbb R^{n\times d_{model}}.
\]

Transformer 学习三个矩阵：

\[
W_Q,\quad W_K,\quad W_V.
\]

然后

\[
Q=XW_Q,
\qquad
K=XW_K,
\qquad
V=XW_V.
\]

也就是说，Q/K/V 最开始都来自同一个 $X$，只是经过不同 learned linear transformations。

## 一个位置的 Query 怎样读取所有 Key

取第 $i$ 个 query $q_i$。它会和所有 keys 做 [Dot Product](/mathematics/linear-algebra/dot-product/)：

\[
q_i^\top k_1,
q_i^\top k_2,
\ldots,
q_i^\top k_n.
\]

这些分数回答的是：**按照当前学到的 Q/K 特征空间，第 $i$ 个读取者与哪些候选最匹配。**

经过 scale 和 softmax 后得到权重 $\alpha_{ij}$。最后真正拿走的是 Values：

\[
y_i=\sum_j\alpha_{ij}v_j.
\]

因此 Key 主要参与“决定权重”，Value 主要参与“构造输出”。

## Key 与 Value 不能混成一个概念

一个位置的 Key 和 Value 都来自它自己的 representation，但职责不同。

例如某个 token 可以学习到：

- Key 中保留“我是一个地点词、现在处在句子前半段”等适合匹配的信息；
- Value 中保留“如果别人关注我，我应该提供这些语义内容”。

这不是人为规定某个维度必须是什么语义，而是训练过程中 learned projections 形成的功能分工。

## Matrix form

一次对所有 queries 同时计算：

\[
S=QK^\top.
\]

如果

\[
Q\in\mathbb R^{n_q\times d_k},
\qquad
K\in\mathbb R^{n_k\times d_k},
\]

则

\[
S\in\mathbb R^{n_q\times n_k}.
\]

第 $i$ 行就是 query $i$ 对所有 keys 的 scores。

Softmax 后：

\[
A=\operatorname{softmax}(S/\sqrt{d_k}),
\]

再乘

\[
Y=AV.
\]

若

\[
V\in\mathbb R^{n_k\times d_v},
\]

则

\[
Y\in\mathbb R^{n_q\times d_v}.
\]

这个 shape 很重要：**输出数量由 Query 的数量决定。**

## 这个 shape 对 Cross-Attention 很关键

如果有 100 个 action queries 去读取 500 个 observation tokens：

\[
Q\in\mathbb R^{100\times d_k},
\qquad
K,V\in\mathbb R^{500\times d_k/d_v},
\]

attention score matrix 大小就是

\[
100\times500.
\]

最终输出仍然有 100 个位置。每个 action query 得到一个从 observation memory 中读取出的表示。

这正是理解 ACT / DETR decoder 的关键：**Query 数量决定输出 slots 数量，Key/Value 数量决定可读取的 memory positions 数量。**

## QKV 是 learned roles，不是固定语义

Q 不一定就是“问题句子”，K 不一定是“数据库索引”，V 也不一定是“答案文本”。这些只是帮助理解的比喻。

正式上，它们是 learned vector projections；其功能由训练目标决定。

## Sources

- Vaswani et al., **Attention Is All You Need**, 2017. https://arxiv.org/abs/1706.03762
