---
title: "Attention"
kind: "canonical"
domain: "Deep Learning / Attention"
parent: "Deep Learning"
canonical: "/deep-learning/attention/"
prerequisites:
  - "/mathematics/linear-algebra/dot-product/"
  - "/deep-learning/core/softmax/"
related:
  - "/deep-learning/transformer/"
---

# Attention

Attention 是一种**根据当前 query，动态从一组信息中读取内容**的机制。

Attention 作为机制早于 Transformer；Transformer 后来把它提升为网络的核心计算，并定义了 Scaled Dot-Product Attention 与 Multi-Head Attention 等具体形式。

它最核心的结构可以写成：

```text
Query
  ↓
和每个 Key 计算匹配程度
  ↓
得到 weights
  ↓
对对应 Values 做 weighted sum
  ↓
Output
```

因此 attention 并不是“让模型关注重要位置”的一句抽象口号。它有一个非常具体的计算对象：

- Query：当前想读取什么；
- Key：每个候选位置拿什么来参与匹配；
- Value：真正被读取和汇总的内容。

完整角色见 [Query / Key / Value](/deep-learning/attention/qkv/)。

## 从 Weighted Sum 开始

假设有三个 value vectors：

\[
v_1,v_2,v_3.
\]

如果已经有 weights：

\[
\alpha_1,\alpha_2,\alpha_3,
\qquad
\sum_i\alpha_i=1,
\]

可以得到：

\[
o=
\alpha_1v_1+
\alpha_2v_2+
\alpha_3v_3.
\]

这一步只是普通 weighted sum。

Attention 真正特别的地方是：这些 weights 不是固定 parameters，而是根据当前 query 和 keys **动态计算**：

\[
\alpha_i
=
\operatorname{softmax}(s(q,k_i)).
\]

因此同一组 values 面对不同 query，会得到不同 reading pattern。

## Attention 是 Content-Dependent Routing

普通 Linear Layer 使用固定 weights：

\[
y=Wx.
\]

parameters $W$ 训练完后，对所有输入使用同一套 mapping。

Attention 中，真正用于 mixing values 的 coefficients $\alpha_i$ 会随输入变化。

所以可以把 attention 看成一种 content-dependent information routing：

> **输入内容决定这一次应该从哪些位置读取多少信息。**

## Score Function

Attention 并不只存在 dot-product 一种形式。

Bahdanau attention 使用 learned additive scoring function；Transformer 使用 [Scaled Dot-Product Attention](/deep-learning/transformer/scaled-dot-product-attention/)：

\[
s(q,k_i)
=\frac{q^\top k_i}{\sqrt{d_k}}.
\]

然后通过 Softmax 得到 normalized weights。

因此 attention 是更上层的机制；scaled dot-product 是其中一种具体 score / aggregation implementation。

## Matrix Form

若：

\[
Q\in\mathbb R^{N_q\times d_k},
\quad
K\in\mathbb R^{N_k\times d_k},
\quad
V\in\mathbb R^{N_k\times d_v},
\]

Transformer-style attention：

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V.
\]

Shape 逐步变化：

\[
QK^\top
\in\mathbb R^{N_q\times N_k},
\]

表示每个 query 对每个 key 的 score。

Softmax 后 shape 不变，再乘：

\[
V\in\mathbb R^{N_k\times d_v},
\]

得到：

\[
O\in\mathbb R^{N_q\times d_v}.
\]

因此 output 数量由 query 数量 $N_q$ 决定。

## Self-Attention 与 Cross-Attention

如果 Q/K/V 都来自同一组 input representations：

\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V,
\]

得到 [Self-Attention](/deep-learning/attention/self-attention/)。

如果 queries 来自一组 representations，而 keys / values 来自另一组：

\[
Q=YW_Q,
\qquad
K=XW_K,
\qquad
V=XW_V,
\]

得到 [Cross-Attention](/deep-learning/attention/cross-attention/)。

它们的核心 attention computation 相同，区别是 information source。

## Multi-Head Attention

单次 attention 在一个 learned projection space 中计算 matching 与 aggregation。

[Multi-Head Attention](/deep-learning/transformer/multi-head-attention/) 并行建立多个 projection spaces：

\[
head_1,\ldots,head_h,
\]

再 concat / project 回 model dimension。

这让同一层可以并行建立多组不同的 reading relationships。

## Attention 不自动理解顺序

如果没有 positional information，并且我们同时对 token order 做一致 permutation，self-attention 的计算会对应地 permutation-equivariant。

所以 attention 本身没有“第一个 token / 第二个 token”的固有顺序概念。

Transformer 需要额外的 [Positional Encoding](/deep-learning/sequence-modeling/positional-encoding/) 或 position-dependent mechanism。

## Attention 的计算成本

Full self-attention 对 $N$ 个 tokens 构造：

\[
N\times N
\]

score matrix。

因此 score computation / memory 对 sequence length 常表现为：

\[
O(N^2).
\]

长序列模型中大量 efficient attention methods，本质上都在尝试改变或近似这张 pairwise interaction matrix。

## Sources

- Bahdanau, Cho, Bengio. *Neural Machine Translation by Jointly Learning to Align and Translate*. 2014.
- Vaswani et al. *Attention Is All You Need*. 2017.
- Vaswani et al. *Attention Is All You Need*. 2017.
