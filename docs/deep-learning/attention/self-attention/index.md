---
title: "Self-Attention"
kind: "canonical"
domain: "Deep Learning / Attention"
parent: "Attention"
canonical: "/deep-learning/attention/self-attention/"
prerequisites:
  - "/deep-learning/attention/"
  - "/deep-learning/attention/qkv/"
related:
  - "/deep-learning/transformer/"
---

# Self-Attention

Self-Attention 是 Attention 的一种使用方式：Queries、Keys、Values 都来自**同一组 input representations**。

给定：

\[
X\in\mathbb R^{N\times d_{model}},
\]

先做 learned projections：

\[
Q=XW_Q,
\qquad
K=XW_K,
\qquad
V=XW_V.
\]

然后：

\[
O=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V.
\]

因为每个 position 同时可以作为 query，也可以作为 key/value source，所以每个输出位置都能读取同一 sequence 中其他位置的信息。

Self-Attention 的形式早于 Transformer；Transformer 的关键变化是把它提升为 encoder / decoder 的主要计算并与自己的 attention 设计组合。

## Single-Position Computation

对 position $i$：

\[
q_i
\]

会和所有 keys：

\[
k_1,k_2,\ldots,k_N
\]

计算 scores。

得到：

\[
\alpha_{i1},\ldots,\alpha_{iN}.
\]

最后：

\[
o_i=
\sum_{j=1}^{N}
\alpha_{ij}v_j.
\]

所以新 representation $o_i$ 不再只包含原来的 $x_i$，而是根据 learned relevance 聚合整个 sequence 的 information。

## Self-Attention 与 Fixed Neighborhood 不同

Convolution 中，一个位置通常读取固定 local neighborhood。

Full self-attention 中，一个 query 理论上可以直接连接 sequence 中任意 key：

```text
x1 ↔ x2 ↔ x3 ↔ ... ↔ xN
```

因此 long-range interaction 只需经过一层 attention 就能建立。

代价是 score matrix 需要：

\[
N^2
\]

pairwise entries。

## Self-Attention 本身不编码顺序

如果 inputs 只有内容 vectors，没有 positions，那么把 sequence 做 permutation 后，outputs 也会对应 permutation。

所以 vanilla self-attention 不知道：

- 哪个 token 在前；
- 距离是多少；
- 绝对位置是什么。

Transformer 通过 [Positional Encoding](/deep-learning/sequence-modeling/positional-encoding/) 等机制补充 position information。

## Bidirectional Self-Attention

如果没有 causal mask，每个 position 可以读取前后所有 positions。

这常用于 encoder-style representation learning：

```text
position i
reads
past + current + future
```

例如 BERT encoder 采用双向 self-attention。

## Causal Self-Attention

加入 [Causal Mask](/deep-learning/sequence-modeling/causal-mask/) 后，position $i$ 只能读取：

\[
j\le i.
\]

这适合 autoregressive generation，因为预测当前位置时不能偷看未来 tokens。

因此“self-attention”只说明 Q/K/V source 相同；是否 bidirectional 由 mask 决定。

## Vision 中的 Self-Attention

如果 image 被切成 patches，每个 patch representation 也可以组成 sequence：

\[
X\in\mathbb R^{N_{patch}\times d}.
\]

Self-attention 可以让一个 patch 直接读取其他 patches 的信息。

所以 self-attention 不依赖文本；它操作的是一组 vectors。

## Sources

- Cheng, Dong, Lapata. *Long Short-Term Memory-Networks for Machine Reading*. 2016.
- Lin et al. *A Structured Self-attentive Sentence Embedding*. 2017.
- Vaswani et al. *Attention Is All You Need*. 2017.
- Dosovitskiy et al. *An Image Is Worth 16×16 Words*. 2021.
