---
title: "Cross-Attention"
kind: "canonical"
domain: "Deep Learning / Transformer / Attention"
parent: "Attention"
canonical: "/deep-learning/transformer/attention/cross-attention/"
prerequisites:
  - "/deep-learning/transformer/attention/scaled-dot-product-attention/"
related:
  - "/deep-learning/transformer/transformer-decoder/"
  - "/robot-learning/act/architecture/"
---

# Cross-Attention

Cross-Attention 指 query 与 key/value 来自不同的信息集合。它让一组表示主动读取另一组表示中的信息。

## 数学结构

设 query source 为

\[
X_q\in\mathbb R^{n_q\times d_{model}},
\]

memory source 为

\[
X_m\in\mathbb R^{n_m\times d_{model}}.
\]

常见计算为

\[
Q=X_qW_Q,
\qquad
K=X_mW_K,
\qquad
V=X_mW_V.
\]

随后

\[
O=\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V.
\]

score matrix 的 shape 是

\[
n_q\times n_m.
\]

所以每个 query position 都可以独立决定如何读取 memory 中的 $n_m$ 个位置。

## 与 Self-Attention 的差别

两者使用相同的 attention 公式。差别在数据来源：

- Self-Attention：$Q,K,V$ 来自同一组表示；
- Cross-Attention：$Q$ 来自一组表示，$K,V$ 来自另一组表示。

因此 Cross-Attention 不是一种全新的数学运算，而是 attention 的输入组织方式不同。

## Transformer Decoder

原始 encoder-decoder Transformer 中，decoder 的 cross-attention 让 decoder state 读取 encoder output。这里 encoder output 同时生成 keys 与 values。

## ACT 中的角色

ACT 延续 DETR 风格的 decoder：一组 learnable query embeddings 形成 action slots，decoder 通过 cross-attention 从 observation memory 中读取视觉特征、proprioception 与 latent style 信息。每个 query slot 最终对应 action chunk 中的一个位置。

这与语言生成中的“前一个 token 产生下一个 token”不同。ACT 的 query slots 本身就是一组并行的输出位置，因此不应把 ACT decoder 机械理解成自回归文本 decoder。

## Sources

- [Attention Is All You Need — Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
