---
title: "Multi-Head Attention"
kind: "canonical"
domain: "Deep Learning / Transformer"
parent: "Transformer"
canonical: "/deep-learning/transformer/multi-head-attention/"
prerequisites:
  - "/deep-learning/transformer/scaled-dot-product-attention/"
related:
  - "/deep-learning/attention/self-attention/"
  - "/deep-learning/attention/cross-attention/"
---

# Multi-Head Attention

Multi-Head Attention 把一次 attention 拆成多个并行 heads。每个 head 都有自己的 Q/K/V projections，因此可以在不同 learned subspaces 中建立不同的信息读取关系。

原始 Transformer 定义：

\[
\operatorname{MultiHead}(Q,K,V)
=
\operatorname{Concat}(head_1,\ldots,head_h)W_O,
\]

其中：

\[
head_i
=
\operatorname{Attention}
(QW_i^Q,KW_i^K,VW_i^V).
\]

## Dimension 怎样分配

假设：

\[
d_{model}=512,
\qquad h=8.
\]

原始 Transformer 常设置：

\[
d_k=d_v=\frac{d_{model}}h=64.
\]

每个 head 在 64-dimensional projection space 中做 attention。

concat 后：

\[
8\times64=512,
\]

再通过 output projection $W_O$ 回到 model space。

## 每个 Head 不是把同一 Attention 复制八遍

如果所有 heads 使用相同 projections，确实只是重复计算。

实际每个 head 有独立 parameters：

\[
W_i^Q,W_i^K,W_i^V.
\]

所以相同 input 会被投影到不同 representation subspaces。

一个 head 可以学习对某类 relation 敏感，另一个 head 可以形成不同 matching pattern。

具体 head 的行为由训练决定，并不保证每个 head 都对应可被人类稳定命名的单一语义；不同 heads 之间也可能存在 redundancy。

## Multiple Heads 与 One Wider Head

单头 attention 的一套 Softmax weights 对每个 query 形成一套 reading distribution。

Multi-head 允许同时形成多套独立 reading distributions：

\[
A^{(1)},A^{(2)},\ldots,A^{(h)}.
\]

因此不是简单“增加 vector dimension”，而是增加并行关系通道。

## Shape

若 batch size 为 $B$，sequence length 为 $N$：

\[
X\in\mathbb R^{B\times N\times d_{model}}.
\]

projection 后常 reshape 成：

\[
Q,K,V
\in
\mathbb R^{B\times h\times N\times d_k}.
\]

每个 head 独立计算：

\[
A
\in
\mathbb R^{B\times h\times N_q\times N_k}.
\]

最后 transpose / concat 回：

\[
O\in\mathbb R^{B\times N_q\times d_{model}}.
\]

这些 tensor shapes 决定了实际实现中的 reshape、transpose 与 batched matrix multiplication。

## Multi-Head 不等于多个 Outputs

Heads 是内部 parallel computations，最终通常 concat 后投影为一个 output representation per query position。

所以：

- 8 heads 不意味着每个 token 输出 8 个独立 tokens；
- 它意味着每个 token 的新 representation 由 8 个 attention subspaces 共同构成。

## Modern Variants

后续大模型还出现 Multi-Query Attention、Grouped-Query Attention 等变体，通过让多个 query heads 共享 K/V heads 来降低 KV cache 与 inference cost。

这些变化进一步说明“multi-head”是一种具体 parameterization design，而 Attention 的核心 matching / weighted-reading 机制仍然保持。

## Sources

- Vaswani et al. *Attention Is All You Need*. 2017.
