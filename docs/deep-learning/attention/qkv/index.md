---
title: "Query / Key / Value"
kind: "canonical"
domain: "Deep Learning / Attention"
parent: "Attention"
canonical: "/deep-learning/attention/qkv/"
prerequisites:
  - "/deep-learning/attention/"
related:
  - "/deep-learning/transformer/scaled-dot-product-attention/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Query / Key / Value

> **知识边界**：本文的 canonical 对象是 **Query / Key / Value**。依赖机制由 [Attention](/deep-learning/attention/) 的 canonical page 定义；本文只在当前语境中调用其接口。


Query、Key、Value 是 key-value attention 中的三种计算角色。

对一个 query $q$ 与 memory entries $(k_i,v_i)$，attention 首先用 $q$ 与每个 $k_i$ 计算匹配 score，再把得到的权重作用于对应 $v_i$：

\[
s_i=s(q,k_i),
\qquad
\alpha_i=\operatorname{softmax}(s)_i,
\qquad
o=\sum_i\alpha_i v_i.
\]

因此：

- Query 决定当前读取请求的表示；
- Key 决定候选信息如何参与匹配；
- Value 决定该候选被读取时贡献什么内容。

这些角色不要求具有人工可解释的固定语义。

## Key–Value Memory Formulation

设 memory 包含

\[
(k_1,v_1),\ldots,(k_N,v_N).
\]

Key 与 Value 成对出现。对某个 query，score 由 key 计算，但归一化后的权重作用于对应 value。

因此通常有相同数量的 keys 与 values：

\[
N_k=N_v.
\]

Query 数量则可以不同于 memory size。

## Learned Projections

Transformer-style attention 通常从输入 representations 通过不同 affine projections 得到 Q/K/V：

\[
Q=X_QW_Q,
\qquad
K=X_KW_K,
\qquad
V=X_VW_V.
\]

在 self-attention 中，$X_Q=X_K=X_V=X$；在 cross-attention 中，query source 与 key/value source 不同。

独立的 $W_Q,W_K,W_V$ 允许模型分别学习用于匹配请求、匹配索引和内容传输的 representation spaces。

## Single-Query Computation

对

\[
q\in\mathbb R^{d_k},
\qquad
K\in\mathbb R^{N\times d_k},
\]

scores 为

\[
s=qK^\top\in\mathbb R^N.
\]

Softmax 得到

\[
\alpha\in\mathbb R^N,
\]

再与

\[
V\in\mathbb R^{N\times d_v}
\]

相乘：

\[
o=\alpha V\in\mathbb R^{d_v}.
\]

一个 query 因此产生一个 aggregated output vector。

## Multiple Queries

若

\[
Q\in\mathbb R^{N_q\times d_k},
\qquad
K\in\mathbb R^{N_k\times d_k},
\]

则

\[
QK^\top\in\mathbb R^{N_q\times N_k}.
\]

第 $(i,j)$ 个元素对应 $q_i$ 与 $k_j$ 的匹配 score。每一 row 经过 normalization 后成为一个 query 对全部 memory entries 的读取权重。

## Query Count and Output Count

因为每个 query 产生一个 output，attention output 的 slot 数由 $N_q$ 决定，而不是由 key/value 数量决定。

这在 cross-attention 与 learned-query architectures 中尤其重要：

- source memory 可以有 $N_k$ 个 positions；
- decoder 可以有 $N_q$ 个 learned queries；
- output 仍有 $N_q$ 个 slots。

## Self-Attention and Cross-Attention

Self-Attention：

\[
Q=XW_Q,
\quad
K=XW_K,
\quad
V=XW_V.
\]

Cross-Attention：

\[
Q=YW_Q,
\quad
K=XW_K,
\quad
V=XW_V.
\]

因此 Q/K/V 是统一计算角色，不绑定某一种 architecture。

## Query Embedding and Q Matrix

Learnable query embedding 与 attention computation 中的 $Q$ 不是同一个概念。

例如 [Object Query](/deep-learning/detr/object-query/) 是 decoder 的 learned input / slot representation。它经过网络与 query projection 后，才会在具体 attention layer 中产生承担 query role 的 vectors。

因此“query token / query embedding”描述 architecture input identity，而 $Q$ 描述某次 attention computation 的 projected query representation。

## Sources

- Miller et al. *Key-Value Memory Networks for Directly Reading Documents*. 2016.
- Vaswani et al. *Attention Is All You Need*. 2017.
