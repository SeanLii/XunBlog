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

Query、Key、Value 是 attention 中三个不同角色的向量。它们都只是向量，但模型通过不同的可学习投影让这些向量承担不同功能。

## 数学形式

给定输入表示矩阵

\[
X\in\mathbb R^{n\times d_{model}},
\]

self-attention 中常通过三个独立的 linear projections 得到

\[
Q=XW_Q,
\qquad
K=XW_K,
\qquad
V=XW_V.
\]

若每个 head 的 query/key dimension 为 $d_k$，value dimension 为 $d_v$，则

\[
W_Q,W_K\in\mathbb R^{d_{model}\times d_k},
\qquad
W_V\in\mathbb R^{d_{model}\times d_v}.
\]

所以同一个输入 token 可以同时产生一个 query、一个 key 和一个 value，但它们经过不同参数变换，不是同一个向量复制三次。

## Key 决定匹配，Value 提供内容

对单个 query $q$，attention 先计算

\[
s_i=q\cdot k_i.
\]

这里参与“该不该关注第 $i$ 个位置”判断的是 key $k_i$。随后得到权重 $\alpha_i$ 后，真正进入输出的是

\[
\sum_i\alpha_i v_i.
\]

因此 key 和 value 不可简单合并成一个概念：一个负责被比较，一个负责被读取。

## 角色来自计算位置

QKV 并没有人为规定的物理功能。功能来自计算图和训练目标：

- $W_Q$ 只通过 query 的使用路径接收梯度；
- $W_K$ 通过 compatibility score 的 key 路径接收梯度；
- $W_V$ 通过 weighted sum 的 value 路径接收梯度。

长期训练后，这三个投影会学习适合各自计算位置的表示。也就是说，“这是 Query”首先是它在公式中的角色，然后才是模型学出的语义结构。

## Self-Attention 与 Cross-Attention

在 [Self-Attention](/deep-learning/transformer/attention/self-attention/) 中，$Q,K,V$ 来自同一组输入表示。在 [Cross-Attention](/deep-learning/transformer/attention/cross-attention/) 中，query 与 key/value 来自不同序列。

ACT 的 Transformer decoder 使用后一种结构：learnable query slots 作为 decoder queries，而视觉、关节状态与 latent 表示构成被读取的 encoder memory。

## Sources

- [Attention Is All You Need — Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
