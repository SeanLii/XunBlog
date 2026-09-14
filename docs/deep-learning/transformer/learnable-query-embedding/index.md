---
title: "Learnable Query Embedding"
kind: "canonical"
domain: "Deep Learning / Transformer"
parent: "Transformer"
canonical: "/deep-learning/transformer/learnable-query-embedding/"
prerequisites:
  - "/deep-learning/core/embedding/"
  - "/deep-learning/transformer/attention/cross-attention/"
related:
  - "/deep-learning/transformer/transformer-decoder/"
  - "/robot-learning/act/architecture/"
---

# Learnable Query Embedding

Learnable Query Embedding 是一组直接作为模型参数学习的 query slots。它们不必由当前输入 token 经过 projection 得到，而可以作为固定数量的可学习向量，在每次前向计算中用于向 encoder memory 发起查询。

## 参数化

设需要 $K$ 个 query slots，hidden dimension 为 $d$，可以定义

\[
E_q\in\mathbb R^{K\times d}.
\]

$E_q$ 的每一行都是可学习参数。训练过程中，梯度会根据每个 slot 最终承担的输出任务更新这些向量。

它们在不同样本之间共享参数，但 cross-attention 读取的 memory 会随样本变化，所以 decoder output 仍然依赖当前输入。

## DETR 中的来源

DETR 使用 fixed set of learned object queries。每个 query 通过 Transformer decoder 从 image features 中读取信息，最终产生一个 object prediction slot。Query 自己不是一个检测到的物体；它是一个可学习输出槽位。

## ACT 中的改造

ACT 的官方实现定义

\[
K=\text{num\_queries}=\text{chunk size}
\]

个 query embeddings。它们不再代表“可能的物体槽位”，而对应 action chunk 中的一组输出 slots。Decoder 让这些 queries 读取当前视觉、proprioception 和 latent-conditioned memory，最后每个 slot 投影成一个 14 维 action。

因此 query embedding 的数学机制可以从 DETR 理解，但其任务语义由 ACT 重新定义。不能把 ACT 的 query 继续解释成 object query。

## Sources

- [DETR — Carion et al., 2020](https://arxiv.org/abs/2005.12872)
- [ACT official implementation](https://github.com/tonyzhaozh/act)
