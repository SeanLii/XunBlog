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

Learnable Query Embedding 是一组由训练直接学习的 query vectors。它们不必来自输入 token，而可以作为固定数量的“输出槽位”送入 Transformer decoder。

DETR 是理解这种设计的典型例子：模型设置一组 learned object queries，每个 query 通过 decoder 从 image memory 中读取信息，并产生一个 object prediction slot。

## Query 不需要先对应真实对象

假设有 $N$ 个 query parameters：

\[
Q_{learned}\in\mathbb R^{N\times d}.
\]

训练开始时它们只是随机初始化的 vectors。通过任务 loss，模型逐渐学会怎样利用这些 queries 从 memory 中取信息。

因此它们不是“把第 1 个物体的坐标编码进去”，而是 neural network parameters。

## 输出数量与 Query 数量

Cross-attention 的输出位置数由 Query 数量决定。

所以如果有 100 个 learned queries，decoder 可以并行产生 100 个 output representations：

\[
H\in\mathbb R^{100\times d}.
\]

后面的 prediction head 再把每个 representation 变成任务需要的输出。

## ACT 的 Action Queries

ACT 把 DETR-style queries 改成 future action slots。

如果 chunk size 为 $k$：

\[
Q_{action}\in\mathbb R^{k\times d}.
\]

Decoder 输出：

\[
H_{action}\in\mathbb R^{k\times d},
\]

再映射成：

\[
\hat A\in\mathbb R^{k\times d_a}.
\]

第 $i$ 个 query 对应 future chunk 中第 $i$ 个 action position。

这意味着 action sequence 不是 autoregressively 一步一步生成，而是由一组 queries **并行形成多个未来动作槽位**。

## Query Embedding 与 Q 矩阵

术语上要区分：

- learnable query embedding：decoder 输入的一组 learned states / parameters；
- attention 中的 Query matrix $Q$：这些 states 经过 $W_Q$ projection 后真正用于 dot-product attention 的矩阵。

两者相关，但不是同一个数学对象。

## Sources

- Carion et al., **End-to-End Object Detection with Transformers**, 2020. https://arxiv.org/abs/2005.12872
- Zhao et al., **ACT**, 2023. https://arxiv.org/abs/2304.13705
