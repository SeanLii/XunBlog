---
title: "Transformer Decoder"
kind: "canonical"
domain: "Deep Learning / Transformer"
parent: "Transformer"
canonical: "/deep-learning/transformer/transformer-decoder/"
prerequisites:
  - "/deep-learning/transformer/attention/self-attention/"
  - "/deep-learning/transformer/attention/cross-attention/"
  - "/deep-learning/transformer/feed-forward-network/"
related:
  - "/deep-learning/transformer/causal-mask/"
  - "/deep-learning/transformer/learnable-query-embedding/"
  - "/robot-learning/act/architecture/"
---

# Transformer Decoder

Transformer Decoder 是一组以 query representations 为中心、能够读取 encoder memory 的 decoder layers。原始 Transformer 用它做自回归序列生成；后续模型也可以保留 decoder 的 cross-attention 结构而采用非自回归 queries。

## 原始结构

原始 Transformer decoder layer 包含：

1. masked multi-head self-attention；
2. encoder-decoder cross-attention；
3. position-wise feed-forward network。

第一步在语言生成中只允许读取已经出现的位置；第二步让 decoder state 从 encoder memory 中提取相关信息。

## Cross-Attention

设 decoder states 为 $X_d$，encoder memory 为 $M$。Cross-attention 中常见来源是

\[
Q=X_dW_Q,
\qquad
K=MW_K,
\qquad
V=MW_V.
\]

因此 decoder 的每个 query 都能针对自己的需求读取同一份 encoder memory。

## Decoder 不等于自回归

Causal mask 是原始文本生成任务所需的约束，而不是所有 Transformer Decoder 的数学定义。DETR 使用 fixed number of learned object queries，同时读取 image memory；ACT 延续这种设计，把 queries 改成 action sequence slots。

因此在 ACT 中，不应该假设“decoder 第 3 个 action 必须先看到 decoder 第 2 个 action 的真实输出”。它预测的是一个并行 action chunk，并通过 decoder layers 在 query slots 和 memory 之间建立关系。

## ACT 的输出

ACT 的 decoder 输出每个 action query 对应的 hidden representation，再经过输出 projection 得到 action dimension。论文中 action chunk 是 $k\times14$；released implementation 的 `action_head` 是从 hidden dimension 到 14 维的 linear layer。

这个具体 shape 属于 ACT，而不是 Transformer Decoder 的一般定义。

## Sources

- [Attention Is All You Need — Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
- [DETR — Carion et al., 2020](https://arxiv.org/abs/2005.12872)
- [ACT official implementation](https://github.com/tonyzhaozh/act)
