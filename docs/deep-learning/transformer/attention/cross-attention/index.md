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

Cross-Attention 是“Query 来自一组表示，而 Key/Value 来自另一组表示”的 attention。

最清楚的结构是：

```text
query states ──→ Q

memory states ─→ K
memory states ─→ V
```

于是 query side 负责提出读取需求，memory side 负责提供可检索的信息。

## 与 Self-Attention 的区别

Self-attention：

\[
Q=XW_Q,
K=XW_K,
V=XW_V.
\]

Cross-attention：

\[
Q=X_qW_Q,
K=X_mW_K,
V=X_mW_V.
\]

其中 $X_q$ 与 $X_m$ 是不同来源。

计算公式仍然是

\[
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
\]

## 输出数量由 Query 数量决定

假设有 10 个 queries、300 个 memory tokens：

\[
Q\in\mathbb R^{10\times d_k},
\qquad
K,V\in\mathbb R^{300\times d_k/d_v}.
\]

score matrix 为

\[
10\times300.
\]

最后输出仍然有 10 个位置。

所以 cross-attention 可以理解为：**每个 query 生成一个“从整份 memory 读取后的新表示”。**

## 原始 Transformer Decoder

机器翻译中，decoder states 作为 queries，encoder outputs 作为 memory：

```text
decoder representations
        │
        ↓ Q
   Cross-Attention
        ↑ K,V
        │
encoder representations
```

这样每个 target position 都可以读取 source sentence 中相关信息。

## DETR 与 ACT 中的意义

DETR 把固定数量的 learnable object queries 送入 decoder，让每个 query 从 image memory 中读取信息并形成一个 object prediction slot。

ACT 延续类似思路，把 learnable queries 解释为 future action slots：

```text
action query 1 ─┐
action query 2 ─┤
...              ├→ cross-attend observation memory
query k ─────────┘
```

最终得到 $k$ 个 future action representations。

所以 cross-attention 不只属于语言任务；它是一种通用的“query set 读取 memory set”的机制。

## Sources

- Vaswani et al., **Attention Is All You Need**, 2017. https://arxiv.org/abs/1706.03762
- Carion et al., **End-to-End Object Detection with Transformers**, 2020. https://arxiv.org/abs/2005.12872
