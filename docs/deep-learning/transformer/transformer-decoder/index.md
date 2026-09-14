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

Transformer Decoder 可以理解成：**用一组 decoder states / queries 去读取 memory，并逐层更新这些 queries。**

在原始机器翻译 Transformer 中，它还需要 masked self-attention 来处理已生成 target tokens；在 DETR、ACT 等模型里，decoder 的使用方式会改变。

## 原始 Decoder Layer

原始 Transformer decoder layer 有三个主要 sublayers：

```text
decoder states
     │
     ↓
masked self-attention
     │
     ↓
cross-attention ← encoder memory
     │
     ↓
feed-forward network
     │
     ↓
updated decoder states
```

每个 sublayer 周围还有 residual connection 与 layer normalization。

## Cross-Attention 是 encoder-decoder 连接点

Decoder query $Q$ 来自 decoder states，而 $K,V$ 来自 encoder memory：

\[
\operatorname{Attention}(Q_{dec},K_{mem},V_{mem}).
\]

所以 decoder 输出的数量由 query positions 数量决定。

## Decoder 不等于 autoregressive generation

“Transformer decoder”常让人想到 GPT 或逐词生成，但这是某些架构的使用方式，不是 decoder 机制的唯一用途。

DETR 使用固定数量 learned object queries，全部并行送入 decoder；ACT 使用 action queries，全部并行产生 future action representations。

```text
learned queries
     │
     ↓
decoder reads memory
     │
     ↓
parallel output slots
```

因此没有要求“第 2 个 query 必须等第 1 个输出生成完才开始”。

## ACT 中的 Decoder

如果 chunk size 是 $k$，ACT 有 $k$ 个 learnable action queries。Decoder 让这些 queries 从 observation memory 中读取信息，最后得到 $k$ 个 hidden states：

\[
H_{dec}\in\mathbb R^{k\times d}.
\]

每个 hidden state 再通过 action head 映射为一个 joint target。

这就是“Transformer decoder 为什么可以一次输出整个 action chunk”的结构基础。

## Sources

- Vaswani et al., **Attention Is All You Need**, 2017. https://arxiv.org/abs/1706.03762
- Carion et al., **DETR**, 2020. https://arxiv.org/abs/2005.12872
- Zhao et al., **ACT**, 2023. https://arxiv.org/abs/2304.13705
