---
title: "Transformer Decoder"
kind: "canonical"
domain: "Deep Learning / Transformer"
parent: "Transformer"
canonical: "/deep-learning/transformer/transformer-decoder/"
prerequisites:
  - "/deep-learning/attention/self-attention/"
  - "/deep-learning/attention/cross-attention/"
  - "/deep-learning/transformer/position-wise-feed-forward-network/"
related:
  - "/deep-learning/transformer/transformer-encoder/"
---

# Transformer Decoder

Transformer Decoder 是以一组 target-side states / queries 为中心，通过 self-attention、可选 cross-attention 和 feed-forward transformation 产生输出 representations 的 Transformer stack。

“Decoder”这个名字容易让人误以为它一定是 autoregressive text generator。实际上 decoder layer 是一种 information-flow architecture；是否 autoregressive 取决于 query source、mask 与 output procedure。

## 原始 Encoder-Decoder Transformer

原始 machine translation Transformer 中，一个 decoder layer 有三类 sublayers：

```text
target states
    ↓
Masked Self-Attention
    ↓
Cross-Attention to Encoder Memory
    ↓
Position-Wise Feed-Forward Network
    ↓
output states
```

每个 sublayer 外还有 [Residual Connection](/deep-learning/cnn/resnet/residual-connection/) 与 [Layer Normalization](/deep-learning/core/layer-normalization/)。

## Masked Self-Attention

目标 sequence 在 autoregressive training 中使用 [Causal Mask](/deep-learning/sequence-modeling/causal-mask/)：

\[
position\ i
\text{ 只能读取 } j\le i.
\]

这防止 target position 直接看到未来 ground-truth tokens。

## Cross-Attention 是 Source–Target 的连接点

Encoder 产生 memory：

\[
M\in\mathbb R^{N_s\times d}.
\]

Decoder hidden states 产生 queries：

\[
Q\in\mathbb R^{N_t\times d_k}.
\]

memory 提供 keys / values：

\[
K=MW_K,
\qquad
V=MW_V.
\]

因此 decoder 中每个 target position 都可以动态读取 source sequence。

## Decoder-Only Transformer

GPT-style architecture 常被叫 decoder-only Transformer。

它保留 causal self-attention + FFN stack，但没有 encoder memory cross-attention。

所以 “decoder” 在这个语境中主要强调 causal autoregressive stack lineage，而不是一定存在一个单独 encoder。

## Non-Autoregressive Decoder

Decoder 也可以使用一组 learnable queries，一次并行输出多个 slots。

例如 [DETR](/deep-learning/detr/)：

```text
learnable object queries
        ↓
Transformer Decoder
        ↓
set of object representations
```

这些 queries 可以互相 self-attend，再 cross-attend image encoder memory。

没有 token-by-token causal generation。

所以：

> **Transformer Decoder 不等于 autoregressive decoding。**

## ACT 中的 Decoder

ACT 使用 action queries 作为 decoder-side slots，从 encoder memory 中读取视觉、proprioception 与 latent-conditioned information，并行产生 action chunk representations。

这是 non-autoregressive decoder 的一个例子。

因此理解 ACT decoder 时，应该从“queries 读取 memory”的 decoder architecture 出发，而不是套用语言模型“每次生成一个 token”的 mental model。

## Sources

- Vaswani et al. *Attention Is All You Need*. 2017.
- Carion et al. *End-to-End Object Detection with Transformers*. 2020.
