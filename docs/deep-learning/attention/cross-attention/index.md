---
title: "Cross-Attention"
kind: "canonical"
domain: "Deep Learning / Attention"
parent: "Attention"
canonical: "/deep-learning/attention/cross-attention/"
prerequisites:
  - "/deep-learning/attention/"
  - "/deep-learning/attention/qkv/"
related:
  - "/deep-learning/transformer/transformer-decoder/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Cross-Attention

> **知识边界**：本文的 canonical 对象是 **Cross-Attention**。依赖机制由 [Attention](/deep-learning/attention/)、[Query / Key / Value](/deep-learning/attention/qkv/) 的 canonical page 定义；本文只在当前语境中调用其接口。


Cross-Attention 是 Attention 的另一种信息流：Query 来自一组 representations，而 Key / Value 来自**另一组 representations**。

设：

\[
Y\in\mathbb R^{N_q\times d_y}
\]

是 query-side representations，

\[
X\in\mathbb R^{N_s\times d_x}
\]

是 source memory。

通常：

\[
Q=YW_Q,
\]

\[
K=XW_K,
\qquad
V=XW_V.
\]

然后：

\[
O=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V.
\]

所以每个 query-side position 可以从 source memory 中动态读取信息。

跨 sequence 的 attention 机制早于 Transformer；Transformer Decoder 将这种信息流实现为 Q/K/V matrix computation，并与 Multi-Head Attention 结合。

## 与 Self-Attention 的核心区别

Self-Attention：

```text
X → Q
X → K
X → V
```

Cross-Attention：

```text
Y → Q
X → K
X → V
```

attention formula 本身没有变化，区别在于 Query 与 Key / Value 来自不同 representation sources。

## Output 数量由 Query 决定

因为：

\[
QK^\top
\in\mathbb R^{N_q\times N_s},
\]

最后：

\[
O\in\mathbb R^{N_q\times d_v}.
\]

所以 output sequence length 是 $N_q$，不是 source length $N_s$。

这使 cross-attention 非常适合“固定若干 queries 从一个较大 memory 中读出若干 outputs”的结构。

## 原始 Transformer Decoder

Machine translation 中：

- encoder 把 source sentence 编成 memory；
- decoder target states 产生 queries；
- cross-attention 从 encoder memory 读取 source information。

```text
source sentence
      ↓
   Encoder
      ↓
    memory ──────┐
                 ↓ K,V
 target → Decoder Cross-Attention
                 ↑ Q
```

这让 decoder 在生成每个 target position 时，都能动态读取 source sentence 的不同部分。

## Multimodal Fusion

Cross-attention 也广泛用于 multimodal models。

例如 language queries 可以读取 image features：

```text
language hidden states → Q
visual features        → K,V
```

或者反过来，让 visual queries 读取 text memory。

所以 cross-attention 是一种通用的“两个 representation sets 之间建立可学习读取关系”的机制。

## Learnable Queries

在 DETR 一类模型中，queries 可以不是来自现有 target tokens，而是 learned query embeddings。

这些 queries 通过 cross-attention 从 image memory 中读取 object-related information。

ACT 的 decoder action queries 也利用了类似结构，但用途不同。

这说明 Cross-Attention 并不要求 query 一定对应语言 token；query 可以代表任何需要从 source memory 中读取信息的 slots。

## Sources

- Bahdanau, Cho, Bengio. *Neural Machine Translation by Jointly Learning to Align and Translate*. 2014.
- Vaswani et al. *Attention Is All You Need*. 2017.
