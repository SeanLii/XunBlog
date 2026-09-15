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
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Transformer Decoder

Transformer Decoder 是一种以 target-side states 或 query states 为中心，通过 self-attention、可选 cross-attention 与 position-wise feed-forward transformation 产生输出 representations 的 Transformer stack。

“Decoder”描述的是 architecture lineage 与信息流，不等同于“逐 token 生成文字”。是否 autoregressive 由 mask、input/query construction 与 output procedure 决定。

## Original Encoder–Decoder Transformer

原始 Transformer 的 decoder layer 包含三类 sublayers：

```text
target states
    ↓
Masked Multi-Head Self-Attention
    ↓
Residual + LayerNorm
    ↓
Cross-Attention to Encoder Memory
    ↓
Residual + LayerNorm
    ↓
Position-Wise Feed-Forward Network
    ↓
Residual + LayerNorm
    ↓
output states
```

其中 self-attention 建模 target-side dependencies，cross-attention 连接 source encoder 与 target decoder。

## Masked Self-Attention

在 autoregressive sequence modeling 中，第 $i$ 个 target position 只能读取

\[
j\le i
\]

的位置。通过 [Causal Mask](/deep-learning/sequence-modeling/causal-mask/)，训练时可以并行计算整个 target sequence，同时防止当前位置直接访问 future ground-truth tokens。

## Cross-Attention to Encoder Memory

设 encoder 输出

\[
M\in\mathbb R^{N_s\times d_{model}},
\]

decoder states 为

\[
Y\in\mathbb R^{N_t\times d_{model}}.
\]

Cross-attention 中通常有

\[
Q=YW_Q,
\qquad
K=MW_K,
\qquad
V=MW_V.
\]

因此 target-side query slots 动态读取 source memory。Cross-attention output 数量由 target/query positions 决定，而 memory length 可以不同。

## Decoder Output

Decoder stack 输出的仍然是 hidden representations，而不是任务最终预测本身。Machine translation 中，hidden state 还需要经过 vocabulary projection 与 Softmax 得到 next-token distribution；DETR 中需要分类与 box heads；ACT 中需要 action head。

因此 decoder architecture 与 task-specific output head 应分开理解。

## Decoder-Only Transformer

GPT-style architecture 通常被称为 decoder-only Transformer。它保留 causal self-attention、FFN、residual 与 normalization stack，但删除 encoder memory 与 cross-attention。

这里“decoder-only”强调其 causal Transformer block lineage，而不是存在一个未显示的 encoder。

## Non-Autoregressive Query Decoder

Decoder 也可以接收一组并行 query slots，并在一次 forward 中形成多个 output representations。

[DETR](/deep-learning/detr/) 使用 learned object queries：

```text
object queries
      ↓
Transformer Decoder ← image memory
      ↓
object representations
```

[ACT](/robot-learning/act/) 使用 learned action queries，对 future action positions 进行并行预测。

这些 architecture 可以包含 query self-attention 与 cross-attention，却不进行逐 token causal generation。

## Query Source and Mask Define Behavior

同一个 Transformer Decoder abstraction 可以对应不同系统行为。关键设计变量包括：

- query states 从哪里来；
- self-attention 是否 causal；
- 是否存在 encoder memory；
- cross-attention 读取什么 source；
- outputs 如何映射到 task predictions。

因此判断一个 decoder 的功能，应检查这些具体结构，而不是仅依据 “Decoder” 名称推断生成方式。

## Sources

- Vaswani et al. *Attention Is All You Need*. 2017.
- Radford et al. *Language Models are Unsupervised Multitask Learners*. 2019.
- Carion et al. *End-to-End Object Detection with Transformers*. 2020.
