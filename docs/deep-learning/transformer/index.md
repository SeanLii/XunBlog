---
title: "Transformer"
kind: "canonical"
domain: "Deep Learning / Transformer"
parent: "Transformer"
canonical: "/deep-learning/transformer/"
prerequisites:
  - "/deep-learning/transformer/attention/"
  - "/deep-learning/transformer/positional-encoding/"
related:
  - "/deep-learning/transformer/transformer-encoder/"
  - "/deep-learning/transformer/transformer-decoder/"
  - "/robot-learning/act/architecture/"
---

# Transformer

Transformer 是一种让一组输入彼此交换信息，再产生一组新的表示的神经网络架构。

先不要从 Q、K、V 开始。假设输入是一串 token：

```text
x1   x2   x3   x4   x5
```

每个 $x_i$ 一开始只是当前位置自己的向量。经过 Transformer 后，得到：

```text
h1   h2   h3   h4   h5
```

关键变化是：$h_3$ 不再只由 $x_3$ 决定。通过 [Attention](/deep-learning/transformer/attention/)，它可以读取其他位置中与自己有关的信息。因此每个输出都成为一个带有上下文的表示。

这就是理解 Transformer 的第一层：

> **Transformer 的核心工作，是让多个表示相互读取信息，然后更新自己。**

## Transformer Layer 的基本数据流

最简单的 encoder layer 可以先看成两步：

```text
输入表示 X
   │
   ↓
Self-Attention
   │    每个位置读取其他位置
   ↓
Feed-Forward Network
   │    每个位置再独立做非线性变换
   ↓
输出表示 H
```

真实结构还会在这些子层周围加入 [Residual Connection](/deep-learning/core/residual-connection/) 与 [Layer Normalization](/deep-learning/core/layer-normalization/)。但如果一开始把所有组件同时塞进图里，很容易失去主线。

所以先把职责分开：

- Attention 负责 **位置之间的信息交换**；
- Feed-Forward Network 负责 **每个位置内部的表示变换**；
- Residual / LayerNorm 帮助深层网络稳定训练。

多个 layer 叠起来，就能反复进行“读别人 → 更新自己”。

## Attention 是 Transformer 的核心，但不等于 Transformer

Attention 的基础计算可以写成

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
\]

这条公式描述的是一次 attention computation，不是整个 Transformer。

要理解它，可以先把三个量看成不同职责：

- Query：当前这个位置想找什么；
- Key：每个候选位置用什么特征表示“我适不适合被你读取”；
- Value：如果决定读取这个位置，真正拿走什么信息。

完整推导在 [Query / Key / Value](/deep-learning/transformer/attention/qkv/) 与 [Scaled Dot-Product Attention](/deep-learning/transformer/attention/scaled-dot-product-attention/) 中展开。

## 位置信息的必要性

标准 self-attention 主要根据内容相似关系决定读取权重。如果只给一组 token vectors，却不提供它们的位置，网络本身不会自动知道“第一个”“前一个”“后一个”这些顺序信息。

因此 Transformer 还需要 [Positional Encoding](/deep-learning/transformer/positional-encoding/) 或其他位置表示，把位置信息加入 token representation。

于是输入不再只是“词是什么”，而是同时带有“它在哪里”。

## Encoder 和 Decoder 是两种信息流

原始 Transformer 是一个 encoder-decoder 模型。

### Encoder

Encoder 接收输入序列，并把每个位置变成 contextual representation：

```text
input tokens
    │
    ↓
Transformer Encoder
    │
    ↓
encoder memory
```

内部核心是 [Self-Attention](/deep-learning/transformer/attention/self-attention/)：Query、Key、Value 都来自同一组输入表示。

### Decoder

Decoder 的任务是产生另一组输出表示。原始机器翻译模型中，它一方面读取已经生成的 target tokens，另一方面通过 [Cross-Attention](/deep-learning/transformer/attention/cross-attention/) 读取 encoder memory。

```text
decoder state ─────┐
                   ↓
             Cross-Attention ← encoder memory
                   │
                   ↓
               output state
```

这里最重要的是理解：

> **Decoder 的本质不是“生成文字”，而是用一组 query states 去读取 memory，并更新这些 query。**

因此后来的 DETR、ACT 也可以使用 Transformer decoder，却不需要把任务写成语言生成。

## Transformer 在不同任务中可以长得很不一样

“Transformer”不是一张固定不变的网络图。

原始论文使用 encoder-decoder 结构；BERT 主要使用 encoder stack；GPT 类模型主要使用带 causal mask 的 decoder-style stack；DETR 使用 learned object queries；ACT 又把 learned action queries 用来产生未来多个动作位置。

这些模型共享 Attention、FFN、Residual 等基本机制，但数据流和训练目标不同。

所以学习 Transformer 时应把两层知识分开：

1. **通用机制**：Attention、QKV、Multi-Head、FFN、position information；
2. **具体架构**：某个模型怎样把这些机制组合起来。

## 一个具体 shape 例子

设输入有 $n=5$ 个 token，每个 token 的 hidden dimension 是 $d=512$：

\[
X\in\mathbb R^{5\times512}.
\]

Self-attention 不需要把序列压成一个向量。它仍然输出 5 个位置：

\[
H\in\mathbb R^{5\times512}.
\]

区别不在 shape，而在每一行的含义。输入的第 3 行主要表示 token 3 自己；输出的第 3 行则已经混合了它从其他位置读取的信息。

这也是 Transformer 特别适合处理 token set / sequence 的原因之一：**位置数量可以保持不变，但每个位置逐层获得更丰富的上下文。**

## 与 ACT 的连接

ACT 中的视觉 features、proprioception 等会形成一组 representations，进入 Transformer encoder 形成 memory；随后一组 [Learnable Query Embedding](/deep-learning/transformer/learnable-query-embedding/) 作为 action queries，通过 decoder 从 memory 中读取信息，分别对应 future action chunk 中的多个输出位置。

因此在 ACT 中：

```text
observation features
        │
        ↓
Transformer Encoder
        │
        ↓
      memory
        ↑
        │
action queries
        │
        ↓
Transformer Decoder
        │
        ↓
future action representations
```

这和“逐词翻译句子”已经是不同任务，但 Transformer 的核心信息流没有变。

## Sources

- Vaswani et al., **Attention Is All You Need**, 2017. https://arxiv.org/abs/1706.03762
- Carion et al., **End-to-End Object Detection with Transformers**, 2020. https://arxiv.org/abs/2005.12872
