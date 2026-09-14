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

Transformer 是一种以 attention 为核心的序列建模架构。原始论文提出的是 encoder-decoder Transformer，用 self-attention、cross-attention 和 position-wise feed-forward network 取代循环网络中的逐步状态传递。

## 架构边界

“Transformer”不是一个单独的 attention layer。原始架构由两个 stack 组成：

- Transformer Encoder：把输入序列变成 contextual representations；
- Transformer Decoder：在 encoder memory 和已有 decoder states 的条件下产生输出表示。

每个 stack 又由 attention、feed-forward、residual connection 和 layer normalization 等模块组成。

## Encoder

对输入 sequence representations $X$，encoder layer 先进行 multi-head self-attention，再进行 position-wise feed-forward transformation。原始论文在每个子层外使用 residual connection 与 LayerNorm。

Self-attention 使一个位置可以直接读取序列中的其他位置，因此信息传播不依赖 RNN 那样的逐步递归。

## Decoder

原始 decoder layer 包含三类子层：

1. masked multi-head self-attention；
2. 对 encoder output 的 cross-attention；
3. position-wise feed-forward network。

语言生成时 causal mask 阻止当前位置读取未来 token。但“Transformer decoder”并不等于“永远必须自回归”。像 DETR、ACT 这样的架构会保留 decoder 的 query-to-memory attention 结构，却以一组 learnable queries 并行产生输出槽位。

## 位置表示

Attention 本身主要根据内容关系计算权重。如果没有额外的位置信息，模型无法仅靠标准 self-attention 区分许多排列关系。因此 Transformer 把 [Positional Encoding](/deep-learning/transformer/positional-encoding/) 加入输入表示。

原始论文采用 sinusoidal positional encoding，也讨论了 learned positional embeddings。不同后续模型可以使用不同位置编码方案，因此 positional encoding 属于 Transformer 体系中的独立知识，而不是 attention 公式内部的一部分。

## 计算核心

Transformer 的基础 attention 为

\[
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
\]

这个公式只定义一次 attention computation。Multi-head 结构会使用多组 projections 并行执行它。

## 与 ACT 的关系

ACT 使用 Transformer，但并不是把原始机器翻译模型原样搬到机器人控制中。ACT 的 CVAE encoder 使用 BERT-like Transformer encoder；policy 部分使用 ResNet 提取视觉特征，再由 Transformer encoder 融合 observation features，并用 DETR-style Transformer decoder 通过 learnable action queries 预测整段 action chunk。

因此 ACT 页面只需要描述它如何组合这些模块；Transformer 的通用理论保留在这里及其子页面中。

## Sources

- [Attention Is All You Need — Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
