---
title: "Transformer Encoder"
kind: "canonical"
domain: "Deep Learning / Transformer"
parent: "Transformer"
canonical: "/deep-learning/transformer/transformer-encoder/"
prerequisites:
  - "/deep-learning/attention/self-attention/"
  - "/deep-learning/transformer/position-wise-feed-forward-network/"
  - "/deep-learning/cnn/resnet/residual-connection/"
  - "/deep-learning/core/layer-normalization/"
related:
  - "/deep-learning/transformer/transformer-decoder/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Transformer Encoder

Transformer Encoder 接收一组 input representations，通过多层 self-attention 与 feed-forward transformation，把它们变成一组 contextual representations。

输入：

\[
X^{(0)}
\in\mathbb R^{N\times d_{model}}.
\]

经过 $L$ 个 encoder layers：

\[
X^{(0)}
\rightarrow
X^{(1)}
\rightarrow\cdots\rightarrow
X^{(L)}.
\]

通常 sequence length 与 model dimension 在 encoder 内保持不变：

\[
X^{(l)}\in\mathbb R^{N\times d_{model}}.
\]

## 一个 Encoder Layer

原始 Transformer 的 encoder layer 有两个主要 sublayers：

```text
input
  ↓
Multi-Head Self-Attention
  ↓
Residual + LayerNorm
  ↓
Position-Wise Feed-Forward Network
  ↓
Residual + LayerNorm
  ↓
output
```

[Self-Attention](/deep-learning/attention/self-attention/) 负责 positions 间信息交换；[Position-Wise Feed-Forward Network](/deep-learning/transformer/position-wise-feed-forward-network/) 负责每个 position 内的 nonlinear feature transformation。

## Contextual Representation

进入 encoder 前，一个 token representation 主要由自身 content embedding 与 position information 构成。

经过 self-attention 后：

\[
h_i^{(1)}
\]

已经可以读取其他 positions。

多层堆叠后：

\[
h_i^{(L)}
\]

逐渐融合更复杂的 global context。

因此 Encoder 的输出不是把 sequence 压成单个 vector，而通常仍然保留每个 position 一个 representation。

## Encoder Memory

在 encoder-decoder architecture 中，最终输出：

\[
M=X^{(L)}
\]

常被叫 encoder memory。

Decoder cross-attention 使用：

\[
K=MW_K,
\qquad
V=MW_V.
\]

所以 “memory” 不是另一个独立神经模块，它只是 encoder 输出在 decoder reading context 中的角色名称。

## Masking

Encoder self-attention 通常不是 causal 的，因为 encoder 任务经常允许一个 input position 看到整个 input sequence。

但会使用 padding mask，避免读取 padding positions。

因此“Transformer Encoder = bidirectional”是常见情况，但不是由 encoder 名称在数学上强制决定；attention connectivity 仍由 mask design 控制。

## Pre-Norm 与 Post-Norm

原始 Transformer 使用 post-norm：

\[
\operatorname{LN}(x+F(x)).
\]

很多现代 implementations 使用 pre-norm：

\[
x+F(\operatorname{LN}(x)).
\]

因此在阅读具体模型时，需要确认 normalization placement，而不能只看到“Transformer Encoder”就假设实现完全等同原论文。

## Encoder-Only Models

Transformer Encoder 可以独立使用，不一定必须搭配 Decoder。

例如 BERT-style models 只堆 encoder layers，用 contextual representations 完成 classification、token prediction 等任务。

Vision Transformer 也是 encoder-like structure：patch tokens 通过 self-attention encoder stack 形成 visual representations。

## Sources

- Vaswani et al. *Attention Is All You Need*. 2017.
