---
title: "Causal Mask"
kind: "canonical"
domain: "Deep Learning / Transformer"
parent: "Transformer"
canonical: "/deep-learning/transformer/causal-mask/"
prerequisites:
  - "/deep-learning/transformer/attention/scaled-dot-product-attention/"
related:
  - "/deep-learning/transformer/transformer-decoder/"
---

# Causal Mask

Causal Mask 是自回归 attention 中限制信息流的掩码：位置 $t$ 只能读取当前位置及之前允许的信息，不能读取未来位置。

## Score Masking

在 self-attention score matrix

\[
S=\frac{QK^\top}{\sqrt{d_k}}
\]

上，令

\[
M_{ij}=
\begin{cases}
0,& j\le i,\\
-\infty,& j>i.
\end{cases}
\]

再计算

\[
A=\operatorname{softmax}(S+M).
\]

被加上 $-\infty$ 的未来位置在 softmax 后权重为 0，因此当前 query 无法使用未来 key/value。

## Causality 的任务含义

在语言模型训练中，模型预测当前位置时不应该偷看未来目标 token。Causal mask 通过网络结构保证这一约束。

## 与 Padding Mask 的区别

Padding mask 屏蔽的是“这不是有效数据”的位置；causal mask 屏蔽的是“这个位置虽然有效，但当前时间上不允许访问”。两种 mask 可以同时存在，但含义不同。

## ACT 并不依赖标准自回归 Causal Mask

ACT 预测的是整个 future action chunk，并使用 DETR-style learned queries。其核心不是把 action $a_t,a_{t+1},\ldots$ 当成语言 token 逐个自回归生成。因此理解原始 Transformer Decoder 时需要知道 causal mask，而理解 ACT 时还必须知道它没有简单照搬文本生成的 decoder 使用方式。

## Sources

- [Attention Is All You Need — Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
- [ACT official implementation](https://github.com/tonyzhaozh/act)
