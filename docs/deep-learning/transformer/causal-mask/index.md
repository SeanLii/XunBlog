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

Causal Mask 用来阻止一个序列位置在 attention 中读取“未来位置”。

对于 autoregressive sequence，位置 $i$ 只允许读取

\[
j\le i.
\]

## Mask 怎样进入 Attention

在 softmax 前有 score matrix：

\[
S=\frac{QK^\top}{\sqrt{d_k}}.
\]

加入 mask $M$：

\[
S'=S+M.
\]

对未来位置设置

\[
M_{ij}=-\infty,
\qquad j>i.
\]

于是 softmax 后这些位置权重为 0。

## 一个四位置例子

允许读取关系为：

```text
position 1: [1]
position 2: [1,2]
position 3: [1,2,3]
position 4: [1,2,3,4]
```

矩阵上是下三角结构。

## Causal Mask 与 Padding Mask

两者都可以通过把某些 attention scores 屏蔽掉实现，但语义不同：

- causal mask：因为时间因果关系，未来信息不应该被看见；
- padding mask：某些位置只是为了 batch shape 补齐，不代表真实数据。

ACT 的 training-only action-sequence encoder 需要处理 padded future action positions，但它并不是为了做 autoregressive action generation，因此不要把 padding mask 自动解释成 causal mask。

## Sources

- Vaswani et al., **Attention Is All You Need**, 2017. https://arxiv.org/abs/1706.03762
