---
title: "Layer Normalization"
kind: "canonical"
domain: "Deep Learning / Core"
parent: "Deep Learning"
canonical: "/deep-learning/core/layer-normalization/"
prerequisites:
  - "/mathematics/probability/expectation/"
  - "/mathematics/probability/variance/"
related:
  - "/deep-learning/transformer/transformer-encoder/"
  - "/deep-learning/transformer/transformer-decoder/"
---

# Layer Normalization

Layer Normalization 对一个样本内部选定的特征维度进行标准化，再用可学习参数恢复模型需要的尺度与偏移。Transformer 广泛使用它来稳定深层网络中的表示。

## 定义

对一个 $d$ 维 hidden vector

\[
\mathbf x=(x_1,\ldots,x_d),
\]

先计算特征维度上的均值与方差：

\[
\mu=\frac1d\sum_{i=1}^{d}x_i,
\]

\[
\sigma^2=\frac1d\sum_{i=1}^{d}(x_i-\mu)^2.
\]

标准化后

\[
\hat x_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}},
\]

再得到

\[
y_i=\gamma_i\hat x_i+\beta_i,
\]

其中 $\gamma_i$ 和 $\beta_i$ 是可学习参数，$\epsilon$ 是防止数值不稳定的小常数。

## 与 Batch Normalization 的区别

Layer Normalization 的统计量通常来自单个样本内部的 feature dimensions，不依赖同一个 mini-batch 中其他样本。因而 sequence length 或 batch size 变化时，它仍可以按相同规则工作。

## Transformer 中的位置

原始 Transformer 论文使用 residual connection 后再做 LayerNorm，也就是常被称为 post-norm 的形式：

\[
\operatorname{LayerNorm}(x+\operatorname{Sublayer}(x)).
\]

后来的 Transformer 也常采用 pre-norm，但那是架构变体。理解 ACT 官方 Transformer 时应以实际实现配置为准，而不是把所有 Transformer 都默认成同一种 normalization 顺序。

## Sources

- [Layer Normalization — Ba, Kiros, Hinton, 2016](https://arxiv.org/abs/1607.06450)
- [Attention Is All You Need — Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
