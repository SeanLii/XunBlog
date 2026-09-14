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

Layer Normalization 对一个样本内部的 feature dimensions 做标准化，再用可学习参数恢复合适的尺度与偏移。

对一个 hidden vector

\[
x=(x_1,\ldots,x_d),
\]

先计算该向量自己的均值与方差：

\[
\mu=\frac1d\sum_i x_i,
\]

\[
\sigma^2=\frac1d\sum_i(x_i-\mu)^2.
\]

标准化：

\[
\hat x_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}.
\]

最后：

\[
y_i=\gamma_i\hat x_i+\beta_i,
\]

其中 $\gamma,\beta$ 是可学习参数。

## 它在什么维度上计算

LayerNorm 的关键是对**同一个样本/位置的 features**做 normalization，而不是依赖整个 batch 的统计量。

对于 Transformer hidden states

\[
X\in\mathbb R^{B\times n\times d},
\]

常见做法是对最后一个 hidden dimension $d$ 独立归一化每个 $(b,i)$ 位置。

## 与 Batch Normalization 的差别

BatchNorm 统计通常依赖 batch dimension；LayerNorm 不需要用其他样本来算当前样本的均值/方差，因此对可变序列长度和小 batch 更自然。

## 在 Transformer 中的位置

原始 Transformer 采用 post-norm 形式：sublayer + residual 后再 LayerNorm。后续很多模型改成 pre-norm：先 LayerNorm，再进入 sublayer。

两者都使用 LayerNorm，但 gradient behavior 与训练稳定性不同。

ACT released Transformer code也保留了这一类 Transformer normalization 结构；具体是否 pre-norm 由实现配置决定。
