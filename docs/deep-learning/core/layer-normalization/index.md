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
  - "/deep-learning/cnn/resnet/residual-connection/"
  - "/deep-learning/transformer/"
---

# Layer Normalization

Layer Normalization（LayerNorm）对**单个样本内部的一组 features**计算 mean 和 variance，然后把这些 features 标准化，再施加可学习的 scale 与 bias。

对 feature vector：

\[
x=[x_1,\ldots,x_d],
\]

先计算：

\[
\mu=\frac1d\sum_{i=1}^{d}x_i,
\]

\[
\sigma^2=\frac1d\sum_{i=1}^{d}(x_i-\mu)^2.
\]

然后：

\[
\hat x_i
=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}.
\]

最后：

\[
y_i=\gamma_i\hat x_i+\beta_i.
\]

其中 $\gamma,\beta$ 是 learned parameters。

## 一次 LayerNorm 改变什么

标准化部分把当前 feature vector 调整到大致：

\[
\text{mean}\approx0,
\qquad
\text{variance}\approx1.
\]

但如果永远强制每个 feature 的最终 scale 和 offset 固定，会限制模型表达能力。

所以 LayerNorm 在标准化后加入：

\[
\gamma_i,\beta_i.
\]

模型可以重新学习每个 feature 适合的尺度与偏移。

## 它沿哪个 Dimension 计算

对于 Transformer hidden states：

\[
X\in\mathbb R^{B\times N\times d_{model}},
\]

LayerNorm 通常独立处理每个 token：

\[
X[b,n,:].
\]

也就是沿最后一个 feature dimension $d_{model}$ 计算 statistics。

不同 batch items 之间不会互相计算 mean；不同 sequence positions 也通常各自独立。

这是理解 LayerNorm 与 BatchNorm 区别的关键。

## 与 Batch Normalization 的差别

BatchNorm 典型做法是在 mini-batch / spatial dimensions 上为每个 channel 估计 statistics，因此计算依赖 batch composition，并且 training 与 inference 常使用不同 statistics 处理方式。

LayerNorm 则对单个 sample 的 feature dimensions 计算：

- 不需要 batch-level statistics；
- batch size 变化不会改变 normalization definition；
- training 与 inference 使用同一种即时计算方式。

Ba、Kiros 与 Hinton 在 2016 年提出 Layer Normalization 时，正是为了避免 BatchNorm 在 recurrent sequence settings 中的一些不便。

## $\epsilon$ 的作用

如果 variance 很小：

\[
\sigma^2\approx0,
\]

直接除以 $\sigma$ 会造成 numerical instability。

因此使用：

\[
\sqrt{\sigma^2+\epsilon}.
\]

$\epsilon$ 是数值稳定项，不是模型用来控制 normalization strength 的主要 hyperparameter。

## Pre-Norm 与 Post-Norm

在 Transformer 中，LayerNorm 与 residual connection 的相对位置形成不同 architecture。

Post-Norm：

\[
y=\operatorname{LN}(x+F(x)).
\]

Pre-Norm：

\[
y=x+F(\operatorname{LN}(x)).
\]

原始 Transformer 使用 post-norm；很多后续大模型更常使用 pre-norm 或相关变体，因为深层 optimization properties 不同。

这些是 Transformer design choices，但 LayerNorm 自身仍然是一个通用 normalization method。

## Sources

- Ba, Kiros, Hinton. *Layer Normalization*. 2016.
