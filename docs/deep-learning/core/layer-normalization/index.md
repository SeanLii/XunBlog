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
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Layer Normalization

Layer Normalization（LayerNorm）对单个样本内部指定的一组 features 计算 [Expectation](/mathematics/probability/expectation/) 与 [Variance](/mathematics/probability/variance/)，完成标准化后再施加可学习的 affine transformation。

对 feature vector

\[
x=(x_1,\ldots,x_d),
\]

定义

\[
\mu=\frac{1}{d}\sum_{i=1}^d x_i,
\qquad
\sigma^2=\frac{1}{d}\sum_{i=1}^d(x_i-\mu)^2.
\]

标准化结果为

\[
\hat x_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}},
\]

最终输出

\[
y_i=\gamma_i\hat x_i+\beta_i,
\]

其中 $\gamma,\beta$ 是 learned scale 与 bias。

## Normalization Axis

LayerNorm 的定义关键在于 **statistics 在单个样本内部计算**。

对 Transformer hidden states

\[
X\in\mathbb R^{B\times N\times d_{model}},
\]

若 `normalized_shape = d_model`，则对每个 $(b,n)$ 独立计算

\[
X[b,n,:].
\]

因此：

- 不同 batch items 不共享 mean / variance；
- 不同 token positions 通常各自计算 statistics；
- normalized dimensions 由实现中的 `normalized_shape` 决定，而不是固定只能归一化最后一个 scalar dimension。

## Learnable Affine Parameters

纯标准化会把 feature vector 强制变为近似零均值、单位方差。LayerNorm 随后通过

\[
\gamma\odot \hat x+\beta
\]

允许模型重新学习每个 normalized feature 的尺度和偏移。

因此 LayerNorm 不是把所有 representation 永久限制在标准正态分布；它只是对当前 sample 的 statistics 做确定性 normalization，再交给可学习 affine parameters 调整。

## Numerical Stability

当 $\sigma^2$ 很小时，直接除以 $\sigma$ 会产生数值不稳定。LayerNorm 使用

\[
\sqrt{\sigma^2+\epsilon}
\]

作为 denominator。$\epsilon$ 的职责是数值稳定，不是主要的 representation-control hyperparameter。

## Relation to Batch Normalization

Batch Normalization 通常对一个 channel 在 mini-batch 与 spatial dimensions 上估计 statistics，因此其结果依赖 batch composition，并在 training / inference 间常维护 running statistics。

LayerNorm 则：

- 对每个样本独立计算 statistics；
- 不需要 running mean / variance；
- training 与 inference 使用同一种 normalization rule；
- 对 variable batch size 与 sequence modeling 更自然。

两者都属于 normalization methods，但 normalization axes 与统计依赖完全不同。

## Invariance Properties

对 scalar $a>0$ 与常数偏移 $c$，若忽略 $\epsilon$，对同一 normalized group 进行

\[
x' = ax+c\mathbf 1
\]

会得到相同的 normalized direction：

\[
\frac{x'-\mu'}{\sigma'}
=
\frac{x-\mu}{\sigma}.
\]

这说明 LayerNorm 会消除组内统一的 shift，并对正的统一 scaling 具有归一化不变性。Learned $\gamma,\beta$ 再重新引入任务所需的 scale 与 offset。

## Pre-Norm and Post-Norm

在 Transformer 中，LayerNorm 与 residual branch 的相对位置会改变 optimization behavior。

Post-Norm：

\[
y=\operatorname{LN}(x+F(x)).
\]

Pre-Norm：

\[
y=x+F(\operatorname{LN}(x)).
\]

原始 Transformer 使用 post-norm；许多现代深层 Transformer 使用 pre-norm 或相关变体。二者不是 LayerNorm 本身的不同定义，而是 LayerNorm 在 residual architecture 中的不同 placement。

## Related Normalization Methods

RMSNorm 等方法保留了“按单个 token / sample feature group 归一化”的思想，但不减去 mean，而只使用 root mean square 进行 scaling。它们与 LayerNorm 相关，但不是同一个 operation。

## 由机制产生的边界

LayerNorm 也会改变 representation 的绝对尺度信息。如果某个任务需要显式保留这类尺度，architecture 必须通过 residual path、learned affine parameters 或其他分支重新表达。

此外，LayerNorm 并不能单独解决深层网络的所有 optimization 问题；初始化、residual design、learning rate、activation 与 architecture depth 仍然共同决定训练稳定性。

## Sources

- Ba, Kiros, Hinton. *Layer Normalization*. 2016.
- Xiong et al. *On Layer Normalization in the Transformer Architecture*. 2020.
