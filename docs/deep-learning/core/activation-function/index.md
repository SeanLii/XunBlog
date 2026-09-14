---
title: "Activation Function"
kind: "canonical"
domain: "Deep Learning / Core"
parent: "Deep Learning"
canonical: "/deep-learning/core/activation-function/"
prerequisites:
  - "/deep-learning/core/linear-layer/"
related:
  - "/deep-learning/transformer/position-wise-feed-forward-network/"
---

# Activation Function

Activation Function 是施加在 neural network intermediate values 上的 nonlinear function。

最简单的 layer 可以写成：

\[
h=\phi(Wx+b),
\]

其中 $\phi$ 就是 activation function。

它最重要的作用不是“模拟神经元是否激活”，而是给网络加入 **nonlinearity**。如果没有 nonlinear activation，多层 Linear Layers 仍然只能合并成一个 affine mapping。

## ReLU

Rectified Linear Unit：

\[
\operatorname{ReLU}(x)=\max(0,x).
\]

也就是：

\[
\operatorname{ReLU}(x)=
\begin{cases}
0,&x<0\\
x,&x\ge0.
\end{cases}
\]

ReLU 的优点是简单、正区间 gradient 为常数 1，并且不会像 sigmoid 那样在大正值区域自动饱和。

但当输入长期落在负区间时，gradient 为 0，可能出现所谓 dead ReLU。

## Sigmoid

\[
\sigma(x)=\frac1{1+e^{-x}}.
\]

输出范围：

\[
0<\sigma(x)<1.
\]

因此它适合把 scalar logit 转成 Bernoulli probability，也常出现在 gates 中。

但作为深层 hidden activation 时，极大或极小输入会进入 saturation：

\[
\sigma'(x)=\sigma(x)(1-\sigma(x))
\]

接近 0，导致 gradient 传播困难。

## Tanh

\[
\tanh(x)
=\frac{e^x-e^{-x}}{e^x+e^{-x}}.
\]

输出范围：

\[
-1<\tanh(x)<1.
\]

相比 sigmoid，它以 0 为中心，但同样会在绝对值很大的区域饱和。

## GELU

Gaussian Error Linear Unit 常用于 Transformer：

\[
\operatorname{GELU}(x)=x\Phi(x),
\]

其中 $\Phi(x)$ 是 standard normal CDF。

直觉上它不像 ReLU 那样在 0 处硬截断，而是根据输入大小平滑地控制保留比例。

实际实现常使用近似形式。

## Activation 改变的是函数族

考虑两层网络：

\[
f(x)=W_2\phi(W_1x+b_1)+b_2.
\]

只要 $\phi$ 是合适的 nonlinear function，网络就不再能被折叠成一个单一 affine transformation。

深度的价值由此开始出现：不同层可以逐步构造更复杂的 piecewise / smooth nonlinear mappings。

## 输出层的 Activation 取决于建模对象

Hidden activation 与 output transformation 不应该混在一起。

例如：

- binary probability：sigmoid；
- categorical probability：[Softmax](/deep-learning/core/softmax/)；
- unrestricted regression：可能不使用 bounded activation；
- positive scale：可能使用 softplus 或 exponential。

选择 activation 的依据是输出对象需要满足什么数学约束，而不是固定模板。

## Transformer 中的位置

Transformer 的 [Position-Wise Feed-Forward Network](/deep-learning/transformer/position-wise-feed-forward-network/) 通常具有：

\[
\operatorname{FFN}(x)
=W_2\phi(W_1x+b_1)+b_2.
\]

原始 Transformer 使用 ReLU，许多后续模型使用 GELU、SwiGLU 等。

这只是 Activation Function 的一个应用环境；activation 本身是 neural network 非线性建模的基础机制。
