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

Activation Function 是神经网络中对 intermediate representation 施加的函数。对一个 affine transformation

\[
z=Wx+b,
\]

常见的神经网络层写成

\[
h=\phi(z),
\]

其中 $\phi$ 是 activation function。

Activation Function 的核心作用是引入 **nonlinearity**。如果连续多层都只有 affine transformation，那么

\[
W_2(W_1x+b_1)+b_2
\]

仍可合并为一个新的 affine transformation，因此增加层数不会扩展到一般的 nonlinear function family。

## Element-Wise Activation

很多 activation 对每个 scalar 独立作用：

\[
h_i=\phi(z_i).
\]

它不会直接在不同 features 之间交换信息；feature mixing 通常由 Linear Layer、Convolution、Attention 等操作完成，activation 则改变每个位置的 nonlinear response。

## ReLU

Rectified Linear Unit 定义为

\[
\operatorname{ReLU}(x)=\max(0,x).
\]

即

\[
\operatorname{ReLU}(x)=
\begin{cases}
0,&x<0,\\
x,&x\ge0.
\end{cases}
\]

它在正区间导数为 1，在负区间导数为 0。ReLU 计算简单，也避免了 sigmoid 在大正值区域的饱和，但长期处于负区间的 unit 可能持续得到零梯度，这通常称为 **dead ReLU**。

Leaky ReLU 等变体在负区间保留一个小斜率，以减弱这一问题。

## Sigmoid

Logistic sigmoid：

\[
\sigma(x)=\frac{1}{1+e^{-x}}.
\]

其输出满足

\[
0<\sigma(x)<1,
\]

导数为

\[
\sigma'(x)=\sigma(x)(1-\sigma(x)).
\]

当 $|x|$ 很大时，导数趋近 0，因此深层网络若大量使用 sigmoid hidden activations，容易出现梯度衰减。Sigmoid 仍适合表示 Bernoulli probability，也常用于 gating mechanism。

## Tanh

\[
\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}.
\]

输出范围为

\[
-1<\tanh(x)<1.
\]

Tanh 以 0 为中心，但同样在大绝对值区域饱和。它在 recurrent networks 与某些 bounded-state models 中仍然常见。

## GELU

Gaussian Error Linear Unit 定义为

\[
\operatorname{GELU}(x)=x\Phi(x),
\]

其中 $\Phi(x)$ 是 standard normal CDF。GELU 是平滑的非线性函数，并允许小的负输入保留非零输出。BERT 等 Transformer 模型使用 GELU，而原始 Transformer 使用 ReLU。

工程实现常使用近似形式，例如

\[
\operatorname{GELU}(x)
\approx
\frac{x}{2}
\left[
1+\tanh\left(
\sqrt{\frac{2}{\pi}}
\left(x+0.044715x^3\right)
\right)
\right].
\]

## SiLU / Swish

SiLU（Sigmoid Linear Unit）定义为

\[
\operatorname{SiLU}(x)=x\sigma(x).
\]

它与 Swish 的常用形式等价，是另一类平滑、非单调的 activation。在现代卷积网络和部分 Transformer-style blocks 中较常见。

## Saturation and Gradient Flow

Activation Function 会直接影响 local derivative：

\[
\frac{\partial h}{\partial z}=\phi'(z).
\]

深层网络的 gradient 需要经过很多这样的 Jacobian factors。若 activation 在大范围内导数接近 0，梯度可能快速衰减；若导数和权重组合长期导致放大，也可能产生 exploding gradients。

因此 activation 的选择不仅决定 forward function family，也会影响 optimization dynamics。

## Hidden Activation and Output Transformation

Hidden activation 与输出层 transformation 的职责不同。输出层应由建模对象决定，例如：

- binary probability：sigmoid；
- categorical probability：[Softmax](/deep-learning/core/softmax/)；
- unrestricted real-valued regression：常直接输出实数；
- positive scalar：可使用 softplus 或 exponential；
- bounded interval：可按目标范围选择 sigmoid、tanh 或重新缩放。

因此不存在适用于所有 layer 的固定 activation。

## Function Approximation

带 nonlinear activation 的 [Multilayer Perceptron](/deep-learning/core/multilayer-perceptron/) 可以表示远比单个 affine mapping 更复杂的函数。Universal approximation results 说明，在一定条件下，具有足够 hidden units 的前馈网络可以逼近很广泛的连续函数类别。

这类定理说明了非线性网络的表达能力，但不意味着任意 architecture 都容易训练，也不说明所需宽度、数据量或 optimization cost 很小。

## Transformer 中的使用

Transformer 的 [Position-Wise Feed-Forward Network](/deep-learning/transformer/position-wise-feed-forward-network/) 通常写成

\[
\operatorname{FFN}(x)=W_2\phi(W_1x+b_1)+b_2.
\]

原始 Transformer 使用 ReLU，BERT 使用 GELU；后续模型还广泛使用 gated variants，例如 GLU / SwiGLU。Activation Function 在这里仍承担同一个基本职责：在两次 learned affine transformations 之间提供 nonlinearity。

## Sources

- Nair, Hinton. *Rectified Linear Units Improve Restricted Boltzmann Machines*. 2010.
- Hendrycks, Gimpel. *Gaussian Error Linear Units (GELUs)*. 2016.
- Ramachandran, Zoph, Le. *Searching for Activation Functions*. 2017.
- Cybenko. *Approximation by Superpositions of a Sigmoidal Function*. 1989.
