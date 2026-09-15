---
title: "Convolution"
kind: "canonical"
domain: "Mathematics / Analysis"
parent: "Mathematics"
canonical: "/mathematics/analysis/convolution/"
prerequisites:
  - "/mathematics/linear-algebra/vector/"
related:
  - "/deep-learning/cnn/convolutional-neural-network/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Convolution

> **知识边界**：本文的 canonical 对象是 **Convolution**。依赖机制由 [Vector](/mathematics/linear-algebra/vector/) 的 canonical page 定义；本文只在当前语境中调用其接口。


Convolution 是把两个 functions / sequences 按相对位移进行乘积累积的运算。它在 signal processing、probability、differential equations 与 convolutional neural networks 中都有独立意义。

## Continuous Convolution

对 functions $f,g$：

\[
(f*g)(t)
=
\int_{-\infty}^{\infty}
f(\tau)g(t-\tau)\,d\tau.
\]

对固定 $t$，把一个 function 翻转并平移后，与另一个 function 做 pointwise product，再对整个 domain 积分。

## Discrete Convolution

对 discrete sequences：

\[
(f*g)[n]
=
\sum_{k=-\infty}^{\infty}
f[k]g[n-k].
\]

Finite sequences / kernels 时，sum 只在有效 overlap 区域计算。

## Kernel Reversal

严格 convolution 中出现：

\[
g[n-k]
\]

而不是 $g[n+k]$。这包含 kernel reversal。

Deep-learning libraries 中名为 convolution 的操作通常实际实现 cross-correlation：

\[
y[n]
=
\sum_k f[n+k]g[k].
\]

由于 CNN kernel parameters 是学习得到的，这个 reversal 差异通常不会改变 model capacity，但数学定义上二者不同。

## Algebraic Properties

在适当条件下，convolution 满足：

### Commutativity

\[
f*g=g*f.
\]

### Associativity

\[
(f*g)*h=f*(g*h).
\]

### Distributivity

\[
f*(g+h)=f*g+f*h.
\]

这些性质使多个 linear filters 可以组合和重排。

## Identity

Dirac delta 在 continuous convolution 中充当 identity：

\[
f*\delta=f.
\]

Discrete case 中 Kronecker delta 具有类似作用。

## Linear Time-Invariant Systems

对于 linear time-invariant（LTI）system，若 impulse response 为 $h$，任意 input $x$ 的 output 可以写成：

\[
y=x*h.
\]

因此 convolution 是 LTI systems 的自然 representation。

## Filtering

不同 kernels 可以执行不同 local filtering，例如 smoothing、edge detection 或 differentiation approximation。

例如一维 moving-average kernel：

\[
h=
\frac13[1,1,1]
\]

对 signal 做 convolution 会产生 local average。

## Two-Dimensional Convolution

对 image-like data：

\[
I\in\mathbb R^{H\times W},
\]

2D convolution：

\[
Y[i,j]
=
\sum_{u,v}
I[i-u,j-v]K[u,v].
\]

Kernel 在 spatial grid 上平移，同一 coefficients 在所有 locations 重复使用。

这产生 translation-equivariant local operator。

## Multi-Channel Data

CNN input 常为：

\[
X\in\mathbb R^{C_{in}\times H\times W}.
\]

Kernel bank：

\[
W\in\mathbb R^{C_{out}\times C_{in}\times K_h\times K_w}.
\]

每个 output channel 对所有 input channels 的 local neighborhoods 做加权求和。

完整 neural-network structure 见 [Convolutional Neural Network](/deep-learning/cnn/convolutional-neural-network/)。

## Probability Convolution

若 independent random variables $X,Y$ 的 densities 为 $f_X,f_Y$，则 sum：

\[
Z=X+Y
\]

的 density 为：

\[
f_Z(z)
=
(f_X*f_Y)(z).
\]

因此 convolution 也描述 independent random variables 相加后的 distribution。

## Frequency-Domain Relation

Fourier transform 把 convolution 转成 multiplication：

\[
\mathcal F\{f*g\}
=
\mathcal F\{f\}
\mathcal F\{g\}.
\]

这称为 convolution theorem，是 signal processing 中分析 filtering 与高效 convolution algorithms 的基础。

## Connections

- [Convolutional Neural Network](/deep-learning/cnn/convolutional-neural-network/)：使用 learned spatial kernels。
- Probability：independent random variables 求和时 distributions 通过 convolution 组合。
