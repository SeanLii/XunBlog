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
---

# Convolution

Convolution 是把一个函数或离散序列与另一个函数进行“滑动、乘积、累加”的运算。

它远早于 CNN，也广泛出现在 signal processing、probability、differential equations 与 machine learning 中。

因此卷积本身属于数学 / 信号处理基础，而 CNN 只是它后来最重要的应用之一。

## 连续形式

对两个函数 $f$ 与 $g$：

\[
(f*g)(t)
=
\int_{-\infty}^{\infty}
f(\tau)g(t-\tau)\,d\tau.
\]

理解这个公式可以分成三步：

1. 固定一个输出位置 $t$；
2. 让 $g(t-\tau)$ 相对 $f(\tau)$ 滑动；
3. 把所有位置的乘积积分起来。

输出仍然是关于 $t$ 的新函数。

## 离散形式

对离散序列：

\[
(f*g)[n]
=
\sum_{k=-\infty}^{\infty}
f[k]g[n-k].
\]

有限序列中，只对实际重叠的位置求和。

直觉仍然相同：

```text
一个序列固定
另一个序列翻转并滑动
每个位置：
逐元素相乘
↓
求和
↓
得到一个输出值
```

## Kernel Flipping

标准 convolution 中出现：

\[
g[n-k]
\]

而不是：

\[
g[n+k].
\]

这个索引结构等价于先翻转 kernel，再做滑动 inner product。

因此数学意义上的 convolution 与 cross-correlation 不完全相同。

## Cross-Correlation

离散 cross-correlation 常写成：

\[
(f\star g)[n]
=
\sum_k f[k]g[n+k].
\]

它没有对 kernel 做相同意义上的翻转。

现代 deep-learning framework 中所谓 `Conv2d` 通常实际实现 cross-correlation，但由于 kernel weights 本身是学习出来的，这个差别不会妨碍 CNN 学习；领域里仍沿用“convolution”这个名称。

## 二维 Convolution

对二维输入：

\[
X\in\mathbb R^{H\times W},
\]

kernel：

\[
K\in\mathbb R^{k_h\times k_w},
\]

CNN-style cross-correlation 常写：

\[
Y_{i,j}
=
\sum_{u=0}^{k_h-1}
\sum_{v=0}^{k_w-1}
K_{u,v}X_{i+u,j+v}.
\]

每个输出位置读取一个 local patch。

## Convolution 的代数性质

在条件合适时，convolution 满足：

### Commutative

\[
f*g=g*f.
\]

### Associative

\[
(f*g)*h=f*(g*h).
\]

### Distributive

\[
f*(g+h)=f*g+f*h.
\]

这些性质解释了为什么连续多个 linear time-invariant filters 可以组合成一个等效 filter。

## Convolution 与 Filtering

在 signal processing 中，kernel / impulse response 决定系统如何处理输入。

例如平滑 kernel 会压低快速变化，高通 kernel 会突出局部变化。

因此 convolution 可以理解为：

> 用一个局部 pattern / response function，在所有位置以相同规则处理输入。

## CNN 中的 Convolution

CNN 把 kernel coefficients 从人工指定变成 learned parameters，并扩展到多 channels：

\[
X\in\mathbb R^{C_{in}\times H\times W},
\]

\[
K\in\mathbb R^{C_{out}\times C_{in}\times k_h\times k_w}.
\]

这同时利用：

- local connectivity；
- weight sharing；
- channel mixing。

Stride、padding、dilation、receptive field 等网络级设计放在 [Convolutional Neural Network](/deep-learning/cnn/convolutional-neural-network/) 中展开。

## Sources

- Standard signal-processing definition of convolution.
- Goodfellow, Bengio, Courville. *Deep Learning*, convolutional networks chapter.
