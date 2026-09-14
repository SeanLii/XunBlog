---
title: "Variance"
kind: "canonical"
domain: "Mathematics / Probability"
parent: "Probability"
canonical: "/mathematics/probability/variance/"
prerequisites:
  - "/mathematics/probability/expectation/"
related:
  - "/mathematics/probability/normal-distribution/"
---

# Variance

方差（variance）衡量随机变量围绕均值的离散程度。均值说明分布大致位于哪里，方差说明取值通常离这个中心有多分散。

## 定义

设

\[
\mu=\mathbb E[X].
\]

方差定义为

\[
\operatorname{Var}(X)
=\mathbb E[(X-\mu)^2].
\]

平方使正负偏差不会互相抵消，也让较大的偏差受到更大的惩罚。

等价地，

\[
\operatorname{Var}(X)
=\mathbb E[X^2]-\bigl(\mathbb E[X]\bigr)^2.
\]

标准差定义为

\[
\sigma=\sqrt{\operatorname{Var}(X)}.
\]

因此方差的单位是原变量单位的平方，而标准差与原变量单位一致。

## 尺度变化

如果

\[
Y=aX+b,
\]

那么

\[
\operatorname{Var}(Y)=a^2\operatorname{Var}(X).
\]

加上常数 $b$ 只平移整个分布，不改变离散程度；乘以 $a$ 会把所有偏差放大 $|a|$ 倍，因此方差放大 $a^2$ 倍。

## 在高斯分布中的作用

[Normal Distribution](/mathematics/probability/normal-distribution/) 用 $\mu$ 和 $\sigma^2$ 完整确定一维分布：

\[
X\sim\mathcal N(\mu,\sigma^2).
\]

VAE 常不直接预测 $\sigma$，而预测 $\log\sigma^2$。这既方便网络输出任意实数，也能通过指数变换得到严格为正的方差。
