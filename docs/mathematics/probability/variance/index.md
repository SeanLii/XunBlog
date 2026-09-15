---
title: "Variance"
kind: "canonical"
domain: "Mathematics / Probability"
parent: "Probability"
canonical: "/mathematics/probability/variance/"
prerequisites:
  - "/mathematics/probability/expectation/"
related:
  - "/mathematics/probability/covariance/"
  - "/mathematics/probability/normal-distribution/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Variance

> **知识边界**：本文的 canonical 对象是 **Variance**。依赖机制由 [Expectation](/mathematics/probability/expectation/) 的 canonical page 定义；本文只在当前语境中调用其接口。


Variance 衡量 random variable 相对其 mean 的平方偏离程度。

设：

\[
\mu=\mathbb E[X].
\]

定义：

\[
\operatorname{Var}(X)
=
\mathbb E[(X-\mu)^2].
\]

Variance 总是 non-negative。

## Computational Identity

展开平方：

\[
(X-\mu)^2
=X^2-2\mu X+\mu^2.
\]

取 expectation：

\[
\operatorname{Var}(X)
=
\mathbb E[X^2]
-
(\mathbb E[X])^2.
\]

这个形式常用于推导与计算。

## Standard Deviation

Standard deviation 定义为：

\[
\sigma_X
=
\sqrt{\operatorname{Var}(X)}.
\]

Variance 的单位是原 variable 单位的平方，而 standard deviation 与 $X$ 具有相同单位，因此更容易直接解释尺度。

## Shift 与 Scale

对 constants $a,b$：

\[
\operatorname{Var}(aX+b)
=
a^2\operatorname{Var}(X).
\]

加 constant 不改变 variance：

\[
\operatorname{Var}(X+b)
=
\operatorname{Var}(X).
\]

乘以 $a$ 则把 variance 放大 $a^2$ 倍。

## Variance of a Sum

对两个 random variables：

\[
\operatorname{Var}(X+Y)
=
\operatorname{Var}(X)
+
\operatorname{Var}(Y)
+
2\operatorname{Cov}(X,Y).
\]

若 $X,Y$ independent，则 covariance 为 0，于是：

\[
\operatorname{Var}(X+Y)
=
\operatorname{Var}(X)
+
\operatorname{Var}(Y).
\]

注意 zero covariance 并不一般意味着 independence。

## Law of Total Variance

Variance 可以分解为：

\[
\operatorname{Var}(X)
=
\mathbb E[
\operatorname{Var}(X\mid Y)
]
+
\operatorname{Var}(
\mathbb E[X\mid Y]
).
\]

第一项描述给定 $Y$ 后仍存在的平均 uncertainty；第二项描述 conditional mean 随 $Y$ 的变化。

## Sample Variance

给定 observations：

\[
x_1,\ldots,x_n,
\]

sample mean：

\[
\bar x
=
\frac1n\sum_i x_i.
\]

常用 unbiased sample variance：

\[
s^2
=
\frac1{n-1}
\sum_{i=1}^{n}(x_i-\bar x)^2.
\]

分母使用 $n-1$ 是 Bessel correction，用来修正使用 sample mean 估计 population mean 后引入的 bias。

如果目标只是描述当前 finite dataset，而不是无偏估计 population variance，也可以使用分母 $n$。

## Variance 与 Scale

Variance 只描述 spread，不描述 distribution 的完整 shape。两个 distributions 可以具有相同 mean 与 variance，但拥有完全不同的 skewness、tails 或 multimodality。

因此：

\[
(\mu,\sigma^2)
\]

通常不足以唯一决定 distribution，除非已经指定 distribution family，例如 Gaussian。

## Chebyshev's Inequality

只要 variance finite，Chebyshev inequality 给出：

\[
P(|X-\mu|\ge k\sigma)
\le
\frac1{k^2}.
\]

它不要求 Normal distribution，因此 variance 可以为一般 distribution 提供最基本的 concentration information。

## Connections

- [Expectation](/mathematics/probability/expectation/)：Variance 是 centered second-moment expectation。
- [Covariance](/mathematics/probability/covariance/)：推广到两个 random variables 的共同变化。
- [Normal Distribution](/mathematics/probability/normal-distribution/)：$\sigma^2$ 是 Gaussian 的 scale parameter。
