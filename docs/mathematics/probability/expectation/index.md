---
title: "Expectation"
kind: "canonical"
domain: "Mathematics / Probability"
parent: "Probability"
canonical: "/mathematics/probability/expectation/"
prerequisites:
  - "/mathematics/probability/random-variable/"
  - "/mathematics/probability/probability-distribution/"
related:
  - "/mathematics/probability/variance/"
---

# Expectation

Expectation 是对 random variable 的所有可能取值按其概率进行加权后得到的平均位置。

它描述的是 distribution 的整体中心趋势，而不是某一次 observation。

对于 discrete random variable $X$：

\[
\mathbb E[X]
=\sum_x x\,P(X=x).
\]

对于 continuous random variable：

\[
\mathbb E[X]
=\int_{-\infty}^{\infty}x\,p(x)\,dx.
\]

## 一个离散例子

公平骰子：

\[
X\in\{1,2,3,4,5,6\},
\qquad P(X=x)=\frac16.
\]

因此：

\[
\mathbb E[X]
=\frac{1+2+3+4+5+6}{6}
=3.5.
\]

3.5 并不是骰子能实际掷出的点数。Expectation 不要求是可能 observation，它描述的是长期平均位置。

## Expectation of a Function

如果关心的不是 $X$ 本身，而是函数 $g(X)$：

\[
\mathbb E[g(X)]
=\sum_x g(x)p(x)
\]

或 continuous 情形：

\[
\mathbb E[g(X)]
=\int g(x)p(x)\,dx.
\]

这比 $\mathbb E[X]$ 更一般。

例如 variance 就依赖：

\[
\mathbb E[(X-\mu)^2].
\]

机器学习中的 expected loss 也是同一个结构：

\[
\mathbb E_{(x,y)\sim p_{data}}
[\ell(f(x),y)].
\]

## Linearity of Expectation

Expectation 最重要的性质之一：

\[
\mathbb E[aX+bY]
=a\mathbb E[X]+b\mathbb E[Y].
\]

这里不要求 $X$ 与 $Y$ independent。

特别地：

\[
\mathbb E\left[\sum_i X_i\right]
=\sum_i\mathbb E[X_i].
\]

这一性质让很多复杂随机量的平均值分析变得简单。

## Sample Mean 与 Expectation

Dataset 中的 sample mean：

\[
\bar x=\frac1N\sum_{i=1}^{N}x_i
\]

是用有限 observations 对 population expectation 的估计。

两者概念不同：

- expectation 属于 probability distribution；
- sample mean 属于一组已经观察到的数据。

样本足够多并满足适当条件时，sample mean 会靠近 expectation，这由 law of large numbers 描述。

## Vector-valued Expectation

如果 $X\in\mathbb R^d$ 是 random vector，则：

\[
\mathbb E[X]
=
\begin{bmatrix}
\mathbb E[X_1]\\
\vdots\\
\mathbb E[X_d]
\end{bmatrix}.
\]

这就是 multivariate distribution 的 mean vector。

Expectation 因此不仅是“求平均”的计算技巧，而是概率论中把整个 distribution 压缩为一个中心位置的基本运算。
