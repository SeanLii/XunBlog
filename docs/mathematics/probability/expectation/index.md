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
  - "/mathematics/information-theory/kl-divergence/"
---

# Expectation

Expectation 描述一个 random variable 在其 probability distribution 下的平均位置。

它不是“一次采样最可能得到的值”，而是如果重复采样很多次，样本平均值趋向的量。

## 离散随机变量

\[
\mathbb E[X]
=
\sum_x xP(X=x).
\]

例如公平骰子：

\[
\mathbb E[X]
=\frac{1+2+3+4+5+6}{6}
=3.5.
\]

但骰子永远不会掷出 3.5。Expectation 是 distribution 的平均，不要求本身是可取值。

## 连续随机变量

若 density 为 $p(x)$：

\[
\mathbb E[X]
=
\int x p(x)\,dx.
\]

更一般地，对函数 $f(X)$：

\[
\mathbb E[f(X)]
=
\int f(x)p(x)\,dx.
\]

## Linearity of Expectation

无论 variables 是否独立：

\[
\mathbb E[aX+bY]
=a\mathbb E[X]+b\mathbb E[Y].
\]

这是概率计算中非常常用的性质。

## 与 Mean 的关系

Distribution 的理论 mean 通常就是 expectation：

\[
\mu=\mathbb E[X].
\]

而 dataset 的 sample mean

\[
\bar x=\frac1n\sum_i x_i
\]

是用有限样本估计这个 population quantity。

## VAE 中的 Expectation

ELBO 中有：

\[
\mathbb E_{q_\phi(z\mid x)}
[\log p_\theta(x\mid z)].
\]

意思是：对 $z$ 按 approximate posterior 分布取值时，decoder 对真实 $x$ 的 log-likelihood 平均是多少。

实际训练往往用少量 Monte Carlo samples 来估计这个 expectation。
