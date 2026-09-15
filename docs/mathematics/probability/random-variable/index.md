---
title: "Random Variable"
kind: "canonical"
domain: "Mathematics / Probability"
parent: "Probability"
canonical: "/mathematics/probability/random-variable/"
prerequisites: []
related:
  - "/mathematics/probability/probability-distribution/"
  - "/mathematics/probability/expectation/"
  - "/mathematics/probability/variance/"
---

# Random Variable

Random Variable 是把随机实验的 outcome 映射为数值的函数。

给定 sample space $\Omega$，实值 random variable 定义为：

\[
X:\Omega\rightarrow\mathbb R.
\]

对一次具体 outcome $\omega\in\Omega$，random variable 给出数值：

\[
X(\omega).
\]

随机性来自实验 outcome $\omega$ 的不确定性；函数 $X$ 本身是确定的映射。

## Sample Space 与 Numerical Quantity

Sample space 描述原始结果，而 random variable 提取其中关心的数值。

例如两次抛硬币：

\[
\Omega=\{HH,HT,TH,TT\}.
\]

定义 $X$ 为正面次数：

\[
X(HH)=2,
\quad
X(HT)=X(TH)=1,
\quad
X(TT)=0.
\]

于是复杂 outcome 被映射为一个 numerical variable。

## Discrete Random Variable

如果 $X$ 的取值集合 finite 或 countable，则称为 discrete random variable。

其 distribution 可由 probability mass function 描述：

\[
p_X(x)=P(X=x).
\]

满足：

\[
p_X(x)\ge0,
\qquad
\sum_x p_X(x)=1.
\]

## Continuous Random Variable

Continuous random variable 常通过 probability density function：

\[
f_X(x)
\]

描述。

区间概率为：

\[
P(a\le X\le b)
=
\int_a^b f_X(x)\,dx.
\]

对于连续分布，单点通常满足：

\[
P(X=x)=0,
\]

即使 $f_X(x)$ 本身可以大于 0。Density 不是 point probability。

## Cumulative Distribution Function

任意 real-valued random variable 都可以定义 CDF：

\[
F_X(x)=P(X\le x).
\]

CDF 具有：

- non-decreasing；
- right-continuous；
- $\lim_{x\to-\infty}F_X(x)=0$；
- $\lim_{x\to\infty}F_X(x)=1$。

它同时适用于 discrete、continuous 与 mixed distributions。

## Transformation of a Random Variable

给定 deterministic function $g$，可以定义新的 random variable：

\[
Y=g(X).
\]

例如：

\[
Y=X^2.
\]

虽然 $g$ 是确定性的，但由于 input $X$ 随机，$Y$ 仍然随机。

Distribution transformation 需要根据 $g$ 与 $X$ 的 distribution 推导 $Y$ 的 distribution。

## Random Vector

多个 random variables 可以组成 random vector：

\[
X=
(X_1,\ldots,X_d)^\top.
\]

其 distribution 是 joint distribution：

\[
p(x_1,\ldots,x_d).
\]

Random vector 可以具有 mean vector 与 covariance matrix，是 multivariate statistics、Gaussian models 与 latent-variable models 的基本对象。

## Random Variable 与 Observation

Random variable $X$ 描述数据生成过程中的不确定 quantity；具体观测值常记为小写：

\[
x.
\]

例如：

\[
X\sim\mathcal N(0,1),
\]

表示 random variable 的 distribution；

\[
x=0.73
\]

表示某次 observation。

区分 random variable 与 realized value 对理解 expectation、likelihood 与 probabilistic models 很重要。

## Functions of Multiple Random Variables

若：

\[
Z=g(X,Y),
\]

则 $Z$ 也是 random variable。

例如：

\[
Z=X+Y.
\]

其 distribution 取决于 $X,Y$ 的 joint distribution，而不只取决于各自 marginal distribution。

这也是 independence、covariance 与 convolution 等概念的重要背景。

## Connections

- [Probability Distribution](/mathematics/probability/probability-distribution/)：描述 random variable 取值的概率规律。
- [Expectation](/mathematics/probability/expectation/)：random variable 的概率加权平均。
- [Variance](/mathematics/probability/variance/)：衡量 random variable 相对 mean 的波动。
- [Latent Variable](/mathematics/probability/latent-variable/)：未被直接观测的 random variable。
