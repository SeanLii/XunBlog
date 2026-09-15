---
title: "Probability Distribution"
kind: "canonical"
domain: "Mathematics / Probability"
parent: "Probability"
canonical: "/mathematics/probability/probability-distribution/"
prerequisites:
  - "/mathematics/probability/random-variable/"
related:
  - "/mathematics/probability/conditional-probability/"
  - "/mathematics/probability/normal-distribution/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Probability Distribution

> **知识边界**：本文的 canonical 对象是 **Probability Distribution**。依赖机制由 [Random Variable](/mathematics/probability/random-variable/) 的 canonical page 定义；本文只在当前语境中调用其接口。


Probability Distribution 描述 random variable 或 random vector 的概率质量如何分配在可能取值上。

若：

\[
X\sim p,
\]

表示 $X$ 的 distribution 为 $p$。Distribution 决定了关于 $X$ 的概率陈述，例如事件概率、expectation、variance 与 quantiles。

## Discrete Distribution

对 discrete random variable，使用 probability mass function：

\[
p_X(x)=P(X=x).
\]

满足：

\[
p_X(x)\ge0,
\qquad
\sum_x p_X(x)=1.
\]

例如 Bernoulli random variable：

\[
P(X=1)=p,
\qquad
P(X=0)=1-p.
\]

## Continuous Distribution

对 continuous random variable，通常使用 probability density function：

\[
f_X(x)\ge0,
\]

且：

\[
\int_{-\infty}^{\infty}f_X(x)\,dx=1.
\]

区间 probability：

\[
P(a\le X\le b)
=
\int_a^b f_X(x)\,dx.
\]

Density value 本身不是 probability；它可以大于 1，只要积分仍为 1。

## Cumulative Distribution Function

CDF 定义为：

\[
F_X(x)=P(X\le x).
\]

对 absolutely continuous distribution：

\[
F_X(x)
=
\int_{-\infty}^{x}f_X(t)\,dt,
\]

若可微，则：

\[
f_X(x)=F_X'(x).
\]

CDF 是对所有 real-valued distributions 都统一适用的表示。

## Support

Distribution 的 support 描述可能出现概率质量的区域。

例如：

- Bernoulli：$\{0,1\}$；
- Exponential：$[0,\infty)$；
- Normal：$\mathbb R$。

Support 是 distribution definition 的一部分。若 model 给一个 support 外的 observation 分配非零 likelihood，或反过来，可能产生建模错误。

## Parametric Distribution

Parametric family 用有限参数表示 distribution：

\[
p(x;\theta).
\]

例如 Normal distribution：

\[
\theta=(\mu,\sigma^2).
\]

Learning 常转化为估计 $\theta$。

## Joint Distribution

多个 random variables 的联合行为由 joint distribution 描述：

\[
p(x,y).
\]

它包含比单独 marginals：

\[
p(x),\qquad p(y)
\]

更多的信息，因为还描述 $X$ 与 $Y$ 的 dependency。

## Marginal Distribution

从 joint distribution 中消去其他 variables：

离散情况：

\[
p(x)=\sum_y p(x,y).
\]

连续情况：

\[
p(x)=\int p(x,y)\,dy.
\]

这个操作称为 marginalization。

## Conditional Distribution

若 $p(y)>0$，conditional distribution 为：

\[
p(x\mid y)
=
\frac{p(x,y)}{p(y)}.
\]

于是 joint distribution 可以 factorize：

\[
p(x,y)=p(x\mid y)p(y).
\]

不同 factorization 是 probabilistic graphical models 与 generative modeling 的基础。

## Independence

若：

\[
p(x,y)=p(x)p(y),
\]

则 $X$ 与 $Y$ independent。

此时：

\[
p(x\mid y)=p(x)
\]

（在条件概率有定义的地方）。

## Transformation of Distributions

若：

\[
Y=g(X),
\]

则 $Y$ 的 distribution 由 $X$ 的 distribution 与 transformation $g$ 决定。

在一维、可逆且可微时：

\[
f_Y(y)
=
f_X(g^{-1}(y))
\left|
\frac{d}{dy}g^{-1}(y)
\right|.
\]

多维情况需要 Jacobian determinant。

## Population Distribution 与 Empirical Distribution

理论中的 $p(x)$ 通常表示 population / data-generating distribution。

有限 dataset：

\[
\{x_1,\ldots,x_N\}
\]

定义 empirical distribution：

\[
\hat p_N(x)
=
\frac1N\sum_{i=1}^{N}\delta_{x_i}(x).
\]

Machine learning 使用有限 samples 推断或逼近更广泛的数据 distribution。

## Likelihood

给定 parametric model $p_\theta(x)$ 与 observed dataset：

\[
D=\{x_i\}_{i=1}^{N},
\]

likelihood 为：

\[
L(\theta;D)
=
\prod_{i=1}^{N}p_\theta(x_i)
\]

（在 i.i.d. 假设下）。

Maximum likelihood 通过：

\[
\max_\theta\sum_i\log p_\theta(x_i)
\]

让 model distribution 对 observations 分配更高概率或 density。

## Connections

- [Random Variable](/mathematics/probability/random-variable/)：distribution 描述的对象。
- [Conditional Probability](/mathematics/probability/conditional-probability/)：conditional distributions 的基础。
- [Normal Distribution](/mathematics/probability/normal-distribution/)：常用 parametric distribution family。
