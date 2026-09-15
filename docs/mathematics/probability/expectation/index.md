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
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Expectation

> **知识边界**：本文的 canonical 对象是 **Expectation**。依赖机制由 [Random Variable](/mathematics/probability/random-variable/)、[Probability Distribution](/mathematics/probability/probability-distribution/) 的 canonical page 定义；本文只在当前语境中调用其接口。


Expectation（期望）是 random variable 在其 probability distribution 下的概率加权平均。

对 discrete random variable：

\[
\mathbb E[X]
=
\sum_x x\,p_X(x).
\]

对 continuous random variable：

\[
\mathbb E[X]
=
\int_{-\infty}^{\infty}x f_X(x)\,dx,
\]

前提是相应积分存在。

Expectation 描述 distribution 的平均位置，但不意味着 random variable 必须实际取到这个值。

## Expectation of a Function

对 function $g$：

\[
\mathbb E[g(X)]
=
\sum_x g(x)p_X(x)
\]

或：

\[
\mathbb E[g(X)]
=
\int g(x)f_X(x)\,dx.
\]

这称为 law of the unconscious statistician（LOTUS）：不需要先求 $Y=g(X)$ 的完整 distribution 就能计算 $\mathbb E[g(X)]$。

## Linearity

Expectation 最重要的性质之一是 linearity：

\[
\mathbb E[aX+bY]
=
a\mathbb E[X]+b\mathbb E[Y].
\]

它不要求 $X,Y$ independent。

进一步：

\[
\mathbb E\left[\sum_iX_i\right]
=
\sum_i\mathbb E[X_i].
\]

## Indicator Variables

对 event $A$，定义 indicator：

\[
\mathbf 1_A=
\begin{cases}
1,&A\text{ occurs},\\
0,&\text{otherwise}.
\end{cases}
\]

则：

\[
\mathbb E[\mathbf 1_A]
=P(A).
\]

这个性质常用于把 counting problem 转成 expectation problem。

## Conditional Expectation

给定 $Y=y$：

\[
\mathbb E[X\mid Y=y]
\]

是在 conditional distribution $p(x\mid y)$ 下对 $X$ 求 expectation。

Random-variable form：

\[
\mathbb E[X\mid Y]
\]

是 $Y$ 的函数。

## Tower Property

Conditional expectation 满足：

\[
\mathbb E[
\mathbb E[X\mid Y]
]
=
\mathbb E[X].
\]

更一般地，若 $\mathcal G\subseteq\mathcal H$ 是 information structures：

\[
\mathbb E[
\mathbb E[X\mid\mathcal H]
\mid\mathcal G]
=
\mathbb E[X\mid\mathcal G].
\]

## Expectation of Products

一般情况下：

\[
\mathbb E[XY]
\neq
\mathbb E[X]\mathbb E[Y].
\]

若 $X,Y$ independent 且 expectations 存在，则：

\[
\mathbb E[XY]
=
\mathbb E[X]\mathbb E[Y].
\]

这个区别与 covariance 直接相关。

## Sample Mean 与 Population Expectation

给定 i.i.d. samples：

\[
X_1,\ldots,X_N,
\]

sample mean：

\[
\bar X
=
\frac1N\sum_{i=1}^{N}X_i
\]

用于估计 population mean：

\[
\mu=\mathbb E[X].
\]

在适当条件下，law of large numbers 给出：

\[
\bar X\rightarrow\mathbb E[X]
\]

随着 sample size 增大成立相应收敛。

## Vector-Valued Expectation

对 random vector：

\[
X\in\mathbb R^d,
\]

expectation 按 component 定义：

\[
\mathbb E[X]
=
\begin{bmatrix}
\mathbb E[X_1]\\
\vdots\\
\mathbb E[X_d]
\end{bmatrix}.
\]

这称为 mean vector。

## Expectation in Machine Learning

很多 learning objectives 都写成 population expectation：

\[
\mathcal L(\theta)
=
\mathbb E_{(x,y)\sim p_{data}}
[\ell(f_\theta(x),y)].
\]

实际训练使用 finite-sample average 或 minibatch estimate：

\[
\hat{\mathcal L}
=
\frac1B\sum_{i=1}^{B}\ell_i.
\]

Stochastic optimization 中常需要估计 expectation 及其 gradient。

## Connections

- [Variance](/mathematics/probability/variance/)：以 expectation 定义 centered second moment。
- [Entropy](/mathematics/information-theory/entropy/)：是 self-information 的 expectation。
- [Evidence Lower Bound](/mathematics/probability/variational-inference/evidence-lower-bound/)：包含对 variational posterior 的 expectation。
