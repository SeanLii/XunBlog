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
---

# Probability Distribution

Probability Distribution 描述一个 [Random Variable](/mathematics/probability/random-variable/) 或一组 random variables 的随机规律。

它回答的不是“这次观测是多少”，而是：

> **在所有可能结果中，不同值出现的概率怎样分配。**

如果我们知道一个 random variable 的完整 distribution，就原则上知道了它的全部概率信息。

## Discrete Distribution

对于 discrete random variable $X$，用 probability mass function：

\[
p(x)=P(X=x).
\]

满足：

\[
p(x)\ge0,
\qquad
\sum_x p(x)=1.
\]

例如公平骰子：

\[
p(x)=\frac16,
\qquad
x\in\{1,2,3,4,5,6\}.
\]

## Continuous Distribution

对于 continuous random variable，用 probability density function：

\[
p(x)\ge0,
\qquad
\int_{-\infty}^{\infty}p(x)\,dx=1.
\]

区间概率由面积给出：

\[
P(a\le X\le b)
=\int_a^b p(x)\,dx.
\]

因此 density 可以大于 1；要求等于 1 的是整个空间下的积分，而不是每个点的函数值。

## CDF

另一种统一描述 distribution 的方式是 cumulative distribution function：

\[
F_X(x)=P(X\le x).
\]

CDF 对 discrete 和 continuous random variables 都适用，并且从 0 单调增长到 1。

## Parametric Distribution

许多 distribution 可以用少量 parameters 描述。

例如 [Normal Distribution](/mathematics/probability/normal-distribution/)：

\[
X\sim\mathcal N(\mu,\sigma^2).
\]

$\mu$ 和 $\sigma^2$ 决定整个 density 的位置与尺度。

学习这些 parameters，和学习一个具体 sample，不是同一件事。模型经常输出 distribution parameters，然后再从 distribution 取样或计算 likelihood。

## Joint Distribution

如果同时研究两个 random variables $X,Y$，需要 joint distribution：

\[
p(x,y).
\]

它描述两者一起取某组值的概率规律。

从 joint distribution 可以得到 marginal distribution，例如离散情形：

\[
p(x)=\sum_y p(x,y).
\]

连续情形对应积分：

\[
p(x)=\int p(x,y)\,dy.
\]

这个过程叫 marginalization。

## Conditional Distribution

在已经知道 $X=x$ 的条件下，$Y$ 的 distribution 为：

\[
p(y\mid x).
\]

它不是另一个无关的 distribution，而是 joint distribution 在给定条件后的重新归一化：

\[
p(y\mid x)
=\frac{p(x,y)}{p(x)},
\qquad p(x)>0.
\]

完整机制见 [Conditional Probability](/mathematics/probability/conditional-probability/)。

## Distribution 与 Dataset Histogram

Dataset histogram 是有限样本得到的 empirical summary；theoretical probability distribution 是产生数据的概率模型。

样本数增加时，empirical distribution 可以越来越接近 underlying distribution，但两者不能直接画等号。

这一区分在机器学习中很重要：训练数据是有限 observations，而模型试图学习关于数据生成规律的 distribution 或 conditional mapping。
