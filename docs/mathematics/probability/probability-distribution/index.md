---
title: "Probability Distribution"
kind: "canonical"
domain: "Mathematics / Probability"
parent: "Probability"
canonical: "/mathematics/probability/probability-distribution/"
prerequisites:
  - "/mathematics/probability/random-variable/"
related:
  - "/mathematics/probability/normal-distribution/"
  - "/mathematics/probability/conditional-probability/"
---

# Probability Distribution

Probability Distribution 描述一个 random variable 的概率如何分布在不同可能取值上。

它回答的不是“这一次一定是多少”，而是：

> 在重复观察这个随机过程时，各种结果分别有多可能出现？

## 离散分布

对离散 random variable $X$，Probability Mass Function（PMF）为

\[
p(x)=P(X=x).
\]

满足

\[
p(x)\ge0,
\qquad
\sum_xp(x)=1.
\]

例如公平硬币可以定义

\[
P(X=0)=0.5,
\qquad
P(X=1)=0.5.
\]

## 连续分布

连续变量通常用 Probability Density Function（PDF）$f(x)$。

一个具体点的概率不是直接看 $f(x)$，而看区间下的面积：

\[
P(a\le X\le b)
=
\int_a^b f(x)\,dx.
\]

总面积满足

\[
\int_{-\infty}^{\infty}f(x)\,dx=1.
\]

[Normal Distribution](/mathematics/probability/normal-distribution/) 就是最重要的连续分布之一。

## Distribution 不等于 Dataset Histogram

Histogram 是有限样本的统计结果；probability distribution 是我们对随机变量总体规律的数学描述。

真实数据可以被某个 distribution model 近似，但样本 histogram 不会和理论 density 完全一致。

## 参数决定分布形状

一个 distribution family 通常由参数决定具体成员。

例如 Normal Distribution：

\[
X\sim\mathcal N(\mu,\sigma^2).
\]

$\mu$ 决定中心，$\sigma^2$ 决定扩散程度。

## Joint、Marginal 与 Conditional

多个 random variables 可以有 joint distribution：

\[
p(x,y).
\]

只看 $X$ 得到 marginal：

\[
p(x)=\sum_y p(x,y)
\]

或连续情况下积分掉 $y$。

已经知道 $Y=y$ 后，$X$ 的分布是 [Conditional Probability](/mathematics/probability/conditional-probability/)：

\[
p(x\mid y).
\]

这些概念构成 VAE/CVAE 概率表达的基础。
