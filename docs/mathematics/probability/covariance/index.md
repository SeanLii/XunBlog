---
title: "Covariance"
kind: "canonical"
domain: "Mathematics / Probability"
parent: "Probability"
canonical: "/mathematics/probability/covariance/"
prerequisites:
  - "/mathematics/probability/expectation/"
  - "/mathematics/probability/variance/"
related:
  - "/mathematics/probability/multivariate-normal-distribution/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Covariance

> **知识边界**：本文的 canonical 对象是 **Covariance**。依赖机制由 [Expectation](/mathematics/probability/expectation/)、[Variance](/mathematics/probability/variance/) 的 canonical page 定义；本文只在当前语境中调用其接口。


Covariance 衡量两个 random variables 相对各自 means 的共同线性变化。

设：

\[
\mu_X=\mathbb E[X],
\qquad
\mu_Y=\mathbb E[Y].
\]

定义：

\[
\operatorname{Cov}(X,Y)
=
\mathbb E[(X-\mu_X)(Y-\mu_Y)].
\]

等价地：

\[
\operatorname{Cov}(X,Y)
=
\mathbb E[XY]
-
\mathbb E[X]\mathbb E[Y].
\]

## Sign

- $\operatorname{Cov}(X,Y)>0$：两者倾向于同方向偏离各自 mean；
- $\operatorname{Cov}(X,Y)<0$：两者倾向于反方向变化；
- $\operatorname{Cov}(X,Y)=0$：没有线性 covariance。

Zero covariance 不意味着一般意义上的 independence。

## Scaling

对 constants $a,b,c,d$：

\[
\operatorname{Cov}(aX+b,cY+d)
=
ac\operatorname{Cov}(X,Y).
\]

因此 covariance 的数值依赖 variables 的单位和尺度。

## Correlation

为了移除 scale，可以定义 Pearson correlation：

\[
\rho_{X,Y}
=
\frac{\operatorname{Cov}(X,Y)}
{\sigma_X\sigma_Y},
\]

当两个 standard deviations 非零时成立。

根据 Cauchy–Schwarz inequality：

\[
-1\le\rho_{X,Y}\le1.
\]

Correlation 是 normalized covariance，但仍主要反映线性 dependency。

## Variance as Self-Covariance

令 $Y=X$：

\[
\operatorname{Cov}(X,X)
=
\operatorname{Var}(X).
\]

因此 variance 是 covariance 的特殊情况。

## Covariance Matrix

对 random vector：

\[
X\in\mathbb R^d,
\qquad
\mu=\mathbb E[X],
\]

covariance matrix 定义为：

\[
\Sigma
=
\mathbb E[(X-\mu)(X-\mu)^\top].
\]

元素：

\[
\Sigma_{ij}
=
\operatorname{Cov}(X_i,X_j).
\]

Diagonal elements 是各 coordinate variance；off-diagonal elements 描述 pairwise covariance。

## Positive Semidefinite Property

对任意 vector $a$：

\[
a^\top\Sigma a
=
\operatorname{Var}(a^\top X)
\ge0.
\]

所以 covariance matrix 一定 positive semidefinite。

它的 eigenvectors / eigenvalues 可以描述 distribution 在线性 directions 上的主要 variance structure。

## Linear Transformation

若：

\[
Y=AX+b,
\]

则：

\[
\operatorname{Cov}(Y)
=
A\Sigma_XA^\top.
\]

这条公式在 multivariate Gaussian、Kalman filtering、PCA 与 uncertainty propagation 中非常常见。

## Independence 与 Gaussian Special Case

一般情况下：

\[
\operatorname{Cov}(X,Y)=0
\]

不能推出 $X,Y$ independent。

但如果 $X,Y$ jointly Gaussian，则 zero covariance 可以推出 independence。

这是 Gaussian family 的特殊性质，不应推广到任意 distributions。

## Sample Covariance

给定 paired observations：

\[
(x_i,y_i),\quad i=1,\ldots,n,
\]

常用 sample covariance：

\[
s_{XY}
=
\frac1{n-1}
\sum_{i=1}^{n}
(x_i-\bar x)(y_i-\bar y).
\]

多维数据对应 sample covariance matrix。

## Connections

- [Variance](/mathematics/probability/variance/)：self-covariance。
- [Multivariate Normal Distribution](/mathematics/probability/multivariate-normal-distribution/)：covariance matrix 决定 Gaussian 的线性 dependency 与 geometry。
- [Linear Transformation](/mathematics/linear-algebra/linear-transformation/)：covariance 在 linear mapping 下按 $A\Sigma A^\top$ 变换。
