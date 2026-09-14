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
---

# Covariance

Variance 描述一个 random variable 自己波动多少；Covariance 描述两个 random variables 是否倾向于一起变化。

设：

\[
\mu_X=\mathbb E[X],
\qquad
\mu_Y=\mathbb E[Y].
\]

Covariance 定义为：

\[
\operatorname{Cov}(X,Y)
=\mathbb E[(X-\mu_X)(Y-\mu_Y)].
\]

## 符号在表达什么

如果某次：

- $X$ 高于自己的 mean；
- $Y$ 也高于自己的 mean；

那么两个 centered deviations 的乘积为正。

如果一个高于 mean、另一个低于 mean，乘积为负。

所以平均后：

- covariance > 0：倾向同方向变化；
- covariance < 0：倾向反方向变化；
- covariance 接近 0：没有明显的**线性**共同变化。

注意 covariance 为 0 一般不等于 independent；它只排除了某类线性依赖。

## Variance 是 Covariance 的特殊情况

令 $Y=X$：

\[
\operatorname{Cov}(X,X)
=\mathbb E[(X-\mu_X)^2]
=\operatorname{Var}(X).
\]

所以 covariance 自然扩展了 variance。

## Covariance Matrix

对于 random vector：

\[
X\in\mathbb R^d,
\qquad
\mu=\mathbb E[X],
\]

定义 covariance matrix：

\[
\Sigma
=\mathbb E[(X-\mu)(X-\mu)^\top].
\]

其第 $(i,j)$ 个元素：

\[
\Sigma_{ij}
=\operatorname{Cov}(X_i,X_j).
\]

对角线：

\[
\Sigma_{ii}=\operatorname{Var}(X_i).
\]

非对角线描述不同 coordinates 的共同变化。

## Geometry

二维情况下，如果两个 coordinates independent 且 variance 相同，distribution 的等密度轮廓可能接近圆。

如果：

- 两个方向 variance 不同；
- coordinates 之间存在 covariance；

轮廓会拉伸并旋转成 ellipse。

因此 covariance matrix 不只是一个统计表格，它决定 multivariate distribution 的几何尺度和方向。

## Correlation

Covariance 会受到变量本身单位和尺度影响。

标准化后得到 correlation：

\[
\rho_{XY}
=\frac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y}.
\]

只要 standard deviations 非零，有：

\[
-1\le\rho_{XY}\le1.
\]

Correlation 更适合比较不同尺度变量之间的线性关系，但它并没有替代 covariance 在概率模型中的作用。

## 在 Multivariate Normal 中

[Multivariate Normal Distribution](/mathematics/probability/multivariate-normal-distribution/) 用：

\[
\mathcal N(\mu,\Sigma)
\]

完整表达中心和二阶依赖结构。

因此理解 covariance matrix 是理解 multivariate Gaussian、PCA、state estimation 和很多 probabilistic models 的基础。
