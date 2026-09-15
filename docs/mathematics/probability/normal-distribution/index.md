---
title: "Normal Distribution"
kind: "canonical"
domain: "Mathematics / Probability"
parent: "Probability"
canonical: "/mathematics/probability/normal-distribution/"
prerequisites:
  - "/mathematics/probability/probability-distribution/"
  - "/mathematics/probability/expectation/"
  - "/mathematics/probability/variance/"
related:
  - "/mathematics/probability/multivariate-normal-distribution/"
---

# Normal Distribution

Normal Distribution（Gaussian Distribution）是由 mean 与 variance 参数化的连续 probability distribution。

记作：

\[
X\sim\mathcal N(\mu,\sigma^2),
\qquad \sigma>0.
\]

其 probability density function 为：

\[
f(x)
=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(
-\frac{(x-\mu)^2}{2\sigma^2}
\right).
\]

参数 $\mu$ 控制位置，$\sigma^2$ 控制尺度。

## Mean, Median, Mode

Normal distribution 关于 $\mu$ 对称，因此：

\[
\operatorname{mean}
=
\operatorname{median}
=
\operatorname{mode}
=
\mu.
\]

其 expectation 与 variance 为：

\[
\mathbb E[X]=\mu,
\qquad
\operatorname{Var}(X)=\sigma^2.
\]

## Probability Density

Density $f(x)$ 描述 probability mass 在实数轴上的局部分布，但：

\[
f(x)\neq P(X=x).
\]

对连续 random variable：

\[
P(X=x)=0.
\]

区间 probability 由积分得到：

\[
P(a\le X\le b)
=
\int_a^b f(x)\,dx.
\]

## Symmetry 与 Scale

Density 只依赖：

\[
(x-\mu)^2,
\]

因此关于 $x=\mu$ 对称。

$\sigma$ 越大，distribution 越分散；$\sigma$ 越小，density 越集中在 $\mu$ 附近。

Normal family 是 location-scale family：若：

\[
Z\sim\mathcal N(0,1),
\]

则：

\[
X=\mu+\sigma Z
\]

满足：

\[
X\sim\mathcal N(\mu,\sigma^2).
\]

## Standard Normal Distribution

Standard normal 定义为：

\[
Z\sim\mathcal N(0,1).
\]

任意 Gaussian variable：

\[
X\sim\mathcal N(\mu,\sigma^2)
\]

都可以 standardize：

\[
Z=\frac{X-\mu}{\sigma}.
\]

这里减 $\mu$ 把 distribution center 移到 0；除以 $\sigma$ 把 scale 调整为 unit standard deviation。

因此 z-score：

\[
z=\frac{x-\mu}{\sigma}
\]

表示 observation 距离 mean 有多少个 standard deviations。

## Cumulative Distribution Function

Standard normal CDF 通常记为：

\[
\Phi(z)
=
P(Z\le z).
\]

它没有 elementary closed-form expression，但可以数值计算。

对一般 Gaussian：

\[
P(X\le x)
=
\Phi\left(
\frac{x-\mu}{\sigma}
\right).
\]

因此 standardization 可以把任意一维 Gaussian probability calculation 转为 standard normal CDF calculation。

## Quantiles

若：

\[
\Phi(z_p)=p,
\]

则 $z_p$ 是 standard-normal $p$-quantile。

一般 Gaussian 的对应 quantile 为：

\[
x_p
=
\mu+\sigma z_p.
\]

Quantile 用于 confidence intervals、thresholds 与 probabilistic calibration。

## 68–95–99.7 Rule

对 Normal distribution，大约：

\[
P(|X-\mu|\le\sigma)\approx0.6827,
\]

\[
P(|X-\mu|\le2\sigma)\approx0.9545,
\]

\[
P(|X-\mu|\le3\sigma)\approx0.9973.
\]

这些数值是 Gaussian family 的性质，不是所有 distributions 的通用规律。

## Affine Transformation

若：

\[
X\sim\mathcal N(\mu,\sigma^2),
\]

并令：

\[
Y=aX+b,
\]

则：

\[
Y
\sim
\mathcal N(a\mu+b,a^2\sigma^2).
\]

Normal family 对 affine transformation 封闭。

## Sum of Independent Gaussian Variables

若：

\[
X\sim\mathcal N(\mu_X,\sigma_X^2),
\]

\[
Y\sim\mathcal N(\mu_Y,\sigma_Y^2),
\]

且 independent，则：

\[
X+Y
\sim
\mathcal N(
\mu_X+\mu_Y,
\sigma_X^2+\sigma_Y^2
).
\]

更一般地，jointly Gaussian variables 的 linear combination 仍为 Gaussian。

## Maximum Entropy Property

在所有具有固定 mean 与 variance 的连续 distributions 中，Gaussian distribution 具有最大的 differential entropy。

这个性质说明：如果只约束一阶与二阶 moments，而不额外加入结构，Gaussian 是最少附加假设的一种分布。

## Central Limit Theorem

在适当条件下，大量 independent random variables 的标准化和会趋近于 Normal distribution。Central Limit Theorem 是 Gaussian 在统计推断中频繁出现的重要原因之一。

这并不意味着现实数据本身都服从 Gaussian；它描述的是某类 aggregate quantities 的 limiting behavior。

## Connections

- [Variance](/mathematics/probability/variance/)：$\sigma^2$ 控制 Gaussian scale。
- [Multivariate Normal Distribution](/mathematics/probability/multivariate-normal-distribution/)：推广到 random vector。
- [Variational Autoencoder](/generative-models/variational-autoencoder/)：常使用 Gaussian prior 与 approximate posterior。
