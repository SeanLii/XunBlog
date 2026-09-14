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

Normal Distribution 是一个由中心位置和波动尺度控制的连续概率分布。它的形状是熟悉的钟形曲线，但“钟形”只是结果，真正定义它的是两个 parameters：mean $\mu$ 和 variance $\sigma^2$。

写作：

\[
X\sim\mathcal N(\mu,\sigma^2).
\]

其 probability density function 为：

\[
p(x)=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(
-\frac{(x-\mu)^2}{2\sigma^2}
\right).
\]

## 从“中心”和“尺度”理解公式

公式中最关键的是：

\[
(x-\mu)^2.
\]

它测量 $x$ 离中心 $\mu$ 有多远。离中心越远，负指数越大，density 越低。

而 $\sigma$ 控制“多远才算远”。

- $\sigma$ 小：distribution 集中在 $\mu$ 附近；
- $\sigma$ 大：distribution 更宽。

因此 mean 和 standard deviation 分别控制位置与尺度。

## Density 不是单点概率

因为 Normal Distribution 是 continuous distribution：

\[
P(X=x)=0.
\]

例如即使 $x=\mu$ 是 density 最大的位置，单点概率仍然为 0。

真正的概率来自面积：

\[
P(a\le X\le b)
=
\int_a^b p(x)\,dx.
\]

所以曲线“更高”表示附近小区间聚集了更多 probability mass，而不是那个点本身有一个有限概率。

## Mean、Median 与 Symmetry

Normal Distribution 关于 $\mu$ 对称。

因此：

\[
\text{mean}=	ext{median}=	ext{mode}=\mu.
\]

左右两侧具有相同 probability mass：

\[
P(X<\mu)=P(X>\mu)=\frac12.
\]

## Standard Normal Distribution

当：

\[
\mu=0,
\qquad
\sigma=1,
\]

得到 standard normal distribution：

\[
Z\sim\mathcal N(0,1).
\]

它不是“更标准、更真实”的 normal distribution，而是把不同 normal distributions 统一到同一个坐标尺度后的参考形式。

## Standardization

如果：

\[
X\sim\mathcal N(\mu,\sigma^2),
\]

定义：

\[
Z=\frac{X-\mu}{\sigma}.
\]

则：

\[
Z\sim\mathcal N(0,1).
\]

这个公式包含两个完全不同的操作。

第一步：

\[
X-\mu
\]

把分布中心平移到 0。

第二步：

\[
\frac{X-\mu}{\sigma}
\]

把原本“一个 standard deviation”的距离统一缩放成 1。

所以不能只减 mean 而不除 standard deviation。只减 mean 只能统一中心，不能统一尺度。

## z-score

具体 observation $x$ 的 standardized value：

\[
z=\frac{x-\mu}{\sigma}
\]

叫 z-score。

它的解释非常直接：

> $x$ 距离 mean 有多少个 standard deviations。

例如：

\[
\mu=60,\quad\sigma=10,\quad x=80,
\]

则：

\[
z=2.
\]

也就是这个 observation 位于 mean 上方 2 个 standard deviations。

这让不同单位、不同中心和不同尺度的数据获得统一的相对位置描述。

## 68–95–99.7 Rule

对于 normal distribution，大约：

- 68% 落在 $\mu\pm1\sigma$；
- 95% 落在 $\mu\pm2\sigma$；
- 99.7% 落在 $\mu\pm3\sigma$。

这不是 normal distribution 的定义，而是由其具体形状得到的常用性质。

## Linear Transformation 下仍然是 Normal

若：

\[
X\sim\mathcal N(\mu,\sigma^2),
\qquad
Y=aX+b,
\]

则：

\[
Y\sim\mathcal N(a\mu+b,a^2\sigma^2).
\]

这类 closure property 是 Gaussian 在统计建模里极其方便的重要原因之一。

## 从一维到多维

当 random variable 变成 random vector，需要用 mean vector 与 covariance matrix 描述：

\[
X\sim\mathcal N(\mu,\Sigma).
\]

完整结构见 [Multivariate Normal Distribution](/mathematics/probability/multivariate-normal-distribution/)。

在 VAE 等模型中使用 Gaussian，是这个概率分布的一个具体应用；Normal Distribution 本身不依赖 VAE 才成立。
