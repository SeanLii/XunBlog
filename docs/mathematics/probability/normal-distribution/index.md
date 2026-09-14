---
title: "Normal Distribution"
kind: "canonical"
domain: "Mathematics / Probability"
parent: "Probability"
canonical: "/mathematics/probability/normal-distribution/"
prerequisites:
  - "/mathematics/probability/expectation/"
  - "/mathematics/probability/variance/"
related:
  - "/mathematics/probability/multivariate-normal-distribution/"
  - "/generative-models/variational-autoencoder/"
---

# Normal Distribution

Normal Distribution（正态分布）是一类由 mean $\mu$ 和 variance $\sigma^2$ 决定的连续 probability distribution：

\[
X\sim\mathcal N(\mu,\sigma^2).
\]

它的图像是熟悉的钟形曲线，但真正需要理解的不是“长得像钟”，而是两个参数怎样决定一个随机变量的中心与波动尺度。

## 从一组会波动的数据开始

假设一个班多次考试的成绩集中在 70 分附近。有的人 60 多，有的人 80 多，离 70 越远的人越来越少。

可以把这种分布的中心用

\[
\mu
\]

描述，把“通常会离中心波动多远”用 standard deviation

\[
\sigma
\]

描述。

Normal Distribution 的 density 为

\[
f(x)
=
\frac{1}{\sigma\sqrt{2\pi}}
\exp\left(
-\frac{(x-\mu)^2}{2\sigma^2}
\right).
\]

先不要被公式吓到。它的结构其实正对应“中心”和“距离中心多远”。

## μ 决定中心

公式中出现

\[
x-\mu.
\]

当 $x=\mu$ 时，偏差为 0，指数项最大，所以 density peak 位于 $\mu$。

改变 $\mu$ 主要会把整条曲线左右移动：

```text
small μ              large μ
    /\                   /\
  /   \                /   \
_/     \_            _/     \_
```

因此 mean 描述“数据整体落在哪个位置”。

## σ 决定宽窄

公式中偏差平方除以

\[
2\sigma^2.
\]

当 $\sigma$ 大时，同样的偏差 $|x-\mu|$ 被除以更大的数，density 下降得更慢，所以曲线更宽。

当 $\sigma$ 小时，数据更集中在 mean 附近。

```text
small σ:         /\
               /   \
______________/     \____________

large σ:      ____
            _/    \_
___________/        \___________
```

所以 $\sigma$ 决定的是 scale，而不是“平均成绩高不高”。

## Density 不是单点概率

Normal Distribution 是 continuous distribution。严格地说，某个精确点的概率

\[
P(X=x)=0.
\]

我们真正计算的是区间概率：

\[
P(a<X<b)
=
\int_a^b f(x)\,dx.
\]

曲线高度是 probability density，不是“这个点本身的概率”。

## Standard Normal Distribution

当

\[
\mu=0,
\qquad
\sigma=1,
\]

得到 Standard Normal Distribution：

\[
Z\sim\mathcal N(0,1).
\]

任何一维 normal variable

\[
X\sim\mathcal N(\mu,\sigma^2)
\]

都可以通过

\[
Z=\frac{X-\mu}{\sigma}
\]

变成 standard normal。

## 减去均值：统一中心

先看

\[
X-\mu.
\]

这一步把“绝对数值”变成“距离自己分布中心有多远”。

例如：

- A 群体平均 160 cm，一个人 170 cm：高出平均 10 cm；
- B 群体平均 180 cm，一个人 190 cm：也高出平均 10 cm。

减均值后两人都是 +10。

但现在仍然不能说他们在各自群体中“同样高”。因为两个群体的波动可能完全不同。

## 除以标准差：统一尺度

假设 A 群体 standard deviation 是 5 cm，B 群体是 20 cm。

A：

\[
z_A=\frac{170-160}{5}=2.
\]

B：

\[
z_B=\frac{190-180}{20}=0.5.
\]

两人虽然都比各自 mean 高 10 cm，但 A 是高出 **2 个标准差**，B 只高出 **0.5 个标准差**。

所以除以 $\sigma$ 做的是单位换算：

> 不再用 cm、分数或别的原始单位衡量偏差，而改用“几个 standard deviations”衡量。

这就是为什么标准化不能只减 mean。

减均值只统一了**中心**；除 standard deviation 才统一了**尺度**。

## z-score 的解释

\[
z=1
\]

表示高于 mean 一个 standard deviation；

\[
z=-2
\]

表示低于 mean 两个 standard deviations。

对于 normal distribution，大约：

- 68% 数据落在 $\mu\pm1\sigma$；
- 95% 落在 $\mu\pm2\sigma$；
- 99.7% 落在 $\mu\pm3\sigma$。

这让 z-score 有清楚的相对位置意义。

## 在 VAE 中

VAE 常选择 prior：

\[
p(z)=\mathcal N(0,I).
\]

一维看就是每个 latent dimension 以 0 为中心、单位 variance；多维情况见 [Multivariate Normal Distribution](/mathematics/probability/multivariate-normal-distribution/)。

Encoder 常输出自己的 $\mu(x)$ 与 $\sigma(x)$，定义

\[
q_\phi(z\mid x)
=
\mathcal N(\mu(x),\operatorname{diag}(\sigma^2(x))).
\]

因此理解 Normal Distribution 之后，VAE 中的“prior 是标准正态”“encoder 输出 mean 和 log variance”“reparameterization”会连成同一套概率结构。
