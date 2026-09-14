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

正态分布（normal distribution，也称 Gaussian distribution）是一族由均值和方差决定的连续概率分布。VAE 中最常见的 prior 与 approximate posterior 都使用它，是因为高斯分布既容易参数化，也能方便地进行重参数化和 KL 计算。

## 定义

一维随机变量

\[
X\sim\mathcal N(\mu,\sigma^2)
\]

的概率密度为

\[
p(x)=\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right).
\]

其中：

- $\mu\in\mathbb R$ 是均值，决定分布中心；
- $\sigma^2>0$ 是方差，决定分布有多分散；
- $\sigma$ 是标准差。

当 $x$ 离 $\mu$ 越远时，指数项越小，因此密度逐渐下降。

## 均值与尺度

改变 $\mu$ 会整体平移分布。改变 $\sigma$ 会改变横向尺度：$\sigma$ 越大，分布越宽；$\sigma$ 越小，概率质量越集中在均值附近。

标准正态分布是特殊情况

\[
Z\sim\mathcal N(0,1).
\]

任意非退化的一维正态变量都可以标准化为

\[
Z=\frac{X-\mu}{\sigma}.
\]

减去 $\mu$ 把中心移动到 0；再除以 $\sigma$ 把“一单位”重新定义成原分布的一个标准差。只减均值仍会保留不同数据尺度，因此还不能把不同方差的分布放到同一个标准尺度上。

## 从标准正态生成一般正态

反过来，如果

\[
\epsilon\sim\mathcal N(0,1),
\]

那么

\[
X=\mu+\sigma\epsilon
\]

满足

\[
X\sim\mathcal N(\mu,\sigma^2).
\]

这条关系是 [Reparameterization Trick](/generative-models/reparameterization-trick/) 的数学基础。随机性被放在 $\epsilon$ 上，而 $\mu$ 与 $\sigma$ 通过普通可微运算决定最终 sample。

## 正态分布不是“把数据变标准”

正态分布描述概率规律；标准化是对变量进行变换。数据可以被标准化到均值约 0、标准差约 1，却不因此自动变成正态分布。标准正态分布要求的不只是前两阶统计量，还要求整个概率密度具有高斯形式。

## 在 VAE 中的角色

VAE 常设置

\[
p(z)=\mathcal N(0,I),
\]

并让 encoder 输出近似 posterior 的 $\mu$ 与方差。训练中的 KL term 会约束 posterior 不要任意偏离这个 prior。ACT 沿用了这种结构，并在推理时使用 prior 的均值 $z=0$。
