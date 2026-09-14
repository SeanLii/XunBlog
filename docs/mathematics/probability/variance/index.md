---
title: "Variance"
kind: "canonical"
domain: "Mathematics / Probability"
parent: "Probability"
canonical: "/mathematics/probability/variance/"
prerequisites:
  - "/mathematics/probability/expectation/"
related:
  - "/mathematics/probability/covariance/"
  - "/mathematics/probability/normal-distribution/"
---

# Variance

Expectation 描述 distribution 的中心，但只知道中心还不够。

例如下面两组数据都可能均值为 0：

```text
A: -1, 0, 1
B: -100, 0, 100
```

它们的波动尺度显然完全不同。

Variance 用来描述 random variable 围绕其 mean 的离散程度。

设：

\[
\mu=\mathbb E[X].
\]

Variance 定义为：

\[
\operatorname{Var}(X)
=\mathbb E[(X-\mu)^2].
\]

## Centering：先减 Mean

\[
X-\mu
\]

表示每个取值相对 distribution center 的偏差。

如果不减 mean，$\mathbb E[X^2]$ 会同时受到“整体位置”和“波动大小”影响，无法单独表示 spread。

## Squared Deviation

如果直接平均 deviation：

\[
\mathbb E[X-\mu]=0.
\]

正偏差和负偏差会互相抵消。

平方后：

- 正负偏差都变成非负；
- 更大的 deviation 会被更强地放大；
- 数学上可微且具有良好的代数性质。

因此：

\[
\operatorname{Var}(X)\ge0.
\]

## 等价形式

展开：

\[
(X-\mu)^2=X^2-2\mu X+\mu^2.
\]

取 expectation：

\[
\operatorname{Var}(X)
=\mathbb E[X^2]-2\mu\mathbb E[X]+\mu^2.
\]

因为 $\mathbb E[X]=\mu$：

\[
\boxed{
\operatorname{Var}(X)
=\mathbb E[X^2]-\mathbb E[X]^2
}
\]

这个形式在推导里非常常见。

## Standard Deviation

Variance 的单位是原变量单位的平方。

例如身高单位是 cm，则 variance 单位是 cm²。

因此常定义 standard deviation：

\[
\sigma=\sqrt{\operatorname{Var}(X)}.
\]

它恢复到与原变量相同的单位，更容易解释实际尺度。

Standard deviation 与 variance 是同一 spread 信息的两种表达，所以这里把它作为 Variance 的直接组成部分，而不是另拆一个孤立页面。

## Scaling

若：

\[
Y=aX+b,
\]

则：

\[
\operatorname{Var}(Y)
=a^2\operatorname{Var}(X).
\]

加常数 $b$ 只移动中心，不改变 spread；乘 $a$ 会把 deviation 放大 $|a|$ 倍，因此 variance 放大 $a^2$ 倍。

## 从 Variance 到 Covariance

Variance 只描述一个 random variable 自己的波动。

当我们想描述两个 variables 是否一起变化，需要 [Covariance](/mathematics/probability/covariance/)：

\[
\operatorname{Cov}(X,Y)
=\mathbb E[(X-\mu_X)(Y-\mu_Y)].
\]

并且：

\[
\operatorname{Var}(X)=\operatorname{Cov}(X,X).
\]

因此 covariance 是 variance 的多变量扩展，而不是另一个无关的统计量。
