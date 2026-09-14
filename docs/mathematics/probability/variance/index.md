---
title: "Variance"
kind: "canonical"
domain: "Mathematics / Probability"
parent: "Probability"
canonical: "/mathematics/probability/variance/"
prerequisites:
  - "/mathematics/probability/expectation/"
related:
  - "/mathematics/probability/normal-distribution/"
---

# Variance

Variance 描述 random variable 围绕其 mean 的分散程度。

若

\[
\mu=\mathbb E[X],
\]

则

\[
\operatorname{Var}(X)
=
\mathbb E[(X-\mu)^2].
\]

## 先减去均值

$X-\mu$ 表示一次取值距离中心有多远。

如果不减 mean，而直接看 $X^2$，结果会同时受到“整体数值基准”和“分散程度”影响。Variance 只想描述 spread，所以先把中心移到 0。

## 再平方

如果直接平均偏差：

\[
\mathbb E[X-\mu]=0.
\]

正负偏差会抵消。因此用平方：

\[
(X-\mu)^2.
\]

它同时让偏差非负，并让较大偏差贡献更大。

## Standard Deviation

Variance 的单位被平方了。例如身高单位是 cm，variance 单位是 cm²。

Standard deviation 定义为

\[
\sigma=\sqrt{\operatorname{Var}(X)}.
\]

它恢复到和原变量相同的单位，因此更容易直接解释。

## 等价形式

展开平方可以得到

\[
\operatorname{Var}(X)
=
\mathbb E[X^2]-\mathbb E[X]^2.
\]

这在推导中很常用。

## Normal Distribution 中的角色

\[
X\sim\mathcal N(\mu,\sigma^2)
\]

的第二个参数是 variance $\sigma^2$，不是 standard deviation $\sigma$。

$\sigma$ 越大，distribution 越宽；$\sigma$ 越小，越集中在 mean 附近。

## Standardization 中除以 σ 的作用

对

\[
Z=\frac{X-\mu}{\sigma},
\]

先减 $\mu$ 把中心移到 0；再除 $\sigma$ 把“一份标准差”重新定义成 1 个单位。

只有减均值，没有除标准差时，不同 distributions 虽然中心都在 0，但 spread 仍然不同，仍不能直接比较“离自己群体中心有多异常”。
