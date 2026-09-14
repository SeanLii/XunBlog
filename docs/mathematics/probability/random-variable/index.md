---
title: "Random Variable"
kind: "canonical"
domain: "Mathematics / Probability"
parent: "Probability"
canonical: "/mathematics/probability/random-variable/"
prerequisites:
  []
related:
  - "/mathematics/probability/probability-distribution/"
  - "/generative-models/latent-variable/"
---

# Random Variable

Random Variable 是把随机实验的每一种可能结果映射成一个数的函数。

例如掷一枚骰子，结果空间是

```text
{点数1, 点数2, ..., 点数6}
```

定义随机变量 $X$ 为“掷出的点数”，那么

\[
X\in\{1,2,3,4,5,6\}.
\]

随机性来自实验结果不确定；Random Variable 让我们可以用数学数值描述这种不确定性。

## 离散与连续

离散随机变量只取离散值，例如骰子点数。

连续随机变量可以在连续区间取值，例如人的身高、传感器噪声、VAE latent 中某一维的 Gaussian sample。

## Random Variable 与普通变量的区别

普通代数变量 $x=3$ 只是一个确定数。

随机变量 $X$ 在实验发生前不是一个固定结果，而由概率规律决定。真正观测到一次结果后，可以得到一个 realization，例如

\[
X=3.
\]

因此通常用大写 $X$ 表示 random variable，小写 $x$ 表示它的一次具体取值。

## Distribution 描述它怎样随机

只知道“$X$ 是随机变量”还不够。还需要 [Probability Distribution](/mathematics/probability/probability-distribution/) 描述每个可能值出现的概率。

例如公平骰子：

\[
P(X=i)=\frac16,
\qquad i=1,\ldots,6.
\]

连续变量则通过 probability density 等方式描述。

## 在生成模型中

VAE 的 latent $Z$ 是 random variable：

\[
Z\sim\mathcal N(0,I).
\]

这句话不是说 $Z$ 永远等于 0，而是说它的取值按一个 Normal Distribution 随机产生。

看到数据 $x$ 后，posterior

\[
p(z\mid x)
\]

又会描述“在这个观测条件下 latent variable 可能取哪些值”。
