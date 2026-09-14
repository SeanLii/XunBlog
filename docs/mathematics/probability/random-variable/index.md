---
title: "Random Variable"
kind: "canonical"
domain: "Mathematics / Probability"
parent: "Probability"
canonical: "/mathematics/probability/random-variable/"
prerequisites: []
related:
  - "/mathematics/probability/probability-distribution/"
  - "/mathematics/probability/expectation/"
  - "/mathematics/probability/variance/"
---

# Random Variable

Random Variable 不是“一个会自己随机变化的普通变量”。它是把随机实验的结果映射成数值的函数。

设随机实验的 sample space 为 $\Omega$。Random variable $X$ 是一个 mapping：

\[
X:\Omega\rightarrow\mathbb R.
\]

例如掷一枚硬币两次，sample space 可以写成：

\[
\Omega=\{HH,HT,TH,TT\}.
\]

定义 $X$ 为“正面出现的次数”，那么：

\[
X(HH)=2,
\quad
X(HT)=1,
\quad
X(TH)=1,
\quad
X(TT)=0.
\]

随机性来自实验结果 $\omega\in\Omega$，而 $X$ 把这些结果转成我们关心的数值。

## Random Variable 与一次观测值

要区分：

\[
X
\]

和

\[
x.
\]

通常大写 $X$ 表示 random variable，小写 $x$ 表示它某次可能取得的具体值。

写：

\[
P(X=x)
\]

表示 random variable $X$ 取值为 $x$ 的概率。

这个区分在 probabilistic model 中非常重要：模型定义的是 random variables 之间的 distribution，而数据集给我们的是这些 variables 的具体 observations。

## Discrete Random Variable

如果 $X$ 只取有限或可数多个值，它是 discrete random variable。

例如骰子：

\[
X\in\{1,2,3,4,5,6\}.
\]

它的概率由 probability mass function 描述：

\[
p_X(x)=P(X=x).
\]

并满足：

\[
\sum_x p_X(x)=1.
\]

## Continuous Random Variable

如果 $X$ 可以在连续范围取值，通常用 probability density function：

\[
p_X(x).
\]

此时单点概率通常为：

\[
P(X=x)=0.
\]

真正有意义的是区间概率：

\[
P(a\le X\le b)
=\int_a^b p_X(x)\,dx.
\]

因此 density 的数值本身不是“这个点的概率”。

## Random Vector

多个 random variables 可以组成 random vector：

\[
X=
\begin{bmatrix}
X_1\\
\vdots\\
X_d
\end{bmatrix}.
\]

它的取值是 $\mathbb R^d$ 中的 vector。

这正是 multivariate distributions、latent variables、robot states 等概率建模的基础。

## Distribution 描述随机性的规律

Random variable 定义“我们从随机结果中读取什么数”；[Probability Distribution](/mathematics/probability/probability-distribution/) 则描述这些数值出现的规律。

例如同样是实值 random variable：

\[
X\sim \mathcal N(0,1)
\]

和

\[
Y\sim \mathcal N(10,0.1^2)
\]

都取实数，但其 distribution 完全不同。

后续的 [Expectation](/mathematics/probability/expectation/)、[Variance](/mathematics/probability/variance/) 都是对 random variable distribution 的整体性质进行总结。
