---
title: "Conditional Probability"
kind: "canonical"
domain: "Mathematics / Probability"
parent: "Probability"
canonical: "/mathematics/probability/conditional-probability/"
prerequisites:
  - "/mathematics/probability/probability-distribution/"
related:
  - "/mathematics/probability/bayes-theorem/"
---

# Conditional Probability

Conditional Probability 描述在已知事件 $B$ 发生的条件下，事件 $A$ 的概率。

若：

\[
P(B)>0,
\]

定义：

\[
P(A\mid B)
=
\frac{P(A\cap B)}{P(B)}.
\]

它改变的是计算 probability 时所采用的 reference population：原 sample space 被限制到 $B$ 中。

## Product Rule

由定义直接得到：

\[
P(A\cap B)
=
P(A\mid B)P(B).
\]

也可写成：

\[
P(A\cap B)
=
P(B\mid A)P(A).
\]

这两个表达式是 Bayes' theorem 的基础。

## Chain Rule

对多个 events / variables：

\[
p(x_1,\ldots,x_n)
=
\prod_{i=1}^{n}
p(x_i\mid x_1,\ldots,x_{i-1}).
\]

例如：

\[
p(x,y,z)
=
p(x)p(y\mid x)p(z\mid x,y).
\]

Chain rule 不需要 independence assumption；它是 joint probability 的一般 factorization。

## Conditional Distribution

对 discrete random variables：

\[
p(x\mid y)
=
\frac{p(x,y)}{p(y)}.
\]

对 continuous variables 使用 conditional density：

\[
f_{X\mid Y}(x\mid y)
=
\frac{f_{X,Y}(x,y)}{f_Y(y)},
\]

在 denominator 有定义时成立。

对固定 $y$，conditional distribution 关于 $x$ 必须归一化。

## Law of Total Probability

若 $B_1,\ldots,B_k$ 构成 sample space 的 partition，则：

\[
P(A)
=
\sum_{i=1}^{k}P(A\mid B_i)P(B_i).
\]

连续 latent variable 时，对应形式为：

\[
p(x)
=
\int p(x\mid z)p(z)\,dz.
\]

这就是 latent-variable model 中的 marginalization。

## Independence

若 events $A,B$ independent：

\[
P(A\cap B)=P(A)P(B),
\]

则：

\[
P(A\mid B)=P(A).
\]

对于 random variables：

\[
p(x,y)=p(x)p(y).
\]

Independence 表示知道一个 variable 不改变另一个 variable 的 distribution。

## Conditional Independence

Variables $X,Y$ 在给定 $Z$ 后 conditionally independent，记作：

\[
X\perp Y\mid Z,
\]

若：

\[
p(x,y\mid z)
=
p(x\mid z)p(y\mid z).
\]

Conditional independence 不等于 marginal independence。两个 variables 可能 marginally dependent，但在 conditioning on a common cause 后变得 independent。

这类结构是 probabilistic graphical models 中 factorization 的核心。

## Bayes' Theorem

由 product rule 的两种写法：

\[
p(x,y)=p(x\mid y)p(y)=p(y\mid x)p(x),
\]

得到：

\[
p(x\mid y)
=
\frac{p(y\mid x)p(x)}{p(y)}.
\]

完整含义见 [Bayes' Theorem](/mathematics/probability/bayes-theorem/)。

## Conditional Expectation

给定 $Y=y$，可以对 $X$ 的 conditional distribution 求 expectation：

\[
\mathbb E[X\mid Y=y].
\]

Random-variable form：

\[
\mathbb E[X\mid Y]
\]

本身也是一个关于 $Y$ 的 random variable。

它满足 tower property：

\[
\mathbb E[
\mathbb E[X\mid Y]
]
=
\mathbb E[X].
\]

## Conditional Modeling in Machine Learning

Supervised learning 常建模：

\[
p(y\mid x).
\]

Conditional generative models 则可能建模：

\[
p(y\mid x,z)
\]

或：

\[
p(a\mid o).
\]

这里 conditional probability 提供了统一概率语言，而具体 neural architecture 只是这些 distributions 的 parameterization。

## Connections

- [Bayes' Theorem](/mathematics/probability/bayes-theorem/)：交换 conditioning 方向。
- [Latent Variable](/mathematics/probability/latent-variable/)：通过 conditional distribution 描述生成过程。
- [Conditional Variational Autoencoder](/generative-models/conditional-variational-autoencoder/)：学习 conditional latent-variable distribution。
