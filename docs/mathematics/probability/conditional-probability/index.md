---
title: "Conditional Probability"
kind: "canonical"
domain: "Mathematics / Probability"
parent: "Probability"
canonical: "/mathematics/probability/conditional-probability/"
prerequisites:
  - "/mathematics/probability/probability-distribution/"
related:
  - "/generative-models/conditional-variational-autoencoder/"
---

# Conditional Probability

Conditional Probability 描述：**在已经知道某个条件成立以后，另一个事件的概率如何重新计算。**

对事件 $A,B$，且 $P(B)>0$：

\[
P(A\mid B)
=
\frac{P(A\cap B)}{P(B)}.
\]

它相当于把原来的样本空间缩小到 $B$ 中，再看其中有多少同时属于 $A$。

## 一个简单例子

假设 100 人中：

- 40 人学过线性代数；
- 20 人同时学过线性代数和概率论。

在“已知这个人学过线性代数”的条件下，他也学过概率论的概率为

\[
P(Probability\mid LinearAlgebra)
=\frac{20}{40}=0.5.
\]

分母不再是全部 100 人，而是条件限定后的 40 人。

## 乘法规则

由定义立即得到：

\[
P(A\cap B)
=P(A\mid B)P(B).
\]

随机变量分布也可以写成：

\[
p(x,y)=p(x\mid y)p(y).
\]

这就是很多 probabilistic model factorization 的基础。

## Bayes Rule

同一个 joint probability 也可以写成

\[
p(x,y)=p(y\mid x)p(x).
\]

于是

\[
p(x\mid y)
=
\frac{p(y\mid x)p(x)}{p(y)}.
\]

Bayes rule 把“从 $x$ 生成 $y$”的概率关系转成“看到 $y$ 后 $x$ 可能是什么”。

## VAE 中的条件关系

VAE 生成模型写成

\[
p(z)p_\theta(x\mid z).
\]

看到 $x$ 后关心 posterior：

\[
p_\theta(z\mid x).
\]

CVAE 则进一步建模：

\[
p_\theta(y\mid x,z),
\]

其中 $x$ 是已知 condition，$z$ 是 latent variable。

所以条件概率不是“给公式加一条竖线”，而是在明确：**当前哪些信息已经被视为已知。**
