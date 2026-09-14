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

随机变量（random variable）把随机试验的结果映射成数值。它的“随机”来自结果在观测前不确定，而不是变量自己随意改变。

## 定义

设样本空间为 $\Omega$。随机变量 $X$ 是一个映射

\[
X:\Omega\rightarrow\mathbb{R}.
\]

对于每个可能结果 $\omega\in\Omega$，$X(\omega)$ 给出一个数值。

例如掷一次骰子时，样本结果可以直接由 $1$ 到 $6$ 表示，于是 $X$ 的取值集合就是 $\{1,2,3,4,5,6\}$。更复杂的情况下，样本本身可以是图像、轨迹或其他对象，而随机变量只抽取其中某个数值属性。

## 离散与连续

离散随机变量的可能取值可以逐个列举，并用概率质量函数描述：

\[
p_X(x)=P(X=x).
\]

连续随机变量通常用概率密度函数 $p_X(x)$ 描述。对连续变量，单个精确点的概率通常为零；区间概率由积分得到：

\[
P(a\le X\le b)=\int_a^b p_X(x)\,dx.
\]

## 在生成模型中的位置

VAE 中的 $z$ 是随机变量，而不是一个固定编码。encoder 给出的不是单一 $z$，而是一个条件分布 $q_\phi(z|x)$；随后从这个分布中得到具体的 latent sample。理解这一点需要把“随机变量”和“它的一次取值”区分开。

[Latent Variable](/generative-models/latent-variable/) 进一步要求这个随机变量没有被数据直接观测到，而是作为概率模型内部的隐藏因素出现。
