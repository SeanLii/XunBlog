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

条件概率（conditional probability）描述：在已经知道某个条件成立之后，另一个事件发生的概率如何改变。

## 定义

对事件 $A$ 和 $B$，只要 $P(B)>0$，条件概率定义为

\[
P(A|B)=\frac{P(A\cap B)}{P(B)}.
\]

分母把讨论范围限制到 $B$ 已经发生的部分；分子是在这个范围中同时满足 $A$ 的部分。

## 条件分布

对随机变量 $X$ 与 $Y$，$p(x|y)$ 表示已经知道 $Y=y$ 后，$X$ 的概率分布。联合分布可以分解为

\[
p(x,y)=p(x|y)p(y).
\]

同样也可以写成

\[
p(x,y)=p(y|x)p(x).
\]

这些关系是概率生成模型的基础。

## 在 CVAE 中的含义

普通 VAE 建模数据分布时常写成 $p_\theta(x|z)$。CVAE 再加入条件 $c$：

\[
p_\theta(x|z,c).
\]

这里不是把 $c$ 当作额外随机噪声，而是明确告诉模型：要在给定 $c$ 的情况下描述 $x$ 的分布。

在 ACT 中，可以把当前机器人观测看作条件，把未来 action chunk 看作要生成的输出。这样，模型学习的是“当前观测给定以后，合理的未来动作序列怎样分布”，而不是无条件地产生动作。
