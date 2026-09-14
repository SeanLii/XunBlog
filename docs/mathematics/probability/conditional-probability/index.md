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

Conditional Probability 描述：**在已经知道某件事发生的前提下，另一件事的概率应该怎样更新。**

对 events $A,B$，若 $P(B)>0$：

\[
P(A\mid B)
=\frac{P(A\cap B)}{P(B)}.
\]

这里的 $B$ 不是“额外乘一个条件”，而是在告诉我们：原来的 sample space 已经缩小到 $B$ 发生的部分，需要在这个新范围内重新归一化概率。

## 从 sample space 缩小来理解

假设一副标准扑克牌中随机抽一张。

令：

- $A$：抽到 Ace；
- $B$：抽到 Spade。

原本：

\[
P(A)=\frac4{52}.
\]

但如果已经知道这张牌是 Spade，那么可能结果只剩 13 张 Spades，其中只有一张 Ace：

\[
P(A\mid B)=\frac1{13}.
\]

条件信息改变了我们计算概率的参考范围。

## Product Rule

由定义直接得到：

\[
P(A\cap B)=P(A\mid B)P(B).
\]

也可以反过来：

\[
P(A\cap B)=P(B\mid A)P(A).
\]

因此：

\[
P(A\mid B)P(B)
=P(B\mid A)P(A).
\]

这正是 [Bayes’ Theorem](/mathematics/probability/bayes-theorem/) 的起点。

## Random Variables 的 conditional distribution

对于 discrete random variables：

\[
p(y\mid x)
=\frac{p(x,y)}{p(x)}.
\]

对于 continuous variables，形式仍然类似，只是用 density：

\[
p(y\mid x)
=\frac{p(x,y)}{p(x)}.
\]

这里的 $p(x)$ 是对 $y$ marginalize 后得到：

\[
p(x)=\int p(x,y)\,dy.
\]

## Independence

如果 $A$ 与 $B$ independent：

\[
P(A\mid B)=P(A).
\]

等价地：

\[
P(A\cap B)=P(A)P(B).
\]

意思是知道 $B$ 没有改变我们对 $A$ 的概率判断。

对于 random variables，也可以通过 joint distribution 是否 factorize 来定义 independence。

## Conditional Probability 在建模中的意义

大量机器学习任务本质上都在学习 conditional distribution：

\[
p(y\mid x).
\]

例如：

- 给定图像 $x$，类别 $y$ 的概率；
- 给定文本上下文 $x$，下一个 token $y$ 的概率；
- 给定 observation $x$，机器人 action $a$ 的分布；
- 给定输入 $x$，生成模型输出 $y$ 的分布。

因此 conditional probability 是概率建模本身的基本语言，而不是 VAE 或某个模型专属的背景公式。
