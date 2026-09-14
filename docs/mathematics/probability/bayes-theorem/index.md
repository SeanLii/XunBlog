---
title: "Bayes' Theorem"
kind: "canonical"
domain: "Mathematics / Probability"
parent: "Probability"
canonical: "/mathematics/probability/bayes-theorem/"
prerequisites:
  - "/mathematics/probability/conditional-probability/"
related:
  - "/mathematics/probability/variational-inference/"
---

# Bayes' Theorem

Bayes' Theorem 给出一种把“正向条件概率”反过来的方法。

对于 events $A,B$：

\[
P(A\mid B)
=\frac{P(B\mid A)P(A)}{P(B)},
\qquad P(B)>0.
\]

它之所以成立，不是因为某种特殊 Bayesian 技巧，而是因为 joint probability 可以从两个方向展开：

\[
P(A\cap B)=P(A\mid B)P(B),
\]

同时：

\[
P(A\cap B)=P(B\mid A)P(A).
\]

把两式相等并整理，就得到 Bayes' theorem。

## Prior、Likelihood、Posterior

在统计建模中常把公式写成：

\[
p(z\mid x)
=\frac{p(x\mid z)p(z)}{p(x)}.
\]

这里：

- $p(z)$：prior，在看到 observation 前对 $z$ 的分布；
- $p(x\mid z)$：likelihood，假设 $z$ 已知时产生 $x$ 的规律；
- $p(z\mid x)$：posterior，看到 $x$ 后对 $z$ 的更新判断；
- $p(x)$：evidence / marginal likelihood，用于归一化。

这个结构是 Bayesian inference 的核心。

## Evidence 从哪里来

若 $z$ 是 discrete：

\[
p(x)=\sum_z p(x\mid z)p(z).
\]

若 $z$ 是 continuous：

\[
p(x)=\int p(x\mid z)p(z)\,dz.
\]

所以 Bayes' theorem 真正困难的地方常常不是公式本身，而是 denominator：要对所有可能 latent states 做 sum 或 integral。

这正是 [Variational Inference](/mathematics/probability/variational-inference/) 等 approximate inference 方法出现的重要背景。

## 一个数值例子

假设某事件 $D$ 的 prior probability 为：

\[
P(D)=0.01.
\]

某测试结果 $T$ 满足：

\[
P(T\mid D)=0.95,
\qquad
P(T\mid \neg D)=0.05.
\]

则：

\[
P(T)=0.95\times0.01+0.05\times0.99=0.059.
\]

所以：

\[
P(D\mid T)
=\frac{0.95\times0.01}{0.059}
\approx0.161.
\]

虽然测试在 $D$ 发生时很容易为 positive，但由于 prior 很低，positive 后的 posterior 也不是 95%。

这说明 Bayes' theorem 把**先验发生率**和**观测证据强度**一起考虑。

## Bayes' Theorem 与机器学习

它出现在：

- Bayesian parameter inference；
- latent-variable models；
- probabilistic graphical models；
- filtering 与 state estimation；
- VAE 中真实 posterior $p(z\mid x)$ 的定义。

这些应用彼此不同，但都使用同一个概率反演结构。
