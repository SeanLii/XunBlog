---
title: "Probability Distribution"
kind: "canonical"
domain: "Mathematics / Probability"
parent: "Probability"
canonical: "/mathematics/probability/probability-distribution/"
prerequisites:
  - "/mathematics/probability/random-variable/"
related:
  - "/mathematics/probability/normal-distribution/"
  - "/mathematics/probability/conditional-probability/"
---

# Probability Distribution

概率分布（probability distribution）描述一个随机变量可能取哪些值，以及这些值具有怎样的概率规律。

## 离散分布

对离散随机变量 $X$，概率质量函数满足

\[
p_X(x)=P(X=x),\qquad p_X(x)\ge 0,
\]

并且所有可能取值的概率之和为

\[
\sum_x p_X(x)=1.
\]

## 连续分布

对连续随机变量，概率密度函数 $p_X(x)$ 满足

\[
p_X(x)\ge0,
\qquad
\int_{-\infty}^{\infty}p_X(x)\,dx=1.
\]

区间概率由密度在该区间上的积分给出：

\[
P(a\le X\le b)=\int_a^b p_X(x)\,dx.
\]

密度值本身不是“取到这个点的概率”。它描述概率在连续空间中的分布强弱。

## 参数化分布

许多分布由少量参数决定。例如 [Normal Distribution](/mathematics/probability/normal-distribution/) 可以写成

\[
X\sim\mathcal N(\mu,\sigma^2),
\]

其中 $\mu$ 决定中心位置，$\sigma^2$ 决定离散程度。

神经网络可以不直接预测随机变量的一个取值，而是预测分布参数。VAE encoder 就常输出 $\mu$ 与 $\log\sigma^2$，从而确定一个高斯近似 posterior。模型随后可以从这个分布中采样。
