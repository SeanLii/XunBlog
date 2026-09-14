---
title: "Expectation"
kind: "canonical"
domain: "Mathematics / Probability"
parent: "Probability"
canonical: "/mathematics/probability/expectation/"
prerequisites:
  - "/mathematics/probability/random-variable/"
  - "/mathematics/probability/probability-distribution/"
related:
  - "/mathematics/probability/variance/"
  - "/mathematics/information-theory/kl-divergence/"
---

# Expectation

期望（expectation）是随机变量在其概率分布下的加权平均。它描述的是分布的平均位置，而不是一次随机采样一定会得到的值。

## 离散随机变量

若 $X$ 的可能取值为 $x$，概率质量函数为 $p(x)$，则

\[
\mathbb E[X]=\sum_x x\,p(x).
\]

每个取值按其出现概率参与平均。

## 连续随机变量

若 $X$ 具有概率密度 $p(x)$，则

\[
\mathbb E[X]=\int_{-\infty}^{\infty}x\,p(x)\,dx.
\]

更一般地，对任意合适的函数 $f$，

\[
\mathbb E_{X\sim p}[f(X)]
=\int f(x)p(x)\,dx.
\]

这个写法在机器学习里非常常见：先说明 $X$ 服从哪个分布，再对某个量求平均。

## 线性性质

对常数 $a,b$ 和随机变量 $X,Y$，

\[
\mathbb E[aX+bY]=a\mathbb E[X]+b\mathbb E[Y].
\]

这个性质不要求 $X$ 与 $Y$ 独立。

## 在生成模型中的位置

VAE 的 ELBO 中包含

\[
\mathbb E_{q_\phi(z|x)}[\log p_\theta(x|z)].
\]

它表示：让 $z$ 按 encoder 给出的分布 $q_\phi(z|x)$ 变化，并平均评估 decoder 对 $x$ 的解释能力。训练时通常用有限次 sample 近似这个期望，而不是对所有可能的 $z$ 做解析积分。
