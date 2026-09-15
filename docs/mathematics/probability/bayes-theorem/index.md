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
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Bayes' Theorem

> **知识边界**：本文的 canonical 对象是 **undefined**。依赖机制由 [Conditional Probability](/mathematics/probability/conditional-probability/) 的 canonical page 定义；本文只在当前语境中调用其接口。


Bayes' Theorem 描述观察 evidence 后，如何由 likelihood 与 prior 计算 posterior。

对 events：

\[
P(A\mid B)
=
\frac{P(B\mid A)P(A)}{P(B)},
\]

前提是：

\[
P(B)>0.
\]

对 probability densities，同样写成：

\[
p(z\mid x)
=
\frac{p(x\mid z)p(z)}{p(x)}.
\]

## Prior, Likelihood, Posterior, Evidence

在：

\[
p(z\mid x)
=
\frac{p(x\mid z)p(z)}{p(x)},
\]

各部分分别是：

### Prior

\[
p(z)
\]

表示观察 $x$ 之前对 $z$ 的 distribution。

### Likelihood

\[
p(x\mid z)
\]

表示假设 $z$ 给定时，observed data $x$ 的概率或 density。

### Posterior

\[
p(z\mid x)
\]

表示观察 $x$ 后对 $z$ 更新后的 distribution。

### Evidence / Marginal Likelihood

\[
p(x)
=
\int p(x\mid z)p(z)\,dz
\]

或离散情况：

\[
p(x)
=
\sum_z p(x\mid z)p(z).
\]

Evidence 负责使 posterior 归一化。

## Derivation

由 joint probability 的两种 factorization：

\[
p(x,z)=p(x\mid z)p(z),
\]

\[
p(x,z)=p(z\mid x)p(x),
\]

令两者相等：

\[
p(z\mid x)p(x)
=
p(x\mid z)p(z),
\]

得到 Bayes' theorem。

## Posterior Is Proportional to Likelihood Times Prior

对固定 observation $x$，$p(x)$ 与 $z$ 无关，因此：

\[
p(z\mid x)
\propto
p(x\mid z)p(z).
\]

这常用于只关心 posterior relative shape 或 MAP optimization 的场景。

但若需要 normalized posterior probability、marginal likelihood 或 model comparison，不能忽略 evidence。

## Posterior Odds

对两个 hypotheses $H_1,H_2$：

\[
\frac{P(H_1\mid D)}{P(H_2\mid D)}
=
\frac{P(D\mid H_1)}{P(D\mid H_2)}
\frac{P(H_1)}{P(H_2)}.
\]

即：

\[
\text{posterior odds}
=
\text{Bayes factor}
\times
\text{prior odds}.
\]

这直接展示 evidence 如何修改 prior belief。

## Sequential Bayesian Update

若 observations $x_1,\ldots,x_n$ 在给定 parameter $\theta$ 后 conditionally independent：

\[
p(\theta\mid x_{1:n})
\propto
p(\theta)
\prod_{i=1}^{n}p(x_i\mid\theta).
\]

也可以递归更新：

\[
p(\theta\mid x_{1:t})
\propto
p(x_t\mid\theta)
 p(\theta\mid x_{1:t-1}).
\]

前一步 posterior 成为下一步 prior。

## MAP 与 Maximum Likelihood

Maximum likelihood：

\[
\theta_{ML}
=
\arg\max_\theta p(D\mid\theta).
\]

Maximum a posteriori：

\[
\theta_{MAP}
=
\arg\max_\theta p(\theta\mid D)
=
\arg\max_\theta p(D\mid\theta)p(\theta).
\]

MAP 比 maximum likelihood 多使用 prior information。

## Latent-Variable Inference

在 latent-variable model 中：

\[
p(z\mid x)
=
\frac{p(x\mid z)p(z)}{p(x)}.
\]

困难通常集中在 evidence：

\[
p(x)=\int p(x\mid z)p(z)\,dz.
\]

当该积分难以计算时，需要 approximate inference，例如 [Variational Inference](/mathematics/probability/variational-inference/)。

## Conditional Bayes' Theorem

在给定 condition $c$ 后：

\[
p(z\mid x,c)
=
\frac{p(x\mid z,c)p(z\mid c)}{p(x\mid c)}.
\]

这是 conditional latent-variable models 中常见的形式。

## Connections

- [Conditional Probability](/mathematics/probability/conditional-probability/)：Bayes' theorem 的直接基础。
- [Variational Inference](/mathematics/probability/variational-inference/)：posterior intractable 时的 approximate inference。
- [Variational Autoencoder](/generative-models/variational-autoencoder/)：用 amortized inference 近似 latent posterior。
