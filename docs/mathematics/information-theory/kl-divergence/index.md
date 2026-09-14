---
title: "KL Divergence"
kind: "canonical"
domain: "Mathematics / Information Theory"
parent: "Information Theory"
canonical: "/mathematics/information-theory/kl-divergence/"
prerequisites:
  - "/mathematics/probability/probability-distribution/"
  - "/mathematics/probability/expectation/"
related:
  - "/generative-models/evidence-lower-bound/"
  - "/generative-models/variational-autoencoder/"
---

# KL Divergence

KL Divergence（Kullback–Leibler divergence）衡量：如果数据实际上按分布 $q$ 出现，却用分布 $p$ 来描述它，会产生多大的对数概率差异。它是两个分布之间的非对称差异量，不是距离度量。

## 定义

离散情形中，

\[
D_{\mathrm{KL}}(q\|p)
=\sum_x q(x)\log\frac{q(x)}{p(x)}.
\]

连续情形中，

\[
D_{\mathrm{KL}}(q\|p)
=\int q(x)\log\frac{q(x)}{p(x)}\,dx.
\]

也可以写成对 $q$ 的期望：

\[
D_{\mathrm{KL}}(q\|p)
=\mathbb E_{x\sim q}
\left[\log q(x)-\log p(x)\right].
\]

这里第一个参数 $q$ 决定“在哪些地方取平均”。因此交换顺序一般会改变结果：

\[
D_{\mathrm{KL}}(q\|p)\ne D_{\mathrm{KL}}(p\|q).
\]

## 基本性质

KL divergence 满足

\[
D_{\mathrm{KL}}(q\|p)\ge0,
\]

并且在通常条件下，仅当两个分布几乎处处相同时取 0。它不满足对称性，也不满足一般的三角不等式，所以不能当作普通欧氏距离理解。

## VAE 中的方向

VAE 使用的典型项是

\[
D_{\mathrm{KL}}
\left(q_\phi(z|x)\|p(z)\right).
\]

$q_\phi(z|x)$ 是看到数据 $x$ 后 encoder 给出的 approximate posterior，$p(z)$ 是预先规定的 prior。这个项推动 approximate posterior 不要偏离 prior 太远，从而让 latent space 保持可采样的整体结构。

它并不是要求每个输入都产生完全相同的 $q_\phi(z|x)$。如果 KL 权重有限，reconstruction term 仍然可以让不同输入使用不同的 latent 分布；训练目标是在信息保留和 prior regularization 之间取得由目标函数决定的折中。

## 对角高斯与标准正态

若

\[
q(z|x)=\mathcal N(\mu,\operatorname{diag}(\sigma^2)),
\qquad
p(z)=\mathcal N(0,I),
\]

则 KL 有闭式形式

\[
D_{\mathrm{KL}}(q\|p)
=\frac12\sum_{j=1}^{d}
\left(
\mu_j^2+\sigma_j^2-1-\log\sigma_j^2
\right).
\]

每个 latent dimension 都贡献一项。$\mu_j$ 远离 0、$\sigma_j^2$ 远离 1，都会增加 KL。ACT 官方实现正是利用这个对角高斯闭式计算 latent regularization。
