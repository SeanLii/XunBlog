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

KL Divergence（Kullback–Leibler Divergence）衡量一个 probability distribution $Q$ 与另一个 distribution $P$ 的差异。

离散形式：

\[
D_{KL}(Q\|P)
=
\sum_x Q(x)
\log\frac{Q(x)}{P(x)}.
\]

连续形式：

\[
D_{KL}(Q\|P)
=
\int q(x)
\log\frac{q(x)}{p(x)}\,dx.
\]

它也可以写成 expectation：

\[
D_{KL}(Q\|P)
=
\mathbb E_{x\sim Q}
\left[
\log\frac{Q(x)}{P(x)}
\right].
\]

## KL Divergence 的比较方向

从 $Q$ 中经常出现的区域出发，如果 $P$ 也给这些区域较高 probability，log ratio 不大；如果 $Q$ 认为很常见而 $P$ 认为非常罕见，贡献就会变大。

所以 $D_{KL}(Q\|P)$ 可以理解成：**以 Q 的视角，看 P 与 Q 有多不匹配。**

## 非负性

KL divergence 满足

\[
D_{KL}(Q\|P)\ge0.
\]

当两 distributions 几乎处处相同时：

\[
D_{KL}(Q\|P)=0.
\]

这也是 ELBO 能成为 lower bound 的关键数学性质。

## KL 不是距离 metric

一般情况下：

\[
D_{KL}(Q\|P)\neq D_{KL}(P\|Q).
\]

而且它不满足普通 metric 的所有性质。

所以不能把 KL 当成 Euclidean distance 使用。

顺序很重要：

\[
D_{KL}(q(z\mid x)\|p(z))
\]

与反过来不是同一个目标。

## VAE 中的 KL

VAE 训练中：

\[
D_{KL}\big(q_\phi(z\mid x)\|p(z)\big)
\]

让 encoder 给某个样本产生的 approximate posterior 不要任意远离 prior。

当

\[
q_\phi(z\mid x)
=
\mathcal N(\mu,\operatorname{diag}(\sigma^2))
\]

且

\[
p(z)=\mathcal N(0,I),
\]

KL 有解析形式：

\[
D_{KL}
=
\frac12
\sum_{j=1}^{d}
\left(
\mu_j^2+
\sigma_j^2-
\log\sigma_j^2-
1
\right).
\]

因此不需要 Monte Carlo 才能计算这一项。

## 公式每一项的意义

如果 $\mu_j$ 离 0 很远，$\mu_j^2$ 让 KL 变大。

如果 $\sigma_j^2$ 远离 1，

\[
\sigma_j^2-
\log\sigma_j^2-1
\]

也会增大。

所以最小值出现在

\[
\mu=0,
\qquad
\sigma^2=1,
\]

也就是 approximate posterior 正好等于 standard normal prior。

## KL 在 ACT 中的作用

ACT training 使用

\[
D_{KL}(q_\phi(z\mid q_t,A_t)\|\mathcal N(0,I)).
\]

这让 training-time latent posterior 被约束在 standard normal prior 附近，使 inference 能够使用 prior mean $z=0$ 作为稳定输入。

但 KL 权重过强也可能让 latent 携带的信息过少，这与 [Posterior Collapse](/generative-models/posterior-collapse/) 有关。
