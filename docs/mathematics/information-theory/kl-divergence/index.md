---
title: "KL Divergence"
kind: "canonical"
domain: "Mathematics / Information Theory"
parent: "Information Theory"
canonical: "/mathematics/information-theory/kl-divergence/"
prerequisites:
  - "/mathematics/information-theory/entropy/"
  - "/mathematics/information-theory/cross-entropy/"
related:
  - "/mathematics/probability/variational-inference/"
  - "/generative-models/variational-autoencoder/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# KL Divergence

> **知识边界**：本文的 canonical 对象是 **KL Divergence**。依赖机制由 [Entropy](/mathematics/information-theory/entropy/)、[Cross-Entropy](/mathematics/information-theory/cross-entropy/) 的 canonical page 定义；本文只在当前语境中调用其接口。


Kullback–Leibler Divergence 衡量一个 probability distribution $q$ 相对于 reference distribution $p$ 的 relative information difference。

Discrete case：

\[
D_{KL}(p\|q)
=
\sum_x
p(x)
\log
\frac{p(x)}{q(x)}.
\]

Continuous density case：

\[
D_{KL}(p\|q)
=
\int
p(x)
\log
\frac{p(x)}{q(x)}\,dx.
\]

它是 expectation：

\[
D_{KL}(p\|q)
=
\mathbb E_{x\sim p}
\left[
\log p(x)-\log q(x)
\right].
\]

## Non-Negativity

Gibbs' inequality 给出：

\[
D_{KL}(p\|q)\ge0.
\]

并且在合适条件下：

\[
D_{KL}(p\|q)=0
\iff
p=q
\quad\text{almost everywhere}.
\]

因此 KL divergence 可以衡量 distributions 的不一致，但它不是普通 Euclidean distance。

## Asymmetry

一般情况下：

\[
D_{KL}(p\|q)
\neq
D_{KL}(q\|p).
\]

所以 KL divergence 不满足 symmetry，也不满足一般 metric 所需的 triangle inequality。

方向决定了 expectation 在哪个 distribution 下计算，也决定了对 support mismatch 的惩罚方式。

## Relation to Cross-Entropy

Cross-entropy：

\[
H(p,q)
=
-\mathbb E_p[\log q(x)].
\]

Entropy：

\[
H(p)
=
-\mathbb E_p[\log p(x)].
\]

因此：

\[
D_{KL}(p\|q)
=
H(p,q)-H(p).
\]

对固定 $p$，minimize cross-entropy 等价于 minimize $D_{KL}(p\|q)$。

## Support Mismatch

若存在某个 region：

\[
p(x)>0,
\qquad
q(x)=0,
\]

则：

\[
D_{KL}(p\|q)=\infty.
\]

因此 forward KL 强烈惩罚 $q$ 漏掉 $p$ 支持的概率区域。

反向：

\[
D_{KL}(q\|p)
\]

则以 $q$ 作为 expectation distribution，对 behavior 的偏好不同。

## Forward and Reverse KL

### Forward KL

\[
D_{KL}(p\|q)
\]

倾向于要求 $q$ 覆盖 $p$ 的高概率 regions，因为漏掉 $p$ 支持的 region 代价非常大。

### Reverse KL

\[
D_{KL}(q\|p)
\]

更关注 $q$ 自己放置 probability mass 的 regions。当 variational family 受限且 target multimodal 时，reverse KL 可能偏向其中一个 mode。

这些“mode-covering / mode-seeking”描述是典型 intuition，不是对所有 parameterizations 的绝对定理。

## Maximum Likelihood

设 data distribution 为 $p_{data}$，model 为 $q_\theta$。

因为：

\[
D_{KL}(p_{data}\|q_\theta)
=
H(p_{data},q_\theta)-H(p_{data}),
\]

且 $H(p_{data})$ 与 $\theta$ 无关，所以 minimize forward KL 等价于 maximum likelihood / minimize expected negative log-likelihood。

## KL in Variational Inference

Variational Inference 使用 approximate posterior $q_\phi(z\mid x)$ 逼近 true posterior $p_\theta(z\mid x)$。常见目标是：

\[
D_{KL}
(q_\phi(z\mid x)
\|
 p_\theta(z\mid x)).
\]

由于 true posterior 的 normalization 往往难算，实际优化转化为 [Evidence Lower Bound](/mathematics/probability/variational-inference/evidence-lower-bound/)。

## Gaussian KL

对一维 Gaussians：

\[
q=\mathcal N(\mu_q,\sigma_q^2),
\qquad
p=\mathcal N(\mu_p,\sigma_p^2),
\]

KL 有 closed form：

\[
D_{KL}(q\|p)
=
\log\frac{\sigma_p}{\sigma_q}
+
\frac{
\sigma_q^2+(\mu_q-\mu_p)^2
}{2\sigma_p^2}
-
\frac12.
\]

VAE 常见特殊情况：

\[
p(z)=\mathcal N(0,I),
\]

\[
q(z\mid x)
=
\mathcal N(
\mu,
\operatorname{diag}(\sigma^2)
),
\]

则：

\[
D_{KL}(q\|p)
=
\frac12
\sum_j
\left(
\mu_j^2+
\sigma_j^2-
1-
\log\sigma_j^2
\right).
\]

这个 closed form 使 VAE 的 KL term 可以直接精确计算，而不需要 Monte Carlo estimate。

## Connections

- [Entropy](/mathematics/information-theory/entropy/)：KL 与 entropy / cross-entropy 直接相关。
- [Cross-Entropy](/mathematics/information-theory/cross-entropy/)：$H(p,q)=H(p)+D_{KL}(p\|q)$。
- [Variational Inference](/mathematics/probability/variational-inference/)：使用 reverse KL 进行 approximate posterior fitting。
- [Variational Autoencoder](/generative-models/variational-autoencoder/)：ELBO 中包含 posterior-to-prior KL term。
