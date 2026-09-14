---
title: "Multivariate Normal Distribution"
kind: "canonical"
domain: "Mathematics / Probability"
parent: "Probability"
canonical: "/mathematics/probability/multivariate-normal-distribution/"
prerequisites:
  - "/mathematics/probability/normal-distribution/"
  - "/mathematics/linear-algebra/vector/"
  - "/mathematics/linear-algebra/matrix/"
related:
  - "/generative-models/variational-autoencoder/"
  - "/robot-learning/act/cvae-in-act/"
---

# Multivariate Normal Distribution

Multivariate Normal Distribution 是一维 Normal Distribution 在多维 random vector 上的扩展。

一维：

\[
X\sim\mathcal N(\mu,\sigma^2).
\]

多维：

\[
X\sim\mathcal N(\mu,\Sigma),
\]

其中

\[
X\in\mathbb R^d,
\qquad
\mu\in\mathbb R^d,
\qquad
\Sigma\in\mathbb R^{d\times d}.
\]

## Mean 变成向量

每一维都有自己的中心：

\[
\mu=
[\mu_1,\ldots,\mu_d]^\top.
\]

## Variance 变成 Covariance Matrix

\[
\Sigma_{ii}=\operatorname{Var}(X_i)
\]

描述每一维自己的 variance。

非对角元素

\[
\Sigma_{ij}
=
\operatorname{Cov}(X_i,X_j)
\]

描述两个 dimensions 是否一起变化。

二维下：

\[
\Sigma=
\begin{bmatrix}
\sigma_1^2 & \operatorname{cov}_{12}\\
\operatorname{cov}_{12}&\sigma_2^2
\end{bmatrix}.
\]

## Geometry

二维 independent standard normal 的等密度线近似圆形：

\[
\Sigma=I.
\]

不同 dimensions variance 不同时会拉成椭圆；存在 covariance 时椭圆还会旋转。

所以 covariance matrix 同时决定 distribution 在各方向上的尺度与相关结构。

## Diagonal Gaussian

VAE 常为了简化采用 diagonal covariance：

\[
\Sigma=
\operatorname{diag}(\sigma_1^2,\ldots,\sigma_d^2).
\]

这表示在该 Gaussian approximation 中，各 latent dimensions 的 covariance 被设为 0。

Encoder 只需输出 $d$ 个 means 与 $d$ 个 variances，而不必输出完整 $d\times d$ covariance matrix。

## Standard Multivariate Normal

\[
Z\sim\mathcal N(0,I).
\]

表示：

- mean vector 全 0；
- 每一维 variance 为 1；
- covariance matrix 为 identity。

ACT 的 latent prior 就采用这种 standard normal form。
