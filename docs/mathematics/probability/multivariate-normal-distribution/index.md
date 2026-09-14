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

多元正态分布（multivariate normal distribution）把一维高斯扩展到向量随机变量。它由均值向量和协方差矩阵决定。

## 定义

设

\[
\mathbf z\in\mathbb R^d.
\]

若

\[
\mathbf z\sim\mathcal N(\boldsymbol\mu,\Sigma),
\]

其中

\[
\boldsymbol\mu\in\mathbb R^d,
\qquad
\Sigma\in\mathbb R^{d\times d},
\]

则 $\boldsymbol\mu$ 描述每个维度的中心，$\Sigma$ 是协方差矩阵。对非退化情形，其密度为

\[
p(\mathbf z)=
\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}
\exp\left[-\frac12(\mathbf z-\boldsymbol\mu)^\top
\Sigma^{-1}(\mathbf z-\boldsymbol\mu)\right].
\]

## 协方差

协方差矩阵第 $(i,j)$ 个元素为

\[
\Sigma_{ij}
=\mathbb E[(Z_i-\mu_i)(Z_j-\mu_j)].
\]

对角元素是各维方差；非对角元素描述不同维度之间的线性共同变化。

## 对角高斯

VAE 常使用对角协方差：

\[
\Sigma=\operatorname{diag}(\sigma_1^2,\ldots,\sigma_d^2).
\]

这意味着近似 posterior 在给定 $x$ 后把各 latent dimension 的条件协方差设为 0。此时网络只需输出 $d$ 个均值和 $d$ 个方差参数，而不必输出完整的 $d\times d$ 协方差矩阵。

采样可以逐维写成

\[
z_i=\mu_i+\sigma_i\epsilon_i,
\qquad
\epsilon_i\sim\mathcal N(0,1).
\]

向量形式则是

\[
\mathbf z=\boldsymbol\mu+\boldsymbol\sigma\odot\boldsymbol\epsilon,
\]

其中 $\odot$ 表示逐元素乘法。

ACT 官方实现把 latent dimension 设为 32，并让 encoder 输出 32 维 $\mu$ 和 32 维 $\log\sigma^2$，对应的正是这种 diagonal Gaussian 参数化。
