---
title: "Multivariate Normal Distribution"
kind: "canonical"
domain: "Mathematics / Probability"
parent: "Probability"
canonical: "/mathematics/probability/multivariate-normal-distribution/"
prerequisites:
  - "/mathematics/probability/normal-distribution/"
  - "/mathematics/probability/covariance/"
  - "/mathematics/linear-algebra/matrix/"
related:
  - "/generative-models/variational-autoencoder/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Multivariate Normal Distribution

Multivariate Normal Distribution 将一维 Normal distribution 推广到 random vector。

设：

\[
X\in\mathbb R^d.
\]

若：

\[
X\sim\mathcal N(\mu,\Sigma),
\]

其中：

\[
\mu\in\mathbb R^d,
\qquad
\Sigma\in\mathbb R^{d\times d},
\]

则 $\mu$ 是 mean vector，$\Sigma$ 是 covariance matrix。

## Density

当 $\Sigma$ positive definite 时：

\[
p(x)
=
\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}
\exp\left(
-\frac12
(x-\mu)^\top
\Sigma^{-1}
(x-\mu)
\right).
\]

Quadratic term：

\[
(x-\mu)^\top\Sigma^{-1}(x-\mu)
\]

是 squared Mahalanobis distance。

## Mean 与 Covariance

\[
\mathbb E[X]=\mu,
\]

\[
\operatorname{Cov}(X)=\Sigma.
\]

Diagonal elements：

\[
\Sigma_{ii}=\operatorname{Var}(X_i),
\]

off-diagonal elements：

\[
\Sigma_{ij}=\operatorname{Cov}(X_i,X_j).
\]

因此 covariance matrix 同时控制各 dimensions 的 scale 与 linear dependency。

## Geometry

等 density contours 满足：

\[
(x-\mu)^\top\Sigma^{-1}(x-\mu)=c.
\]

它们在二维中是 ellipses，在高维中是 ellipsoids。

若 eigendecomposition：

\[
\Sigma=Q\Lambda Q^\top,
\]

则 eigenvectors 给出 principal directions，eigenvalues 给出这些 directions 上的 variances。

## Diagonal Covariance

若：

\[
\Sigma
=
\operatorname{diag}(
\sigma_1^2,\ldots,\sigma_d^2
),
\]

则 Gaussian factorizes：

\[
p(x)
=
\prod_{i=1}^{d}
\mathcal N(x_i;\mu_i,\sigma_i^2).
\]

在 multivariate Gaussian 中，diagonal covariance 表示 coordinates mutually independent。

## Standard Multivariate Normal

若：

\[
Z\sim\mathcal N(0,I),
\]

称为 standard multivariate normal。

若 $L$ 满足：

\[
LL^\top=\Sigma,
\]

例如 Cholesky factor，则：

\[
X=\mu+LZ
\]

满足：

\[
X\sim\mathcal N(\mu,\Sigma).
\]

这给出 multivariate Gaussian sampling 与 reparameterization 的基础。

## Linear Transformation

若：

\[
X\sim\mathcal N(\mu,\Sigma)
\]

且：

\[
Y=AX+b,
\]

则：

\[
Y
\sim
\mathcal N(
A\mu+b,
A\Sigma A^\top
).
\]

因此 multivariate Gaussian 对 affine transformation 封闭。

## Marginal Distribution

将 vector 分为：

\[
X=
\begin{bmatrix}
X_1\\X_2
\end{bmatrix},
\qquad
\mu=
\begin{bmatrix}
\mu_1\\\mu_2
\end{bmatrix},
\]

\[
\Sigma=
\begin{bmatrix}
\Sigma_{11}&\Sigma_{12}\\
\Sigma_{21}&\Sigma_{22}
\end{bmatrix}.
\]

则 marginal：

\[
X_1
\sim
\mathcal N(\mu_1,\Sigma_{11}).
\]

Gaussian marginal 仍然是 Gaussian。

## Conditional Distribution

Conditional distribution 同样为 Gaussian：

\[
X_1\mid X_2=x_2
\sim
\mathcal N(
\mu_{1\mid2},
\Sigma_{1\mid2}
),
\]

其中：

\[
\mu_{1\mid2}
=
\mu_1+
\Sigma_{12}\Sigma_{22}^{-1}(x_2-\mu_2),
\]

\[
\Sigma_{1\mid2}
=
\Sigma_{11}
-
\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}.
\]

Conditional mean 对 observed variable 是 affine function。

## Zero Covariance and Independence

对于 jointly Gaussian variables：

\[
\operatorname{Cov}(X_i,X_j)=0
\]

可以推出 $X_i,X_j$ independent。

这一性质不是任意 distribution 都成立；它是 Gaussian family 的特殊结构。

## Linear Combinations

对任意 vector $a$：

\[
a^\top X
\sim
\mathcal N(
a^\top\mu,
a^\top\Sigma a
).
\]

事实上，一个 random vector 是 multivariate Gaussian 的等价刻画之一，就是所有 linear combinations 都是一维 Gaussian。

## Connections

- [Covariance](/mathematics/probability/covariance/)：$\Sigma$ 决定 multivariate Gaussian geometry。
- [Reparameterization Trick](/mathematics/probability/variational-inference/reparameterization-trick/)：Gaussian latent sampling 的常用 gradient construction。
- [Variational Autoencoder](/generative-models/variational-autoencoder/)：常使用 diagonal multivariate Gaussian posterior。
