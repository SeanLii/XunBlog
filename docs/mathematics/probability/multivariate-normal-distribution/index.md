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
---

# Multivariate Normal Distribution

一维 Normal Distribution 描述一个 scalar random variable；Multivariate Normal Distribution 把它扩展到 random vector：

\[
X\in\mathbb R^d.
\]

写作：

\[
X\sim\mathcal N(\mu,\Sigma),
\]

其中：

\[
\mu\in\mathbb R^d
\]

是 mean vector，

\[
\Sigma\in\mathbb R^{d\times d}
\]

是 covariance matrix。

## Mean Vector 决定中心

\[
\mu=\mathbb E[X]
\]

逐维给出 distribution 的中心：

\[
\mu=
\begin{bmatrix}
\mathbb E[X_1]\\
\vdots\\
\mathbb E[X_d]
\end{bmatrix}.
\]

二维时，$\mu$ 就是 density ellipse 的中心位置。

## Covariance Matrix 决定尺度与方向

\[
\Sigma
=\mathbb E[(X-\mu)(X-\mu)^\top].
\]

对角线元素：

\[
\Sigma_{ii}=\operatorname{Var}(X_i)
\]

控制各 coordinate 的 spread。

非对角线：

\[
\Sigma_{ij}=\operatorname{Cov}(X_i,X_j)
\]

描述 coordinates 之间的线性共同变化。

因此 multivariate Gaussian 的几何形状不仅有“宽窄”，还有“朝哪个方向拉伸”。

## Density

当 $\Sigma$ positive definite 时：

\[
p(x)=
\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}
\exp\left(
-\frac12(x-\mu)^\top
\Sigma^{-1}
(x-\mu)
\right).
\]

其中：

\[
(x-\mu)^\top\Sigma^{-1}(x-\mu)
\]

可以看作考虑 covariance 后的“标准化平方距离”。

如果某个方向 variance 很大，那么同样的 Euclidean displacement 在那个方向上不会被认为特别罕见。

## 等密度面的 Geometry

满足：

\[
(x-\mu)^\top\Sigma^{-1}(x-\mu)=c
\]

的点形成 ellipse / ellipsoid。

Covariance matrix 的 eigenvectors 给出主轴方向，eigenvalues 控制各主轴尺度。

这把 probability distribution 与 linear algebra 直接连接起来。

## Diagonal Gaussian

如果：

\[
\Sigma=
\operatorname{diag}(\sigma_1^2,\ldots,\sigma_d^2),
\]

则不同 dimensions 的 covariance 为 0。

在 Gaussian 情况下，diagonal covariance 还意味着这些 coordinates independent。

此时 density 可以 factorize：

\[
p(x)=\prod_{i=1}^d
\mathcal N(x_i;\mu_i,\sigma_i^2).
\]

这极大简化了参数量和计算，因此很多 latent-variable models 使用 diagonal Gaussian approximate posterior。

## Standard Multivariate Normal

当：

\[
\mu=0,
\qquad
\Sigma=I,
\]

得到：

\[
Z\sim\mathcal N(0,I).
\]

各 coordinates 都是 standard normal，并且互相 independent。

这正是许多 generative models 选择的简单 prior，但它首先是一个独立的 probability distribution。

## Affine Transformation

若：

\[
X\sim\mathcal N(\mu,\Sigma),
\qquad
Y=AX+b,
\]

则：

\[
Y\sim
\mathcal N(A\mu+b,
A\Sigma A^\top).
\]

Gaussian 对 affine transformation 的这种封闭性，是它在统计推断、控制、state estimation 和 generative modeling 中非常重要的原因。
