---
title: "Reparameterization Trick"
kind: "canonical"
domain: "Mathematics / Probability / Variational Inference"
parent: "Variational Inference"
canonical: "/mathematics/probability/variational-inference/reparameterization-trick/"
prerequisites:
  - "/mathematics/probability/variational-inference/"
related:
  - "/mathematics/probability/variational-inference/evidence-lower-bound/"
---

# Reparameterization Trick

Reparameterization Trick 把“从一个依赖 parameters 的 distribution 中随机采样”改写成“从固定 noise distribution 采样，再做 deterministic transformation”。

以 Gaussian 为例，原本：

\[
z\sim\mathcal N(\mu,\sigma^2).
\]

改写为：

\[
\epsilon\sim\mathcal N(0,1),
\]

\[
\boxed{z=\mu+\sigma\epsilon}.
\]

得到的 $z$ distribution 完全相同：

\[
z\sim\mathcal N(\mu,\sigma^2).
\]

变化的不是 probability model，而是 computation graph 中随机性的放置位置。

## 在 Variational Inference 中的位置

今天深度学习里常说的 Reparameterization Trick，通常指 **pathwise gradient estimator** 在 stochastic variational inference 中的使用：把 parameter-dependent random variable 表示成 parameter-free noise 与 deterministic transformation 的组合。

这项思想并不属于 VAE 本身。Pathwise gradient / infinitesimal perturbation 一类方法在更早的 stochastic simulation 与 gradient estimation 中已经存在；Kingma & Welling 和 Rezende 等工作把它带入了现代 deep variational inference，并使 VAE 可以用 ordinary backpropagation 高效训练。

因此它更适合作为 **Variational Inference 中的一般梯度估计技术** 来理解；VAE 是它最重要、也最广为人知的使用场景之一。

## 原问题：Random Node 依赖 Parameters

假设 objective：

\[
L(\mu,\sigma)
=
\mathbb E_{z\sim q_{\mu,\sigma}(z)}[f(z)].
\]

如果直接在 graph 中写：

```text
μ, σ
 ↓
random sample z ~ N(μ, σ²)
 ↓
f(z)
```

普通 pathwise backprop 不容易把 gradient 穿过 sampling operation 回到 $\mu,\sigma$。

## 把 Randomness 移出去

Reparameterization 后：

```text
ε ~ N(0,1)      μ, σ
      \          /
       deterministic
       z = μ + σε
            ↓
           f(z)
```

随机变量 $\epsilon$ 与 model parameters 无关。

给定某次 sampled $\epsilon$，$z$ 对 $\mu,\sigma$ 是 deterministic differentiable function。

例如：

\[
\frac{\partial z}{\partial\mu}=1,
\qquad
\frac{\partial z}{\partial\sigma}=\epsilon.
\]

所以 gradient 可以沿普通 computation path 回传。

## Distribution Preservation

若：

\[
\epsilon\sim\mathcal N(0,1),
\]

那么 affine transformation：

\[
z=\mu+\sigma\epsilon
\]

满足：

\[
\mathbb E[z]=\mu,
\]

\[
\operatorname{Var}(z)=\sigma^2.
\]

并且 Gaussian 在 affine transformation 下仍为 Gaussian，因此 distribution 正好是：

\[
\mathcal N(\mu,\sigma^2).
\]

## Multivariate Diagonal Gaussian

VAE 中常见：

\[
q(z\mid x)
=
\mathcal N(
\mu,
\operatorname{diag}(\sigma^2)
).
\]

使用：

\[
\epsilon\sim\mathcal N(0,I),
\]

\[
z=\mu+\sigma\odot\epsilon.
\]

这里 $\odot$ 表示 element-wise multiplication。

## Log-Variance Parameterization

神经网络可以输出任意 real number，但 standard deviation 必须：

\[
\sigma>0.
\]

常让 network 输出：

\[
\log\sigma^2.
\]

再计算：

\[
\sigma
=
\exp\left(\frac12\log\sigma^2\right).
\]

这样 positivity 自动满足，也更适合数值计算。

## Pathwise Gradient 的意义

Reparameterization estimator 的核心优势是：同一份 sampled noise 下，output 会随 parameters 平滑变化，因此 gradient 能利用 function $f(z)$ 的局部 derivative information。

这通常比纯 score-function estimator 具有更低 variance，但它要求 distribution 可以写成合适的 differentiable transformation。

## 不只是 Gaussian

一般形式：

\[
\epsilon\sim p(\epsilon),
\]

\[
z=g_\phi(\epsilon),
\]

只要 base noise 与 $\phi$ 无关，且 $g_\phi$ differentiable，就可以形成 reparameterized path。

因此它是 stochastic gradient estimation 的一般技术，不是 VAE 专属 trick。

## Sources

- Kingma & Welling. *Auto-Encoding Variational Bayes*. 2013/2014. https://arxiv.org/abs/1312.6114
- Rezende, Mohamed & Wierstra. *Stochastic Backpropagation and Approximate Inference in Deep Generative Models*. 2014. https://proceedings.mlr.press/v32/rezende14.html
- Jankowiak & Obermeyer. *Pathwise Derivatives Beyond the Reparameterization Trick*. 2018. https://proceedings.mlr.press/v80/jankowiak18a.html
