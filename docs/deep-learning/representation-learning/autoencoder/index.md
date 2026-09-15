---
title: "Autoencoder"
kind: "canonical"
domain: "Deep Learning / Representation Learning"
parent: "Deep Learning"
canonical: "/deep-learning/representation-learning/autoencoder/"
prerequisites:
  - "/deep-learning/core/multilayer-perceptron/"
related:
  - "/generative-models/variational-autoencoder/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Autoencoder

Autoencoder 是通过**重建输入**学习中间表示的神经网络框架。它由 Encoder 与 Decoder 两部分组成：

\[
h=f_\phi(x),
\]

\[
\hat x=g_\theta(h).
\]

其中：

- $x$ 是输入；
- $h$ 是 latent code / representation；
- $\hat x$ 是 reconstruction；
- $f_\phi$ 与 $g_\theta$ 分别是 Encoder 与 Decoder。

训练通过 reconstruction objective 使 $\hat x$ 保留 $x$ 中的信息，例如：

\[
\mathcal L_{rec}
=
\|x-\hat x\|_2^2.
\]

Autoencoder 的关键不在于“输入和输出相同”，而在于 **representation path 受到什么约束**。如果 Encoder、Decoder 容量足够大且没有有效约束，网络可以接近 identity mapping，而不必形成有用的 representation。

## Encoder、Latent Code 与 Decoder

Encoder 定义从 input space 到 representation space 的映射：

\[
f_\phi:\mathcal X\rightarrow\mathcal H.
\]

Decoder 定义反向的 reconstruction mapping：

\[
g_\theta:\mathcal H\rightarrow\mathcal X.
\]

完整模型为：

\[
\hat x
=
g_\theta(f_\phi(x)).
\]

因此 Autoencoder 的训练目标可以统一写成：

\[
\min_{\phi,\theta}
\mathbb E_{x\sim p_{data}}
\left[
\ell\bigl(x,g_\theta(f_\phi(x))\bigr)
\right].
\]

损失 $\ell$ 的选择应与数据和建模目标一致。连续实值数据常使用 squared error；二值或概率化输出可以采用其他 likelihood-based reconstruction objectives。

## Undercomplete Autoencoder

最直接的 representation constraint 是 bottleneck：

\[
x\in\mathbb R^D,
\qquad
h\in\mathbb R^d,
\qquad
d<D.
\]

此时 Encoder 必须把高维输入压缩到较低维空间。若 Decoder 仍能较好重建输入，$h$ 必须保留对 reconstruction 有价值的信息。

这种结构称为 undercomplete autoencoder。

Bottleneck 本身并不保证得到“语义表示”。它只是限制可传递的信息容量；最终学到什么仍由数据分布、网络结构与 reconstruction objective 决定。

## Linear Autoencoder 与 PCA

考虑线性 Encoder 与 Decoder，并使用 squared reconstruction error。若 latent dimension 小于输入维度，在适当条件下，最优 linear autoencoder 学到的子空间与 Principal Component Analysis（PCA）的 principal subspace 相同。

这说明 Autoencoder 可以看作 nonlinear representation learning 的推广：

- linear mapping + squared error 对应低维线性子空间；
- nonlinear Encoder / Decoder 可以学习更复杂的低维 representations。

但两者并非所有层面都完全等价。Autoencoder 的参数化并不唯一，PCA 则具有明确的正交主方向结构。

## Overcomplete Representation

如果：

\[
d\ge D,
\]

latent representation 并不构成维度 bottleneck。这样的 overcomplete autoencoder 仍然可以学习有用 representation，但必须依赖其他约束，否则 identity mapping 很容易成为低损失解。

常见约束包括：

- sparsity；
- denoising；
- contractive regularization；
- weight constraints；
- architectural restrictions。

因此“Autoencoder = 压缩到更低维”不是完整定义。低维 bottleneck 只是其中一种约束方式。

## Sparse Autoencoder

Sparse Autoencoder 对 latent activations 加入 sparsity constraint，使单个输入通常只激活少量 representation units。

抽象 objective 可以写成：

\[
\mathcal L
=
\mathcal L_{rec}
+
\lambda\,\Omega(h),
\]

其中 $\Omega(h)$ 鼓励 sparse activation。

即使 latent dimension 大于 input dimension，sparsity 也能限制信息表达方式，使模型不能简单依靠所有 units 同时传递输入。

## Denoising Autoencoder

Denoising Autoencoder 先构造 corrupted input：

\[
\tilde x\sim C(\tilde x\mid x),
\]

再训练：

\[
\hat x=g_\theta(f_\phi(\tilde x))
\]

去重建原始 clean input $x$：

\[
\mathcal L
=
\ell(x,\hat x).
\]

模型必须学习在 corruption 下仍然稳定的结构，而不能只实现逐元素 identity mapping。不同 corruption process 会改变模型被迫学习的 invariances。

## Contractive Autoencoder

Contractive Autoencoder 直接约束 Encoder 对输入小变化的敏感度。令：

\[
J_f(x)
=
\frac{\partial f_\phi(x)}{\partial x},
\]

可加入 penalty：

\[
\mathcal L
=
\mathcal L_{rec}
+
\lambda\|J_f(x)\|_F^2.
\]

这鼓励附近输入映射到相对稳定的 latent representation。

Sparse、denoising 与 contractive variants 的共同目的都是：通过额外约束决定什么样的 representation 才能以低损失重建数据。

## Reconstruction Objective 与 Representation Bias

Autoencoder 直接优化的是 reconstruction，而不是抽象语义。

如果 reconstruction loss 强烈奖励 pixel-level precision，latent code 可能优先保存纹理、位置和局部细节；如果输入经过 corruption、masking 或其他结构性处理，模型则可能被迫学习更稳定的结构。

因此 representation quality 不能只由 reconstruction loss 数值判断。较低的 reconstruction error 不自动意味着 representation 对分类、检索或控制任务更有用。

## Deterministic Representation

普通 Autoencoder 通常定义：

\[
h=f_\phi(x),
\]

因此给定同一个 $x$，latent code $h$ 是确定的。

它没有自动定义：

\[
p(h),
\qquad
p(x\mid h),
\qquad
p(h\mid x).
\]

这使普通 Autoencoder 与 probabilistic latent-variable model 有本质区别。

## Autoencoder 与 Generative Modeling

训练后可以把某个 latent code 输入 Decoder：

\[
h\rightarrow g_\theta(h),
\]

但普通 Autoencoder 并没有规定应该如何从 latent space 采样 $h$。训练样本对应的 latent codes 可能只占空间中的复杂区域；任意采样：

\[
h\sim\mathcal N(0,I)
\]

没有一般理论依据，也不保证 Decoder 输出 data-like sample。

因此普通 Autoencoder 本身不是一个完整的 probabilistic generative model。

[Variational Autoencoder](/generative-models/variational-autoencoder/) 的关键变化不是简单“给 latent 加噪声”，而是明确建立：

\[
p(z),
\qquad
p_\theta(x\mid z),
\qquad
q_\phi(z\mid x),
\]

从而把 representation learning 与概率生成模型结合起来。

## 表示能力导出的用途与边界

Autoencoder 可以用于：

- dimensionality reduction；
- representation learning；
- denoising；
- anomaly detection；
- compression-like representation；
- feature pretraining。

主要限制包括：

- reconstruction objective 与 downstream usefulness 不一定一致；
- 高容量模型可能接近 identity mapping；
- latent space 不自动具有可采样概率结构；
- latent factors 不自动具有 disentanglement 或可解释性；
- reconstruction quality 不等价于 generative quality。

## Sources

- Hinton & Salakhutdinov. *Reducing the Dimensionality of Data with Neural Networks*. Science, 2006.
- Vincent et al. *Extracting and Composing Robust Features with Denoising Autoencoders*. ICML, 2008.
- Rifai et al. *Contractive Auto-Encoders: Explicit Invariance During Feature Extraction*. ICML, 2011.
