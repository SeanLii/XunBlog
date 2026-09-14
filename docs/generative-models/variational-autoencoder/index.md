---
title: "Variational Autoencoder"
kind: "canonical"
domain: "Generative Models"
parent: "Generative Models"
canonical: "/generative-models/variational-autoencoder/"
prerequisites:
  - "/deep-learning/representation-learning/autoencoder/"
  - "/mathematics/probability/latent-variable/"
  - "/mathematics/probability/variational-inference/evidence-lower-bound/"
related:
  - "/mathematics/probability/variational-inference/reparameterization-trick/"
  - "/generative-models/conditional-variational-autoencoder/"
  - "/generative-models/posterior-collapse/"
---

# Variational Autoencoder

Variational Autoencoder（VAE）是一种把**deep latent-variable generative model**和**amortized variational inference**结合起来的模型框架。

它不是“普通 Autoencoder 加噪声”。

最重要的 mental model 是两条方向：

```text
Generative direction:
z → x

Inference direction:
x → approximate distribution over z
```

Generative model 想描述：

\[
p_\theta(x,z)=p(z)p_\theta(x\mid z).
\]

Inference network 则近似难算 posterior：

\[
q_\phi(z\mid x)
\approx
p_\theta(z\mid x).
\]

## 先看完整数据流

对一个 training sample $x$：

```text
x
↓
Encoder / Inference Network
↓
μ(x), log σ²(x)
↓
qφ(z|x)
↓
sample z
↓
Decoder / Generative Network
↓
parameters of pθ(x|z)
↓
ELBO objective
```

这里 encoder 输出的不是“最终 latent vector”，而是 approximate posterior distribution 的 parameters。

## Generative Model

常见 prior：

\[
p(z)=\mathcal N(0,I).
\]

Decoder 定义 likelihood：

\[
p_\theta(x\mid z).
\]

生成时：

\[
z\sim p(z),
\]

再：

\[
x\sim p_\theta(x\mid z).
\]

这才是 VAE 作为 generative model 的正向过程。

## Encoder / Recognition Model

真实 posterior：

\[
p_\theta(z\mid x)
\]

通常难以直接计算。

VAE 使用 neural network 产生：

\[
q_\phi(z\mid x).
\]

常设 diagonal Gaussian：

\[
q_\phi(z\mid x)
=
\mathcal N(
\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))
).
\]

encoder 因而输出：

\[
\mu_\phi(x),
\qquad
\log\sigma_\phi^2(x).
\]

## Reparameterized Sampling

直接写：

\[
z\sim
\mathcal N(\mu,\sigma^2)
\]

会让 stochastic sampling node 难以直接用普通 backpropagation 表达 parameter gradient。

VAE 使用 [Reparameterization Trick](/mathematics/probability/variational-inference/reparameterization-trick/)：

\[
\epsilon\sim\mathcal N(0,I),
\]

\[
z=\mu+\sigma\odot\epsilon.
\]

随机性被移到 parameter-independent noise $\epsilon$ 中。

## Training Objective

VAE maximize ELBO：

\[
\mathcal L(x)
=
\mathbb E_{q_\phi(z\mid x)}
[
\log p_\theta(x\mid z)
]
-
D_{KL}
(q_\phi(z\mid x)\|p(z)).
\]

第一项要求 sampled latent 能解释 observation；第二项让 approximate posterior 不要任意偏离 prior。

这两个 terms 共同来自概率推导，不是经验上随意拼出的两个 losses。

## Reconstruction Term 取决于 Likelihood

如果：

\[
p_\theta(x\mid z)
=
\mathcal N(
\mu_\theta(z),
\sigma_x^2 I
)
\]

并固定 $\sigma_x$，negative log-likelihood 与 squared error 只差 scale / constants。

如果是 Bernoulli likelihood，则对应不同 reconstruction form。

所以 VAE 的根本对象是 likelihood，而不是固定“必须用 MSE”。

## Generation

训练完成后不需要先给一个真实 $x$ 才能生成。

可以：

\[
z\sim\mathcal N(0,I),
\]

然后 decoder：

\[
p_\theta(x\mid z).
\]

这就是 VAE 相比 deterministic Autoencoder 的重要区别：prior 定义了 latent sampling mechanism。

## Latent Space Geometry

KL term 让各 observations 的 approximate posteriors 与共同 prior 建立约束，使 latent regions 不至于任意碎裂。

但“VAE latent 一定平滑、一定 disentangled”不是理论保证。具体 geometry 取决于 objective、capacity 与 data。

## Posterior Collapse

如果 decoder 很强，model 可能学会不使用 $z$：

\[
q_\phi(z\mid x)
\approx p(z),
\]

而 decoder 主要依靠自身能力解释 data。

这叫 [Posterior Collapse](/generative-models/posterior-collapse/)，是 VAE training 中的重要 failure mode。

## 从 VAE 到 CVAE

如果生成结果还需要由 observed condition $c$ 控制，可以建立：

\[
p(y,z\mid c).
\]

这进入 [Conditional Variational Autoencoder](/generative-models/conditional-variational-autoencoder/)。

## Sources

- Kingma & Welling. *Auto-Encoding Variational Bayes*. ICLR 2014.
- Kingma & Welling. *An Introduction to Variational Autoencoders*. 2019.
