---
title: "VAE：为什么我们需要 Variational Autoencoder？"
description: "从 latent-variable model 的 posterior intractability 出发，逐步推导 approximate posterior、ELBO、reconstruction term、KL divergence 与 reparameterization trick，建立对 VAE 的完整概率学理解。"
status: reviewed
pageType: concept
canonical: /generative-models/vae
updated: "2026-09-15"
---

# VAE：为什么我们需要 Variational Autoencoder？

很多 VAE 教程从这样一张图开始：

```text
x
↓
Encoder
↓
μ, σ²
↓
sample z
↓
Decoder
↓
x̂
```

然后告诉你：

> VAE 是一个“带随机采样的 Autoencoder”。

这个说法可以帮助你快速认识网络结构。

但它没有回答最重要的问题：

> **为什么我们非要把 Autoencoder 改成这样？**

为什么 encoder 不直接输出一个向量 $z$？

为什么要输出：

$$
\mu,\sigma^2
$$

为什么还要从一个分布里采样？

为什么 loss 里面突然出现：

$$
D_{KL}
$$

为什么训练时要让 latent distribution 接近：

$$
\mathcal N(0,I)
$$

以及最关键的：

> **Variational 到底是什么意思？**

如果这些问题没有回答，VAE 很容易变成一套需要死记的结构。

这一篇我们不从网络结构开始。

而是从 VAE 原论文真正的问题开始：

> **我们有一个 latent-variable generative model，但真实 posterior 很难计算。怎么办？**

---

## 1. 前置知识：Latent Variable Model

如果还没有读过：

- [Latent Variable：模型为什么需要一个“看不见的变量”？](./latent-variable.md)

建议先读这一篇。

这里快速复习。

我们假设观测数据：

$$
x
$$

不是直接凭空产生的。

模型引入一个没有直接观测到的 latent variable：

$$
z
$$

生成过程为：

$$
z\sim p(z)
$$

然后：

$$
x\sim p_\theta(x\mid z)
$$

因此联合分布：

$$
\boxed{
p_\theta(x,z)
=
p(z)p_\theta(x\mid z)
}
$$

可以画成：

```text
z
│
│  pθ(x | z)
▼
x
```

其中：

- $z$：latent variable；
- $x$：observed variable；
- $p(z)$：prior；
- $p_\theta(x\mid z)$：generative distribution / likelihood；
- $\theta$：generative model 的参数。

VAE 的核心问题，就是：

> **如何高效训练这样一个复杂的 latent-variable model？**

---

## 2. 我们真正想训练的是什么？

假设训练数据为：

$$
\mathcal D
=
\{x^{(1)},x^{(2)},\ldots,x^{(N)}\}
$$

我们希望模型给真实数据较高概率。

也就是说，希望最大化：

$$
p_\theta(x)
$$

对于整个数据集，通常最大化 log-likelihood：

$$
\sum_{i=1}^{N}
\log p_\theta(x^{(i)})
$$

所以问题似乎非常简单：

> 直接计算 $\log p_\theta(x)$，然后对 $\theta$ 做梯度优化不就行了吗？

困难马上出现。

---

## 3. 因为 z 没有被观察，所以必须把它积分掉

模型定义的是：

$$
p_\theta(x,z)
$$

但数据集里只有：

$$
x
$$

没有真正的：

$$
z
$$

因此为了得到：

$$
p_\theta(x)
$$

必须对所有可能的 $z$ 做 marginalization：

$$
\boxed{
p_\theta(x)
=
\int
p_\theta(x,z)\,dz
}
$$

根据：

$$
p_\theta(x,z)
=
p(z)p_\theta(x\mid z)
$$

得到：

$$
\boxed{
p_\theta(x)
=
\int
p(z)p_\theta(x\mid z)\,dz
}
$$

于是：

$$
\log p_\theta(x)
=
\log
\int
p(z)p_\theta(x\mid z)\,dz
$$

这就是我们真正希望最大化的 marginal log-likelihood。

如果不熟悉 marginalization，可以阅读：

- Marginalization

---

## 4. 问题：这个积分通常算不出来

如果：

$$
p_\theta(x\mid z)
$$

是一个非常简单的数学函数，

有时积分可以解析求解。

但 VAE 希望：

$$
p_\theta(x\mid z)
$$

可以由神经网络参数化。

例如：

```text
z
↓
MLP / CNN / Transformer
↓
distribution parameters of x
```

这时：

$$
\int
p(z)p_\theta(x\mid z)\,dz
$$

通常没有一个简单的闭式解。

Kingma & Welling 在 VAE 原论文中把这称为：

> **intractable marginal likelihood**

也就是说：

> 理论上定义得出来，但实际无法直接精确计算或高效求导。

这不仅影响：

$$
p_\theta(x)
$$

还直接影响另一个更关键的量。

---

## 5. 我们还想知道：看到 x 后，z 应该是什么？

生成方向是：

$$
z\rightarrow x
$$

但训练数据给我们的方向却是：

$$
x
$$

已经在眼前。

我们自然会问：

> 哪些 latent variable $z$ 最可能解释这个 $x$？

这就是 posterior：

$$
p_\theta(z\mid x)
$$

根据 Bayes' Rule：

$$
\boxed{
p_\theta(z\mid x)
=
\frac{
p_\theta(x\mid z)p(z)
}{
p_\theta(x)
}
}
$$

而分母：

$$
p_\theta(x)
=
\int
p(z)p_\theta(x\mid z)\,dz
$$

恰好就是刚才那个难算的积分。

所以：

$$
\boxed{
p_\theta(z\mid x)
}
$$

通常也无法直接精确计算。

这就是 VAE 真正的起点。

---

## 6. VAE 解决的核心问题，不是“如何压缩图片”

VAE 原论文开头真正提出的问题是：

> 当 directed probabilistic model 中存在 continuous latent variables，并且 posterior distribution 是 intractable 的时候，如何高效地进行 inference 和 learning？

所以 VAE 的故事不是：

```text
普通 Autoencoder
↓
觉得 latent space 不够漂亮
↓
加一点随机性
↓
得到 VAE
```

更准确的逻辑是：

```text
Latent-variable generative model
        ↓
希望最大化 pθ(x)
        ↓
需要 marginalize z
        ↓
积分难算
        ↓
真实 posterior pθ(z | x) 也难算
        ↓
怎么办？
```

答案是：

> **不要强行精确求 posterior。找一个可以计算的分布去近似它。**

---

## 7. 引入 Approximate Posterior

VAE 引入：

$$
\boxed{
q_\phi(z\mid x)
}
$$

去近似真正但难计算的：

$$
p_\theta(z\mid x)
$$

也就是希望：

$$
q_\phi(z\mid x)
\approx
p_\theta(z\mid x)
$$

这里：

- $q$：我们选择的近似分布；
- $\phi$：近似分布的参数；
- $p_\theta(z\mid x)$：真实 posterior；
- $q_\phi(z\mid x)$：approximate posterior / variational posterior。

在 VAE 中：

$$
q_\phi(z\mid x)
$$

通常由一个 neural network 参数化。

所以这个网络也叫：

> **Inference Model**

或者：

> **Recognition Model**

也可以叫：

> **Probabilistic Encoder**

---

## 8. 为什么它叫 Probabilistic Encoder？

普通 Autoencoder 的 encoder 可能是：

$$
z=f_\phi(x)
$$

给定 $x$，直接输出一个固定向量 $z$。

而 VAE encoder 做的是：

$$
x
\longrightarrow
q_\phi(z\mid x)
$$

也就是说：

> 给定 $x$，输出一个关于“哪些 $z$ 可能解释这个 $x$”的概率分布。

在经典 Gaussian VAE 中：

$$
q_\phi(z\mid x)
=
\mathcal N
\left(
\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))
\right)
$$

因此 encoder 输出的不是最终固定的：

$$
z
$$

而是这个 distribution 的参数：

$$
\mu_\phi(x)
$$

以及：

$$
\sigma_\phi^2(x)
$$

实际代码中，经常输出：

$$
\log\sigma^2
$$

而不是直接输出 variance。

后面再从这个分布采样：

$$
z\sim q_\phi(z\mid x)
$$

---

## 9. 但现在出现一个新问题

我们虽然定义了：

$$
q_\phi(z\mid x)
$$

但怎么训练它？

我们真正希望：

$$
q_\phi(z\mid x)
$$

接近：

$$
p_\theta(z\mid x)
$$

最直接想到的是：

> 用 KL divergence 测量两个分布的差距。

即：

$$
D_{KL}
\left(
q_\phi(z\mid x)
\parallel
p_\theta(z\mid x)
\right)
$$

然后把它最小化。

听起来很好。

但有一个致命问题：

$$
p_\theta(z\mid x)
$$

本来就是我们算不出来的东西。

所以这个 KL 也不能直接计算。

我们似乎绕回了原点。

---

## 10. ELBO 就是在这个地方出现的

关键恒等式是：

$$
\boxed{
\log p_\theta(x)
=
\mathcal L(\theta,\phi;x)
+
D_{KL}
\left(
q_\phi(z\mid x)
\parallel
p_\theta(z\mid x)
\right)
}
$$

其中：

$$
\mathcal L(\theta,\phi;x)
$$

叫：

> **Evidence Lower Bound**

简称：

> **ELBO**

为什么叫 Lower Bound？

因为 KL divergence 永远：

$$
D_{KL}(q\parallel p)\ge0
$$

所以：

$$
\log p_\theta(x)
\ge
\mathcal L(\theta,\phi;x)
$$

也就是：

$$
\boxed{
\mathcal L(\theta,\phi;x)
\le
\log p_\theta(x)
}
$$

ELBO 是 marginal log-likelihood 的一个下界。

---

## 11. 这个恒等式为什么成立？

我们一步一步推。

先从：

$$
D_{KL}
\left(
q_\phi(z\mid x)
\parallel
p_\theta(z\mid x)
\right)
$$

的定义开始：

$$
D_{KL}
=
\mathbb E_{q_\phi(z\mid x)}
\left[
\log
\frac{
q_\phi(z\mid x)
}{
p_\theta(z\mid x)
}
\right]
$$

根据 Bayes：

$$
p_\theta(z\mid x)
=
\frac{
p_\theta(x,z)
}{
p_\theta(x)
}
$$

代进去：

$$
D_{KL}
=
\mathbb E_q
\left[
\log
\frac{
q_\phi(z\mid x)
}{
p_\theta(x,z)/p_\theta(x)
}
\right]
$$

整理：

$$
D_{KL}
=
\mathbb E_q
\left[
\log q_\phi(z\mid x)
-
\log p_\theta(x,z)
+
\log p_\theta(x)
\right]
$$

注意：

$$
\log p_\theta(x)
$$

与 $z$ 无关。

所以可以从 expectation 中拿出来：

$$
D_{KL}
=
\log p_\theta(x)
+
\mathbb E_q
\left[
\log q_\phi(z\mid x)
-
\log p_\theta(x,z)
\right]
$$

移项：

$$
\log p_\theta(x)
=
D_{KL}
+
\mathbb E_q
\left[
\log p_\theta(x,z)
-
\log q_\phi(z\mid x)
\right]
$$

于是定义：

$$
\boxed{
\mathcal L(\theta,\phi;x)
=
\mathbb E_{q_\phi(z\mid x)}
\left[
\log p_\theta(x,z)
-
\log q_\phi(z\mid x)
\right]
}
$$

就得到：

$$
\boxed{
\log p_\theta(x)
=
\mathcal L
+
D_{KL}(q_\phi(z\mid x)\parallel p_\theta(z\mid x))
}
$$

这就是 ELBO 的来源。

它不是凭空设计出来的一项 loss。

---

## 12. 为什么最大化 ELBO 有意义？

回到：

$$
\log p_\theta(x)
=
\mathcal L
+
D_{KL}
\left(
q_\phi(z\mid x)
\parallel
p_\theta(z\mid x)
\right)
$$

KL：

$$
\ge0
$$

所以提高：

$$
\mathcal L
$$

会推动我们提高：

$$
\log p_\theta(x)
$$

的下界。

同时，ELBO 与 approximate posterior：

$$
q_\phi(z\mid x)
$$

有关。

因此我们可以：

> **同时训练 generative model $p_\theta$ 和 inference model $q_\phi$。**

这就是 variational learning 的核心。

---

## 13. ELBO 还可以写成更熟悉的形式

我们已经有：

$$
\mathcal L
=
\mathbb E_q
[
\log p_\theta(x,z)
-
\log q_\phi(z\mid x)
]
$$

联合分布：

$$
p_\theta(x,z)
=
p_\theta(x\mid z)p(z)
$$

所以：

$$
\log p_\theta(x,z)
=
\log p_\theta(x\mid z)
+
\log p(z)
$$

代进去：

$$
\mathcal L
=
\mathbb E_q
[
\log p_\theta(x\mid z)
+
\log p(z)
-
\log q_\phi(z\mid x)
]
$$

拆开：

$$
\mathcal L
=
\mathbb E_q
[
\log p_\theta(x\mid z)
]
+
\mathbb E_q
[
\log p(z)
-
\log q_\phi(z\mid x)
]
$$

而：

$$
D_{KL}
(q_\phi(z\mid x)\parallel p(z))
=
\mathbb E_q
\left[
\log q_\phi(z\mid x)
-
\log p(z)
\right]
$$

所以：

$$
\boxed{
\mathcal L
=
\mathbb E_{q_\phi(z\mid x)}
[
\log p_\theta(x\mid z)
]
-
D_{KL}
\left(
q_\phi(z\mid x)
\parallel
p(z)
\right)
}
$$

这就是 VAE 最经典的目标函数形式。

---

## 14. 终于出现了大家熟悉的两部分

ELBO：

$$
\boxed{
\mathcal L
=
\underbrace{
\mathbb E_{q_\phi(z\mid x)}
[
\log p_\theta(x\mid z)
]
}_{\text{reconstruction / likelihood term}}
-
\underbrace{
D_{KL}
(
q_\phi(z\mid x)
\parallel
p(z)
)
}_{\text{KL regularization}}
}
$$

训练时我们希望：

$$
\max_{\theta,\phi}\mathcal L
$$

机器学习代码通常用 gradient descent 最小化 loss。

所以会写成负 ELBO：

$$
\boxed{
\mathcal J_{\text{VAE}}
=
-
\mathbb E_q[\log p_\theta(x\mid z)]
+
D_{KL}(q_\phi(z\mid x)\parallel p(z))
}
$$

于是大家常说：

```text
VAE Loss
=
Reconstruction Loss
+
KL Loss
```

但这里要非常小心：

> 这只是把 **negative ELBO** 写成 loss 的形式。

不是研究者随便把两个 loss 拼起来。

---

## 15. Reconstruction Term 真正是什么？

第一项：

$$
\mathbb E_{q_\phi(z\mid x)}
[
\log p_\theta(x\mid z)
]
$$

含义是：

> 从 approximate posterior 中得到 $z$ 后，decoder 应该给真实数据 $x$ 较高概率。

因此 decoder 定义：

$$
p_\theta(x\mid z)
$$

VAE 想让：

$$
\log p_\theta(x\mid z)
$$

尽量大。

也就是：

> 给定这个 latent $z$，原始 $x$ 在 decoder distribution 下应该很合理。

这就是所谓 reconstruction 的概率学来源。

---

## 16. Reconstruction Loss 不总是 MSE

这是一个特别重要的严谨性问题。

经常有人直接写：

$$
\mathcal L_{\text{recon}}
=
\|x-\hat x\|^2
$$

然后说：

> VAE 的 reconstruction loss 就是 MSE。

**不准确。**

真正的目标是：

$$
-\log p_\theta(x\mid z)
$$

具体长什么样，取决于你为：

$$
p_\theta(x\mid z)
$$

选择什么概率分布。

---

## 17. 如果 decoder 使用 Gaussian likelihood

假设：

$$
p_\theta(x\mid z)
=
\mathcal N
(
\mu_\theta(z),
\sigma_x^2I
)
$$

并且：

$$
\sigma_x^2
$$

固定。

Gaussian negative log-likelihood 可以写成：

$$
-\log p_\theta(x\mid z)
=
\frac{1}{2\sigma_x^2}
\|x-\mu_\theta(z)\|^2
+
C
$$

其中：

$$
C
$$

与模型输出无关。

所以在这种特定假设下：

> 最大化 Gaussian log-likelihood 等价于最小化一个按方差缩放的 squared error。

这就是 MSE 和 reconstruction term 的关系。

---

## 18. 如果 decoder 使用 Bernoulli likelihood

对于 binary data，VAE 原论文示例使用 Bernoulli decoder。

此时：

$$
p_\theta(x\mid z)
$$

是 Bernoulli distribution。

对应的：

$$
-\log p_\theta(x\mid z)
$$

会变成类似 binary cross-entropy 的形式。

所以：

```text
Gaussian likelihood
→ squared-error-like objective

Bernoulli likelihood
→ binary-cross-entropy-like objective
```

因此最严谨的说法应该是：

> **VAE 的 reconstruction term 是 expected log-likelihood；MSE 或 BCE 是在特定 likelihood 假设下得到的具体形式。**

---

## 19. KL Term 在做什么？

第二项：

$$
D_{KL}
\left(
q_\phi(z\mid x)
\parallel
p(z)
\right)
$$

要求 approximate posterior：

$$
q_\phi(z\mid x)
$$

不要离 prior：

$$
p(z)
$$

太远。

经典 VAE 选择：

$$
\boxed{
p(z)=\mathcal N(0,I)
}
$$

因此 KL 项鼓励：

$$
q_\phi(z\mid x)
$$

接近标准多元高斯。

Kingma & Welling 原论文明确把这一项解释成：

> 对 approximate posterior 的 regularization。

---

## 20. 为什么不能只做 Reconstruction？

假设没有 KL：

$$
\mathcal J
=
\mathcal J_{\text{reconstruction}}
$$

encoder 可以非常自由地给每个训练样本分配不同的 latent region。

例如：

```text
x₁ → latent far left
x₂ → latent far upper-right
x₃ → latent another isolated island
```

只要 decoder 能重建回来，训练 objective 可能就满意。

于是 latent space 可能变成：

```text
●               ●


         ●


                            ●
```

中间存在大片没有训练数据对应的“洞”。

这对普通 reconstruction 也许没有问题。

但对 generative model 来说有问题。

因为我们希望训练后可以：

$$
z\sim p(z)
$$

然后：

$$
x\sim p_\theta(x\mid z)
$$

生成合理样本。

如果 encoder 学出来的 latent codes 与 prior 完全不匹配，

从：

$$
p(z)
$$

随机采样的 $z$ 可能掉到 decoder 从未学好的区域。

---

## 21. KL 让“编码分布”和“生成时采样的分布”对得上

训练时 latent 来自：

$$
q_\phi(z\mid x)
$$

生成时 latent 来自：

$$
p(z)
$$

如果这两个世界完全没有关系：

```text
训练时 z
来自某些奇怪小岛

生成时 z
来自 N(0, I)
```

那么 generation 就会失败。

所以 KL：

$$
D_{KL}
(
q_\phi(z\mid x)\parallel p(z)
)
$$

在做一个非常关键的事情：

> **让训练时使用的 latent distribution 不要和生成时使用的 prior 完全脱节。**

这也是 VAE 为什么能够：

```text
sample z ~ N(0, I)
↓
decoder
↓
generate x
```

的重要基础。

---

## 22. KL 是不是强迫每一个样本都 μ=0、σ²=1？

这是非常常见的误解。

经典 Gaussian VAE 中：

$$
q_\phi(z\mid x)
=
\mathcal N
(
\mu(x),
\operatorname{diag}(\sigma^2(x))
)
$$

prior：

$$
p(z)=\mathcal N(0,I)
$$

如果只考虑：

$$
D_{KL}(q_\phi(z\mid x)\parallel p(z))
$$

这一项，

它确实在：

$$
\mu=0,\qquad\sigma^2=1
$$

时达到最小值。

但是 VAE **不是只优化 KL**。

它同时还要做好 reconstruction：

$$
\mathbb E_q[\log p_\theta(x\mid z)]
$$

所以训练实际上是两种需求之间的平衡：

```text
Reconstruction
希望 z 保留足够多的 x 信息

          ↕

KL
希望 q(z | x) 不要偏离 prior 太远
```

因此正常训练时：

> 不应该理解成“一个 linear layer 专门学习把所有 $\mu$ 变成 0、所有 variance 变成 1”。

更准确的是：

> **整个 encoder 在 ELBO 的共同作用下学习 distribution parameters；KL 给这些 distributions 一个朝 prior 靠近的压力，而 reconstruction 又要求它们保留足够的数据相关信息。**

---

## 23. 这也是为什么 μ 和 variance 不是随便来的

经典 VAE 假设：

$$
q_\phi(z\mid x)
=
\mathcal N
(
\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))
)
$$

所以为了完整描述这个 Gaussian，

encoder 必须给出：

$$
\mu_\phi(x)
$$

和：

$$
\sigma_\phi^2(x)
$$

也就是说：

```text
x
↓
encoder backbone
↓
hidden representation h
↙                  ↘
μ head              variance head
↓                    ↓
μ(x)                 log σ²(x)
```

这些 linear heads 的作用是：

> **把 encoder representation 映射成 approximate posterior 的 distribution parameters。**

它们不是被单独监督成：

```text
μ target = 0
variance target = 1
```

而是通过整个 VAE objective 一起学习。

---

## 24. 为什么代码里经常输出 log σ²？

理论可以写：

$$
\sigma^2>0
$$

但是 neural network 的普通 linear output：

$$
r\in(-\infty,+\infty)
$$

没有天然保证输出为正数。

所以常见实现让网络输出：

$$
\log\sigma^2
$$

它可以是任何实数。

再通过：

$$
\sigma^2
=
\exp(\log\sigma^2)
$$

保证 variance：

$$
>0
$$

随后：

$$
\sigma
=
\exp
\left(
\frac12\log\sigma^2
\right)
$$

所以代码中经常看到：

```python
std = torch.exp(0.5 * logvar)
```

这不是神秘技巧。

只是因为：

$$
\log\sigma^2
$$

比直接让网络输出一个必须为正的 variance 更方便参数化。

---

## 25. Gaussian KL 可以写成闭式公式

对于：

$$
q_\phi(z\mid x)
=
\mathcal N
(
\mu,
\operatorname{diag}(\sigma^2)
)
$$

和：

$$
p(z)
=
\mathcal N(0,I)
$$

KL divergence 有解析解：

$$
\boxed{
D_{KL}
(q_\phi(z\mid x)\parallel p(z))
=
\frac12
\sum_{j=1}^{d}
\left(
\mu_j^2
+
\sigma_j^2
-
\log\sigma_j^2
-
1
\right)
}
$$

其中：

$$
d
$$

是 latent dimension。

如果代码使用：

$$
\text{logvar}_j
=
\log\sigma_j^2
$$

那么：

$$
\sigma_j^2
=
e^{\text{logvar}_j}
$$

所以常见代码为：

```python
kl = -0.5 * torch.sum(
    1 + logvar - mu.pow(2) - logvar.exp()
)
```

它和上面的数学公式完全等价。

完整推导建议单独阅读：

- [KL Divergence](../mathematics/kl-divergence.md)

---

## 26. 到这里还有一个困难：我们需要从 q(z|x) 采样

ELBO 第一项：

$$
\mathbb E_{q_\phi(z\mid x)}
[
\log p_\theta(x\mid z)
]
$$

包含一个 expectation。

实际训练时通常使用 Monte Carlo sample：

$$
z\sim q_\phi(z\mid x)
$$

然后计算：

$$
\log p_\theta(x\mid z)
$$

所以训练流程变成：

```text
x
↓
encoder
↓
μ, σ
↓
sample z
↓
decoder
↓
reconstruction objective
```

看起来没问题。

但现在出现了反向传播问题。

---

## 27. 为什么直接“sample z”会让梯度变麻烦？

假设：

$$
z\sim
\mathcal N(
\mu_\phi(x),
\sigma_\phi^2(x)
)
$$

其中：

$$
\mu_\phi(x),\sigma_\phi(x)
$$

由 encoder 参数：

$$
\phi
$$

决定。

我们希望 reconstruction loss 的梯度能够一路传回 encoder：

```text
loss
↑
decoder
↑
z
↑
μ, σ
↑
encoder φ
```

但如果把：

$$
z\sim q_\phi(z\mid x)
$$

当成一个直接的随机采样操作，

随机节点本身并不是普通 deterministic differentiable function。

我们不能简单地像：

$$
z=f_\phi(x)
$$

那样直接对 sample operation 进行普通 backpropagation。

VAE 原论文的关键贡献之一，就是用：

> **Reparameterization Trick**

解决这个问题。

---

## 28. Reparameterization Trick

对于 Gaussian：

$$
z\sim
\mathcal N(\mu,\sigma^2)
$$

等价地可以：

先采样：

$$
\epsilon\sim\mathcal N(0,I)
$$

然后：

$$
\boxed{
z
=
\mu+\sigma\odot\epsilon
}
$$

其中：

$$
\odot
$$

表示逐元素乘法。

于是随机性被移到了：

$$
\epsilon
$$

里。

$\epsilon$ 与：

$$
\phi
$$

无关。

而：

$$
z
=
\mu_\phi(x)
+
\sigma_\phi(x)\odot\epsilon
$$

对于固定的 $\epsilon$ 来说，是一个普通 differentiable function。

于是梯度可以：

```text
loss
↑
z = μ + σ ⊙ ε
↑          ↑
μ          σ
↑          ↑
encoder parameters φ
```

正常传播。

---

## 29. 为什么这没有改变 z 的分布？

因为如果：

$$
\epsilon\sim\mathcal N(0,I)
$$

那么：

$$
\sigma\odot\epsilon
$$

会把标准差从：

$$
1
$$

缩放成：

$$
\sigma
$$

再加：

$$
\mu
$$

会把均值从：

$$
0
$$

平移到：

$$
\mu
$$

所以：

$$
z=\mu+\sigma\odot\epsilon
$$

仍然满足：

$$
z\sim\mathcal N(\mu,\operatorname{diag}(\sigma^2))
$$

因此我们只是：

> **换了一种等价的采样写法。**

没有改变目标 distribution。

详细内容：

- [Reparameterization Trick](./reparameterization-trick.md)

---

## 30. ε 和 z 千万不要混

在公式：

$$
z
=
\mu+\sigma\odot\epsilon
$$

中：

#### $\epsilon$

是：

> auxiliary noise

通常：

$$
\epsilon\sim\mathcal N(0,I)
$$

---

#### $z$

是：

> latent sample

它服从：

$$
q_\phi(z\mid x)
$$

所以：

$$
\boxed{
z\neq\epsilon
}
$$

只有在特殊情况：

$$
\mu=0,\qquad\sigma=1
$$

时数值分布才相同。

---

## 31. VAE 的完整训练流程

现在终于可以把整个网络串起来。

---

### Step 1：输入 x

$$
x
$$

进入 encoder。

---

### Step 2：Encoder 输出 approximate posterior 参数

$$
\mu_\phi(x)
$$

以及：

$$
\log\sigma_\phi^2(x)
$$

于是定义：

$$
q_\phi(z\mid x)
=
\mathcal N
(
\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))
)
$$

---

### Step 3：采样 ε

$$
\epsilon
\sim
\mathcal N(0,I)
$$

---

### Step 4：Reparameterize

$$
z
=
\mu
+
\sigma\odot\epsilon
$$

---

### Step 5：Decoder 定义 p(x|z)

将：

$$
z
$$

输入 decoder，

得到：

$$
p_\theta(x\mid z)
$$

的 distribution parameters。

例如 Gaussian decoder 可能输出：

$$
\mu_\theta(z)
$$

---

### Step 6：计算 Reconstruction / Likelihood Term

$$
-\log p_\theta(x\mid z)
$$

或者它对应的具体实现形式。

---

### Step 7：计算 KL

$$
D_{KL}
(
q_\phi(z\mid x)
\parallel
p(z)
)
$$

---

### Step 8：组合 negative ELBO

$$
\mathcal J
=
-
\mathbb E_q
[
\log p_\theta(x\mid z)
]
+
D_{KL}
(q_\phi(z\mid x)\parallel p(z))
$$

---

### Step 9：Backpropagation

同时更新：

$$
\phi
$$

和：

$$
\theta
$$

也就是：

- encoder；
- decoder。

---

## 32. 用一张图看完整数据流

```text
                         ┌──────────────────────┐
                         │       input x        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                            ┌──────────────┐
                            │   Encoder    │
                            │    qφ(z|x)   │
                            └──────┬───────┘
                                   │
                       ┌───────────┴───────────┐
                       ▼                       ▼
                      μ(x)                 log σ²(x)
                       │                       │
                       │                       ▼
                       │                      σ(x)
                       │                       │
                       └───────────┬───────────┘
                                   │
                ε ~ N(0, I) ───────┤
                                   ▼
                       z = μ + σ ⊙ ε
                                   │
                                   ▼
                            ┌──────────────┐
                            │   Decoder    │
                            │   pθ(x|z)    │
                            └──────┬───────┘
                                   │
                                   ▼
                        likelihood / reconstruction


同时：

qφ(z|x)
   │
   └────────── KL ────────── p(z)=N(0,I)
```

两部分一起构成 ELBO。

---

## 33. 为什么叫 Variational？

现在终于可以解释名字里的：

> **Variational**

VAE 使用的是：

> **Variational Inference**

核心思想是：

真实 posterior：

$$
p_\theta(z\mid x)
$$

很难算。

于是我们选一个容易处理的 distribution family：

$$
q_\phi(z\mid x)
$$

然后在这个 family 中寻找一个尽可能好的近似。

换句话说：

> 把“直接计算 posterior”的问题，转成“优化一个近似 distribution”的问题。

这个“在一族 distribution 中通过优化寻找最佳近似”的思想，就是 variational inference 的核心。

---

## 34. 那为什么叫 Autoencoder？

因为当：

$$
q_\phi(z\mid x)
$$

和：

$$
p_\theta(x\mid z)
$$

都由 neural networks 参数化时，

整个计算图看起来像：

```text
x
↓
encoder
↓
z
↓
decoder
↓
x
```

与 Autoencoder 很像。

Kingma & Welling 也明确指出这种 connection。

但：

> VAE 的理论核心不是普通 Autoencoder 的“压缩—解压”。

而是：

> **latent-variable probabilistic model + variational inference + stochastic gradient optimization。**

---

## 35. VAE 和普通 Autoencoder 到底差在哪？

### 普通 Autoencoder

典型形式：

$$
z=f_\phi(x)
$$

$$
\hat x=g_\theta(z)
$$

latent：

$$
z
$$

通常是一个 deterministic point。

训练目标往往直接是：

$$
\|x-\hat x\|
$$

---

### VAE

Encoder：

$$
q_\phi(z\mid x)
$$

输出的是 distribution。

sample：

$$
z\sim q_\phi(z\mid x)
$$

Decoder：

$$
p_\theta(x\mid z)
$$

也是概率 distribution。

训练目标是：

$$
\text{ELBO}
$$

也就是：

$$
\mathbb E_q[\log p_\theta(x\mid z)]
-
D_{KL}(q_\phi(z\mid x)\parallel p(z))
$$

所以 VAE 从头到尾都是：

> **概率生成模型。**

---

## 36. VAE 训练好以后怎么生成新数据？

这是 KL prior matching 真正发挥作用的地方。

训练完成以后，不需要输入：

$$
x
$$

也不需要 encoder。

直接：

$$
z\sim p(z)
$$

例如：

$$
z\sim\mathcal N(0,I)
$$

然后：

$$
x\sim p_\theta(x\mid z)
$$

也就是：

```text
sample z from prior
↓
decoder
↓
generate x
```

所以 generation 阶段：

> **只需要 generative model。**

encoder 主要用于：

> inference / training / representation。

---

## 37. Reconstruction 和 Generation 是两条不同流程

### Reconstruction

有一个已有数据：

$$
x
$$

流程：

```text
x
↓
qφ(z | x)
↓
z
↓
pθ(x | z)
↓
reconstruct x
```

---

### Generation

没有输入数据：

```text
z ~ p(z)
↓
pθ(x | z)
↓
new x
```

这个区别非常重要。

VAE 不是只会：

> “把输入还原回来”。

它真正学习的是一个：

$$
p_\theta(x,z)
$$

生成模型。

---

## 38. ELBO 为什么既训练 Encoder 又训练 Decoder？

看：

$$
\mathcal L
=
\mathbb E_{q_\phi(z\mid x)}
[
\log p_\theta(x\mid z)
]
-
D_{KL}
(
q_\phi(z\mid x)
\parallel
p(z)
)
$$

#### $\theta$

出现在：

$$
p_\theta(x\mid z)
$$

所以 reconstruction / likelihood term 会训练 decoder。

---

#### $\phi$

出现在：

$$
q_\phi(z\mid x)
$$

因此：

- sample $z$ 依赖 $\phi$；
- KL 也依赖 $\phi$。

通过 reparameterization：

$$
z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon
$$

reconstruction gradient 也能传回：

$$
\phi
$$

所以：

> encoder 和 decoder 可以 end-to-end 联合训练。

这正是 AEVB 方法的重要贡献之一。

---

## 39. ELBO 的另一个非常重要的理解

回到恒等式：

$$
\log p_\theta(x)
=
\mathcal L
+
D_{KL}
(
q_\phi(z\mid x)
\parallel
p_\theta(z\mid x)
)
$$

如果：

$$
q_\phi(z\mid x)
=
p_\theta(z\mid x)
$$

那么：

$$
D_{KL}=0
$$

于是：

$$
\boxed{
\mathcal L
=
\log p_\theta(x)
}
$$

也就是说：

> approximate posterior 越接近 true posterior，ELBO 越紧。

所以叫：

> **lower bound**

不是说这个 bound 永远很松。

如果 posterior approximation 很好，它可以非常接近真正的 log-likelihood。

---

## 40. 两个 KL 千万不要混

VAE 学习时经常看到两个形式不同的 KL。

---

### KL 1：理论 identity 中

$$
D_{KL}
(
q_\phi(z\mid x)
\parallel
p_\theta(z\mid x)
)
$$

这是：

> approximate posterior 和 true posterior 之间的差距。

它出现在：

$$
\log p_\theta(x)
=
ELBO
+
KL
$$

但因为 true posterior 难算，

我们通常不能直接计算它。

---

### KL 2：ELBO 中真正可计算的 regularizer

$$
D_{KL}
(
q_\phi(z\mid x)
\parallel
p(z)
)
$$

这是：

> approximate posterior 和 prior 之间的差距。

它出现在：

$$
ELBO
=
\mathbb E_q[\log p_\theta(x\mid z)]
-
D_{KL}(q_\phi(z\mid x)\parallel p(z))
$$

对于 Gaussian case 通常可以解析计算。

这两个 KL：

> **不是同一个东西。**

这是 VAE 学习中非常容易混淆的地方。

---

## 41. 为什么 ELBO 里的 KL 是 q(z|x) 对 prior，而不是 true posterior？

因为我们把：

$$
p_\theta(x,z)
$$

分解成：

$$
p_\theta(x\mid z)p(z)
$$

以后，

ELBO：

$$
\mathbb E_q[
\log p_\theta(x,z)
-
\log q(z\mid x)
]
$$

自然变成：

$$
\mathbb E_q[
\log p_\theta(x\mid z)
]
-
D_{KL}(q(z\mid x)\parallel p(z))
$$

所以：

> prior KL 不是为了“人工让 latent 好看”额外塞进去的。

它是 ELBO 数学展开后自然出现的。

---

## 42. 为什么 Reconstruction 和 KL 会产生 Trade-off？

如果只追求 reconstruction：

> encoder 希望给不同输入提供非常有区分度的 latent information。

如果只追求 KL：

最简单的方法是让所有输入都产生：

$$
q_\phi(z\mid x)
=
p(z)
$$

这样 KL：

$$
=0
$$

但此时：

$$
z
$$

几乎不再携带关于 $x$ 的信息。

decoder 可能无法重建。

所以：

```text
Reconstruction
要求 latent 有信息

           ↕

KL
限制 latent 不要任意偏离 prior
```

VAE 正是在这两个目标之间平衡。

---

## 43. 这也解释了 Posterior Collapse

如果 decoder 非常强，

它可能几乎不需要 $z$ 就能很好地建模 $x$。

这时优化可能走向：

$$
q_\phi(z\mid x)
\approx
p(z)
$$

于是：

$$
D_{KL}\approx0
$$

同时：

$$
z
$$

几乎不包含 $x$ 的有效信息。

这叫：

> **Posterior Collapse**

也就是：

> 模型“有 latent variable”，但 decoder 基本把它忽略了。

所以不能看到网络里存在：

$$
z
$$

就自动认为模型一定学到了有意义的 latent representation。

---

## 44. 一个最小 PyTorch 结构

下面不是 production implementation。

只是为了把数学和代码对应起来。

```python
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        z = mu + std * eps

        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        x_hat = self.decode(z)

        return x_hat, mu, logvar
```

核心数据流：

```text
x
↓
h
↓
μ, logσ²
↓
σ = exp(0.5 logσ²)
↓
ε ~ N(0,I)
↓
z = μ + σ ε
↓
decoder
↓
x̂
```

---

## 45. 一个最小 Gaussian-style loss

如果把 reconstruction 简化成 squared error：

```python
def vae_loss(x_hat, x, mu, logvar):
    recon = F.mse_loss(
        x_hat,
        x,
        reduction="sum",
    )

    kl = -0.5 * torch.sum(
        1
        + logvar
        - mu.pow(2)
        - logvar.exp()
    )

    return recon + kl
```

但再次强调：

> **这里用 MSE 是基于特定 Gaussian reconstruction 假设的简化实现。**

数学上最一般的 VAE objective 仍然是：

$$
-\mathbb E_q[
\log p_\theta(x\mid z)
]
+
D_{KL}
(
q_\phi(z\mid x)
\parallel
p(z)
)
$$

---

## 46. 常见误解一：VAE 就是普通 Autoencoder 加噪声

**不准确。**

随机采样只是实现的一部分。

VAE 的核心来自：

> latent-variable probabilistic model 中的 variational inference。

真正的理论主线是：

```text
intractable posterior
↓
approximate posterior qφ(z | x)
↓
ELBO
↓
reparameterized stochastic gradient
```

---

## 47. 常见误解二：Encoder 的目标就是让 μ→0、variance→1

**错误。**

KL 对：

$$
q_\phi(z\mid x)
$$

确实施加向：

$$
\mathcal N(0,I)
$$

靠近的压力。

但 reconstruction 同时要求 $z$ 保留关于 $x$ 的信息。

所以 encoder 最终学习的是这两者的折中。

---

## 48. 常见误解三：KL 是人为加的 regularization trick

**不完整。**

它确实具有 regularization 作用。

但它不是凭经验随手添加的。

它直接来自 ELBO：

$$
\mathcal L
=
\mathbb E_q[\log p_\theta(x\mid z)]
-
D_{KL}(q_\phi(z\mid x)\parallel p(z))
$$

---

## 49. 常见误解四：Reconstruction Loss 就是 MSE

**错误。**

正确对象是：

$$
-\log p_\theta(x\mid z)
$$

MSE 只是某些 Gaussian likelihood 假设下对应的形式。

---

## 50. 常见误解五：Encoder 输出 z

严格来说：

> **经典 Gaussian VAE encoder 输出的是 approximate posterior 的参数。**

例如：

$$
\mu(x),\log\sigma^2(x)
$$

然后：

$$
z
=
\mu+\sigma\odot\epsilon
$$

得到 sample。

---

## 51. 常见误解六：采样意味着无法反向传播

如果直接把 sampling node 当成参数相关随机操作，普通 pathwise backprop 确实有困难。

但 reparameterization：

$$
z
=
\mu_\phi(x)
+
\sigma_\phi(x)\odot\epsilon
$$

把随机性转移到与 $\phi$ 无关的：

$$
\epsilon
$$

于是对：

$$
\mu,\sigma
$$

可以正常求梯度。

---

## 52. 常见误解七：训练完成后生成数据还要先输入 x

**错误。**

真正 generation：

$$
z\sim p(z)
$$

然后：

$$
x\sim p_\theta(x\mid z)
$$

不需要 encoder。

---

## 53. 常见误解八：q(z|x) 就是真实 posterior

**错误。**

它是：

> **approximate posterior**

真实 posterior：

$$
p_\theta(z\mid x)
$$

通常正是因为难算，我们才引入：

$$
q_\phi(z\mid x)
$$

---

## 54. 常见误解九：ELBO = reconstruction + KL

符号上要小心。

ELBO 是要：

$$
\boxed{
\text{maximize}
}
$$

的：

$$
\mathcal L
=
\text{expected log-likelihood}
-
KL
$$

而代码里的 VAE loss 通常是要：

$$
\boxed{
\text{minimize}
}
$$

的 negative ELBO：

$$
-\mathcal L
=
\text{negative expected log-likelihood}
+
KL
$$

所以：

> “reconstruction + KL”

说的是 **loss / negative ELBO**，不是 ELBO 本身。

---

## 55. 用四条公式记住 VAE

如果最后只记住四条公式，应该是下面这些。

---

### 1. Generative Model

$$
\boxed{
p_\theta(x,z)
=
p(z)p_\theta(x\mid z)
}
$$

---

### 2. Approximate Posterior

$$
\boxed{
q_\phi(z\mid x)
\approx
p_\theta(z\mid x)
}
$$

---

### 3. ELBO

$$
\boxed{
\mathcal L
=
\mathbb E_{q_\phi(z\mid x)}
[
\log p_\theta(x\mid z)
]
-
D_{KL}
(
q_\phi(z\mid x)
\parallel
p(z)
)
}
$$

---

### 4. Gaussian Reparameterization

$$
\boxed{
z
=
\mu_\phi(x)
+
\sigma_\phi(x)\odot\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I)
}
$$

几乎整个经典 VAE 都可以围绕这四条式子展开。

---

## 56. 一句话重新理解 VAE

> **VAE 的本质不是“会随机采样的 Autoencoder”，而是用一个可学习的 approximate posterior $q_\phi(z\mid x)$ 去解决复杂 latent-variable generative model 中真实 posterior 难以计算的问题，并通过最大化 ELBO 同时学习 inference model 和 generative model。**

Encoder：

$$
q_\phi(z\mid x)
$$

回答：

> “看到这个 $x$，哪些 $z$ 可能解释它？”

Decoder：

$$
p_\theta(x\mid z)
$$

回答：

> “给定这个 $z$，什么样的 $x$ 可能被生成？”

ELBO 把二者连接起来。

Reparameterization 则让整个随机模型能够使用标准 backpropagation 高效训练。

---

## 57. 这和 ACT 有什么关系？

ACT 使用的并不是普通 VAE，而是：

> **Conditional Variational Autoencoder（CVAE）**

但核心思想完全建立在 VAE 之上。

ACT 训练阶段同样会：

```text
输入 demonstration information
↓
encoder
↓
μ, logσ²
↓
sample z
↓
policy / decoder
↓
predict action chunk
```

同时加入：

$$
D_{KL}
$$

让 approximate posterior 受到 prior 的约束。

如果不先理解这一篇中的：

- prior；
- posterior；
- approximate posterior；
- $\mu,\sigma^2$；
- KL；
- reparameterization；

那么 ACT 里的：

$$
z
$$

很容易被误解成：

> “随便采样出来的一个风格向量。”

下一步我们会先把条件 $c$ 加进 VAE：

> [CVAE：条件信息到底改变了什么？](./cvae.md)

然后再回到：

> [CVAE in ACT](../robot-learning/act/cvae-in-act.md)

看 ACT 原论文到底怎样构造它自己的 encoder 和 decoder。

---

### Primary Source

Diederik P. Kingma, Max Welling.  
**Auto-Encoding Variational Bayes.**  
arXiv:1312.6114, first submitted 2013; later published at ICLR 2014.

- Paper: https://arxiv.org/abs/1312.6114
- HTML: https://arxiv.org/html/1312.6114

本文主要依据原论文：

- Section 1 — Introduction
- Section 2.1 — Problem Scenario
- Section 2.2 — The Variational Bound
- Section 2.3 — The SGVB Estimator and AEVB Algorithm
- Section 2.4 — The Reparameterization Trick
- Section 3 — Example: Variational Auto-Encoder
- Appendix B — Gaussian KL
- Appendix C — Probabilistic Encoder / Decoder

---

### 本文知识连接

#### 前置知识

- [Latent Variable](./latent-variable.md)
- Probability Distribution
- Conditional Probability
- Bayes' Rule
- Marginalization

#### 数学核心

- Expectation
- [Normal Distribution](../mathematics/normal-distribution.md)
- Standard Normal Distribution
- Mean
- Variance
- [KL Divergence](../mathematics/kl-divergence.md)

#### VAE 核心

- [Reparameterization Trick](./reparameterization-trick.md)
- ELBO
- [CVAE](./cvae.md)

#### Robot Learning

- [ACT](../robot-learning/act/act-what-problem-does-it-solve.md)
- [CVAE in ACT](../robot-learning/act/cvae-in-act.md)
