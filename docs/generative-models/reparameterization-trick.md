---
title: "Reparameterization Trick：随机采样为什么还能反向传播？"
description: "从 VAE 中对 approximate posterior 的采样问题出发，严格解释普通路径梯度为何受阻、score-function estimator 为什么不理想，以及 z = μ + σ⊙ε 如何把随机性与参数依赖分离。"
status: reviewed
pageType: concept
canonical: /generative-models/reparameterization-trick
updated: "2026-09-15"
---

# Reparameterization Trick：随机采样为什么还能反向传播？

在 [VAE](./vae.md) 中，我们已经得到一个 approximate posterior：

\[
q_\phi(z\mid x)
\]

经典 Gaussian VAE 通常写成：

\[
q_\phi(z\mid x)
=
\mathcal N
\left(
\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))
\right)
\]

训练时，我们需要从这个分布里采样：

\[
z\sim q_\phi(z\mid x)
\]

再把 \(z\) 输入 decoder：

\[
z
\longrightarrow
p_\theta(x\mid z)
\]

然后根据 reconstruction / likelihood term 更新模型。

问题来了：

> **\(z\) 是随机采样出来的，梯度怎么从 decoder 穿过这个随机采样，再回到 encoder？**

这就是 **Reparameterization Trick（重参数化技巧）** 要解决的问题。

经典 Gaussian VAE 中，它最终变成一句非常有名的公式：

\[
\boxed{
z
=
\mu+\sigma\odot\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I)
}
\]

但如果只是背下这条公式，我们其实仍然不知道：

- 为什么直接采样会有问题？
- “随机采样不可导”这句话到底准确不准确？
- \(\epsilon\) 和 \(z\) 有什么区别？
- 为什么换一种写法以后分布没有变化？
- 为什么梯度突然就能传播了？
- 随机性是不是被消除了？
- 这个技巧是不是只能用于 Gaussian？

这一篇只解决这些问题。

---

# 1. 先明确 VAE 中真正需要求的梯度

VAE 的 ELBO 包含：

\[
\mathcal L(\theta,\phi;x)
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
\]

对于经典 Gaussian VAE，KL 项通常有解析解。

真正需要通过 sampling 估计的主要是：

\[
\mathbb E_{q_\phi(z\mid x)}
[
\log p_\theta(x\mid z)
]
\]

为了简化符号，把：

\[
f_\theta(z)
=
\log p_\theta(x\mid z)
\]

于是问题变成：

\[
\boxed{
\mathbb E_{z\sim q_\phi(z\mid x)}
[
f_\theta(z)
]
}
\]

训练时，我们不仅要对 decoder 参数：

\[
\theta
\]

求梯度。

还要对 encoder 参数：

\[
\phi
\]

求梯度。

也就是：

\[
\nabla_\phi
\mathbb E_{q_\phi(z\mid x)}
[
f_\theta(z)
]
\]

麻烦就出现在这里。

---

# 2. 如果 z 不是随机的，一切都很简单

先想一个完全 deterministic 的模型。

假设：

\[
z=f_\phi(x)
\]

然后：

\[
\hat x=g_\theta(z)
\]

loss：

\[
L(\hat x,x)
\]

计算图：

```text
x
↓
fφ
↓
z
↓
gθ
↓
x̂
↓
loss
```

因为：

\[
z=f_\phi(x)
\]

是一个普通可微函数，

chain rule 可以直接写：

\[
\frac{\partial L}{\partial\phi}
=
\frac{\partial L}{\partial z}
\frac{\partial z}{\partial\phi}
\]

所以 reconstruction 的梯度可以从 decoder 一路传回 encoder。

普通 Autoencoder 大致就是这种情况。

---

# 3. VAE 不一样：z 是“从分布里抽出来的”

VAE 中不是：

\[
z=f_\phi(x)
\]

而是：

\[
\boxed{
z\sim q_\phi(z\mid x)
}
\]

例如：

\[
z
\sim
\mathcal N
(
\mu_\phi(x),
\sigma_\phi^2(x)
)
\]

计算图表面上变成：

```text
x
↓
Encoder φ
↓
μ, σ
↓
Random Sample
↓
z
↓
Decoder θ
↓
loss
```

现在如果我们把：

> `Random Sample`

当作一个黑盒操作，

就不像之前那样有一个明确的 deterministic equation：

\[
z=f(\mu,\sigma)
\]

供普通 chain rule 直接穿过去。

这就是问题。

---

# 4. 但“随机采样不可导”其实说得太粗糙

很多教程会说：

> “Sampling 不可导，所以不能反向传播。”

这是一个方便初学者进入问题的说法。

但严格来说并不准确。

真正的问题是：

> **我们需要计算一个分布依赖参数 \(\phi\) 的 expectation 对 \(\phi\) 的梯度，而直接从这个参数化分布采样时，普通 pathwise backpropagation 没有显式的 deterministic sample path 可以沿着传播。**

事实上，即使不使用 reparameterization，也存在其他梯度估计方法。

VAE 原论文就写出了一个所谓的 naïve Monte Carlo gradient estimator，也就是常见的：

> **Score-Function Estimator**

或：

> **Likelihood-Ratio Estimator**

所以：

\[
\boxed{
\text{不是“数学上完全没有梯度”}
}
\]

而是：

> **直接处理这种梯度通常不够方便，而且某些 estimator 的方差很高。**

Kingma & Welling 明确指出，他们讨论的 naïve estimator 具有 very high variance，因此不适合这里的训练。  

---

# 5. 先看看不用 Reparameterization 能怎么办

考虑：

\[
J(\phi)
=
\mathbb E_{q_\phi(z)}
[
f(z)
]
\]

写成积分：

\[
J(\phi)
=
\int
q_\phi(z)f(z)\,dz
\]

对 \(\phi\) 求导：

\[
\nabla_\phi J
=
\int
\nabla_\phi q_\phi(z)
f(z)\,dz
\]

利用：

\[
\nabla_\phi q_\phi(z)
=
q_\phi(z)
\nabla_\phi\log q_\phi(z)
\]

得到：

\[
\nabla_\phi J
=
\int
q_\phi(z)
f(z)
\nabla_\phi\log q_\phi(z)
\,dz
\]

也就是：

\[
\boxed{
\nabla_\phi
\mathbb E_{q_\phi(z)}
[f(z)]
=
\mathbb E_{q_\phi(z)}
[
f(z)\nabla_\phi\log q_\phi(z)
]
}
\]

这就是 score-function estimator 的基础。

用 Monte Carlo 可以估计：

\[
\nabla_\phi J
\approx
\frac1L
\sum_{l=1}^{L}
f(z^{(l)})
\nabla_\phi
\log q_\phi(z^{(l)})
\]

其中：

\[
z^{(l)}
\sim q_\phi(z)
\]

所以：

> **没有 reparameterization 也不是完全不能训练。**

问题是这个 estimator 往往：

> **variance 很高。**

VAE 原论文正是因此寻找另一种更适合连续 latent variable 的方法。

---

# 6. 我们真正想要的是“像普通神经网络一样求梯度”

理想情况是：

\[
z
=
g_\phi(\text{something})
\]

这样：

\[
f(z)
=
f(g_\phi(\text{something}))
\]

然后直接：

\[
\nabla_\phi f(g_\phi(\cdot))
\]

使用普通 chain rule。

也就是说，我们希望把：

```text
参数 φ
↓
一个随机抽样黑盒
↓
z
```

改写成：

```text
与 φ 无关的随机量
        +
由 φ 控制的可微确定性变换
        ↓
        z
```

这就是 Reparameterization Trick 的核心思想。

---

# 7. Reparameterization 到底“重新参数化”了什么？

原来我们写：

\[
\boxed{
z\sim q_\phi(z\mid x)
}
\]

随机变量 \(z\) 的分布本身依赖：

\[
\phi
\]

现在我们寻找一个辅助随机变量：

\[
\epsilon
\]

它来自一个**不依赖 \(\phi\)** 的固定分布：

\[
\epsilon\sim p(\epsilon)
\]

然后定义一个可微 deterministic transformation：

\[
\boxed{
z
=
g_\phi(\epsilon,x)
}
\]

要求：

> 当 \(\epsilon\sim p(\epsilon)\) 时，通过 \(g_\phi\) 得到的 \(z\) 恰好服从原来的 \(q_\phi(z\mid x)\)。

于是原来的：

\[
z\sim q_\phi(z\mid x)
\]

被改写成：

\[
\epsilon\sim p(\epsilon)
\]

\[
z=g_\phi(\epsilon,x)
\]

这就是：

> **Reparameterization**

---

# 8. Gaussian 情况最简单

经典 VAE 假设：

\[
q_\phi(z\mid x)
=
\mathcal N
(
\mu_\phi(x),
\sigma_\phi^2(x)
)
\]

先只看一维。

原本：

\[
z
\sim
\mathcal N(\mu,\sigma^2)
\]

我们可以改成：

\[
\epsilon
\sim
\mathcal N(0,1)
\]

然后：

\[
\boxed{
z=\mu+\sigma\epsilon
}
\]

这就是 Gaussian reparameterization。

多维 diagonal Gaussian 则是：

\[
\boxed{
\mathbf z
=
\boldsymbol\mu
+
\boldsymbol\sigma
\odot
\boldsymbol\epsilon
}
\]

其中：

\[
\boldsymbol\epsilon
\sim
\mathcal N(\mathbf 0,I)
\]

而：

\[
\odot
\]

表示逐元素乘法。

---

# 9. 为什么 μ + σε 还是 N(μ, σ²)？

这是理解 reparameterization 的关键。

先有：

\[
\epsilon\sim\mathcal N(0,1)
\]

因此：

\[
\mathbb E[\epsilon]=0
\]

以及：

\[
\operatorname{Var}(\epsilon)=1
\]

定义：

\[
z=\mu+\sigma\epsilon
\]

那么：

## 均值

\[
\mathbb E[z]
=
\mathbb E[\mu+\sigma\epsilon]
\]

\[
=
\mu+\sigma\mathbb E[\epsilon]
\]

\[
=
\mu
\]

---

## 方差

因为加常数不改变方差：

\[
\operatorname{Var}(z)
=
\operatorname{Var}(\sigma\epsilon)
\]

根据：

\[
\operatorname{Var}(aX)
=
a^2\operatorname{Var}(X)
\]

得到：

\[
\operatorname{Var}(z)
=
\sigma^2
\]

所以：

\[
\boxed{
z
\sim
\mathcal N(\mu,\sigma^2)
}
\]

也就是说：

> 我们只是换了一种产生同样 distribution sample 的方式。

没有改变目标分布。

---

# 10. 用更直觉的方式理解：先标准化，再缩放和平移

标准正态：

\[
\epsilon\sim\mathcal N(0,1)
\]

可以把它理解为一团：

> 中心在 0、标准差为 1 的随机样本。

先乘：

\[
\sigma
\]

得到：

\[
\sigma\epsilon
\]

会改变这团数据的宽度。

标准差变成：

\[
\sigma
\]

然后加：

\[
\mu
\]

整个分布整体移动到：

\[
\mu
\]

附近。

所以：

```text
ε ~ N(0,1)
     │
     │ × σ
     ▼
N(0,σ²)
     │
     │ + μ
     ▼
N(μ,σ²)
```

如果不熟悉这里的均值、方差和标准正态，可以阅读：

- Mean
- Variance
- [Normal Distribution](../mathematics/normal-distribution.md)
- Standard Normal Distribution

---

# 11. 随机性到底去哪了？

它没有消失。

原来随机性写在：

\[
z
\sim
\mathcal N(\mu,\sigma^2)
\]

现在随机性写在：

\[
\epsilon\sim\mathcal N(0,1)
\]

然后：

\[
z=\mu+\sigma\epsilon
\]

所以变化不是：

```text
随机
→
不随机
```

而是：

```text
随机性和模型参数纠缠在同一个 sampling operation 里
```

变成：

```text
参数无关的随机源 ε
+
参数控制的 deterministic transform
```

所以更准确地说：

> **Reparameterization 把随机性从“参数化 sampling distribution”中抽离出来，放到一个固定 auxiliary noise distribution 中。**

---

# 12. 这为什么能帮助反向传播？

现在：

\[
\mu=\mu_\phi(x)
\]

\[
\sigma=\sigma_\phi(x)
\]

以及：

\[
\epsilon\sim\mathcal N(0,I)
\]

对于一次具体 forward pass，

一旦：

\[
\epsilon
\]

被采样出来，它在这一次计算中就是一个普通数值 tensor。

于是：

\[
z
=
\mu_\phi(x)
+
\sigma_\phi(x)\odot\epsilon
\]

就是关于：

\[
\phi
\]

的可微函数。

所以 reconstruction loss：

\[
L(z)
\]

对 \(\phi\) 的梯度可以通过 chain rule：

\[
\frac{\partial L}{\partial\phi}
=
\frac{\partial L}{\partial z}
\frac{\partial z}{\partial\phi}
\]

继续展开：

\[
\frac{\partial z}{\partial\phi}
=
\frac{\partial\mu_\phi(x)}{\partial\phi}
+
\epsilon
\frac{\partial\sigma_\phi(x)}{\partial\phi}
\]

于是 decoder 的 reconstruction signal 能够传回：

- \(\mu\) head；
- \(\sigma\) head；
- encoder backbone。

---

# 13. 用计算图看最清楚

## 直接 sampling 的抽象写法

```text
x
↓
Encoder φ
↓
μ, σ
↓
sample z ~ N(μ, σ²)
↓
Decoder
↓
Loss
```

如果把 sampling 当作黑盒，普通 autodiff 看不到：

\[
z
\]

是如何以 deterministic computational path 依赖：

\[
\mu,\sigma
\]

的。

---

## Reparameterization 后

```text
x
↓
Encoder φ
↓
μ ───────────────┐
                  │
σ ────────────┐   │
              │   │
ε ~ N(0,I) ───×───+──→ z
                        ↓
                     Decoder
                        ↓
                       Loss
```

这里：

\[
z=\mu+\sigma\epsilon
\]

是显式的可微运算。

所以 autodiff 可以直接知道：

\[
\frac{\partial z}{\partial\mu}=1
\]

以及：

\[
\frac{\partial z}{\partial\sigma}=\epsilon
\]

这就是整个技巧在工程上的核心。

---

# 14. 一个具体数字例子

假设 encoder 对某个输入 \(x\) 输出：

\[
\mu=2
\]

\[
\sigma=0.5
\]

这次 forward 随机采样：

\[
\epsilon=-0.8
\]

那么：

\[
z
=
2+0.5(-0.8)
\]

\[
=1.6
\]

decoder 使用：

\[
z=1.6
\]

得到 reconstruction loss。

假设反向传播得到：

\[
\frac{\partial L}{\partial z}=3
\]

那么：

\[
\frac{\partial z}{\partial\mu}=1
\]

所以：

\[
\frac{\partial L}{\partial\mu}
=
3\times1
=
3
\]

而：

\[
\frac{\partial z}{\partial\sigma}
=
\epsilon
=
-0.8
\]

所以：

\[
\frac{\partial L}{\partial\sigma}
=
3\times(-0.8)
=
-2.4
\]

现在：

\[
\mu
\]

和：

\[
\sigma
\]

都得到了明确梯度。

然后梯度继续传进 encoder。

这就是“采样以后还能反向传播”的具体含义。

---

# 15. 但 ε 自己需要梯度吗？

通常不需要。

\[
\epsilon
\sim
\mathcal N(0,I)
\]

只是一个 auxiliary random variable。

它不是需要学习的模型参数。

我们并不优化：

\[
\epsilon
\]

我们优化的是：

\[
\phi
\]

也就是生成：

\[
\mu_\phi(x)
\]

和：

\[
\sigma_\phi(x)
\]

的 encoder parameters。

所以计算图可以把：

\[
\epsilon
\]

视为：

> **随机输入常量**

类似每个 forward pass 临时给网络的一组随机数字。

---

# 16. 这其实是在做 Pathwise Gradient

Reparameterization 后：

\[
J(\phi)
=
\mathbb E_{\epsilon\sim p(\epsilon)}
[
f(g_\phi(\epsilon,x))
]
\]

注意：

\[
p(\epsilon)
\]

不依赖：

\[
\phi
\]

所以可以把参数梯度作用到 expectation 内部的 deterministic transformation：

\[
\boxed{
\nabla_\phi J(\phi)
=
\mathbb E_{p(\epsilon)}
[
\nabla_\phi
f(g_\phi(\epsilon,x))
]
}
\]

然后 Monte Carlo 估计：

\[
\nabla_\phi J
\approx
\frac1L
\sum_{l=1}^{L}
\nabla_\phi
f(g_\phi(\epsilon^{(l)},x))
\]

其中：

\[
\epsilon^{(l)}
\sim
p(\epsilon)
\]

这种梯度估计通常叫：

> **Pathwise Derivative Estimator**

或者：

> **Reparameterization Gradient**

因为梯度是沿着具体 sampled path：

\[
\epsilon
\rightarrow
z
\rightarrow
f(z)
\]

传播的。

---

# 17. 和 Score-Function Estimator 到底有什么不同？

两者都可以处理：

\[
\nabla_\phi
\mathbb E_{q_\phi(z)}
[f(z)]
\]

但思路不同。

---

## Score-Function

使用：

\[
\boxed{
\nabla_\phi
\mathbb E_q[f(z)]
=
\mathbb E_q
[
f(z)\nabla_\phi\log q_\phi(z)
]
}
\]

它不要求：

\[
f(z)
\]

对 \(z\) 可微。

这个性质很强。

但 estimator 往往 variance 较高。

---

## Reparameterization / Pathwise

把：

\[
z\sim q_\phi(z)
\]

改写为：

\[
z=g_\phi(\epsilon)
\]

然后：

\[
\boxed{
\nabla_\phi
\mathbb E_\epsilon
[
f(g_\phi(\epsilon))
]
}
\]

直接沿：

\[
g_\phi
\]

的计算路径求导。

它要求：

> \(g_\phi\) 和后续 computation 具有合适的 differentiability。

对于 VAE 的 continuous latent variables，这通常非常方便。

VAE 原论文正是利用这种方式构造低方差、可用标准 stochastic gradient 方法优化的 estimator。citeturn453800view0

---

# 18. Reparameterization 改写的其实是 Expectation

原来的 expectation：

\[
\mathbb E_{q_\phi(z\mid x)}
[
f(z)
]
\]

可以写成：

\[
\int
q_\phi(z\mid x)
f(z)\,dz
\]

重参数化后：

\[
z
=
g_\phi(\epsilon,x)
\]

其中：

\[
\epsilon\sim p(\epsilon)
\]

于是同一个 expectation 可以写成：

\[
\boxed{
\mathbb E_{p(\epsilon)}
[
f(g_\phi(\epsilon,x))
]
}
\]

Gaussian 情况：

\[
\boxed{
\mathbb E_{
z\sim\mathcal N(\mu,\sigma^2)
}
[
f(z)
]
=
\mathbb E_{
\epsilon\sim\mathcal N(0,1)
}
[
f(\mu+\sigma\epsilon)
]
}
\]

这条式子比：

\[
z=\mu+\sigma\epsilon
\]

本身更能说明 reparameterization 的理论意义。

它真正改变的是：

> **我们用哪个随机变量来表达同一个 expectation。**

---

# 19. Monte Carlo 在这里做什么？

即使 reparameterize 以后：

\[
\mathbb E_{\epsilon}
[
f(\mu+\sigma\epsilon)
]
\]

通常仍然没有直接把 expectation 精确算完。

实际训练时还是用 sample。

例如采：

\[
L
\]

个：

\[
\epsilon^{(1)},\ldots,\epsilon^{(L)}
\]

然后：

\[
\mathbb E_\epsilon[f(\mu+\sigma\epsilon)]
\approx
\frac1L
\sum_{l=1}^{L}
f(
\mu+\sigma\epsilon^{(l)}
)
\]

VAE 原论文的实验中指出，当 minibatch 足够大时，每个 datapoint 使用：

\[
L=1
\]

个 latent sample 就可以工作得很好。citeturn453800view0

所以常见 VAE 代码每个 forward：

```python
eps = torch.randn_like(std)
z = mu + std * eps
```

只采一次。

---

# 20. 一次 sample 怎么能代表整个 distribution？

它不能精确代表。

它只是：

> Monte Carlo estimator 的一个随机样本。

单次估计有随机误差。

但训练过程中：

- 有很多 datapoints；
- 有很多 minibatches；
- 有很多 optimization steps；
- 每次都会产生新的 \(\epsilon\)。

所以整体上形成 stochastic estimation。

这与 SGD 本身也有类似精神：

> 每一次更新都不是看完整数据集，而是用 minibatch 近似总体梯度。

因此：

\[
L=1
\]

并不意味着：

> “一个 \(z\) 就等于整个 posterior。”

而是：

> **这一轮用一个 sample 来估计 expectation。**

---

# 21. 为什么不是直接用 μ，不采样？

这是一个非常自然的问题。

既然 encoder 已经输出：

\[
\mu
\]

为什么不直接：

\[
z=\mu
\]

这样：

- 完全可微；
- 没有随机性；
- reconstruction 也更稳定。

问题是：

> 这样训练的就不再是原本那个 stochastic latent-variable objective。

VAE 希望优化的是：

\[
\mathbb E_{q_\phi(z\mid x)}
[
\log p_\theta(x\mid z)
]
\]

这里 expectation 是对整个：

\[
q_\phi(z\mid x)
\]

而言。

如果永远只使用：

\[
z=\mu
\]

实际上只评估 distribution 中一个特殊点。

variance：

\[
\sigma^2
\]

对 reconstruction path 的意义会被严重削弱。

也无法正确训练 decoder 在 posterior distribution 中的不同 latent samples 上工作。

所以：

> **训练时 sampling 是概率目标的一部分，不是可以随意删除的噪声。**

---

# 22. 为什么 z=μ+σε 中 σ 很重要？

如果：

\[
\sigma\rightarrow0
\]

那么：

\[
z
=
\mu+\sigma\epsilon
\]

会变成：

\[
z\approx\mu
\]

随机性很小。

posterior 非常集中。

---

如果：

\[
\sigma
\]

较大，

同一个 \(x\) 可以产生分散得更开的：

\[
z
\]

samples。

例如：

\[
\mu=2
\]

如果：

\[
\sigma=0.1
\]

samples 大多在：

\[
2
\]

附近很窄的范围。

如果：

\[
\sigma=2
\]

samples 则分散得非常广。

因此：

\[
\sigma
\]

不是“为了让公式完整才有的”。

它真正控制：

> approximate posterior 的 uncertainty / spread。

---

# 23. 为什么通常输出 log variance？

VAE encoder 常见：

```python
mu = mu_head(h)
logvar = logvar_head(h)
```

然后：

```python
std = torch.exp(0.5 * logvar)
```

原因是：

\[
\sigma^2>0
\]

但 neural network 普通 linear output 可以是任意实数。

所以让网络输出：

\[
\log\sigma^2
\in(-\infty,+\infty)
\]

再恢复：

\[
\sigma
=
\exp
\left(
\frac12\log\sigma^2
\right)
\]

自然保证：

\[
\sigma>0
\]

然后：

\[
z
=
\mu
+
\exp
\left(
\frac12\log\sigma^2
\right)
\odot\epsilon
\]

---

# 24. PyTorch 中最核心的三行

```python
std = torch.exp(0.5 * logvar)

eps = torch.randn_like(std)

z = mu + std * eps
```

逐行对应数学：

### 第一行

\[
\sigma
=
\exp
\left(
\frac12\log\sigma^2
\right)
\]

---

### 第二行

\[
\epsilon
\sim
\mathcal N(0,I)
\]

---

### 第三行

\[
z
=
\mu+\sigma\odot\epsilon
\]

这三行就是 Gaussian reparameterization。

---

# 25. Autograd 到底会对谁求梯度？

代码：

```python
eps = torch.randn_like(std)
z = mu + std * eps
```

通常：

\[
\epsilon
\]

不需要 gradient。

但是：

\[
\mu
\]

和：

\[
std
\]

来自 encoder，所以带有 computation graph。

因此：

```text
loss
↓ backward
z
├──→ mu
└──→ std
      ↓
    logvar
      ↓
 encoder parameters
```

于是 reconstruction loss 可以训练：

- `mu_head`
- `logvar_head`
- encoder 之前的所有层。

---

# 26. 一个最小代码例子

```python
import torch

mu = torch.tensor(
    [2.0],
    requires_grad=True,
)

logvar = torch.tensor(
    [-1.0],
    requires_grad=True,
)

std = torch.exp(0.5 * logvar)

eps = torch.randn_like(std)

z = mu + std * eps

loss = z.pow(2).sum()

loss.backward()

print(mu.grad)
print(logvar.grad)
```

尽管：

```python
eps
```

是随机生成的，

`mu.grad` 和 `logvar.grad` 仍然存在。

因为：

\[
z
\]

对：

\[
\mu,\log\sigma^2
\]

有一条显式可微路径。

---

# 27. Reparameterization 让结果变 deterministic 了吗？

**没有。**

每次 forward：

\[
\epsilon
\]

仍可能不同。

例如：

第一次：

\[
\epsilon=0.2
\]

得到：

\[
z_1=\mu+0.2\sigma
\]

第二次：

\[
\epsilon=-1.1
\]

得到：

\[
z_2=\mu-1.1\sigma
\]

所以：

\[
z_1\neq z_2
\]

模型仍然是 stochastic 的。

Reparameterization 只是让：

> 对于一次已经抽出的 \(\epsilon\)，从参数到 \(z\) 的路径是 deterministic 且可微的。

这是一个非常重要的表述。

---

# 28. “固定 ε 后可微”是什么意思？

训练一次 forward 时：

\[
\epsilon
\]

被 sample 成一个具体 tensor。

例如：

\[
\epsilon=-0.8
\]

这一轮计算里：

\[
z=\mu-0.8\sigma
\]

现在它就是一个普通函数。

反向传播时，不需要问：

> “为什么随机数是 -0.8？”

只需要问：

> “如果 \(\mu\) 稍微变化，这个 \(z\) 会怎么变？”

以及：

> “如果 \(\sigma\) 稍微变化，这个 \(z\) 会怎么变？”

答案分别是：

\[
\frac{\partial z}{\partial\mu}=1
\]

\[
\frac{\partial z}{\partial\sigma}=\epsilon
\]

因此可以优化。

下一轮再重新 sample 新的：

\[
\epsilon
\]

即可。

---

# 29. 一个很好的直觉：把“随机骰子”移到参数外面

原始写法可以直觉看作：

```text
μ, σ
  ↓
一个规则会随着 μ,σ 改变的随机盒子
  ↓
z
```

Reparameterization 改成：

```text
固定分布的随机骰子 ε ~ N(0,1)
                ↓
              一个数
                ↓
μ, σ ──→ 可微公式 μ + σε
                ↓
                z
```

骰子仍然在掷。

但是：

> **骰子的分布不再由 encoder parameters 控制。**

encoder parameters 只控制：

- 平移多少；
- 缩放多少。

而平移与缩放都是标准可微运算。

---

# 30. 为什么标准正态特别方便？

Gaussian 属于：

> **location-scale family**

标准形式：

\[
\epsilon\sim\mathcal N(0,1)
\]

任意：

\[
\mathcal N(\mu,\sigma^2)
\]

都可以通过：

\[
z=\mu+\sigma\epsilon
\]

得到。

也就是：

```text
standard distribution
↓
scale
↓
shift
↓
target distribution
```

VAE 原论文指出，这种方法并不只适用于 Gaussian。

很多 location-scale distributions 都可以类似处理。citeturn453800view0

---

# 31. Reparameterization 不只适用于 Gaussian

原论文 Section 2.4 讨论了几类可以构造 differentiable transformation 的情况。

例如某些分布可以通过：

### Inverse CDF

\[
\epsilon\sim U(0,1)
\]

然后：

\[
z=F_\phi^{-1}(\epsilon)
\]

---

### Location-Scale Family

\[
z
=
\text{location}
+
\text{scale}\cdot\epsilon
\]

Gaussian 就属于这种情况。

---

### Composition

某些随机变量可以表达成其他基础随机变量的 transformation。

所以：

> **Reparameterization Trick 是一个更一般的思想。**

Gaussian 的：

\[
z=\mu+\sigma\epsilon
\]

只是最经典、最简单的例子。

---

# 32. 那离散变量怎么办？

经典 VAE reparameterization 对 continuous latent variables 特别自然。

但如果：

\[
z
\]

是离散变量，例如：

\[
z\in\{1,2,3\}
\]

普通 sampling：

```text
choose category 1 / 2 / 3
```

不是简单的 continuous differentiable location-scale transform。

因此不能直接套：

\[
z=\mu+\sigma\epsilon
\]

这也是为什么后来出现了其他方法，例如：

- score-function estimators；
- Gumbel-Softmax / Concrete relaxation；
- 其他离散梯度估计方法。

所以不要记成：

> “所有 random sampling 都可以用 μ+σε 解决。”

准确说法是：

> **Gaussian location-scale reparameterization 是 continuous reparameterization 的经典特例。**

---

# 33. Reparameterization 是不是 VAE 独有的？

不是。

它是一种：

> **随机变量 / stochastic computation 的梯度估计技术。**

VAE 让它非常出名，是因为：

\[
q_\phi(z\mid x)
\]

中的 continuous Gaussian latent variable 非常适合用这种技巧。

但 reparameterization gradient 的思想也可以出现在：

- variational inference；
- stochastic optimization；
- Bayesian neural networks；
- continuous stochastic policies；
- probabilistic models。

所以：

> VAE 使用 Reparameterization Trick。

不是：

> VAE 发明了“随机变量可微”这个概念。

Kingma & Welling 的 AEVB 工作的重要贡献之一，是把这种重参数化用于构造实用的 stochastic variational lower-bound estimator。citeturn933259view0

---

# 34. 为什么它通常比 Score-Function Gradient 更稳定？

只做直觉理解。

Score-function estimator：

\[
f(z)
\nabla_\phi\log q_\phi(z)
\]

更多依赖：

> sample 最终得到的整体 reward / function value \(f(z)\)

来判断 distribution parameters 应该怎么变化。

而 pathwise gradient 直接使用：

\[
\frac{\partial f}{\partial z}
\]

知道：

> 如果这个 sample \(z\) 往哪个方向稍微移动，loss 会怎样变化。

也就是说，它利用了：

> **函数局部导数信息。**

因此对于 smooth continuous problems，通常能够得到更低 variance、更直接的 gradient signal。

但这里最好记成经验与估计性质：

> **通常更低 variance。**

不要理解成：

> 所有情况下 reparameterization estimator 都必然优于所有其他 estimator。

---

# 35. Reparameterization 和 KL 项是什么关系？

很多初学者会把它们混在一起。

它们解决的是不同问题。

## KL

\[
D_{KL}
(
q_\phi(z\mid x)\parallel p(z)
)
\]

回答：

> approximate posterior 应该受到什么概率分布约束？

---

## Reparameterization

\[
z
=
\mu+\sigma\epsilon
\]

回答：

> 如何从这个 parameterized distribution 采样，同时让 reconstruction path 可以高效地对 encoder 参数求梯度？

所以：

```text
KL
= objective 的一部分

Reparameterization
= gradient estimation / sampling technique
```

不是同一个东西。

---

# 36. Reparameterization 和 μ、σ 的学习是什么关系？

encoder 生成：

\[
\mu_\phi(x)
\]

和：

\[
\sigma_\phi(x)
\]

它们受到两条梯度来源影响。

---

## Reconstruction path

通过：

\[
z
=
\mu+\sigma\epsilon
\]

reconstruction loss 的梯度可以回到：

\[
\mu,\sigma
\]

再回到 encoder。

---

## KL path

KL 通常直接是：

\[
\mu,\sigma
\]

的解析函数。

例如：

\[
D_{KL}
=
\frac12
\sum_j
(
\mu_j^2+\sigma_j^2-\log\sigma_j^2-1
)
\]

它也直接对：

\[
\mu,\sigma
\]

提供梯度。

因此 encoder 的最终更新来自：

```text
Reconstruction gradient
        +
KL gradient
```

Reparameterization 的作用主要是确保：

> reconstruction gradient 能通过 stochastic latent sample 回到 encoder。

---

# 37. 如果没有 Reparameterization，KL 还能训练 Encoder 吗？

对于 Gaussian closed-form KL：

**可以。**

因为 KL 直接依赖：

\[
\mu,\sigma
\]

所以：

\[
\nabla_\phi KL
\]

可以直接计算。

真正困难的是：

\[
\mathbb E_{q_\phi(z\mid x)}
[
\log p_\theta(x\mid z)
]
\]

这一部分对 \(\phi\) 的梯度。

如果 sampling path 没有合适的 estimator，

reconstruction signal 就很难高效传回 encoder。

这个区别很重要。

---

# 38. 为什么训练中同一个 x 每次可能得到不同 z？

因为每次都会重新 sample：

\[
\epsilon
\]

例如同一个：

\[
x
\]

encoder 输出固定：

\[
\mu=1
\]

\[
\sigma=0.5
\]

第一次：

\[
\epsilon_1=0.2
\]

得到：

\[
z_1=1.1
\]

第二次：

\[
\epsilon_2=-0.8
\]

得到：

\[
z_2=0.6
\]

第三次：

\[
\epsilon_3=1.0
\]

得到：

\[
z_3=1.5
\]

它们都来自同一个：

\[
q_\phi(z\mid x)
\]

因此同一个 \(x\) 并不是固定对应一个 \(z\)。

这正是：

> **probabilistic encoder**

与 deterministic encoder 的区别。

---

# 39. 为什么这反而不会让 decoder 崩掉？

因为训练过程中 decoder 不断看到：

> 同一个 \(x\) 对应 posterior 附近的不同 \(z\) samples。

它因此被迫学习：

> posterior 附近的一片 latent region 都应该能够合理解释 / 重建这个 \(x\)。

再加上 KL 让不同：

\[
q_\phi(z\mid x)
\]

不要完全散落在 prior 空间的无关位置，

最终 latent space 才更适合：

- sampling；
- interpolation；
- generation。

注意：

> 这是直觉解释。

严格的训练目标仍然是：

\[
ELBO
\]

不能把“latent space 会很光滑”当成无条件数学保证。

---

# 40. 一个很重要的思想：随机性和可微性可以同时存在

初学深度学习时，我们很容易形成：

```text
deterministic
→ 可微

random
→ 不可微
```

这种二分法。

其实并不正确。

Reparameterization 给出的思想是：

> **随机 computation 可以被表示成：随机输入 + 可微确定性计算。**

例如：

\[
\epsilon\sim p(\epsilon)
\]

然后：

\[
z=g_\phi(\epsilon)
\]

模型整体仍然 stochastic。

但对固定 random draw：

\[
\epsilon
\]

而言，

\[
g_\phi
\]

仍然是可微函数。

这是一种非常重要的 probabilistic deep learning 思维。

---

# 41. 现在回到 ACT

ACT 使用 CVAE。

训练阶段，ACT encoder 会输出 latent distribution 的参数：

\[
\mu
\]

和：

\[
\log\sigma^2
\]

然后使用同样的 Gaussian reparameterization：

\[
\epsilon
\sim
\mathcal N(0,I)
\]

\[
\boxed{
z
=
\mu+\sigma\odot\epsilon
}
\]

接着：

\[
z
\]

进入 ACT policy / decoder，帮助预测 action chunk。

所以 ACT 中的这一步并不是：

> “为了随机而随机”。

它继承的是 VAE / CVAE 的完整概率训练机制。

---

# 42. ACT 中为什么训练时需要 sample z？

训练阶段 approximate posterior：

\[
q_\phi(z\mid \text{demonstration information})
\]

描述的是：

> 哪些 latent style values 能够解释当前这条 demonstration action sequence。

如果只永远使用：

\[
z=\mu
\]

就没有真正按照 posterior distribution 进行 stochastic variational training。

所以训练时使用：

\[
z=\mu+\sigma\epsilon
\]

是 CVAE objective 的组成部分。

而到了 inference，ACT 做了另外一个设计：

\[
z=0
\]

这不是 Reparameterization Trick 本身规定的。

而是 ACT 特定的 inference choice。

后面会单独讨论：

- [为什么 ACT 推理时令 z = 0？](../robot-learning/act/why-z-zero-at-inference.md)

---

# 43. Reparameterization 并不能解释 ACT 为什么 z=0

这是知识边界必须分清的地方。

Reparameterization 只告诉我们：

> **训练时怎样从 Gaussian approximate posterior 采样，并让梯度能传回 encoder。**

它不回答：

> 推理时应该选择哪个 \(z\)。

ACT 推理：

\[
z=0
\]

需要结合：

- CVAE；
- prior；
- KL regularization；
- ACT 的具体训练 / inference 设计；

一起解释。

所以不要把下面两件事混成一句：

```text
因为 reparameterization，
所以 inference z = 0
```

这是错误逻辑。

---

# 44. 常见误解一：Sampling 完全没有梯度

**不准确。**

更正确的是：

> 直接对参数化 sampling distribution 求 pathwise gradient 不方便；其他 estimator 仍然存在。

VAE 原论文甚至明确讨论了 naïve Monte Carlo / score-function estimator，只是指出它的 variance 太高，不适合这里。citeturn933259view0

---

# 45. 常见误解二：Reparameterization 消除了随机性

**错误。**

随机性仍然来自：

\[
\epsilon\sim\mathcal N(0,I)
\]

每次 forward 都可以得到不同的：

\[
z
\]

改变的只是随机性的表达方式。

---

# 46. 常见误解三：z 就是 ε

**错误。**

\[
\epsilon
\]

是 auxiliary standard noise。

而：

\[
z
=
\mu+\sigma\epsilon
\]

是 latent sample。

所以：

\[
\boxed{
z\neq\epsilon
}
\]

一般情况下它们的 distribution 也不同。

---

# 47. 常见误解四：μ 和 σ 也是随机采样出来的

经典 VAE 中不是。

对于给定：

\[
x
\]

encoder deterministic 地计算：

\[
\mu_\phi(x)
\]

和：

\[
\sigma_\phi(x)
\]

随机的是：

\[
\epsilon
\]

从而使：

\[
z
\]

随机。

---

# 48. 常见误解五：Reparameterization 是为了让 z 接近 N(0,I)

**错误。**

让：

\[
q_\phi(z\mid x)
\]

受到 prior：

\[
p(z)=\mathcal N(0,I)
\]

约束的是：

\[
D_{KL}
\]

Reparameterization 的任务是：

> 高效求 stochastic expectation 对参数的梯度。

两者功能不同。

---

# 49. 常见误解六：只要写成 μ+σε，模型就一定学得好

**错误。**

Reparameterization 只是 gradient estimator / sampling mechanism。

最终训练质量还依赖：

- model architecture；
- likelihood；
- prior；
- KL weighting；
- optimization；
- data；
- posterior family；
- decoder capacity。

它不是模型性能保证。

---

# 50. 常见误解七：所有分布都能直接 μ+σε

**错误。**

这种形式特别适合：

> location-scale family

尤其 Gaussian。

其他分布可能需要：

- inverse-CDF transform；
- composition；
- relaxation；
- 其他 gradient estimators。

---

# 51. 常见误解八：Reparameterization 让随机 sample 本身变得“可导”

这个说法也不够精确。

更准确的是：

> **它构造了一条对模型参数可微的 sample path。**

我们不需要对：

> “随机事件为什么发生”

求导。

我们需要的是：

> 固定一次随机 draw 后，sample value 如何随着 \(\phi\) 改变。

这才是 optimization 真正需要的信息。

---

# 52. 用两条公式记住全部内容

如果整篇只留下两条公式：

## 原始 stochastic variable

\[
\boxed{
z\sim q_\phi(z\mid x)
}
\]

---

## Reparameterized form

\[
\boxed{
z
=
g_\phi(\epsilon,x),
\qquad
\epsilon\sim p(\epsilon)
}
\]

Gaussian VAE 中：

\[
\boxed{
z
=
\mu_\phi(x)
+
\sigma_\phi(x)\odot\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I)
}
\]

真正重要的思想是：

\[
p(\epsilon)
\]

**不依赖 \(\phi\)**，

而：

\[
g_\phi
\]

是可微的。

---

# 53. 一句话重新理解

> **Reparameterization Trick 没有把随机采样变成确定性，也没有删除随机性；它只是把随机性放进一个与模型参数无关的辅助变量 \(\epsilon\)，再通过一个依赖参数且可微的确定性函数生成 \(z\)。**

这样：

\[
z
\]

仍然具有正确的 stochastic distribution，

但 reconstruction objective 对 encoder 参数：

\[
\phi
\]

可以通过：

\[
z=g_\phi(\epsilon,x)
\]

使用普通 backpropagation 高效求梯度。

对于 Gaussian VAE：

\[
\epsilon\sim\mathcal N(0,I)
\]

\[
z=\mu+\sigma\odot\epsilon
\]

就是这个思想最经典的形式。

---

# 54. 下一步

现在 VAE 链条已经有：

```text
Latent Variable
        ↓
VAE
        ↓
Reparameterization Trick
```

接下来要真正理解 ACT 的 CVAE，还缺一个关键变化：

> **如果我不只想根据 latent \(z\) 生成结果，而是希望在已知某个 condition \(c\) 的情况下生成结果，会发生什么？**

也就是从：

\[
p(x\mid z)
\]

变成：

\[
p(x\mid z,c)
\]

同时 posterior 从：

\[
q(z\mid x)
\]

变成带条件的形式。

这会引出下一篇：

> [CVAE：条件信息到底改变了什么？](./cvae.md)

之后我们就可以正式回到 ACT：

> [CVAE in ACT](../robot-learning/act/cvae-in-act.md)

---

## Primary Source

Diederik P. Kingma, Max Welling.  
**Auto-Encoding Variational Bayes.**  
arXiv:1312.6114; ICLR 2014.

- Paper: https://arxiv.org/abs/1312.6114
- HTML: https://arxiv.org/html/1312.6114

本文主要依据原论文：

- Section 1 — Introduction
- Section 2.2 — The Variational Bound
- Section 2.3 — The SGVB Estimator and AEVB Algorithm
- Section 2.4 — The Reparameterization Trick
- Section 3 — Example: Variational Auto-Encoder

原论文的通用重参数化形式为：

\[
\widetilde z
=
g_\phi(\epsilon,x),
\qquad
\epsilon\sim p(\epsilon)
\]

并在 Gaussian VAE 例子中使用：

\[
z
=
\mu+\sigma\odot\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I)
\]

---

## 本文知识连接

### 前置知识

- [Latent Variable](./latent-variable.md)
- [VAE](./vae.md)
- Expectation
- [Normal Distribution](../mathematics/normal-distribution.md)
- Standard Normal Distribution
- Mean
- Variance

### 相关数学

- Monte Carlo Estimation
- Gradient & Chain Rule
- [KL Divergence](../mathematics/kl-divergence.md)

### 下一步

- [CVAE](./cvae.md)

### Robot Learning

- [ACT](../robot-learning/act/act-what-problem-does-it-solve.md)
- [CVAE in ACT](../robot-learning/act/cvae-in-act.md)
- [为什么 ACT 推理时令 z = 0？](../robot-learning/act/why-z-zero-at-inference.md)
