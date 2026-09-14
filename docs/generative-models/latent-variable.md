---
title: "Latent Variable：模型为什么需要一个“看不见的变量”？"
description: "从概率生成模型出发，严格理解 latent variable、observed variable、prior、likelihood、posterior 与 marginalization，并为 VAE、CVAE 和 ACT 中的 latent z 建立正确直觉。"
status: reviewed
pageType: concept
canonical: /generative-models/latent-variable
updated: "2026-09-15"
---

# Latent Variable：模型为什么需要一个“看不见的变量”？

如果你正在学习 VAE、CVAE 或 ACT，很快就会遇到一个字母：

\[
z
\]

然后文章往往会告诉你：

> \(z\) 是 latent variable。

再往后：

```text
encoder
↓
μ, σ²
↓
sample z
↓
decoder
```

很多人第一次看到这里，会自然产生一个问题：

> **这个 \(z\) 到底是什么？**

常见解释包括：

- “它是压缩后的特征。”
- “它代表风格。”
- “它是隐藏信息。”
- “它把输入编码成一个向量。”
- “它是模型自己学出来的抽象语义。”

这些说法有时能帮助理解。

但如果把它们当成正式定义，就很容易在后面学 VAE、CVAE 和 ACT 时产生误解。

这一篇我们先不碰复杂网络。

我们只解决一个最基础的问题：

> **在概率生成模型里，latent variable 到底是什么？**

---

# 1. 先从“我们能看到什么”开始

假设我们有一个数据集：

\[
\mathcal D
=
\{x^{(1)},x^{(2)},\ldots,x^{(N)}\}
\]

这些 \(x\) 是我们真正能够观察到的数据。

例如：

- 一张图片的像素；
- 一段声音；
- 一个机器人的动作轨迹；
- 一个学生的考试成绩；
- 一组传感器读数。

这种变量叫：

> **Observed Variable（观测变量）**

因为数据集里真的记录了它。

例如一张 MNIST 图片：

\[
x\in\mathbb R^{28\times28}
\]

每一个像素值我们都能直接读取。

所以：

\[
x
\]

是 observed。

---

# 2. 但“观测到数据”不等于“知道数据为什么长这样”

假设我们看到两张手写数字 7。

它们都属于“7”。

但它们可能：

- 一个写得很斜；
- 一个写得很直；
- 一个笔画粗；
- 一个笔画细；
- 一个顶部横线长；
- 一个顶部横线短。

我们看到的是最终像素：

\[
x
\]

但是：

> **产生这些像素的所有内部变化因素，并没有作为标签完整记录在数据里。**

例如我们通常没有一个数据表写着：

```text
slant = 0.72
stroke_width = 0.31
writer_style = ...
```

这些潜在因素可能影响数据。

但我们没有直接观察到它们。

于是可以提出一种建模思想：

> 假设观测数据 \(x\) 背后还存在某个没有直接观测到的变量 \(z\)。

这个：

\[
z
\]

就是：

> **Latent Variable（潜变量）**

---

# 3. Latent 的意思到底是什么？

`latent` 的意思接近：

> hidden / not directly observed

也就是：

> **没有被直接观测到。**

因此在概率模型语境中，一个最核心的区别是：

```text
x
= observed variable
= 我们在数据中真的看到了


z
= latent variable
= 模型假设存在，但数据没有直接告诉我们它的值
```

注意：

> “没有观测到”不等于“现实世界里一定真的存在一个叫 \(z\) 的物理量”。

\(z\) 是：

> **模型内部用于描述数据生成过程的随机变量。**

---

# 4. Kingma & Welling 的 VAE 从这里开始

VAE 原论文《Auto-Encoding Variational Bayes》在定义问题时，直接假设数据由一个包含未观测连续随机变量 \(z\) 的随机过程产生。

生成过程分成两步：

首先：

\[
z\sim p_\theta(z)
\]

然后：

\[
x\sim p_\theta(x\mid z)
\]

也就是：

```text
先产生 z
    ↓
再根据 z 产生 x
```

写成图：

```text
z
│
│  pθ(x | z)
▼
x
```

其中：

- \(z\)：latent variable；
- \(x\)：observed variable；
- \(p_\theta(z)\)：latent variable 的 prior；
- \(p_\theta(x\mid z)\)：给定 \(z\) 后生成 \(x\) 的条件分布。

这就是一个最基本的：

> **Latent Variable Model**

---

# 5. 什么叫“生成模型”？

这里的“生成”不要理解成：

> 一定要生成漂亮图片。

更基础的含义是：

> **模型描述数据可能是怎样产生出来的。**

例如模型假设：

```text
先从某个分布中得到 z
↓
z 决定一些隐藏因素
↓
根据 z 产生 x
```

数学上：

\[
z\sim p(z)
\]

\[
x\sim p(x\mid z)
\]

于是联合分布可以写成：

\[
\boxed{
p(x,z)=p(z)p(x\mid z)
}
\]

这来自概率的乘法规则：

\[
p(x,z)=p(x\mid z)p(z)
\]

如果还不熟悉条件概率，可以阅读：

- Conditional Probability

---

# 6. 一个简单但重要的例子：高斯混合模型

为了真正理解 latent variable，我们先看一个完全不需要神经网络的例子。

假设我们观察到一维数据：

\[
x
\]

它大概长这样：

```text
      ****
    ********

                       *****
                    **********
------------------------------------------------
```

数据明显有两个 cluster。

我们可以假设：

> 每个数据点其实先属于某个隐藏类别，再从该类别对应的高斯分布中产生。

于是定义：

\[
z\in\{1,2\}
\]

其中：

- \(z=1\)：来自左边 cluster；
- \(z=2\)：来自右边 cluster。

生成过程：

\[
z\sim \text{Categorical}(\pi)
\]

然后：

\[
x\mid z=1
\sim
\mathcal N(\mu_1,\sigma_1^2)
\]

\[
x\mid z=2
\sim
\mathcal N(\mu_2,\sigma_2^2)
\]

这里：

\[
x
\]

是我们观察到的数据。

但是如果数据集只记录数值：

```text
1.12
0.95
5.34
...
```

并没有告诉我们：

```text
这个点来自 cluster 1
这个点来自 cluster 2
```

那么：

\[
z
\]

就是 latent variable。

---

# 7. 这个例子告诉我们一个关键点

Latent variable 并不是 VAE 发明的。

也不是神经网络才有。

早在各种经典统计模型中就存在 latent variables，例如：

- mixture models；
- factor analysis；
- hidden Markov models；
- probabilistic PCA；
- topic models；
- state-space models。

VAE 只是：

> **把复杂 latent-variable generative model 和神经网络、variational inference 结合起来。**

所以理解 latent variable 时：

> 不要先从 encoder / decoder 开始。

先从概率模型开始会清楚很多。

---

# 8. 为什么我们需要 latent variable？

最根本的原因是：

> **观测数据中的复杂变化，可能更容易通过一些没有直接观测到的内部变量来解释。**

假设我们直接建模：

\[
p(x)
\]

这意味着模型需要一次性描述：

> \(x\) 本身所有复杂变化。

但如果引入：

\[
z
\]

我们可以改成：

\[
p(x,z)=p(z)p(x\mid z)
\]

直觉上：

```text
复杂的数据分布 p(x)

变成

“隐藏因素怎么变化”
p(z)

+

“给定隐藏因素后数据怎么产生”
p(x | z)
```

有时这样的分解会让问题容易得多。

---

# 9. Latent Variable 最重要的作用不是“压缩”

很多 VAE 教程会说：

> \(z\) 是压缩表示。

在某些 VAE 架构里，这个说法确实有一定直觉价值。

例如：

\[
x\in\mathbb R^{784}
\]

而：

\[
z\in\mathbb R^{20}
\]

确实从高维像素进入了较低维 latent space。

但是：

> **“压缩”不是 latent variable 的定义。**

一个 latent variable：

- 可以是低维；
- 可以是高维；
- 可以是离散变量；
- 可以是连续变量；
- 可以比某些 observed representation 还大。

真正决定它是不是 latent 的，不是维度。

而是：

> **它有没有被直接观察到。**

---

# 10. Latent Variable 也不等于神经网络 Hidden Layer

这是另一个极常见误解。

假设普通神经网络：

\[
h=f_\theta(x)
\]

这里：

\[
h
\]

是 hidden representation。

因为：

\[
h
\]

位于网络中间层。

但它并不因此自动成为概率模型意义上的 latent variable。

如果：

\[
h=f_\theta(x)
\]

完全由 \(x\) 确定，

那么它只是：

> **deterministic hidden representation**

也就是确定性的中间表示。

而 VAE 中：

\[
z\sim q_\phi(z\mid x)
\]

\(z\) 是：

> **random variable**

给定同一个 \(x\)，我们描述的是一个关于 \(z\) 的概率分布。

因此：

```text
hidden layer
≠
latent variable
```

两者有时都被叫“隐藏表示”，但数学意义不同。

---

# 11. 为什么 latent variable 要是“随机变量”？

在概率 latent-variable model 中：

\[
z
\]

不是一个固定常数。

它服从某个分布：

\[
z\sim p(z)
\]

这意味着：

> 模型允许不同的 \(z\) 取值对应不同的潜在生成情况。

例如：

```text
z₁
↓
生成某种 x

z₂
↓
生成另一种 x

z₃
↓
又生成另一种 x
```

因此：

\[
p(x)
\]

实际上是把所有可能的 \(z\) 情况都考虑进去以后得到的结果。

这就引出 latent-variable model 最关键的数学操作之一：

> **Marginalization（边缘化）**

---

# 12. 我们看不到 z，那怎么得到 p(x)？

我们真正观察的是：

\[
x
\]

而不是：

\[
(x,z)
\]

所以如果模型定义了联合分布：

\[
p(x,z)
\]

我们最终还是需要得到：

\[
p(x)
\]

怎么把 \(z\) 消掉？

答案是：

> 对所有可能的 \(z\) 求和或积分。

---

# 13. 如果 z 是离散的：求和

例如：

\[
z\in\{1,2,3\}
\]

那么：

\[
\boxed{
p(x)
=
\sum_zp(x,z)
}
\]

利用：

\[
p(x,z)=p(x\mid z)p(z)
\]

得到：

\[
\boxed{
p(x)
=
\sum_z
p(x\mid z)p(z)
}
\]

意思是：

> \(x\) 出现的总概率 = 把所有可能隐藏情况 \(z\) 对 \(x\) 的贡献加起来。

---

# 14. 如果 z 是连续的：积分

如果：

\[
z\in\mathbb R^d
\]

那么求和变成积分：

\[
\boxed{
p_\theta(x)
=
\int
p_\theta(x,z)\,dz
}
\]

进一步：

\[
\boxed{
p_\theta(x)
=
\int
p_\theta(x\mid z)
p_\theta(z)
\,dz
}
\]

这就是 latent-variable model 中非常核心的：

> **Marginal Likelihood**

我们把看不见的：

\[
z
\]

积分掉以后，

只留下真正观测到的：

\[
x
\]

---

# 15. 为什么叫 Marginalization？

因为联合分布本来描述：

\[
(x,z)
\]

而我们把：

\[
z
\]

消掉：

\[
p(x,z)
\longrightarrow
p(x)
\]

得到某个变量自己的 marginal distribution。

因此：

\[
\int p(x,z)\,dz
=
p(x)
\]

如果不熟悉这个操作，可以单独阅读：

- Marginal Probability & Marginalization

它后面会频繁出现在：

- VAE；
- HMM；
- Bayesian inference；
- mixture model；
- probabilistic graphical model。

---

# 16. 一个非常重要的直觉：我们不知道 z，但不能假装它不存在

想象：

> 一张图片 \(x\) 是由许多隐藏因素共同产生的。

我们虽然没有直接观察这些因素，

但如果模型认为这些因素对 \(x\) 的生成很重要，那么在计算：

\[
p(x)
\]

时就不能简单忽略它们。

我们需要：

> 把所有可能的 latent configurations 都考虑进去。

所以：

\[
p(x)
=
\int
p(x\mid z)p(z)dz
\]

可以直觉理解为：

> “遍历所有可能的隐藏情况，看每种情况产生 \(x\) 的可能性有多大，再按照该隐藏情况本身出现的概率加权。”

---

# 17. Prior：还没有看到 x 时，我们怎么描述 z？

模型首先定义：

\[
p(z)
\]

它叫：

> **Prior Distribution（先验分布）**

意思是：

> 在还没有观察某一个具体 \(x\) 之前，模型对 \(z\) 可能取值的总体假设。

例如 VAE 中非常常见：

\[
\boxed{
p(z)=\mathcal N(0,I)
}
\]

也就是标准多元高斯。

为什么经常选择这种分布，我们会在：

- [VAE](./vae.md)
- Standard Normal Distribution

中详细讨论。

这里暂时只需要知道：

\[
p(z)
\]

描述：

> **生成数据之前，latent variable 怎么分布。**

---

# 18. Likelihood：如果 z 已经知道，x 会怎么产生？

接下来模型定义：

\[
p_\theta(x\mid z)
\]

这表示：

> 如果 latent variable 是 \(z\)，那么观测 \(x\) 应该服从什么分布？

这就是：

> **Likelihood / Conditional Generative Distribution**

例如非常抽象地：

```text
z
↓
neural network
↓
产生 pθ(x | z) 的参数
↓
从这个分布生成 x
```

在 VAE 中，decoder 通常就是负责参数化：

\[
p_\theta(x\mid z)
\]

所以严格来说：

> decoder 不是简单地“把 z 变回 x”。

它更准确地是在定义：

\[
\boxed{
p_\theta(x\mid z)
}
\]

这个条件分布。

---

# 19. Generative Direction：z → x

到这里，生成模型的方向是：

\[
\boxed{
z\rightarrow x
}
\]

流程：

```text
sample z from p(z)
↓
given z, obtain pθ(x | z)
↓
sample / generate x
```

这叫：

> **Generative Process**

VAE 原论文就是用这个方向定义 latent-variable model。

但实际使用时，我们经常面对完全相反的问题。

---

# 20. 真正拿到一张 x 后，我们最想问什么？

现实中我们通常不是先看到：

\[
z
\]

然后看到：

\[
x
\]

而是：

> 数据 \(x\) 已经摆在我们面前。

例如已经有一张图片。

于是自然想问：

> **什么样的 latent variable \(z\) 可能生成了这张 \(x\)？**

数学上：

\[
\boxed{
p_\theta(z\mid x)
}
\]

这叫：

> **Posterior Distribution（后验分布）**

---

# 21. Prior 和 Posterior 千万不要混

这是后面 VAE 最重要的区别之一。

### Prior

\[
p(z)
\]

含义：

> 还没有看到当前数据 \(x\) 时，对 \(z\) 的总体假设。

---

### Posterior

\[
p(z\mid x)
\]

含义：

> 已经看到了具体 \(x\) 以后，哪些 \(z\) 更可能解释它？

可以画成：

```text
还没看到 x：

        p(z)
         ↓
     对 z 的先验


看到 x 以后：

         x
         ↓
     p(z | x)
         ↓
哪些 z 更可能生成这个 x？
```

---

# 22. Posterior 从哪里来？

根据 Bayes' Rule：

\[
\boxed{
p_\theta(z\mid x)
=
\frac{
p_\theta(x\mid z)p(z)
}{
p_\theta(x)
}
}
\]

分子很好理解：

\[
p_\theta(x\mid z)p(z)
\]

表示：

> \(z\) 本身有多合理，以及这个 \(z\) 产生 \(x\) 的可能性有多大。

困难在分母：

\[
p_\theta(x)
\]

而我们刚刚知道：

\[
p_\theta(x)
=
\int
p_\theta(x\mid z)p(z)dz
\]

所以：

\[
\boxed{
p_\theta(z\mid x)
=
\frac{
p_\theta(x\mid z)p(z)
}{
\int p_\theta(x\mid z)p(z)dz
}
}
\]

现在你应该已经能看到 VAE 的问题为什么会出现了。

---

# 23. 真正困难的地方：这个积分可能算不出来

如果：

\[
p_\theta(x\mid z)
\]

非常简单，

有时：

\[
\int p_\theta(x\mid z)p(z)dz
\]

可以解析计算。

但如果：

\[
p_\theta(x\mid z)
\]

由复杂神经网络参数化，

这个积分通常可能变得：

> **intractable**

也就是实际无法直接精确计算。

于是：

\[
p_\theta(z\mid x)
\]

也很难直接得到。

Kingma & Welling 的 VAE 原论文正是从这个问题出发：

> 如何在连续 latent variables 的 posterior 难以计算时，仍然高效地完成 inference 和 learning？

这就是：

> **Variational Inference**

和：

> **VAE**

真正进入故事的地方。

---

# 24. VAE 为什么需要 Encoder？

现在我们终于可以用严格方式理解 VAE encoder。

真实 posterior：

\[
p_\theta(z\mid x)
\]

很难算。

所以 VAE 引入另一个分布：

\[
\boxed{
q_\phi(z\mid x)
}
\]

去近似：

\[
p_\theta(z\mid x)
\]

即：

\[
q_\phi(z\mid x)
\approx
p_\theta(z\mid x)
\]

这个：

\[
q_\phi(z\mid x)
\]

通常由神经网络参数化。

这就是 VAE 中所谓的：

> **Probabilistic Encoder / Inference Model / Recognition Model**

所以 VAE encoder 真正做的不是：

> “把图片压缩成一个 z。”

更准确地说：

> **给定 \(x\)，它输出一个关于可能 latent variable \(z\) 的近似 posterior distribution。**

例如：

\[
q_\phi(z\mid x)
=
\mathcal N(
\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))
)
\]

encoder 的直接输出实际上通常是：

\[
\mu
\]

和：

\[
\sigma^2
\]

或者：

\[
\log\sigma^2
\]

而不是直接输出一个固定 \(z\)。

然后才从这个 distribution 中采样：

\[
z
\]

---

# 25. 所以同一个 x 可以对应多个 z 吗？

在概率模型中：

**可以。**

因为 encoder 定义的是：

\[
q_\phi(z\mid x)
\]

一个 distribution。

不是：

\[
z=f(x)
\]

一个固定点。

例如：

\[
q_\phi(z\mid x)
=
\mathcal N(\mu,\sigma^2)
\]

同一个 \(x\) 下，

可以采样：

\[
z_1
\]

也可以采样：

\[
z_2
\]

也可以采样：

\[
z_3
\]

它们都来自：

\[
q_\phi(z\mid x)
\]

这就是 VAE 和普通 deterministic autoencoder 的一个根本区别。

---

# 26. Latent Variable 不等于“随机噪声”

这里也容易误解。

VAE 训练时会使用随机采样：

\[
z\sim q_\phi(z\mid x)
\]

于是有人会说：

> \(z\) 就是随机噪声。

不准确。

真正的随机噪声通常是后面 [Reparameterization Trick](./reparameterization-trick.md) 中引入的：

\[
\epsilon\sim\mathcal N(0,I)
\]

然后：

\[
z
=
\mu+\sigma\odot\epsilon
\]

其中：

- \(\epsilon\)：辅助随机噪声；
- \(z\)：latent variable sample。

所以：

\[
\boxed{
z\neq\epsilon
}
\]

虽然二者都涉及随机性。

---

# 27. Latent Variable 一定代表人类能理解的语义吗？

**不一定。**

这是极其重要的一点。

我们经常喜欢说：

```text
z₁ = 颜色
z₂ = 角度
z₃ = 风格
z₄ = 速度
```

这种图很容易让人理解 latent space。

但普通 VAE 并不保证：

> 每个 latent dimension 自动对应一个清晰的人类语义。

模型优化的是数学 objective。

它可能学到一些对生成数据有用的 representation。

但这些 representation：

- 可能纠缠在一起；
- 可能很难解释；
- 可能发生旋转、置换等变化；
- 不一定对应我们心中命名好的因素。

因此最安全的说法是：

> **\(z\) 表示模型用于解释数据变化的 latent representation。**

不要无依据地说：

> “第 3 维就是动作速度。”

除非有实验真正证明。

---

# 28. Latent Variable 一定是真实世界中的“原因”吗？

也不一定。

如果我们写：

\[
z\rightarrow x
\]

这表示：

> 在当前概率生成模型的生成方向里，先有 \(z\)，再生成 \(x\)。

但这并不自动证明：

> \(z\) 就是真实世界中的 causal variable。

概率生成关系：

\[
p(x\mid z)
\]

和严格的 causal claim：

> \(z\) 在现实中导致 \(x\)

不是同一件事。

因此除非模型和研究本身进行了 causal identification，

否则不要随意把 latent variable 写成：

> “真实世界隐藏原因”。

---

# 29. 一个更准确的 mental model

可以把 latent variable 想成：

> **模型允许自己提出的一组未观测解释变量。**

模型说：

> “虽然数据只给了我 \(x\)，但如果我假设还有一个看不见的 \(z\)，那么 \(x\) 的复杂分布可能更容易被描述。”

于是：

\[
p(x)
\]

被改写成：

\[
p(x)
=
\int
p(x\mid z)p(z)dz
\]

这是 latent-variable modeling 的核心。

---

# 30. 为什么 Latent Variable 特别适合多模态数据？

假设在同一个大致条件下，可能出现两种完全不同的结果：

```text
模式 A
********

模式 B
                    ********
```

如果没有 latent variable，

模型可能需要直接用一个复杂的：

\[
p(x)
\]

同时表示两个 mode。

而引入离散 latent variable：

\[
z\in\{A,B\}
\]

以后：

\[
p(x)
=
p(x\mid z=A)p(z=A)
+
p(x\mid z=B)p(z=B)
\]

于是：

```text
z=A
↓
负责模式 A

z=B
↓
负责模式 B
```

从建模角度会非常自然。

---

# 31. 但“latent variable = multimodality”也不能说得太绝对

引入 latent variables 可以帮助模型表达多种潜在生成情况。

但不能反过来说：

> 只要有 latent variable，就一定解决 multimodality。

模型是否真的学到多个有意义 mode，还依赖：

- model family；
- training objective；
- latent dimension；
- optimization；
- data；
- posterior approximation；
- decoder capacity。

所以 latent variable 是：

> **一种建模机制。**

不是自动保证某种语义结果的魔法按钮。

---

# 32. 一个最简单的 latent generative model

假设：

\[
z\sim\mathcal N(0,1)
\]

并定义：

\[
x=2z+\epsilon
\]

其中：

\[
\epsilon\sim\mathcal N(0,0.1^2)
\]

这里：

\[
z
\]

没有被我们直接观测。

我们只看到：

\[
x
\]

生成过程：

```text
sample z
↓
乘以 2
↓
加一点 observation noise
↓
得到 x
```

如果某次我们看到：

\[
x=4.1
\]

自然会反推：

> 哪些 \(z\) 比较可能产生 4.1？

这正是在计算：

\[
p(z\mid x=4.1)
\]

也就是 posterior inference。

所以 latent-variable model 同时自然产生两个方向：

```text
Generative:

z → x


Inference:

x → possible z
```

VAE 正是在处理这两个方向。

---

# 33. Generative Model 和 Inference Model 不要混

VAE 中存在两套 distribution：

### Generative Model

\[
p_\theta(z)
\]

\[
p_\theta(x\mid z)
\]

方向：

\[
z\rightarrow x
\]

---

### Inference Model

\[
q_\phi(z\mid x)
\]

方向：

\[
x\rightarrow z
\]

它的目标是近似：

\[
p_\theta(z\mid x)
\]

因此可以画成：

```text
Generative direction

z ─────────→ x
   pθ(x | z)


Inference direction

x ─────────→ z
   qφ(z | x)
```

不要因为代码里 encoder 写在 decoder 前面，就误以为：

> encoder 是整个概率模型真正的生成方向。

生成模型的定义仍然是：

\[
z\rightarrow x
\]

---

# 34. 为什么 VAE 叫 Autoencoder，容易误导初学者？

普通 Autoencoder 的故事通常是：

```text
x
↓
encoder
↓
h
↓
decoder
↓
x̂
```

于是初学 VAE 时很自然地把它理解为：

```text
x
↓
compress
↓
z
↓
decompress
↓
x̂
```

这可以作为早期直觉。

但如果停在这里，就会漏掉 VAE 最重要的概率意义。

VAE 的真正结构更接近：

```text
Approximate Inference:

x
↓
qφ(z | x)
↓
sample z


Generative Model:

z
↓
pθ(x | z)
↓
model x
```

其中：

\[
z
\]

是随机变量。

而不是普通 autoencoder 中一个确定性的 bottleneck vector。

---

# 35. 现在终于可以回到 ACT

ACT 为什么也有：

\[
z
\]

？

因为 ACT 不只是一个普通 deterministic action regressor。

它把 policy 训练成：

> **Conditional Variational Autoencoder（CVAE）**

训练时，ACT 的 CVAE encoder 接收：

- 当前 proprioceptive observation；
- demonstration action sequence；

然后构造一个关于：

\[
z
\]

的 distribution。

ACT 论文把 \(z\) 称为：

> **style variable**

decoder / policy 再根据：

- 当前 observation；
- \(z\)；

预测未来 action chunk。

从 latent-variable model 的角度：

> \(z\) 给模型提供了一个没有直接出现在 observation 中的 latent degree of freedom，用于描述 demonstration action sequence 中的变化。

---

# 36. 但 ACT 的 z 不能被随意解释成某个具体因素

例如我们可以用直觉说：

> 不同 \(z\) 可能对应不同的动作风格或轨迹变化。

这是合理的教学直觉。

但不能未经实验就断言：

```text
z₁ = 速度
z₂ = 力量
z₃ = 左右偏好
z₄ = 操作者状态
```

ACT 训练目标并没有显式监督这些语义。

所以：

> **style variable 是模型里的 latent representation，不是一组预先定义好的人类语义标签。**

这是后面理解 ACT 的 \(z\) 时非常重要的边界。

---

# 37. 为什么 ACT 训练时需要看 action sequence 才推断 z？

思考一个问题。

如果当前 observation 是：

\[
o_t
\]

但同一个 observation 附近可能存在多个不同的合理 action chunks：

\[
A_t^{(1)}
\]

\[
A_t^{(2)}
\]

\[
A_t^{(3)}
\]

仅仅看：

\[
o_t
\]

可能无法知道：

> 当前这条 training demonstration 具体采用的是哪一种轨迹变化。

所以训练阶段 CVAE encoder 还会看到：

\[
A_t
\]

然后推断：

\[
q_\phi(z\mid A_t,\bar o_t)
\]

直觉上相当于问：

> “看到这次 demonstration 实际是这么做的，那么什么样的 latent \(z\) 可以帮助 decoder 重现这一条 action chunk？”

于是：

```text
demonstration action chunk
        ↓
infer latent distribution
        ↓
sample z
        ↓
decoder reconstructs that chunk
```

这里已经开始进入 CVAE。

详细内容会在：

- [CVAE](./cvae.md)
- [CVAE in ACT](../robot-learning/act/cvae-in-act.md)

继续展开。

---

# 38. 一个与你后面理解 ACT 非常相关的问题

你可能会马上问：

> 如果训练时 \(z\) 是从 demonstration action sequence 推断出来的，
> 那推理时根本没有 ground-truth future actions，
> \(z\) 从哪里来？

这是非常关键的问题。

ACT 的答案是：

> 推理阶段丢弃 CVAE encoder，并设置：

\[
z=0
\]

也就是 prior：

\[
\mathcal N(0,I)
\]

的均值。

但现在还不要急着把：

\[
z=0
\]

理解成：

> “把 latent information 删除掉。”

要理解为什么这样做能成立，需要先理解：

1. [Normal Distribution](../mathematics/normal-distribution.md)
2. Standard Normal Distribution
3. [VAE](./vae.md)
4. [KL Divergence](../mathematics/kl-divergence.md)
5. [CVAE](./cvae.md)

这些知识连起来以后，

ACT 推理时：

\[
z=0
\]

才会真正变得清楚。

---

# 39. 常见误解一：latent variable 就是模型不知道的任何东西

**太宽泛。**

现实里当然有无数我们“不知道”的因素。

但模型中的 latent variable：

> 必须是模型概率结构中的变量。

也就是说它参与：

\[
p(x,z)
\]

或者：

\[
p(x\mid z)
\]

等概率关系。

不是随便一个“我们不知道的现实因素”都自动成为模型的 \(z\)。

---

# 40. 常见误解二：latent variable = hidden layer

**错误。**

普通 hidden activation：

\[
h=f(x)
\]

可能只是确定性中间计算。

概率 latent variable：

\[
z
\]

是模型中的随机变量。

VAE 中：

\[
z\sim q_\phi(z\mid x)
\]

两者不能只因为都“看不见”就混为一谈。

---

# 41. 常见误解三：latent space 每一维都有明确含义

**没有保证。**

普通 VAE objective 并不会自动给每个维度贴上：

```text
angle
speed
color
style
```

这种标签。

如果想获得更明确的 disentanglement 或 identifiability，需要额外假设、objective 或 supervision。

---

# 42. 常见误解四：z 就是压缩后的 x

**不够准确。**

对于 VAE：

\[
q_\phi(z\mid x)
\]

描述的是：

> 给定 \(x\) 后 latent variable 的 approximate posterior。

encoder 通常输出 distribution parameters：

\[
\mu(x),\sigma^2(x)
\]

再从中 sample：

\[
z
\]

所以它不是简单的 deterministic compression。

---

# 43. 常见误解五：z 就是随机噪声

**错误。**

在 Gaussian VAE 的 reparameterization 中：

\[
\epsilon\sim\mathcal N(0,I)
\]

才是辅助 noise。

然后：

\[
z=\mu+\sigma\odot\epsilon
\]

所以：

\[
z
\]

和：

\[
\epsilon
\]

不是同一个变量。

---

# 44. 常见误解六：latent variable 一定是真实 causal factor

**没有这种保证。**

生成模型使用：

\[
z\rightarrow x
\]

作为建模方向，

并不自动证明模型恢复了现实世界真正的因果变量。

---

# 45. 常见误解七：有 z 就一定能产生多样结果

也不保证。

如果训练中出现：

- posterior collapse；
- decoder 忽略 \(z\)；
- latent capacity 不合适；
- objective 权衡不好；

那么模型可能几乎不用 latent variable。

所以真正的问题不是：

> “模型有没有一个变量叫 \(z\)？”

而是：

> **训练后的生成模型有没有真正利用它。**

这个问题会在 VAE 中详细讨论。

---

# 46. 用三条公式记住 Latent Variable Model

如果这一篇所有东西最后只留下三条公式，应该是：

---

## 生成模型

\[
\boxed{
p_\theta(x,z)
=
p(z)p_\theta(x\mid z)
}
\]

意思：

> 先有 latent \(z\)，再生成 observed \(x\)。

---

## Marginal Likelihood

\[
\boxed{
p_\theta(x)
=
\int
p_\theta(x\mid z)p(z)dz
}
\]

意思：

> 因为 \(z\) 没有被观察，所以计算 \(x\) 的概率时，要把所有可能的 \(z\) 都考虑进去。

---

## Posterior

\[
\boxed{
p_\theta(z\mid x)
=
\frac{
p_\theta(x\mid z)p(z)
}{
p_\theta(x)
}
}
\]

意思：

> 已经看见 \(x\) 后，反推哪些 latent \(z\) 更可能解释它。

VAE 的故事几乎就是从：

> **这个 posterior 很难算**

开始的。

---

# 47. 一句话重新理解

> **Latent variable 不是“神秘特征向量”，而是概率模型中没有被直接观测到、却被模型用来描述数据生成过程的随机变量。**

我们只观察：

\[
x
\]

模型却假设：

\[
z\rightarrow x
\]

于是：

\[
p(x)
=
\int p(x\mid z)p(z)dz
\]

而看到某个具体 \(x\) 后，我们又希望反推：

\[
p(z\mid x)
\]

这就是为什么 latent variable 一出现，

后面自然会跟着出现：

- prior；
- posterior；
- Bayes；
- marginalization；
- variational inference；
- VAE；
- CVAE。

---

# 48. 下一步：为什么需要 VAE？

现在我们已经有 latent-variable model：

\[
p(z)p_\theta(x\mid z)
\]

下一步最大的困难是：

\[
p_\theta(z\mid x)
\]

需要：

\[
p_\theta(x)
=
\int
p_\theta(x\mid z)p(z)dz
\]

而这个积分在复杂神经生成模型中往往无法直接计算。

于是我们会问：

> 能不能训练另一个网络：

\[
q_\phi(z\mid x)
\]

去近似真正的：

\[
p_\theta(z\mid x)
\]

？

如果可以，

我们又如何训练：

\[
q_\phi
\]

和：

\[
p_\theta
\]

？

为什么目标函数最终会出现：

\[
D_{KL}
\]

和 reconstruction term？

为什么 encoder 输出的是：

\[
\mu,\sigma^2
\]

而不是直接输出 \(z\)？

以及：

> 为什么采样操作还能反向传播？

这些问题会在下一篇：

> [VAE：为什么我们需要 Variational Autoencoder？](./vae.md)

开始完整展开。

---

## Primary Source

Diederik P. Kingma, Max Welling.  
**Auto-Encoding Variational Bayes.**  
arXiv:1312.6114, first submitted 2013.

- Paper: https://arxiv.org/abs/1312.6114
- HTML: https://arxiv.org/html/1312.6114

本文中关于连续 latent-variable generative model 的核心概率结构主要依据该论文：

- Section 2.1 — Problem Scenario
- Figure 1
- Section 2.2 — The Variational Bound

尤其是原论文定义的生成过程：

\[
z\sim p_\theta(z)
\]

\[
x\sim p_\theta(x\mid z)
\]

以及：

\[
p_\theta(x)
=
\int p_\theta(z)p_\theta(x\mid z)dz
\]

---

## 与 ACT 的关系

Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.  
**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.**  
Robotics: Science and Systems (RSS), 2023.

- Paper: https://arxiv.org/abs/2304.13705

ACT 将 policy 训练为 CVAE，并使用 latent style variable \(z\) 描述 demonstration action sequence 中的潜在变化。

本文只负责建立：

> **latent variable 本身的概率学意义。**

ACT 中 \(z\) 的具体构造将在：

- [CVAE in ACT](../robot-learning/act/cvae-in-act.md)

单独讨论。

---

## 本文知识连接

### 数学前置知识

- Probability Distribution
- Conditional Probability
- Bayes' Rule
- Marginalization

### 下一步

- [VAE](./vae.md)
- [Reparameterization Trick](./reparameterization-trick.md)
- [CVAE](./cvae.md)

### VAE 中会用到

- [Normal Distribution](../mathematics/normal-distribution.md)
- Standard Normal Distribution
- Mean
- Variance
- [KL Divergence](../mathematics/kl-divergence.md)

### Robot Learning

- [ACT](../robot-learning/act/act-what-problem-does-it-solve.md)
- [CVAE in ACT](../robot-learning/act/cvae-in-act.md)
