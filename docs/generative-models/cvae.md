---
title: "CVAE：条件信息到底改变了什么？"
description: "从 VAE 到 Conditional VAE，严格理解 one-to-many 条件生成、conditional prior、recognition model、conditional ELBO，以及训练和推理时 latent z 的来源为什么不同。"
status: reviewed
pageType: concept
canonical: /generative-models/cvae
updated: "2026-09-15"
---

# CVAE：条件信息到底改变了什么？

在 [VAE](./vae.md) 中，我们学习了一个生成模型：

\[
z\sim p(z)
\]

\[
x\sim p_\theta(x\mid z)
\]

它回答的问题是：

> **如果我只想学习整个数据分布 \(p(x)\)，怎样用 latent variable \(z\) 来描述数据中的变化？**

但很多现实任务不是：

> “随便生成一个 \(x\)。”

而是：

> **“在已经知道某些条件的情况下，生成一个合理的结果。”**

例如：

```text
给定一张局部图片
→ 补全剩下的图片

给定一句文字
→ 生成符合文字的图片

给定机器人当前 observation
→ 生成接下来合理的 action sequence

给定类别 label
→ 生成这个类别中的样本
```

这里我们不再只关心：

\[
p(y)
\]

而是关心：

\[
\boxed{
p(y\mid c)
}
\]

其中：

- \(c\)：condition，已经知道的信息；
- \(y\)：我们希望生成 / 预测的输出。

Conditional Variational Autoencoder：

> **CVAE**

要解决的就是这个问题。

但千万不要把它简单记成：

> “VAE 多输入一个 condition。”

条件 \(c\) 加进来以后，真正改变的是整个概率模型：

- 我们要建模的目标变了；
- posterior 变了；
- generative distribution 变了；
- prior **也可能**变成 conditional prior；
- 训练和推理时 \(z\) 的来源也出现了新的差异。

这一篇我们从概率模型开始，把这些变化全部讲清楚。

---

# 1. 先看一个 CVAE 最适合解决的问题：One-to-Many

假设我们只看到一张手写数字图片的左半边：

```text
┌────────┬────────┐
│        │        │
│  已知  │   ?    │
│ 左半边 │ 右半边 │
│        │        │
└────────┴────────┘
```

现在要求模型预测右半边。

问题是：

> **同一个左半边，可能对应不止一种合理的完整数字。**

例如一个模糊的左半边，可能来自：

- 3；
- 5；
- 8；

甚至同一个数字也可以有不同书写方式。

所以：

\[
c
\]

并不一定唯一决定：

\[
y
\]

而可能存在：

\[
y^{(1)},y^{(2)},y^{(3)},\ldots
\]

都与同一个条件：

\[
c
\]

相容。

这就是：

> **one-to-many mapping**

Sohn, Lee & Yan 2015 提出 CVAE 时，核心动机就是 structured output prediction 中这种：

> 一个输入可能对应多个合理输出

的问题。

---

# 2. 为什么普通确定性网络容易有问题？

假设使用一个普通神经网络：

\[
\hat y=f_\theta(c)
\]

给定：

\[
c
\]

它输出唯一的：

\[
\hat y
\]

如果训练数据中：

\[
c
\]

附近存在多个不同但合理的输出：

\[
y^{(1)},y^{(2)},y^{(3)}
\]

一个 deterministic regressor 在某些 loss 下可能倾向于输出某种“平均结果”。

对于一维数字也许没问题。

但对于：

- 图像；
- segmentation；
- 动作轨迹；
- 复杂 structured output；

多个有效 mode 的平均不一定仍然是一个有效结果。

例如两条都合理的机器人轨迹：

```text
轨迹 A：
先向左绕
↓
再向前


轨迹 B：
先向右绕
↓
再向前
```

简单平均可能变成：

```text
直接向前
↓
撞上障碍
```

所以我们真正想学的不是：

\[
\boxed{
y=f(c)
}
\]

而是：

\[
\boxed{
p_\theta(y\mid c)
}
\]

即：

> **给定 condition \(c\)，所有合理输出 \(y\) 的条件概率分布。**

---

# 3. 只写 p(y|c) 还不够

如果：

\[
p_\theta(y\mid c)
\]

本身具有多个 mode，

直接建模它可能非常复杂。

于是我们沿用 latent-variable model 的思想：

> 在 condition \(c\) 和 output \(y\) 之间，再引入一个 latent variable \(z\)。

可以先直觉画成：

```text
condition c
     │
     ├─────────────┐
     │             │
     ▼             ▼
latent z ───────→ output y
```

\(z\) 用来表达：

> 在同一个 condition 下，输出仍然可以怎样变化。

于是：

```text
同一个 c
+
不同 z
↓
不同但合理的 y
```

例如：

```text
c = 当前机器人状态

z₁
→ 一种合理动作轨迹

z₂
→ 另一种合理动作轨迹
```

但注意：

> **这只是 latent variable 的直觉作用。**

不能未经实验就说：

```text
z₁ = “向左”
z₂ = “向右”
```

更严格地说：

> \(z\) 是条件生成模型中的未观测随机变量，用于帮助表示 \(p(y\mid c)\) 中的剩余变化。

---

# 4. VAE 和 CVAE 的目标到底差在哪里？

普通 VAE 希望建模：

\[
\boxed{
p_\theta(x)
}
\]

CVAE 希望建模：

\[
\boxed{
p_\theta(y\mid c)
}
\]

这就是最根本的改变。

---

## VAE

问题：

> 数据 \(x\) 整体怎么分布？

latent model：

\[
p_\theta(x,z)
=
p(z)p_\theta(x\mid z)
\]

---

## CVAE

问题：

> 已知 condition \(c\) 后，output \(y\) 怎么分布？

latent model：

\[
p_\theta(y,z\mid c)
\]

然后把 latent \(z\) marginalize：

\[
\boxed{
p_\theta(y\mid c)
=
\int
p_\theta(y,z\mid c)\,dz
}
\]

所以 CVAE 仍然是一个 latent-variable model。

只是：

> **整个模型现在处于 condition \(c\) 之下。**

---

# 5. 原论文中的 CVAE：Prior 也依赖 Condition

Sohn et al. 2015 的 CVAE 定义为：

先给定输入：

\[
c
\]

然后从 conditional prior 中采样：

\[
\boxed{
z\sim p_\theta(z\mid c)
}
\]

再生成输出：

\[
\boxed{
y\sim p_\theta(y\mid c,z)
}
\]

因此：

\[
\boxed{
p_\theta(y,z\mid c)
=
p_\theta(z\mid c)
p_\theta(y\mid c,z)
}
\]

最后：

\[
\boxed{
p_\theta(y\mid c)
=
\int
p_\theta(z\mid c)
p_\theta(y\mid c,z)
\,dz
}
\]

这就是原论文 Section 4 中的 conditional generative process。

---

# 6. 为什么 Prior 也可以依赖 c？

先回忆普通 VAE：

\[
p(z)
=
\mathcal N(0,I)
\]

无论输入数据是什么，prior 都一样。

但在 conditional generation 中，可以允许：

\[
p_\theta(z\mid c)
\]

随着 condition 改变。

例如：

```text
condition c₁
↓
latent prior A


condition c₂
↓
latent prior B
```

直觉上：

> 不同 condition 下，合理的 latent variations 本来就可能不同。

Sohn et al. 的原始 CVAE formulation 正是这样做：

> input observation 调制 latent variable 的 prior。

不过这里有一个极其重要的细节。

---

# 7. Conditional Prior 不是 CVAE 的唯一可能形式

原论文明确指出：

> 可以放宽这个约束，让 latent variable 与 condition 统计独立。

也就是：

\[
\boxed{
p_\theta(z\mid c)=p(z)
}
\]

于是生成模型变成：

\[
p_\theta(y,z\mid c)
=
p(z)
p_\theta(y\mid c,z)
\]

以及：

\[
\boxed{
p_\theta(y\mid c)
=
\int
p(z)p_\theta(y\mid c,z)\,dz
}
\]

这个版本非常重要。

因为很多后来被称为 CVAE 的模型——包括我们后面要学习的 ACT——采用的就是：

> **固定 prior \(p(z)\)，而 decoder / policy 被 condition 所条件化。**

所以以后看到“CVAE”时，不要机械认为：

\[
p(z\mid c)
\]

一定存在一个单独的 conditional prior network。

应该查看具体论文。

---

# 8. 为了统一，接下来用 c 表示 condition

原始 Sohn et al. 论文使用：

- \(x\)：input / condition；
- \(y\)：output；
- \(z\)：latent。

为了和我们前面 VAE 文章里：

\[
x=\text{被生成的数据}
\]

的记号避免混乱，这一篇改用：

\[
c=\text{condition}
\]

\[
y=\text{output}
\]

\[
z=\text{latent}
\]

所以原论文：

\[
p_\theta(z\mid x)
\]

在本文写成：

\[
p_\theta(z\mid c)
\]

原论文：

\[
p_\theta(y\mid x,z)
\]

在本文写成：

\[
p_\theta(y\mid c,z)
\]

数学含义完全一样。

---

# 9. Generative Process：真正生成 y 时发生什么？

先看原论文形式。

给定：

\[
c
\]

### Step 1

从 conditional prior 采样：

\[
z\sim p_\theta(z\mid c)
\]

### Step 2

根据 condition 和 latent 生成输出：

\[
y\sim p_\theta(y\mid c,z)
\]

所以生成方向：

```text
condition c
   │
   ▼
pθ(z | c)
   │
   ▼
   z
   │
   ├──────────┐
   │          │
   ▼          ▼
         pθ(y | c,z)
               │
               ▼
               y
```

更简洁地画：

```text
c ───────→ y
│          ▲
│          │
└──→ z ────┘
```

---

# 10. 为什么同一个 c 可以产生多个 y？

因为：

\[
z
\]

是随机变量。

固定：

\[
c
\]

以后，

可以采样：

\[
z_1,z_2,z_3,\ldots
\]

然后：

\[
p_\theta(y\mid c,z_1)
\]

\[
p_\theta(y\mid c,z_2)
\]

\[
p_\theta(y\mid c,z_3)
\]

可以产生不同输出。

因此：

```text
       z₁ → y₁
      /
c ──── z₂ → y₂
      \
       z₃ → y₃
```

这就是 CVAE 表达 one-to-many mapping 的核心机制。

Sohn et al. 的实验正是为了展示：

> 相同或部分相同输入条件下，模型能够生成 diverse but realistic outputs。

---

# 11. 现在到了训练时真正困难的地方

生成时：

\[
c
\]

已知，

但：

\[
z
\]

不可观测。

训练数据给我们的是：

\[
(c,y)
\]

也就是说：

```text
condition
+
ground-truth output
```

但没有：

```text
ground-truth z
```

我们自然想知道：

> 对于这条训练样本 \((c,y)\)，什么样的 \(z\) 最可能解释这个具体输出 \(y\)？

这就是：

\[
\boxed{
p_\theta(z\mid c,y)
}
\]

即真实 conditional posterior。

---

# 12. 和 VAE 一样，真实 Posterior 通常难算

根据 Bayes：

\[
p_\theta(z\mid c,y)
=
\frac{
p_\theta(y\mid c,z)
p_\theta(z\mid c)
}{
p_\theta(y\mid c)
}
\]

而：

\[
p_\theta(y\mid c)
=
\int
p_\theta(y\mid c,z)
p_\theta(z\mid c)
\,dz
\]

这个积分通常 intractable。

所以：

\[
p_\theta(z\mid c,y)
\]

也很难直接计算。

这与 VAE 的问题完全同构。

于是我们再次引入：

> **Approximate Posterior / Recognition Model**

---

# 13. CVAE 的 Recognition Model

CVAE 定义：

\[
\boxed{
q_\phi(z\mid c,y)
}
\]

去近似：

\[
p_\theta(z\mid c,y)
\]

即：

\[
q_\phi(z\mid c,y)
\approx
p_\theta(z\mid c,y)
\]

注意和 VAE 的区别。

普通 VAE：

\[
q_\phi(z\mid x)
\]

CVAE：

\[
\boxed{
q_\phi(z\mid c,y)
}
\]

为什么 training encoder 要同时看到：

\[
c
\]

和：

\[
y
\]

？

因为它要回答：

> **“在这个 condition 下，既然真实 output 是这个 \(y\)，什么 latent \(z\) 可以解释这条具体输出？”**

---

# 14. 一个机器人例子

假设 condition：

\[
c
=
\text{机器人当前状态}
\]

同一个 \(c\) 下有两条合理动作：

```text
y₁:
先略微左移
→ 抓取


y₂:
先略微右移
→ 抓取
```

训练数据中的某个 sample 实际给的是：

\[
(c,y_1)
\]

recognition model：

\[
q_\phi(z\mid c,y_1)
\]

可以利用：

\[
y_1
\]

知道：

> 这条 training example 具体采用了第一种轨迹。

如果只看：

\[
c
\]

它可能根本无法知道训练数据这次选的是：

\[
y_1
\]

还是：

\[
y_2
\]

这就是为什么：

> **training recognition model 通常能看 ground-truth output \(y\)。**

---

# 15. 但这里马上产生 Train–Test Gap

训练时有：

\[
y
\]

所以可以使用：

\[
q_\phi(z\mid c,y)
\]

但真正推理时，我们想预测的正是：

\[
y
\]

还没有 ground-truth \(y\)。

所以 inference 时不可能先计算：

\[
q_\phi(z\mid c,y)
\]

因为这相当于：

> 为了预测答案，先把答案喂给模型。

因此测试时必须从：

\[
\boxed{
p_\theta(z\mid c)
}
\]

采样，

或者在 fixed-prior CVAE 中：

\[
\boxed{
p(z)
}
\]

采样。

这是 CVAE 中非常重要的：

> **training / inference asymmetry**

---

# 16. Training 和 Inference 的两条路径

## Training

我们有：

\[
(c,y)
\]

所以：

```text
condition c
+
ground-truth y
        ↓
qφ(z | c,y)
        ↓
sample z
        ↓
pθ(y | c,z)
        ↓
reconstruct / explain y
```

---

## Inference

只有：

\[
c
\]

没有 ground-truth：

\[
y
\]

所以：

```text
condition c
        ↓
pθ(z | c)
        ↓
sample z
        ↓
pθ(y | c,z)
        ↓
generate prediction y
```

或者 fixed prior：

```text
z ~ p(z)
+
condition c
        ↓
pθ(y | c,z)
        ↓
y
```

这两条路径必须分清。

---

# 17. CVAE 的 Conditional ELBO

我们的目标是最大化：

\[
\log p_\theta(y\mid c)
\]

但是：

\[
p_\theta(y\mid c)
=
\int
p_\theta(y\mid c,z)
p_\theta(z\mid c)
\,dz
\]

通常难算。

于是和 VAE 一样，引入：

\[
q_\phi(z\mid c,y)
\]

可以得到 conditional ELBO：

\[
\boxed{
\log p_\theta(y\mid c)
\ge
\mathbb E_{
q_\phi(z\mid c,y)
}
[
\log p_\theta(y\mid c,z)
]
-
D_{KL}
\left(
q_\phi(z\mid c,y)
\parallel
p_\theta(z\mid c)
\right)
}
\]

这就是 Sohn et al. 原论文 Equation (4) 的核心形式。

---

# 18. 这和 VAE 的 ELBO 有多像？

普通 VAE：

\[
\mathcal L_{\text{VAE}}
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
\]

CVAE：

\[
\mathcal L_{\text{CVAE}}
=
\mathbb E_{q_\phi(z\mid c,y)}
[
\log p_\theta(y\mid c,z)
]
-
D_{KL}
(
q_\phi(z\mid c,y)
\parallel
p_\theta(z\mid c)
)
\]

把它们对齐：

```text
VAE

target:
x

posterior:
q(z | x)

prior:
p(z)

decoder:
p(x | z)
```

```text
CVAE

condition:
c

target:
y

posterior:
q(z | c,y)

prior:
p(z | c)

decoder:
p(y | c,z)
```

所以 CVAE 不是一套完全新的理论。

它是：

> **Variational latent-variable modeling 在 conditional distribution \(p(y\mid c)\) 上的版本。**

---

# 19. Conditional ELBO 怎么推出来？

和 VAE 几乎一样。

我们人为乘除：

\[
q_\phi(z\mid c,y)
\]

从：

\[
\log p_\theta(y\mid c)
\]

出发：

\[
\log p_\theta(y\mid c)
=
\log
\int
p_\theta(y,z\mid c)\,dz
\]

插入：

\[
\frac{
q_\phi(z\mid c,y)
}{
q_\phi(z\mid c,y)
}
\]

得到：

\[
=
\log
\int
q_\phi(z\mid c,y)
\frac{
p_\theta(y,z\mid c)
}{
q_\phi(z\mid c,y)
}
dz
\]

写成 expectation：

\[
=
\log
\mathbb E_q
\left[
\frac{
p_\theta(y,z\mid c)
}{
q_\phi(z\mid c,y)
}
\right]
\]

利用 Jensen's inequality：

\[
\log\mathbb E[X]
\ge
\mathbb E[\log X]
\]

得到：

\[
\log p_\theta(y\mid c)
\ge
\mathbb E_q
\left[
\log
\frac{
p_\theta(y,z\mid c)
}{
q_\phi(z\mid c,y)
}
\right]
\]

因为：

\[
p_\theta(y,z\mid c)
=
p_\theta(y\mid c,z)
p_\theta(z\mid c)
\]

所以：

\[
=
\mathbb E_q[
\log p_\theta(y\mid c,z)
]
+
\mathbb E_q[
\log p_\theta(z\mid c)
-
\log q_\phi(z\mid c,y)
]
\]

第二项正好是：

\[
-
D_{KL}
(
q_\phi(z\mid c,y)
\parallel
p_\theta(z\mid c)
)
\]

于是：

\[
\boxed{
\mathcal L_{\text{CVAE}}
=
\mathbb E_q[
\log p_\theta(y\mid c,z)
]
-
D_{KL}
(
q_\phi(z\mid c,y)
\parallel
p_\theta(z\mid c)
)
}
\]

完成。

---

# 20. 两个 Term 分别在做什么？

## Conditional Reconstruction / Likelihood Term

\[
\mathbb E_q[
\log p_\theta(y\mid c,z)
]
\]

要求：

> 给定 condition \(c\) 和从 recognition posterior 中得到的 \(z\)，generator 应该能够很好地解释真实 output \(y\)。

也就是：

```text
c
+
z
↓
应该生成当前训练样本中的 y
```

---

## KL Term

\[
D_{KL}
(
q_\phi(z\mid c,y)
\parallel
p_\theta(z\mid c)
)
\]

要求：

> 训练时看到 ground-truth \(y\) 得到的 latent distribution，不要和测试时只能依赖 \(c\) 得到的 prior 相差太远。

这个作用在 CVAE 中比普通 VAE 更容易直觉理解。

因为：

```text
训练：
z 来自 q(z | c,y)

测试：
z 来自 p(z | c)
```

如果二者完全不一致：

> decoder 在训练时看到的 \(z\) 和测试时看到的 \(z\) 会来自完全不同的区域。

那 inference 就会出问题。

---

# 21. KL 在 CVAE 中可以理解成“让训练和测试的 z 接得上”

这是一种很好的直觉。

训练时：

\[
q_\phi(z\mid c,y)
\]

有一个巨大优势：

> 它偷看了 ground-truth \(y\)。

所以很容易找到一个 latent code 来重建 \(y\)。

但测试时：

\[
y
\]

不存在。

我们只能用：

\[
p_\theta(z\mid c)
\]

如果：

```text
q(z | c,y)
在左边

p(z | c)
在右边
```

两者毫无重叠，

decoder 训练时学到的 latent region 在测试时就采不到。

所以 KL：

\[
D_{KL}
(
q_\phi(z\mid c,y)
\parallel
p_\theta(z\mid c)
)
\]

在施加压力：

```text
training posterior
        ↓
不要离
        ↓
test-time prior
太远
```

但和 VAE 一样：

> 这不是要求它们无条件完全相等。

reconstruction 仍然要求 posterior 保留足够的信息来解释当前 \(y\)。

---

# 22. 这也是原论文特别讨论的问题

Sohn et al. 明确指出：

> CVAE training 使用 recognition network \(q_\phi(z\mid c,y)\)，而 testing 使用 prior \(p_\theta(z\mid c)\)。

因为 recognition model 能看到：

\[
y
\]

训练阶段其实更接近：

> “重建已经知道的 \(y\)”

而测试阶段才是真正：

> “根据 \(c\) 预测未知 \(y\)”。

因此二者存在 discrepancy。

论文中的 KL 项正是在尝试缩小这两条 pipeline 的 gap。

作者后来还提出 GSNN 和 hybrid objective 来进一步研究这个 train–test discrepancy。

对于理解 CVAE 原理，我们暂时不需要展开 GSNN。

但要记住：

> **training posterior 和 inference prior 不同，是 CVAE 的核心结构特征，不是实现细节。**

---

# 23. Fixed-Prior CVAE：很多现代模型更常见的版本

如果我们选择：

\[
p_\theta(z\mid c)
=
p(z)
\]

并使用：

\[
p(z)
=
\mathcal N(0,I)
\]

那么 CVAE ELBO 变成：

\[
\boxed{
\mathcal L_{\text{CVAE}}
=
\mathbb E_{
q_\phi(z\mid c,y)
}
[
\log p_\theta(y\mid c,z)
]
-
D_{KL}
\left(
q_\phi(z\mid c,y)
\parallel
p(z)
\right)
}
\]

这时候：

```text
训练：
q(z | c,y)

测试：
p(z) = N(0,I)
```

condition：

\[
c
\]

仍然进入 generator：

\[
p_\theta(y\mid c,z)
\]

但不再调制 prior。

这个版本和普通 VAE 的形式更像。

---

# 24. 所以“CVAE = 把 condition 拼到 encoder 和 decoder”准确吗？

只能算工程直觉。

例如最简单实现可能真的写：

```python
encoder_input = concat(c, y)
decoder_input = concat(c, z)
```

于是：

\[
q_\phi(z\mid c,y)
\]

由 encoder 实现，

\[
p_\theta(y\mid c,z)
\]

由 decoder 实现。

但真正定义它是 CVAE 的不是：

> `torch.cat`

而是概率模型：

\[
q_\phi(z\mid c,y)
\]

\[
p_\theta(y\mid c,z)
\]

以及 conditional objective：

\[
\log p_\theta(y\mid c)
\]

如果使用 conditional prior，还包括：

\[
p_\theta(z\mid c)
\]

所以：

> **“拼接 condition”只是实现概率条件关系的一种网络设计，不是 CVAE 的数学定义。**

---

# 25. 一个最小 Fixed-Prior CVAE

假设：

- condition：\(c\)
- target：\(y\)
- latent：\(z\)

training encoder：

\[
q_\phi(z\mid c,y)
\]

可以写成：

```python
encoder_input = torch.cat([c, y], dim=-1)

h = encoder(encoder_input)

mu = mu_head(h)
logvar = logvar_head(h)
```

然后：

```python
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
z = mu + std * eps
```

decoder：

```python
decoder_input = torch.cat([c, z], dim=-1)

y_hat = decoder(decoder_input)
```

KL：

```python
kl = -0.5 * torch.sum(
    1 + logvar - mu.pow(2) - logvar.exp()
)
```

如果 reconstruction 使用 squared error：

```python
recon = F.mse_loss(y_hat, y)
```

最终：

```python
loss = recon + beta * kl
```

其中：

\[
\beta
\]

是具体模型可能引入的 KL 权重。

注意：

> 这是一种常见工程实现，不是 Sohn et al. 原论文所有网络细节的复刻。

---

# 26. 推理时发生什么？

训练完成以后，我们只有 condition：

\[
c
\]

没有真实：

\[
y
\]

所以不能再运行：

\[
q_\phi(z\mid c,y)
\]

---

## 如果使用 Conditional Prior

采样：

\[
z\sim p_\theta(z\mid c)
\]

然后：

\[
y\sim p_\theta(y\mid c,z)
\]

多采几次：

\[
z_1,z_2,z_3
\]

可以得到：

\[
y_1,y_2,y_3
\]

多个可能输出。

---

## 如果使用 Fixed Prior

采样：

\[
z\sim p(z)
\]

例如：

\[
z\sim\mathcal N(0,I)
\]

再：

\[
y\sim p_\theta(y\mid c,z)
\]

同样可以获得多个输出。

---

# 27. 为什么 Condition 和 Latent 两个都需要？

这是理解 CVAE 非常关键的问题。

假设：

\[
c
\]

已经告诉模型：

> “现在是什么任务 / 环境 /上下文”。

那为什么还需要：

\[
z
\]

？

因为二者负责的信息不同。

可以直觉理解为：

### Condition \(c\)

告诉模型：

> **哪些输出是与当前上下文相容的？**

### Latent \(z\)

允许模型在这些相容输出之间表达：

> **还存在什么没有被 condition 唯一决定的变化？**

例如机器人：

```text
c:
当前相机画面 + 关节状态

z:
在这种状态下，
具体采用哪一种合理动作变化
```

所以：

\[
c
\]

并不是：

\[
z
\]

的替代品。

反过来也一样。

---

# 28. 一个条件分布的简单例子

假设：

\[
c=0
\]

时：

\[
y
\]

可能集中在：

\[
-2
\]

或：

\[
+2
\]

也就是双峰。

而：

\[
c=1
\]

时，

两个 mode 可能变成：

\[
5
\]

和：

\[
8
\]

那么 condition 决定：

> 整个输出分布处在什么上下文。

latent 可以决定：

> 当前 sample 落在哪个 mode 或 mode 内的哪个位置。

可以想象：

```text
c = 0:
      mode A       mode B
        ●             ●


c = 1:
                           mode A    mode B
                             ●         ●
```

因此：

\[
p(y\mid c)
\]

随着：

\[
c
\]

改变。

而 \(z\) 帮助描述每个条件分布内部的 variation。

---

# 29. CVAE 是不是专门为 Multimodality 设计的？

Sohn et al. 的核心动机之一确实是：

> 建模 structured output conditional distribution 中的 multiple modes，并生成 diverse predictions。

所以用 CVAE 来处理：

> one-to-many mapping

是非常典型的用途。

但仍然不要把它说成：

> “用了 CVAE 就一定能学到所有 mode。”

实际是否成功还受：

- latent dimension；
- posterior family；
- decoder capacity；
- KL weighting；
- optimization；
- data coverage；
- posterior collapse；

等因素影响。

CVAE 提供的是：

> **能够表达 stochastic conditional variation 的概率建模框架。**

不是 multimodality 的自动保证。

---

# 30. CVAE 中的 z 一定等于“style”吗？

**不一定。**

某些论文——包括 ACT——会把 latent \(z\) 解释为：

> style variable

这是对该模型用途的命名。

但 CVAE 的一般数学定义只要求：

\[
z
\]

是 conditional latent variable。

它可以用于表达：

- style；
- trajectory variation；
- shape；
- ambiguity；
- unobserved factors；
- stochastic output variation；

但具体语义是否可解释，需要由模型和实验决定。

所以：

> **“CVAE 里的 z 就是风格”不是普遍定义。**

---

# 31. CVAE Encoder 真正编码的是什么？

普通 VAE：

\[
q_\phi(z\mid x)
\]

可以说：

> 给定数据 \(x\)，推断解释这个数据的 latent posterior。

CVAE：

\[
q_\phi(z\mid c,y)
\]

更加准确地说：

> **给定 condition \(c\) 和这次真正发生的 output \(y\)，推断什么 latent \(z\) 能解释“为什么在这个 condition 下出现的是这个 \(y\)”**。

这个区别非常重要。

因为 \(z\) 不是单独描述：

\[
c
\]

也不是单独描述：

\[
y
\]

而是服务于：

\[
p(y\mid c)
\]

这个 conditional generation 问题。

---

# 32. 一个很好的问题：既然 Encoder 已经看到 y，模型会不会作弊？

某种意义上：

> training recognition model 确实拥有 test-time 没有的信息。

它看到：

\[
y
\]

以后，更容易找到能够 reconstruct \(y\) 的 latent code。

但这不是 bug。

Variational inference 本来就允许 approximate posterior 使用观测到的 target information 来推断 latent variable。

真正需要防止的是：

> training posterior 学到一个 test-time prior 根本无法产生的 latent space。

所以 KL：

\[
D_{KL}
(
q_\phi(z\mid c,y)
\parallel
p_\theta(z\mid c)
)
\]

非常重要。

Sohn et al. 论文还专门讨论了这种 train–test discrepancy。

---

# 33. Conditional Prior Network 是怎么工作的？

原论文的 CVAE 使用：

\[
p_\theta(z\mid c)
\]

如果 \(z\) 是 Gaussian：

\[
p_\theta(z\mid c)
=
\mathcal N
\left(
\mu_p(c),
\operatorname{diag}(\sigma_p^2(c))
\right)
\]

那么可以用一个网络：

```text
condition c
↓
prior network
↙          ↘
μₚ(c)      log σₚ²(c)
```

测试时：

\[
z
=
\mu_p(c)
+
\sigma_p(c)\odot\epsilon
\]

其中：

\[
\epsilon\sim\mathcal N(0,I)
\]

所以 conditional prior 本身也是一个可学习 distribution。

---

# 34. Recognition Network 同时看 c 和 y

如果：

\[
q_\phi(z\mid c,y)
\]

也是 Gaussian：

\[
q_\phi(z\mid c,y)
=
\mathcal N
\left(
\mu_q(c,y),
\operatorname{diag}(\sigma_q^2(c,y))
\right)
\]

则：

```text
condition c
+
target y
↓
recognition network
↙             ↘
μq(c,y)       log σq²(c,y)
```

然后：

\[
z
=
\mu_q(c,y)
+
\sigma_q(c,y)\odot\epsilon
\]

训练时使用这一条 path。

---

# 35. KL 现在比较的是两个 Gaussian

在 conditional-prior CVAE 中：

\[
D_{KL}
\left(
q_\phi(z\mid c,y)
\parallel
p_\theta(z\mid c)
\right)
\]

也就是：

```text
Recognition posterior:

N(μq(c,y), σq²(c,y))


          VS


Conditional prior:

N(μp(c), σp²(c))
```

不再一定是和：

\[
\mathcal N(0,I)
\]

比较。

所以普通 VAE 中大家熟悉的：

\[
\frac12
\sum_j
(
\mu_j^2+\sigma_j^2-\log\sigma_j^2-1
)
\]

只适用于：

\[
p(z)=\mathcal N(0,I)
\]

这个特殊 prior。

如果 prior 本身：

\[
p(z\mid c)
\]

具有不同的均值和方差，

需要使用两个 Gaussian 之间更一般的 KL 公式。

---

# 36. 这就是为什么要先看具体论文，再写 KL 公式

看到“CVAE”三个字以后直接写：

\[
KL(q(z\mid x,y)\parallel\mathcal N(0,I))
\]

不一定正确。

你必须先问：

> 这篇论文的 prior 到底是什么？

可能是：

\[
p(z)
=
\mathcal N(0,I)
\]

也可能是：

\[
p_\theta(z\mid c)
\]

甚至可能是更复杂的 distribution。

XunBlog 后面介绍任何具体 CVAE 模型，都应该遵守：

> **先看原论文的 graphical model / probability factorization，再决定公式。**

---

# 37. 现在终于可以和 ACT 对上了

ACT 论文说：

> policy 被训练为 CVAE，用 latent variable \(z\) 建模 human demonstrations 中的 variability。

在 ACT 的语境里，可以粗略对应：

### Condition \(c\)

当前 observation。

包括例如：

- robot joint positions；
- camera observations；

具体输入在 [CVAE in ACT](../robot-learning/act/cvae-in-act.md) 中详细拆。

---

### Output \(y\)

未来的：

\[
\text{action chunk}
\]

---

### Latent \(z\)

用于表达 demonstration 中的 latent style / variation。

---

所以 ACT 的目标可以从高层写成：

\[
p(
\text{action chunk}
\mid
\text{current observation}
)
\]

这是一个 conditional distribution。

这就是为什么 CVAE 很自然。

---

# 38. 但 ACT 和 Sohn 原始 CVAE 有一个重要不同

Sohn et al. 原论文的主要 CVAE formulation：

\[
p_\theta(z\mid c)
\]

是 conditional prior。

而 ACT 采用的是：

\[
\boxed{
p(z)=\mathcal N(0,I)
}
\]

也就是 fixed standard normal prior。

因此 ACT 更接近前面介绍的：

> **fixed-prior CVAE variant**

训练时：

\[
q_\phi(z\mid c,y)
\]

被 KL regularize toward：

\[
\mathcal N(0,I)
\]

而不是 toward 一个单独学习出来的：

\[
p_\theta(z\mid c)
\]

这就是为什么：

> 不能把 Sohn CVAE 的网络图机械复制到 ACT。

我们需要看 ACT 自己的原论文。

---

# 39. ACT 甚至还做了更特别的推理选择

一般 fixed-prior CVAE 在 inference 时可以：

\[
z\sim\mathcal N(0,I)
\]

采不同 \(z\) 得到多样输出。

但 ACT 推理时没有这样做。

ACT 直接使用：

\[
\boxed{
z=0
}
\]

也就是 standard normal prior 的均值。

因此 ACT inference 是：

> deterministic decoding from the prior mean

而不是不断随机 sample 不同动作风格。

为什么这样设计、为什么还能工作，会在：

- [为什么 ACT 推理时令 z = 0？](../robot-learning/act/why-z-zero-at-inference.md)

单独讨论。

这个行为是：

> **ACT-specific inference choice**

不是 CVAE 的一般定义。

---

# 40. 一个很容易出现的错误理解

有人会说：

> “CVAE 在训练时学到了各种风格；推理时随机采一个 \(z\)，选一个风格。”

作为一般 CVAE intuition：

> 有一定道理。

但拿来描述 ACT：

> **不准确。**

因为 ACT 推理时：

\[
z=0
\]

并没有随机采样一个不同风格。

所以以后要严格区分：

```text
General CVAE
```

和：

```text
ACT's use of CVAE
```

---

# 41. CVAE 和 Conditional Autoencoder 有什么区别？

普通 Conditional Autoencoder 可以是：

\[
z=f_\phi(c,y)
\]

然后：

\[
\hat y=g_\theta(c,z)
\]

其中 \(z\) 是 deterministic。

CVAE 则：

\[
q_\phi(z\mid c,y)
\]

是 probability distribution。

例如：

\[
\mathcal N(\mu,\sigma^2)
\]

然后：

\[
z\sim q_\phi(z\mid c,y)
\]

同时存在：

- prior；
- KL；
- variational objective；
- stochastic latent generation。

所以：

> **有 condition + encoder/decoder，不代表就是 CVAE。**

Variational probabilistic structure 才是关键。

---

# 42. CVAE 和 VAE 的最小区别可以怎么记？

如果只想先记一个版本：

## VAE

\[
\boxed{
q(z\mid y)
}
\]

\[
\boxed{
p(y\mid z)
}
\]

---

## CVAE

把 condition \(c\) 提供给两边：

\[
\boxed{
q(z\mid c,y)
}
\]

\[
\boxed{
p(y\mid c,z)
}
\]

如果采用 conditional prior，再加：

\[
\boxed{
p(z\mid c)
}
\]

如果采用 fixed prior：

\[
p(z)
\]

即可。

这比记：

> “VAE 加 condition”

更准确，因为你知道 condition 加进的是哪些 probability distributions。

---

# 43. 一个完整 Training Data Flow

以 Gaussian posterior 为例：

```text
             condition c
                 │
                 │
        ┌────────┴─────────┐
        │                  │
        ▼                  ▼
ground-truth y        generation path
        │                  ▲
        └──────┐           │
               ▼           │
        Recognition        │
           Network         │
       qφ(z | c,y)         │
               │           │
        ┌──────┴──────┐    │
        ▼             ▼    │
       μq          logσq²   │
        │             │    │
        └──────┬──────┘    │
               │           │
          ε ~ N(0,I)       │
               │           │
               ▼           │
          z = μq+σqε       │
               │           │
               └──────┬────┘
                      ▼
             pθ(y | c,z)
                      │
                      ▼
                reconstruct y
```

同时计算：

\[
KL(
q_\phi(z\mid c,y)
\parallel
p_\theta(z\mid c)
)
\]

或 fixed-prior 版本：

\[
KL(
q_\phi(z\mid c,y)
\parallel
p(z)
)
\]

---

# 44. 一个完整 Inference Data Flow

## Conditional-prior CVAE

```text
condition c
    │
    ├───────────────┐
    │               │
    ▼               │
Prior Network       │
pθ(z | c)           │
    │               │
    ▼               │
sample z            │
    │               │
    └───────┬───────┘
            ▼
       pθ(y | c,z)
            │
            ▼
            y
```

注意：

> recognition network 不再出现。

因为：

\[
y
\]

还不知道。

---

# 45. 为什么 Training Encoder 在 Inference 时可以被丢掉？

因为：

\[
q_\phi(z\mid c,y)
\]

从一开始就不是 generative model 的必要组成部分。

它是：

> **用于近似难算 posterior、帮助训练 generative model 的 inference network。**

真正生成模型是：

\[
p_\theta(z\mid c)
p_\theta(y\mid c,z)
\]

或者 fixed-prior：

\[
p(z)p_\theta(y\mid c,z)
\]

因此模型训练完成后，真正生成数据时：

> 可以不需要 recognition network。

这和 VAE 中：

> generation 只需要 prior + decoder

完全一致。

---

# 46. 所以“Encoder 和 Decoder 是一对必须永远一起使用的网络”是错的

在 VAE / CVAE 语境下：

Encoder / recognition model：

\[
q_\phi
\]

主要负责：

> posterior inference。

Generator / decoder：

\[
p_\theta
\]

负责：

> generative process。

训练时两者一起出现。

生成时：

> recognition model 可以完全不参与。

这一点对 ACT 特别重要。

ACT 推理时 CVAE encoder 被丢弃，并不是奇怪 hack。

它符合 variational latent-variable model 中：

> inference network 与 generative model 角色不同

这一基本结构。

---

# 47. 常见误解一：CVAE 就是 VAE 多拼一个 condition

**不够准确。**

工程上可能通过 concatenation 实现。

数学上改变的是：

\[
p(y)
\rightarrow
p(y\mid c)
\]

以及相应的：

\[
q(z\mid c,y)
\]

\[
p(y\mid c,z)
\]

并且可能还有：

\[
p(z\mid c)
\]

---

# 48. 常见误解二：所有 CVAE 的 prior 都是 N(0,I)

**错误。**

Sohn et al. 的主要 formulation 使用：

\[
p_\theta(z\mid c)
\]

conditional prior。

论文明确说明也可以放宽为：

\[
p(z)
\]

独立 prior。

所以必须看具体模型。

---

# 49. 常见误解三：所有 CVAE 的 prior 都必须依赖 condition

同样错误。

Fixed-prior CVAE 完全成立。

ACT 就使用：

\[
p(z)=\mathcal N(0,I)
\]

---

# 50. 常见误解四：Training 和 Testing 都从 q(z|c,y) 采 z

**错误。**

testing 时真实：

\[
y
\]

未知。

所以不能使用：

\[
q(z\mid c,y)
\]

一般必须使用：

\[
p(z\mid c)
\]

或：

\[
p(z)
\]

---

# 51. 常见误解五：KL 只是让 latent space 更漂亮

**太浅。**

CVAE 中 KL 有非常具体的概率学作用：

\[
q_\phi(z\mid c,y)
\]

是 training-time recognition posterior。

\[
p_\theta(z\mid c)
\]

是 generative prior。

KL 让二者不要完全脱节，从而使 test-time sampling 有意义。

---

# 52. 常见误解六：z 一定对应明确的人类“风格”

**没有保证。**

\(z\) 是 latent random variable。

是否具有可解释语义要靠具体实验验证。

---

# 53. 常见误解七：同一个 condition 采不同 z，一定产生不同输出

也不保证。

如果模型发生：

> posterior collapse

或者 decoder 几乎忽略：

\[
z
\]

那么：

\[
z_1\neq z_2
\]

可能仍然产生几乎相同的：

\[
y
\]

所以存在 latent variable 不等于模型一定真正利用它。

---

# 54. 常见误解八：CVAE 的 Encoder 在做普通压缩

不准确。

Recognition model：

\[
q_\phi(z\mid c,y)
\]

是在进行：

> **approximate posterior inference**

它输出的是 distribution parameters。

不是简单把：

\[
(c,y)
\]

压缩成一个 deterministic vector。

---

# 55. 用四个 Distribution 彻底记住 CVAE

如果学习一个 conditional-prior CVAE，只需要牢牢记住四个对象。

---

## 1. 我们真正想建模

\[
\boxed{
p_\theta(y\mid c)
}
\]

---

## 2. Generative distribution

\[
\boxed{
p_\theta(y\mid c,z)
}
\]

---

## 3. Prior

\[
\boxed{
p_\theta(z\mid c)
}
\]

---

## 4. Approximate posterior

\[
\boxed{
q_\phi(z\mid c,y)
}
\]

然后 ELBO：

\[
\boxed{
\mathcal L_{\text{CVAE}}
=
\mathbb E_q[
\log p_\theta(y\mid c,z)
]
-
D_{KL}
(
q_\phi(z\mid c,y)
\parallel
p_\theta(z\mid c)
)
}
\]

这就是 CVAE 的核心骨架。

---

# 56. 如果是 Fixed-Prior CVAE，只改一个地方

把：

\[
p_\theta(z\mid c)
\]

换成：

\[
p(z)
\]

于是：

\[
\boxed{
\mathcal L
=
\mathbb E_q[
\log p_\theta(y\mid c,z)
]
-
D_{KL}
(
q_\phi(z\mid c,y)
\parallel
p(z)
)
}
\]

其他核心思想保持不变。

这就是后面理解 ACT 最有用的版本。

---

# 57. 一句话重新理解 CVAE

> **CVAE 的本质，是用 latent variable \(z\) 来建模“给定 condition \(c\) 后仍然没有被唯一确定的输出变化”，从而学习完整的 conditional distribution \(p(y\mid c)\)，而不是只学习一个确定映射 \(y=f(c)\)。**

训练时：

\[
q_\phi(z\mid c,y)
\]

可以利用真实 \(y\) 来推断这条 sample 的 latent explanation。

推理时真实 \(y\) 不存在，所以：

\[
z
\]

必须来自 generative prior：

\[
p_\theta(z\mid c)
\]

或者：

\[
p(z)
\]

然后 generator：

\[
p_\theta(y\mid c,z)
\]

产生输出。

KL 的重要作用之一，就是让：

> training-time posterior

和：

> test-time prior

能够接得起来。

---

# 58. 下一步：CVAE 到底怎样进入 ACT？

现在我们已经理解一般 CVAE。

下一步不能再泛泛谈：

```text
condition
output
latent
```

而要逐个对应 ACT 原论文：

> ACT 的 \(c\) 到底是什么？

> ACT 的 \(y\) 到底是什么？

> CVAE encoder 具体输入哪些 token？

> 为什么 action sequence 会进入 encoder？

> `[CLS]` token 在里面做什么？

> \(\mu\) 和 \(\log\sigma^2\) 从哪里得到？

> ACT 的 prior 为什么是：

\[
\mathcal N(0,I)
\]

> 为什么训练时 sample \(z\)，推理时却：

\[
z=0
\]

> CVAE decoder 和 ACT Transformer policy 到底是什么关系？

这些问题全部属于下一篇：

> [CVAE in ACT：z 到底是怎么产生和使用的？](../robot-learning/act/cvae-in-act.md)

从下一篇开始，我们会直接回到 ACT 原论文网络结构，而不再停留在一般 CVAE。

---

## Primary Source

Kihyuk Sohn, Xinchen Yan, Honglak Lee.  
**Learning Structured Output Representation using Deep Conditional Generative Models.**  
Advances in Neural Information Processing Systems 28 (NIPS / NeurIPS), 2015.

- Paper: https://papers.neurips.cc/paper/5775-learning-structured-output-representation-using-deepconditional-generative-models
- PDF: https://papers.neurips.cc/paper_files/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf

本文主要依据原论文：

- Section 1 — Introduction
- Section 3 — Preliminary: Variational Auto-encoder
- Section 4 — Deep Conditional Generative Models for Structured Output Prediction
- Equation (4) — Conditional variational lower bound
- Equation (5) — Empirical CVAE objective
- Figure 1 — Conditional graphical models
- Section 4.1 — Output inference and conditional likelihood
- Section 4.2 — Learning to predict structured output
- Section 5.1 — MNIST one-to-many toy experiment

原论文使用：

\[
x=\text{input condition},
\qquad
y=\text{output},
\qquad
z=\text{latent}
\]

本文为了避免与 VAE 文章中的 observed variable \(x\) 混淆，将原论文的 condition \(x\) 改记为：

\[
c
\]

因此本文：

\[
p_\theta(z\mid c)
\]

对应原论文：

\[
p_\theta(z\mid x)
\]

本文：

\[
q_\phi(z\mid c,y)
\]

对应原论文：

\[
q_\phi(z\mid x,y)
\]

---

## Background Source

Diederik P. Kingma, Max Welling.  
**Auto-Encoding Variational Bayes.**  
ICLR 2014.

- Paper: https://arxiv.org/abs/1312.6114

CVAE 的 variational learning framework 建立在 VAE / SGVB 的基础之上。

---

## 与 ACT 的关系

Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.  
**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.**  
Robotics: Science and Systems (RSS), 2023.

- Paper: https://arxiv.org/abs/2304.13705

ACT 使用 CVAE 建模 demonstration variability，但其 prior 采用：

\[
p(z)=\mathcal N(0,I)
\]

而不是 Sohn et al. 主 formulation 中单独学习的：

\[
p_\theta(z\mid c)
\]

因此 ACT 属于本文讨论的 fixed-prior conditional VAE 形式。

---

## 本文知识连接

### 前置知识

- [Latent Variable](./latent-variable.md)
- [VAE](./vae.md)
- [Reparameterization Trick](./reparameterization-trick.md)
- Conditional Probability
- Bayes' Rule
- Marginalization

### 数学核心

- Expectation
- [KL Divergence](../mathematics/kl-divergence.md)
- [Normal Distribution](../mathematics/normal-distribution.md)
- Standard Normal Distribution

### 下一步

- [CVAE in ACT](../robot-learning/act/cvae-in-act.md)
- [为什么 ACT 推理时令 z = 0？](../robot-learning/act/why-z-zero-at-inference.md)

### Robot Learning

- [ACT 到底解决了什么问题？](../robot-learning/act/act-what-problem-does-it-solve.md)
- [Action Chunking](../robot-learning/act/action-chunking.md)
- [Temporal Ensemble](../robot-learning/act/temporal-ensemble.md)
