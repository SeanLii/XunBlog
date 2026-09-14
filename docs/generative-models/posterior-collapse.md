---
title: "Posterior Collapse：CVAE 明明有 z，Decoder 为什么可能完全不理它？"
description: "从 VAE/CVAE 的 ELBO、KL、mutual information 与 rate–distortion 视角理解 posterior collapse：为什么强 Decoder 可能忽略 latent z、KL≈0 到底意味着什么、β 如何改变 latent usage，以及如何诊断 ACT 的 32-D style latent 是否真的被使用。"
status: reviewed
pageType: concept
canonical: /generative-models/posterior-collapse
updated: "2026-09-15"
---

# Posterior Collapse：CVAE 明明有 \(z\)，Decoder 为什么可能完全不理它？

前面我们已经知道 ACT 的 CVAE 训练流程：

\[
[\text{current qpos},\text{future action chunk}]
\rightarrow
\text{CVAE Encoder}
\rightarrow
\mu,\log\sigma^2
\]

然后：

\[
z
=
\mu+\sigma\epsilon
\]

再把：

\[
z
\]

和：

- 当前 images；
- 当前 joints；

一起输入 policy：

\[
\boxed{
\hat a_{t:t+k-1}
=
f_\theta(
o_t,
z
)
}
\]

训练目标：

\[
\boxed{
L
=
L_{\text{recon}}
+
\beta L_{\text{KL}}
}
\]

ACT 官方实现中：

\[
L_{\text{recon}}
=
L_1
\]

并且 canonical：

\[
\beta=10
\]

到这里我们很容易产生一个直觉：

> 既然网络里专门有一个 \(z\)，而且还花了一整个 CVAE Encoder 去预测 \(\mu\) 和 \(\log\sigma^2\)，那 \(z\) 肯定学到了 demonstration style。

但这句话：

\[
\boxed{
\text{不一定成立。}
}
\]

一个 VAE/CVAE 可以：

- 有 Encoder；
- 有 \(\mu\)；
- 有 \(\log\sigma^2\)；
- 有 reparameterization；
- 有 latent tensor；
- 有 KL loss；

甚至 training loss 看起来也很正常，

但 Decoder 最终可能学会：

\[
\boxed{
\text{几乎完全忽略 }z
}
\]

这就是：

\[
\boxed{
\text{Posterior Collapse}
}
\]

后验坍缩。

这是理解 VAE、CVAE，以及 ACT 中 style latent 真正意义时非常重要的一层。

这篇会回答：

1. 什么叫 posterior collapse？
2. “posterior collapse”到底是哪一个 posterior collapsed？
3. 为什么：
   \[
   q_\phi(z|x)\approx p(z)
   \]
   意味着 latent 几乎不携带 input information？
4. 为什么 collapse 时 variance 不是趋近 0，而往往是趋近 prior variance 1？
5. 为什么一个很强的 Decoder 反而可能更容易忽略 \(z\)？
6. KL 为什么一方面让 latent space可采样，一方面又会“惩罚信息”？
7. Mutual Information：
   \[
   I(X;Z)
   \]
   和 KL 到底什么关系？
8. CVAE 中应该看：
   \[
   I(Y;Z|C)
   \]
   而不是简单 \(I(X;Z)\) 吗？
9. 什么是 aggregated posterior？
10. Rate–Distortion 视角是什么？
11. KL 很小一定代表 collapse 吗？
12. KL 很大一定代表 latent 很有用吗？
13. 某些 latent dimensions collapse 和整个 latent collapse有什么区别？
14. 什么叫 Active Units？
15. \(\beta\) 太大为什么可能压死 \(z\)？
16. \(\beta\) 太小又为什么会造成 prior mismatch？
17. KL annealing为什么有用？
18. Free Bits 是什么？
19. 为什么 weakening decoder有时可以逼 decoder 使用 \(z\)？
20. “强 Decoder 导致 collapse”是不是完整原因？
21. Lagging Inference Networks 提出了什么不同解释？
22. ACT 的 Decoder 会不会发生 posterior collapse？
23. ACT 的 \(z=0\) inference 和 collapse是什么关系？
24. 如果 ACT 已经 collapse，\(z=0\) 还有意义吗？
25. ACT 的 CVAE ablation 从 human-data success 35.3% 降到 2% 到底能证明什么？
26. 它又不能证明什么？
27. 怎么真正实验诊断 ACT 的 32-D latent 是否被使用？
28. 怎样画 per-dimension KL？
29. 怎样做 latent sensitivity test？
30. 怎样做 posterior-vs-zero ablation？
31. 怎样做 latent swapping？
32. 怎样估计 latent usage，而不是只看总 loss？
33. 为什么“有一个 latent variable”不等于“学到了可解释因素”？

---

# 1. 先回到 VAE 最基本的目标

普通 VAE：

\[
p_\theta(x,z)
=
p(z)p_\theta(x|z)
\]

我们希望：

\[
z
\]

解释：

> \(x\) 中的重要 variation。

但真正 posterior：

\[
p_\theta(z|x)
\]

通常难算。

所以引入：

\[
q_\phi(z|x)
\]

作为 approximate posterior。

---

# 2. ELBO

经典：

\[
\boxed{
\log p_\theta(x)
\ge
\mathbb E_{q_\phi(z|x)}
[
\log p_\theta(x|z)
]
-
D_{KL}
(
q_\phi(z|x)
\|
p(z)
)
}
\]

最大化 ELBO。

如果写成最小化 loss：

\[
\boxed{
L
=
L_{\text{reconstruction}}
+
L_{\text{KL}}
}
\]

---

# 3. 两项的目标其实在“拉扯”

Reconstruction 项希望：

> \(z\) 尽可能携带对重建有帮助的信息。

KL 项希望：

\[
q_\phi(z|x)
\]

不要离 prior：

\[
p(z)
\]

太远。

所以：

\[
\boxed{
\text{information usefulness}
\leftrightarrow
\text{prior regularization}
}
\]

之间存在 tradeoff。

---

# 4. 为什么需要 KL？

如果完全没有 KL，

Encoder可以把每个：

\[
x
\]

映射到非常任意、互不相关的位置。

Decoder可以很好重建，

但训练后从：

\[
z\sim\mathcal N(0,I)
\]

采样时：

> 可能落到 Encoder从没使用过的 latent regions。

所以 VAE 希望 posterior latent space：

> 和 prior兼容。

---

# 5. 但 KL 有一个副作用

如果：

\[
q_\phi(z|x)
\]

携带很多关于：

\[
x
\]

的信息，

通常它就必须：

> 根据不同 \(x\) 发生变化。

例如：

\[
x_1
\rightarrow
q(z|x_1)
\]

和：

\[
x_2
\rightarrow
q(z|x_2)
\]

不同。

---

# 6. 可是 KL 在不断说

每个：

\[
q(z|x)
\]

都最好靠近：

\[
p(z)
\]

如果最极端：

\[
\boxed{
q_\phi(z|x)
=
p(z)
\quad
\forall x
}
\]

那么：

\[
D_{KL}=0
\]

达到 KL 项的最小值。

---

# 7. 但此时发生了什么？

如果所有 input：

\[
x
\]

得到完全相同的 latent distribution：

\[
q(z|x)=p(z)
\]

那么知道：

\[
z
\]

几乎不能告诉你：

> 这是哪个 \(x\)。

也就是说：

\[
\boxed{
z
\text{ 不再携带 }x\text{ 的信息}
}
\]

---

# 8. 这就是 Posterior Collapse 的核心状态

最典型定义：

\[
\boxed{
q_\phi(z|x)
\approx
p(z)
}
\]

同时 Decoder：

\[
p_\theta(x|z)
\]

实际上：

> 对 \(z\) 很不敏感，甚至完全忽略它。

---

# 9. 为什么叫 Posterior Collapse？

因为本来：

\[
q_\phi(z|x)
\]

应该随着：

\[
x
\]

变化。

collapse后：

\[
q_\phi(z|x)
\]

全部“塌”到：

\[
p(z)
\]

附近。

于是：

```text
x₁ ─┐
x₂ ─┤
x₃ ─┤→ same posterior ≈ prior
x₄ ─┘
```

input-specific posterior structure消失。

---

# 10. 一个非常重要的纠错：Collapse 不是 variance → 0

很多二手解释会说：

> “posterior collapse就是 latent variance变成0。”

这是：

\[
\boxed{
\text{错误的。}
}
\]

---

# 11. 标准 Gaussian VAE 的 Prior

\[
p(z)
=
\mathcal N(0,I)
\]

如果完全 collapse：

\[
q(z|x)
=
\mathcal N(0,I)
\]

那么：

\[
\boxed{
\mu(x)\rightarrow0
}
\]

以及：

\[
\boxed{
\sigma^2(x)\rightarrow1
}
\]

---

# 12. 用 log-variance 表示

如果：

\[
\log\sigma^2=0
\]

那么：

\[
\sigma^2
=
e^0
=
1
\]

所以 collapsed posterior常见形式：

\[
\boxed{
\mu\approx0,
\qquad
\logvar\approx0
}
\]

不是：

\[
\sigma^2\approx0
\]

---

# 13. Variance → 0 是另一种现象

如果：

\[
\sigma^2\rightarrow0
\]

那么 posterior：

\[
q(z|x)
\]

反而会变得：

> 非常尖锐、接近 deterministic encoding。

这可能导致：

> KL变大，

而不是标准 posterior collapse。

---

# 14. 所以一定区分

### Posterior Collapse

\[
q(z|x)
\rightarrow
p(z)
=
\mathcal N(0,I)
\]

因此：

\[
\mu\rightarrow0,
\quad
\sigma^2\rightarrow1
\]

---

### Deterministic / Variance Collapse

\[
\sigma^2\rightarrow0
\]

Posterior变得：

> 极度确定。

这不是同一问题。

---

# 15. 为什么 Decoder 会愿意忽略 z？

设 VAE Decoder：

\[
p_\theta(x|z)
\]

如果 Decoder能力很强，

它可能发现：

> 即使不读取 \(z\)，也可以把数据分布建模得不错。

于是：

\[
p_\theta(x|z)
\approx
p_\theta(x)
\]

---

# 16. 如果 Decoder 不需要 z

那 Encoder再往：

\[
z
\]

里塞信息有什么好处？

Reconstruction：

> 几乎没改善。

但 KL：

> 会增加。

所以 optimizer最划算的选择：

\[
\boxed{
q(z|x)=p(z)
}
\]

KL直接：

\[
0
\]

---

# 17. 这就是经典“Powerful Decoder”解释

Bowman et al. 在 sentence VAE 中观察到：

> 强 RNN decoder可以优先学会解释数据，而不依赖全局 latent。

一旦进入这种状态：

- Decoder忽略 Encoder；
- KL降到0；
- Encoder获得的有用 reconstruction gradient减少；

形成一个稳定的坏平衡。

---

# 18. 一个极端例子

假设 Decoder能直接看到完整 target：

\[
x
\]

那么它根本不需要：

\[
z
\]

即可完美输出：

\[
x
\]

此时最优：

\[
q(z|x)=p(z)
\]

因为：

- recon完美；
- KL=0。

---

# 19. 现实中 Decoder 不会直接看到 target

但它可能有很多其他强信息。

例如 language VAE：

Decoder每一步看到：

> ground-truth previous words。

于是局部语言上下文已经足够强。

Global latent：

\[
z
\]

变得可有可无。

---

# 20. CVAE 中这个问题更明显

CVAE：

\[
p_\theta(y|c,z)
\]

其中：

\[
c
\]

是 condition。

如果 condition：

\[
c
\]

已经足够预测：

\[
y
\]

那么：

\[
z
\]

就更加容易被忽略。

---

# 21. ACT 正是 CVAE

ACT 可以写成：

\[
\boxed{
p_\theta(
A
|
O,z
)
}
\]

其中：

\[
O
\]

代表当前：

- images；
- qpos。

\[
A
\]

代表：

> future action chunk。

---

# 22. 如果当前 Observation 已经足够预测 Action Chunk

那么 Decoder可以学：

\[
\boxed{
p_\theta(A|O,z)
\approx
p_\theta(A|O)
}
\]

此时：

> \(z\) 没必要。

---

# 23. 这就是 ACT 可能发生 Collapse 的理论入口

ACT 的 policy decoder很强：

- ResNet18；
- Transformer Encoder；
- Transformer Decoder；
- large FFN；
- current observation。

如果这些信息足以决定 action chunk，

模型理论上完全可能：

> 忽略 latent input。

---

# 24. 但 Human Demonstrations 为什么让 z 更有机会被使用？

因为同一个或非常相近的：

\[
O
\]

可能对应多个：

\[
A
\]

例如人类 demonstration存在：

- 不同微小姿势；
- pause timing；
- trajectory style；
- recovery variation；
- 操作习惯。

于是：

\[
O
\]

不能完全解释：

\[
A
\]

---

# 25. 这时 latent 有一个明确用途

Encoder训练时看到：

\[
(O,A)
\]

可以把：

> 当前 observation无法解释、但 action chunk中存在的 variation

编码进：

\[
z
\]

然后 decoder：

\[
p(A|O,z)
\]

利用它重建对应 trajectory。

---

# 26. 所以 ACT 中真正希望 z 编码什么？

理想上：

\[
\boxed{
\text{action variation not already determined by current observation}
}
\]

可以粗略叫：

> style / demonstration variation。

但仍然不能说：

> 一定是“睡眠状态”“操作者力量”“策略意图”等具体因果变量。

---

# 27. 现在进入 Mutual Information

如果我们想严谨表达：

> “z 是否携带 x 的信息”

最自然的量是：

\[
\boxed{
I(X;Z)
}
\]

Mutual Information。

---

# 28. Mutual Information 的定义

\[
\boxed{
I(X;Z)
=
D_{KL}
(
q(x,z)
\|
q(x)q(z)
)
}
\]

等价：

\[
\boxed{
I(X;Z)
=
\mathbb E_{q(x)}
[
D_{KL}
(
q(z|x)
\|
q(z)
)
]
}
\]

---

# 29. 如果 X 和 Z 独立

\[
q(z|x)=q(z)
\]

则：

\[
\boxed{
I(X;Z)=0
}
\]

所以知道：

\[
z
\]

不会减少对：

\[
x
\]

的不确定性。

---

# 30. Posterior Collapse 对 Mutual Information 意味着什么？

如果：

\[
q(z|x)=p(z)
\]

对所有：

\[
x
\]

成立，

那么 aggregated posterior：

\[
q(z)=p(z)
\]

因此：

\[
\boxed{
I(X;Z)=0
}
\]

---

# 31. 这比“KL 变小”更本质

Posterior Collapse真正可怕的地方不是：

> KL这个数变成0。

而是：

\[
\boxed{
\text{latent stops carrying information about the data}
}
\]

---

# 32. 为什么 Expected KL 和 Mutual Information 有关系？

考虑：

\[
R
=
\mathbb E_{q(x)}
[
D_{KL}
(
q(z|x)
\|
p(z)
)
]
\]

把：

\[
q(z)
\]

插进去，可以得到：

\[
\boxed{
R
=
I(X;Z)
+
D_{KL}
(
q(z)
\|
p(z)
)
}
\]

---

# 33. 推导第一步

\[
R
=
\mathbb E_{q(x,z)}
\left[
\log
\frac{q(z|x)}{p(z)}
\right]
\]

乘除：

\[
q(z)
\]

：

\[
=
\mathbb E
\left[
\log
\frac{q(z|x)}{q(z)}
+
\log
\frac{q(z)}{p(z)}
\right]
\]

---

# 34. 第一项

\[
\mathbb E
\left[
\log
\frac{q(z|x)}{q(z)}
\right]
=
\boxed{
I(X;Z)
}
\]

---

# 35. 第二项

对：

\[
x
\]

积分掉后：

\[
\boxed{
D_{KL}
(
q(z)
\|
p(z)
)
}
\]

所以：

\[
\boxed{
\mathbb E_x KL(q(z|x)\|p(z))
=
I(X;Z)
+
KL(q(z)\|p(z))
}
\]

---

# 36. 这条公式非常重要

因为右边两项都：

\[
\ge0
\]

所以：

\[
\boxed{
I(X;Z)
\le
\mathbb E_x
KL(
q(z|x)\|p(z)
)
}
\]

---

# 37. 所以如果 Average KL → 0

那么必然：

\[
\boxed{
I(X;Z)\rightarrow0
}
\]

以及：

\[
q(z)\rightarrow p(z)
\]

也就是说：

> latent确实几乎不携带 input-specific information。

---

# 38. 这也是为什么 KL 是 Collapse Diagnostic

如果训练后：

\[
KL\approx0
\]

尤其每个维度都接近0，

这是非常强的警告：

\[
\boxed{
z\text{ 可能几乎没被使用}
}
\]

---

# 39. 但反过来不成立

如果：

\[
KL>0
\]

不能直接得出：

\[
\boxed{
z\text{ 一定对 Decoder 有用}
}
\]

为什么？

因为 KL包含两个部分：

\[
I(X;Z)
\]

和：

\[
KL(q(z)\|p(z))
\]

---

# 40. KL 可以大，但只是 Aggregated Posterior 没对齐 Prior

也就是说：

\[
q(z)
\]

整体偏离：

\[
p(z)
\]

很多，

但不同：

\[
x
\]

之间可能没有携带很多区分信息。

所以：

\[
\boxed{
KL\text{ nonzero}
\not\Rightarrow
\text{useful latent}
}
\]

---

# 41. CVAE 中真正关心的是 Conditional Mutual Information

ACT的 latent不是单独解释：

\[
A
\]

而是解释：

> 已经知道 observation \(O\) 以后，action chunk中剩余的 variation。

所以更自然的量：

\[
\boxed{
I(A;Z|O)
}
\]

---

# 42. Conditional Mutual Information 在这里意味着什么？

它问：

> 已经知道当前 observation \(O\) 后，再知道 \(Z\)，还能额外获得多少关于 future action chunk \(A\) 的信息？

如果：

\[
I(A;Z|O)=0
\]

意味着：

> 给定 observation后，latent不再提供额外 action information。

---

# 43. 这正是 CVAE Collapse 的理想数学定义之一

如果：

\[
\boxed{
q_\phi(z|O,A)
\approx
p(z|O)
}
\]

而 Decoder也：

\[
p_\theta(A|O,z)
\approx
p_\theta(A|O)
\]

那么 latent branch基本失效。

---

# 44. ACT 的 Prior 更简单

ACT 使用固定：

\[
\boxed{
p(z)
=
\mathcal N(0,I)
}
\]

而不是 learned conditional prior：

\[
p(z|O)
\]

所以 collapse趋向：

\[
\boxed{
q_\phi(z|O,A)
\approx
\mathcal N(0,I)
}
\]

---

# 45. ACT 的 Conditional KL Decomposition

定义 aggregated posterior：

\[
q(z|O)
=
\mathbb E_{A\sim p_{\text{data}}(A|O)}
[
q(z|O,A)
]
\]

那么：

\[
\boxed{
\mathbb E_{O,A}
[
KL(
q(z|O,A)
\|
p(z)
)
]
}
\]

可以分解为：

\[
\boxed{
I(A;Z|O)
+
\mathbb E_O
[
KL(
q(z|O)
\|
p(z)
)
]
}
\]

---

# 46. 这对 ACT 意义非常大

ACT 的 KL loss同时在做两件事：

### 1. 惩罚 Conditional Information Rate

\[
I(A;Z|O)
\]

也就是：

> action chunk通过 latent传递多少额外信息。

### 2. 让 Aggregated Posterior 靠近 Prior

\[
q(z|O)
\approx
p(z)
\]

以便 test-time prior latent有意义。

---

# 47. 所以 KL 本质上确实在“收费”

如果 Encoder想通过：

\[
z
\]

告诉 Decoder很多关于：

\[
A
\]

的信息，

通常就要支付：

\[
\boxed{
KL\text{ cost}
}
\]

---

# 48. 这就是 Rate–Distortion 视角

Alemi et al. 把 VAE训练解释成：

\[
\boxed{
\text{Rate}
\leftrightarrow
\text{Distortion}
}
\]

tradeoff。

---

# 49. Distortion 是什么？

Distortion大致代表：

> reconstruction有多差。

例如：

\[
D
=
-\mathbb E
[
\log p_\theta(x|z)
]
\]

ACT里可类比：

\[
\boxed{
D
\sim
L_1
}
\]

---

# 50. Rate 是什么？

典型 VAE中：

\[
\boxed{
R
=
\mathbb E
[
KL(
q(z|x)
\|
p(z)
)
]
}
\]

它与：

> latent传递的信息量

密切相关。

---

# 51. 所以 Objective 可以粗略理解为

\[
\boxed{
L
=
D+\beta R
}
\]

---

# 52. β 大时

信息变贵。

Encoder每多通过：

\[
z
\]

传一点信息，

都要付更多：

\[
\beta KL
\]

代价。

于是模型倾向：

\[
\boxed{
\text{更低 Rate}
}
\]

---

# 53. β 极大时可能怎样？

最便宜：

\[
R\rightarrow0
\]

也就是：

\[
q(z|x)\rightarrow p(z)
\]

如果 Decoder还能维持 acceptable reconstruction：

> collapse非常诱人。

---

# 54. β 小时

latent information变便宜。

Encoder可以更自由地：

> 把 input/target information塞进 z。

所以 reconstruction可能改善。

---

# 55. 但 β 太小也有问题

如果几乎：

\[
\beta=0
\]

模型接近普通 stochastic/deterministic autoencoder。

Posterior可以：

> 远离 prior。

于是训练时：

\[
z\sim q(z|x)
\]

很好用，

但测试从：

\[
z\sim p(z)
\]

采样：

> 可能落到 decoder陌生的 latent regions。

---

# 56. 所以 β 不是“越小越防 Collapse越好”

真正需要平衡：

\[
\boxed{
\text{use latent}
}
\]

和：

\[
\boxed{
\text{make prior usable}
}
\]

---

# 57. ACT 的 β = 10 要怎样理解？

ACT canonical：

\[
\boxed{
\beta=10
}
\]

但不能单独看到“10很大”就说：

> KL特别强。

因为：

\[
L_1
\]

和：

\[
KL
\]

本身 scale不同。

---

# 58. Loss Weight 的绝对意义依赖 Scale

如果：

\[
L_1=0.02
\]

而：

\[
KL=0.001
\]

乘10：

\[
0.01
\]

两者同量级。

如果：

\[
KL=10
\]

乘10就是：

\[
100
\]

完全不同。

所以必须看：

\[
\boxed{
\text{actual loss magnitudes and gradients}
}
\]

---

# 59. ACT Objective 还有一个严格数学细节

标准 VAE ELBO reconstruction term：

\[
-\mathbb E_q
[
\log p_\theta(A|O,z)
]
\]

ACT official implementation直接使用：

\[
\boxed{
L_1(A,\hat A)
}
\]

---

# 60. L1 可以有 Probabilistic Interpretation

如果假设：

\[
p_\theta(
A|O,z
)
\]

是 fixed-scale Laplace distribution，

那么 negative log-likelihood：

> 与 L1 距离成正比，加常数。

---

# 61. 但 ACT 还用了 β=10

所以它更像：

\[
\boxed{
\text{β-weighted CVAE-style objective}
}
\]

而不是：

> 朴素标准 ELBO 的严格 unweighted形式。

因此本文在讨论 rate–distortion时：

> 把 ACT 视为 CVAE/β-VAE-style tradeoff，

而不是声称其代码恰好等于某个唯一校准 likelihood 下的标准 ELBO。

---

# 62. 什么叫 Full Collapse？

假设 latent：

\[
z\in\mathbb R^{32}
\]

如果所有32维：

\[
q(z_j|x)
\approx
\mathcal N(0,1)
\]

且 Decoder不依赖任何：

\[
z_j
\]

那么：

\[
\boxed{
\text{full posterior collapse}
}
\]

---

# 63. Partial Collapse

更常见：

> 只有一部分 latent dimensions真的 active。

例如32维：

```text
dim 0  KL = 0.8
dim 1  KL = 0.3
dim 2  KL = 0.0001
dim 3  KL = 0.0000
...
```

可能只有：

\[
5
\]

维真正被使用。

---

# 64. 所以只看 Total KL 不够

假设：

\[
KL_{\text{total}}=8
\]

可能：

### Case A

32维每个：

\[
0.25
\]

都在工作。

### Case B

1维：

\[
8
\]

其他31维：

\[
0
\]

两者表示完全不同的 latent usage。

---

# 65. ACT 官方 `kl_divergence` 已经计算了 Dim-Wise KL

`policy.py`：

```python
total_kld, dim_wise_kld, mean_kld =
    kl_divergence(
        mu,
        logvar
    )
```

虽然最终 loss主要使用：

```python
total_kld[0]
```

但函数同时返回：

> per-dimension KL statistics。

---

# 66. 这正好可以拿来诊断 ACT Latent Usage

训练后应该至少画：

\[
\boxed{
KL_j
}
\]

for：

\[
j=1,\ldots,32
\]

---

# 67. Gaussian Per-Dimension KL

ACT：

\[
q(z_j)
=
\mathcal N(
\mu_j,
\sigma_j^2
)
\]

prior：

\[
\mathcal N(0,1)
\]

单维：

\[
\boxed{
KL_j
=
\frac12
(
\mu_j^2
+
\sigma_j^2
-
1
-
\log\sigma_j^2
)
}
\]

---

# 68. 如果完全 collapsed

\[
\mu_j=0
\]

\[
\sigma_j^2=1
\]

则：

\[
KL_j
=
\frac12(
0+1-1-0
)
=
\boxed{
0
}
\]

---

# 69. 为什么 KL_j > 0？

可能因为：

\[
\mu_j\neq0
\]

或者：

\[
\sigma_j^2\neq1
\]

或者两者都有。

说明：

> posterior在这一维偏离 prior。

---

# 70. 但再次强调：偏离 Prior ≠ Decoder 真正使用

Encoder可能产生 nonzero KL，

但 Decoder weights对：

\[
z
\]

很不敏感。

所以必须结合：

> decoder sensitivity tests。

---

# 71. 什么叫 Active Units？

VAE文献常用一个 metric：

> Active Units（AU）。

一种常见思路是看 posterior mean：

\[
\mu_j(x)
\]

跨数据集是否有足够 variance。

---

# 72. 如果某维 μ 几乎总是0

\[
Var_x[
\mu_j(x)
]
\approx0
\]

那么这个维度很可能：

> 没有编码 input-specific variation。

---

# 73. 但 Active Units 只是 Diagnostic

它不是：

> latent usefulness的完整定义。

因为：

- variance可能存在但 decoder忽略；
- posterior variance也携带信息；
- nonlinear usage更复杂。

所以最好多指标联合。

---

# 74. 一个更直接的问题：改变 z，Output 会不会变？

这是 ACT 最有意义的实验之一。

固定 observation：

\[
O
\]

改变：

\[
z
\]

观察：

\[
\hat A(O,z)
\]

是否变化。

---

# 75. 如果 Decoder 完全忽略 z

那么：

\[
\boxed{
\hat A(O,z_1)
\approx
\hat A(O,z_2)
}
\]

对各种：

\[
z_1,z_2
\]

都成立。

---

# 76. 这就是 Latent Sensitivity Test

可以固定一条 observation，

采样：

\[
z^{(1)},...,z^{(M)}
\sim
\mathcal N(0,I)
\]

计算：

\[
A^{(m)}
=
f(O,z^{(m)})
\]

---

# 77. 然后测 Output Variance

例如：

\[
\boxed{
S_z
=
\frac1{k d_a}
\sum_{t,j}
Var_m[
A^{(m)}_{t,j}
]
}
\]

如果：

\[
S_z\approx0
\]

说明：

> prior latent sample几乎不影响动作预测。

---

# 78. 但 Prior Sensitivity 还不够

因为 ACT训练时 Decoder主要看到：

\[
z\sim q(z|O,A)
\]

而不是随机任意：

\[
z\sim p(z)
\]

所以还应测试：

> posterior latents。

---

# 79. Posterior Reconstruction Test

对 training/validation example：

\[
(O,A)
\]

计算：

\[
\mu,\logvar
\]

取：

\[
z=\mu
\]

或者 posterior sample。

得到：

\[
\hat A_{\text{post}}
\]

---

# 80. 再与 z=0 比较

\[
\hat A_{0}
=
f(O,0)
\]

比较：

\[
L_1(
A,
\hat A_{\text{post}}
)
\]

vs：

\[
L_1(
A,
\hat A_0
)
\]

---

# 81. 如果两者几乎完全一样

这提示：

> posterior latent可能没有给 reconstruction提供明显帮助。

但不能单凭一次 batch下结论。

应该统计整个 validation set。

---

# 82. 定义 Reconstruction Gain

例如：

\[
\boxed{
G
=
L_1(A,f(O,0))
-
L_1(A,f(O,\mu))
}
\]

如果：

\[
G>0
\]

说明 posterior mean：

> 帮助重建。

如果长期：

\[
G\approx0
\]

要怀疑 latent usage很弱。

---

# 83. 还可以做 Latent Shuffle Test

一个非常强的实验：

Batch里有：

\[
(O_i,A_i,z_i)
\]

把 latent随机打乱：

\[
z_i
\rightarrow
z_{\pi(i)}
\]

再 decode：

\[
\hat A_i
=
f(
O_i,
z_{\pi(i)}
)
\]

---

# 84. 如果 Shuffle 完全不影响 Reconstruction

那说明：

\[
\boxed{
Decoder\text{ 很可能没在用 sample-specific }z
}
\]

---

# 85. 如果 Reconstruction 明显变差

说明：

> 正确 posterior latent和对应 action chunk之间确实存在 useful association。

这是比仅看 KL：

> 更直接的 causal-style ablation。

---

# 86. Latent Zeroing Test

训练时本来：

\[
z=\mu+\sigma\epsilon
\]

人为改成：

\[
z=0
\]

保持所有其他输入不变。

看：

- reconstruction loss；
- gradients；
- rollout；

如何变化。

---

# 87. Latent Mean vs Sample Test

比较：

\[
z=\mu
\]

和：

\[
z=\mu+\sigma\epsilon
\]

如果二者输出差异很小：

> posterior stochasticity可能不重要。

但：

\[
\mu
\]

仍可能携带大量信息。

---

# 88. 这和 Collapse 不一样

Posterior collapse问的是：

> \(z\) 是否携带/传递 data-specific information。

而“sampling noise是否重要”问：

> distribution variance是否影响具体 decode。

不要混淆。

---

# 89. Latent Interpolation Test

固定：

\[
O
\]

选两个 posterior means：

\[
z_A,z_B
\]

插值：

\[
z(\alpha)
=
(1-\alpha)z_A
+
\alpha z_B
\]

观察：

\[
f(O,z(\alpha))
\]

动作 trajectory是否：

> 连续发生系统变化。

---

# 90. 但 ACT 中这个实验要小心

如果：

\[
z_A,z_B
\]

来自不同 observations，

直接固定同一个：

\[
O
\]

进行 swap/interpolation，

可能产生：

> training distribution外的 \(O,z\) 组合。

所以实验是：

> sensitivity probe，

不一定代表自然 latent semantics。

---

# 91. 更好的 Same-Condition Experiment

如果数据里有：

> 非常相近 observation / state，但不同 human trajectories，

可以比较这些样本的：

\[
\mu
\]

是否系统分离。

这样更接近：

\[
\boxed{
\text{style latent}
}
\]

的原始目的。

---

# 92. Latent Nearest-Neighbor Analysis

取：

\[
\mu_i
\]

作为 latent embedding。

在 latent space找 nearest neighbors。

看它们是否共享：

- trajectory timing；
- pause pattern；
- approach style；
- joint coordination；

等。

---

# 93. 但这只能做 Exploratory Interpretation

不能看见 cluster就直接说：

> latent dimension 7 是“疲劳”。

需要：

- controlled labels；
- intervention；
- statistical validation。

否则只是：

> correlation / visualization。

---

# 94. Jacobian Sensitivity 是更数学的诊断

对于固定：

\[
O
\]

计算：

\[
\boxed{
J_z
=
\frac{\partial \hat A}{\partial z}
}
\]

shape：

\[
[k\times14,\ 32]
\]

---

# 95. 如果 Decoder 完全忽略 z

那么：

\[
\boxed{
J_z
\approx0
}
\]

---

# 96. 可以看 Frobenius Norm

\[
\boxed{
\|J_z\|_F
}
\]

作为 local sensitivity metric。

如果大量 samples：

\[
\|J_z\|_F
\]

都非常小，

说明：

> action decoder对 latent局部不敏感。

---

# 97. 但 Jacobian 很大也不自动等于“Latent 有意义”

它只说明：

> output对 latent敏感。

不保证：

- semantics；
- robustness；
- useful reconstruction；
- prior calibration。

仍要结合其他指标。

---

# 98. Decoder First-Layer Weight 能不能直接看？

ACT：

\[
z
\in\mathbb R^{32}
\]

经过：

```python
latent_out_proj =
    nn.Linear(
        32,
        512
    )
```

如果这个 weight norm非常小：

> 可能提示 latent path弱。

---

# 99. 但 Weight Norm 也不是充分证据

即使：

\[
\|W_z\|
\]

不小，

后续 Transformer可能：

> 抵消/忽略 latent token。

反过来，小 weight也可能经过后续 amplification。

所以真正应看：

> end-to-end functional sensitivity。

---

# 100. Posterior Collapse 为什么在 Powerful Decoder 中经典？

Bowman et al. 的 sentence VAE发现：

> RNN decoder很容易利用previous words，绕开 global latent。

他们报告：

> 没有 KL annealing和word dropout时，模型可靠地出现接近0 KL。

这是 posterior collapse研究非常经典的早期案例。

---

# 101. Bowman 的第一个 Remedy：KL Cost Annealing

训练初期：

\[
\beta=0
\]

所以 objective主要：

\[
L\approx L_{\text{recon}}
\]

---

# 102. 这时 Encoder 可以“免费”往 z 塞信息

不会立即因为：

\[
KL
\]

受到惩罚。

所以 Decoder有机会先学会：

> 使用 latent。

---

# 103. 然后逐渐增加 β

\[
0
\rightarrow
1
\]

让 posterior逐渐：

> 压向 prior-compatible latent space。

希望最终：

- latent仍有用；
- prior也可用。

---

# 104. KL Annealing 的直觉

不要一开始就告诉模型：

> “用 \(z\) 很贵。”

否则 Decoder可能最早学会：

> “那我不用。”

先让：

\[
z
\]

建立功能价值，

再慢慢收取 rate cost。

---

# 105. 但 KL Annealing 不是保证

训练 dynamics复杂。

有些模型 anneal后：

> 仍会 collapse。

它是一种 heuristic/training strategy，

不是数学保证。

---

# 106. Bowman 的第二个 Remedy：Weaken Decoder

在 language model里用：

> word dropout / historyless decoding。

减少 Decoder直接获得的 local context。

---

# 107. 为什么 Weaken Decoder 有用？

原本：

```text
decoder local context
↓↓↓↓↓↓↓↓↓↓↓↓↓↓
z 可有可无
```

削弱 local context后：

```text
decoder local context
↓↓↓
z 变得更有价值
```

为了 reconstruction，

Decoder被迫：

> 读取 \(z\)。

---

# 108. 这是非常通用的思想

\[
\boxed{
\text{If decoder can solve task without latent, latent is easy to ignore.}
}
\]

减少 shortcut information：

> 提高 latent的必要性。

---

# 109. 但“削弱 Decoder”也有代价

你可能为了强迫使用：

\[
z
\]

故意让 Decoder变差。

这可能损害：

> 最终 likelihood / reconstruction。

所以不能把：

> latent usage

当成唯一目标。

---

# 110. 这也是 Alemi et al. 的核心提醒之一

**Fixing a Broken ELBO** 强调：

> 好 likelihood / ELBO 不一定自动意味着好 representation。

反过来：

> 强迫 latent信息也可能改变 generative performance tradeoff。

因此应明确：

> 你的最终目标是什么。

---

# 111. Rate–Distortion 不存在唯一“最好点”

不同模型可以在：

\[
(R,D)
\]

平面上取得不同 tradeoff。

例如：

### Model A

\[
R\approx0
\]

但 Decoder很强：

\[
D\text{ 仍然不错}
\]

---

# 112. Model B

\[
R\text{ 较高}
\]

latent携带更多信息，

reconstruction：

> 更好。

---

# 113. 两者可能甚至具有类似 ELBO

因为：

\[
D+R
\]

tradeoff不同。

所以只看：

> total ELBO/loss

可能看不出 latent representation质量。

---

# 114. 这对 ACT 特别重要

ACT README甚至提醒：

> policy success/smoothness可能在 validation loss已经 plateau后继续改善。

而对于 CVAE：

> total loss也不能单独告诉你 latent到底怎么用。

所以应该分开记录：

- L1；
- KL；
- per-dim KL；
- rollout success；
- latent diagnostics。

---

# 115. Free Bits 是什么？

Kingma et al. 在 IAF 工作中使用一种 modified objective：

> 给 latent groups一个最低“免费信息预算”。

概念上：

\[
\boxed{
L
=
D
+
\sum_j
\max(
\lambda,
KL_j
)
}
\]

取决于 sign convention。

---

# 116. 为什么叫 “Free Bits”？

当：

\[
KL_j<\lambda
\]

时，

把它进一步从：

\[
0.4
\]

压到：

\[
0.1
\]

并不会降低这部分 objective penalty。

---

# 117. 所以在这个区间里

Encoder可以使用：

> 一定量 latent information，

而不用因为每一点 KL都被立刻惩罚。

这降低：

> collapse到0 KL的压力。

---

# 118. 注意一个常见错误解释

Free Bits并不是简单：

> “强制 KL 一定大于 \(\lambda\)”。

更精确地说：

> 它修改 objective，使低于阈值的 KL reduction不再获得优化奖励。

实际不同实现可能：

- per-dimension；
- per-group；
- aggregate；

形式不同。

---

# 119. β-VAE 又是什么关系？

β-VAE：

\[
\boxed{
L
=
D
+
\beta R
}
\]

如果：

\[
\beta>1
\]

通常强化：

> compression / prior pressure。

---

# 120. 所以如果目标只是“防 collapse”

简单把：

\[
\beta
\]

调得更大：

> 往往方向相反。

因为更大的 KL weight：

> 更鼓励低 rate。

---

# 121. 但 β-VAE 的原始目标并不是“防 Collapse”

它主要研究：

> disentangled representations。

这和：

> 保证 latent不被忽略

不是完全同一个问题。

---

# 122. ACT β=10 也不能直接套 β-VAE Disentanglement 结论

ACT没有声称：

> 32维 style latent被 disentangle成独立人类因素。

\[
\beta=10
\]

在 ACT 中首先是：

> reconstruction vs KL regularization权衡的超参数。

---

# 123. 什么叫 KL Floor / Minimum Desired Rate？

一些方法希望：

\[
R
\]

不要低于某个：

\[
R_{\min}
\]

因为：

> rate太低代表 representation capacity没有被使用。

这种思路与 Free Bits相关。

---

# 124. 但 Rate 高也不等于 Semantics 好

一个 Encoder可以：

> 在 z 中编码很多无关细节，

得到很高 KL。

所以：

\[
\boxed{
\text{information quantity}
\neq
\text{information quality}
}
\]

---

# 125. Posterior Collapse 的原因只有 Powerful Decoder 吗？

不是。

这是经典直觉，

但后续研究给出了更丰富解释。

---

# 126. He et al. 2019：Lagging Inference Networks

他们从：

> training dynamics

角度研究 collapse。

核心观察：

> early training时 inference network \(q_\phi\) 跟不上不断变化的 true posterior。

---

# 127. 什么叫 Inference Network “Lagging”？

Decoder/generative model parameters：

\[
\theta
\]

一直改变。

所以 true posterior：

\[
p_\theta(z|x)
\]

也是：

> moving target。

Encoder：

\[
q_\phi(z|x)
\]

需要不断追它。

---

# 128. 如果 Encoder 跟不上

当前：

\[
q_\phi
\]

给出的 latent可能：

> 对 Decoder帮助不大。

于是 generative model发现：

> 依赖 z并不划算。

---

# 129. 然后 Decoder 更倾向不用 z

一旦 Decoder逐渐学会：

> 不需要 latent，

true posterior本身也会趋近：

\[
p(z)
\]

最终：

\[
q(z|x)\approx p(z)
\]

collapse。

---

# 130. 所以后验坍缩可以是 Training Dynamics 的自强化过程

```text
inference network lags
↓
z initially unhelpful
↓
decoder learns to ignore z
↓
posterior moves toward prior
↓
q also collapses to prior
```

---

# 131. He et al. 的方法

他们提出：

> aggressive inference-network updates。

在 generative model每次更新前：

> 让 Encoder多优化几步，

使：

\[
q_\phi
\]

更好跟上当前 posterior。

---

# 132. 这说明一个重要事实

不能简单说：

\[
\boxed{
\text{posterior collapse = decoder太强}
}
\]

更完整的原因可能包括：

- objective tradeoff；
- decoder expressivity；
- inference lag；
- optimization path；
- initialization；
- data structure。

---

# 133. Lucas et al. 等工作进一步说明

即使非常简单的 latent-variable models，

collapse-like behavior也可能和：

> optimization landscape / model structure

有关，

而不只是“神经网络 Decoder太强”。

所以 posterior collapse：

> 是一个比单一口诀更复杂的问题。

---

# 134. 那怎样判断“ACT 有没有 Collapse”？

不能只靠：

> ACT论文说有 CVAE。

应该实际测。

至少做：

\[
\boxed{
5\text{ 类诊断}
}
\]

---

# 135. 诊断 1：Total KL Curve

训练时记录：

\[
KL_{\text{total}}
\]

随 epoch变化。

如果：

\[
KL\rightarrow0
\]

而长期保持接近0：

> 强烈怀疑 collapse。

---

# 136. 但 KL 的绝对数值要看 Reduction

ACT代码：

> 对 latent dimensions求和，再对 batch求平均。

所以：

\[
KL=1
\]

是：

> 32维总 rate量级，

不是每维1。

解释时必须知道 reduction。

---

# 137. 诊断 2：Per-Dimension KL

画：

```text
dim 0: 0.00
dim 1: 0.43
dim 2: 0.00
...
dim31: 0.81
```

看：

> 多少 latent dimensions active。

---

# 138. 一个可能的 Pattern

```text
32 dims
↓
5 dims have meaningful KL
27 dims ≈ 0
```

这叫：

> partial collapse / inactive dimensions。

并不代表整个 latent完全失效。

---

# 139. 诊断 3：Posterior Mean Distribution

统计：

\[
\mu_j
\]

在 validation set上的：

- mean；
- std；
- histogram。

Full collapse时：

\[
\mu_j
\]

通常：

> 都挤在0附近。

---

# 140. 同时看 logvar

Full collapse：

\[
\logvar_j
\approx0
\]

即：

\[
\sigma_j^2\approx1
\]

---

# 141. 诊断 4：Decoder Sensitivity to z

固定：

\[
O
\]

改变：

\[
z
\]

看：

\[
\hat A
\]

是否改变。

可以做：

- \(z=0\)；
- \(z=\mu\)；
- posterior samples；
- prior samples；
- shuffled z。

---

# 142. 诊断 5：Reconstruction Dependence on z

比较：

\[
L_{\text{post}}
=
L_1(
A,
f(O,z_{\text{post}})
)
\]

和：

\[
L_0
=
L_1(
A,
f(O,0)
)
\]

以及：

\[
L_{\text{shuffle}}
=
L_1(
A,
f(O,z_{\text{wrong}})
)
\]

---

# 143. 理想上

如果 latent有用：

\[
\boxed{
L_{\text{post}}
<
L_{\text{shuffle}}
}
\]

通常也希望：

\[
L_{\text{post}}
<
L_0
\]

至少在训练-style reconstruction setting中。

---

# 144. 如果三者几乎一样

\[
L_{\text{post}}
\approx
L_0
\approx
L_{\text{shuffle}}
\]

说明：

> Decoder可能几乎完全不关心 latent。

---

# 145. 诊断 6：Gradient to Latent Path

可以 inspect：

\[
\left\|
\frac{\partial L_{L1}}{\partial z}
\right\|
\]

如果 reconstruction loss对：

\[
z
\]

长期 gradient很小，

说明 Decoder：

> 很可能没有从 latent得到多少 reconstruction signal。

---

# 146. 但 Gradient 也要谨慎解释

某个 checkpoint局部 gradient小：

> 可能只是处于局部平坦区域。

所以应跨：

- samples；
- epochs；

统计。

---

# 147. 诊断 7：Ablate Latent Input Entirely

训练完成后：

> 把 `latent_out_proj` 输入强制为0。

比较 rollout success。

如果成功率：

> 完全不变，

说明 deployment policy不依赖 latent。

---

# 148. 但 ACT 原本 Inference 就 z=0

所以这个实验需要更仔细设计。

ACT test-time：

\[
z=0
\]

本来就是固定的。

所以不能说：

> “推理z=0，所以latent没被使用。”

这是错误的。

---

# 149. Training-Time Latent 可能只是 Training Scaffold

ACT 的 Encoder：

> 训练时存在，

推理时删除。

因此 latent可能发挥：

\[
\boxed{
\text{training-time representation / regularization role}
}
\]

而不需要 test-time随机控制 style。

---

# 150. 这很像某些 Privileged Information

训练时模型可以利用：

> future action chunk

产生：

\[
z
\]

帮助学习。

推理时：

> future actions不可见，

改用 prior center：

\[
z=0
\]

---

# 151. 所以 ACT 的问题比普通 Generative VAE 更特殊

普通生成模型通常希望：

> 推理/生成时采样不同 z得到不同 outputs。

ACT canonical则故意：

\[
\boxed{
z=0
}
\]

做 deterministic decoding。

---

# 152. 那 ACT 为什么还怕 Posterior Collapse？

因为如果训练阶段：

\[
z
\]

完全没被 Decoder使用，

那么 CVAE Encoder：

> 可能基本没有提供 action-style information。

CVAE objective想解决的人类 demonstration variability：

> 就没有按预期发挥。

---

# 153. 但即使 z test-time固定，Training-Time Usage 仍可重要

训练时：

\[
z
\]

可以帮助 Decoder区分：

> 同一个 observation附近的多个 demonstration trajectories。

这样 Decoder不必把多种 modes：

> 硬平均成一个动作。

---

# 154. 一个 Toy Example

同样 observation：

\[
O
\]

human data有两种 chunk：

\[
A^{(1)}
\]

和：

\[
A^{(2)}
\]

---

# 155. 没有 Latent

deterministic model：

\[
f(O)
\]

可能被 L1/MSE迫使：

> 找一个折中。

---

# 156. 有 Latent

Encoder：

\[
(O,A^{(1)})
\rightarrow
z_1
\]

\[
(O,A^{(2)})
\rightarrow
z_2
\]

Decoder：

\[
f(O,z_1)
\approx
A^{(1)}
\]

\[
f(O,z_2)
\approx
A^{(2)}
\]

这样可以：

> 在训练中解释 multimodality。

---

# 157. Test-Time z=0

ACT选择 prior center：

\[
z=0
\]

输出：

> 某种 canonical deterministic trajectory。

这不等于训练时：

> z没有作用。

---

# 158. 这正是为什么“Inference 固定 z=0”不能诊断 Collapse

你必须看：

> training-time posterior z 是否改变 reconstruction。

---

# 159. 如果模型真的 Collapse 呢？

那么：

\[
f(O,z)
\approx
f(O)
\]

训练时：

\[
z_1,z_2
\]

也几乎没用。

CVAE就退化成：

> conditional deterministic chunk predictor + 一个无效 Encoder branch。

---

# 160. Collapse 时 z=0 为什么当然也能 Work？

因为 Decoder本来：

> 就没在看 z。

所以：

\[
z=0
\]

\[
z=random
\]

\[
z=\mu
\]

输出都近似相同。

---

# 161. 所以一个很有用的诊断

如果：

\[
f(O,0)
\approx
f(O,z_1)
\approx
f(O,z_2)
\]

对很多显著不同：

\[
z_1,z_2
\]

都成立，

非常像：

> latent ignored。

---

# 162. ACT 的 CVAE Ablation 告诉我们什么？

ACT论文 Ablation 中，

作者比较：

> 有 CVAE objective

和：

> 去掉 CVAE、直接根据当前 observation预测 action sequence，只用 L1。

---

# 163. Scripted Data

因为 scripted demonstrations：

> 基本 deterministic，

移除 CVAE：

> 几乎没有影响。

---

# 164. Human Data

论文报告 aggregate success：

\[
\boxed{
35.3\%
\rightarrow
2\%
}
\]

当移除 CVAE objective时显著下降。

这说明：

\[
\boxed{
\text{CVAE-style training is crucial for their human demonstration setting.}
}
\]

---

# 165. 但这能不能证明“32个 z 维度都没 Collapse”？

\[
\boxed{
不能。
}
\]

Ablation比较的是：

> 整个 CVAE objective / latent training mechanism存在与否。

它没有逐维测：

- KL_j；
- mutual information；
- active units；
- sensitivity。

---

# 166. 能不能证明 z 编码了“睡眠”“力量”“意图”？

当然不能。

Success ablation只证明：

> 这种 training design显著改善 human-data performance。

不提供：

> latent causal semantics。

---

# 167. 能不能证明 test-time z=0 是“average style”？

也不能。

即使 training latent有用，

\[
z=0
\]

只是：

> prior mean / latent origin。

Decoder nonlinear：

\[
f(E[z])
\neq
E[f(z)]
\]

一般成立。

---

# 168. 所以 ACT Ablation 的正确结论

可以说：

\[
\boxed{
\text{CVAE objective materially matters for human-data ACT performance.}
}
\]

不能扩大成：

\[
\boxed{
\text{all latent dimensions encode interpretable human styles and zero is the average style.}
}
\]

---

# 169. ACT 会不会 Partial Collapse？

完全可能。

\[
latent\_dim=32
\]

只是：

> available capacity。

实际可能只有：

\[
r<32
\]

个 dimensions在使用。

---

# 170. 这甚至不一定是坏事

如果真实 demonstration variation只需要：

\[
4
\]

个 latent degrees of freedom，

模型只使用4维：

> 很合理。

目标不是：

> 强迫所有32维都非零。

---

# 171. 所以“多少 Active Dimensions 才算好”没有统一答案

要看：

- data complexity；
- decoder；
- prior；
- objective；
- downstream performance。

---

# 172. Full Utilization 不是目标

一个32维 latent：

> 32维全部高KL

不一定比：

> 6维有效、26维inactive

更好。

Representation应该：

> 有效且足够，

而不是机械填满。

---

# 173. KL≈0 一定就是灾难吗？

也要看任务。

如果 data本来 deterministic：

\[
A=f(O)
\]

根本没有 observation之外的 multimodality，

那么 latent没必要携带额外信息。

---

# 174. ACT Scripted Data 就是很好的例子

论文发现：

> scripted deterministic data中去掉 CVAE几乎没影响。

这说明：

\[
\boxed{
\text{latent usefulness depends on irreducible conditional variation in the data.}
}
\]

---

# 175. 所以 Collapse 的“坏”与否和 Modeling Goal 有关

如果你的目标只是：

> 最佳 deterministic prediction，

latent unused可能没问题。

但如果设计 CVAE就是为了：

> 表达 multimodality/style，

那 collapse就意味着：

> 核心设计目标失败。

---

# 176. Posterior Collapse 与 Identifiability 也不同

假设 latent确实 active：

\[
I(A;Z|O)>0
\]

也不代表：

> 每个 z dimension拥有唯一真实语义。

---

# 177. Latent Representation 可以发生旋转/重参数化

例如：

\[
z'
=
Rz
\]

某些 Decoder/Encoder可以相应变换，

仍表示相同信息。

所以 latent coordinate语义：

> 通常不是 identifiable。

---

# 178. 所以三个问题要分开

### 1. Is z used?

\[
I(A;Z|O)>0?
\]

### 2. Is z useful?

它是否改善：

- recon；
- generation；
- rollout。

### 3. Is z interpretable/identifiable?

每维是否对应：

> 清晰人类因素。

这三个不是同一件事。

---

# 179. Posterior Collapse 主要针对第一个

\[
\boxed{
\text{Is latent being used at all?}
}
\]

---

# 180. Disentanglement 主要针对第三个

\[
\boxed{
\text{What structure does latent information have?}
}
\]

不要混淆。

---

# 181. Prior Collapse 和 Posterior Collapse 是一个词吗？

标准术语：

> posterior collapse。

核心是 approximate posterior：

\[
q(z|x)
\]

趋近：

\[
p(z)
\]

有些文献也会说：

- KL collapse；
- latent collapse；
- inactive latent units。

具体语义可能略有不同。

---

# 182. KL Vanishing

通常指：

\[
KL
\rightarrow0
\]

它是 posterior collapse的重要表现。

但最好不要只凭某一个 batch：

\[
KL\text{ 很小}
\]

就立即下最终结论。

---

# 183. 为什么 Total KL 很小可能只是 Scaling 问题？

如果 latent dimensions少：

> total KL自然小。

如果 loss单位/reduction不同：

> 数值不可直接横向比较。

所以要：

- 知道公式；
- 知道 reduction；
- 做 baseline；
- 看趋势。

---

# 184. ACT 中正确记录 KL

建议同时记录：

\[
KL_{total}
\]

\[
KL_{mean-per-dim}
\]

\[
KL_j,\ j=1...32
\]

---

# 185. 再记录 μ Statistics

每维：

\[
E[\mu_j]
\]

\[
Std[\mu_j]
\]

以及：

\[
E[\logvar_j]
\]

---

# 186. Collapse 典型 Pattern

可能看到：

```text
KL total → ~0
μ mean → 0
μ std  → ~0
logvar mean → 0
output insensitive to z
```

几项一起出现：

> 证据非常强。

---

# 187. Partial Collapse Pattern

```text
total KL > 0

dims 0,3,8,12:
    KL significantly > 0
    μ varies across examples

other dims:
    KL ~ 0
    μ ~ 0
    logvar ~ 0
```

说明：

> latent capacity只部分使用。

---

# 188. Non-Collapsed but Prior-Mismatched Pattern

```text
KL high
μ strongly varies
output sensitive to z
reconstruction improves
but prior z samples produce strange actions
```

这说明：

> latent被用得很多，

但 prior regularization可能不够。

---

# 189. 这正是 β 太小可能出现的情况

Train posterior：

> 很好。

Test prior：

> 不好。

对于 ACT尤其重要，

因为 inference使用：

\[
z=0
\]

prior center。

---

# 190. 所以 ACT 需要的是一种特殊平衡

训练 posterior：

> 必须提供 enough variation information。

同时：

\[
z=0
\]

附近必须：

> 是 Decoder训练见过、可用的 latent region。

KL帮助把 posterior：

> 拉回 prior附近。

---

# 191. 这可能是 β=10 的设计动机之一

ACT需要：

> inference-time zero latent稳定工作。

如果 posterior means遍布很远：

\[
\| \mu \|\gg0
\]

那：

\[
z=0
\]

可能是 Decoder训练时很少见的位置。

---

# 192. KL 则让 Posterior Distribution 靠近 N(0,I)

因此：

\[
z=0
\]

更可能位于：

> high-density、熟悉的 latent region附近。

---

# 193. 但“靠近 prior”太强又会 Collapse

所以 ACT 的 latent训练就是：

\[
\boxed{
\text{make }z\text{ informative enough}
\quad
\text{but not so unconstrained that }z=0\text{ becomes meaningless}
}
\]

---

# 194. 这就是真正的 Tradeoff

左边极端：

\[
\beta\rightarrow0
\]

可能：

> posterior highly informative but prior mismatch。

右边极端：

\[
\beta\rightarrow\infty
\]

可能：

> posterior=prior, no information。

中间：

> useful regularized latent。

---

# 195. 一个示意图

```text
β too small
│
│  z carries lots of info
│  recon good
│  prior mismatch high
│  z=0 may be unfamiliar
│
├──────────── useful regime
│
│  enough latent information
│  reasonable prior alignment
│
│
│  z carries almost no info
│  KL ≈ 0
│  decoder ignores z
│
β too large
```

实际并不是单调这么简单，

但这个图适合建立直觉。

---

# 196. 怎样给 ACT 做 β Sweep？

例如训练：

\[
\beta
\in
\{
0,\ 0.1,\ 1,\ 10,\ 50
\}
\]

对每个 checkpoint记录：

1. validation L1；
2. total KL；
3. per-dim KL；
4. posterior-vs-zero recon gap；
5. latent shuffle penalty；
6. rollout success；
7. z=0 rollout；
8. prior-sample behavior。

---

# 197. 不要只选 Total Validation Loss 最低的 β

因为：

\[
L=L1+\beta KL
\]

不同 β：

> total loss本身不可直接公平横向比较。

更应该比较：

- task metric；
- reconstruction；
- latent diagnostics。

---

# 198. KL Annealing 能不能直接用于 ACT？

理论上可以尝试：

\[
\beta(t):
0\rightarrow10
\]

让模型先学：

> 使用 latent重建 human action variation，

然后逐渐加强 prior alignment。

---

# 199. 但这是一个新实验，不是 Original ACT Canonical

Official ACT training：

\[
\boxed{
\beta=10
}
\]

固定使用。

所以不能写成：

> ACT原论文使用KL annealing。

没有。

---

# 200. Free Bits 能不能用于 ACT？

理论上也可以。

例如 per-dim：

\[
L_{KL}
=
\sum_j
\max(
\lambda,
KL_j
)
\]

降低把每维压到0的动力。

---

# 201. 但同样

这不是：

> canonical ACT。

它是：

> 可能用于 latent-collapse研究的改动。

知识库必须区分：

\[
\boxed{
\text{ACT Fact}
}
\]

vs：

\[
\boxed{
\text{Possible Extension}
}
\]

---

# 202. Weakening ACT Decoder 是否合理？

理论上：

> 如果 Decoder太容易忽略 z，可以减少 shortcut。

但实际 ACT需要：

> 强视觉控制能力。

故意削弱 Transformer：

> 可能直接降低policy能力。

所以在机器人策略中：

> 不一定是最优 remedy。

---

# 203. 更实际的 ACT Diagnostic-first Strategy

不要一上来改 architecture。

先测：

1. KL；
2. per-dim activity；
3. latent sensitivity；
4. shuffle test；
5. reconstruction gain；
6. rollout ablations。

只有确认：

> latent usage确实异常，

再考虑改训练。

---

# 204. 一个最小 PyTorch Per-Dim KL

```python
def kl_per_dim(mu, logvar):
    # mu, logvar: [B, Z]

    kl =
        -0.5 * (
            1
            + logvar
            - mu.pow(2)
            - logvar.exp()
        )

    # [B, Z]
    return kl
```

---

# 205. Dataset-Level Statistic

```python
kl =
    kl_per_dim(
        mu,
        logvar
    )

mean_kl_per_dim =
    kl.mean(dim=0)

total_kl =
    mean_kl_per_dim.sum()
```

得到：

\[
[32]
\]

的 latent-dimension profile。

---

# 206. Posterior Mean Activity

```python
mu_variance =
    mu.var(
        dim=0,
        unbiased=False
    )
```

看：

\[
Var[\mu_j]
\]

哪些维度明显大于0。

---

# 207. Latent Shuffle Test 伪代码

```python
z =
    reparameterize(
        mu,
        logvar
    )

pred_correct =
    decoder(
        obs,
        z
    )

perm =
    torch.randperm(
        z.shape[0]
    )

pred_shuffled =
    decoder(
        obs,
        z[perm]
    )
```

然后：

```python
loss_correct = L1(
    pred_correct,
    actions
)

loss_shuffled = L1(
    pred_shuffled,
    actions
)
```

---

# 208. 定义 Shuffle Gap

\[
\boxed{
\Delta_{\text{shuffle}}
=
L_{\text{shuffle}}
-
L_{\text{correct}}
}
\]

如果：

\[
\Delta_{\text{shuffle}}\gg0
\]

说明：

> sample-specific latent对 reconstruction重要。

---

# 209. 如果 Δshuffle ≈ 0

可能：

1. collapse；
2. observation已经足够；
3. latent编码的信息很弱；
4. decoder用 latent的方式不影响 L1很多。

需要结合其他指标。

---

# 210. z=0 Gap

\[
\boxed{
\Delta_0
=
L(
A,
f(O,0)
)
-
L(
A,
f(O,z_{\text{post}})
)
}
\]

---

# 211. Prior-Sample Output Diversity

固定：

\[
O
\]

采：

\[
z_m\sim N(0,I)
\]

测：

\[
Var_m[
f(O,z_m)
]
\]

---

# 212. Posterior-Sample Output Diversity

固定 example：

\[
(O,A)
\]

采：

\[
z_m
\sim
q(z|O,A)
\]

测：

> stochastic uncertainty影响。

---

# 213. Latent Traversal

选某维：

\[
j
\]

其他维设0，

让：

\[
z_j
\in
\{-2,-1,0,1,2\}
\]

观察 predicted trajectory。

---

# 214. 这个实验能看什么？

如果输出：

> 随 \(z_j\) 平滑系统变化，

说明这一维：

> Decoder确实敏感。

---

# 215. 但不要立刻给它人类语义

例如看到：

> 手臂更高

不能直接说：

> “z7就是抬高手臂因素。”

需要：

- 多 observation重复；
- statistical consistency；
- intervention validation。

---

# 216. Mutual Information 怎么估？

高维连续：

\[
I(A;Z|O)
\]

精确估计并不容易。

所以实践中常用 proxies：

- expected KL；
- active units；
- reconstruction ablations；
- MI estimators；
- auxiliary probes。

---

# 217. Expected KL 是 Mutual Information Upper Bound-like Quantity

如前面分解：

\[
R
=
I(A;Z|O)
+
KL_{\text{aggregate}}
\]

所以：

\[
\boxed{
I(A;Z|O)
\le R
}
\]

---

# 218. 如果 R≈0

MI几乎一定：

\[
\approx0
\]

---

# 219. 如果 R 很大

MI可能：

> 大，

也可能有一部分只是：

> aggregated posterior mismatch。

所以还需其他 diagnostics。

---

# 220. 为什么 Strong Conditional Decoder 更容易 Collapse？

CVAE里 Decoder已有 condition：

\[
O
\]

相当于：

> 给它一个 shortcut。

如果：

\[
O\rightarrow A
\]

mapping本身很强，

latent的边际价值：

> 降低。

---

# 221. 所以 CVAE Posterior Collapse 比普通 VAE 更值得警惕

尤其 sequence-to-sequence任务：

- source sentence；
- current robot observation；
- context embedding；

已经包含大量 target information。

latent非常容易：

> 成为可选项。

---

# 222. Conditional VAE 研究因此经常直接促进 MI

有工作会在 objective中：

> 显式鼓励 latent和target之间的 mutual information。

例如 conditional variational NMT研究就用：

- modified ELBO；
- auxiliary bag-of-words objective；

促进：

\[
I(Z;Y)
\]

或相应 conditional information。

---

# 223. ACT 没有这些额外 Anti-Collapse Mechanisms

Canonical ACT主要：

- CVAE encoder；
- L1；
- β KL。

没有：

- free bits；
- KL annealing；
- explicit MI reward；
- auxiliary latent prediction；

等。

---

# 224. 但 ACT Human Ablation 表明 CVAE Objective 确实重要

这意味着：

> 在他们的数据/architecture/training regime中，

CVAE design没有简单等价于：

> “完全没作用。”

---

# 225. 但不能由此知道 Latent 的精细信息结构

ACT paper没有系统报告：

- per-dim KL；
- AU；
- MI；
- latent traversal；
- latent swap；

所以这些仍是：

> 很好的复现/研究方向。

---

# 226. 这甚至可以做成一个很不错的小研究项目

例如：

> **What Does ACT’s Style Latent Actually Encode?**

实验：

1. reproduce ACT；
2. log posterior statistics；
3. identify active dimensions；
4. latent swaps；
5. trajectory clustering；
6. β sweep；
7. zero/shuffle ablation；
8. human vs scripted comparison。

---

# 227. Scripted vs Human 特别适合做 Control

Original ACT已经显示：

- scripted data：
  > CVAE removal影响小；
- human data：
  > CVAE removal影响大。

进一步可以问：

\[
\boxed{
\text{Human data是否拥有更高 conditional latent rate？}
}
\]

---

# 228. 一个可检验假设

Human model：

\[
KL_{\text{human}}
>
KL_{\text{scripted}}
\]

或者：

\[
\#ActiveUnits_{\text{human}}
>
\#ActiveUnits_{\text{scripted}}
\]

---

# 229. 但结果不一定如此

因为：

- β固定；
- decoder capacity；
- data normalization；
- optimization；

都可能影响 KL。

所以这是：

> empirical hypothesis，

不是理论保证。

---

# 230. 另一个实验：Latent Shuffle 对 Human Data 是否更伤？

如果 human variation确实通过 z编码，

预期：

\[
\Delta_{\text{shuffle,human}}
>
\Delta_{\text{shuffle,scripted}}
\]

可能成立。

这会比只看 success ablation：

> 更直接说明 sample-specific latent usage。

---

# 231. 还有一个很重要的问题：ACT 训练后为什么不 Sample z？

Canonical inference：

\[
z=0
\]

不是：

\[
z\sim N(0,I)
\]

因为机器人控制更需要：

> deterministic, stable behavior。

---

# 232. 如果 latent表示多种 Human Styles

随机 sample：

> 可能每次 policy call选择不同 style，

导致：

- trajectory switching；
- jitter；
- inconsistent intent。

所以 ACT使用 prior mean：

> 固定 style。

---

# 233. 这和 Posterior Collapse 的关系

两种系统都可能出现：

\[
z=0
\]

推理，

但原因完全不同。

---

# 234. Case A：Healthy Latent

训练时：

\[
z
\]

显著帮助解释 demonstration variation。

推理时：

\[
z=0
\]

人为选择：

> 一个固定 canonical latent。

---

# 235. Case B：Collapsed Latent

训练时 Decoder本来就忽略：

\[
z
\]

推理：

\[
z=0
\]

只是：

> 随便给什么都一样。

---

# 236. 两者外表可能完全一样

都看到代码：

```python
latent_sample =
    torch.zeros(...)
```

但训练机制意义：

> 完全不同。

所以必须：

> 看 latent usage diagnostics。

---

# 237. 这也是为什么代码阅读不能只看 Inference

要同时看：

- training graph；
- loss；
- ablation；
- latent statistics。

---

# 238. Common Misconception 1：有 z 就代表模型一定学了 Latent Information

**错误。**

Decoder可以完全忽略 z。

---

# 239. Common Misconception 2：Posterior Collapse = σ²→0

**错误。**

标准 collapse趋向：

\[
q(z|x)=N(0,I)
\]

所以：

\[
\sigma^2\rightarrow1
\]

---

# 240. Common Misconception 3：μ→0 就一定 Collapse

单独看：

\[
\mu
\]

不够。

Posterior variance和 decoder usage也重要。

---

# 241. Common Misconception 4：KL→0 是好事，因为 Regularization 完美

对于 latent representation目标：

> 可能是严重警报。

KL=0表示 posterior完全匹配 prior，

也可能意味着：

> latent不携带 data information。

---

# 242. Common Misconception 5：KL 越大 Latent 越好

**错误。**

可能只是：

> posterior与prior严重 mismatch。

---

# 243. Common Misconception 6：Nonzero KL 保证 Decoder 使用 z

**错误。**

必须测试 output sensitivity / reconstruction dependence。

---

# 244. Common Misconception 7：Posterior Collapse 只发生在 Text VAE

**错误。**

任何 latent-variable model：

> 如果 decoder能够绕开 latent，

都可能发生。

---

# 245. Common Misconception 8：强 Decoder 一定 Collapse

**错误。**

Powerful decoder提高风险，

但训练 dynamics和objective等共同决定。

---

# 246. Common Misconception 9：Collapse 的唯一原因是 Decoder 太强

**过度简化。**

Lagging inference等工作说明：

> optimization dynamics也非常重要。

---

# 247. Common Misconception 10：KL Annealing 能保证不 Collapse

**不能。**

它是 heuristic。

---

# 248. Common Misconception 11：Free Bits 强制每一维一定携带 λ bits 的真实 Mutual Information

**不严格。**

它修改：

> KL penalty structure。

KL并不等于纯 MI。

---

# 249. Common Misconception 12：β 越大越能学好 Latent

如果 β太大：

> 反而可能压低 information rate。

---

# 250. Common Misconception 13：β 越小越好

太小可能：

> prior mismatch，

让 inference-time prior latent不好用。

---

# 251. Common Misconception 14：ACT β=10 意味着 KL 一定比 L1 强10倍

**错误。**

还取决于：

> loss/gradient scales。

---

# 252. Common Misconception 15：ACT 的 L1+βKL 就必须被称为标准 ELBO

更准确：

> CVAE-style / β-weighted variational objective。

L1可对应 fixed-scale Laplace likelihood，

但 β和实际 scaling需要单独考虑。

---

# 253. Common Misconception 16：ACT inference z=0 说明 z 没用

**错误。**

z可以在 training-time解释 multimodality，

test-time再固定 prior center。

---

# 254. Common Misconception 17：ACT z=0 能 Work 就证明 Posterior Collapse

**错误。**

Healthy latent training也完全可以配 deterministic z=0 inference。

---

# 255. Common Misconception 18：ACT CVAE Ablation 证明 z 的每个维度都有语义

**错误。**

Ablation只证明：

> CVAE objective对human-data performance非常重要。

---

# 256. Common Misconception 19：35.3%→2% 证明 z=0 是 Average Human Style

**完全不能证明。**

---

# 257. Common Misconception 20：一个 Latent Dimension KL≈0，整个 Model 就 Collapse

**错误。**

可能只是：

> inactive dimension。

---

# 258. Common Misconception 21：所有32维都必须 Active

**错误。**

真实所需 intrinsic latent dimension可能小于32。

---

# 259. Common Misconception 22：高 Active Units 一定带来更高 Rollout Success

**不保证。**

Representation metric与task performance需分开测。

---

# 260. Common Misconception 23：Latent Traversal 改变动作，就证明 Dimension 是可解释因素

不够。

只能证明：

> decoder对该维敏感。

---

# 261. Common Misconception 24：Posterior Mean Cluster 就证明 Causal Style

**错误。**

Cluster是 correlation evidence。

---

# 262. Common Misconception 25：Mutual Information 高就等于 Disentangled

**错误。**

MI高只说明：

> 携带较多信息。

不说明信息如何组织。

---

# 263. Common Misconception 26：Posterior Collapse 和 Distribution Shift 是同一问题

**不是。**

Posterior collapse：

> latent-variable training failure。

Distribution shift：

> deployment state distribution mismatch。

---

# 264. Common Misconception 27：Posterior Collapse 和 Mode Collapse 是同一问题

**不是。**

Mode collapse通常更多关联 GAN：

> generator只覆盖数据少数 modes。

Posterior collapse：

> VAE latent posterior趋近 prior并被 decoder忽略。

---

# 265. Common Misconception 28：Posterior Collapse 和 Latent Variance Collapse 是同一个

**不是。**

再次记住：

\[
posterior\ collapse
\rightarrow
\sigma^2\approx1
\]

对于 standard-normal prior。

---

# 266. Common Misconception 29：Decoder Output 对随机 z 变化小就一定 Collapse

还需要确认：

> posterior z 是否有用。

Prior samples可能落在 decoder局部不敏感方向，

但 posterior variations可能仍有作用。

---

# 267. Common Misconception 30：只看 Total Training Loss 就能诊断 Posterior Collapse

**完全不够。**

至少拆开：

- reconstruction；
- KL；
- per-dim statistics；
- latent sensitivity。

---

# 268. 一张图理解 Healthy VAE

```text
x
│
▼
Encoder
│
├─ μ(x)
└─ σ(x)
   │
   ▼
z carries x-specific information
   │
   ▼
Decoder genuinely uses z
   │
   ▼
reconstruction
```

不同：

\[
x
\]

产生不同：

\[
q(z|x)
\]

同时整体又大致：

> 和 prior兼容。

---

# 269. 一张图理解 Posterior Collapse

```text
x₁ ─┐
x₂ ─┤
x₃ ─┤
x₄ ─┘
     │
     ▼
Encoder
     │
     ▼
q(z|x) ≈ N(0,I)
for almost every x
     │
     ▼
z tells decoder almost nothing
     │
     ▼
Decoder solves task without z
```

---

# 270. 一张图理解 CVAE Collapse

```text
condition c ──────────────┐
                          │
target y → Encoder → z    │
                  │       │
                  ▼       ▼
                 Decoder
                    │
                    ▼
                    y


Healthy:
decoder uses c + z

Collapsed:
decoder effectively uses c only
z ≈ prior noise
```

---

# 271. 一张图理解 ACT

```text
TRAINING

current qpos + future action chunk
              │
              ▼
        CVAE Encoder
              │
         μ, logvar
              │
              ▼
              z
              │
              ├──────────────┐
images ───────┤              │
qpos ─────────┤              │
              ▼              │
        Policy Transformer   │
              │              │
              ▼              │
        predicted chunk      │
              │              │
              └──── L1 ◄─────┘

μ,logvar
   │
   ▼
  KL
```

---

# 272. Healthy ACT Latent

```text
future action variation
↓
encoder
↓
z
↓
policy actually changes predicted chunk
↓
better reconstruction of human modes
```

---

# 273. Collapsed ACT Latent

```text
encoder
↓
μ ≈ 0
logvar ≈ 0
↓
z ≈ N(0,I)
↓
policy ignores latent token
↓
actions determined almost entirely by observation
```

---

# 274. ACT 的一个很有趣的极端

如果 human demonstrations其实非常 deterministic：

\[
A\approx f(O)
\]

那么最合理模型可能本来就是：

\[
\boxed{
z\text{ unused}
}
\]

这时“collapse”从 representation角度存在，

但对task performance：

> 不一定有害。

---

# 275. 所以必须问“为什么我们想要 z？”

ACT的答案：

> human demonstrations有 variability / nonstationarity。

因此只有在：

\[
A|O
\]

确实存在 residual variation时，

latent才有明确建模价值。

---

# 276. 一句话真正理解 Posterior Collapse

> **Posterior Collapse 不是“latent variance变成0”，而是 approximate posterior \(q_\phi(z|x)\) 或 CVAE 中的 \(q_\phi(z|c,y)\) 被优化得越来越像 prior，使不同样本不再通过 \(z\) 传递足够的样本特定信息；如果 decoder同时有能力依赖其他输入独立完成 reconstruction，它就会逐渐忽略 latent，于是模型虽然形式上仍有 Encoder、\(\mu\)、\(\log\sigma^2\) 和 sampling，功能上却退化成一个几乎不使用 \(z\) 的模型。**

---

# 277. 一句话理解 KL 与 Information

> **Expected KL 不只是“让 posterior长得像 prior”的正则项，它还对通过 latent传递的信息量收费：\(\mathbb E KL(q(z|x)\|p(z))=I(X;Z)+KL(q(z)\|p(z))\)，而在 CVAE 中对应地包含 \(I(Y;Z|C)\) 与 conditional aggregated-posterior mismatch，因此把 KL压到0会同时把 latent information压到0，但 nonzero KL本身又不能保证这些信息真的被 decoder有效利用。**

---

# 278. 一句话理解 ACT 中 β 的作用

> **ACT 的 \(L=L_1+\beta KL\) 可以看成一个 information-rate 与 action-reconstruction 的权衡：L1鼓励 posterior latent携带那些当前 observation无法独自解释的 human-action variation，而 KL又要求这些 posteriors不要离 \(\mathcal N(0,I)\) 太远，以便推理时使用 prior center \(z=0\) 仍落在模型熟悉的 latent region；β太强可能让 latent collapse，太弱则可能让 posterior很好用但 prior/zero latent不好用。**

---

# 279. 一句话理解 ACT z=0 与 Collapse 的关系

> **ACT 推理时固定 \(z=0\) 并不是 posterior collapse 的证据：一个健康的 ACT-CVAE完全可能在训练时依赖 posterior \(z\) 来区分不同 human demonstration modes，再在推理时人为选择 prior mean作为稳定的 canonical latent；真正的 collapse 是训练时改变、打乱或移除 posterior \(z\) 也几乎不影响 reconstruction，且 \(q(z|O,A)\) 本身趋近 \(\mathcal N(0,I)\)。**

---

# 280. 如果你真的想判断 ACT 有没有使用 z

不要只看：

```text
loss ↓
```

应该至少同时看：

\[
\boxed{
KL_{\text{total}}
}
\]

\[
\boxed{
KL_1,\ldots,KL_{32}
}
\]

\[
\boxed{
Var[\mu_j]
}
\]

\[
\boxed{
L_{\text{posterior}}
-
L_{z=0}
}
\]

\[
\boxed{
L_{\text{shuffled-z}}
-
L_{\text{correct-z}}
}
\]

\[
\boxed{
\left\|
\partial\hat A/\partial z
\right\|
}
\]

以及最终：

\[
\boxed{
\text{rollout ablations}
}
\]

这才叫真正研究：

> latent usage。

---

# 281. 下一篇建议：ACT Vision Pipeline

到这里：

- Transformer；
- Attention；
- QKV；
- Decoder Queries；
- CVAE；
- z；
- Posterior Collapse；
- Behavior Cloning；

都已经进入比较深的层级。

ACT 仍然有一个值得专门补齐的高级模块：

> **视觉到底怎样从 4 张 RGB 图变成 Transformer 可以读取的 1200 个 tokens？**

下一篇建议：

> **[ACT Vision Pipeline：4 张 RGB 图像到底怎样变成 1200 个 Transformer Tokens？](../robot-learning/act/vision-pipeline.md)**

会集中讲：

- ResNet18 在 ACT 中到底保留了哪一层；
- 为什么 480×640 最终是 15×20；
- receptive field；
- feature map一个 cell到底代表什么；
- channel=512是什么意思；
- `input_proj` 为什么还要做 \(1\times1\) Conv；
- 2D sine positional encoding；
- row/column positional components；
- 四个 camera为什么直接沿 width/token axis拼接；
- camera identity到底在哪里；
- flatten：
  \[
  [B,512,15,20]
  \rightarrow
  [300,B,512]
  \]
- 4 cameras：
  \[
  4\times300=1200
  \]
- 为什么 Transformer不直接吃 raw pixels；
- ResNet feature token和 ViT patch token的异同；
- action query最终怎样 cross-attend这些视觉 tokens。

---

## Primary Source：VAE

Diederik P. Kingma, Max Welling.

**Auto-Encoding Variational Bayes.**  
ICLR 2014 / arXiv 2013.

- arXiv: https://arxiv.org/abs/1312.6114
- PDF: https://arxiv.org/pdf/1312.6114

经典 VAE objective：

\[
\mathcal L
=
\mathbb E_{q_\phi(z|x)}
[
\log p_\theta(x|z)
]
-
KL(
q_\phi(z|x)
\|
p(z)
)
\]

本文关于：

- variational posterior；
- prior；
- KL；
- reparameterization；

的基础建立在该工作上。

---

## Classic Posterior-Collapse Source：Sentence VAE

Samuel R. Bowman, Luke Vilnis, Oriol Vinyals, Andrew M. Dai, Rafał Jozefowicz, Samy Bengio.

**Generating Sentences from a Continuous Space.**  
CoNLL 2016.

- arXiv: https://arxiv.org/abs/1511.06349
- ACL Anthology: https://aclanthology.org/K16-1002/
- PDF: https://aclanthology.org/K16-1002.pdf

这篇是 posterior-collapse讨论的经典来源之一。

作者观察到：

> 强大的 RNN decoder可以优先解释数据并忽略 global latent，使 KL cost趋近0。

并提出两种经典 training techniques：

### KL Cost Annealing

从：

\[
\beta=0
\]

逐渐增加到：

\[
1
\]

先允许 Encoder通过 z编码信息，

再逐步施加 prior regularization。

### Word Dropout / Historyless Decoding

削弱 Decoder直接获得的 local conditioning information，

迫使它更多依赖：

\[
z
\]

做预测。

---

## Rate–Distortion / Mutual Information Source

Alexander A. Alemi, Ben Poole, Ian Fischer, Joshua V. Dillon, Rif A. Saurous, Kevin Murphy.

**Fixing a Broken ELBO.**

- arXiv: https://arxiv.org/abs/1711.00464
- PDF: https://arxiv.org/pdf/1711.00464

该工作强调：

> 最大化 likelihood / ELBO 并不自动保证得到好的 latent representation。

并从：

- mutual information；
- rate；
- distortion；

角度分析 VAE representation tradeoff。

本文关于：

\[
\boxed{
\text{Rate}
\leftrightarrow
\text{Distortion}
}
\]

的解释主要以此为背景。

---

## Training-Dynamics Source

Junxian He, Daniel Spokoyny, Graham Neubig, Taylor Berg-Kirkpatrick.

**Lagging Inference Networks and Posterior Collapse in Variational Autoencoders.**  
ICLR 2019.

- arXiv: https://arxiv.org/abs/1901.05534
- PDF: https://arxiv.org/pdf/1901.05534

该工作把 posterior collapse解释为一个重要的：

> training-dynamics / inference-lag

问题。

作者发现 early training中：

> inference network不能快速跟上不断变化的 true posterior，

从而促使 generative model学会忽略 latent。

其方法通过：

> aggressive inference-network optimization

缓解 collapse。

这说明：

\[
\boxed{
\text{Powerful Decoder}
}
\]

虽然是经典解释，

但不是 posterior collapse唯一可能机制。

---

## Free Bits Source

Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, Max Welling.

**Improving Variational Inference with Inverse Autoregressive Flow.**

- arXiv: https://arxiv.org/abs/1606.04934
- PDF: https://arxiv.org/pdf/1606.04934

论文使用 modified objective 中的：

> **Free Bits**

思想，给 latent groups一个最低信息预算，使优化器不会因为继续把已经很小的 KL压向0而不断获得收益。

这成为后续 mitigation posterior collapse时经常使用的技术之一。

---

## ACT Primary Source

Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.

**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.**  
RSS 2023.

- arXiv: https://arxiv.org/abs/2304.13705
- PDF: https://arxiv.org/pdf/2304.13705
- Project: https://tonyzhaozh.github.io/aloha/
- Code: https://github.com/tonyzhaozh/act

ACT明确将 policy训练为 CVAE：

- Encoder训练时读取：
  - current joints；
  - demonstration action sequence；
- 输出：
  \[
  \mu,\sigma
  \]
  of style latent \(z\)；
- Policy读取：
  - images；
  - joints；
  - \(z\)；
- test-time Encoder被丢弃；
- \(z\)设为 prior mean：
  \[
  0
  \]

论文的人类数据 ablation报告：

> 移除 CVAE objective后，aggregate success 从约35.3%降到2%，而 deterministic scripted data中影响很小。

这说明 CVAE-style objective对人类 demonstration setting有显著作用，

但它本身并不报告：

- per-dimension latent utilization；
- latent mutual information；
- posterior-collapse diagnostics。

因此不应把这项 ablation扩大解释成：

> “所有32维都拥有清晰 style semantics”。

---

## ACT Official Loss

Official repository:

https://github.com/tonyzhaozh/act

`policy.py`:

https://github.com/tonyzhaozh/act/blob/main/policy.py

训练：

```python
a_hat, is_pad_hat, (mu, logvar) =
    self.model(
        qpos,
        image,
        env_state,
        actions,
        is_pad
    )

total_kld, dim_wise_kld, mean_kld =
    kl_divergence(
        mu,
        logvar
    )

all_l1 =
    F.l1_loss(
        actions,
        a_hat,
        reduction='none'
    )

l1 =
    (
        all_l1
        *
        ~is_pad.unsqueeze(-1)
    ).mean()

loss =
    l1
    +
    kl_weight * total_kld[0]
```

canonical：

\[
\boxed{
kl\_weight=10
}
\]

---

## ACT Official Latent

`detr/models/detr_vae.py`:

https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py

```python
self.latent_dim = 32

self.latent_proj =
    nn.Linear(
        hidden_dim,
        self.latent_dim * 2
    )

self.latent_out_proj =
    nn.Linear(
        self.latent_dim,
        hidden_dim
    )
```

training：

\[
h_{CLS}
\rightarrow
[\mu,\logvar]
\rightarrow
z
\rightarrow
Linear(32,512)
\]

inference则使用：

\[
\boxed{
z=0
}
\]

---

## Conditional-VAE Posterior Collapse Background

Arya D. McCarthy, Xian Li, Jiatao Gu, Ning Dong.

**Addressing Posterior Collapse with Mutual Information for Improved Variational Neural Machine Translation.**  
ACL 2020.

- ACL Anthology: https://aclanthology.org/2020.acl-main.753/
- PDF: https://aclanthology.org/2020.acl-main.753.pdf

该工作专门研究 conditional VAE / Transformer setting中的 posterior collapse，

通过：

- modified variational objective；
- explicit mutual-information encouragement；
- auxiliary target prediction；

促进 latent真正携带 target information。

它说明：

> conditional decoder已有强 source/context信息时，latent usage同样需要专门关注。

---

## 本文知识连接

### Generative Modeling

- [Latent Variable](./latent-variable.md)
- [VAE](./vae.md)
- [CVAE](./cvae.md)
- [Reparameterization Trick](./reparameterization-trick.md)
- ELBO
- β-VAE
- Rate–Distortion
- KL Annealing
- Free Bits

### Information Theory

- [KL Divergence](../mathematics/kl-divergence.md)
- Entropy
- Mutual Information
- Conditional Mutual Information

### Deep Learning

- [Backpropagation](../deep-learning/backpropagation.md)
- [Decoder](../deep-learning/transformer-decoder.md)
- Representation Learning

### ACT

- [CVAE in ACT](../robot-learning/act/cvae-in-act.md)
- [为什么 ACT 推理时令 z = 0？](../robot-learning/act/why-z-zero-at-inference.md)
- [ACT Training](../robot-learning/act/training.md)
- [ACT Architecture](../robot-learning/act/architecture.md)
- [ACT Complete Data Flow](../robot-learning/act/complete-data-flow.md)

### 下一步

- [ACT Vision Pipeline：4 张 RGB 图像到底怎样变成 1200 个 Transformer Tokens？](../robot-learning/act/vision-pipeline.md)
