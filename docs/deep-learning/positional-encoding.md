---
title: "Positional Encoding：没有 RNN 后，Transformer 怎么知道顺序？"
description: "从 Self-Attention 的 permutation equivariance 出发，严格推导 sinusoidal positional encoding，解释为什么直接加到 embedding 上就能提供位置、为什么 sin/cos 能表达相对位移，以及 ACT 如何把 1D/2D/learned position information 分别用于 action、image、joint 与 latent tokens。"
status: reviewed
pageType: concept
canonical: /deep-learning/positional-encoding
updated: "2026-09-15"
---

# Positional Encoding：没有 RNN 后，Transformer 怎么知道顺序？

在 [Self-Attention](./self-attention.md) 中，我们已经证明过一个非常关键的性质。

如果：

- 没有 positional information；
- 没有 causal / position-dependent mask；
- 只是普通 full Self-Attention；

那么对任意 permutation matrix：

$$
P
$$

都有：

$$
\boxed{
SA(PX)=P\,SA(X)
}
$$

也就是说：

> 输入 tokens 怎样重排，输出 representations 就跟着怎样重排。

这叫：

> **Permutation Equivariance**

这并不意味着 Self-Attention “什么都看不懂”。

它仍然可以根据 token content 判断：

> 谁和谁相关。

但它自己不知道：

> 谁是第 1 个、谁是第 2 个、谁在谁前面。

于是就出现一个很直接的问题。

---

## 1. Transformer 去掉 RNN 后，到底丢掉了什么？

RNN：

$$
h_t=f(h_{t-1},x_t)
$$

计算路径本身已经携带顺序：

```text
x₁
↓
h₁
↓
x₂
↓
h₂
↓
x₃
↓
h₃
```

所以：

$$
x_1
$$

和：

$$
x_3
$$

天然处在不同 computation steps。

---

Transformer 则希望一次并行处理：

$$
x_1,x_2,\ldots,x_n
$$

Self-Attention 不再沿：

$$
1\rightarrow2\rightarrow3
$$

的 recurrent chain 传播。

这带来了强大的并行性。

但代价是：

> **sequence order 不再由 computation graph 自动提供。**

所以原始 Transformer 论文 Section 3.5 非常直接地说：

> 因为模型既没有 recurrence，也没有 convolution，为了让模型使用 sequence order，必须注入 token 相对或绝对位置的信息。

这就是：

$$
\boxed{
\text{Positional Encoding}
}
$$

---

## 2. 一个最简单的反例

考虑：

```text
dog bites man
```

和：

```text
man bites dog
```

词集合完全相同：

$$
\{
dog,
bites,
man
\}
$$

但顺序不同。

语义完全不同。

如果模型只看到 token content，

而完全没有 position：

```text
dog
bites
man
```

只是：

> 三个 vectors。

换成：

```text
man
bites
dog
```

只是把这三行 vectors 换了位置。

Full Self-Attention 会跟着同样重排输出，

但没有额外信息告诉它：

> “谁是 position 0？”

> “谁在 bites 前面？”

> “谁在 bites 后面？”

所以必须人为给 token 加入：

> **position identity。**

---

## 3. 最直觉的解决方案：给每个位置一个向量

假设 token embedding：

$$
e_i
\in
\mathbb R^{d_{\text{model}}}
$$

再给 position $i$ 一个位置向量：

$$
p_i
\in
\mathbb R^{d_{\text{model}}}
$$

最终输入：

$$
\boxed{
x_i=e_i+p_i
}
$$

于是：

```text
dog at position 0
```

表示为：

$$
e_{\text{dog}}+p_0
$$

而：

```text
dog at position 2
```

表示为：

$$
e_{\text{dog}}+p_2
$$

两者不再相同。

---

## 4. 为什么只要 Position Vector 不同，就已经打破对称性？

原来：

$$
e_{\text{dog}}
$$

无论放在哪里都相同。

加入 position 后：

$$
e_{\text{dog}}+p_0
\neq
e_{\text{dog}}+p_2
$$

一般成立。

于是：

$$
Q=(E+P)W_Q
$$

$$
K=(E+P)W_K
$$

都会受到位置影响。

所以 attention score：

$$
QK^\top
$$

现在不仅依赖：

> token content，

还依赖：

> token position。

Permutation symmetry 被打破。

---

## 5. 为什么是“加”，而不是“拼接”？

原始 Transformer 选择：

$$
\boxed{
x_i=e_i+p_i
}
$$

而不是：

$$
[e_i;p_i]
$$

论文给出的直接结构事实是：

> positional encoding 与 embedding 具有同样的 $d_{\text{model}}$ 维度，因此可以相加。

但我们还可以进一步理解这个设计。

---

## 6. Addition 不改变 Transformer Width

如果：

$$
e_i,p_i
\in
\mathbb R^{512}
$$

相加后：

$$
x_i\in\mathbb R^{512}
$$

后面的：

$$
W_Q,W_K,W_V
$$

shape 不需要改变。

---

如果 concat：

$$
[e_i;p_i]
$$

会变成：

$$
1024
$$

维。

那么需要：

- 更大的 projection；
- 或额外 projection 压回 512。

所以 addition 是一个：

> 结构简单、参数经济的融合方式。

但这不意味着：

> concat 数学上绝对不可行。

它只是另一种 architecture design。

---

## 7. 更重要的是：Addition 仍然允许模型利用 Content 和 Position

因为 Linear Projection 满足：

$$
(e_i+p_i)W_Q
=
e_iW_Q
+
p_iW_Q
$$

所以：

$$
\boxed{
q_i
=
q_i^{content}
+
q_i^{position}
}
$$

同理：

$$
k_j
=
k_j^{content}
+
k_j^{position}
$$

因此：

$$
q_i^\top k_j
$$

展开：

$$
(
e_iW_Q+p_iW_Q
)^\top
(
e_jW_K+p_jW_K
)
$$

得到四类 interaction：

$$
\boxed{
\text{content-content}
}
$$

$$
\boxed{
\text{content-position}
}
$$

$$
\boxed{
\text{position-content}
}
$$

$$
\boxed{
\text{position-position}
}
$$

所以“直接相加”并不意味着位置会被简单淹没。

它会通过 Q/K projections 参与 attention matching。

---

## 8. 但 Addition 会不会把 Content 和 Position 混在一起？

会。

它们确实进入同一个：

$$
d_{\text{model}}
$$

representation。

Transformer 并没有保留：

```text
前 512 维 = content
后 512 维 = position
```

这样的硬分区。

但网络是端到端训练的。

Token embeddings、Q/K/V projections、后续 layers 都可以共同适应：

$$
e_i+p_i
$$

这种输入形式。

所以更准确地说：

> **Addition 提供一个共享 representation space，让模型学习怎样利用 content 与 position 的叠加信息。**

不要理解成：

> 网络一定能把两者“完美解码还原”。

这不是必要条件。

---

## 9. Positional Encoding 有两大类最直接方案

原始 Transformer 论文明确提到：

> positional encoding 有很多选择，包括 learned 和 fixed。

最常见的两类：

---

### Learned Positional Embedding

每个位置：

$$
i
$$

对应一个可训练向量：

$$
p_i
$$

例如：

```python
nn.Embedding(max_length, d_model)
```

训练直接学：

$$
p_0,p_1,\ldots,p_{L-1}
$$

---

### Fixed Positional Encoding

位置向量不是 learned parameter。

而是通过一个确定函数生成：

$$
p_i=f(i)
$$

原始 Transformer 选择：

> **sinusoidal positional encoding。**

---

## 10. 原始 Transformer 的 Sinusoidal Formula

对于 position：

$$
pos
$$

第：

$$
2i
$$

维：

$$
\boxed{
PE_{(pos,2i)}
=
\sin
\left(
\frac{
pos
}{
10000^{2i/d_{\text{model}}}
}
\right)
}
$$

第：

$$
2i+1
$$

维：

$$
\boxed{
PE_{(pos,2i+1)}
=
\cos
\left(
\frac{
pos
}{
10000^{2i/d_{\text{model}}}
}
\right)
}
$$

也就是说：

> 每两个 dimensions 共用一个 frequency。

一个用：

$$
\sin
$$

一个用：

$$
\cos
$$

---

## 11. 先把公式改写得更容易理解

定义：

$$
\omega_i
=
10000^{-2i/d_{\text{model}}}
$$

那么：

$$
PE_{(pos,2i)}
=
\sin(
\omega_i pos
)
$$

$$
PE_{(pos,2i+1)}
=
\cos(
\omega_i pos
)
$$

所以每一个 sin/cos pair 就是：

$$
\boxed{
[
\sin(\omega_i pos),
\cos(\omega_i pos)
]
}
$$

真正变化的是：

$$
\omega_i
$$

即频率。

---

## 12. 一个 d_model = 4 的超小例子

如果：

$$
d_{\text{model}}=4
$$

一共有两个 sin/cos pairs。

---

### 第一个 Pair

$$
i=0
$$

所以：

$$
10000^{0}=1
$$

得到：

$$
[
\sin(pos),
\cos(pos)
]
$$

---

### 第二个 Pair

$$
i=1
$$

因为：

$$
10000^{2/4}
=
100
$$

得到：

$$
[
\sin(pos/100),
\cos(pos/100)
]
$$

所以：

$$
\boxed{
PE(pos)
=
[
\sin(pos),
\cos(pos),
\sin(pos/100),
\cos(pos/100)
]
}
$$

---

## 13. position = 0 是什么？

$$
PE(0)
=
[
0,1,0,1
]
$$

因为：

$$
\sin0=0
$$

$$
\cos0=1
$$

---

position = 1：

$$
PE(1)
=
[
\sin1,
\cos1,
\sin0.01,
\cos0.01
]
$$

近似：

$$
[
0.8415,
0.5403,
0.0100,
0.99995
]
$$

---

position = 2：

$$
[
\sin2,
\cos2,
\sin0.02,
\cos0.02
]
$$

不同 positions 得到不同 pattern。

---

## 14. 为什么不同 Dimensions 使用不同 Frequency？

如果所有 dimensions 都使用：

$$
\sin(pos)
$$

那么 512 维只是在重复同一个数字，

没有意义。

原论文让 wavelengths 按几何级数变化。

论文表述是：

> wavelengths form a geometric progression from $2\pi$ to $10000\cdot2\pi$。

所以模型拥有：

- 高频变化 dimensions；
- 低频变化 dimensions。

---

## 15. 高频和低频可以怎样直觉理解？

这是一个数学直觉，而不是论文唯一官方解释。

高频：

$$
\sin(pos)
$$

位置稍微变化，

数值可能变化很多。

所以对：

> 局部 position difference

很敏感。

---

低频：

$$
\sin(pos/10000)
$$

位置变化很多以后，

数值才慢慢变化。

所以可以提供：

> 更大尺度的位置变化模式。

因此多频率一起使用有点像：

> 用多个不同尺度的“时钟”共同描述一个位置。

---

## 16. 一个“多只时钟”的类比

想象你只看：

> 秒针。

60 秒以后它回到原位。

你就不知道：

> 是第 1 分钟还是第 100 分钟。

但如果同时看：

- 秒针；
- 分针；
- 时针；

它们的组合状态会更容易区分时间。

Sinusoidal encoding 也是类似：

> 不同频率的周期信号组合起来形成位置 signature。

当然实际公式不是普通时钟整数周期，

这个类比只是帮助理解多尺度频率的价值。

---

## 17. 但 Sin/Cos 是周期函数，不会重复吗？

单独一个：

$$
\sin(\omega pos)
$$

当然会周期重复。

但 positional vector 同时包含大量不同 frequencies：

$$
\omega_0,\omega_1,\ldots
$$

整个 vector 要完全重复，

必须让许多不同周期同时对齐。

因此组合能在很大范围内产生丰富的位置区分。

不过不要把它夸张成：

> “任何无限大的整数位置都 mathematically guaranteed 唯一。”

工程上真正关心的是：

> 在模型工作长度范围内提供有效的位置结构。

---

## 18. 为什么一定同时用 Sin 和 Cos？

这是 Sinusoidal Encoding 最漂亮的一点之一。

如果只用：

$$
\sin(\omega pos)
$$

从：

$$
\sin\theta
$$

一个数，

无法唯一知道相位方向。

而一对：

$$
[
\sin\theta,
\cos\theta
]
$$

可以看成单位圆上的一个点。

它完整表示：

> 当前 phase。

更重要的是：

> **固定位置偏移可以写成这一对向量的线性变换。**

这就是原论文特别强调的性质。

---

## 19. 原论文最重要的 Positional Encoding 理由

论文说作者选择这个函数，是因为他们假设它能让模型更容易学习：

> relative positions。

原因是：

> 对任意固定 offset $k$，$PE_{pos+k}$ 都能表示为 $PE_{pos}$ 的线性函数。

我们现在直接证明。

---

## 20. 先看一个 Frequency Pair

定义：

$$
\theta
=
\omega pos
$$

固定 offset：

$$
k
$$

对应 phase offset：

$$
\delta
=
\omega k
$$

当前位置 pair：

$$
p(pos)
=
\begin{bmatrix}
\sin\theta\\
\cos\theta
\end{bmatrix}
$$

我们想求：

$$
p(pos+k)
$$

即：

$$
\begin{bmatrix}
\sin(\theta+\delta)\\
\cos(\theta+\delta)
\end{bmatrix}
$$

---

## 21. 使用三角加法公式

$$
\sin(\theta+\delta)
=
\sin\theta\cos\delta
+
\cos\theta\sin\delta
$$

$$
\cos(\theta+\delta)
=
\cos\theta\cos\delta
-
\sin\theta\sin\delta
$$

写成矩阵：

$$
\boxed{
\begin{bmatrix}
\sin(\theta+\delta)\\
\cos(\theta+\delta)
\end{bmatrix}
=
\begin{bmatrix}
\cos\delta & \sin\delta\\
-\sin\delta & \cos\delta
\end{bmatrix}
\begin{bmatrix}
\sin\theta\\
\cos\theta
\end{bmatrix}
}
$$

---

## 22. 这意味着什么？

对于固定：

$$
k
$$

和固定 frequency：

$$
\omega
$$

矩阵：

$$
R_k
=
\begin{bmatrix}
\cos(\omega k) & \sin(\omega k)\\
-\sin(\omega k) & \cos(\omega k)
\end{bmatrix}
$$

只取决于：

$$
k
$$

不取决于：

$$
pos
$$

所以：

$$
\boxed{
p(pos+k)
=
R_k p(pos)
}
$$

这就是：

> **固定 relative offset 对应一个固定线性变换。**

---

## 23. 这实际上是一个 Rotation

矩阵：

$$
R_k
$$

本质上是一个二维旋转矩阵的等价排列形式。

因为 sin/cos pair 位于单位圆：

$$
\sin^2\theta+\cos^2\theta=1
$$

增加 position：

$$
pos\rightarrow pos+k
$$

相当于：

> 在这个 frequency 对应的单位圆上旋转固定角度：

$$
\omega k
$$

所以 Positional Encoding 可以被理解成：

> 每个 frequency pair 都是一只以不同速度旋转的二维“相位时钟”。

---

## 24. 多 Frequency 时会发生什么？

整个：

$$
PE(pos)
$$

由很多二维 pairs 组成：

$$
p_0(pos),
p_1(pos),\ldots
$$

对固定 offset：

$$
k
$$

每一个 pair 都有自己的 rotation：

$$
R_{k,0},
R_{k,1},
\ldots
$$

所以整体：

$$
PE(pos+k)
$$

可以由一个 block-diagonal linear transform 作用在：

$$
PE(pos)
$$

上得到。

这就是原论文那句：

> $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$

背后的具体数学。

---

## 25. 这是不是说明 Transformer 会自动精确计算 Relative Distance？

不是。

必须区分：

#### Mathematical Property

Sinusoidal encoding 确实具有：

> fixed offset → fixed linear relation。

#### Model Behavior

模型是否真的学会利用这个结构：

> 由训练决定。

原论文说的是：

> 作者**假设**这个性质会让模型更容易学习 relative-position attention。

并不是证明：

> Transformer 必然恢复精确距离。

---

## 26. 为什么 learned Position Embedding 也能工作？

原始 Transformer 还实验了：

> learned positional embeddings。

结果和 sinusoidal 版本：

> nearly identical。

所以模型并不依赖：

> sin/cos 本身具有某种不可替代的魔法。

只要提供：

> 足够有区分度的位置 representation，

网络就可以学习使用。

---

## 27. 那原论文为什么最后选择 Sinusoidal？

论文给出的理由是：

> sinusoidal encoding **可能**让模型 extrapolate 到训练时未见过的更长 sequence。

因为固定函数：

$$
PE(pos)
$$

可以直接计算任意：

$$
pos
$$

而 learned table：

```text
p₀
p₁
...
p₅₁₁
```

如果训练和参数表只定义到 512，

position 1000 就没有现成 learned vector。

注意原论文用的是：

> “may allow”

不是证明一定能很好 extrapolate。

---

## 28. Learned Embedding 的优势是什么？

Learned position：

$$
p_i
$$

不需要服从预设 sin/cos structure。

模型可以自己找到：

> 最适合任务的位置表示。

优点：

- 灵活；
- 简单；
- 常常效果很好。

缺点之一：

> 通常绑定到预定义最大长度。

而且没有 sinusoidal 那种显式 relative-shift linear structure。

---

## 29. Fixed Sinusoidal 有参数吗？

没有 learned positional parameters。

$$
PE(pos)
$$

由：

- position；
- $d_{\text{model}}$；
- 常数 10000；

直接计算。

所以它是一张：

> deterministic lookup table / function。

虽然最终网络会学习：

> 怎样使用它。

---

## 30. 为什么 10000？

这是原始 Transformer 的 design choice。

它控制 frequency / wavelength 范围。

不要赋予：

$$
10000
$$

某种深刻自然常数意义。

真正结构思想是：

> 使用按几何尺度分布的多个 sinusoidal frequencies。

10000 是该具体参数化中的 base。

---

## 31. Position Encoding 是不是 Token 的“第几个位置”这个整数本身？

不是。

模型通常不是直接把：

$$
pos=37
$$

这个 scalar 输入 Attention。

而是把它映射成：

$$
PE(37)
\in
\mathbb R^{d_{\text{model}}}
$$

这让位置和 token embedding：

> 处在同一个 vector dimensionality 中。

然后可以共同参与：

$$
Q/K/V
$$

projections。

---

## 32. 为什么不用一个 Scalar Position？

例如直接输入：

$$
x_i=e_i+[i,i,\ldots,i]
$$

可能导致很多问题：

- 数值尺度随 length 线性增大；
- 表达结构单一；
- 只有一个位置自由度方向；
- 不同尺度关系不丰富。

Sinusoidal encoding 把一个 scalar position：

$$
pos
$$

展开成：

> 多频率高维 feature representation。

这是一种经典的 feature mapping 思想。

---

## 33. Positional Encoding 和 Fourier Features 有关系吗？

从数学形式看：

> 有明显联系。

Sin/cos functions 可以作为 frequency basis。

现代机器学习里也大量使用：

> Fourier features / frequency features

把低维坐标映射到高维周期特征。

但要注意历史与定义边界：

> 原始 Transformer 论文直接提出的是 sinusoidal positional encoding，并没有把它用今天所有 Fourier-feature 理论来包装。

可以从现代视角建立联系，

但不要把后来的理论反向写成原论文主张。

---

## 34. Position 信息最终怎样影响 Attention Score？

假设：

$$
x_i=e_i+p_i
$$

那么：

$$
q_i=(e_i+p_i)W_Q
$$

$$
k_j=(e_j+p_j)W_K
$$

所以：

$$
s_{ij}
=
q_i^\top k_j
$$

展开后包含：

$$
(e_iW_Q)^\top(e_jW_K)
$$

content-content，

加：

$$
(e_iW_Q)^\top(p_jW_K)
$$

content-position，

加：

$$
(p_iW_Q)^\top(e_jW_K)
$$

position-content，

以及：

$$
(p_iW_Q)^\top(p_jW_K)
$$

position-position。

因此：

> Attention 可以联合利用“你是谁”和“你在哪里”。

---

## 35. 为什么这对语言很重要？

模型可能需要学习：

#### Local Relation

> “我前一个 token 是谁？”

#### Directional Relation

> “某个 noun 在 verb 前还是后？”

#### Long-Range Relation

> “距离我 20 个位置的 token 是否相关？”

Position information 让 Q/K compatibility 有能力区分：

$$
j=i-1
$$

和：

$$
j=i+1
$$

即使两个位置 token content 相似。

---

## 36. 为什么 Causal Mask 不能替代 Position Encoding？

Causal Mask 只告诉 Decoder：

> 哪些位置允许访问。

例如 position 5 可以看：

$$
0,1,2,3,4,5
$$

不能看：

$$
6,7,\ldots
$$

但在允许区域内部，

如果没有额外 position representation，

它仍然缺少精确：

> “这是第几个？”

的信息。

Mask 提供的是：

> visibility structure。

Position Encoding 提供的是：

> position identity / geometry。

二者不同。

---

## 37. 反过来，Position Encoding 也不能替代 Causal Mask

即使模型知道：

$$
position=10
$$

如果 full Attention 允许它直接看到：

$$
position=11
$$

训练 autoregressive next-token prediction 时仍然会泄漏未来。

所以：

$$
\boxed{
Position
\neq
Causality
}
$$

Position 告诉：

> where。

Mask 规定：

> what is allowed to be seen。

---

## 38. Absolute Position 和 Relative Position 是什么区别？

#### Absolute

直接表示：

> “我是 position 37。”

例如：

$$
p_{37}
$$

---

#### Relative

直接表示：

> “j 相对 i 偏移 +4。”

例如：

$$
r_{j-i}
$$

原始 sinusoidal encoding表面上是：

> absolute positional encoding，

因为它给每个：

$$
pos
$$

一个 vector。

但由于 sin/cos 的线性 shift property，

作者希望模型能容易学到：

> relative offset。

---

## 39. 现代 Transformer 都还用原始 Sinusoidal 吗？

不一定。

后来出现很多 position mechanisms，例如：

- learned absolute embedding；
- relative position representations；
- relative position bias；
- RoPE；
- ALiBi；
- 2D / 3D position encodings；
- modality-specific positions。

所以：

$$
\boxed{
\text{Transformer}
\neq
\text{必须使用原始 sinusoidal PE}
}
$$

原始 sinusoidal 是：

> 2017 Transformer 的经典实现。

---

## 40. 为什么 Position Encoding 在 Vision 更复杂？

Text 主要是一维：

$$
0,1,2,\ldots,n-1
$$

Image 是二维：

$$
(y,x)
$$

例如：

$$
15\times20
$$

feature map 中：

```text
row 3, col 7
```

和：

```text
row 7, col 3
```

空间位置不同。

如果直接 flatten：

$$
15\times20
\rightarrow300
$$

只给一个 flat index：

$$
0\ldots299
$$

虽然理论上仍能编码某些位置，

但不直接体现二维 row / column geometry。

所以 Vision Transformer 系统经常使用：

> 2D positional information。

---

## 41. ACT 为什么特别需要 2D Position？

ACT 每张 RGB image：

$$
480\times640\times3
$$

经过 ResNet18：

$$
15\times20\times512
$$

也就是每张图像有：

$$
300
$$

spatial feature positions。

论文明确写：

> flatten spatial dimension 得到 $300\times512$，为了保留 spatial information，加入 2D sinusoidal position embedding。

因为 flatten 后，

如果没有 position：

> Transformer 只看见 300 个 feature vectors，

不知道它们原本来自：

- 左上；
- 中央；
- 右下。

---

## 42. 2D Sinusoidal Position Encoding 的核心想法

每个 spatial token 有：

$$
(y,x)
$$

两个坐标。

可以分别编码：

$$
PE_y(y)
$$

和：

$$
PE_x(x)
$$

然后合并：

$$
\boxed{
PE_{2D}(y,x)
=
[
PE_y(y);
PE_x(x)
]
}
$$

所以位置向量同时告诉模型：

> 这是第几行，

和：

> 这是第几列。

---

## 43. ACT 官方代码具体怎么做？

ACT 的 position encoding 代码继承 DETR 风格。

对于：

$$
hidden\_dim=512
$$

构建时：

```python
N_steps = hidden_dim // 2
```

所以：

$$
N_{\text{steps}}=256
$$

`PositionEmbeddingSine` 分别产生：

- $256$-D 的 $y$ encoding；
- $256$-D 的 $x$ encoding。

然后：

$$
\boxed{
256+256=512
}
$$

拼成最终 2D positional representation。

---

## 44. 官方 2D Code 的核心结构

概念上：

```python
y_embed = coordinate_y
x_embed = coordinate_x

pos_y = sinusoidal(y_embed)
pos_x = sinusoidal(x_embed)

pos = concat(pos_y, pos_x)
```

所以最终每个：

$$
(y,x)
$$

位置都有：

$$
512
$$

维 PE。

这和 visual feature：

$$
512
$$

维匹配。

---

## 45. ACT Code 还会 Normalize x/y Coordinates

官方 `PositionEmbeddingSine` 在：

```python
normalize=True
```

时会把：

$$
x,y
$$

位置尺度归一化，

再乘：

$$
2\pi
$$

的 scale。

因此它不是简单把 row index：

$$
1,2,\ldots,15
$$

原样塞进原始文本公式。

这是 DETR 为图像二维位置做的一个泛化实现。

---

## 46. 为什么 Scale = 2π 很自然？

Sin/cos 的一个完整周期是：

$$
2\pi
$$

归一化坐标后乘：

$$
2\pi
$$

可以让 coordinate domain 映射到一个自然的角度尺度。

但真正编码仍包含：

> 多个 frequency scales。

所以不是说整张图只用一个正弦周期。

---

## 47. ACT 4 个 Cameras 怎么处理 Position？

论文：

> 对每张图像分别经过 ResNet、flatten、加 2D sinusoidal position embedding，然后将四张 feature sequences 拼接。

官方代码也会收集每个 camera 的：

- features；
- position encoding；

然后拼起来。

概念上：

$$
300\times512
$$

每张，

四张：

$$
1200\times512
$$

---

## 48. 官方代码的一个值得知道的实现细节

代码先对每张 camera feature map 分别生成二维 position grid，

然后把多个 cameras 的 feature maps 和 position maps 沿空间维拼接。

因此 2D encoding 的主要语义是：

> **camera 内部的 spatial coordinate。**

当前这份官方实现中没有在这段 position-encoding code 里额外加入一个显式：

> camera-ID positional embedding。

不同 camera 的区分主要来自：

- 不同图像内容；
- 固定输入组织；
- 网络学习到的 representation。

这属于：

> released-code implementation detail，

不是理解 ACT Positional Encoding 必须背诵的理论核心。

---

## 49. ACT 不只有 Visual Position Encoding

ACT 至少有三类位置/slot information 值得区分：

1. CVAE encoder 的 sequence position；
2. Policy encoder 的 image / joint / latent position；
3. Decoder action-query position。

如果把它们全部叫：

> positional encoding

但不区分来源，很容易混乱。

---

## 50. 第一类：CVAE Encoder 的 1D Sinusoidal Table

Training 时 CVAE encoder 输入：

```text
[CLS]
qpos
a₀
a₁
...
aₖ₋₁
```

总长度：

$$
k+2
$$

官方代码注册：

```python
pos_table =
get_sinusoid_encoding_table(
    1 + 1 + num_queries,
    hidden_dim
)
```

即为：

- `[CLS]`；
- qpos；
- action sequence；

提供一维 sinusoidal positions。

---

## 51. 为什么 CVAE Action Sequence 需要 Position？

如果 action token：

$$
a_0,a_1,a_2,a_3
$$

只看内容，

Self-Attention 不天然知道：

> 哪个动作更早，哪个动作更晚。

但 action chunk 的时间结构非常重要：

$$
a_t
\neq
a_{t+3}
$$

即使两个 joint vectors 数值接近。

所以：

> action tokens 必须携带时间 slot identity。

---

## 52. `[CLS]` 和 qpos 也占 Position Slot

官方 table 长度：

$$
1+1+k
$$

意味着：

```text
position 0 → [CLS]
position 1 → qpos
position 2 → first action
position 3 → second action
...
```

它们一起作为 Transformer Encoder sequence。

所以 CVAE encoder 并不是：

> 只给 actions 加 position，

而是整个 input sequence 有对应位置编码。

---

## 53. 第二类：Policy Encoder 的 2D Visual Position

视觉 feature grid：

$$
15\times20
$$

使用：

> 2D sine position encoding。

目的：

> 保留 spatial structure。

---

## 54. joint 和 latent token 没有 2D Image Coordinate 怎么办？

官方代码：

```python
self.additional_pos_embed =
    nn.Embedding(2, hidden_dim)
```

这两个 learned position embeddings 分别对应：

- latent token；
- proprio / joint token。

随后：

> 它们被 prepend 到 visual positional sequence。

所以 Policy Encoder 的最终 position information 概念上是：

```text
latent token
→ learned positional identity

joint token
→ learned positional identity

visual tokens
→ 2D sinusoidal spatial positions
```

---

## 55. 为什么 joint / latent 不应该强行给一个 (x,y)？

因为它们不是图像 patch。

joint token 没有：

> “图像第 7 行第 12 列”

这种空间含义。

latent token 也没有。

所以给这两个 special tokens 独立 learned positional identities：

> 比伪造一个 image coordinate 更合理。

---

## 56. 第三类：Action Query Position

ACT Decoder 要输出：

$$
k
$$

个 future action slots。

这些 slots 必须互相可区分：

```text
slot 0
slot 1
...
slot k-1
```

否则如果每个 slot 完全相同，

整个 Decoder 对这些 output positions 会有很强的对称性。

所以需要：

> action-query positional identity。

---

## 57. 论文和 Released Code 在这里有差异

ACT 论文正文描述：

> Transformer decoder input sequence is a fixed position embedding，shape $k\times512$。

Appendix C 进一步写：

> first-layer queries are fixed sinusoidal embeddings。

但是当前官方仓库：

```python
self.query_embed =
    nn.Embedding(
        num_queries,
        hidden_dim
    )
```

使用的是：

> learned query embeddings。

所以需要明确写：

$$
\boxed{
\text{Paper Fact}
\neq
\text{Released-Code Detail}
}
$$

---

## 58. 但两者解决的是同一个核心问题

无论：

#### Fixed Sinusoidal

还是：

#### Learned Query Embedding

目的之一都是：

> 让第 0、1、2、...、$k-1$ 个 action slots 具有不同 identity。

这样：

$$
q_i
$$

和：

$$
q_j
$$

才不必因初始完全对称而产生一样的行为。

---

## 59. Position Encoding 和 Query Embedding 是不是完全一样？

不应该简单画等号。

`query_embed` 在 DETR-style decoder 里更像：

> learned output-slot identity / query positional representation。

而普通 token positional encoding：

> 给已有 content token 提供位置。

二者功能有重叠：

> 都打破 slot/position 对称性。

但 architecture context 不同。

所以最好区分：

- token positional encoding；
- query positional embedding。

---

## 60. DETR-style Transformer 如何实际加入 Position？

ACT 官方 `transformer.py` 没有简单地一开始永久做：

$$
src=src+pos
$$

然后之后所有地方都只用 src。

它采用一个 helper：

```python
with_pos_embed(tensor, pos):
    return tensor + pos
```

然后在 Attention 中：

```text
Q = content + position
K = content + position
V = content
```

这种模式非常值得理解。

---

## 61. Encoder Self-Attention 中

官方代码概念上：

$$
q=k=src+pos
$$

但：

$$
value=src
$$

于是：

> Position 直接参与“谁和谁匹配”。

真正被传递的 Value：

> 主要是 content representation。

这和我们前面说的：

> position 特别影响 routing

非常一致。

---

## 62. Decoder Cross-Attention 中

官方代码概念上：

$$
query=tgt+query\_pos
$$

$$
key=memory+pos
$$

$$
value=memory
$$

所以：

#### Query Side

output slot content + query identity

#### Key Side

memory content + memory position

#### Value Side

memory content

这使 Cross-Attention score 同时知道：

> “哪个 output slot 在问”

和：

> “memory 信息位于哪里”。

---

## 63. 为什么这种实现不和“Embedding + PE”原论文冲突？

它们是不同具体 architecture 的位置注入方式。

原始 Transformer：

> 在 Encoder/Decoder stack 底部把 PE 加到 token embeddings。

DETR-style architecture：

> 还会在每层 Attention 的 Q/K 构造时显式重新加入 position embeddings。

所以不能把：

> “Transformer 使用位置编码”

误解成只有一种固定代码写法。

核心原则是：

$$
\boxed{
\text{Attention must have access to positional structure}
}
$$

具体怎样注入：

> 是 architecture design choice。

---

## 64. 为什么 Position 特别适合加到 Q/K？

因为 Q/K 负责：

> routing / matching。

如果你想让模型学：

> “优先看附近位置”

或者：

> “这个 action slot 读取图像左侧”

position 必须影响：

$$
q^\top k
$$

也就是 attention logits。

把 position 加进 Q/K source：

> 直接让位置影响匹配。

---

## 65. Value 完全不需要 Position 吗？

不能泛化成绝对规则。

原始 Transformer 的 embedding 本身已经包含 position：

$$
e+p
$$

所以 Value 也间接包含位置。

DETR-style code 则在某些 attention calls 里：

> positional tensor 显式只加到 Q/K，而 Value 使用 content。

这只是具体设计。

现代 architecture 有很多不同做法。

所以更安全的知识结论是：

> **位置必须以某种方式影响 network；具体进入 Q/K/V 哪些路径取决于 architecture。**

---

## 66. 为什么 Position Encoding 不等于“给模型一个计数器”？

因为模型不会显式执行：

```python
if position == 5:
    ...
```

它得到的是高维 vector：

$$
p_i
$$

然后通过 learned projections：

$$
W_Q,W_K,\ldots
$$

形成连续几何关系。

所以 Position Encoding 更像：

> 把离散位置映射到一个神经网络可以通过线性代数利用的 feature space。

---

## 67. 为什么 Sinusoidal PE 对 Nearby Positions 看起来相似？

对于低频 dimensions：

$$
\sin(\omega pos)
$$

如果：

$$
\omega
$$

很小，

从：

$$
pos
$$

到：

$$
pos+1
$$

变化：

$$
\approx \omega
$$

很小。

所以邻近 positions 的某些低频 features 变化平滑。

高频 dimensions则变化更明显。

因此整个 PE 同时有：

> 局部敏感 + 长尺度平滑

的多尺度结构。

---

## 68. PE(pos) 和 PE(pos+1) 一定非常相似吗？

不应绝对化。

相似程度取决于：

- $d_{\text{model}}$；
- frequency distribution；
- similarity metric。

有些 high-frequency dimensions 变化明显，

有些 low-frequency dimensions 变化很小。

所以正确理解是：

> encoding 在多种频率尺度上连续变化。

不是简单：

> “相邻位置的向量一定几乎一样。”

---

## 69. Sinusoidal PE 会随着 Position 变大而数值爆炸吗？

不会。

因为：

$$
-1\le\sin(\cdot)\le1
$$

$$
-1\le\cos(\cdot)\le1
$$

所以所有 positional dimensions 始终有界。

这相比直接输入巨大：

$$
pos
$$

数值有一个很自然的稳定性优势。

---

## 70. 为什么 Pair 的 Norm 是固定的？

每一个 frequency pair：

$$
[
\sin\theta,
\cos\theta
]
$$

都有：

$$
\sin^2\theta+\cos^2\theta=1
$$

所以每个 pair 始终位于单位圆。

如果总共有：

$$
d_{\text{model}}/2
$$

对，

忽略实现细节，

整个 pure sinusoidal PE 的 squared norm 是：

$$
d_{\text{model}}/2
$$

因此它的整体 magnitude 不随：

$$
pos
$$

增长。

这也是一个漂亮性质。

---

## 71. Position 向量 Norm 固定意味着什么？

它避免：

> 越靠后的 token 仅仅因为 position 数字大，就获得越来越大 positional magnitude。

position 主要通过：

> phase pattern

而不是：

> vector length 不断增长

来编码。

这和直接：

$$
[1,1,\ldots], [2,2,\ldots], [1000,1000,\ldots]
$$

非常不同。

---

## 72. Sin/Cos Pair 还能直接表示 Relative Offset 的 Dot Product

考虑同一个 frequency：

$$
p(a)
=
[
\sin(\omega a),\cos(\omega a)
]
$$

$$
p(b)
=
[
\sin(\omega b),\cos(\omega b)
]
$$

点积：

$$
p(a)^\top p(b)
$$

等于：

$$
\sin(\omega a)\sin(\omega b)
+
\cos(\omega a)\cos(\omega b)
$$

利用：

$$
\cos(x-y)
=
\cos x\cos y+\sin x\sin y
$$

得到：

$$
\boxed{
p(a)^\top p(b)
=
\cos(
\omega(a-b)
)
}
$$

这非常漂亮。

---

## 73. 这说明什么？

同 frequency pair 的 inner product：

> 只依赖两个 positions 的差：

$$
a-b
$$

而不依赖它们的绝对共同平移。

例如：

$$
(5,8)
$$

和：

$$
(100,103)
$$

relative offset 都是：

$$
3
$$

该 pair 的 dot product 相同。

这进一步说明：

> sin/cos structure 天然包含 relative-position geometry。

---

## 74. 但完整 Transformer Attention 不是直接 PE·PE

注意：

真正 Q/K：

$$
(E+P)W_Q
$$

和：

$$
(E+P)W_K
$$

还混合：

- content；
- learned projections；
- multi-head structure。

所以：

$$
PE_i^\top PE_j
=
f(i-j)
$$

这个漂亮性质只是：

> Positional Encoding 自身的数学结构。

不等于：

> 最终 attention score 只由 relative distance 决定。

---

## 75. 为什么这种数学结构对 Learning 有帮助？

因为网络不必从完全无结构的 arbitrary position IDs 中：

> 从零发现所有 relative geometry。

Sin/cos 给了它一个具有：

- multi-scale periodicity；
- linear shift relation；
- relative-difference inner-product structure；

的位置 basis。

但到底利用多少：

> 由 training objective 决定。

---

## 76. Positional Encoding 会不会覆盖 Token Embedding？

两者直接相加：

$$
e_i+p_i
$$

所以确实共享 numerical scale。

原始 Transformer 对 word embeddings 还乘：

$$
\sqrt{d_{\text{model}}}
$$

再与 positional encoding 相加。

这会影响 content 与 PE 的初始尺度关系。

不过网络后续还有：

- projections；
- residual；
- normalization；

并联合训练。

所以不用把它想成：

> “两组信息在 512 个格子里互相覆盖掉。”

它们是 vector superposition，

后续线性/非线性层利用组合结构。

---

## 77. 为什么 PE 通常不单独做一个 Attention Token？

理论上可以设计 special position tokens，

但原始 Transformer 使用：

> position vector 直接附着到每个 token representation。

因为位置本身是：

> 每个 token 的属性。

“dog 在 position 3”

不是：

> 一个独立 token 与 dog 无关。

所以：

$$
e_{\text{dog}}+p_3
$$

是一种自然融合方式。

---

## 78. 2D Image PE 里的 x 和 y 为什么通常分开？

因为二维位置有两个独立坐标：

$$
(x,y)
$$

如果只用 flatten index：

$$
i=yW+x
$$

模型理论上可以学习映射，

但 row/column structure 被隐式化。

把：

$$
PE_x(x)
$$

和：

$$
PE_y(y)
$$

分别编码，

可以显式告诉模型：

> 横向位置和纵向位置。

然后合并成完整空间位置。

---

## 79. ACT 的 Feature Grid 为什么是 15 × 20？

论文中每张：

$$
480\times640
$$

RGB image 经 ResNet18：

$$
15\times20\times512
$$

所以空间降采样比例大约：

$$
32
$$

即：

$$
480/32=15
$$

$$
640/32=20
$$

每个 feature location 对应原图一个较大的 receptive-field region。

Positional encoding标记的是：

> feature-map spatial location，

不是单个 raw pixel 坐标。

---

## 80. Flatten 后 2D Structure 真的还在吗？

Tensor shape 变成：

$$
300\times512
$$

以后，

二维数组结构本身不再显式存在。

但每个 token 带有：

$$
PE(y,x)
$$

所以：

> 2D geometry 被编码进 token features / attention positions。

因此 Transformer 可以在 flattened sequence 上仍利用空间结构。

---

## 81. 为什么图像 Position 和文本 Position 本质是同一个问题？

Text：

> token 的顺序来自 1D coordinate。

Image：

> patch / feature 的位置来自 2D coordinate。

Video：

> 还可能加入 time coordinate。

Robot trajectories：

> action slot 有 temporal coordinate。

所以 Positional Encoding 更一般地说是在解决：

$$
\boxed{
\text{Token content alone does not specify token geometry}
}
$$

我们需要给模型：

> token 所在结构空间中的坐标 / identity。

---

## 82. 以后 3D / Embodied 模型会出现什么？

在 3D、point cloud、robotics 中可能有：

- $x,y,z$ spatial coordinates；
- time $t$；
- camera ID；
- embodiment/joint identity；
- modality identity。

这些都可以被看成更广义：

> structural / positional metadata。

Position Encoding 不必局限于：

> 文本里的“第几个词”。

---

## 83. Token Type Embedding 和 Positional Embedding 一样吗？

不一样。

Position：

> 我在哪里？

Type / modality embedding：

> 我是什么类型？

例如：

```text
visual token
joint token
latent token
```

可以有 type identity。

ACT 当前代码对 latent/joint 使用 `additional_pos_embed` 作为两个特殊位置 identity，

功能上也部分承担：

> special-token distinction。

但概念上最好区分：

- spatial/temporal position；
- token type / slot identity。

---

## 84. Camera ID 也不是普通 2D Position

对于四个 camera：

```text
top
front
left wrist
right wrist
```

camera identity 是：

> view/source identity。

每张图内：

$$
(x,y)
$$

是 spatial position。

这两个结构维度不同。

一个更复杂的模型可以同时拥有：

$$
PE_{spatial}(x,y)
+
E_{camera}(c)
$$

但 ACT 原论文核心描述主要强调：

> 每张图的 2D sinusoidal spatial encoding。

---

## 85. 为什么理解这种区别很重要？

因为以后读多模态模型时常会看到：

$$
x
=
content
+
position
+
segment
+
modality
+
role
$$

这些向量虽然都通过“加法”进入 hidden state，

但语义完全不同：

- position：在哪里；
- segment：属于哪段；
- modality：来自图像还是文字；
- role：system/user/assistant；
- camera：来自哪个视角。

不要因为都叫 embedding 就混成一个概念。

---

## 86. Positional Encoding 是不是训练目标？

通常不是单独监督：

```text
predict position 7
```

原始 Transformer 的 PE 是固定。

模型的最终 translation loss 会训练其他参数：

> 怎样利用 PE。

Learned positional embeddings则：

> 参数本身也接受任务 loss 的 gradient。

但同样没有单独位置标签 loss。

---

## 87. Learned PE 是怎样被训练的？

假设：

$$
p_i
$$

是 learned vector。

某个 training sample 使用 position $i$，

它进入：

$$
x_i=e_i+p_i
$$

最终 loss：

$$
L
$$

gradient 会回到：

$$
p_i
$$

所以模型逐渐学：

> position $i$ 应该用什么 representation 才有助于任务。

---

## 88. Fixed Sinusoidal PE 会收到 Gradient 吗？

通常：

> 不作为 trainable parameter 更新。

它就是固定 tensor。

但 gradient 仍然可以通过：

$$
x=e+p
$$

流向：

- embedding；
- Q/K/V weights；
- 后续 network。

只是：

$$
p
$$

本身不更新。

---

## 89. 为什么 Sinusoidal PE 能处理未见位置，但不保证泛化好？

因为：

$$
PE(10000)
$$

可以直接计算。

这是：

> **representational availability。**

但模型训练可能只见过：

$$
0\ldots512
$$

它的 learned Q/K/FFN 参数是否会对 position 10000 的新 phase pattern 正确响应：

> 不是公式自动保证的。

所以：

$$
\boxed{
\text{can compute unseen PE}
\neq
\text{guaranteed length generalization}
}
$$

这也是原论文只说：

> may allow extrapolation。

---

## 90. 为什么 Absolute Learned Embedding 通常更难直接外推？

因为参数表可能只有：

$$
p_0,\ldots,p_{L-1}
$$

位置：

$$
L
$$

以后根本没有 learned vector。

你必须：

- 扩表；
- 插值；
- 重新训练；
- 使用其他方法。

这是固定 function PE 的一个明显结构优势。

---

## 91. 为什么现在很多 LLM 使用 RoPE？

因为现代 long-context autoregressive模型很重视：

> relative-position behavior

和：

> extrapolation / efficient attention structure。

RoPE 把 position 作为旋转作用到 Q/K 上，

与我们刚刚 sin/cos pair 的 rotation 推导有很强数学联系。

但 RoPE 是后来的方法，

不应该混进原始 Transformer 定义。

以后可以单独写：

> **RoPE：为什么旋转 Q/K 就能编码相对位置？**

---

## 92. 原始 Sinusoidal 和 RoPE 最大的高层区别

原始 Transformer：

$$
x=e+PE
$$

即：

> position 加到 token representation。

RoPE：

> 直接根据 position 旋转 Q/K representation。

所以位置进入 Attention 的方式不同。

二者都用 sin/cos，

但不能因为都有 trigonometric functions 就认为：

> 是同一个方法。

---

## 93. Positional Encoding 会影响 V 吗？

取决于实现。

#### 原始 Transformer

PE 加到输入 embedding：

$$
x=e+p
$$

随后：

$$
V=xW_V
$$

所以 position 会进入 V。

---

#### ACT/DETR-style Attention

显式代码中常：

```text
q = content + pos
k = content + pos
value = content
```

所以额外 `pos` tensor 主要影响 Q/K。

因此必须：

> 以具体 architecture 为准。

---

## 94. 这就是为什么我们不能只背一个代码模板

“Transformer Positional Encoding”是一个概念。

具体实现可能：

- stack bottom addition；
- every-layer Q/K injection；
- learned query embeddings；
- relative bias；
- rotary Q/K；
- spatial coordinates。

如果只背：

```python
x = x + pos
```

会在读 DETR/ACT 时立刻困惑。

真正应该理解的是：

> **模型必须有机制打破 content-only Attention 的位置对称性，并让位置结构能够影响信息路由。**

---

## 95. ACT 的位置系统可以整理成一张表

| 部分 | Token / Slot | Position Mechanism | 目的 |
|---|---|---|---|
| CVAE Encoder | `[CLS] + qpos + actions` | 1D sinusoidal table | 区分 sequence/time slots |
| Policy Encoder | image features | 2D sinusoidal | 保留 image spatial geometry |
| Policy Encoder | latent + joint | learned 2-token positional embedding | 区分 special non-image tokens |
| Policy Decoder | action query slots | Paper: fixed sinusoidal; released code: learned `nn.Embedding` | 区分 future action output slots |

这张表很重要。

因为 ACT 并不是：

> “全模型只用一种 Position Encoding。”

---

## 96. 为什么 CVAE Encoder 和 Policy Encoder 用不同 Position 形式？

因为它们的数据结构不同。

CVAE sequence：

```text
[CLS], qpos, action₀, action₁, ...
```

本质是：

> 1D sequence。

---

Visual feature map：

$$
15\times20
$$

本质是：

> 2D spatial grid。

---

latent / joint：

> special non-spatial tokens。

所以不同结构使用不同 position representation 很合理。

---

## 97. 为什么不能给所有东西统一一个 1D index？

理论上可以 flatten 全部：

```text
latent = 0
joint = 1
camera1 patch0 = 2
camera1 patch1 = 3
...
```

但这样位置 representation 不直接表达：

- image x/y geometry；
- token type distinction。

2D / special embeddings提供更合适的 inductive structure。

所以 Positional Encoding 设计往往应该匹配：

> data topology。

---

## 98. Position Encoding 本质上是一种 Inductive Bias

它告诉模型：

> “位置关系可能有结构，你不需要只靠内容从零猜。”

Sinusoidal 进一步告诉模型：

> 位置可以被表示成多尺度连续 phase structure。

2D Encoding 告诉模型：

> 图像具有横向和纵向坐标。

这种 structure 是人为加入的：

> inductive bias。

最终如何使用仍由 learning 决定。

---

## 99. 常见误解一：Self-Attention 已经按 Tensor Row 顺序计算，所以自然知道顺序

**错误。**

矩阵 row 的存储顺序并不会自动成为模型可用 feature。

如果没有 position-dependent signal，

交换 rows：

> computation 会等变地交换 outputs。

模型没有“Python index”这种神秘额外输入。

---

## 100. 常见误解二：Positional Encoding 是为了让 GPU 知道 Token 顺序

**错误。**

GPU 本来当然知道 tensor memory layout。

需要位置的是：

> **模型的 learned function。**

PE 是给网络 representation 用的，

不是给硬件用的。

---

## 101. 常见误解三：Positional Encoding 就是给每个 Token 加一个整数

**错误。**

经典方法把 position 映射成：

$$
d_{\text{model}}
$$

维 vector。

---

## 102. 常见误解四：Sinusoidal PE 是 Learned

原始 Transformer：

**不是。**

它是 deterministic fixed function。

---

## 103. 常见误解五：Transformer 必须使用 Sinusoidal PE

**错误。**

原论文自己就测试了 learned positional embedding，

效果几乎相同。

现代模型还有大量其他方法。

---

## 104. 常见误解六：Sin/Cos 的目的只是“因为它们周期性”

不完整。

原论文特别强调：

> fixed offset 可以通过线性 transformation 表示。

这是 sin/cos pair 很重要的结构性质。

---

## 105. 常见误解七：PE(pos+k) 本身等于 PE(pos)+PE(k)

**错误。**

Sinusoidal encoding不是简单：

$$
PE(pos+k)=PE(pos)+PE(k)
$$

正确是：

> 对固定 $k$，存在一个线性旋转变换 $R_k$：

$$
PE(pos+k)=R_kPE(pos)
$$

按各 frequency pairs 组成 block-wise transformation。

---

## 106. 常见误解八：相邻 Position 的所有维度都会相邻

**不准确。**

不同 dimensions 有不同 frequencies。

有些变化快，

有些变化慢。

---

## 107. 常见误解九：10000 是一个理论推导出的唯一最佳常数

**错误。**

它是原始设计选择。

---

## 108. 常见误解十：加 PE 会永久污染 Content，模型无法分辨

**错误。**

网络正是被训练来利用：

$$
content+position
$$

的联合 representation。

Linear projections 可以对两类成分产生可学习响应。

---

## 109. 常见误解十一：Causal Mask 可以替代 Position Encoding

**错误。**

Mask 决定可见性，

PE 表示位置。

---

## 110. 常见误解十二：Position Encoding 可以替代 Causal Mask

**错误。**

知道未来 token 在 position 10，

不等于禁止当前 position 9 读取它。

---

## 111. 常见误解十三：图片 Flatten 后只要顺序固定，就不需要 2D Position

**错误。**

“固定存储顺序”不会自动变成模型可识别的空间坐标。

需要 position signal。

---

## 112. 常见误解十四：ACT 的 1200 Visual Tokens 只用普通 1D Text Position

**错误。**

论文明确使用：

> 2D sinusoidal position embedding。

---

## 113. 常见误解十五：ACT 中 joint 和 latent 也有 Image x/y Coordinates

**错误。**

官方代码给它们：

> 两个 learned additional positional embeddings。

---

## 114. 常见误解十六：ACT 论文和代码的 Action Query Position 完全一样

当前版本不一样。

论文：

> fixed / sinusoidal。

released repository：

> learned `nn.Embedding`。

需要分开记录。

---

## 115. 常见误解十七：Position Encoding 就等于 Attention Weight

完全不是。

PE：

> 输入/结构 representation。

Attention Weight：

$$
softmax(QK^\top)
$$

是：

> 当前 forward 动态算出的 routing coefficients。

PE 只是会影响 Q/K，

进而影响 Attention Weight。

---

## 116. 常见误解十八：Sinusoidal Encoding 证明 Transformer 一定能外推无限长度

**错误。**

它可以为未见 positions 生成 encoding，

但 learned network 是否泛化：

> 没有自动保证。

---

## 117. 用四层理解 Position Encoding

### 第一层：为什么需要？

因为 content-only Self-Attention：

$$
\boxed{
\text{does not intrinsically encode order}
}
$$

---

### 第二层：怎么加入？

经典 Transformer：

$$
\boxed{
x_i=e_i+p_i
}
$$

---

### 第三层：Sinusoidal 为什么有意义？

$$
\boxed{
PE(pos)
=
[\sin(\omega_i pos),\cos(\omega_i pos)]_i
}
$$

多 frequencies 提供多尺度 phase pattern。

---

### 第四层：Relative Shift Property

对于固定：

$$
k
$$

$$
\boxed{
PE(pos+k)=R_kPE(pos)
}
$$

其中：

$$
R_k
$$

是只依赖 offset 的 block-wise linear rotation。

---

## 118. 用一句话理解 Sinusoidal PE

> **Sinusoidal Positional Encoding 把一个离散位置映射成由多种频率 sin/cos 相位组成的高维向量，使每个位置获得可区分但有连续几何结构的表示；同一频率的 sin/cos pair 在位置平移时只发生固定旋转，因此固定 relative offset 可以由位置向量之间的简单线性关系表达。**

---

## 119. 用一句话理解 Positional Encoding

> **Positional Encoding 的本质不是“告诉 Transformer 一个数字编号”，而是把 token 所处的顺序或空间结构转换成神经网络可利用的向量特征，从而打破 content-only Self-Attention 的 permutation symmetry，让 Q/K matching 同时依赖“这个 token 是什么”和“它在哪里”。**

---

## 120. 用一句话连接 ACT

> **ACT 把 Positional Encoding 当作结构信息接口：CVAE action sequence 使用 1D sinusoidal positions 表示时间槽位，ResNet visual grid 使用 2D sinusoidal positions 保留图像空间坐标，latent/joint special tokens 使用 learned positional identities，而 Decoder 的 $k$ 个 action-query slots 也需要彼此不同的位置/slot identity，才能并行表示未来动作序列中的不同时间位置。**

---

## 121. 下一步：Causal Mask

现在已经知道：

> Position Encoding 告诉模型“你在哪里”。

但原始 Transformer Decoder 还需要另一个机制：

> “你能看到谁？”

这就是：

> **Causal Mask。**

下一篇：

> **[Causal Mask：为什么 Decoder 训练时不能偷看未来？](./causal-mask.md)**

会重点解释：

- 为什么训练时 target sequence 已经全部存在，仍然不会造成 label leakage；
- 为什么必须用 upper-triangular mask；
- 为什么把非法 logits 设成 $-\infty$；
- Softmax 后为什么精确变 0；
- teacher forcing 和 causal mask 的关系；
- 为什么训练可以并行，而 inference 仍然 autoregressive；
- Encoder 为什么通常不 causal；
- ACT Decoder 为什么恰恰 **不需要** causal mask；
- non-autoregressive action chunk prediction 与语言 next-token generation 的根本区别。

---

### Primary Source：Transformer

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,  
Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin.

**Attention Is All You Need.**  
NeurIPS 2017.

- arXiv: https://arxiv.org/abs/1706.03762
- HTML: https://arxiv.org/html/1706.03762

本文主要依据 Section 3.5 — Positional Encoding。

原论文明确指出：

- Transformer 没有 recurrence 和 convolution；
- 因此必须向模型注入 relative / absolute position information；
- Positional Encoding 与 embedding 具有相同：
  $$
  d_{\text{model}}
  $$
  所以直接相加；
- 使用：
  $$
  PE_{(pos,2i)}
  =
  \sin
  \left(
  pos/10000^{2i/d_{\text{model}}}
  \right)
  $$
  $$
  PE_{(pos,2i+1)}
  =
  \cos
  \left(
  pos/10000^{2i/d_{\text{model}}}
  \right)
  $$
- wavelengths 按几何级数变化；
- 作者选择该形式的一个理由是：对于固定 offset $k$，$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数；
- learned positional embedding 实验结果与 sinusoidal 几乎相同；
- 作者最终选择 sinusoidal，因为它可能更容易 extrapolate 到更长 sequence。

---

### ACT Primary Source

Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.  
**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.**  
RSS 2023.

- arXiv: https://arxiv.org/abs/2304.13705
- HTML: https://arxiv.org/html/2304.13705v1

Section IV-C 明确写：

- 每张：
  $$
  480\times640
  $$
  RGB image 经 ResNet18 得到：
  $$
  15\times20\times512
  $$
- flatten 得到：
  $$
  300\times512
  $$
- 为保留 spatial information，加入：
  > **2D sinusoidal position embedding**
- 4 cameras 拼成：
  $$
  1200\times512
  $$
- 加 joint 与 latent 后：
  $$
  1202\times512
  $$

Appendix C 同样再次确认 image features：

> add a 2D sinusoidal position embedding to preserve spatial information。

论文同时描述 Decoder input/query positions 为 fixed positional representation，Appendix C 写 first-layer queries 为 fixed sinusoidal embeddings。

---

### Official ACT Implementation

Official repository:

https://github.com/tonyzhaozh/act

#### `detr/models/position_encoding.py`

官方代码的：

```python
PositionEmbeddingSine
```

明确说明：

> similar to Attention Is All You Need positional embedding, generalized to work on images。

对于：

$$
hidden\_dim=512
$$

构建：

```python
N_steps = hidden_dim // 2
```

即：

$$
256
$$

维用于每一个 spatial axis。

代码分别构造：

```text
pos_y → 256 dims
pos_x → 256 dims
```

再：

```python
torch.cat((pos_y, pos_x), dim=3)
```

得到：

$$
512
$$

维 2D position representation。

---

#### `detr/models/detr_vae.py`

CVAE Encoder：

```python
self.register_buffer(
    'pos_table',
    get_sinusoid_encoding_table(
        1 + 1 + num_queries,
        hidden_dim
    )
)
```

对应：

```text
[CLS]
qpos
action sequence
```

的一维 sinusoidal positions。

Policy Encoder 中：

```python
self.additional_pos_embed =
    nn.Embedding(2, hidden_dim)
```

为：

- latent；
- proprio / joint；

提供两个 learned special positional embeddings。

---

#### `detr/models/transformer.py`

ACT/DETR-style attention 使用：

```python
with_pos_embed(tensor, pos):
    return tensor + pos
```

Encoder Self-Attention：

```text
Q = src + pos
K = src + pos
V = src
```

Decoder Cross-Attention：

```text
Q = tgt + query_pos
K = memory + pos
V = memory
```

这说明当前 released architecture 中：

> positional information 被显式用于 Attention matching path。

---

### Paper vs Released Code

ACT Paper：

> Decoder input/query sequence 使用 fixed positional representation；Appendix C 描述 first-layer queries 为 fixed sinusoidal embeddings。

Current official repository：

```python
self.query_embed =
    nn.Embedding(
        num_queries,
        hidden_dim
    )
```

即：

> learned action-query embeddings。

两者都解决：

> $k$ 个 action slots 必须拥有不同 positional / slot identity。

但知识库应明确标记：

$$
\boxed{
\text{Paper Fact}
\neq
\text{Released-Code Detail}
}
$$

而不是把两者混写。

---

### 本文知识连接

#### 前置知识

- [Transformer](./transformer.md)
- [Self-Attention](./self-attention.md)
- [Query / Key / Value](./qkv.md)
- [Dot Product](./dot-product.md)
- Permutation
- Equivariance

#### 数学

- Sine and Cosine
- Trigonometric Addition Formulas
- Rotation Matrix
- Frequency and Wavelength
- Geometric Progression

#### Transformer

- [Causal Mask](./causal-mask.md)
- [Cross-Attention](./cross-attention.md)
- [Multi-Head Attention](./multi-head-attention.md)
- [Transformer Encoder](./transformer-encoder.md)
- [Transformer Decoder](./transformer-decoder.md)
- RoPE
- Relative Position Encoding

#### Vision

- [2D Positional Encoding](./positional-encoding.md)
- Vision Transformer
- ResNet

#### Robot Learning

- [ACT Architecture](../robot-learning/act/architecture.md)
- [CVAE in ACT](../robot-learning/act/cvae-in-act.md)
- [ACT Complete Data Flow](../robot-learning/act/complete-data-flow.md)

#### 下一步

- [Causal Mask](./causal-mask.md)
