---
title: "Residual Connection：为什么网络要把输入直接加回来？"
description: "从 ResNet 的 degradation problem 出发，严格理解 y=x+F(x) 为什么把深层网络重写成 identity + residual update，推导其前向与反向传播性质，并连接到 Transformer 的 Post-LN / Pre-LN 与 ACT 中 Attention、Cross-Attention、FFN 外的残差路径。"
status: reviewed
pageType: concept
canonical: /deep-learning/residual-connection
updated: "2026-09-15"
---

# Residual Connection：为什么网络要把输入直接加回来？

在前面的 Transformer 文章中，我们不断看到同一个结构：

$$
\boxed{
x + \operatorname{Sublayer}(x)
}
$$

例如 Encoder：

$$
x
\rightarrow
\operatorname{SelfAttention}(x)
$$

却不是直接把 Attention 输出当作下一层输入。

而是：

$$
\boxed{
x+\operatorname{Attention}(x)
}
$$

FFN 也是：

$$
\boxed{
h+\operatorname{FFN}(h)
}
$$

Decoder 甚至有三次：

$$
x+\operatorname{SelfAttention}(x)
$$

$$
h+\operatorname{CrossAttention}(h,M)
$$

$$
u+\operatorname{FFN}(u)
$$

然后再配上 LayerNorm。

为什么？

为什么神经网络辛辛苦苦算出：

$$
F(x)
$$

之后，还要把原来的：

$$
x
$$

直接加回来？

如果：

$$
F(x)
$$

足够强，

直接让：

$$
y=F(x)
$$

不是更干净吗？

Residual Connection 真正改变了什么？

它是不是单纯：

> “防止信息丢失”？

是不是：

> “解决梯度消失”？

是不是：

> “让网络记住输入”？

为什么：

$$
x+F(x)
$$

这个看起来极其简单的加法，会成为 ResNet、Transformer、现代深度学习架构最核心的结构之一？

这一篇从 ResNet 原论文开始，把它讲完整。

---

## 1. Residual Connection 最经典的形式

设输入：

$$
x
$$

一个需要学习的 transformation：

$$
F(x)
$$

普通网络：

$$
\boxed{
y=F(x)
}
$$

Residual block：

$$
\boxed{
y=x+F(x)
}
$$

其中：

$$
x
$$

直接绕过：

$$
F
$$

这条路径叫：

- shortcut connection；
- skip connection；
- identity shortcut；

取决于具体上下文。

最经典的 identity shortcut：

$$
\boxed{
\text{Shortcut}(x)=x
}
$$

---

## 2. 先不要从“梯度”开始理解

很多教程一上来就说：

> Residual Connection 是为了解决 vanishing gradient。

这句话太粗糙。

ResNet 原论文的历史动机其实更具体。

论文首先承认：

> vanishing / exploding gradients 是早期深层网络的重要困难。

但作者随后指出，到 2015 年时：

- normalized initialization；
- intermediate normalization，例如 BatchNorm；

已经能让几十层网络：

> **开始正常收敛。**

然而新的问题出现了。

这个问题叫：

$$
\boxed{
\text{Degradation Problem}
}
$$

---

## 3. 什么是 Degradation Problem？

直觉上你可能认为：

> 网络越深，至少训练集上应该不会更差吧？

因为更深网络参数更多、表达能力更强。

但 ResNet 论文观察到：

> 当 plain network 深度继续增加时，训练误差反而变高。

论文举了 CIFAR-10 的典型现象：

- 20-layer plain network；
- 56-layer plain network。

结果：

> 更深的 56-layer plain network 具有更高的 training error。

这非常关键。

因为如果只是：

> 测试误差变差，

可能是 overfitting。

但现在连：

$$
\boxed{
\text{training error}
}
$$

都更高。

所以问题不是简单：

> “深网络太强，过拟合了。”

而是：

> **优化器没有成功找到本来应该存在的好解。**

---

## 4. 为什么更深网络理论上“不应该”训练得更差？

ResNet 原论文给了一个非常漂亮的构造性论证。

假设有一个已经训练得很好的较浅网络：

$$
f(x)
$$

现在我们在它后面多加几层。

理论上，新加的这些层至少可以学：

$$
\boxed{
Identity(x)=x
}
$$

那么整个更深网络就可以做到：

$$
x
\rightarrow
f(x)
\rightarrow
Identity
\rightarrow
Identity
\rightarrow
\cdots
$$

输出和较浅网络一样。

所以：

> 更深网络的 hypothesis space 至少包含“复制浅网络”的一个解。

因此从“存在解”的角度：

$$
\boxed{
\text{deeper network should be able to match the shallower training error}
}
$$

---

## 5. 但实验里它做不到

ResNet 论文的核心观察是：

> 这个好解虽然理论上存在，但当时的优化器很难在 plain deep network 中找到它。

也就是说：

$$
\boxed{
\text{representation capacity is not the main issue}
}
$$

真正的问题更接近：

$$
\boxed{
\text{optimization difficulty}
}
$$

这就是 degradation problem 的核心。

---

## 6. 为什么“学 Identity”可能并不容易？

假设一个普通多层 nonlinear network需要学：

$$
H(x)=x
$$

也就是：

> 输入什么，就输出什么。

听起来很简单。

但网络内部可能是：

$$
x
\rightarrow
W_1
\rightarrow
ReLU
\rightarrow
W_2
\rightarrow
ReLU
\rightarrow
\cdots
$$

要让一堆 nonlinear transformations组合起来恰好实现：

$$
H(x)\approx x
$$

优化器不一定容易找到。

所以 ResNet 作者提出：

> 与其让这些层直接学习完整映射 $H(x)$，不如让它们只学“和输入相比需要改多少”。

---

## 7. Residual Learning 的核心重参数化

设我们真正希望某个 block 学到：

$$
\mathcal H(x)
$$

ResNet 不直接让 nonlinear branch 学：

$$
\mathcal H(x)
$$

而定义：

$$
\boxed{
\mathcal F(x)
=
\mathcal H(x)-x
}
$$

那么：

$$
\boxed{
\mathcal H(x)
=
\mathcal F(x)+x
}
$$

于是 block 写成：

$$
\boxed{
y=
\mathcal F(x)+x
}
$$

---

## 8. 为什么叫 Residual？

数学里 residual 可以理解成：

> 目标值与参考值之间剩下的差。

这里：

$$
\mathcal H(x)
$$

是目标映射，

$$
x
$$

是 identity reference。

所以：

$$
\boxed{
\mathcal F(x)
=
\mathcal H(x)-x
}
$$

就是：

> **目标输出相对于输入还需要补上的那部分。**

这就是 residual。

---

## 9. 一个最简单的数值例子

假设：

$$
x=10
$$

真正希望：

$$
H(x)=10.2
$$

普通网络必须直接学：

$$
10\rightarrow10.2
$$

Residual formulation 则写成：

$$
F(x)=0.2
$$

然后：

$$
y=10+0.2=10.2
$$

如果目标映射很接近 identity，

那么：

$$
F(x)
$$

只需要学一个小修正。

---

## 10. Identity 情况尤其漂亮

如果最优映射就是：

$$
H(x)=x
$$

那么 residual target：

$$
F(x)=H(x)-x=0
$$

所以网络只需要学：

$$
\boxed{
F(x)\approx0
}
$$

而不是让一串 nonlinear layers 精确逼近：

$$
H(x)=x
$$

这正是 ResNet 原论文明确给出的核心直觉。

---

## 11. Residual Connection 不是让网络“只能做小修改”

注意：

$$
y=x+F(x)
$$

并不要求：

$$
F(x)
$$

一定很小。

如果目标：

$$
H(x)
$$

和 input差很远，

网络完全可以学：

$$
F(x)=H(x)-x
$$

一个很大的 residual。

所以 residual architecture 不是硬约束：

> “每层只能做一点点。”

更准确：

> 它给 optimization 一个 identity reference。

---

## 12. 这是 Architecture 变化，还是目标函数变化？

主要是：

> **网络参数化方式改变。**

原目标输出仍然是：

$$
H(x)
$$

但原来 nonlinear branch直接表示：

$$
H
$$

现在表示：

$$
F=H-I
$$

然后 architecture固定加：

$$
x
$$

回去。

所以是：

$$
\boxed{
\text{same desired mapping, different parameterization}
}
$$

这种 parameterization 可以显著影响 optimization difficulty。

---

## 13. 为什么存在同样表达能力，优化难度却可能不同？

这是机器学习中非常重要的观念。

两个 parameterizations可能都能表示：

$$
H(x)
$$

但 loss surface：

- geometry；
- gradients；
- initialization附近的行为；

可能不同。

所以：

$$
\boxed{
\text{can represent}
\neq
\text{easy to optimize}
}
$$

Residual Learning正是经典例子。

---

## 14. ResNet 原论文的核心假设

论文并没有证明：

> 所有目标函数都一定更适合 residual form。

作者更谨慎：

> 如果最优函数比起 zero mapping 更接近 identity mapping，那么以 identity为参考学习 perturbation可能更容易。

换句话说：

普通参数化相当于：

> 从“零映射”附近开始想象目标。

Residual参数化则给了一个：

$$
\boxed{
\text{identity baseline}
}
$$

然后学习偏移。

---

## 15. 为什么 Identity 是很有意义的默认基线？

深网络中我们经常希望：

> 如果这一层没有什么特别有用的新东西，就至少不要毁掉已有 representation。

Residual block可以在：

$$
F(x)\approx0
$$

时近似：

$$
y\approx x
$$

所以每一层都有一个自然的：

> “什么都不改”

参考状态。

这对深层网络很有价值。

---

## 16. Plain Layer 没有这么直接的“什么都不做”模式

普通：

$$
y=F(x)
$$

如果想什么都不做，

就必须让：

$$
F(x)\approx x
$$

也就是整个 nonlinear branch主动实现 identity。

Residual：

$$
y=x+F(x)
$$

想什么都不做：

$$
F(x)\approx0
$$

就够。

这就是两种 parameterization 的直观差别。

---

## 17. Shortcut Connection 到底是什么？

Residual block：

```text
x ───────────────────────┐
│                        │
│                        ▼
│                      Add
│                        │
▼                        │
F(x) ────────────────────┘
```

上面的：

$$
x
$$

直接绕过 nonlinear branch。

这条线就是：

> shortcut / skip connection。

如果完全不变：

$$
Shortcut(x)=x
$$

就叫：

> identity shortcut。

---

## 18. Identity Shortcut 有参数吗？

没有。

ResNet 原论文明确强调：

> identity shortcut不会引入额外 trainable parameter。

计算上只多一个：

> element-wise addition。

所以当 shape一致时：

$$
\boxed{
y=F(x)+x
}
$$

非常经济。

---

## 19. 为什么必须 Shape 一样才能相加？

如果：

$$
x\in\mathbb R^{512}
$$

而：

$$
F(x)\in\mathbb R^{256}
$$

无法直接：

$$
x+F(x)
$$

所以 identity residual要求：

$$
\boxed{
shape(x)=shape(F(x))
}
$$

这也是 Transformer为什么让 sub-layer最终都回到：

$$
d_{\text{model}}
$$

的重要原因之一。

---

## 20. Shape 不一样怎么办？

ResNet 原论文给出 projection shortcut：

$$
\boxed{
y=F(x)+W_sx
}
$$

其中：

$$
W_s
$$

把 shortcut branch投影到合适维度。

例如：

$$
256\rightarrow512
$$

这样才能与：

$$
F(x)\in\mathbb R^{512}
$$

相加。

---

## 21. 这还是 Identity Shortcut 吗？

严格来说：

> 不是纯 identity。

它是：

> projection shortcut。

真正 identity shortcut：

$$
W_s=I
$$

或者根本没有矩阵。

ResNet 作者更偏好：

> 能用 identity 就用 identity；

只在维度匹配需要时使用 projection。

---

## 22. Transformer 为什么通常不需要 Projection Shortcut？

Transformer设计所有 sub-layers：

- Attention；
- FFN；

最终输出都保持：

$$
d_{\text{model}}
$$

例如：

$$
512\rightarrow512
$$

FFN内部虽然：

$$
512\rightarrow2048\rightarrow512
$$

但最后回到 512。

因此 residual可以直接：

$$
\boxed{
x+F(x)
}
$$

无需：

$$
W_sx
$$

这使 Transformer residual path非常干净。

---

## 23. 现在开始看 Gradient

Residual常被解释成：

> 帮助梯度传播。

这个说法有数学基础，

但需要严谨。

先看最简单：

$$
y=x+F(x)
$$

对：

$$
x
$$

求 Jacobian：

$$
\boxed{
\frac{\partial y}{\partial x}
=
I+
\frac{\partial F}{\partial x}
}
$$

这是最重要的公式之一。

---

## 24. 普通 Layer 的 Gradient

如果：

$$
y=F(x)
$$

那么：

$$
\frac{\partial y}{\partial x}
=
\frac{\partial F}{\partial x}
$$

Gradient完全依赖：

$$
F
$$

这条 transformation path。

---

## 25. Residual Layer 多出一个 Identity Term

Residual：

$$
y=x+F(x)
$$

所以：

$$
\frac{\partial y}{\partial x}
=
I+J_F
$$

其中：

$$
J_F
=
\frac{\partial F}{\partial x}
$$

因此反向：

$$
\boxed{
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial y}
\left(
I+J_F
\right)
}
$$

概念上出现了一条：

$$
\boxed{
I
}
$$

直接项。

---

## 26. “Identity Gradient Path”是什么意思？

展开：

$$
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial y}
+
\frac{\partial L}{\partial y}J_F
$$

第一项：

$$
\frac{\partial L}{\partial y}
$$

不需要穿过：

$$
F
$$

内部全部权重和 nonlinearities。

所以 shortcut给 backward signal提供：

> 一条直接 additive route。

这也是 residual network容易优化的重要数学直觉之一。

---

## 27. 但不能说“有 Residual 就绝不会梯度消失”

这是必须纠正的。

因为实际深网络有：

- normalization；
- nonlinear activation；
- dropout；
- many layers；
- parameter scaling；

梯度传播取决于完整 Jacobian chain。

$$
I+J_F
$$

并不保证整个乘积永远：

$$
=I
$$

也不保证 singular values永远稳定。

所以正确说法：

> residual / identity shortcut提供直接 signal path，并通常显著改善深层优化；

而不是：

> residual从数学上彻底消灭 vanishing/exploding gradients。

---

## 28. ResNet 原论文也没有把 Degradation 简化成 Vanishing Gradient

原论文明确区分：

1. vanishing/exploding gradients；
2. deeper plain networks仍然出现的 degradation problem。

作者指出第一类问题当时已经通过 initialization / normalization得到很大缓解。

之后才提出：

> 即使网络能收敛，更深 plain network仍可能有更高 training error。

所以：

$$
\boxed{
\text{Residual Connection 的经典动机不只是 Vanishing Gradient}
}
$$

---

## 29. 一个多层 Residual Chain

考虑最简单无额外 normalization 的 residual stack：

$$
x_{l+1}
=
x_l+F_l(x_l)
$$

那么：

$$
x_{l+2}
=
x_{l+1}+F_{l+1}(x_{l+1})
$$

代入：

$$
x_{l+2}
=
x_l
+
F_l(x_l)
+
F_{l+1}(x_{l+1})
$$

继续展开到第：

$$
L
$$

层：

$$
\boxed{
x_L
=
x_l
+
\sum_{i=l}^{L-1}
F_i(x_i)
}
$$

这是一个很漂亮的表达。

---

## 30. 这说明什么？

在这种理想 identity residual chain里，

深层 representation可以看成：

$$
\boxed{
\text{earlier representation}
+
\text{many learned residual updates}
}
$$

所以深网络不是每层都完全“推翻重建”。

而是：

> 在一个持续存在的 representation stream 上反复写入修正量。

---

## 31. Residual Stream 的直觉

现代 Transformer语境常把：

$$
x_l
$$

称为：

> residual stream。

可以直觉化成一条主干：

```text
x₀
│
├─ + attention update
│
x₁
│
├─ + FFN update
│
x₂
│
├─ + attention update
│
...
```

Attention / FFN像不同模块：

> 往这条主干里写东西。

这是非常好用的 mental model。

但：

> “residual stream”这个说法是后续解释语言，不是 ResNet 2015论文的正式术语。

---

## 32. 为什么 Add，而不是 Concat？

Residual使用：

$$
x+F(x)
$$

意味着：

> 两条路径落在同一个 representation space 中。

Concat：

$$
[x;F(x)]
$$

会：

- 增加维度；
- 需要后续 projection；
- 不再直接实现 identity + update。

Residual addition的独特含义是：

$$
\boxed{
\text{base representation}
+
\text{correction in the same coordinates}
}
$$

---

## 33. 为什么不是 Average？

Residual不是：

$$
\frac{x+F(x)}{2}
$$

而是：

$$
x+F(x)
$$

因为它不是：

> 两个模型投票。

它是在表示：

$$
H(x)=x+F(x)
$$

也就是：

> identity baseline + learned residual。

Average会引入固定：

$$
1/2
$$

scale，

改变参数化。

---

## 34. Residual Branch 的输出是不是“误差”？

不要和 supervised loss的 error混淆。

这里 residual：

$$
F(x)
$$

指：

> desired mapping相对于 identity reference的差。

它不是：

$$
y_{true}-y_{pred}
$$

那种训练误差。

所以：

$$
\boxed{
\text{Residual function}
\neq
\text{Loss residual/error}
}
$$

---

## 35. 为什么 Residual Connection 可以保留已有信息？

从 forward看：

$$
y=x+F(x)
$$

input：

$$
x
$$

显式参与 output。

所以 branch不需要：

> 重新编码所有原信息。

如果某些 feature已经很好，

理论上：

$$
F
$$

可以对它们少改。

这就是“保留信息”的直觉来源。

但这不是唯一作用。

---

## 36. 为什么“防止信息丢失”仍然不够完整？

因为普通网络也可能学 identity、复制信息。

Residual真正特别的是：

- identity path被 architecture显式提供；
- optimization不必让 nonlinear branch自己学完整 identity；
- forward / backward都有直接路径。

所以更准确：

$$
\boxed{
\text{information preservation}
+
\text{optimization reparameterization}
+
\text{signal propagation}
}
$$

共同构成 residual的价值。

---

## 37. Original ResNet 其实不是纯 y=x+F(x) 后立刻结束

ResNet 2015原始 basic block中，

作者在 addition后还有 ReLU。

即概念上：

$$
y=F(x)+x
$$

然后：

$$
x_{\text{next}}=ReLU(y)
$$

所以最早的 ResNet并不是：

> 一条完全无任何 nonlinear transform的长 identity highway。

---

## 38. 为什么后来又有 Pre-Activation ResNet？

He 等人在 2016 年进一步研究：

**Identity Mappings in Deep Residual Networks**

他们分析：

> 如果 shortcut和 after-addition mapping都尽量保持 identity，forward/backward signal可以更直接传播。

于是提出：

> full pre-activation residual unit。

把 BN / ReLU移到 weight layers之前，

让 addition之后更接近：

$$
\boxed{
x_{l+1}=x_l+F(\cdot)
}
$$

纯 additive identity propagation。

这对理解 Transformer Pre-LN很有启发。

---

## 39. 但 Pre-Activation ResNet 和 Transformer Pre-LN 不是同一个东西

两者都强调：

> 更干净的 identity residual path。

但具体模块完全不同：

ResNet：

- convolution；
- BatchNorm；
- ReLU。

Transformer：

- Attention / FFN；
- LayerNorm。

所以可以比较思想，

但不能说：

> “Pre-LN Transformer就是 Pre-Activation ResNet。”

---

## 40. 现在回到 Transformer

《Attention Is All You Need》原论文明确规定：

$$
\boxed{
LayerNorm(
x+
Sublayer(x)
)
}
$$

其中：

$$
Sublayer
$$

可以是：

- Self-Attention；
- Cross-Attention；
- FFN。

这就是：

> Residual Connection + LayerNorm。

---

## 41. 原始 Transformer Encoder 的第一条 Residual

输入：

$$
X
$$

Self-Attention：

$$
A=MHA(X)
$$

加入 Dropout：

$$
\tilde A=Dropout(A)
$$

Residual：

$$
\boxed{
R_1=X+\tilde A
}
$$

然后：

$$
\boxed{
H=LN(R_1)
}
$$

所以准确顺序：

```text
Self-Attention
↓
Dropout
↓
Add input
↓
LayerNorm
```

---

## 42. Encoder 第二条 Residual

FFN：

$$
F=FFN(H)
$$

Dropout：

$$
\tilde F=Dropout(F)
$$

Residual：

$$
\boxed{
R_2=H+\tilde F
}
$$

再：

$$
\boxed{
Y=LN(R_2)
}
$$

所以每个 Encoder layer有：

$$
2
$$

条 residual shortcuts。

---

## 43. Decoder 有三条

原始 Decoder：

#### Self-Attention residual

$$
H_1=
LN(
H+SelfAttn(H)
)
$$

#### Cross-Attention residual

$$
H_2=
LN(
H_1+CrossAttn(H_1,M)
)
$$

#### FFN residual

$$
H_3=
LN(
H_2+FFN(H_2)
)
$$

省略 dropout。

所以一层 Decoder：

$$
\boxed{
3\text{ residual additions}
}
$$

---

## 44. 为什么 Transformer 所有 Sub-Layers 都保持 d_model？

现在就非常清楚。

因为要做：

$$
x+Sublayer(x)
$$

必须：

$$
shape(x)=shape(Sublayer(x))
$$

所以：

- MHA最后 output projection回 $d_{\text{model}}$；
- FFN从 $d_{\text{model}}$ 扩维后再压回 $d_{\text{model}}$；
- Cross-Attention output也回 $d_{\text{model}}$。

Residual shape constraint深刻影响了 Transformer模块设计。

---

## 45. Attention Residual 在语义上怎么理解？

第 $i$ 个 token原 representation：

$$
x_i
$$

Attention从其他 token读回来 context：

$$
a_i
$$

Residual：

$$
x_i+a_i
$$

直觉上：

$$
\boxed{
\text{我原来是谁}
+
\text{我从其他 Tokens 获取的新信息}
}
$$

这是非常自然的 contextual update。

---

## 46. FFN Residual 又怎么理解？

Attention后：

$$
h_i
$$

已经是 contextual token。

FFN计算：

$$
f_i
$$

作为 feature transformation。

Residual：

$$
h_i+f_i
$$

可以直觉化：

$$
\boxed{
\text{当前 representation}
+
\text{本地 nonlinear feature update}
}
$$

所以 Attention和FFN都不必：

> 从零重新创建下一层 representation。

---

## 47. Cross-Attention Residual 呢？

Decoder当前状态：

$$
h_i
$$

Cross-Attention从 Encoder Memory读取：

$$
c_i
$$

Residual：

$$
h_i+c_i
$$

直觉上：

$$
\boxed{
\text{我当前的 output-side state}
+
\text{我刚从 source memory 获取的信息}
}
$$

这也是为什么 residual和 Decoder结构非常契合。

---

## 48. 为什么 Dropout 只作用 Residual Branch，而不是 Identity Path？

原始 Transformer训练：

$$
x+
Dropout(
Sublayer(x)
)
$$

Identity shortcut：

$$
x
$$

保持直接。

这意味着 training dropout主要扰动：

> learned update branch。

而主 identity path不被这一个 sub-layer dropout随机切断。

这有助于保持稳定信息通路。

---

## 49. 如果 Dropout 把整个 Residual Branch 某些值变 0，会怎样？

对应那些被丢弃的 update components，

输出仍有：

$$
x
$$

所以 block在局部更接近：

> identity。

这也是 residual + dropout组合很自然的一点。

但标准 dropout有 scaling，

具体 expectation由 framework处理。

---

## 50. Transformer 原始结构为什么叫 Post-LN？

因为：

$$
\boxed{
y=LN(x+F(x))
}
$$

LayerNorm在：

> residual addition之后。

所以：

$$
\boxed{
Post\text{-}LN
}
$$

---

## 51. Post-LN 的 Gradient Path 不是纯 Identity

这是一个非常重要的细节。

如果只有：

$$
y=x+F(x)
$$

则：

$$
\frac{\partial y}{\partial x}
=
I+J_F
$$

但 Post-LN：

$$
\boxed{
y=LN(x+F(x))
}
$$

设：

$$
z=x+F(x)
$$

则：

$$
\frac{\partial y}{\partial x}
=
\frac{\partial LN(z)}{\partial z}
\left(
I+J_F
\right)
$$

所以 gradient即使沿 shortcut，

仍然必须经过：

$$
\boxed{
J_{LN}
}
$$

---

## 52. 因此不能对 Original Transformer 说“梯度沿 Identity 无修改直通所有层”

不严谨。

因为每一个 Post-LN block的 addition后：

> 都有 LayerNorm。

所以跨很多层的 gradient path仍然包含：

$$
J_{LN}^{(l)}
$$

等因素。

Residual仍然有重要价值，

但不是一个完全裸露的 identity highway。

---

## 53. Pre-LN 是什么？

Pre-LN把 normalization移到 sub-layer前：

$$
\boxed{
y=
x+
F(
LN(x)
)
}
$$

这里 addition之后：

> 没有立刻 LayerNorm。

因此 residual stream：

$$
x
$$

可以直接加到下一状态。

---

## 54. Pre-LN 的 Jacobian

$$
y=x+F(LN(x))
$$

对：

$$
x
$$

求导：

$$
\boxed{
\frac{\partial y}{\partial x}
=
I
+
J_F
J_{LN}
}
$$

关键是：

$$
I
$$

在最外层直接出现。

所以相比 Post-LN：

$$
J_{LN}(I+J_F)
$$

Pre-LN保留了更直接的 identity derivative term。

---

## 55. 这就是为什么现代深 Transformer 常讨论 Pre-LN

后续研究例如：

**On Layer Normalization in the Transformer Architecture**

系统分析了 Post-LN vs Pre-LN 的 gradient behavior。

论文指出：

> LayerNorm位置会显著影响初始化时的 gradient distribution和训练稳定性。

现代很多深 Transformer因此使用：

> Pre-LN 或相关变体。

但一定记住：

$$
\boxed{
\text{2017 Original Transformer = Post-LN}
}
$$

---

## 56. Post-LN 不代表 Residual Connection “没用了”

当然不是。

即使：

$$
LN(x+F(x))
$$

Residual addition仍然改变：

- forward representation；
- optimization geometry；
- local gradient Jacobian；
- identity/reference structure。

只是不能把 ResNet最理想的纯 identity-path推导：

> 原封不动套到 Post-LN Transformer所有层上。

---

## 57. 为什么 Pre-LN 常被称为更干净的 Residual Stream？

因为：

$$
x_{l+1}
=
x_l
+
F_l(
LN(x_l)
)
$$

可以连续展开：

$$
\boxed{
x_L
=
x_0+
\sum_{l=0}^{L-1}
F_l(
LN(x_l)
)
}
$$

主 residual state显式累加。

这更接近：

> identity highway + updates。

---

## 58. Original Post-LN 不能这样简单 Telescope

Post-LN：

$$
x_{l+1}
=
LN(
x_l+
F_l(x_l)
)
$$

由于：

$$
LN
$$

每层都重新变换 representation，

不能简单写成：

$$
x_L=x_0+\sum F_l
$$

所以：

$$
\boxed{
\text{Pre-LN residual stream intuition}
}
$$

比 Post-LN更加字面成立。

---

## 59. ACT 当前代码两种都支持

ACT官方 `transformer.py`：

```python
normalize_before
```

控制：

- `forward_post`
- `forward_pre`

---

Post-LN：

```python
src2 = self.self_attn(...)
src = src + dropout1(src2)
src = norm1(src)
```

---

Pre-LN：

```python
src2 = norm1(src)
src2 = self.self_attn(...)
src = src + dropout1(src2)
```

所以代码明确展示了两种 residual / norm排列。

---

## 60. ACT 当前默认是什么？

当前 Transformer constructor：

```python
normalize_before=False
```

而 ACT config中的 `pre_norm` 是可选 flag。

默认不开：

$$
\boxed{
normalize\_before=False
}
$$

所以 canonical released ACT默认：

$$
\boxed{
Post\text{-}LN
}
$$

---

## 61. ACT Policy Encoder 的 Residual

当前 `forward_post`：

```python
src2 =
    self.self_attn(...)[0]

src =
    src +
    self.dropout1(src2)

src =
    self.norm1(src)
```

然后 FFN：

```python
src2 =
    self.linear2(
        self.dropout(
            self.activation(
                self.linear1(src)
            )
        )
    )

src =
    src +
    self.dropout2(src2)

src =
    self.norm2(src)
```

所以每层 Encoder有：

$$
\boxed{
2\text{ residual additions}
}
$$

---

## 62. ACT Policy Decoder 的 Residual

Decoder Post-LN：

Self-Attention：

$$
\boxed{
tgt
\leftarrow
LN(
tgt+
SelfAttn(tgt)
)
}
$$

Cross-Attention：

$$
\boxed{
tgt
\leftarrow
LN(
tgt+
CrossAttn(tgt,memory)
)
}
$$

FFN：

$$
\boxed{
tgt
\leftarrow
LN(
tgt+
FFN(tgt)
)
}
$$

所以每层 Decoder：

$$
\boxed{
3\text{ residual additions}
}
$$

---

## 63. ACT 的 CVAE Encoder 也有同样 Residual Layer结构

CVAE training-only Transformer Encoder使用相同类型 Encoder Layer abstraction。

因此：

```text
[CLS]
qpos
action tokens
```

每一层也是：

```text
Self-Attention update
+ residual

FFN update
+ residual
```

所以 residual不是只存在于：

> Policy Transformer。

---

## 64. ACT 中 Residual Shape 为什么总能匹配？

hidden dimension统一：

$$
512
$$

Attention：

$$
512\rightarrow512
$$

FFN：

$$
512\rightarrow3200\rightarrow512
$$

所以最终：

$$
F(x)\in\mathbb R^{512}
$$

可以直接和：

$$
x\in\mathbb R^{512}
$$

相加。

---

## 65. ACT Visual Token经过 Attention 后为什么还能保留自己原来的 Feature？

假设第：

$$
i
$$

个 visual token：

$$
x_i
$$

Self-Attention返回：

$$
a_i
$$

Residual：

$$
x_i+a_i
$$

所以它不必完全被 weighted context替换。

这很重要：

> Attention output是从所有 Values聚合得到的 context；

> residual把原 token representation直接保留在更新中。

---

## 66. 如果没有 Residual，会发生什么？

Encoder Self-Attention后直接：

$$
x_i'=a_i
$$

那么新的 token完全由：

$$
\sum_j\alpha_{ij}v_j
$$

决定。

原 representation只有通过 Value path间接存在。

Residual加入：

$$
x_i
$$

提供显式保留。

这并不证明“无 residual模型一定不能训练”，

但结构差异非常显著。

---

## 67. Decoder Action Slot也同样

Action slot当前状态：

$$
t_i
$$

Cross-Attention返回：

$$
c_i
$$

Residual：

$$
t_i+c_i
$$

意味着：

> observation context不是把 action slot identity /已有状态完全覆盖，

而是在已有 decoder state上增加 observation-conditioned update。

---

## 68. 这和我们前面“Communication + Computation”可以进一步统一

Transformer每层：

#### Attention

生成：

$$
\Delta_{\text{communication}}
$$

然后：

$$
x\leftarrow x+\Delta_{\text{communication}}
$$

---

#### FFN

生成：

$$
\Delta_{\text{computation}}
$$

然后：

$$
x\leftarrow x+\Delta_{\text{computation}}
$$

所以 residual viewpoint可以把 Transformer看成：

$$
\boxed{
\text{representation}
+
\text{series of learned updates}
}
$$

---

## 69. 为什么这个视角比“每层重新编码”更准确？

因为每个 sub-layer只负责输出：

$$
F(x)
$$

然后 architecture强制：

$$
x+F(x)
$$

所以模块天然处在：

> update role。

这也是为什么现代 mechanistic interpretability常用：

> modules write to residual stream

这种语言。

---

## 70. Residual Connection 会不会让网络变成线性累加？

不会。

因为：

$$
F_l
$$

本身高度 nonlinear：

- Attention有 Softmax；
- FFN有 activation；
- LayerNorm也非固定线性；
- 多层 input不断变化。

所以虽然外层是加法，

整体仍然是复杂 nonlinear function。

---

## 71. 为什么简单加法反而这么强？

因为简单 add operation不会成为主要表达瓶颈。

复杂性都放在：

$$
F(x)
$$

里面。

Shortcut只提供：

> stable reference path。

可以理解：

> “主路尽量简单，复杂变化放支路。”

这种架构思想非常强。

---

## 72. Residual Branch 可以学负值吗？

当然。

$$
F(x)
$$

可以：

- 增加 feature；
- 减少 feature；
- 翻转；
- 重组。

例如：

$$
x=5
$$

如果目标：

$$
y=2
$$

那么：

$$
F(x)=-3
$$

所以 residual不是：

> 只能“往上加”。

加法里的 residual可以是负值。

---

## 73. 这也是为什么 ReLU 位置很重要

如果强制 residual branch最后：

$$
F(x)\ge0
$$

那么它无法自由表示：

> negative correction。

2016 Identity Mappings论文就讨论了 activation placement对 residual form和signal propagation的影响。

所以 residual function最好能够表示：

$$
(-\infty,+\infty)
$$

范围的 correction。

---

## 74. Transformer Attention Output 可以是负的吗？

可以。

虽然 Attention weights：

$$
\alpha_{ij}\ge0
$$

且和为：

$$
1
$$

但 Value vectors：

$$
v_j
$$

本身各维可以正或负。

随后还有 output projection：

$$
W_O
$$

所以 Attention sub-layer output：

$$
F(x)
$$

完全可以包含正负 residual updates。

---

## 75. FFN Residual 也可以为负

即使中间使用 ReLU：

$$
r\ge0
$$

第二层 Linear：

$$
rW_2+b_2
$$

可以产生：

> 正或负输出。

所以最终 FFN residual不是只能增加 feature value。

---

## 76. 为什么 Identity Shortcut 没有 Learned Scale？

经典：

$$
y=x+F(x)
$$

shortcut coefficient固定是：

$$
1
$$

这保证了：

> identity baseline确实存在。

现代架构有时会使用：

- learned residual scaling；
- LayerScale；
- ReZero；
- gated residual；

但这些是后续变体。

经典 ResNet / Transformer主线先理解：

$$
\boxed{
1\cdot x+F(x)
}
$$

---

## 77. 如果变成 y=0.1x+F(x) 还是 Residual 吗？

广义上仍可叫 skip/residual-style connection。

但它不再是：

> pure identity shortcut。

Forward / backward signal都被：

$$
0.1
$$

缩放。

这可能显著影响深层传播。

所以 identity coefficient：

$$
1
$$

本身很重要。

---

## 78. Residual Addition 会让数值越来越大吗？

如果一直：

$$
x_{l+1}=x_l+F_l(x_l)
$$

确实需要关心 activation scale。

实际网络使用：

- normalization；
- initialization；
- residual scaling；
- training dynamics；

来控制。

Transformer里 LayerNorm正是整体稳定机制的重要部分。

所以不能只看：

$$
+
$$

号就认为值必然线性爆炸。

---

## 79. LayerNorm 为什么和 Residual 经常一起出现？

Residual会持续累积不同 sub-layer updates。

LayerNorm帮助：

> 控制每个 token hidden features的统计尺度。

原始 Transformer直接采用：

$$
LN(x+F(x))
$$

这使每个 sub-layer之后 representation重新归一化。

现代 Pre-LN则在 update计算前规范输入。

两种都体现：

> residual + normalization需要联合设计。

---

## 80. Residual Connection 本身会 Normalize 吗？

不会。

$$
x+F(x)
$$

只是 element-wise addition。

它不会自动：

- mean=0；
- variance=1；
- 限制 norm。

所以 Residual 与 LayerNorm必须严格区分。

---

## 81. Residual 和 Dropout 也不是一回事

Residual：

> 提供 shortcut / identity reference。

Dropout：

> 随机 regularization。

标准 Transformer把二者组合：

$$
x+Dropout(F(x))
$$

但功能完全不同。

---

## 82. Residual 和 Dense Connection 也不同

DenseNet式连接常用：

$$
[x,F_1(x),F_2(\cdot),\ldots]
$$

concat features。

Residual：

$$
x+F(x)
$$

addition。

Dense connection会增加 channel dimension，

residual通常保持维度。

所以都叫 skip-type architectures，

但数学结构不同。

---

## 83. Residual Connection 和 Highway Network 也不一样

Highway Networks使用 learned gates控制：

> shortcut与transform path比例。

ResNet原论文特别比较：

> 其 identity shortcut是 parameter-free、始终开放。

因此：

$$
\boxed{
\text{ResNet identity shortcut}
\neq
\text{gated highway}
}
$$

---

## 84. 为什么 Residual 不等于 Ensemble？

有一些理论会把 ResNet从某些角度联系到许多短路径/ensemble-like behavior。

但 canonical definition不是：

> 多模型投票。

Residual block就是一个单一 computation graph：

$$
y=x+F(x)
$$

不要把类比误当定义。

---

## 85. Residual Connection 会不会增加模型表达能力？

这是一个微妙问题。

如果 plain network已经足够通用，

理论函数表达集合可能本来就能表示：

$$
H(x)
$$

Residual重参数化的经典优势主要是：

> optimization / signal propagation。

当然具体 architecture变化也会影响 function class，

但 ResNet原始动机不是：

> “普通网络不能表示这些函数。”

而是：

> “深层 plain network难以优化。”

---

## 86. Degradation 和 Overfitting 的区别

#### Overfitting

Training error低，

test error高。

---

#### Degradation problem

随着 plain network加深：

$$
\boxed{
training error itself becomes higher
}
$$

所以不是简单泛化问题。

这是理解 ResNet历史的关键。

---

## 87. Degradation 和 Vanishing Gradient 的区别

#### Vanishing Gradient

反向传播数值越来越小，

导致早层难训练。

---

#### Degradation

即使网络已经能开始收敛，

更深 plain architecture仍然更难达到好 training solution。

二者可能相关，

但 ResNet原论文明确把 degradation作为：

> 在传统 vanishing/exploding已有缓解之后仍暴露的问题。

---

## 88. Identity Mapping Argument 为什么这么有力？

因为它不是说：

> “我猜更深模型应该更好。”

而是构造一个明确解：

如果浅模型：

$$
f(x)
$$

很好，

深模型可以：

$$
f(x)
$$

后面新增层全部设成：

$$
Identity
$$

于是至少与浅模型一样。

如果训练却找不到，

说明：

$$
\boxed{
\text{优化过程本身有问题}
}
$$

这个逻辑非常漂亮。

---

## 89. Residual Learning 如何直接回应这个构造？

如果新增 block需要 identity，

Residual branch只需：

$$
F(x)=0
$$

所以 architecture让：

> “新增层什么也不做”

成为一个非常自然的状态。

这正好针对 degradation problem。

---

## 90. “F=0 很容易学”是不是严格定理？

不是。

网络参数、normalization、optimization仍很复杂。

原论文说的是：

> 作者假设 / 实验证明 residual formulation更易优化。

不能把这个直觉升级成：

> “SGD必然先找到 F=0。”

正确结论来自：

> strong empirical evidence + architecture reasoning。

---

## 91. Residual Branch初始化如果接近 0，会怎样？

如果：

$$
F(x)\approx0
$$

初始 block：

$$
y\approx x
$$

整个深网络初始更接近：

> identity perturbation stack。

现代架构有很多专门设计让 residual branch初始较小。

但经典 Transformer原论文没有把所有 residual branches显式初始化为严格 0。

所以这是后续设计思想，

不是原始方法定义。

---

## 92. ACT 当前 Xavier Initialization 会让 Residual Branch严格为 0 吗？

不会。

ACT `Transformer._reset_parameters()` 对 dimension > 1 parameters使用：

```python
nn.init.xavier_uniform_
```

所以 Attention/FFN weights不是全零。

Residual shortcut仍是 identity，

但 branch初始化：

> 不是严格零映射。

---

## 93. 那为什么仍然叫 Residual Learning？

因为 architecture形式：

$$
x+F(x;\theta)
$$

存在。

Residual并不要求：

$$
F
$$

初始化必须为 0。

核心是：

> nonlinear branch表示相对于 shortcut baseline的 update。

---

## 94. ACT Encoder 一个 Token 的 Residual Flow

设某个 token：

$$
x_i
$$

Self-Attention输出：

$$
a_i
$$

Post-LN：

$$
h_i
=
LN(
x_i+
Dropout(a_i)
)
$$

FFN：

$$
f_i
=
FFN(h_i)
$$

最后：

$$
\boxed{
y_i
=
LN(
h_i+
Dropout(f_i)
)
}
$$

这就是 ACT Policy Encoder每层。

---

## 95. ACT Decoder 一个 Action Slot 的 Residual Flow

当前 action slot：

$$
t_i
$$

Self-Attention：

$$
s_i
$$

$$
u_i=
LN(
t_i+s_i
)
$$

Cross-Attention：

$$
c_i
$$

$$
v_i=
LN(
u_i+c_i
)
$$

FFN：

$$
f_i
$$

$$
\boxed{
t_i'
=
LN(
v_i+f_i
)
}
$$

省略 dropout。

一个 Decoder Layer对 action slot进行了三次：

> update + preserve。

---

## 96. 为什么 Action Query Identity 不会第一层就被 Cross-Attention完全洗掉？

一个原因就是 residual结构。

query-side representation产生 cross-attention context后：

$$
u_i+c_i
$$

原有 slot state：

$$
u_i
$$

仍显式参与。

所以 observation information是：

> 写入已有 action-slot representation，

而不是直接替换它。

---

## 97. 为什么 Policy Encoder Visual Token也不会只剩全局平均信息？

Self-Attention output是 weighted contexts，

但 residual：

$$
x_i+a_i
$$

保留了原 token-specific representation。

所以 global communication不等于：

> 所有 tokens逐层变成同一个平均向量。

Position、content、residual和FFN共同维持差异化 representations。

---

## 98. Residual Connection 会不会阻止 Token 变化？

不会。

如果 Attention / FFN输出很强，

$$
F(x)
$$

可以显著修改：

$$
x
$$

Residual只是提供一个基线和直接路径。

它不限制：

$$
y
$$

必须接近：

$$
x
$$

---

## 99. 为什么网络不会因为 Identity Path而“偷懒什么都不学”？

如果：

$$
F=0
$$

导致 task loss很高，

gradient会推动：

$$
F
$$

学习有用 updates。

Identity只提供一个：

> 可优化基线。

是否需要大变化由：

> task objective

决定。

---

## 100. Residual Connection 会增加 Loss 项吗？

不会。

它是 architecture内部 forward formula。

不会自动增加：

$$
L_{residual}
$$

这种 loss。

ACT仍然训练：

$$
L_{L1}+\beta L_{KL}
$$

Residual只是决定：

> 梯度怎样通过模型传播、representation怎样更新。

---

## 101. 为什么 Residual 对 End-to-End Training特别自然？

整个：

$$
x+F(x)
$$

完全可微。

Gradient既流：

- shortcut；
- branch。

无需额外训练阶段。

ResNet原论文就强调：

> 网络仍可直接用普通 SGD/backprop端到端训练。

Transformer同样。

---

## 102. Residual Addition 的 Gradient 对 F 参数是什么？

设：

$$
y=x+F(x;\theta)
$$

那么：

$$
\frac{\partial y}{\partial\theta}
=
\frac{\partial F}{\partial\theta}
$$

shortcut没有参数，

所以 branch parameters仍然正常获得 gradient。

Identity path不会：

> 绕开到让 F完全学不到。

---

## 103. Shortcut 是不是会“抢走”梯度？

不能这么理解。

Loss对 branch仍有：

$$
\frac{\partial L}{\partial y}
\frac{\partial F}{\partial\theta}
$$

gradient。

Shortcut只是额外提供：

> input-to-output直接依赖。

不是二选一 routing。

---

## 104. 如果 F 很小，网络是不是接近 Euler Integration？

从数学类比上，

$$
x_{l+1}
=
x_l+F_l(x_l)
$$

确实和离散 dynamical system / Euler step形状相似。

后来 Neural ODE等工作进一步发展这个视角。

但这是后来的数学解释。

ResNet canonical定义不需要把它解释成：

> 微分方程求解器。

可以作为扩展阅读，

不能替代基本 residual原理。

---

## 105. 为什么 Residual Connection 广泛出现在 Vision、Language、Robotics？

因为问题非常通用：

> 深层网络需要持续变换 representation，同时又希望优化和信息传播稳定。

Residual提供的：

$$
\text{identity baseline}+\text{learned update}
$$

与 modality无关。

所以它适用于：

- CNN；
- Transformer；
- multimodal model；
- robot policy。

---

## 106. ACT 为什么特别依赖深层 Residual结构？

ACT Policy Transformer包含：

- 4 Encoder layers；
- 7 Decoder layers；

每层多个 sub-layers。

如果没有 skip connections，

一个 action prediction gradient需要连续穿过大量 Attention / FFN transformations。

Residual architecture让这些 block都变成：

> update-on-existing-state。

这与标准 Transformer训练方式一致。

---

## 107. ACT ResNet18 Backbone本身也有 Residual Connections

有趣的是：

ACT 的图像 backbone：

> ResNet18。

所以 ACT实际上同时在两类网络里使用 residual思想：

#### Vision Backbone

ResNet residual blocks。

#### Transformer

Attention / FFN residual sub-layers。

二者公式精神相同：

$$
\boxed{
x+F(x)
}
$$

但内部：

$$
F
$$

完全不同。

---

## 108. ResNet18里的 F 是什么？

典型 basic block中：

$$
F(x)
$$

由 convolution、normalization、activation等组成。

---

## 109. Transformer里的 F 是什么？

可能是：

$$
F(x)=MHA(x)
$$

或者：

$$
F(x)=FFN(x)
$$

Decoder还可能：

$$
F(x)=CrossAttention(x,M)
$$

所以 residual pattern是一种：

> 可复用 architecture wrapper。

---

## 110. 为什么这篇文章属于 Deep Learning Canonical，而不是 ACT 专属？

因为：

$$
x+F(x)
$$

远早于 ACT，

也是现代深度学习通用基础。

所以 canonical home page应该是：

> `residual-connection.md`

ACT页面只链接回来说明：

> 在 ACT 的哪里使用。

而不是在 ACT architecture中重新复制整套 ResNet历史和梯度推导。

---

## 111. 常见误解一：Residual Connection 就是解决 Gradient Vanishing

**不完整。**

ResNet原论文的核心直接动机是：

> deep plain network degradation / optimization difficulty。

梯度传播是重要解释，

但不能把历史问题压缩成一个词。

---

## 112. 常见误解二：Degradation 就是 Overfitting

**错误。**

degradation现象包括：

$$
\boxed{
更深 plain net 的 training error 更高
}
$$

所以不是普通 overfitting。

---

## 113. 常见误解三：网络更深参数更多，所以一定训练得更好

**错误。**

表达能力更强：

$$
\neq
$$

优化器一定能找到更好的解。

ResNet历史正是反例。

---

## 114. 常见误解四：Residual 让网络直接学习 H(x)

**错误。**

branch学的是：

$$
F(x)=H(x)-x
$$

整体输出才：

$$
H(x)=x+F(x)
$$

---

## 115. 常见误解五：Residual Function 就是 Loss Error

**错误。**

它是：

> mapping相对 identity的差。

不是 supervised residual：

$$
y-\hat y
$$

---

## 116. 常见误解六：F(x) 必须很小

**错误。**

Residual parameterization允许任意需要的 correction。

---

## 117. 常见误解七：Residual 就是把两路平均

**错误。**

经典是：

$$
x+F(x)
$$

不是：

$$
(x+F(x))/2
$$

---

## 118. 常见误解八：Shortcut一定有参数

**错误。**

identity shortcut：

$$
x
$$

没有 trainable parameters。

---

## 119. 常见误解九：Shape 不一样也能直接相加

**错误。**

必须 shape兼容。

不匹配时需要 projection /其他匹配机制。

---

## 120. 常见误解十：有 Residual 就数学保证不会梯度消失

**错误。**

Residual提供直接 identity derivative term，

但完整深网络 gradient仍取决于：

- branch Jacobians；
- normalization；
- activations；
- depth。

---

## 121. 常见误解十一：Transformer 的 residual path 和纯 ResNet identity chain完全一样

**不准确。**

原始 Transformer是 Post-LN：

$$
LN(x+F(x))
$$

LayerNorm位于 addition之后。

---

## 122. 常见误解十二：Original Transformer 是 Pre-LN

**错误。**

2017论文：

$$
\boxed{
Post\text{-}LN
}
$$

---

## 123. 常见误解十三：Pre-LN 只是把代码顺序改一下，没有数学影响

**错误。**

Post-LN：

$$
J_{LN}(I+J_F)
$$

Pre-LN：

$$
I+J_FJ_{LN}
$$

gradient path结构不同。

---

## 124. 常见误解十四：LayerNorm 就是 Residual 的一部分

概念上应分开。

Residual：

$$
x+F(x)
$$

LayerNorm：

$$
LN(\cdot)
$$

Transformer把它们组合使用，

但它们是不同 mechanisms。

---

## 125. 常见误解十五：Dropout 发生在 Identity Shortcut 上

原始 Transformer sub-layer dropout主要作用：

$$
F(x)
$$

branch，

然后：

$$
x+Dropout(F(x))
$$

---

## 126. 常见误解十六：Residual 会阻止网络忘掉旧信息

不保证。

$$
F(x)
$$

可以产生：

$$
-x
$$

或其他强 correction。

Residual只是提供 direct path，

不强制 output保留所有输入。

---

## 127. 常见误解十七：Residual Connection 会自动让模型更浅

网络计算深度仍然存在。

只是有 shortcut paths。

不能说：

> “100层 ResNet实际只有1层。”

---

## 128. 常见误解十八：Attention Residual 是把 Attention Weights 加回去

**错误。**

加回的是：

$$
\boxed{
Attention\ module\ output
}
$$

不是：

$$
[N,N]
$$

attention weight matrix。

---

## 129. 常见误解十九：FFN Residual 是把 3200-D hidden直接加回512-D

**错误。**

FFN先：

$$
512\rightarrow3200\rightarrow512
$$

回到 512 后才 residual add。

---

## 130. 常见误解二十：ACT Residual只在 ResNet18 Backbone里

**错误。**

ACT Transformer Encoder和Decoder每个 sub-layer都有 residual。

---

## 131. 一张最核心的 Residual 图

```text
                 ┌───────────────────────┐
                 │                       │
                 │                       ▼
Input x ─────────┼────────────────────── Add ──→ y
                 │                       ▲
                 ▼                       │
                F(x) ────────────────────┘
```

数学：

$$
\boxed{
y=x+F(x)
}
$$

其中：

$$
x
$$

是 identity path，

$$
F(x)
$$

是 learned update。

---

## 132. 一张 Transformer Encoder 图

```text
x
│
├──────────────────────────┐
│                          │
▼                          │
Multi-Head Attention       │
│                          │
Dropout                    │
│                          │
└──────────────→ Add ←─────┘
                  │
                  ▼
             LayerNorm
                  │
                  ▼
                  h
                  │
├──────────────────────────┐
│                          │
▼                          │
FFN                        │
│                          │
Dropout                    │
│                          │
└──────────────→ Add ←─────┘
                  │
                  ▼
             LayerNorm
                  │
                  ▼
                  y
```

这就是原始 Post-LN Transformer Encoder。

---

## 133. 一张 ACT Decoder Residual 图

```text
Action slot state
      │
      ├──────────────┐
      ▼              │
Self-Attention       │
      │              │
      └────→ Add ←───┘
              │
            Norm
              │
              ├──────────────┐
              ▼              │
      Cross-Attention        │
              │              │
              └────→ Add ←───┘
                      │
                    Norm
                      │
                      ├──────────────┐
                      ▼              │
                     FFN             │
                      │              │
                      └────→ Add ←───┘
                              │
                            Norm
                              │
                              ▼
                    next action-slot state
```

每个 layer：

$$
3
$$

次 update-on-existing-state。

---

## 134. 最值得记住的 Identity Argument

如果新增网络层没有必要，

理论上希望它能：

$$
\boxed{
H(x)=x
}
$$

Plain network：

> nonlinear branch自己学 identity。

Residual network：

$$
H(x)=x+F(x)
$$

只需：

$$
\boxed{
F(x)=0
}
$$

这个重参数化就是 Residual Learning最核心的思想之一。

---

## 135. 最值得记住的 Gradient Argument

普通：

$$
y=F(x)
$$

$$
\frac{\partial y}{\partial x}
=
J_F
$$

Residual：

$$
y=x+F(x)
$$

$$
\boxed{
\frac{\partial y}{\partial x}
=
I+J_F
}
$$

shortcut带来：

$$
I
$$

direct term。

但：

$$
\boxed{
\text{direct term}
\neq
\text{absolute gradient guarantee}
}
$$

---

## 136. 最值得记住的 Transformer Nuance

Original Transformer：

$$
\boxed{
y=LN(x+F(x))
}
$$

所以：

$$
\boxed{
J
=
J_{LN}(I+J_F)
}
$$

不是纯 identity chain。

Pre-LN：

$$
\boxed{
y=x+F(LN(x))
}
$$

所以：

$$
\boxed{
J
=
I+J_FJ_{LN}
}
$$

这就是为什么 norm placement和 residual path要一起理解。

---

## 137. 一句话真正理解 Residual Connection

> **Residual Connection 并不是简单“把旧信息保存下来”，而是把一个深层模块的任务从直接学习完整目标映射 $H(x)$，重参数化为在显式 identity baseline 上学习修正量 $F(x)=H(x)-x$，使“这一层什么也不改”可以通过 $F(x)\approx0$ 自然实现；同时 shortcut 在前向提供直接信息路径，在反向产生 $I+J_F$ 中的 identity term，从而通常使深网络更容易优化和传播信号。**

---

## 138. 一句话真正理解 Transformer 里的 Residual

> **Transformer 把 Attention、Cross-Attention 和 FFN 都当作对当前 hidden representation 的“更新函数”而不是完整替代函数：模块先计算一个新的 context / feature update $F(x)$，再通过 $x+F(x)$ 写回已有 representation；因此每一层更像不断在同一 hidden state 上增加新的通信结果和局部计算结果。**

---

## 139. 一句话连接 ACT

> **ACT 的 Policy Encoder、Policy Decoder 和 training-only CVAE Encoder 都继承了这种 residual update结构：visual/joint/latent memory tokens 与 action-query slots并不会在每次 Attention或 FFN后被完全替换，而是把新的 cross-token、observation-to-action和feature-processing信息作为 residual update加到当前 512-D state上，再由 LayerNorm进入下一阶段。**

---

## 140. 下一篇：Layer Normalization

Residual Connection已经解释了：

$$
x+F(x)
$$

但 Transformer原论文实际使用的是：

$$
\boxed{
LayerNorm(
x+F(x)
)
}
$$

于是下一个问题自然出现：

> **为什么相加之后还要做 LayerNorm？**

下一篇：

> **[Layer Normalization：为什么 Transformer 每一层都要重新“标准化”？](./layer-normalization.md)**

会详细解释：

- LayerNorm到底对哪些维度求 mean / variance；
- 为什么它和 BatchNorm完全不同；
- 为什么：
  $$
  mean=0,\ variance=1
  $$
  不意味着“服从标准正态分布”；
- $\gamma,\beta$ 为什么让 normalization仍然可学习；
- 为什么 LayerNorm不需要 batch statistics；
- Transformer Post-LN / Pre-LN 的数学差别；
- LayerNorm Jacobian为什么会影响 residual gradient path；
- ACT中 `nn.LayerNorm(512)` 到底对 `[1202,B,512]` / `[k,B,512]` 哪个轴工作；
- inference时 LayerNorm为什么不需要 running mean / running variance。

---

### Primary Source：Deep Residual Learning

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

**Deep Residual Learning for Image Recognition.**  
CVPR 2016.

- arXiv: https://arxiv.org/abs/1512.03385
- HTML: https://arxiv.org/html/1512.03385

论文核心事实：

#### Degradation Problem

作者观察到：

> deeper plain networks can have higher **training error** than shallower counterparts。

因此该问题：

> 不是单纯 overfitting。

论文进一步指出：

> 若新增 layers能够实现 identity mapping，则更深模型至少存在一个与浅模型同样好的构造解。

实际 optimizer却难以找到这样的解，

说明：

> optimization difficulty 是核心问题。

---

#### Residual Reformulation

目标 mapping：

$$
\mathcal H(x)
$$

改写成：

$$
\boxed{
\mathcal F(x)
=
\mathcal H(x)-x
}
$$

于是：

$$
\boxed{
\mathcal H(x)
=
\mathcal F(x)+x
}
$$

经典 residual block：

$$
\boxed{
y
=
\mathcal F(x,\{W_i\})
+
x
}
$$

当 shape不一致：

$$
\boxed{
y
=
\mathcal F(x,\{W_i\})
+
W_sx
}
$$

---

#### Identity Shortcut

论文强调 identity shortcut：

- 不引入 trainable parameter；
- 除 element-wise addition外几乎不增加计算；
- 当 input/output shape一致时直接使用。

---

### Background：Identity Mappings

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

**Identity Mappings in Deep Residual Networks.**  
ECCV 2016.

- arXiv: https://arxiv.org/abs/1603.05027

该论文进一步分析：

> identity shortcut与 after-addition identity mapping 对 forward/backward signal propagation的重要性。

并提出：

> full pre-activation residual unit。

这一工作是理解后续“clean identity path”思想的重要背景，

但不等于 Transformer Pre-LN 的定义。

---

### Transformer Primary Source

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,  
Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin.

**Attention Is All You Need.**  
NeurIPS 2017.

- arXiv: https://arxiv.org/abs/1706.03762
- PDF: https://arxiv.org/pdf/1706.03762

Section 3.1 明确规定：

> Encoder / Decoder 每个 sub-layer外都使用 residual connection followed by layer normalization。

即：

$$
\boxed{
LayerNorm(
x+
Sublayer(x)
)
}
$$

所有 sub-layer和 embedding output：

$$
\boxed{
d_{\text{model}}=512
}
$$

以便 residual addition维度匹配。

---

### Pre-LN / Post-LN Background

Ruibin Xiong et al.

**On Layer Normalization in the Transformer Architecture.**  
ICML 2020.

- arXiv: https://arxiv.org/abs/2002.04745

该工作分析 LayerNorm位置对 Transformer gradient behavior与训练稳定性的影响，

区分：

#### Post-LN

$$
\boxed{
LN(
x+F(x)
)
}
$$

#### Pre-LN

$$
\boxed{
x+
F(
LN(x)
)
}
$$

本文使用这一工作帮助解释：

> 为什么 residual path不能脱离 LayerNorm placement单独讨论。

---

### ACT Official Implementation

Official repository:

https://github.com/tonyzhaozh/act

File:

https://github.com/tonyzhaozh/act/blob/main/detr/models/transformer.py

#### Encoder Post-LN

当前代码：

```python
src2 =
    self.self_attn(
        q,
        k,
        value=src,
        ...
    )[0]

src =
    src +
    self.dropout1(src2)

src =
    self.norm1(src)

src2 =
    self.linear2(
        self.dropout(
            self.activation(
                self.linear1(src)
            )
        )
    )

src =
    src +
    self.dropout2(src2)

src =
    self.norm2(src)
```

即：

$$
\boxed{
SelfAttention
\rightarrow
Add
\rightarrow
Norm
}
$$

和：

$$
\boxed{
FFN
\rightarrow
Add
\rightarrow
Norm
}
$$

---

#### Decoder Post-LN

当前代码：

```python
tgt =
    tgt +
    self.dropout1(
        self_attn_output
    )

tgt =
    self.norm1(tgt)

tgt =
    tgt +
    self.dropout2(
        cross_attn_output
    )

tgt =
    self.norm2(tgt)

tgt =
    tgt +
    self.dropout3(
        ffn_output
    )

tgt =
    self.norm3(tgt)
```

所以 Decoder每层拥有三条 residual update paths：

- Self-Attention；
- Cross-Attention；
- FFN。

---

#### Pre-LN Support

官方代码同时提供：

```python
forward_pre(...)
```

其结构类似：

$$
\boxed{
x
+
Sublayer(
LayerNorm(x)
)
}
$$

并由：

```python
normalize_before
```

切换。

当前 constructor默认：

```python
normalize_before=False
```

所以 canonical released ACT默认：

$$
\boxed{
Post\text{-}LN
}
$$

---

### 本文知识连接

#### 前置

- [Transformer Encoder](./transformer-encoder.md)
- [Transformer Decoder](./transformer-decoder.md)
- [Feed-Forward Network](./feed-forward-network.md)

#### Deep Learning Fundamentals

- Gradient
- Jacobian
- Chain Rule
- Identity Matrix
- Identity Mapping

#### Transformer Components

- [Layer Normalization](./layer-normalization.md)
- [Dropout](./dropout.md)
- [Self-Attention](./self-attention.md)
- [Cross-Attention](./cross-attention.md)
- Pre-LN vs Post-LN

#### Vision

- ResNet

#### Robot Learning

- [ACT Architecture](../robot-learning/act/architecture.md)
- [ACT Training](../robot-learning/act/training.md)
- [ACT Complete Data Flow](../robot-learning/act/complete-data-flow.md)

#### 下一步

- [Layer Normalization](./layer-normalization.md)
