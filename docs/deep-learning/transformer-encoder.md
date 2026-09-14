---
title: "Transformer Encoder：一层到底对 Token 做了什么？"
description: "把 Multi-Head Self-Attention、Residual Connection、LayerNorm 与 Position-wise FFN 组装成完整 Transformer Encoder Layer，并解释 Post-LN / Pre-LN、shape flow、参数共享边界，以及 ACT 的 CVAE Encoder 与 Policy Encoder 如何分别使用 Encoder stack。"
status: reviewed
pageType: concept
canonical: /deep-learning/transformer-encoder
updated: "2026-09-15"
---

# Transformer Encoder：一层到底对 Token 做了什么？

前面我们已经把 Transformer 的很多零件分别拆开：

- [Attention](./attention.md)
- [Query / Key / Value](./qkv.md)
- [Dot Product](./dot-product.md)
- [Softmax](./softmax.md)
- [Self-Attention](./self-attention.md)
- [Multi-Head Attention](./multi-head-attention.md)
- [Positional Encoding](./positional-encoding.md)
- [Causal Mask](./causal-mask.md)

现在终于可以回答一个更完整的问题：

> **一个 token 进入 Transformer Encoder Layer 之后，到底经历了什么？**

很多教程会直接画：

```text
Input
↓
Multi-Head Attention
↓
Add & Norm
↓
Feed Forward
↓
Add & Norm
```

这张图没有错。

但如果只停在这里，很容易留下很多真正关键的问题：

1. Attention 已经很强了，为什么后面还需要 FFN？
2. Residual 到底加的是谁和谁？
3. LayerNorm 是对 batch 做，还是对 token 做？
4. 为什么 FFN 要先从 512 扩到 2048，再缩回 512？
5. FFN 会不会让不同 token 再次互相交流？
6. Transformer 原论文里的 LayerNorm 到底放在 residual 前还是后？
7. “6 个 identical encoder layers”是不是 6 层共享同一套权重？
8. ACT 为什么用 4 个 Encoder layers，却把 FFN 扩成 3200？
9. ACT 里面明明有两个 Transformer Encoder，它们是不是同一个模块？
10. Policy Encoder 的 $1202\times512$ 到底怎样一层层变成 observation memory？

这一篇把完整 Encoder stack 拆开。

---

## 1. Transformer Encoder 的任务是什么？

先不要急着看公式。

Encoder 接收一组 input representations：

$$
X^{(0)}
=
[
x_1^{(0)},
x_2^{(0)},
\ldots,
x_n^{(0)}
]
$$

每个 token：

$$
x_i^{(0)}
\in
\mathbb R^{d_{\text{model}}}
$$

Encoder 的目标不是：

> 立刻生成最终分类结果或动作。

它主要做的是：

> **把每个 token 从“当前局部 representation”逐层更新成“结合全局 context 的 representation”。**

最终：

$$
X^{(N)}
=
[
x_1^{(N)},
\ldots,
x_n^{(N)}
]
$$

仍然有：

$$
n
$$

个 positions。

但每个 position 的 vector 已经融合了其他位置的信息。

所以 Encoder 更像：

$$
\boxed{
\text{contextual representation builder}
}
$$

---

## 2. 原始 Transformer Encoder 有多少层？

《Attention Is All You Need》Section 3.1：

$$
\boxed{
N=6
}
$$

也就是：

> 6 个相同结构的 Encoder Layers 堆叠。

但这里的：

> “identical layers”

必须立刻澄清。

它指：

> **architecture 相同。**

并不是：

> **所有层共享同一套参数。**

---

## 3. “相同结构”不等于“参数共享”

第 1 层有自己的：

$$
W_Q^{(1)},
W_K^{(1)},
W_V^{(1)},
W_O^{(1)}
$$

以及 FFN：

$$
W_1^{(1)},W_2^{(1)}
$$

第 2 层有自己的：

$$
W_Q^{(2)},
W_K^{(2)},\ldots
$$

通常：

$$
W_Q^{(1)}
\neq
W_Q^{(2)}
$$

它们是不同 trainable parameters。

所以：

$$
\boxed{
\text{same layer design}
\neq
\text{shared weights}
}
$$

这也是为什么不同深度的 layers 可以学到不同层级的 representations。

---

## 4. 一个 Encoder Layer 只有两个核心 Sub-Layers

原始论文定义每一个 Encoder Layer 包含：

#### Sub-layer 1

$$
\boxed{
\text{Multi-Head Self-Attention}
}
$$

#### Sub-layer 2

$$
\boxed{
\text{Position-wise Feed-Forward Network}
}
$$

每个 sub-layer 外面再加：

- Residual Connection；
- Layer Normalization。

原始 2017 Transformer 的形式是：

$$
\boxed{
LayerNorm(
x+Sublayer(x)
)
}
$$

这就是经典：

> **Post-LN Transformer**

---

## 5. 原始 Encoder Layer 的完整公式

假设输入：

$$
X
$$

第一步 Self-Attention：

$$
A
=
MHA(X)
$$

然后 residual：

$$
X+A
$$

然后 LayerNorm：

$$
\boxed{
H
=
LN(
X+MHA(X)
)
}
$$

接着 FFN：

$$
F
=
FFN(H)
$$

再 residual：

$$
H+F
$$

再 LayerNorm：

$$
\boxed{
Y
=
LN(
H+FFN(H)
)
}
$$

所以一整层：

$$
\boxed{
X
\rightarrow
MHA
\rightarrow
Add
\rightarrow
LN
\rightarrow
FFN
\rightarrow
Add
\rightarrow
LN
\rightarrow
Y
}
$$

---

## 6. Dropout 放在哪里？

原始 Transformer 论文训练部分明确写：

> sub-layer output 在与 residual input 相加并进行 normalization 前应用 dropout。

所以更完整：

$$
\boxed{
H=
LN(
X+
Dropout(
MHA(X)
)
)
}
$$

以及：

$$
\boxed{
Y=
LN(
H+
Dropout(
FFN(H)
)
)
}
$$

Base Transformer：

$$
P_{drop}=0.1
$$

---

## 7. Encoder Layer 为什么不会改变 Token 数量？

假设输入：

$$
X
\in
\mathbb R^{n\times d_{\text{model}}}
$$

MHA 输出：

$$
\in
\mathbb R^{n\times d_{\text{model}}}
$$

Residual 要求：

$$
X+MHA(X)
$$

shape 完全一致。

LayerNorm：

> 不改变 shape。

FFN：

$$
d_{\text{model}}
\rightarrow
d_{ff}
\rightarrow
d_{\text{model}}
$$

最终又回到：

$$
n\times d_{\text{model}}
$$

所以一整层：

$$
\boxed{
[n,d_{\text{model}}]
\rightarrow
[n,d_{\text{model}}]
}
$$

---

## 8. 为什么 d_model 必须在 Sub-Layer 外保持不变？

因为 residual connection 需要：

$$
x+F(x)
$$

如果：

$$
x\in\mathbb R^{512}
$$

而：

$$
F(x)\in\mathbb R^{300}
$$

就不能直接相加。

所以原始 Transformer 明确让：

> 所有 sub-layers 和 embedding layers 的输出都具有：

$$
d_{\text{model}}=512
$$

这样 residual path 才能自然工作。

---

## 9. 第一个 Sub-Layer：Multi-Head Self-Attention

输入：

$$
X
=
[x_1,\ldots,x_n]
$$

每个 head：

$$
Q_h=XW_h^Q
$$

$$
K_h=XW_h^K
$$

$$
V_h=XW_h^V
$$

然后：

$$
head_h
=
softmax
\left(
\frac{
Q_hK_h^\top
}{
\sqrt{d_k}
}
\right)V_h
$$

所有 heads：

$$
MHA(X)
=
Concat(
head_1,\ldots,head_H
)W^O
$$

所以 Self-Attention 的职责可以概括成：

> **让不同 token positions 交换信息。**

---

## 10. 为什么 Attention 是“跨 Token”操作？

对于第 $i$ 个 token：

$$
o_i
=
\sum_j
\alpha_{ij}v_j
$$

它显式包含：

$$
j=1,\ldots,n
$$

其他 positions 的信息。

所以：

$$
x_i
$$

经过 Attention 后，

可以吸收：

- token 1；
- token 2；
- ...
- token n；

的 representation。

这就是：

$$
\boxed{
\text{token mixing}
}
$$

---

## 11. Attention 之后为什么还需要 FFN？

这是理解 Encoder 最关键的问题之一。

如果 Attention 已经把不同 token 的信息都融合进来，

是不是直接多堆几层 Attention 就够了？

Transformer 选择不是这样。

Attention 主要做：

> **跨 positions 的动态信息路由与线性加权聚合。**

而 FFN 提供：

> **每个 position 内部的 nonlinear feature transformation。**

所以可以非常粗略地分工：

$$
\boxed{
Attention
=
\text{where to get information from}
}
$$

$$
\boxed{
FFN
=
\text{how to transform the resulting features}
}
$$

这不是数学上的严格唯一分解，

但非常有用。

---

## 12. Position-Wise FFN 的原始公式

Transformer 原论文 Section 3.3：

$$
\boxed{
FFN(x)
=
\max(
0,
xW_1+b_1
)
W_2+b_2
}
$$

也就是：

$$
\boxed{
Linear
\rightarrow
ReLU
\rightarrow
Linear
}
$$

对于 Base Transformer：

$$
d_{\text{model}}=512
$$

$$
d_{ff}=2048
$$

所以：

$$
\boxed{
512
\rightarrow
2048
\rightarrow
512
}
$$

---

## 13. 为什么叫 Position-Wise？

假设：

$$
X
\in
\mathbb R^{n\times512}
$$

FFN 对每个 row：

$$
x_i
$$

独立应用同一个函数：

$$
FFN(x_i)
$$

也就是说：

$$
\boxed{
y_i=FFN(x_i)
}
$$

不会直接计算：

$$
x_i
$$

和：

$$
x_j
$$

之间的 interaction。

所以：

> FFN 本身不跨 token。

---

## 14. 但所有 Token 使用的是同一个 FFN 参数

在同一 Encoder layer 中：

$$
x_1
$$

用：

$$
W_1,W_2,b_1,b_2
$$

$$
x_2
$$

也用同一套：

$$
W_1,W_2,b_1,b_2
$$

所以它不是：

> position 1 一套 MLP，position 2 一套 MLP。

而是：

$$
\boxed{
\text{same FFN function applied independently to every position}
}
$$

这就是论文所谓：

> applied to each position separately and identically。

---

## 15. 不同 Encoder Layers 的 FFN 参数一样吗？

不一样。

原论文明确说明：

> 不同 positions 共享同一层参数，但不同 layers 使用不同参数。

所以：

$$
FFN^{(1)}
\neq
FFN^{(2)}
$$

通常成立。

这又回到：

> identical architecture ≠ weight sharing。

---

## 16. FFN 为什么先扩维再缩回？

原始：

$$
512
\rightarrow
2048
\rightarrow
512
$$

中间宽度是：

$$
4\times
$$

model width。

为什么不直接：

$$
512\rightarrow512
$$

？

关键在于：

> 中间高维空间 + 非线性激活提供更大的 feature transformation capacity。

第一层：

$$
h=xW_1+b_1
$$

把 512-D representation 映射到：

$$
2048
$$

个 intermediate features。

然后 ReLU：

$$
h'=\max(0,h)
$$

引入非线性。

最后：

$$
h'W_2+b_2
$$

重新组合回 512-D。

---

## 17. 如果没有 ReLU，两层 Linear 会怎样？

如果：

$$
FFN(x)
=
(xW_1)W_2
$$

没有非线性，

那么：

$$
=
x(W_1W_2)
$$

仍然只是：

> 一个 Linear Transformation。

即使：

$$
512\rightarrow2048\rightarrow512
$$

也可以合并成一个：

$$
512\rightarrow512
$$

矩阵。

所以：

$$
\boxed{
\text{hidden expansion 本身不是重点}
}
$$

真正关键是：

$$
\boxed{
\text{expansion + nonlinear activation}
}
$$

---

## 18. ReLU 提供了什么？

$$
ReLU(z)
=
\max(0,z)
$$

它让不同 input regions：

> 激活不同 intermediate dimensions。

因此 FFN 不再是一个固定全局线性映射。

它能实现：

> piecewise nonlinear feature transformation。

现代 Transformer 常用：

- GELU；
- SiLU；
- SwiGLU；

等其他 FFN variants。

但原始 Transformer 是：

$$
ReLU
$$

---

## 19. 为什么 Attention 本身还不够 Nonlinear？

Attention 有 Softmax，

所以整体确实不是纯线性网络。

不能说：

> “没有 FFN，Transformer 就完全线性。”

这是错误的。

更准确：

> FFN 额外提供一个强大的 per-token nonlinear transformation stage。

它和 Attention 承担互补的 computation pattern：

```text
Attention:
跨 token interaction

FFN:
token 内 feature transformation
```

---

## 20. 一个很好用的二维思维

Transformer representation 有两个主要轴：

$$
\boxed{
\text{Sequence Axis}
}
$$

和：

$$
\boxed{
\text{Feature Axis}
}
$$

Self-Attention 主要沿：

> sequence axis

把不同 positions 的信息混合。

FFN 主要沿：

> feature axis

对每个 position 的 feature vector 做 nonlinear remapping。

所以可以粗略记：

$$
\boxed{
\text{Attention mixes tokens}
}
$$

$$
\boxed{
\text{FFN mixes channels/features}
}
$$

---

## 21. 但 Attention 也会混合 Features

严格来说：

Multi-Head Attention 里面：

$$
W_Q,W_K,W_V,W_O
$$

本身也做 feature projections。

所以“Attention 只混 token、FFN 只混 feature”并非数学上完全隔离。

更准确的教学描述：

> Attention 的**独特作用**是建立跨 token 的动态 interaction；

> FFN 的**独特作用**是提供不跨 token 的强 nonlinear per-position transform。

---

## 22. Residual Connection 到底是什么？

假设一个 sub-layer：

$$
F(x)
$$

Residual Connection：

$$
\boxed{
y=x+F(x)
}
$$

不是：

$$
y=F(x)
$$

原始 ResNet 工作把这种设计解释为：

> 学习 residual function 相对于 identity reference。

Transformer 把同样思想包围在：

- Attention；
- FFN；

两个 sub-layers 外。

---

## 23. 为什么叫 Skip Connection？

因为 input：

$$
x
$$

有一条路径：

> 跳过 sub-layer，

直接到加法节点。

```text
x ─────────────────┐
│                  │
▼                  │
F(x)               │
│                  │
└────── + ◀────────┘
        │
        ▼
      x+F(x)
```

所以也叫：

> skip connection。

---

## 24. Residual 的直觉是什么？

没有 residual：

$$
x\rightarrow F(x)
$$

sub-layer 必须自己产生：

> 完整新 representation。

有 residual：

$$
x\rightarrow x+F(x)
$$

sub-layer 可以更自然地学习：

> “在原 representation 上增加什么更新？”

所以可以理解：

$$
\boxed{
F(x)
=
\text{representation update}
}
$$

而：

$$
x
$$

保留原信息的直接路径。

---

## 25. Attention Residual 可以怎样理解？

输入 token representation：

$$
x_i
$$

Attention 返回：

$$
a_i
$$

其中包含：

> 从其他 tokens 读来的 context。

Residual：

$$
x_i+a_i
$$

可以直觉化：

> **我原来是谁 + 我从上下文中新读到了什么。**

这是一个非常自然的信息更新结构。

---

## 26. FFN Residual 又怎样理解？

经过 Attention/Norm 后：

$$
h_i
$$

FFN 提供：

$$
f_i=FFN(h_i)
$$

Residual：

$$
h_i+f_i
$$

可以理解：

> **保留当前 contextual representation，再加入 nonlinear feature update。**

---

## 27. Residual 为什么对 Deep Network 有帮助？

ResNet 原论文的核心动机就是：

> 更深网络难优化，residual formulation 能缓解 optimization degradation。

从梯度角度，

若：

$$
y=x+F(x)
$$

那么：

$$
\frac{\partial y}{\partial x}
=
I+
\frac{\partial F}{\partial x}
$$

所以 gradient 有一条：

$$
I
$$

identity path。

这能帮助深层网络中的信息与梯度传播。

不要把它简化成：

> “Residual 彻底解决 gradient vanishing。”

更准确是：

> 它提供了直接 identity route，通常使深层优化更容易。

---

## 28. LayerNorm 又为什么出现？

Residual addition 后：

$$
x+F(x)
$$

不同 features 的数值分布可能变化很大。

Transformer 使用：

> Layer Normalization

稳定 representation。

原论文的 Post-LN 形式：

$$
\boxed{
LN(
x+F(x)
)
}
$$

---

## 29. LayerNorm 到底 Normalize 什么？

对于单个 token：

$$
x=
[x_1,x_2,\ldots,x_d]
$$

先计算该 token feature dimensions 的 mean：

$$
\boxed{
\mu
=
\frac1d
\sum_{j=1}^{d}x_j
}
$$

variance：

$$
\boxed{
\sigma^2
=
\frac1d
\sum_{j=1}^{d}
(x_j-\mu)^2
}
$$

然后：

$$
\hat x_j
=
\frac{
x_j-\mu
}{
\sqrt{\sigma^2+\epsilon}
}
$$

最后 learned affine：

$$
\boxed{
y_j
=
\gamma_j\hat x_j+\beta_j
}
$$

---

## 30. LayerNorm 是对 Batch 求 Mean 吗？

**不是。**

这点一定要和 BatchNorm 分清。

Transformer 中常见：

$$
X
\in
[B,N,D]
$$

LayerNorm 通常对：

$$
D
$$

也就是最后一个 feature dimension 做 normalization。

对于每一个：

```text
sample b
token n
```

分别计算自己的：

$$
\mu_{b,n}
$$

和：

$$
\sigma_{b,n}
$$

所以：

$$
\boxed{
\text{一个 token 的 LN 不需要其他 batch samples}
}
$$

---

## 31. LayerNorm 会让不同 Tokens 互相交流吗？

不会。

Token 1 的 LayerNorm：

> 使用 token 1 自己 512 个 features 的 statistics。

Token 2：

> 使用 token 2 自己的 statistics。

所以：

$$
\boxed{
LayerNorm
\text{ does not perform token mixing}
}
$$

跨 token 信息流仍主要由：

> Attention

完成。

---

## 32. 为什么 LayerNorm 特别适合 Sequence Models？

LayerNorm 原论文的重要特点之一是：

> normalization statistics 来自同一个 sample 的 layer activations，而不是 mini-batch statistics。

因此它不像 BatchNorm 那样强依赖：

- batch size；
- batch composition。

对变长 sequence、autoregressive/modeling architectures 很自然。

---

## 33. LayerNorm 后不是 Mean=0、Variance=1 吗？

在：

$$
\hat x
$$

阶段：

> 是。

但最终还有 learned：

$$
\gamma,\beta
$$

所以输出：

$$
y=\gamma\odot\hat x+\beta
$$

不要求最终严格：

$$
mean=0,\quad variance=1
$$

LayerNorm 不是把网络永远锁死在标准正态分布。

它只是先标准化，再允许模型通过 learned scale/shift 恢复有用表示。

---

## 34. LayerNorm 也不是“把 Representation 变成高斯分布”

Normalization：

$$
(x-\mu)/\sigma
$$

只约束：

- sample feature mean；
- sample feature variance。

它不意味着整个 feature distribution：

> 服从 Gaussian。

这和我们之前学习的：

- Normal Distribution；
- Standardization；

要严格区分。

---

## 35. 原始 Transformer 为什么叫 Post-LN？

因为 LayerNorm 放在 sub-layer 和 residual addition：

> **之后。**

即：

$$
\boxed{
y=
LN(
x+F(x)
)
}
$$

所以叫：

> Post-LayerNorm / Post-Norm。

---

## 36. Pre-LN 又是什么？

后来非常常见的设计是：

$$
\boxed{
y=
x+
F(
LN(x)
)
}
$$

即先：

$$
LN(x)
$$

再进入 sub-layer。

例如 Attention：

$$
x
+
MHA(
LN(x)
)
$$

FFN：

$$
h
+
FFN(
LN(h)
)
$$

这叫：

> Pre-LN。

---

## 37. Post-LN vs Pre-LN 图

### Post-LN

```text
x
│
├───────────────┐
│               │
▼               │
Sublayer(x)     │
│               │
└────── + ◀─────┘
        │
        ▼
    LayerNorm
        │
        ▼
        y
```

---

### Pre-LN

```text
x ─────────────────────┐
│                      │
▼                      │
LayerNorm              │
│                      │
▼                      │
Sublayer               │
│                      │
└──────── + ◀───────────┘
           │
           ▼
           y
```

---

## 38. 哪一个才是“Transformer 正宗结构”？

如果问：

> **2017 原始 Transformer**

答案明确是：

$$
\boxed{
Post-LN
}
$$

论文 Section 3.1：

$$
\boxed{
LayerNorm(
x+Sublayer(x)
)
}
$$

---

现代很多 Transformer 使用 Pre-LN。

所以不能说：

> “Transformer 必然先 LayerNorm。”

要看具体 architecture。

---

## 39. ACT 官方代码支持两种

ACT `TransformerEncoderLayer` 有：

```python
normalize_before
```

如果：

```python
False
```

走：

```python
forward_post
```

如果：

```python
True
```

走：

```python
forward_pre
```

所以代码同时实现：

- Post-LN；
- Pre-LN。

---

## 40. ACT 当前默认是哪一个？

当前 `detr/main.py`：

```python
parser.add_argument(
    '--pre_norm',
    action='store_true'
)
```

默认：

$$
False
$$

而 `Transformer` 默认：

```python
normalize_before=False
```

所以 released default configuration 是：

$$
\boxed{
Post-LN
}
$$

也就是和原始 Transformer 的 Add → Norm 顺序一致。

---

## 41. ACT Post-LN Encoder Layer 的官方代码骨架

官方 `forward_post` 逻辑：

```python
q = k = src + pos

src2 = self_attn(
    q,
    k,
    value=src
)[0]

src = src + dropout1(src2)

src = norm1(src)

src2 = linear2(
    dropout(
        activation(
            linear1(src)
        )
    )
)

src = src + dropout2(src2)

src = norm2(src)
```

这几乎就是我们刚才的完整公式。

---

## 42. 把 ACT 代码翻译成数学

设当前 input：

$$
X
$$

position：

$$
P
$$

Attention：

$$
\boxed{
A=
MHA(
Q=X+P,
K=X+P,
V=X
)
}
$$

然后：

$$
\boxed{
H=
LN_1(
X+
Dropout(A)
)
}
$$

FFN：

$$
\boxed{
F=
W_2
\,
Dropout(
ReLU(
W_1H+b_1
)
)
+b_2
}
$$

然后：

$$
\boxed{
Y=
LN_2(
H+
Dropout(F)
)
}
$$

这就是当前 ACT default Encoder Layer。

---

## 43. 为什么 ACT 的 Position 只显式加到 Q/K，而 V=src？

这是 DETR-style Transformer 的设计。

Attention matching：

$$
QK^\top
$$

需要知道：

> token 位于哪里。

所以：

$$
src+pos
$$

用于 Q/K。

Value：

$$
src
$$

主要传递 content representation。

这与原始文本 Transformer：

> 最底部 embedding + PE

的具体注入方式不同。

但核心原则一样：

> position 必须能够影响 routing。

---

## 44. ACT 的 FFN 是 512→2048→512 吗？

**论文最终配置不是。**

原始 Transformer Base：

$$
512
\rightarrow
2048
\rightarrow
512
$$

ACT Table III：

$$
\boxed{
hidden\ dimension=512
}
$$

$$
\boxed{
feedforward\ dimension=3200
}
$$

所以 ACT：

$$
\boxed{
512
\rightarrow
3200
\rightarrow
512
}
$$

---

## 45. 为什么 ACT Code 里 default 写 2048？

因为代码 function 定义：

```python
dim_feedforward=2048
```

继承了通用 DETR/Transformer 默认值。

但 ACT training configuration 会传入：

```text
--dim_feedforward 3200
```

README 示例也是：

```text
--hidden_dim 512
--dim_feedforward 3200
```

所以：

> **function default 不等于 ACT experiment hyperparameter。**

这是读代码时一个很重要的习惯。

---

## 46. ACT 一共有多少 Encoder Layers？

ACT Table III：

$$
\boxed{
4\text{ encoder layers}
}
$$

当前 training script 也固定：

```python
enc_layers = 4
```

所以 Policy Transformer Encoder：

$$
4
$$

层。

CVAE Transformer Encoder 的 `build_encoder(args)` 也使用：

```python
args.enc_layers
```

因此同一配置下也是：

$$
4
$$

层。

---

## 47. 但 ACT 其实有两个不同的 Encoder Stack

这是非常重要的一点。

---

### Encoder A：CVAE Encoder

Training-only。

输入：

```text
[CLS]
joint token
action tokens
```

长度：

$$
k+2
$$

作用：

> 根据 current joints + demonstration action chunk 推断：

$$
\mu,\log\sigma^2
$$

最终得到 latent：

$$
z
$$

---

### Encoder B：Policy Observation Encoder

Training 和 inference 都使用。

输入：

```text
latent token
joint token
1200 visual tokens
```

长度：

$$
1202
$$

作用：

> 形成 observation memory，供 Decoder Cross-Attention读取。

---

## 48. 这两个 Encoder 是不是共享参数？

**不是。**

官方 `DETRVAE` 中：

- `self.encoder` 是 CVAE encoder；
- `self.transformer.encoder` 是 policy transformer 内部的 observation encoder。

它们是两个独立 module objects。

即使：

- 都是 Transformer Encoder；
- 都用 4 layers；
- 都有 hidden dim 512；
- 都有 8 heads；

也不代表参数共享。

所以：

$$
\boxed{
\text{same architecture/config}
\neq
\text{same network weights}
}
$$

---

## 49. 为什么 ACT 要两个 Encoder？

因为它们解决完全不同的问题。

CVAE Encoder：

> **解释 demonstration action sequence 中的 latent variation。**

Policy Encoder：

> **把当前 observation 整理成可以被 action decoder读取的 memory。**

所以虽然都叫 Encoder，

输入和功能完全不同。

---

## 50. CVAE Encoder 一层发生什么？

假设：

$$
k=100
$$

input sequence：

$$
102
$$

tokens：

```text
[CLS]
qpos
a₀
...
a₉₉
```

shape：

$$
\boxed{
[B,102,512]
}
$$

每一层：

$$
[B,102,512]
\rightarrow
[B,102,512]
$$

4 层后仍：

$$
[B,102,512]
$$

最后只取：

$$
[CLS]
$$

位置：

$$
h_{\text{CLS}}
\in
\mathbb R^{512}
$$

再 Linear：

$$
512\rightarrow64
$$

split：

$$
32
$$

维：

$$
\mu
$$

和：

$$
32
$$

维：

$$
\log\sigma^2
$$

---

## 51. 为什么 `[CLS]` 能汇总整个 Action Sequence？

因为 Self-Attention 里：

$$
q_{\text{CLS}}
$$

可以读取：

- qpos token；
- 所有 action tokens。

经过第一层后：

$$
h_{\text{CLS}}^{(1)}
$$

已经可以包含全 sequence context。

后面几层继续：

> 重新基于 contextual representations 做 attention + FFN refinement。

所以最终：

$$
h_{\text{CLS}}^{(4)}
$$

可以成为 latent posterior prediction 的 sequence summary。

---

## 52. Policy Encoder 一层发生什么？

ACT paper：

4 张图像：

$$
4\times
(15\times20)
$$

总 visual positions：

$$
1200
$$

再加：

- latent token；
- qpos token；

总：

$$
\boxed{
1202
}
$$

hidden：

$$
512
$$

所以：

$$
X^{(0)}
\in
\mathbb R^{1202\times512}
$$

忽略 batch。

一层后：

$$
X^{(1)}
\in
\mathbb R^{1202\times512}
$$

4 层后：

$$
X^{(4)}
\in
\mathbb R^{1202\times512}
$$

这个：

$$
X^{(4)}
$$

就是：

> observation memory。

---

## 53. Encoder 并不会把 1202 Tokens 压成一个 Vector

这是非常重要的。

Policy Encoder：

$$
1202\times512
$$

进去，

仍然：

$$
1202\times512
$$

出来。

它不会：

$$
1202\times512
\rightarrow
512
$$

因为后面的 Decoder Cross-Attention 希望保留：

> 多个 memory positions。

这样不同 action queries 可以分别读取不同 memory locations。

---

## 54. 那么“Encoder synthesizes information”是什么意思？

ACT 论文说 Transformer Encoder：

> synthesizes information from different camera viewpoints, joint positions, and style variable。

这不意味着：

> 把它们压成一个单独 global vector。

而是：

> 每个 memory token 都经过多层 Self-Attention，变成 contextualized representation。

例如某个 wrist-camera token 在第 4 层后：

> 已经有可能包含来自其他 camera、joint 和 z 的信息。

---

## 55. 所以 Memory Token 已经不是“纯 Patch”了

Layer 0：

$$
x_j^{(0)}
$$

可能主要是：

> 某个 camera spatial feature。

经过：

$$
x_j^{(1)}
=
EncoderLayer_1(x_j^{(0)},X^{(0)})
$$

它可以吸收其他 tokens。

再经过：

$$
x_j^{(2)}
$$

继续融合。

所以最终：

$$
m_j=x_j^{(4)}
$$

是：

> contextualized memory token。

因此 Decoder Cross-Attention 读取：

> contextualized observation memory，

不是 raw patch features。

---

## 56. 为什么要堆多层？

一层 Self-Attention 已经能让任意 token 直接读任意 token。

那为什么还要：

$$
4
$$

层或：

$$
6
$$

层？

因为：

> “能直接交换信息”不等于“一次交换就完成全部 computation”。

第一层可以建立初步 relations。

FFN 转换这些 features。

第二层再基于新的 representations：

> 重新计算 Q/K/V 和新的 attention patterns。

所以每层都有机会重新问：

> 现在我需要从谁那里读什么？

---

## 57. 第二层 Attention 看到的已经不是原始输入

第 1 层输入：

$$
X^{(0)}
$$

第 1 层输出：

$$
X^{(1)}
$$

第 2 层：

$$
Q^{(2)}
=
X^{(1)}W_Q^{(2)}
$$

而不是：

$$
X^{(0)}W_Q^{(2)}
$$

所以第二层 matching geometry 建立在：

> 第一层已经 contextualized 的 representations 上。

这使深度真正有意义。

---

## 58. 一个逐层直觉

可以非常粗略地想象：

#### Layer 1

“哪些 tokens 可能相关？”

#### FFN 1

“把刚刚聚合的信息变成更有用的 features。”

#### Layer 2

“基于新的理解，再次判断哪些 tokens 应该交互。”

#### FFN 2

“再加工。”

……

这不是说每层真的有固定语义阶段。

只是帮助理解：

> deep Transformer 是 iterative representation refinement。

---

## 59. 为什么每一层 Q/K/V 参数不同？

如果所有层共享：

$$
W_Q,W_K,W_V
$$

每层更像重复同一种 matching rule。

独立参数让不同层可以学习：

> 不同 representation level 的 relationships。

例如底层可能更依赖局部 feature，

高层可能形成更抽象 task relation。

这些具体语义不是 architecture guarantee，

但独立参数提供了这种自由度。

---

## 60. ACT Official Encoder Layer 的 Shape

Policy Encoder 当前：

$$
D=512
$$

heads：

$$
H=8
$$

所以每 head：

$$
d_k=d_v=64
$$

sequence：

$$
N=1202
$$

Multi-Head Self-Attention：

$$
Q,K,V:
[B,8,1202,64]
$$

scores：

$$
[B,8,1202,1202]
$$

head outputs：

$$
[B,8,1202,64]
$$

concat：

$$
[B,1202,512]
$$

然后 FFN：

$$
[B,1202,512]
\rightarrow
[B,1202,3200]
\rightarrow
[B,1202,512]
$$

整层 shape 不变。

---

## 61. FFN 3200 中间表示会变成 1202×3200 吗？

是。

概念上：

$$
X
\in
[B,1202,512]
$$

Linear 1：

$$
\boxed{
[B,1202,3200]
}
$$

ReLU：

$$
[B,1202,3200]
$$

Linear 2：

$$
\boxed{
[B,1202,512]
}
$$

注意：

> 1202 个 tokens 仍然彼此独立地通过 FFN。

只是矩阵运算把它们 batch 化一起计算。

---

## 62. FFN 的参数量其实很大

ACT：

$$
W_1:
512\times3200
$$

参数：

$$
1,638,400
$$

$$
W_2:
3200\times512
$$

也是：

$$
1,638,400
$$

只算这两个 weight matrices：

$$
\boxed{
3,276,800
}
$$

约 3.28M。

所以 FFN 不是 Transformer 中一个“小补丁”。

它往往占很大一部分参数量和计算量。

---

## 63. Attention Projection 参数量对比

MHA 的 Q/K/V/O：

每个大致：

$$
512\times512
$$

4 个：

$$
4\times512^2
$$

$$
=
1,048,576
$$

约：

$$
1.05M
$$

所以在 ACT 的：

$$
d_{ff}=3200
$$

配置下，

一个 Encoder Layer 的 FFN weight 参数：

> 明显多于 MHA projection weights。

这能帮助纠正一个常见印象：

> Transformer 不只是“Attention”。

---

## 64. “Attention Is All You Need” 不等于模型里只有 Attention

原论文标题容易让初学者误解。

Transformer Layer 中非常重要的组件还有：

- FFN；
- Residual；
- LayerNorm；
- Positional Encoding；
- embeddings。

“Attention Is All You Need”主要是在说：

> sequence-to-sequence architecture 不再需要 recurrence / convolution 来建立 sequence dependency。

不是：

> 网络只包含 Attention 算子。

---

## 65. Transformer Encoder 更准确的核心是两种操作交替

可以压成：

$$
\boxed{
\text{Communication}
\rightarrow
\text{Computation}
}
$$

其中：

#### Attention

不同 positions 之间 communication。

#### FFN

每个 position 内部 computation。

然后不断重复。

这是理解 Transformer block 一个很有力量的视角。

---

## 66. 为什么 LayerNorm 放在每个 Sub-Layer 外，而不是整个 Encoder 最后一次？

原始设计希望每次：

- Attention update；
- FFN update；

之后都把 representation 放回稳定的 normalization regime。

所以每一层有两个 LayerNorm。

原始 6-layer Encoder：

> 不是最后才统一做一次 Norm。

---

## 67. ACT Post-LN 默认有没有额外 Encoder Final Norm？

当前 DETR-derived code 在文件头注释里写：

> extra LN at the end of encoder is removed。

代码：

```python
encoder_norm =
    nn.LayerNorm(d_model)
    if normalize_before
    else None
```

ACT 默认：

$$
normalize\_before=False
$$

所以：

$$
encoder\_norm=None
$$

每个 Encoder Layer 自己已经做两次 Post-LN，

stack 末尾没有再额外加一个 LayerNorm。

---

## 68. 为什么 Pre-LN 分支反而有 Encoder Final Norm？

当前代码如果：

$$
normalize\_before=True
$$

每个 layer 是 Pre-LN：

```text
LN → sublayer → residual
```

stack 最后会配置：

$$
encoder\_norm=LayerNorm(d)
$$

因此最后再做一次 LN。

这是一种常见 Pre-LN stack pattern。

但 ACT 默认并不走这条路径。

---

## 69. 这里必须区分“原始论文”和“现代实现”

#### Original Transformer

$$
Post\text{-}LN
$$

#### Many modern Transformers

$$
Pre\text{-}LN
$$

#### ACT released default

$$
Post\text{-}LN
$$

但 ACT code保留：

> `--pre_norm`

开关。

所以以后读论文一定要确认：

> Norm 放哪？

不要只看到 `TransformerEncoderLayer` 就默认一种结构。

---

## 70. 为什么现代模型常喜欢 Pre-LN？

大量后续研究和工程实践发现：

> Pre-LN 往往更容易稳定训练深 Transformer。

直觉之一是 residual stream 提供更直接的 identity gradient route。

但这是 Transformer 之后的发展，

不是 2017 原论文 Encoder 的定义。

这一篇不把后续结果混成原论文事实。

如果需要可以单独写：

> **Pre-LN vs Post-LN：LayerNorm 放哪里为什么会影响深层训练？**

---

## 71. LayerNorm 会删除绝对 Feature Magnitude 信息吗？

Normalization 确实去掉每个 token 的：

- common shift；
- common scale；

然后再通过：

$$
\gamma,\beta
$$

做 learned affine。

所以它改变 representation geometry。

但不能简单说：

> “所有 magnitude information 都彻底丢失。”

因为：

- relative feature pattern仍保留；
- learned network 可以在前后层编码相关信息；
- residual / learned affine 共同作用。

更重要的是：

> LayerNorm 是 architecture transform，不是一个纯无损操作。

---

## 72. Residual + LayerNorm 的顺序为什么重要？

比较：

$$
LN(x+F(x))
$$

和：

$$
x+F(LN(x))
$$

它们不是代数等价。

LayerNorm 是非线性、sample-dependent normalization。

所以：

$$
\boxed{
Post\text{-}LN
\neq
Pre\text{-}LN
}
$$

不能随便交换位置。

---

## 73. Residual 为什么要求 Dropout 在 Add 前？

原始 Transformer residual dropout：

$$
x+
Dropout(
F(x)
)
$$

训练时随机抑制 sub-layer update 的部分 contributions。

identity path：

$$
x
$$

本身仍直接保留。

这和对：

$$
x+F(x)
$$

整体 Dropout 是不同操作。

---

## 74. Evaluation 时 Dropout 怎么办？

训练：

$$
Dropout
$$

随机关闭部分 units / contributions。

调用：

```python
model.eval()
```

后，

标准 Dropout：

> 关闭随机丢弃。

所以 inference 的 Encoder 是 deterministic 的，

假设：

- input 固定；
- parameters 固定；
- 没有其他随机机制。

ACT inference 中：

$$
z=0
$$

且 policy `.eval()`，

因此 latent sampling 和 dropout 随机性都不参与正常 deterministic forward。

---

## 75. Encoder Self-Attention 需要 Causal Mask 吗？

原始 Transformer Encoder：

> 不需要。

ACT Policy Encoder：

> 也不需要 language-style causal mask。

它处理的是：

> 当前已经完整可用的 observation tokens。

CVAE Encoder Training：

> 同样允许 `[CLS] / qpos / action tokens` 全局互看。

所以两者使用：

> full Self-Attention，

但可能有 padding mask。

---

## 76. ACT CVAE Encoder 的 Padding Mask

Training sample 接近 episode 末尾时，

action chunk 不足：

$$
k
$$

步。

剩余位置 padding。

这些 PAD action tokens：

> 不应该被 `[CLS]` 或其他 tokens 当成真实 demonstration。

所以 CVAE encoder 使用：

$$
src\_key\_padding\_mask
$$

屏蔽 padding positions。

这和：

> Causal Mask

完全不同。

---

## 77. Policy Encoder 里的 Padding Mask 呢？

视觉 input 在 ACT 当前 fixed-camera/fixed-grid pipeline 中：

> 1200 visual positions 通常都有有效 feature。

latent/joint 也固定存在。

官方 policy transformer 的 `mask` 路径保留通用接口，

但这里不像 target action chunk padding 那样是核心机制。

所以不要把 CVAE action padding mask 与 Policy Encoder visual mask 混成一件事。

---

## 78. 为什么 Encoder 输出叫 Memory？

因为它接下来被 Decoder 当作：

$$
K/V
$$

source。

经过 Encoder stack：

$$
X^{(0)}
\rightarrow
X^{(4)}
$$

得到：

$$
M
$$

Decoder 每个 action query 通过 Cross-Attention：

$$
Q\leftarrow Decoder
$$

$$
K,V\leftarrow M
$$

所以：

$$
M
$$

就是可以被后续 Query 反复读取的：

> contextual memory。

---

## 79. Memory 不是某种特殊神经网络存储器

它其实就是：

> Encoder 输出 tensor。

例如 ACT：

$$
\boxed{
M
\in
[B,1202,512]
}
$$

在官方 code 里变量名直接是：

```python
memory
```

不要把它理解成：

- database；
- RNN hidden memory；
- long-term agent memory；

它只是 Transformer Encoder outputs 在 Decoder context 下的名称。

---

## 80. 为什么 Decoder 用 Encoder 最后一层，而不是所有层？

原始 Transformer 架构：

> Decoder Cross-Attention 的 memory keys / values 来自 Encoder stack output。

也就是最终 encoder representations。

ACT 当前代码也是：

```python
memory = self.encoder(...)
hs = self.decoder(
    tgt,
    memory,
    ...
)
```

并没有把所有 intermediate Encoder layers 全部堆给 Decoder。

当然其他现代 architectures 可以做 multi-scale / multi-layer memory，

但这不是原始设计。

---

## 81. ACT Policy Encoder 从 1202 个 Token 到底发生什么？

完整写一遍。

初始：

$$
X^{(0)}
=
[
z,
q,
v_1,
v_2,
\ldots,
v_{1200}
]
$$

每个：

$$
512
$$

维。

---

Layer 1：

$$
X^{(1)}
=
EncoderLayer_1(
X^{(0)}
)
$$

---

Layer 2：

$$
X^{(2)}
=
EncoderLayer_2(
X^{(1)}
)
$$

---

Layer 3：

$$
X^{(3)}
=
EncoderLayer_3(
X^{(2)}
)
$$

---

Layer 4：

$$
\boxed{
M=
X^{(4)}
}
$$

shape 始终：

$$
1202\times512
$$

---

## 82. 每一层内发生：

$$
X^{(l)}
$$

先：

$$
Q,K=
X^{(l)}+P
$$

Value：

$$
V=X^{(l)}
$$

Multi-Head Self-Attention：

$$
A^{(l)}
=
MHA(
X^{(l)},P
)
$$

Residual + Norm：

$$
H^{(l)}
=
LN_1(
X^{(l)}+
Dropout(A^{(l)})
)
$$

FFN：

$$
F^{(l)}
=
FFN(
H^{(l)}
)
$$

再：

$$
\boxed{
X^{(l+1)}
=
LN_2(
H^{(l)}+
Dropout(F^{(l)})
)
}
$$

这就是 ACT Policy Encoder 的 layer-by-layer math。

---

## 83. CVAE Encoder 的数学骨架几乎一样

区别主要不是 Encoder Layer 本身，

而是：

> 输入 sequence 不同。

CVAE：

$$
X_{\text{CVAE}}
=
[
CLS,
q,
a_0,\ldots,a_{k-1}
]
$$

Policy：

$$
X_{\text{policy}}
=
[
z,
q,
visual_1,\ldots,visual_{1200}
]
$$

所以同一个 Transformer Encoder abstraction：

> 可以服务两个完全不同任务。

---

## 84. 这说明 Transformer Encoder 是一种 General Sequence Processor

它并不要求 token 一定是：

> 文字。

只要你能把某种对象变成：

$$
d_{\text{model}}
$$

维 tokens，

并提供必要位置/结构信息，

Encoder 就可以对其做：

> global interaction + per-token transformation。

所以 token 可以是：

- words；
- image patches；
- actions；
- joints；
- latent variables；
- multimodal features。

---

## 85. 为什么所有不同 Modality 都要投影到 512？

ACT：

- visual feature → 512；
- joint → Linear → 512；
- z → Linear → 512。

因为它们要进入同一个 Transformer hidden space：

$$
d_{\text{model}}=512
$$

这样：

- MHA projections；
- residual；
- LayerNorm；
- FFN；

都能统一处理。

所以：

$$
\boxed{
\text{common hidden width}
=
\text{shared representation interface}
}
$$

---

## 86. 512 维意味着不同 Token 语义相同吗？

当然不是。

一个 visual token：

> 表示图像 feature。

joint token：

> 表示机器人关节状态。

latent token：

> 表示 style condition。

它们只是都被映射成：

$$
512
$$

维 vector。

同维度意味着：

> 可以进入统一计算空间。

不是说：

> 物理含义相同。

---

## 87. Position Embedding 又帮模型区分结构角色

Policy Encoder 中：

- visual tokens 有 2D positional encoding；
- latent/joint 有 additional learned position embeddings。

所以即使它们都 512-D，

模型仍能利用：

> token content + structural identity

进行 Attention。

---

## 88. Encoder Layer 会不会直接预测 z 或 action？

不会。

CVAE Encoder stack：

> 只产生 contextualized sequence。

之后：

$$
h_{\text{CLS}}
$$

再进入 latent projection：

$$
\rightarrow\mu,\log\sigma^2
$$

---

Policy Encoder：

> 只产生 observation memory。

之后由 Transformer Decoder：

> Cross-Attention + action head

产生 action chunk。

所以：

$$
\boxed{
Encoder
=
\text{representation stage}
}
$$

不是最终 task head。

---

## 89. 为什么一个 Encoder Layer 有两个 Norm，而不是一个？

因为它有两个 independently residualized sub-layers：

1. Attention；
2. FFN。

原始 architecture 对每一个 sub-layer 都定义：

$$
LayerNorm(
x+Sublayer(x)
)
$$

所以自然：

> 两次 Add & Norm。

---

## 90. 为什么 FFN 后还要 Residual？

如果只有 Attention residual，

FFN 会：

$$
H\rightarrow FFN(H)
$$

完全替换 representation。

加入：

$$
H+FFN(H)
$$

让 FFN 也作为：

> incremental feature update。

因此整个 Layer 的两个计算模块都以 residual form 工作。

---

## 91. 为什么 LayerNorm 参数也是 Learned？

LayerNorm：

$$
y=
\gamma\odot
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
+
\beta
$$

其中：

$$
\gamma,\beta
$$

是 trainable。

所以虽然 normalization 会标准化，

模型仍可学习：

> 每个 feature 最合适的输出尺度和偏移。

这防止 normalization 过度限制 representation。

---

## 92. LayerNorm 有多少参数？

如果：

$$
d_{\text{model}}=512
$$

一个 LayerNorm 有：

$$
512
$$

个 $\gamma$，

加：

$$
512
$$

个 $\beta$。

共：

$$
1024
$$

trainable scalars。

一个 Post-LN Encoder Layer 有两个 LN：

$$
2048
$$

LayerNorm parameters。

和几百万 MHA/FFN weights 相比很小，

但作用非常重要。

---

## 93. FFN 是不是一种 1×1 Convolution？

原始 Transformer 论文明确说：

> 另一种描述方式是两个 kernel size 1 的 convolutions。

为什么？

对 sequence 每个 position 独立做：

$$
d_{\text{model}}
\rightarrow d_{ff}
\rightarrow d_{\text{model}}
$$

等价于：

> channel-wise 1×1 convolution across positions。

这再次说明：

> FFN 不进行跨 position spatial/temporal mixing。

---

## 94. ACT 的 1202×3200 会很大，为什么还可行？

因为 FFN 对 token 独立。

它不形成：

$$
1202\times1202\times3200
$$

这种 tensor。

主要计算近似：

$$
N
\cdot
D
\cdot
D_{ff}
$$

而 Attention 主要：

$$
N^2D
$$

当：

$$
N=1202
$$

两部分都不便宜，

但 computation structure 不同。

---

## 95. Attention vs FFN Complexity

粗略：

#### Self-Attention

$$
O(N^2D)
$$

再加 projections：

$$
O(ND^2)
$$

#### FFN

$$
O(
ND D_{ff}
)
$$

如果：

$$
D_{ff}
$$

很大，

FFN 计算和参数也会非常显著。

所以性能优化时不能只盯：

> Attention matrix。

---

## 96. Encoder 深度会不会改变 receptive field？

Full Self-Attention 一层已经：

> global receptive field。

所以不像 CNN：

> 需要多层才逐渐扩大 spatial receptive field。

Encoder 深度主要不是为了：

> “让 token 终于看到远处”。

而是为了：

> 进行多轮 contextual computation / nonlinear refinement。

这是 Transformer 和局部 CNN 一个重要区别。

---

## 97. 为什么一层能全局，但仍需要深度？

可以用一个计算论直觉：

> 看得到所有信息，不等于一次简单运算就能完成复杂推理。

一个人可以一次拿到整本题目的所有数据，

但仍需要：

> 多步加工。

Transformer depth 提供：

> successive computation stages。

所以：

$$
\boxed{
global access
\neq
single-step sufficient computation
}
$$

---

## 98. Self-Attention 每层都会重新生成 Attention Matrix

第 $l$ 层：

$$
A^{(l)}
=
softmax(
Q^{(l)}K^{(l)\top}
)
$$

下一层 input 已变：

$$
X^{(l+1)}
$$

所以：

$$
A^{(l+1)}
$$

也重新计算。

因此信息路由是：

> layer-dependent + input-dependent。

不是第一层生成一张 attention map 后后面重复用。

---

## 99. Residual Stream 可以怎样理解？

现代 Transformer 教学中常把：

$$
X
$$

看成一条：

> residual stream。

Attention 和 FFN 不断向它写入 updates：

$$
X
\leftarrow
X+\Delta_{\text{attn}}
$$

$$
X
\leftarrow
X+\Delta_{\text{ffn}}
$$

在原始 Post-LN 中，每次 update 后再做 normalization。

这是一个很有用的 mental model，

但“residual stream”是后续常用解释语言，

不是原论文的正式术语。

---

## 100. 为什么 Encoder Output 不能简单理解成“Attention Result”？

因为经过一层后：

$$
Y
$$

已经包含：

- input residual；
- MHA output；
- LayerNorm；
- FFN nonlinear transform；
- 第二次 residual；
- 第二次 LayerNorm。

再堆 4/6 层。

所以最终 Encoder Memory：

> 远远不只是 `softmax(QKᵀ)V`。

这是为什么理解完整 Transformer block 非常重要。

---

## 101. 常见误解一：Transformer Encoder 就是一层 Self-Attention

**错误。**

完整 Encoder Layer 还有：

- Residual；
- LayerNorm；
- FFN；
- Dropout。

---

## 102. 常见误解二：FFN 负责 Token 之间的信息交换

**错误。**

标准 Position-wise FFN 对每个 token 独立。

跨 token interaction主要由 Attention 完成。

---

## 103. 常见误解三：FFN 只是为了把 Shape 改回 512

**错误。**

它是重要的 nonlinear feature transformation module。

---

## 104. 常见误解四：512→2048→512 是两个 Linear，所以仍然等于一个 Linear

只有在：

> 中间没有 nonlinear activation

时才成立。

ReLU 使其不能简单合并。

---

## 105. 常见误解五：所有 6 个 Encoder Layers 共用一套参数

**错误。**

结构相同，

参数通常独立。

---

## 106. 常见误解六：LayerNorm 对整个 Batch 算 Mean/Variance

**错误。**

Transformer 常见 LayerNorm 对单个 token 的 feature dimension normalization。

---

## 107. 常见误解七：LayerNorm 会让所有 Tokens 变得一样

**错误。**

每个 token 独立 normalization，

不会把 token values averaging 到一起。

---

## 108. 常见误解八：LayerNorm 会让 Representation 服从标准正态分布

**错误。**

零均值/单位方差 normalization 不等于 Gaussian distribution。

---

## 109. 常见误解九：Residual 就是把上一层输出和这一层输出平均

**错误。**

是：

$$
x+F(x)
$$

普通加法，

不是：

$$
(x+F(x))/2
$$

---

## 110. 常见误解十：Residual 只是防止信息丢失，和训练无关

不完整。

它也提供直接 identity path，

对深层优化和梯度传播很重要。

---

## 111. 常见误解十一：Transformer 原论文是 Pre-LN

**错误。**

原始 2017：

$$
\boxed{
Post-LN
}
$$

---

## 112. 常见误解十二：所有现代 Transformer 都是 Post-LN

也错误。

很多现代架构改成：

> Pre-LN 或其他 normalization variants。

---

## 113. 常见误解十三：ACT 默认使用 Pre-LN

**当前 released default 不是。**

`pre_norm` 是 `store_true` flag，

默认 false。

所以走：

> `forward_post`。

---

## 114. 常见误解十四：ACT 的 FFN 是原始 Transformer 的 2048

**论文实际配置不是。**

ACT：

$$
\boxed{
d_{ff}=3200
}
$$

---

## 115. 常见误解十五：ACT 只有一个 Transformer Encoder

**错误。**

至少需要区分：

- training-only CVAE Encoder；
- Policy Observation Encoder。

---

## 116. 常见误解十六：两个 ACT Encoders 都是 4 层，所以共享权重

**错误。**

它们是独立 module instances。

---

## 117. 常见误解十七：Policy Encoder 把 1202 Tokens 压成一个 Vector

**错误。**

输出仍是：

$$
1202\times512
$$

memory sequence。

---

## 118. 常见误解十八：Encoder 输出每个 visual token 仍只包含那个 patch

**不准确。**

经过多层 Self-Attention，

它已经是 contextualized representation。

---

## 119. 常见误解十九：Attention 已经有 Softmax，所以 FFN 没必要

**错误。**

Attention 与 FFN提供不同 computation structure。

FFN 是 Transformer block 的核心组成。

---

## 120. 常见误解二十：Attention Is All You Need 意味着网络只有 Attention

**错误。**

它表示 sequence dependency 不再依赖 recurrence/convolution，

不是 literal “only attention operations exist”。

---

## 121. 一层 Encoder 的完整 Shape Flow：原始 Transformer Base

输入：

$$
X:
[B,N,512]
$$

---

### Multi-Head Self-Attention

8 heads：

$$
d_h=64
$$

输出：

$$
[B,N,512]
$$

---

### Residual + LayerNorm

$$
[B,N,512]
$$

---

### FFN Linear 1

$$
[B,N,512]
\rightarrow
[B,N,2048]
$$

---

### ReLU

$$
[B,N,2048]
$$

---

### FFN Linear 2

$$
[B,N,2048]
\rightarrow
[B,N,512]
$$

---

### Residual + LayerNorm

$$
\boxed{
[B,N,512]
}
$$

---

## 122. ACT Policy Encoder 的完整 Shape Flow

输入：

$$
\boxed{
[B,1202,512]
}
$$

注意官方 PyTorch internal convention 常转成：

$$
[1202,B,512]
$$

但语义不变。

---

### MHA

$$
[B,1202,512]
$$

---

### Add & Norm

$$
[B,1202,512]
$$

---

### FFN

$$
[B,1202,512]
\rightarrow
[B,1202,3200]
\rightarrow
[B,1202,512]
$$

---

### Add & Norm

$$
[B,1202,512]
$$

一层完成。

重复：

$$
4
$$

次。

最终：

$$
\boxed{
Memory:
[B,1202,512]
}
$$

---

## 123. ACT CVAE Encoder 的完整 Shape Flow

假设：

$$
k=100
$$

输入：

$$
\boxed{
[B,102,512]
}
$$

每层：

$$
[B,102,512]
\rightarrow
[B,102,512]
$$

4 层后：

$$
[B,102,512]
$$

取 CLS：

$$
[B,512]
$$

latent head：

$$
[B,512]
\rightarrow
[B,64]
$$

split：

$$
\mu:
[B,32]
$$

$$
\log\sigma^2:
[B,32]
$$

---

## 124. 为什么 Policy Encoder 和 CVAE Encoder 可以用同一种 Layer？

因为 Transformer Encoder Layer 只要求：

> 输入是一组同 hidden dimension 的 tokens。

它并不关心：

```text
token 1 是词
token 2 是图
token 3 是 action
```

Token 的语义来自：

- upstream projection；
- position encoding；
- training objective。

所以 Transformer 是一种非常通用的 representation processor。

---

## 125. 一句最重要的分工

如果要把整个 Encoder Layer 压成一句最实用的 mental model：

$$
\boxed{
\text{Self-Attention}
:
\text{让 Tokens 互相交流}
}
$$

$$
\boxed{
\text{FFN}
:
\text{让每个 Token 自己做深度非线性加工}
}
$$

$$
\boxed{
\text{Residual}
:
\text{保留 identity path，并学习增量更新}
}
$$

$$
\boxed{
\text{LayerNorm}
:
\text{规范每个 Token 的 feature scale}
}
$$

四者合起来，

才是 Transformer Encoder Layer。

---

## 126. 一句话真正理解 Transformer Encoder

> **Transformer Encoder 不是一个“把 sequence 压缩成向量”的模块，而是一个保持 token 数量不变的深层 contextualization stack：每一层先通过 Multi-Head Self-Attention 让所有允许的位置根据当前内容动态交换信息，再通过 residual 与 LayerNorm 保留并规范 representation，随后用 position-wise FFN 对每个 token 的 feature vector 进行高维 nonlinear transformation，再经过第二次 residual 与 normalization；多层重复后，每个 token 都变成结合整个输入 context 的 memory representation。**

---

## 127. 一句话理解 ACT Policy Encoder

> **ACT Policy Encoder 把 $1200$ 个多摄像头视觉 token、1 个 joint token 和 1 个 latent token 放进统一的 512-D hidden space，通过 4 层、8-head、$512\rightarrow3200\rightarrow512$ 的 Post-LN Transformer Encoder反复进行全局信息交换和 per-token nonlinear refinement，最终仍输出 $1202\times512$ 的 contextual observation memory，供 action-query Decoder 通过 Cross-Attention读取。**

ACT 论文明确写出 Policy Encoder 输入是：

$$
1202\times512
$$

并将 Transformer Encoder描述为融合不同 camera viewpoints、joint positions 和 style variable 的模块；Table III 给出 4 层 Encoder、512 hidden、3200 feedforward、8 heads、dropout 0.1。

---

## 128. 下一步：Transformer Decoder

现在 Encoder 已经完整。

下一篇自然就是：

> **[Transformer Decoder：为什么比 Encoder 多一个 Cross-Attention？](./transformer-decoder.md)**

会正式把：

```text
Decoder Self-Attention
↓
Residual + LayerNorm
↓
Cross-Attention
↓
Residual + LayerNorm
↓
FFN
↓
Residual + LayerNorm
```

完整组装起来。

重点会讲：

- 原始语言 Transformer Decoder 为什么有 3 个 sub-layers；
- Masked Self-Attention 和 Cross-Attention 分别解决什么；
- 为什么 Decoder Query 先和其他 output positions 交流，再读取 Encoder Memory；
- Post-LN / Pre-LN 的三次 normalization；
- ACT 的 Decoder 为什么 Self-Attention 不 causal；
- ACT 中 $k$ 个 action slots 如何从 zero `tgt` + query embedding 开始；
- 每一层怎样把 `[k,512]` action representations 逐步变成 observation-conditioned outputs；
- ACT Paper 的 7 decoder layers 与 current released-code intermediate-output行为如何区分。

---

### Primary Source：Transformer

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,  
Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin.

**Attention Is All You Need.**  
NeurIPS 2017.

- arXiv: https://arxiv.org/abs/1706.03762
- HTML: https://arxiv.org/html/1706.03762

本文主要依据：

#### Section 3.1 — Encoder

原论文规定：

- Encoder stack：
  $$
  N=6
  $$
  个 identical-architecture layers；
- 每层两个 sub-layers：
  1. Multi-Head Self-Attention；
  2. Position-wise FFN；
- 每个 sub-layer 使用 residual connection；
- residual 后做 LayerNorm：
  $$
  \boxed{
  LayerNorm(x+Sublayer(x))
  }
  $$
- 所有 sub-layers 维度：
  $$
  d_{\text{model}}=512
  $$

---

#### Section 3.3 — Position-wise FFN

原论文：

$$
\boxed{
FFN(x)
=
\max(
0,
xW_1+b_1
)W_2+b_2
}
$$

Base Model：

$$
\boxed{
d_{\text{model}}=512
}
$$

$$
\boxed{
d_{ff}=2048
}
$$

同一 layer 内：

> FFN 在所有 positions 使用相同参数；

不同 Encoder Layers：

> 使用不同参数。

---

#### Section 5.4 — Residual Dropout

原论文：

> 对每个 sub-layer output 应用 dropout，再与 sub-layer input 做 residual addition 并 normalization。

Base dropout：

$$
0.1
$$

---

### Residual Connection Background

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

**Deep Residual Learning for Image Recognition.**  
CVPR 2016.

- arXiv: https://arxiv.org/abs/1512.03385

Residual framework 的核心形式：

$$
\boxed{
y=x+F(x)
}
$$

其目的之一是：

> 让深层网络更容易优化，并允许 block 学习相对于 identity mapping 的 residual update。

Transformer 将 residual connection 用在：

- Attention；
- FFN；

sub-layers 外。

---

### Layer Normalization Background

Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton.

**Layer Normalization.**  
2016.

- arXiv: https://arxiv.org/abs/1607.06450

LayerNorm 使用单个 training case 内 layer activations 的统计量，

并带有 learned gain / bias。

在 Transformer hidden vector 上可以写：

$$
\mu
=
\frac1d
\sum_jx_j
$$

$$
\sigma^2
=
\frac1d
\sum_j
(x_j-\mu)^2
$$

$$
\boxed{
LN(x)
=
\gamma
\odot
\frac{
x-\mu
}{
\sqrt{
\sigma^2+\epsilon
}
}
+
\beta
}
$$

---

### ACT Primary Source

Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.

**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.**  
RSS 2023.

- arXiv: https://arxiv.org/abs/2304.13705
- HTML: https://arxiv.org/html/2304.13705

ACT Section IV-C：

- CVAE Encoder 使用 BERT-like Transformer Encoder；
- Policy 使用 ResNet image encoders + Transformer Encoder + Transformer Decoder；
- Transformer Encoder 融合：
  - camera viewpoints；
  - joint positions；
  - latent $z$；
- Policy Encoder 输入：
  $$
  1202\times512
  $$

ACT Table III：

$$
\boxed{
\#encoder\ layers=4
}
$$

$$
\boxed{
hidden\ dimension=512
}
$$

$$
\boxed{
feedforward\ dimension=3200
}
$$

$$
\boxed{
\#heads=8
}
$$

$$
\boxed{
dropout=0.1
}
$$

---

### Official ACT Implementation

Repository:

https://github.com/tonyzhaozh/act

#### `detr/models/transformer.py`

当前 `TransformerEncoderLayer`：

```python
self.self_attn =
    nn.MultiheadAttention(
        d_model,
        nhead,
        dropout=dropout
    )

self.linear1 =
    nn.Linear(
        d_model,
        dim_feedforward
    )

self.linear2 =
    nn.Linear(
        dim_feedforward,
        d_model
    )

self.norm1 =
    nn.LayerNorm(d_model)

self.norm2 =
    nn.LayerNorm(d_model)
```

默认：

```python
normalize_before=False
```

因此走：

```python
forward_post(...)
```

核心顺序：

```text
Self-Attention
→ Residual
→ LayerNorm
→ FFN
→ Residual
→ LayerNorm
```

---

#### Position Injection

官方 Post-LN Encoder：

```python
q = k =
    self.with_pos_embed(
        src,
        pos
    )

src2 =
    self.self_attn(
        q,
        k,
        value=src,
        ...
    )[0]
```

即概念上：

$$
Q,K\leftarrow src+pos
$$

$$
V\leftarrow src
$$

---

#### Encoder Stack

```python
for layer in self.layers:
    output = layer(
        output,
        ...
    )
```

因此每一层的 output：

> 成为下一层 input。

层由：

```python
_get_clones(...)
```

创建为独立 module copies；

它们不是运行时共享同一个 parameter object。

---

#### ACT Training Configuration

`imitate_episodes.py` 当前固定：

```python
enc_layers = 4
dec_layers = 7
nheads = 8
```

并把用户参数：

```python
dim_feedforward
hidden_dim
```

传给 ACT model。

官方 README 示例：

```text
--hidden_dim 512
--dim_feedforward 3200
```

与论文 Table III 一致。

---

### ACT 的两个 Encoder

#### CVAE Encoder

`DETRVAE.self.encoder`

输入：

$$
[CLS]+qpos+actions
$$

Training-only。

---

#### Policy Observation Encoder

`DETRVAE.self.transformer.encoder`

输入：

$$
latent+proprio+visual
$$

Training + Inference 都使用。

两者：

> architecture/config 可以相同，

但：

$$
\boxed{
\text{参数不共享}
}
$$

---

### 本文知识连接

#### 前置

- [Transformer](./transformer.md)
- [Self-Attention](./self-attention.md)
- [Multi-Head Attention](./multi-head-attention.md)
- [Positional Encoding](./positional-encoding.md)

#### Encoder Components

- [Residual Connection](./residual-connection.md)
- [Layer Normalization](./layer-normalization.md)
- [Feed-Forward Network](./feed-forward-network.md)
- [ReLU](./relu.md)
- [Dropout](./dropout.md)

#### 后续

- [Transformer Decoder](./transformer-decoder.md)
- [Cross-Attention](./cross-attention.md)
- Pre-LN vs Post-LN

#### Robot Learning

- [ACT Architecture](../robot-learning/act/architecture.md)
- [CVAE in ACT](../robot-learning/act/cvae-in-act.md)
- [ACT Training](../robot-learning/act/training.md)
- [ACT Complete Data Flow](../robot-learning/act/complete-data-flow.md)

#### 下一步

- [Transformer Decoder](./transformer-decoder.md)
