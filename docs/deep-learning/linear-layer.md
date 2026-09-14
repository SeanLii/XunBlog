---
title: "Linear Layer：神经网络里的 nn.Linear 到底学了什么？"
description: "从单个加权和 y=w^T x+b 出发，严格理解矩阵形式、Linear 与 Affine 的区别、weight/bias 的几何意义、shape 与反向传播，并统一解释 Transformer 的 Q/K/V、W_O、FFN 与 ACT 的 joint/action/latent projections。"
status: reviewed
pageType: concept
canonical: /deep-learning/linear-layer
updated: "2026-09-15"
---

# Linear Layer：神经网络里的 `nn.Linear` 到底学了什么？

如果你把 ACT、Transformer、CVAE 的代码打开，会发现一个模块反复出现：

```python
nn.Linear(...)
```

例如 ACT 中：

```python
self.encoder_action_proj = nn.Linear(14, 512)
self.encoder_joint_proj  = nn.Linear(14, 512)

self.latent_proj = nn.Linear(512, 64)
self.latent_out_proj = nn.Linear(32, 512)

self.action_head = nn.Linear(512, 14)
```

Transformer 的 FFN：

```python
nn.Linear(512, 3200)
nn.Linear(3200, 512)
```

Multi-Head Attention 背后还有：

\[
W_Q,\quad W_K,\quad W_V,\quad W_O
\]

本质上也都是 learned linear / affine projections。

于是你会发现：

> **一个看起来极其简单的 `Linear Layer`，几乎贯穿整个神经网络。**

但它到底在“学习”什么？

当我们写：

\[
y=Wx+b
\]

时，\(W\) 到底是什么意思？

一个：

```python
nn.Linear(512, 64)
```

为什么能把一个 512-D vector 变成：

- Query；
- Key；
- Value；
- \(\mu\)；
- \(\log\sigma^2\)；
- latent embedding；
- action prediction；

这些完全不同的东西？

是不是 `Linear` 内部知道：

> “现在我要算 \(\mu\)”？

当然不是。

这一篇要把 `nn.Linear` 从最底层讲透。

---

# 1. 先从一个最简单的 Neuron 开始

假设输入只有三个 features：

\[
x=
[x_1,x_2,x_3]
\]

我们想输出一个 scalar：

\[
y
\]

最基本的神经元计算：

\[
\boxed{
y=
w_1x_1+w_2x_2+w_3x_3+b
}
\]

也可以写：

\[
\boxed{
y=w^\top x+b
}
\]

其中：

\[
w=
[w_1,w_2,w_3]
\]

---

# 2. Weight 的最直接意义

每个：

\[
w_i
\]

控制输入 feature：

\[
x_i
\]

对输出：

\[
y
\]

产生多大影响。

例如：

\[
y=
2x_1-3x_2+0.5x_3+1
\]

那么：

- \(x_1\) 增加 1，其他不变：
  \[
  y
  \]
  增加 2；
- \(x_2\) 增加 1：
  \[
  y
  \]
  减少 3；
- \(x_3\) 增加 1：
  \[
  y
  \]
  增加 0.5。

所以：

\[
\boxed{
w_i
=
\text{output 对 feature }x_i\text{ 的线性敏感度}
}
\]

---

# 3. Bias 又是什么？

\[
b
\]

是一个与输入无关的 learned offset。

如果：

\[
x=0
\]

那么：

\[
y=b
\]

所以 bias 让模型不必满足：

\[
x=0
\Rightarrow
y=0
\]

这一限制。

---

# 4. 一个数值例子

设：

\[
x=
[2,-1,3]
\]

\[
w=
[0.5,2,-1]
\]

\[
b=4
\]

则：

\[
y
=
0.5(2)
+
2(-1)
-
1(3)
+
4
\]

\[
=
1-2-3+4
\]

\[
\boxed{
y=0
}
\]

Linear Layer 的最底层并没有更神秘。

就是：

> learned weighted sum + bias。

---

# 5. 如果我们想输出不止一个数呢？

假设 input：

\[
x\in\mathbb R^3
\]

但我们想输出两个 features：

\[
y_1,\ y_2
\]

那么可以建立两个神经元：

\[
y_1
=
w_{11}x_1+
w_{12}x_2+
w_{13}x_3+
b_1
\]

\[
y_2
=
w_{21}x_1+
w_{22}x_2+
w_{23}x_3+
b_2
\]

---

# 6. 两个输出就是两套 Weight Vectors

定义：

\[
w_1=
[w_{11},w_{12},w_{13}]
\]

\[
w_2=
[w_{21},w_{22},w_{23}]
\]

那么：

\[
y_1=w_1^\top x+b_1
\]

\[
y_2=w_2^\top x+b_2
\]

所以每一个 output dimension：

> 都拥有自己的一套 learned weighted combination。

---

# 7. Matrix 只是一次把很多 Neurons 写完

把 weights堆成：

\[
W=
\begin{bmatrix}
w_1^\top\\
w_2^\top
\end{bmatrix}
\]

那么：

\[
W
\in
\mathbb R^{2\times3}
\]

bias：

\[
b=
\begin{bmatrix}
b_1\\
b_2
\end{bmatrix}
\]

于是：

\[
\boxed{
y=Wx+b
}
\]

其中：

\[
x\in\mathbb R^3
\]

\[
y\in\mathbb R^2
\]

---

# 8. Matrix Multiplication 的每一行是什么？

对于：

\[
W
\in
\mathbb R^{m\times n}
\]

输入：

\[
x\in\mathbb R^n
\]

输出：

\[
y\in\mathbb R^m
\]

第 \(j\) 个 output：

\[
\boxed{
y_j
=
w_j^\top x+b_j
}
\]

其中：

\[
w_j^\top
\]

是：

> \(W\) 的第 \(j\) 行。

所以：

\[
\boxed{
W\text{ 的每一行对应一个 output unit 的 weight vector}
}
\]

在 column-vector convention 下。

---

# 9. 一个 `Linear(3,2)` 的完整矩阵例子

设：

\[
W=
\begin{bmatrix}
1&2&0\\
-1&0&3
\end{bmatrix}
\]

\[
b=
\begin{bmatrix}
1\\
-2
\end{bmatrix}
\]

输入：

\[
x=
\begin{bmatrix}
2\\
1\\
-1
\end{bmatrix}
\]

则：

\[
Wx
=
\begin{bmatrix}
1(2)+2(1)+0(-1)\\
-1(2)+0(1)+3(-1)
\end{bmatrix}
\]

\[
=
\begin{bmatrix}
4\\
-5
\end{bmatrix}
\]

加 bias：

\[
y=
\begin{bmatrix}
5\\
-7
\end{bmatrix}
\]

---

# 10. `nn.Linear(3,2)` 本质是什么？

就是：

> 两个接受 3-D input 的 affine neurons。

参数：

\[
W:
[2,3]
\]

\[
b:
[2]
\]

总参数：

\[
2\times3+2
=
\boxed{
8
}
\]

---

# 11. PyTorch 官方定义

当前 PyTorch 文档定义：

```python
nn.Linear(
    in_features,
    out_features,
    bias=True
)
```

应用：

\[
\boxed{
y=xA^\top+b
}
\]

PyTorch 存储：

\[
\boxed{
weight:
[out\_features,in\_features]
}
\]

bias：

\[
\boxed{
[out\_features]
}
\]

---

# 12. 为什么 PyTorch 写 \(xA^\top+b\)，我们前面写 \(Wx+b\)？

只是 vector / tensor convention不同。

数学教材常把单个 vector写成 column：

\[
x
\in
\mathbb R^{n\times1}
\]

于是：

\[
y=Wx+b
\]

其中：

\[
W:
[m,n]
\]

---

PyTorch tensor通常把 features放最后一维：

\[
x:
[\ldots,n]
\]

可以把每个 sample想成 row vector，

所以计算写：

\[
\boxed{
y=xW^\top+b
}
\]

其中 PyTorch stored weight仍：

\[
W:
[m,n]
\]

---

# 13. 两种写法完全等价

Column-vector notation：

\[
\boxed{
y=Wx+b
}
\]

PyTorch row-vector notation：

\[
\boxed{
y=xW^\top+b
}
\]

只要 shape convention保持一致，

表达的是同一组 scalar equations：

\[
y_j
=
\sum_i
W_{ji}x_i+b_j
\]

---

# 14. 这是读代码时最容易 Shape Confusion 的地方

如果你在论文里看到：

\[
W:
[512,64]
\]

可能作者使用：

\[
xW
\]

row-vector notation。

但 PyTorch：

```python
nn.Linear(512, 64)
```

内部 stored：

```text
weight.shape = [64, 512]
```

所以不要看到：

\[
[64,512]
\]

就以为 projection方向写反了。

---

# 15. 最安全的方法：永远先看 Input / Output Dimensions

对于：

```python
nn.Linear(512, 64)
```

无论矩阵怎样排版，

你只需要先牢牢记住：

\[
\boxed{
512
\rightarrow
64
}
\]

也就是：

> 每个 512-D input vector变成一个 64-D output vector。

---

# 16. PyTorch 会作用于哪个 Axis？

官方文档：

输入：

\[
(*,H_{in})
\]

输出：

\[
(*,H_{out})
\]

其中：

\[
*
\]

可以是任意数量的前置 dimensions。

也就是说：

> `nn.Linear` 只变最后一个 dimension。

---

# 17. 例如

```python
x.shape =
[B, N, 512]
```

经过：

```python
nn.Linear(512, 64)
```

得到：

\[
\boxed{
[B,N,64]
}
\]

Batch：

\[
B
\]

不变。

Token count：

\[
N
\]

不变。

只改变：

\[
512\rightarrow64
\]

feature axis。

---

# 18. 所以 Linear Layer 不会自动 Mixing Tokens

如果：

\[
X:
[B,N,512]
\]

`Linear(512,64)` 对每一个：

\[
X[b,n,:]
\]

独立使用同一套 weights。

所以：

\[
\boxed{
\text{Linear over last dimension}
\neq
\text{token mixing}
}
\]

这和我们前面讲 FFN position-wise computation是一致的。

---

# 19. Token 之间什么时候交流？

Transformer中主要是：

> Attention。

Linear projections例如：

\[
W_Q,W_K,W_V
\]

只是对每个 token自身 feature vector做变换。

真正 token-to-token interaction直到：

\[
QK^\top
\]

才出现。

---

# 20. 一个重要的数学问题：为什么叫 Linear Layer？

如果：

\[
y=Wx
\]

这在严格线性代数定义中确实是：

> linear transformation。

因为它满足：

\[
T(x_1+x_2)
=
T(x_1)+T(x_2)
\]

以及：

\[
T(cx)
=
cT(x)
\]

---

# 21. 但加了 Bias 以后呢？

如果：

\[
T(x)=Wx+b
\]

且：

\[
b\neq0
\]

那么：

\[
T(0)=b
\]

不再是：

\[
0
\]

而任何严格 linear map必须：

\[
T(0)=0
\]

所以：

\[
\boxed{
Wx+b
\text{ 严格数学上不是 linear map}
}
\]

---

# 22. 它真正叫什么？

严格地说：

\[
\boxed{
T(x)=Wx+b
}
\]

是：

> **Affine Transformation**

也就是：

> linear transformation + translation。

---

# 23. PyTorch 官方自己也承认这一点

PyTorch 对：

```python
nn.Linear
```

的描述就是：

> applies an **affine linear transformation** to incoming data。

公式：

\[
y=xA^\top+b
\]

所以名称叫 `Linear`，

但默认带 bias 时：

> 严格数学上是 affine。

---

# 24. 为什么深度学习还一直叫 Linear Layer？

主要是工程/历史命名。

大家把：

\[
Wx+b
\]

这一类 fully-connected affine operation习惯称为：

- Linear Layer；
- Dense Layer；
- Fully Connected Layer。

所以看到：

> Linear

不要因此忘记：

\[
b
\]

会破坏严格线性。

---

# 25. 如果 `bias=False` 呢？

```python
nn.Linear(
    in_features,
    out_features,
    bias=False
)
```

那么：

\[
y=Wx
\]

这才是严格 linear transformation。

---

# 26. Bias 的几何意义

没有 bias：

\[
y=w^\top x
\]

如果：

\[
y=0
\]

决策/等值 hyperplane：

\[
w^\top x=0
\]

必须经过 origin。

---

有 bias：

\[
y=w^\top x+b
\]

零点：

\[
w^\top x+b=0
\]

hyperplane可以：

> 离开原点平移。

所以 bias增加：

\[
\boxed{
\text{translation / threshold flexibility}
}
\]

---

# 27. 一个 2-D 例子

无 bias：

\[
y=x_1+x_2
\]

\(y=0\)：

\[
x_1+x_2=0
\]

经过 origin。

有 bias：

\[
y=x_1+x_2-3
\]

\(y=0\)：

\[
x_1+x_2=3
\]

整条直线平移。

---

# 28. 一个 Output Neuron 可以看成 Learned Direction

\[
y_j=w_j^\top x+b_j
\]

其中：

\[
w_j
\]

定义 input space中的一个 direction / normal vector。

dot product：

\[
w_j^\top x
\]

衡量 input沿这组 learned coefficients的 signed response。

所以一个 Linear Layer可以理解为：

> 同时用很多 learned directions去“读取”当前 representation。

---

# 29. 但不要直接叫“投影长度”

严格正交投影到 unit vector \(u\) 的 scalar component是：

\[
u^\top x
\]

前提：

\[
\|u\|=1
\]

而神经网络 weight：

\[
w_j
\]

通常：

> 不要求 unit norm。

所以：

\[
w_j^\top x
\]

更准确叫：

> learned linear response / weighted feature combination。

不要自动把每个 neuron叫正交 projection。

---

# 30. Linear Layer 可以做什么几何变换？

矩阵：

\[
W
\]

可以实现或组合：

- rotation；
- scaling；
- reflection；
- shear；
- projection；
- dimension reduction；
- dimension expansion into new coordinates；

具体取决于：

\[
W
\]

shape和结构。

再加：

\[
b
\]

可做 translation。

---

# 31. 但“Dimension Expansion”需要一个重要 Caveat

例如：

\[
x\in\mathbb R^2
\]

通过：

\[
W:
[100,2]
\]

得到：

\[
y\in\mathbb R^{100}
\]

我们可以说：

> representation被写成100个 learned output coordinates。

但它没有凭空创造：

\[
100
\]

个独立信息自由度。

---

# 32. 为什么？

因为：

\[
y=Wx
\]

所有 outputs仍由原始：

\[
2
\]

个 input degrees of freedom决定。

矩阵 rank最多：

\[
\boxed{
rank(W)\le2
}
\]

所以所有输出 \(y\) 只能位于：

> \(\mathbb R^{100}\) 中至多 2-D 的 linear subspace

里，

忽略 bias时。

---

# 33. 加 Bias 后呢？

\[
y=Wx+b
\]

输出位于：

> 一个至多 2-D affine subspace。

所以：

\[
2\rightarrow100
\]

的单个 Linear Layer：

> 增加 coordinate数量，

但不增加独立信息维数。

---

# 34. 那 FFN 为什么要 512 → 3200？

非常好的问题。

单个：

\[
512\rightarrow3200
\]

Linear本身仍然只是 affine map，

rank最多：

\[
512
\]

它的价值之一是：

> 提供大量不同 learned pre-activation combinations。

然后经过：

\[
ReLU
\]

这种非线性，

不同 inputs会激活不同 subsets。

这时整个：

\[
Linear
\rightarrow
ReLU
\rightarrow
Linear
\]

就不能再压成一个 affine map。

---

# 35. 所以“扩维”与“非线性”要一起理解

FFN：

\[
512
\rightarrow
3200
\rightarrow
ReLU
\rightarrow
512
\]

更准确地：

\[
512
\xrightarrow{Linear}
3200
\xrightarrow{ReLU}
3200
\xrightarrow{Linear}
512
\]

第一 Linear创建：

> 3200个 learned candidate responses。

ReLU做：

> input-dependent gating。

这才让宽 hidden layer获得真正更强的 nonlinear function capacity。

---

# 36. 为什么两层 Linear 没 Activation 可以合并？

假设：

\[
h=W_1x+b_1
\]

\[
y=W_2h+b_2
\]

代入：

\[
y
=
W_2(W_1x+b_1)+b_2
\]

\[
=
W_2W_1x+
W_2b_1+b_2
\]

定义：

\[
W'=W_2W_1
\]

\[
b'=W_2b_1+b_2
\]

则：

\[
\boxed{
y=W'x+b'
}
\]

仍然只是一个 affine layer。

---

# 37. 所以 Deep Network 为什么需要 Activation？

因为如果整个网络只有：

\[
Linear
\rightarrow
Linear
\rightarrow
Linear
\]

无论多少层：

> 最终都可以合并成一个 affine transformation。

真正让深层组合无法全部折叠的是：

- ReLU；
- GELU；
- Softmax；
- normalization等非线性 operations。

---

# 38. Linear Layer 本身并不会“理解语义”

这是整篇最重要的思想之一。

```python
nn.Linear(512, 32)
```

只知道：

> 输入512个 numbers，
> 输出32个 learned weighted combinations。

它并不知道：

- 这32维是 latent；
- 这是动作；
- 这是概率；
- 这是 query；
- 这是 mean。

---

# 39. 那 Output Semantics 从哪里来？

来自：

\[
\boxed{
\text{它在 computation graph 中被怎样使用}
}
\]

以及：

\[
\boxed{
\text{最终 loss 怎样给它 gradient}
}
\]

也就是：

> **role is created by usage + training objective。**

---

# 40. 一个最简单的例子

有：

```python
head = nn.Linear(512, 10)
```

如果这10个 outputs随后进入：

```python
cross_entropy(logits, class_label)
```

它们会被训练成：

> 10个 class logits。

---

# 41. 同一个 `Linear(512,10)` 放到别处

如果输出被拿去和：

\[
10
\]

维 robot action做：

\[
MSE
\]

那么它会被训练成：

> action predictor。

Architecture完全一样。

语义完全不同。

---

# 42. 所以 Linear Layer 的名字不会决定它学什么

```text
Linear(512,10)
```

本身没有：

> “classification mode”

或：

> “regression mode”。

是：

- downstream operation；
- target；
- loss；
- training data；

共同赋予输出语义。

---

# 43. 这直接解释 Q / K / V

Transformer中：

\[
q=xW_Q
\]

\[
k=xW_K
\]

\[
v=xW_V
\]

三个 operation从数学层面都是：

> Linear projections。

---

# 44. 为什么 \(W_Q\) 学成“Query”？

不是因为矩阵里写了：

> Q。

而是因为它的 output被放在：

\[
QK^\top
\]

左边，

承担：

> 当前 position发起匹配需求

这一 computation role。

---

# 45. 为什么 \(W_K\) 学成“Key”？

因为它的 output出现在：

\[
QK^\top
\]

另一侧，

用于：

> 被 Query比较。

Gradient会根据最终 task loss不断调整：

\[
W_K
\]

使这种 matching有用。

---

# 46. 为什么 \(W_V\) 学成“Value”？

因为它不直接决定标准 attention score：

\[
QK^\top
\]

而是在：

\[
A V
\]

中承担：

> 被 routing weights读取的 message/content。

所以 role来自：

\[
\boxed{
\text{computation topology}
}
\]

而不是 Linear本身。

---

# 47. 这就是我们在 QKV 文章里说的核心

\[
\boxed{
\text{Role}
\leftarrow
\text{position in computation graph}
+
\text{task gradients}
}
\]

Linear Layer只是：

> 可学习的 feature transform。

---

# 48. 同样逻辑解释 ACT 的 μ 和 logvar

ACT CVAE Encoder得到：

\[
h_{CLS}
\in
\mathbb R^{512}
\]

官方代码：

```python
self.latent_proj =
    nn.Linear(
        hidden_dim,
        latent_dim * 2
    )
```

其中：

\[
hidden\_dim=512
\]

\[
latent\_dim=32
\]

所以：

\[
\boxed{
512\rightarrow64
}
\]

---

# 49. 这个 Linear Output 一开始只是 64 个 Numbers

记作：

\[
z_{raw}
\in
\mathbb R^{64}
\]

`nn.Linear` 本身不知道：

> 哪些是 mean，
> 哪些是 log variance。

---

# 50. 是后续代码赋予语义

官方 forward把：

\[
64
\]

维 output拆成：

\[
32+32
\]

例如概念上：

\[
[\mu,\log\sigma^2]
\]

然后第一半被当作：

\[
\mu
\]

第二半被当作：

\[
\log\sigma^2
\]

进入：

- reparameterization；
- KL divergence；
- reconstruction path。

---

# 51. 于是 Gradient 开始塑造这两半

如果前32维改变，

它们通过：

\[
z=\mu+\sigma\epsilon
\]

和 KL中的 mean项影响 loss。

后32维则通过：

\[
\sigma=\exp(\tfrac12\log\sigma^2)
\]

和 KL中的 variance项影响 loss。

因此 optimizer逐渐把：

> 前32维训练成有用的 posterior means，

> 后32维训练成有用的 posterior log-variances。

---

# 52. 所以不要说 Linear Layer “学习一组参数让 μ 接近0、variance接近1”

更准确：

> `latent_proj` 学习一个 512→64 affine mapping。

它的两半因为后续被解释为：

\[
\mu,\log\sigma^2
\]

并同时受到：

- reconstruction；
- KL；

梯度，

才逐渐承担 posterior parameterization。

KL会鼓励 posterior靠近：

\[
\mathcal N(0,I)
\]

但并不要求：

> 每个 sample的 \(\mu=0\)、variance=1。

---

# 53. 同样解释 ACT Action Head

官方：

```python
self.action_head =
    nn.Linear(
        hidden_dim,
        state_dim
    )
```

其中：

\[
512\rightarrow14
\]

Linear本身只产生：

> 14个 numbers。

---

# 54. 为什么这14维变成 Robot Joint Targets？

因为：

1. ground-truth action也是14维；
2. reconstruction loss比较：
   \[
   \hat a
   \]
   和：
   \[
   a
   \]
3. inference时这14维被 de-normalize；
4. 最后送给 robot/environment作为 target action。

因此：

\[
\boxed{
\text{usage + supervision}
}
\]

赋予这14维：

> joint target语义。

---

# 55. Action Head 并不知道“左肩关节”是什么

例如 output第3维最终可能对应：

> 某个关节。

不是因为：

\[
W_{3,:}
\]

天然具有“肩关节”标签。

而是因为数据 layout规定：

> 第3维 target就是那个 joint。

Loss逐维训练它。

---

# 56. Linear Layer 的 Output Ordering 也来自 Data Convention

如果你把 training target的 joint order换掉，

模型会重新学：

> 新的 output ordering。

Architecture：

```python
Linear(512,14)
```

完全不需要改变。

所以：

\[
\boxed{
\text{output index semantics are external conventions learned through targets}
}
\]

---

# 57. ACT Joint Projection

官方：

```python
self.encoder_joint_proj =
    nn.Linear(
        14,
        hidden_dim
    )
```

所以：

\[
\boxed{
14\rightarrow512
}
\]

输入是：

> current qpos。

---

# 58. 这是不是把14个 Joint 变成512个“虚拟关节”？

不是。

512维是：

> Transformer hidden features。

每一个 output feature：

\[
h_j
=
w_j^\top q+b_j
\]

可以是14个 joints的不同 learned combination。

它们不再一一对应：

> 真实物理关节。

---

# 59. 一个 Hidden Feature 可以同时依赖所有14个 Joints

\[
h_j
=
\sum_{i=1}^{14}
W_{ji}q_i+b_j
\]

所以某个 512-D feature可能综合：

- 左臂；
- 右臂；
- gripper；

多个 joint values。

---

# 60. 为什么要投影到512？

因为 Transformer需要统一：

\[
d_{\text{model}}=512
\]

hidden space。

visual tokens：

\[
512
\]

joint token：

\[
512
\]

latent token：

\[
512
\]

这样才能：

- Attention；
- residual；
- FFN；

在统一维度中工作。

---

# 61. ACT Action Projection 同理

CVAE Encoder：

```python
self.encoder_action_proj =
    nn.Linear(
        14,
        hidden_dim
    )
```

每个 ground-truth action：

\[
a_t\in\mathbb R^{14}
\]

变成：

\[
\boxed{
512\text{-D action token}
}
\]

供 Transformer Encoder处理。

---

# 62. 为什么 Joint 和 Action 都是14维，却需要两个不同 Linear Layers？

因为它们虽然 physical dimension相同：

\[
14
\]

但 role不同。

官方：

```python
encoder_joint_proj
encoder_action_proj
```

是两个不同 modules，

所以：

\[
W_{joint}\neq W_{action}
\]

一般成立。

---

# 63. 为什么不共享？

qpos token表示：

> 当前 robot state。

action token表示：

> target demonstration action at a future position。

即使坐标系统相似，

模型可以需要不同 representation geometry。

独立 projections提供这种自由度。

---

# 64. ACT Latent Output Projection

推理时：

\[
z\in\mathbb R^{32}
\]

官方：

```python
self.latent_out_proj =
    nn.Linear(
        32,
        512
    )
```

所以：

\[
\boxed{
32\rightarrow512
}
\]

变成 latent token。

---

# 65. 当 z=0 时为什么 Linear Output 不一定是0？

这是我们之前专门讲过的重要细节。

Linear：

\[
y=Wz+b
\]

如果：

\[
z=0
\]

那么：

\[
\boxed{
y=b
}
\]

只要 bias存在，

输出就不是 zero vector。

---

# 66. PyTorch `nn.Linear` 默认 bias=True

官方 API：

```python
nn.Linear(
    in_features,
    out_features,
    bias=True
)
```

因此 ACT：

```python
nn.Linear(32, 512)
```

如果没有显式：

```python
bias=False
```

就有：

\[
b\in\mathbb R^{512}
\]

---

# 67. 所以 ACT Inference 的 z=0

进入：

\[
latent\_out\_proj(z)
\]

得到：

\[
W(0)+b
\]

即：

\[
\boxed{
b
}
\]

所以：

> `z=0` 不等于 latent token是全零。

---

# 68. 这个 Bias 是什么含义？

可以把它理解为：

> 当 latent coordinate为0时，Linear层学习的 baseline 512-D representation。

但不要过度语义化。

它只是：

> learned affine offset。

最终它是否表达“canonical style baseline”之类的意义：

> 需要结合整个训练来理解。

---

# 69. 为什么 Bias 很容易被忽略？

因为很多公式简写：

\[
zW
\]

而代码：

```python
nn.Linear(...)
```

默认还有：

\[
+b
\]

这会造成很多错误推断。

例如：

> “输入0，所以输出肯定0。”

只在：

\[
bias=False
\]

或：

\[
b=0
\]

时成立。

---

# 70. Transformer Q/K/V 有 Bias 吗？

这取决于具体 implementation。

概念论文公式常简写：

\[
Q=XW_Q
\]

省略 bias。

而具体 PyTorch `MultiheadAttention` 默认：

> projection通常具有 bias，除非配置关闭。

所以 canonical理论和代码实现要区分。

---

# 71. 为什么论文经常省略 Bias？

因为 Bias通常不改变主要 architecture逻辑。

写：

\[
XW_Q+b_Q
\]

每次会让公式更繁琐。

所以很多论文只突出：

\[
W_Q
\]

projection。

但读实现时：

> Bias可能真实存在。

---

# 72. Linear Layer 到底有多少参数？

```python
nn.Linear(
    n,
    m
)
```

weight：

\[
m\times n
\]

bias：

\[
m
\]

所以：

\[
\boxed{
mn+m
}
\]

如果 bias=True。

---

# 73. `Linear(512,64)`

weight：

\[
64\times512
=
32768
\]

bias：

\[
64
\]

总：

\[
\boxed{
32832
}
\]

---

# 74. `Linear(512,3200)`

weight：

\[
3200\times512
=
1,638,400
\]

bias：

\[
3200
\]

总：

\[
\boxed{
1,641,600
}
\]

---

# 75. `Linear(3200,512)`

weight：

\[
512\times3200
=
1,638,400
\]

bias：

\[
512
\]

总：

\[
\boxed{
1,638,912
}
\]

---

# 76. ACT 一个 FFN 两层总参数

\[
1,641,600
+
1,638,912
\]

\[
=
\boxed{
3,280,512
}
\]

这正好与我们 FFN 文章算过的一致。

---

# 77. `Linear(512,14)` Action Head

weight：

\[
14\times512
=
7168
\]

bias：

\[
14
\]

总：

\[
\boxed{
7182
}
\]

相比大型 FFN：

> Action Head非常小。

---

# 78. 这说明 Representation Learning 在前，Task Head 可以很简单

Decoder已经把每个 action slot变成：

\[
512
\]

维高度 processed representation。

最终只需要：

\[
512\rightarrow14
\]

affine map，

就能输出 action。

所以深度学习常见结构：

\[
\boxed{
\text{complex representation network}
+
\text{simple linear task head}
}
\]

---

# 79. Linear Classifier 为什么常只要一层？

如果 upstream representation已经把 classes分得很好，

最终：

\[
W x+b
\]

就可以用 hyperplanes完成分类。

这也是为什么：

> representation quality

非常重要。

---

# 80. Linear Layer 的几何决策边界

二分类 logit：

\[
s=w^\top x+b
\]

decision：

\[
s>0
\]

vs：

\[
s<0
\]

boundary：

\[
\boxed{
w^\top x+b=0
}
\]

这是一个 hyperplane。

---

# 81. 多分类 Linear Head

\[
z=Wx+b
\]

每个 class：

\[
z_c=w_c^\top x+b_c
\]

选择：

\[
\operatorname{argmax}_c z_c
\]

不同 class logits之间的 equality：

\[
z_a=z_b
\]

也形成 hyperplane：

\[
(w_a-w_b)^\top x+(b_a-b_b)=0
\]

所以 linear classifier在 representation space中形成：

> piecewise hyperplane boundaries。

---

# 82. Linear Layer 自己无法表示 XOR 类非线性边界

如果原始 input geometry不是 linearly separable，

单个 affine layer无法：

> 弯曲 decision boundary。

这就是为什么需要：

- hidden layers；
- nonlinear activations；
- deep feature learning。

---

# 83. 但在 Deep Network 最后一层 Linear 仍然足够

因为前面的 nonlinear network：

\[
h=f_\theta(x)
\]

可以把 raw data变成：

> 更容易线性读取的 representation。

然后：

\[
y=Wh+b
\]

只需做最终 readout。

---

# 84. Linear Layer 也可以理解成 Change of Coordinates 吗？

在某些情况下可以。

方阵：

\[
W:
[n,n]
\]

如果 invertible，

它把 vector表示变到另一套坐标中。

但神经网络 Linear并不要求：

- 方阵；
- invertible；
- orthogonal。

所以更一般地应说：

> learned linear/affine transformation between feature spaces。

---

# 85. Dimension Reduction

例如：

\[
512\rightarrow64
\]

如果：

\[
W:
[64,512]
\]

rank最多：

\[
64
\]

因此它将512-D representation压到一个最多64-D的线性子空间 representation。

这就是一种 learned compression/projection。

---

# 86. Q/K Head Projection 正是这种 Reduction

Multi-Head Attention：

\[
512
\rightarrow64
\]

每个 head的：

\[
W_Q^{(h)}
\]

把当前 token投到：

> 64-D query space。

同样：

\[
W_K^{(h)}
\]

投到：

> 64-D key space。

---

# 87. 为什么 Q 和 K 必须输出相同维度？

因为要做：

\[
q^\top k
\]

所以：

\[
q,k
\in\mathbb R^{d_k}
\]

维度必须一致。

这由 architecture graph决定。

Linear Layer自身并不知道：

> “我必须64维。”

设计者为了后续 dot product设定：

\[
out\_features=64
\]

---

# 88. Value 的 Dimension 可以不同

理论上：

\[
d_v
\]

不必等于：

\[
d_k
\]

因为 Value不会和 Q做 dot product。

它只被：

\[
A V
\]

读取。

原始 Transformer选择：

\[
d_k=d_v=64
\]

是 architecture choice。

---

# 89. W_O 又是另一个 Linear Layer

Concat heads：

\[
[8\times64]=512
\]

然后：

\[
W_O:
512\rightarrow512
\]

作用：

> 把多个 head outputs重新混合到统一 residual space。

所以 MHA实际上充满 Linear transformations。

---

# 90. Multi-Head Attention 可以看成什么？

简化：

```text
Linear projections
↓
Dot products
↓
Softmax
↓
Weighted sum
↓
Linear projection
```

即：

\[
\boxed{
\text{Linear}
\rightarrow
\text{nonlinear/data-dependent routing}
\rightarrow
\text{Linear}
}
\]

---

# 91. 为什么 Q/K/V Projection 非常重要？

如果完全没有 learned projection：

\[
Q=K=V=X
\]

模型只能在原 hidden coordinates中：

> 用同一 representation同时做 matching和message passing。

独立 Linear layers允许：

- matching space；
- payload space；

解耦。

---

# 92. Linear 不是“降维工具”而已

它可以：

- 降维：
  \[
  512\rightarrow64
  \]
- 升维：
  \[
  512\rightarrow3200
  \]
- 保持维度但换 representation：
  \[
  512\rightarrow512
  \]
- 输出 task values：
  \[
  512\rightarrow14
  \]

所以核心不是：

> dimension change。

而是：

\[
\boxed{
\text{learned affine feature transformation}
}
\]

---

# 93. `Linear(512,512)` 也不是“什么都没做”

即使 input/output维度一样，

\[
W
\]

可以：

- rotate；
- scale；
- shear；
- mix features；
- collapse directions；

所以：

\[
512\rightarrow512
\]

不意味着 identity。

Identity只有当：

\[
W=I
\]

且：

\[
b=0
\]

---

# 94. 为什么 Residual Connection 需要相同维度，却不要求同一 Representation？

Residual：

\[
x+F(x)
\]

要求：

> coordinate shape一致。

但 \(F(x)\) 可以是复杂 learned update。

所以 Linear：

\[
512\rightarrow512
\]

常用于在相同 residual space width中：

> 构造新的 update directions。

---

# 95. Linear Layer 与 Dot Product 的关系

一个 neuron：

\[
y_j=w_j^\top x+b_j
\]

本质包含：

\[
\boxed{
\text{dot product}
}
\]

所以一个 Linear Layer就是：

> 同时拿 input与很多 learned weight vectors做 dot products，再加 biases。

---

# 96. `Linear(n,m)` 可以理解成 m 个 Learned Dot Products

\[
W=
\begin{bmatrix}
w_1^\top\\
\vdots\\
w_m^\top
\end{bmatrix}
\]

那么：

\[
\boxed{
y_j=w_j^\top x+b_j,\qquad j=1,\ldots,m
}
\]

这是理解 Linear Layer最有用的 mental model之一。

---

# 97. 为什么它和 Attention Dot Product 又不同？

Linear：

\[
w_j^\top x
\]

其中：

\[
w_j
\]

是：

> learned parameter vector，训练后对所有 samples固定。

Attention score：

\[
q_i^\top k_j
\]

其中：

\[
q_i,k_j
\]

是：

> 当前 input经过 projections得到的 dynamic activations。

所以：

\[
\boxed{
\text{Linear neuron: input vs parameter}
}
\]

\[
\boxed{
\text{Attention score: activation vs activation}
}
\]

---

# 98. 这两个 Dot Product 都重要

Linear先学习：

> 怎样把 raw hidden state变成 Q/K/V。

然后 Attention再比较：

> 当前 Q 与当前 K。

所以：

```text
learned static parameters
↓
create dynamic vectors
↓
dynamic vectors interact
```

---

# 99. Weight 是静态参数，Activation 是动态数据

训练结束后某次 inference：

\[
W
\]

固定。

输入：

\[
x
\]

改变。

所以：

\[
y=Wx+b
\]

随 input动态变化。

这和 Attention matrix：

\[
A(X)
\]

不同：

> Attention weights会随每个 input sequence重新生成。

---

# 100. Linear Layer 的 Weight 在 Training 中怎样学习？

设 loss：

\[
L
\]

Linear：

\[
y=Wx+b
\]

如果我们知道：

\[
g_y
=
\frac{\partial L}{\partial y}
\]

那么可以计算：

- input gradient；
- weight gradient；
- bias gradient。

---

# 101. 单个 Sample 的 Gradient：Column Convention

\[
y=Wx+b
\]

其中：

\[
x\in\mathbb R^n
\]

\[
y\in\mathbb R^m
\]

令：

\[
g=
\frac{\partial L}{\partial y}
\in\mathbb R^m
\]

则：

\[
\boxed{
\frac{\partial L}{\partial x}
=
W^\top g
}
\]

---

# 102. Weight Gradient

第 \(j,i\) 个 weight：

\[
y_j=
\sum_iW_{ji}x_i+b_j
\]

所以：

\[
\frac{\partial y_j}{\partial W_{ji}}
=
x_i
\]

因此：

\[
\boxed{
\frac{\partial L}{\partial W}
=
g x^\top
}
\]

shape：

\[
[m,1]
[1,n]
=
[m,n]
\]

正好和：

\[
W
\]

一样。

---

# 103. Bias Gradient

\[
\frac{\partial y_j}{\partial b_j}=1
\]

所以：

\[
\boxed{
\frac{\partial L}{\partial b}
=
g
}
\]

单样本情况下。

---

# 104. 这说明 Weight 怎样被训练？

如果某个 input feature：

\[
x_i
\]

在当前 sample很大，

同时某个 output：

\[
y_j
\]

收到很强 gradient：

\[
g_j
\]

则：

\[
\frac{\partial L}{\partial W_{ji}}
=
g_jx_i
\]

会相应较大。

所以 optimizer通过大量 samples：

> 学习哪些 input-output feature connections有助于降低 loss。

---

# 105. Batch 情况

PyTorch row convention：

\[
X:
[B,n]
\]

\[
W:
[m,n]
\]

\[
Y=XW^\top+b
\]

令：

\[
G=
\frac{\partial L}{\partial Y}
:
[B,m]
\]

则：

\[
\boxed{
\frac{\partial L}{\partial X}
=
GW
}
\]

---

# 106. Batch Weight Gradient

\[
\boxed{
\frac{\partial L}{\partial W}
=
G^\top X
}
\]

shape：

\[
[m,B]
[B,n]
=
[m,n]
\]

也就是说：

> 一个 batch中所有 samples对同一个 shared weight贡献会累加。

---

# 107. Batch Bias Gradient

如果 bias broadcast到所有 batch samples：

\[
\boxed{
\frac{\partial L}{\partial b}
=
\sum_{r=1}^{B}
G_r
}
\]

实际 reduction还取决于 loss是否 mean/sum，

但 local affine layer gradient structure如此。

---

# 108. Sequence Tensor 也是同样

输入：

\[
X:
[B,N,n]
\]

Linear：

\[
n\rightarrow m
\]

可以把：

\[
B\times N
\]

个 vectors概念上 flatten成：

\[
[B N,n]
\]

全部共享同一：

\[
W,b
\]

所以 gradient会从：

> 所有 batch × token positions

累积到同一 parameter matrix。

---

# 109. 这就是 Transformer Parameter Sharing 的一部分

FFN的：

```python
linear1
```

对所有 tokens共享。

所以：

> 每个 token产生自己的 activation和gradient，

但它们共同更新同一套 \(W_1\)。

---

# 110. 为什么 Linear Layer 能泛化到任意 Sequence Length？

参数：

\[
W:
[m,n]
\]

不依赖 token count：

\[
N
\]

所以训练可能看到：

\[
N=100
\]

理论上同一 layer可以应用：

\[
N=200
\]

只要 feature dimension：

\[
n
\]

一致。

---

# 111. 当然整个模型未必因此支持任意长度

因为还可能有：

- positional embedding limits；
- mask；
- memory；
- task design。

这里只是说：

\[
\boxed{
\text{Linear Layer本身与 sequence length无关}
}
\]

---

# 112. Weight Matrix 的 Rows 与 Columns 到底分别是什么？

PyTorch stored：

\[
W:
[out,in]
\]

### Row \(j\)

\[
W_{j,:}
\]

对应：

> output feature \(j\) 如何读取所有 input features。

---

### Column \(i\)

\[
W_{:,i}
\]

对应：

> input feature \(i\) 对所有 output features的连接权重。

---

# 113. 所以看 Row 更像“一个 Neuron”

第 \(j\) 行：

\[
w_j
\]

完整定义：

> output neuron \(j\) 的 input coefficients。

这是最直观读法。

---

# 114. 看 Column 又有什么用？

第 \(i\) 列告诉你：

> input dimension \(i\) 会怎样影响所有 outputs。

例如某一列全部接近0，

说明：

> 这个 input coordinate在这个 affine map中几乎没被使用。

但实际深网解释仍需谨慎。

---

# 115. Weight Visualization 能直接告诉 Semantic 吗？

通常不能。

因为 hidden dimensions本身：

> 不一定有独立可命名语义。

某个大 weight：

\[
W_{ji}
\]

只说明 local sensitivity/link较强，

不代表：

> feature \(i\) 就是某个人类概念。

深层网络解释需要考虑整个 computation graph。

---

# 116. Linear Layer 的 Jacobian

对于：

\[
y=Wx+b
\]

对 input：

\[
x
\]

的 Jacobian：

\[
\boxed{
J=
\frac{\partial y}{\partial x}
=
W
}
\]

这是 Linear/Affine Layer一个非常漂亮的性质。

---

# 117. Bias 不影响 Input Jacobian

因为：

\[
\frac{\partial b}{\partial x}=0
\]

所以：

\[
\boxed{
\frac{\partial (Wx+b)}{\partial x}=W
}
\]

这也是为什么 weight matrix本身直接描述：

> local linear sensitivity。

---

# 118. Affine Function 的 Jacobian everywhere 一样

无论：

\[
x
\]

在哪里：

\[
J(x)=W
\]

不会随 input变化。

这就是：

> affine layer本身没有 input-dependent local geometry变化。

---

# 119. ReLU 后就不同

\[
y=ReLU(Wx+b)
\]

Jacobian变成：

\[
D(x)W
\]

其中：

\[
D(x)
\]

取决于哪些 neurons active。

所以：

> nonlinearity让 effective transformation随 input改变。

---

# 120. 这又解释 Deep Network 的表达能力

Affine：

\[
J=W
\]

固定。

Nonlinear composition：

> 不同 input区域拥有不同 effective Jacobians。

因此可以形成：

- curved decision regions；
- complex feature transformations。

---

# 121. Linear Layer 能不能“学习公式”？

某种意义上：

> 它学习一组 affine coefficients。

例如真实关系：

\[
y=3x_1-2x_2+5
\]

一个 Linear Layer可以直接学：

\[
w=[3,-2]
\]

\[
b=5
\]

---

# 122. 但复杂函数需要多个模块

例如：

\[
y=x_1x_2
\]

单个 affine layer不能在全空间精确表示这种乘法。

需要：

- nonlinear network；
- special interaction；
- approximation。

所以 Linear Layer是基础 building block，

不是万能函数。

---

# 123. 为什么 Attention 里用 Linear Projection 很合理？

我们希望从同一个 hidden representation：

\[
x
\]

构造不同 views：

- query view；
- key view；
- value view。

Linear projection提供：

> 简单、可微、高效、learned feature recombination。

后续 dot product + softmax再引入：

> dynamic interaction。

---

# 124. 为什么 Q/K/V 不先用很深 MLP？

理论上可以设计。

但原始 Transformer选择 simple learned linear projections：

- 计算高效；
- 参数可控；
- 后续 Attention本身已有非线性 routing；
- 多层 Transformer会反复产生复杂 representation。

Architecture是整体设计平衡。

---

# 125. Linear Projection 和 Embedding 有什么区别？

`nn.Embedding`通常通过：

> integer index查表。

例如 token id：

\[
42
\]

直接选：

\[
E_{42,:}
\]

---

`nn.Linear`：

> 根据连续 input features做 weighted combination。

所以：

\[
\boxed{
Embedding:
index\rightarrow vector
}
\]

\[
\boxed{
Linear:
vector\rightarrow vector
}
\]

---

# 126. ACT Query Embedding 不是 Linear

官方：

```python
self.query_embed =
    nn.Embedding(
        num_queries,
        hidden_dim
    )
```

每个 action slot index：

> 查一个 learned 512-D vector。

它不是把某个 continuous action index通过：

\[
Wx+b
\]

计算出来。

---

# 127. 但 Query Embedding 进入 Attention 后会再被 Linear Project

`nn.MultiheadAttention`内部：

\[
Q
\]

仍需 Q projection。

所以：

```text
learned query embedding
↓
Q Linear Projection
↓
head-specific query vectors
```

这是两个不同 operations。

---

# 128. ACT Image Projection 为什么是 Conv2d，不是 Linear？

官方：

```python
self.input_proj =
    nn.Conv2d(
        backbone_channels,
        hidden_dim,
        kernel_size=1
    )
```

它把 backbone channel dimension投到：

\[
512
\]

---

# 129. 1×1 Conv 和 Linear 有什么关系？

对于每一个 spatial location：

\[
(h,w)
\]

1×1 convolution只看该位置的 channel vector：

\[
x_{h,w}
\in\mathbb R^{C_{in}}
\]

然后计算：

\[
\boxed{
y_{h,w}
=
Wx_{h,w}+b
}
\]

其中同一：

\[
W,b
\]

在所有 spatial locations共享。

---

# 130. 所以 1×1 Conv 本质上像 Spatially Shared Linear Layer

可以理解：

\[
C_{in}\rightarrow C_{out}
\]

per pixel/location affine transform，

并在：

\[
H\times W
\]

位置共享。

所以 ACT视觉 projection与：

```python
Linear(C_in,512)
```

在单个 spatial location上的数学形式非常接近。

---

# 131. 但 `Conv2d(1×1)` 仍不是 API 上的 `nn.Linear`

它有：

- channel/spatial tensor convention；
- convolution implementation；
- shared kernel semantics。

所以正确说：

> 1×1 conv在每个 spatial location执行同一 channel-wise affine map。

而不是：

> “它就是同一个 Python class”。

---

# 132. Linear 与 Normalization 又有什么区别？

Linear：

\[
y=Wx+b
\]

parameters：

> learned固定。

LayerNorm：

\[
\frac{x-\mu(x)}{\sqrt{\sigma^2(x)+\epsilon}}
\]

statistics：

> input-dependent。

所以 LayerNorm不是一个 fixed affine transformation，

虽然最后还有：

\[
\gamma,\beta
\]

---

# 133. Linear 与 Softmax 也完全不同

Linear output：

\[
y_j
\]

可以任意正负，

不要求 sum=1。

Softmax：

\[
p_j
=
\frac{e^{y_j}}{\sum_ke^{y_k}}
\]

把 logits转成：

> positive normalized weights。

所以 classification常：

```text
Linear
→ logits
→ Softmax / CrossEntropy
```

---

# 134. 为什么 Logit 常由 Linear Head 输出？

因为最后 hidden representation：

\[
h
\]

已经包含复杂 nonlinear features。

每个 class只需一个 learned vector：

\[
w_c
\]

计算：

\[
z_c=w_c^\top h+b_c
\]

即可衡量：

> representation与该 class readout direction的匹配。

---

# 135. ACT Action Head 不用 Softmax

因为动作：

\[
a\in\mathbb R^{14}
\]

是连续 regression target，

不是 mutually exclusive classes。

所以：

\[
Linear(512,14)
\]

直接输出 normalized action values。

---

# 136. ACT μ/logvar Head 也不用 Softmax

因为：

\[
\mu
\]

可以任意实数。

\[
\log\sigma^2
\]

也可以任意实数。

真正正 variance通过：

\[
\sigma^2
=
\exp(
\log\sigma^2
)
\]

获得。

所以输出 layer不需要：

> probability normalization。

---

# 137. 为什么输出 logvar 而不是直接 variance？

如果直接预测：

\[
\sigma^2
\]

必须保证：

\[
\sigma^2>0
\]

Linear output本身不保证正。

预测：

\[
\log\sigma^2
\in\mathbb R
\]

则：

\[
\exp(\log\sigma^2)>0
\]

天然满足 positivity。

这是 downstream parameterization赋予 Linear output意义的又一个例子。

---

# 138. Linear Layer 会自动限制输出范围吗？

不会。

\[
Wx+b
\]

理论上可以：

\[
(-\infty,+\infty)
\]

如果需要范围限制，

要额外：

- Sigmoid；
- Tanh；
- Softmax；
- clamp；
- task-specific transform。

---

# 139. 为什么 ACT Action Head 可以不显式 Tanh？

因为 training targets经过 normalization，

网络可以通过 loss学习合适范围。

最终 inference再：

> de-normalize。

Canonical ACT并不需要通过：

\[
tanh
\]

硬限制 action head。

---

# 140. Linear Layer 的 Initialization 为什么重要？

训练刚开始：

\[
W,b
\]

还没有 learned semantics。

需要初始化到：

> 合理尺度，

避免 activations/gradients过大或过小。

---

# 141. PyTorch `nn.Linear` 当前默认初始化

PyTorch官方文档说明 weight：

\[
[out,in]
\]

初始化于一个与：

\[
in\_features
\]

有关的 uniform range。

其当前实现等价于大致：

\[
\boxed{
W_{ij}
\sim
U
\left(
-\frac1{\sqrt{n}},
\frac1{\sqrt{n}}
\right)
}
\]

其中：

\[
n=in\_features
\]

bias若存在，也使用同尺度范围。

---

# 142. 为什么 Scale 随 in_features 变化？

如果 input有：

\[
n
\]

项，

输出：

\[
y_j=
\sum_{i=1}^nW_{ji}x_i
\]

如果 weights尺度完全不随：

\[
n
\]

调整，

当 \(n\) 很大时，

sum variance可能随 dimension快速增大。

合理 initialization希望：

> 控制 forward/backward scale。

这与我们前面 Dot Product scaling有类似“维度累积”背景，

但具体初始化理论是另一个专题。

---

# 143. 不要把 Initialization 当成最终 Weight Distribution

初始化只发生在：

> training开始。

经过 optimizer更新后：

\[
W
\]

不再服从最初 uniform distribution。

所以不能在训练完成后说：

> “Linear weights服从这个均匀分布。”

---

# 144. ACT Transformer 又会做额外 Initialization

ACT/DETR-style `Transformer._reset_parameters()` 会对某些参数：

> 使用 Xavier uniform 初始化。

所以具体模型某个 Linear最终初始化方式：

> 还要看外层 model是否覆盖 module默认 initialization。

这说明：

\[
\boxed{
\text{PyTorch module default}
\neq
\text{specific architecture final initialization recipe}
}
\]

---

# 145. 为什么这篇不把 Initialization 展开太深？

因为初始化本身值得单独 canonical page：

> Xavier / Glorot Initialization。

会涉及：

- activation variance；
- fan-in；
- fan-out；
- forward/backward scale。

Linear Layer这里只建立：

> parameter初始不是有语义的，它通过训练变成有用 mapping。

---

# 146. “Projection”这个词到底是什么意思？

深度学习经常说：

> project 14-D qpos to 512-D embedding。

严格线性代数中：

> projection 有更专门的含义，例如 \(P^2=P\)。

而神经网络工程里：

> projection 常被宽泛用来表示 learned Linear mapping from one feature space to another。

所以：

\[
\boxed{
\text{neural projection}
\text{ 不一定是严格数学投影算子}
}
\]

---

# 147. 例如 Q Projection

我们叫：

\[
W_Q
\]

“Query Projection”。

并不要求：

\[
W_Q^2=W_Q
\]

甚至它可能：

\[
512\rightarrow64
\]

根本不是方阵。

所以这是：

> feature projection/mapping 的工程术语。

---

# 148. “Embedding”这个词也类似

ACT说：

> project qpos into embedding space。

意思是：

> 得到 Transformer hidden representation。

不代表一定通过：

```python
nn.Embedding
```

qpos是 continuous vector，

所以实际上用：

```python
nn.Linear(14,512)
```

---

# 149. Linear Layer 是否有 Memory？

Parameters：

\[
W,b
\]

当然存储训练学到的信息。

但 forward没有：

> recurrent hidden state。

同一个 input和parameters：

> 输出确定。

所以它不是 RNN式动态 memory。

---

# 150. Linear Layer 是否是 End-to-End Learnable？

是。

因为：

\[
y=Wx+b
\]

对：

- \(x\)；
- \(W\)；
- \(b\)；

都可微。

所以 loss gradient可以直接反向传播。

---

# 151. 这就是为什么各种模块都喜欢用 Linear

它同时具有：

- 表达力作为 learned feature mixing；
- 简单数学；
- 高效 GEMM；
- 易于反向传播；
- shape可控。

现代 accelerator又非常擅长：

> dense matrix multiplication。

---

# 152. Linear 的 Compute Complexity

对于一个：

\[
n\rightarrow m
\]

vector，

主要乘加操作量约：

\[
O(nm)
\]

如果有：

\[
B\times N
\]

个 vectors：

\[
O(BNnm)
\]

所以大型 FFN中的 Linear：

> 往往是 Transformer主要 FLOP来源之一。

---

# 153. 为什么 GPU 特别喜欢 Linear Layer？

因为大量 vectors可以组成矩阵：

\[
X
\]

然后统一做：

\[
XW^\top
\]

这是标准：

> GEMM（General Matrix Multiplication）。

GPU/TPU对此高度优化。

---

# 154. 所以 Python 看起来只是一个 `nn.Linear`

背后实际可能是：

> 大型并行矩阵乘 kernel。

例如：

\[
[B,1202,512]
\]

乘：

\[
[512,3200]
\]

一次就处理：

\[
B\times1202
\]

个 token vectors。

---

# 155. Linear Layer 和 Fully Connected 是不是一回事？

经典 MLP语境：

> 基本可以。

因为每个 output unit：

> 连接所有 input features。

所以也叫：

> fully connected layer。

---

# 156. 但在 Transformer Sequence 上为什么又不是“所有 Token 全连接”？

因为 “fully connected” 指：

> feature dimension内部。

例如每个 token：

\[
512\rightarrow3200
\]

所有 3200 outputs都读取该 token的全部512 features。

但不同 tokens之间：

> 不是这个 Linear Layer连接的。

所以“fully connected”要明确：

> 对哪个 axis。

---

# 157. Attention 才是 Dynamic Token-to-Token Connectivity

Linear：

> feature mixing。

Attention：

> token mixing / dynamic routing。

这又回到我们前面建立的分工：

\[
\boxed{
\text{Linear/FFN:
feature-space computation}
}
\]

\[
\boxed{
\text{Attention:
cross-token interaction}
}
\]

当然 Attention内部本身也大量使用 Linear。

---

# 158. 一个 Linear Layer 能同时混 Feature 和 Token 吗？

如果你先 reshape：

\[
[N,D]
\]

成：

\[
[ND]
\]

再对整个 flattened vector做 Linear，

当然可以混全部 positions。

但标准 Transformer position-wise Linear没有这么做。

所以：

> Linear数学本身不认识“token axis”。

是否跨 token取决于：

> 你把哪一维当 in_features。

---

# 159. 为什么 Shape 思维如此重要？

`nn.Linear` 只看：

> 最后一维。

如果你错误 reshape / permute，

就可能把：

- features；
- tokens；
- channels；

混成完全不同意义。

所以读 AI代码首先问：

\[
\boxed{
\text{当前 tensor 每个 axis 代表什么？}
}
\]

---

# 160. ACT 中 Linear Layer 总表

可以把核心 `nn.Linear` 整理成：

| 模块 | Shape | 作用 |
|---|---:|---|
| `encoder_joint_proj` | \(14\to512\) | qpos → CVAE token |
| `encoder_action_proj` | \(14\to512\) | action → CVAE token |
| `latent_proj` | \(512\to64\) | `[CLS]` → \(\mu,\log\sigma^2\) |
| `latent_out_proj` | \(32\to512\) | latent \(z\) → policy latent token |
| `input_proj_robot_state` | \(14\to512\) | current qpos → policy token |
| Transformer FFN `linear1` | \(512\to3200\) | feature expansion |
| Transformer FFN `linear2` | \(3200\to512\) | return to residual width |
| `action_head` | \(512\to14\) | decoder slot → robot action |
| `is_pad_head` | \(512\to1\) | decoder slot → padding score |

另外 MHA内部还有：

\[
W_Q,W_K,W_V,W_O
\]

等 learned projections。

---

# 161. `latent_proj` 的64维怎样 split？

latent dim：

\[
32
\]

所以：

\[
64=32+32
\]

概念上：

\[
latent\_info
=
[
\mu_1,\ldots,\mu_{32},
\log\sigma_1^2,\ldots,\log\sigma_{32}^2
]
\]

具体代码split顺序应以实现为准。

核心是：

> 同一个 Linear同时输出两组 posterior parameters。

---

# 162. 为什么可以用一个 Linear 而不是两个？

数学上：

两个独立：

```python
mu_head = Linear(512,32)
logvar_head = Linear(512,32)
```

等价于：

> 把两套 output rows拼成一个：

```python
Linear(512,64)
```

只要后面正确 split。

---

# 163. 这说明“一个 Linear Layer”其实只是很多 Output Neurons 的打包

前32 rows：

> 可以被后续用作 \(\mu\)。

后32 rows：

> 用作 logvar。

它们虽然存储在同一个 weight tensor，

每一 row仍拥有独立参数。

---

# 164. 是否意味着 μ 与 logvar 完全独立？

它们的 final affine rows不同，

但都共享同一个 input：

\[
h_{CLS}
\]

而这个 upstream representation由共同 encoder产生。

所以整个网络层面：

> 并不独立。

只是 final readout weights不同。

---

# 165. 为什么 `latent_proj` 不需要 Activation？

因为：

\[
\mu
\]

可以是任意 real。

\[
\log\sigma^2
\]

也可以是任意 real。

所以直接 affine output很自然。

如果加 ReLU：

> 会错误限制它们必须非负。

---

# 166. 为什么 Action Head 也没有 ReLU？

Robot joint target经过 normalization后：

> 可能正也可能负。

所以 final Linear直接输出 unrestricted real values。

如果最后加 ReLU：

> 会禁止负 normalized action。

---

# 167. Final Layer 是否需要 Activation 取决于 Output Semantics

例如：

### Binary probability

Linear logit：

\[
z
\]

再：

\[
Sigmoid(z)
\]

---

### Multiclass probability

Linear logits：

\[
z
\]

再：

\[
Softmax(z)
\]

或 CrossEntropy直接接 logits。

---

### Regression

常常：

> Linear直接输出 real values。

---

### Positive scale

可能：

\[
Linear
\rightarrow
Softplus
\]

或者预测 log-scale再 exponentiate。

所以：

\[
\boxed{
\text{Linear output semantics由 downstream transform决定}
}
\]

---

# 168. 这就是为什么“Linear Layer 学的是什么？”没有脱离 Context 的唯一答案

数学上它学：

\[
\boxed{
W,b
}
\]

也就是：

> 一套 affine mapping。

语义上它学什么：

> 取决于它在整个模型中的位置。

---

# 169. Q Projection 学的是怎样的 Mapping？

它学习：

> 把 current hidden representation变成对后续 query-key compatibility计算有用的 feature coordinates。

这不是人工指定。

由 task loss端到端形成。

---

# 170. K Projection 呢？

学习：

> 被 Query拿来匹配时有用的 key coordinates。

---

# 171. V Projection 呢？

学习：

> 当某个 position被 attention读取时，应该传递什么 message features。

---

# 172. W_O 呢？

学习：

> 多个 head outputs应该怎样重新混合成 residual-space update。

---

# 173. FFN Linear1 呢？

学习：

> 从 current token representation构造大量 intermediate pre-activation feature responses。

---

# 174. FFN Linear2 呢？

学习：

> 把经过 activation的 wide features重新写回 residual space。

---

# 175. ACT Joint Projection 呢？

学习：

> 怎样把14维 physical joint configuration表示成 Transformer可用的512-D hidden feature pattern。

---

# 176. ACT Action Projection 呢？

学习：

> 怎样把14维 demonstration action表示成 CVAE Encoder能够处理的512-D action token。

---

# 177. ACT Latent Projection 呢？

学习：

> 从 `[CLS]` summary中读出 posterior parameter values。

---

# 178. ACT Latent Out Projection 呢？

学习：

> 把32-D latent coordinates映射成 policy Transformer hidden space中的 conditioning token。

---

# 179. ACT Action Head 呢？

学习：

> 从 decoder action-slot representation中读出14-D action coordinates。

---

# 180. 所有这些都是同一个数学模块

\[
\boxed{
y=Wx+b
}
\]

区别：

> 不是 `Linear` 公式不同，

而是：

- input是什么；
- output怎么被使用；
- loss从哪里回来。

---

# 181. 这是理解深度学习代码非常重要的一次“去神秘化”

看到：

```python
nn.Linear(512, 64)
```

不要问：

> “这个层怎么知道应该提取什么？”

先问：

1. Input representation是什么？
2. Output接下来去哪里？
3. 哪个 loss监督它？
4. Gradient怎样回来？

这样你才能知道：

> 它为什么最终学成现在的功能。

---

# 182. 一个最小 PyTorch 例子

```python
import torch
import torch.nn as nn

layer = nn.Linear(
    3,
    2
)

x = torch.tensor(
    [[2.0, 1.0, -1.0]]
)

y = layer(x)

print(y.shape)
```

输出 shape：

```text
[1, 2]
```

因为：

\[
3\rightarrow2
\]

---

# 183. 手动复现 PyTorch Forward

PyTorch stored：

```python
layer.weight.shape
# [2, 3]

layer.bias.shape
# [2]
```

手算：

```python
y_manual = (
    x @ layer.weight.T
    + layer.bias
)
```

则：

```python
torch.allclose(
    y,
    y_manual
)
```

应该为：

```text
True
```

忽略数值精度问题。

---

# 184. 为什么 `.T`？

因为：

\[
x:
[B,3]
\]

而：

\[
weight:
[2,3]
\]

需要：

\[
weight^\top:
[3,2]
\]

所以：

\[
[B,3]
[3,2]
\rightarrow
[B,2]
\]

---

# 185. 如果你自己用参数矩阵 [in,out] 呢？

你也可以定义：

\[
W:
[3,2]
\]

然后写：

\[
xW+b
\]

数学完全可以。

只是 PyTorch `nn.Linear` 的 stored weight convention是：

\[
[out,in]
\]

因此 forward内部使用 transpose意义。

---

# 186. 不要把 Storage Convention 当成数学定律

框架可以选择不同 layout。

真正不变的是：

\[
\boxed{
y_j
=
\sum_iw_{ji}x_i+b_j
}
\]

只要知道 scalar equation，

任何 matrix notation都能重新推出来。

---

# 187. Linear Layer 是否一定 Dense？

`nn.Linear` 是 dense affine layer。

也存在：

- sparse linear operators；
- low-rank adapters；
- structured matrices；
- convolution；
- tensor factorization。

它们利用不同结构减少参数或计算。

---

# 188. Low-Rank Linear 又是什么？

如果：

\[
W
\]

很大，

可以近似：

\[
W\approx AB
\]

其中：

\[
A:
[m,r]
\]

\[
B:
[r,n]
\]

且：

\[
r\ll \min(m,n)
\]

这能降低参数量。

LoRA等方法就利用了：

> low-rank parameter update

思想。

但 canonical Linear Layer先理解完整 dense \(W\)。

---

# 189. 为什么 LoRA 也离不开 Linear Layer？

Transformer大量参数就在：

- Q/K/V/O；
- FFN；

这些 Linear maps中。

LoRA通过给某些：

\[
W
\]

增加低秩 update：

\[
\Delta W=BA
\]

来低成本 fine-tune。

所以理解 Linear Layer也是理解 LLM fine-tuning的基础。

---

# 190. Quantization 为什么也重点处理 Linear Weights？

因为大模型大量计算是：

\[
XW^\top
\]

如果：

\[
W
\]

从 FP16量化到：

- INT8；
- INT4；

可以显著：

- 减少 memory；
- 加速 matrix multiplication。

所以 Linear Layer也是 AI Infra的核心算子。

---

# 191. ACT/Robot Policy 部署同样如此

即使不是 LLM，

Transformer policy中大量：

- FFN；
- Attention projection；

仍是 GEMM / affine operations。

所以 Linear Layer不仅是理论基础，

也是：

> 推理优化、量化、加速的核心对象。

---

# 192. Linear Layer 和 Parameter Count 的关系非常直接

\[
n\rightarrow m
\]

参数近似：

\[
nm
\]

所以当：

\[
n,m
\]

都很大，

参数量迅速增长。

例如：

\[
4096\rightarrow11008
\]

这类现代 LLM FFN projection：

> 一个矩阵就非常巨大。

---

# 193. 为什么矩阵乘是现代 AI 芯片核心？

因为神经网络反复执行：

\[
Y=XW
\]

所以 GPU Tensor Cores、TPUs等硬件专门优化：

> multiply-accumulate

大规模并行。

从这个角度：

> `nn.Linear` 是现代深度学习最基础的硬件工作负载之一。

---

# 194. 但 Linear Layer 本身没有“智能”

这一点值得强调。

\[
y=Wx+b
\]

是非常简单的数学函数。

真正复杂能力来自：

- 大量 Linear；
- nonlinearities；
- Attention；
- normalization；
- residual；
- deep composition；
- large-scale data；
- optimization。

所以：

\[
\boxed{
\text{simple primitives}
+
\text{deep composition}
=
\text{complex model behavior}
}
\]

---

# 195. 常见误解一：`nn.Linear` 严格数学上一定是 Linear Map

**默认带 bias 时不是。**

它是：

\[
Wx+b
\]

即 affine transformation。

---

# 196. 常见误解二：`Linear(512,64)` 的 Weight Shape 是 [512,64]

PyTorch存储：

\[
\boxed{
[64,512]
}
\]

因为：

\[
[out,in]
\]

---

# 197. 常见误解三：PyTorch 公式和论文 \(xW\) 冲突

**不冲突。**

只是 row/column convention不同。

---

# 198. 常见误解四：每个 Output 只读取一个 Input Feature

**错误。**

每一个：

\[
y_j
\]

通常读取：

\[
\boxed{
\text{所有 input features}
}
\]

---

# 199. 常见误解五：Linear Layer 会自动让 Tokens 相互交流

**错误。**

标准 `nn.Linear`只作用最后 feature dimension。

---

# 200. 常见误解六：512→3200 创建了3200个独立新信息维度

**错误。**

单 affine map的 rank受 input dimension限制。

---

# 201. 常见误解七：升维本身就是非线性

**错误。**

无论多宽：

\[
Wx+b
\]

仍是 affine。

---

# 202. 常见误解八：两层 Linear 一定比一层更强

**没有 activation 时可以合并成一层 affine map。**

---

# 203. 常见误解九：Bias 只是可有可无的小常数，没有数学作用

它允许：

> translation，

使：

\[
T(0)\neq0
\]

并移动 decision hyperplane。

---

# 204. 常见误解十：输入0，Linear输出一定0

**只有 bias为0时。**

默认：

\[
y=b
\]

---

# 205. 常见误解十一：ACT z=0 所以 latent token是0

**错误。**

`latent_out_proj` 默认含 bias：

\[
Linear(0)=b
\]

---

# 206. 常见误解十二：Linear Layer 自己知道 Output 是 μ

**错误。**

后续 graph把 output当成 \(\mu\)，loss才训练出这种功能。

---

# 207. 常见误解十三：Linear Layer 自己知道 Output 是 Action

**错误。**

Action target、loss和execution convention赋予语义。

---

# 208. 常见误解十四：Q/K/V 是三种特殊神经元类型

**错误。**

它们本质上是普通 learned affine projections，

角色来自 Attention computation。

---

# 209. 常见误解十五：一个 Output Neuron 就是对某个 Input Dimension做 Projection

**不准确。**

通常：

\[
w_j^\top x
\]

混合所有 input dimensions。

---

# 210. 常见误解十六：Neural “Projection” 一定满足数学投影 \(P^2=P\)

**错误。**

深度学习中“projection”通常只是宽泛指 learned mapping。

---

# 211. 常见误解十七：Linear Layer 会把512维“压缩成64维但信息完全不丢”

一般不能保证。

如果：

\[
64<512
\]

rank上限64，

通常存在信息丢失。

---

# 212. 常见误解十八：512→512 一定不丢信息

也不保证。

如果：

\[
W
\]

rank-deficient，

仍可能丢失 directions。

只有 invertible square \(W\)才是一一可逆的线性部分。

---

# 213. 常见误解十九：Weight Matrix 每一列是一个 Output Neuron

在 PyTorch `[out,in]` convention下：

> 每一 **row** 对应一个 output unit。

---

# 214. 常见误解二十：Linear Layer 的 Weight 会随每个 Input 动态变化

标准 `nn.Linear`：

> 不会。

训练完成后 weights固定。

动态变化的是 activation。

---

# 215. 常见误解二十一：Attention Weight 和 Linear Weight 是同一种东西

不是。

### Linear Weight \(W\)

learned parameters。

### Attention Weight \(A(X)\)

由当前 input动态计算的 activations。

---

# 216. 常见误解二十二：Linear Output 默认是 Probability

不是。

它只是 real-valued activations。

需要 Softmax/Sigmoid等才能赋予特定 probability interpretation。

---

# 217. 常见误解二十三：Action Head 必须有 Tanh

不一定。

取决于 target normalization和task设计。

ACT canonical action head就是 Linear readout。

---

# 218. 常见误解二十四：logvar必须由正数输出层产生

不需要。

\[
\log\sigma^2
\]

本来就是任意 real。

Exponentiation后 variance才为正。

---

# 219. 常见误解二十五：1×1 Conv 和 Linear 毫无关系

在单个 spatial position，

1×1 conv就是：

> shared channel-wise affine mapping。

但 API/tensor semantics不同。

---

# 220. 常见误解二十六：Linear Layer 没有 Bias 也叫 Affine

严格地：

\[
Wx
\]

是 linear。

\[
Wx+b
\]

才是 affine。

---

# 221. 常见误解二十七：所有论文中的 \(W\) Shape 都应该和 PyTorch `.weight.shape` 一样

不一定。

论文常使用：

- row-vector；
- column-vector；

不同 notation。

应从公式推 shape。

---

# 222. 常见误解二十八：Linear Layer 的 Expansion 可以替代 Activation

不能。

纯 affine composition仍可合并。

---

# 223. 常见误解二十九：Output Dimension 越大一定越强

不一定。

更大 output width意味着：

- 更多参数；
- 更多 compute；
- 更多 coordinates；

但性能取决于：

- architecture；
- nonlinearity；
- data；
- optimization；
- downstream use。

---

# 224. 常见误解三十：Linear 是一个“简单层”，所以对模型不重要

Transformer中绝大多数大参数矩阵：

> 都属于 Linear/affine transformations。

它是最基础也最重要的算子之一。

---

# 225. 一张图记住 Linear Layer

```text
Input x ∈ R^n
   │
   ├── dot with learned w₁ + b₁ ──→ y₁
   ├── dot with learned w₂ + b₂ ──→ y₂
   ├── dot with learned w₃ + b₃ ──→ y₃
   │
   ...
   └── dot with learned w_m + b_m ─→ y_m
```

统一矩阵写法：

\[
\boxed{
y=Wx+b
}
\]

---

# 226. 一张图记住 PyTorch Shape

```text
nn.Linear(in_features=n, out_features=m)

input:
[..., n]

weight:
[m, n]

bias:
[m]

forward:
input @ weight.T + bias

output:
[..., m]
```

---

# 227. 一张图记住 Transformer 中的 Linear

```text
hidden x
   │
   ├── Linear ─→ Query
   ├── Linear ─→ Key
   ├── Linear ─→ Value
   │
   ▼
Attention
   │
   ▼
Concat heads
   │
   ▼
Linear W_O
   │
   ▼
residual-space output
```

---

# 228. 一张图记住 ACT 中的 Linear

```text
qpos [14]
   │
   ▼
Linear 14→512
   │
   ▼
joint token

action [14]
   │
   ▼
Linear 14→512
   │
   ▼
action token

h_CLS [512]
   │
   ▼
Linear 512→64
   │
   ├── first 32  → μ
   └── second 32 → logσ²

z [32]
   │
   ▼
Linear 32→512
   │
   ▼
latent token

decoder slot [512]
   │
   ▼
Linear 512→14
   │
   ▼
predicted robot action
```

每一条箭头背后：

\[
\boxed{
y=Wx+b
}
\]

---

# 229. 最重要的 Mental Model：Linear = Learned Readout / Re-encoding

一个非常好用的直觉是：

> Linear Layer拿到当前 feature vector后，用很多 learned weight vectors重新“读”它，并把结果写成另一组 coordinates。

输入：

\[
x
\]

每个输出：

\[
y_j=w_j^\top x+b_j
\]

所以：

> 每个 \(w_j\) 都像一种 learned question：

> “当前 input沿我这组 feature combination响应多大？”

这只是直觉，

不是说每个 row都具有可解释语言问题。

---

# 230. 更严谨的说法

\[
\boxed{
\text{A Linear/Affine Layer learns a basis of output responses, not semantics by itself.}
}
\]

这些 responses最终承担什么角色：

> 由整个网络的 computation graph与 objective决定。

---

# 231. 一句话真正理解 Linear Layer

> **神经网络中的 Linear Layer 本质上是一组并行的 learned weighted sums：每个 output dimension都有一个 weight vector \(w_j\)，通过 \(y_j=w_j^\top x+b_j\) 从全部 input features中读取一种 learned linear response；把所有 output neurons堆在一起就是矩阵形式 \(y=Wx+b\)。默认带 bias时它严格说是 affine transformation，而这些输出最终是 Query、Key、Value、latent parameter还是 robot action，并不是 Linear Layer自己知道的，而是由它在 computation graph中的位置、后续运算、training targets和 loss gradients共同赋予的。**

---

# 232. 一句话理解为什么 Linear 到处都是

> **Linear Layer可以用极其高效的矩阵乘完成任意 learned feature mixing，并且对参数和输入都可微；它既能改变 feature dimension，也能在相同维度中重新编码 representation，因此 Transformer可以用它构造 Q/K/V 和 \(W_O\)，FFN用它完成扩维与压回，CVAE用它读取 distribution parameters，ACT用它把 joint/action/latent映射进统一 hidden space并最终把 decoder representation读出为14维动作。**

---

# 233. 一句话连接你之前的 μ / logvar 问题

> **`h_CLS → Linear(512,64)` 并不是一个“知道如何把输入变成均值和方差”的特殊概率层；它只是输出64个 affine responses。之所以前32维最终成为 \(\mu\)、后32维成为 \(\log\sigma^2\)，是因为代码这样解释它们并把它们送入 reparameterization 和 KL objective，于是 reconstruction + KL 的梯度不断调整对应 rows of \(W\)，让这些 numbers逐渐成为对训练有用的 posterior parameters。**

---

# 234. 下一篇：ReLU

现在我们已经真正理解：

\[
Linear
\]

能做什么，

也知道一个严重限制：

\[
Linear
\rightarrow
Linear
\]

仍然只是 Linear/Affine。

那么真正让 MLP 开始拥有 nonlinear expressive power 的那个最小模块就是：

\[
\boxed{
ReLU
}
\]

下一篇：

> **[ReLU：为什么只做 \(\max(0,x)\)，就能让神经网络学非线性？](./relu.md)**

会详细解释：

- 什么叫 linear / nonlinear function；
- 为什么：
  \[
  ReLU(x)=\max(0,x)
  \]
  如此简单却足够关键；
- 为什么两个 Linear中间没有 ReLU可以合并；
- ReLU如何把 input space切成不同 linear regions；
- 一个 neuron的：
  \[
  w^\top x+b=0
  \]
  为什么形成 activation boundary；
- 多个 ReLU如何组成 piecewise-linear function；
- ReLU derivative；
- 为什么负区间 gradient=0；
- dying ReLU；
- 为什么 ACT FFN：
  \[
  512\rightarrow3200\rightarrow ReLU\rightarrow512
  \]
  需要这个 activation；
- ReLU、GELU、Sigmoid、Tanh 的角色区别。

---

## Primary Mathematical Background：Linear Transformations

MIT OpenCourseWare — Gilbert Strang.

**Linear Transformations and Their Matrices.**

- https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/pages/positive-definite-matrices-and-applications/linear-transformations-and-their-matrices/
- https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/resources/lecture-30-linear-transformations-and-their-matrices-1/

核心数学背景：

> matrix multiplication represents a linear transformation between vector spaces.

严格 linear transformation必须保持：

\[
T(x+y)=T(x)+T(y)
\]

\[
T(cx)=cT(x)
\]

因此：

\[
T(0)=0
\]

必然成立。

所以：

\[
Wx
\]

是 linear，

而：

\[
Wx+b,\quad b\neq0
\]

严格说是 affine transformation。

---

## Deep Learning Background

Ian Goodfellow, Yoshua Bengio, Aaron Courville.

**Deep Learning.**  
MIT Press, 2016.

- https://www.deeplearningbook.org/

Feedforward networks使用 affine transformations与 nonlinear activation functions组合构造复杂 functions。

本文中的：

\[
Wx+b
\]

是 dense neural network最基础的 affine building block。

---

## PyTorch Primary Implementation Reference

PyTorch `torch.nn.Linear`:

https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html

当前官方文档定义：

```python
torch.nn.Linear(
    in_features,
    out_features,
    bias=True
)
```

并明确说明：

\[
\boxed{
y=xA^\top+b
}
\]

PyTorch stored parameters：

\[
\boxed{
weight:
[out\_features,in\_features]
}
\]

\[
\boxed{
bias:
[out\_features]
}
\]

输入：

\[
(*,H_{in})
\]

输出：

\[
(*,H_{out})
\]

所以所有前置 dimensions保持不变，

`nn.Linear`只转换最后一个 feature dimension。

---

## ACT Official Implementation

Official repository:

https://github.com/tonyzhaozh/act

Model source:

https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py

当前官方 `DETRVAE` 中可直接看到：

```python
self.action_head =
    nn.Linear(
        hidden_dim,
        state_dim
    )

self.is_pad_head =
    nn.Linear(
        hidden_dim,
        1
    )
```

以及：

```python
self.input_proj_robot_state =
    nn.Linear(
        14,
        hidden_dim
    )
```

CVAE Encoder：

```python
self.encoder_action_proj =
    nn.Linear(
        14,
        hidden_dim
    )

self.encoder_joint_proj =
    nn.Linear(
        14,
        hidden_dim
    )
```

posterior parameter projection：

```python
self.latent_dim = 32

self.latent_proj =
    nn.Linear(
        hidden_dim,
        self.latent_dim * 2
    )
```

因此：

\[
\boxed{
512\rightarrow64
}
\]

再在 forward中解释为：

\[
\mu,\log\sigma^2
\]

latent进入 policy：

```python
self.latent_out_proj =
    nn.Linear(
        self.latent_dim,
        hidden_dim
    )
```

即：

\[
\boxed{
32\rightarrow512
}
\]

---

## ACT Transformer FFN

Official source:

https://github.com/tonyzhaozh/act/blob/main/detr/models/transformer.py

Encoder / Decoder均使用：

```python
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
```

ACT canonical config：

\[
d_{\text{model}}=512
\]

\[
d_{ff}=3200
\]

所以：

\[
\boxed{
512
\rightarrow
3200
\rightarrow
512
}
\]

中间必须有 nonlinear activation，

否则两层 affine transformations仍可合并为一个 affine transformation。

---

## Transformer Q/K/V Background

Ashish Vaswani et al.

**Attention Is All You Need.**  
NeurIPS 2017.

- https://arxiv.org/abs/1706.03762
- https://arxiv.org/pdf/1706.03762

Multi-Head Attention使用 learned projections：

\[
QW_i^Q
\]

\[
KW_i^K
\]

\[
VW_i^V
\]

以及最终：

\[
W^O
\]

这些 projection从神经网络计算角度都属于：

> learned linear/affine feature transformations。

它们本身并不“知道” Query、Key、Value语义；

这些角色由 Attention computation定义。

---

## 本文知识连接

### 数学

- Vector
- Matrix
- Matrix Multiplication
- [Dot Product](./dot-product.md)
- Linear Transformation
- Affine Transformation
- Rank
- Hyperplane
- Jacobian
- Gradient

### Deep Learning

- [MLP](./mlp.md)
- [Feed-Forward Network](./feed-forward-network.md)
- [ReLU](./relu.md)
- Xavier Initialization
- Embedding

### Transformer

- [Q / K / V](./qkv.md)
- [Attention](./attention.md)
- [Multi-Head Attention](./multi-head-attention.md)
- [Transformer Encoder](./transformer-encoder.md)
- [Transformer Decoder](./transformer-decoder.md)

### Generative Modeling

- [VAE](../generative-models/vae.md)
- [CVAE](../generative-models/cvae.md)
- [Reparameterization Trick](../generative-models/reparameterization-trick.md)

### Robot Learning

- [ACT Architecture](../robot-learning/act/architecture.md)
- [CVAE in ACT](../robot-learning/act/cvae-in-act.md)
- [ACT Training](../robot-learning/act/training.md)
- [ACT Complete Data Flow](../robot-learning/act/complete-data-flow.md)

### 下一步

- [ReLU](./relu.md)
