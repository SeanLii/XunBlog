---
title: "Backpropagation：Loss 到底怎样一路传回第一层？"
description: "从 computational graph 和 chain rule 出发，严格理解 backpropagation、upstream gradient、local derivative、Jacobian/VJP、gradient accumulation 与 PyTorch autograd；再完整映射 ACT 的 L1 + βKL 梯度如何流经 action head、Transformer、ResNet、reparameterization 和 CVAE encoder。"
status: reviewed
pageType: concept
canonical: /deep-learning/backpropagation
updated: "2026-09-15"
---

# Backpropagation：Loss 到底怎样一路传回第一层？

在上一篇 [MLP](./mlp.md) 里，我们第一次看到了完整的训练闭环：

\[
\boxed{
Forward
\rightarrow
Prediction
\rightarrow
Loss
\rightarrow
Backward
\rightarrow
Optimizer
}
\]

我们也手算过一个非常简单的例子：

\[
x
\rightarrow
w_1
\rightarrow
ReLU
\rightarrow
w_2
\rightarrow
\hat y
\rightarrow
L
\]

然后利用 Chain Rule：

\[
\frac{\partial L}{\partial w_1}
=
\frac{\partial L}{\partial \hat y}
\frac{\partial \hat y}{\partial h}
\frac{\partial h}{\partial z}
\frac{\partial z}{\partial w_1}
\]

把最终 Loss 对第一层参数的影响算出来。

但是这只是最小例子。

真正的 ACT 训练图里同时有：

- ResNet image backbone；
- joint projection；
- CVAE Transformer Encoder；
- \(\mu\)；
- \(\log\sigma^2\)；
- reparameterization；
- latent projection；
- Policy Transformer Encoder；
- Decoder；
- action head；
- L1 reconstruction loss；
- KL loss；
- residual branches；
- shared parameters；
- masking；
- Dropout；
- LayerNorm。

最后代码只有一句：

```python
loss.backward()
```

然后 PyTorch 就能得到：

```text
几百万、几千万个参数各自的 gradient
```

它到底是怎么做到的？

Backpropagation最容易被一句不够准确的话概括：

> “把误差从后往前传。”

这句话可以作为很早期的直觉，

但如果一直停在这里，就会导致很多误解：

- 难道把 Loss 数字原样传回去了？
- 一个参数离 Loss 很远，怎么知道自己应该变多少？
- 一个节点被两条 branch 同时使用时，gradient怎么办？
- 为什么 Residual Connection 的 gradient 会相加？
- 为什么 `detach()` 可以切断 gradient？
- 为什么 `torch.no_grad()` 和 `model.eval()` 完全不是一回事？
- 为什么 `.backward()` 默认只能直接作用于 scalar loss？
- 为什么 Autograd 不需要真的构造巨大的完整 Jacobian？
- ACT 的 KL Loss 为什么不会直接训练 Action Head？
- ACT 的 L1 Loss 又为什么可以穿过随机采样 \(z\) 回到 \(\mu\) 和 \(\log\sigma^2\)？
- 一个 module明明存在于模型中，为什么有时却没有 gradient？

这一篇的目标就是：

> **把 Backpropagation 从“误差倒着走”的故事，变成真正严谨的 Computational Graph + Chain Rule + Reverse-Mode Automatic Differentiation。**

---

# 1. 最先纠正一句话：Backward 传的不是“Loss 数字”

假设：

\[
L=3.7
\]

Backprop 并不是把：

\[
3.7
\]

复制给上一层，

再复制给再上一层。

真正向后传播的是：

\[
\boxed{
\text{derivative information}
}
\]

更具体：

> 当前 Loss 对某个中间变量的敏感度。

例如：

\[
\frac{\partial L}{\partial h}
\]

它回答：

> **如果 \(h\) 增加一个非常小的量，最终 \(L\) 会怎样变化？**

---

# 2. Gradient 是“敏感度”，不是“误差值”

假设：

\[
\frac{\partial L}{\partial h}=5
\]

意思是局部上：

\[
h\rightarrow h+\Delta h
\]

会造成：

\[
L
\rightarrow
L+5\Delta h
\]

近似成立。

如果：

\[
\frac{\partial L}{\partial h}=-2
\]

则增加一点：

\[
h
\]

反而会让：

\[
L
\]

下降。

所以 gradient真正表达的是：

\[
\boxed{
\text{Loss 对变量的局部变化率}
}
\]

---

# 3. Backprop 的真正目标

训练一个模型有参数：

\[
\theta
\]

Loss：

\[
L(\theta)
\]

Optimizer真正需要的是：

\[
\boxed{
\nabla_\theta L
}
\]

即：

\[
[
\frac{\partial L}{\partial\theta_1},
\frac{\partial L}{\partial\theta_2},
\ldots
]
\]

Backpropagation就是：

> **高效计算这些 gradients 的算法。**

它不是 optimizer本身。

---

# 4. Backprop 和 Optimizer 再严格区分

## Backpropagation

计算：

\[
\boxed{
\nabla_\theta L
}
\]

PyTorch：

```python
loss.backward()
```

---

## Optimizer

使用这些 gradients修改参数。

最简单 Gradient Descent：

\[
\boxed{
\theta
\leftarrow
\theta-\eta\nabla_\theta L
}
\]

PyTorch：

```python
optimizer.step()
```

所以：

\[
\boxed{
backward
\neq
parameter\ update
}
\]

---

# 5. 为什么还要 `zero_grad()`？

PyTorch默认会：

> **累积 leaf parameter 的 `.grad`。**

因此标准训练循环：

```python
optimizer.zero_grad()

loss.backward()

optimizer.step()
```

ACT官方训练代码也是这个模式：

```python
optimizer.zero_grad()

forward_dict = forward_pass(
    data,
    policy
)

loss = forward_dict['loss']

loss.backward()

optimizer.step()

optimizer.zero_grad()
```

---

# 6. 为什么 PyTorch 要默认 Accumulate？

因为有些计算图里：

> 同一个变量会从多条路径影响 Loss。

数学上这些 contribution本来就应该：

> 相加。

另外工程上还允许：

> gradient accumulation across micro-batches。

所以 accumulation不是一个奇怪 bug，

而是和 Chain Rule 的分支结构天然一致。

---

# 7. 从 Computational Graph 开始

Backprop最清楚的语言不是：

> “第1层、第2层、第3层”

而是：

\[
\boxed{
\text{Computational Graph}
}
\]

每一个节点：

> 一个变量 / tensor。

每一条边：

> 一个 operation dependency。

---

# 8. 最简单计算图

假设：

\[
a=wx
\]

\[
b=a+c
\]

\[
L=b^2
\]

图：

```text
w ──┐
    × ──→ a ──┐
x ──┘         + ──→ b ──→ square ──→ L
          c ──┘
```

Forward时：

> 从左往右算值。

Backward时：

> 从 \(L\) 出发，沿依赖关系反向算 derivatives。

---

# 9. Backward 不是“把 Forward 倒着执行一次”

Forward的 operation例如：

\[
b=a+c
\]

Backward不会执行：

> “减法把 b 还原成 a。”

它执行的是：

\[
\frac{\partial b}{\partial a}
\]

以及：

\[
\frac{\partial b}{\partial c}
\]

所以：

\[
\boxed{
\text{reverse graph traversal}
\neq
\text{inverse function computation}
}
\]

---

# 10. 很多 Forward Operation 根本不可逆

ReLU：

\[
-5,-1,0
\]

都可能变：

\[
0
\]

所以无法从：

\[
0
\]

恢复原始 input。

但仍然可以定义 backward derivative convention：

\[
\frac{\partial ReLU(x)}{\partial x}
\]

所以 Backprop根本不需要：

> invert forward operations。

---

# 11. Chain Rule 是整个 Backprop 的数学核心

设：

\[
y=g(x)
\]

\[
L=f(y)
\]

那么：

\[
L=f(g(x))
\]

Chain Rule：

\[
\boxed{
\frac{dL}{dx}
=
\frac{dL}{dy}
\frac{dy}{dx}
}
\]

这就是 Backprop最小核心。

---

# 12. 两个导数分别是什么？

\[
\frac{dL}{dy}
\]

表示：

> downstream Loss 对当前 output \(y\) 的敏感度。

通常称为：

> upstream gradient。

而：

\[
\frac{dy}{dx}
\]

是：

> 当前 operation自己的 local derivative。

两者相乘：

\[
\boxed{
\text{upstream gradient}
\times
\text{local derivative}
=
\text{gradient to previous node}
}
\]

---

# 13. Backprop 的核心口诀

对每个 operation：

```text
receive upstream gradient
↓
multiply by local derivative
↓
send gradient to inputs
```

这就是最重要的 mental model。

---

# 14. 一个极简单的例子

\[
y=3x
\]

\[
L=y^2
\]

假设：

\[
x=2
\]

Forward：

\[
y=6
\]

\[
L=36
\]

---

# 15. 从 L 回到 y

\[
L=y^2
\]

所以：

\[
\frac{dL}{dy}=2y=12
\]

这就是传到：

\[
y
\]

节点的 upstream gradient。

---

# 16. y = 3x 的 Local Derivative

\[
\frac{dy}{dx}=3
\]

所以：

\[
\frac{dL}{dx}
=
12\times3
\]

\[
\boxed{
=36
}
\]

---

# 17. 每个节点不需要知道整个网络

这是 Backprop非常漂亮的地方。

节点：

\[
y=3x
\]

不需要知道：

> 后面 Loss具体是什么复杂函数。

它只收到：

\[
\frac{dL}{dy}=12
\]

然后知道自己的 local derivative：

\[
3
\]

就能计算：

\[
\frac{dL}{dx}=36
\]

---

# 18. 这叫 Local Computation

每个 primitive只负责：

\[
\boxed{
\text{自己的局部 Jacobian / derivative rule}
}
\]

整个网络的 global gradient：

> 通过 Chain Rule自动组合出来。

这就是 automatic differentiation能够扩展到巨大模型的关键。

---

# 19. 再看一个 MLP 节点

Linear：

\[
y=Wx+b
\]

收到：

\[
g_y
=
\nabla_yL
\]

那么：

\[
\boxed{
\nabla_xL=W^\top g_y
}
\]

同时：

\[
\boxed{
\nabla_WL=g_yx^\top
}
\]

\[
\boxed{
\nabla_bL=g_y
}
\]

---

# 20. ReLU 节点

\[
h=ReLU(z)
\]

收到：

\[
g_h
=
\nabla_hL
\]

local derivative：

\[
1[z>0]
\]

所以：

\[
\boxed{
g_z
=
g_h\odot1[z>0]
}
\]

---

# 21. 两个 Local Rules 一接起来

```text
x
↓ Linear
z
↓ ReLU
h
↓ ...
L
```

Backward：

```text
∂L/∂h
↓ multiply ReLU local derivative
∂L/∂z
↓ Linear backward
∂L/∂x
∂L/∂W
∂L/∂b
```

这就是一层 MLP backward。

---

# 22. Computational Graph 为什么比“Layers”更一般？

因为现实模型会有：

- branch；
- skip connection；
- shared tensor；
- multiple losses；
- concatenation；
- residual addition。

这些不一定是简单：

```text
layer 1 → layer 2 → layer 3
```

链结构。

Graph language可以自然描述。

---

# 23. 一个 Branching Graph

假设：

\[
u=x^2
\]

然后：

\[
a=3u
\]

\[
b=u+5
\]

最后：

\[
L=a+b
\]

图：

```text
             ┌→ a = 3u ──┐
x → u=x² ────┤            ├→ L=a+b
             └→ b = u+5 ─┘
```

---

# 24. u 对 Loss 有两条路径

通过：

\[
u\rightarrow a\rightarrow L
\]

和：

\[
u\rightarrow b\rightarrow L
\]

所以：

\[
\frac{dL}{du}
\]

不能只算其中一条。

---

# 25. Gradient Contributions 要相加

一般：

\[
\boxed{
\frac{dL}{du}
=
\frac{\partial L}{\partial a}
\frac{\partial a}{\partial u}
+
\frac{\partial L}{\partial b}
\frac{\partial b}{\partial u}
}
\]

这就是：

> branch merge 时 gradient accumulation。

---

# 26. 代入这个例子

\[
L=a+b
\]

所以：

\[
\frac{\partial L}{\partial a}=1
\]

\[
\frac{\partial L}{\partial b}=1
\]

---

\[
a=3u
\]

所以：

\[
\frac{\partial a}{\partial u}=3
\]

---

\[
b=u+5
\]

所以：

\[
\frac{\partial b}{\partial u}=1
\]

因此：

\[
\boxed{
\frac{dL}{du}
=
3+1
=
4
}
\]

---

# 27. 再回到 x

\[
u=x^2
\]

所以：

\[
\frac{du}{dx}=2x
\]

于是：

\[
\boxed{
\frac{dL}{dx}
=
4(2x)
=
8x
}
\]

---

# 28. 这就是为什么 Gradient 会“相加”

不是 PyTorch随便决定：

> “多个 gradient 就加起来吧。”

而是 multivariable Chain Rule 本来就要求：

> 所有从该节点到 Loss 的路径 contribution求和。

---

# 29. Residual Connection 正是典型 Branch

\[
y=x+F(x)
\]

这里：

\[
x
\]

有两条路径到：

\[
y
\]

### Path 1

identity：

\[
x\rightarrow y
\]

### Path 2

\[
x\rightarrow F(x)\rightarrow y
\]

---

# 30. 所以 Gradient

\[
\boxed{
\frac{\partial y}{\partial x}
=
I+
J_F(x)
}
\]

然后：

\[
\boxed{
\nabla_xL
=
\left(
I+J_F
\right)^\top
\nabla_yL
}
\]

这正是 Residual Connection 的 backward结构。

---

# 31. 为什么 Gradient Accumulation 特别重要？

因为现代模型大量使用：

- residual；
- shared embeddings；
- branching losses；
- multi-task heads；
- skip paths。

一个 tensor可能：

> 被多个 downstream operations同时消费。

Backward必须：

> 等所有 downstream contributions都回来后相加。

---

# 32. 多个 Loss 也是同样

假设：

\[
L=L_1+\beta L_2
\]

那么：

\[
\boxed{
\nabla_\theta L
=
\nabla_\theta L_1
+
\beta\nabla_\theta L_2
}
\]

这对 ACT尤其重要。

---

# 33. ACT 的总 Loss

官方当前代码：

\[
\boxed{
L
=
L_{L1}
+
\beta L_{KL}
}
\]

代码：

```python
loss_dict['loss'] =
    loss_dict['l1']
    +
    loss_dict['kl']
    * self.kl_weight
```

canonical配置：

\[
\beta=10
\]

---

# 34. 因此一个共享参数可能同时收到两种 Gradient

如果某参数：

\[
\theta
\]

同时影响：

\[
L_{L1}
\]

和：

\[
L_{KL}
\]

那么：

\[
\boxed{
\frac{\partial L}{\partial\theta}
=
\frac{\partial L_{L1}}{\partial\theta}
+
\beta
\frac{\partial L_{KL}}{\partial\theta}
}
\]

Optimizer看到的：

> 是合成后的 gradient。

---

# 35. 但不是所有 ACT 参数都同时收到两条 Loss

关键是：

> **这个参数是否位于该 Loss 的可微依赖路径上。**

这就是计算图思维最重要的价值。

---

# 36. 现在进入 Vector 情况

神经网络中的节点通常不是 scalar，

而是：

\[
x\in\mathbb R^n
\]

\[
y=f(x)\in\mathbb R^m
\]

这时：

\[
\frac{\partial y}{\partial x}
\]

不再是一个数。

而是：

\[
\boxed{
Jacobian
}
\]

---

# 37. Jacobian 定义

\[
J_f
=
\frac{\partial y}{\partial x}
=
\begin{bmatrix}
\frac{\partial y_1}{\partial x_1}
&
\cdots
&
\frac{\partial y_1}{\partial x_n}
\\
\vdots
&
\ddots
&
\vdots
\\
\frac{\partial y_m}{\partial x_1}
&
\cdots
&
\frac{\partial y_m}{\partial x_n}
\end{bmatrix}
\]

shape：

\[
\boxed{
[m,n]
}
\]

---

# 38. 如果最终 Loss 是 Scalar

\[
L\in\mathbb R
\]

我们已经有：

\[
g_y
=
\nabla_yL
\in
\mathbb R^m
\]

那么：

\[
\boxed{
\nabla_xL
=
J_f(x)^\top
g_y
}
\]

---

# 39. 这就是 Reverse-Mode 的核心运算

它并不要求真的把：

\[
J_f
\]

整个矩阵 materialize出来。

只需要计算：

\[
\boxed{
J_f^\top g_y
}
\]

这就是 reverse-mode automatic differentiation 的关键效率来源。

---

# 40. VJP 到底是什么？

很多 autodiff文献用 row-vector cotangent notation：

\[
g_y^\top J_f
\]

称：

> **Vector-Jacobian Product (VJP)**。

如果我们用 column-vector gradient：

\[
J_f^\top g_y
\]

是同一个数学 contraction的转置表示。

所以看到：

- VJP；
- Jacobian-transpose-vector product；

不要以为是两种完全不同算法。

---

# 41. 为什么不直接构造整个 Jacobian？

假设：

\[
x
\]

有：

\[
1,000,000
\]

维，

\[
y
\]

也有：

\[
1,000,000
\]

维。

完整 Jacobian：

\[
10^{12}
\]

entries。

完全不可接受。

但 Backprop只需要：

> 当前 upstream gradient与 local Jacobian的乘积。

这样很多 operation有极其高效的 backward formula。

---

# 42. Linear 就是最好例子

\[
y=Wx+b
\]

Jacobian：

\[
J=W
\]

所以：

\[
J^\top g
=
W^\top g
\]

不需要另外构造任何 Jacobian。

---

# 43. ReLU 也一样

Jacobian：

\[
D=
diag(
1[z_i>0]
)
\]

但根本不需要构造巨大的 diagonal matrix。

直接：

\[
\boxed{
g_z
=
g_h
\odot
1[z>0]
}
\]

即可。

---

# 44. Softmax 也不需要显式 Jacobian

Softmax Jacobian：

\[
J
=
diag(p)-pp^\top
\]

但 autodiff实现可以直接计算：

\[
J^\top g
\]

而不是先分配：

\[
n\times n
\]

矩阵。

这就是 backward kernel设计的重要思想。

---

# 45. LayerNorm 也是同样

LayerNorm对一个512-D token的 Jacobian：

> 是 512×512 dense-coupled structure。

实际 backward不会笨拙地创建：

\[
512\times512
\]

矩阵再乘。

它使用推导好的 reduction公式：

> 直接计算 input gradients与参数 gradients。

---

# 46. Backprop 本质是“局部 VJP 的反向组合”

非常精确地说：

\[
\boxed{
\text{Backprop}
=
\text{reverse traversal}
+
\text{local VJP evaluations}
+
\text{gradient accumulation at branches}
}
\]

这比“误差往回传”精确得多。

---

# 47. 为什么 Reverse Mode 特别适合 Neural Network Training？

训练通常：

- 参数数量：
  \[
  P\gg1
  \]
- 最终 Loss：
  \[
  L\in\mathbb R
  \]

也就是说：

\[
\mathbb R^P
\rightarrow
\mathbb R
\]

我们想一次得到：

\[
\nabla_\theta L
\in\mathbb R^P
\]

Reverse mode特别适合：

> many inputs → few outputs，尤其 scalar output。

---

# 48. Forward-Mode 更适合什么直觉？

Forward mode更自然地传播：

> input direction对后续 outputs的变化。

如果：

- 输入维度很小；
- 输出维度很大；

可能更合适。

Neural network training恰好相反：

> 参数巨多，Loss很少。

所以 reverse mode成为主流。

---

# 49. Backprop 与 Reverse-Mode AD 什么关系？

现代语言里：

> Neural-network Backpropagation基本可以理解为在计算图上执行 reverse-mode automatic differentiation 的特定应用。

Backprop强调：

- neural network structure；
- parameter gradients；
- shared subexpressions。

Automatic differentiation是更一般的框架。

---

# 50. Backprop 不是 Symbolic Differentiation

Symbolic differentiation可能产生：

> 巨大代数表达式。

例如不断展开 chain rule。

Autodiff则：

> 在已经执行的 primitive graph上组合局部 derivative rules。

---

# 51. Backprop 也不是 Numerical Differentiation

Finite difference：

\[
\frac{\partial L}{\partial\theta_i}
\approx
\frac{
L(\theta_i+\epsilon)
-
L(\theta_i-\epsilon)
}{
2\epsilon
}
\]

每个参数都要额外 forward。

百万参数：

> 不现实。

而且还有：

- truncation error；
- floating-point error。

---

# 52. Finite Difference 主要用于 Gradient Check

小模型中可以比较：

\[
gradient_{autograd}
\]

和：

\[
gradient_{finite\ difference}
\]

确认自定义 backward是否正确。

但正式训练：

> 使用 autodiff/backprop。

---

# 53. 为什么 Forward 要保存一些东西？

Backward local derivative常需要：

> Forward时的 intermediate values。

例如 ReLU backward需要知道：

\[
z>0?
\]

Sigmoid backward可能需要：

\[
\sigma(z)
\]

Linear weight gradient需要：

\[
x
\]

---

# 54. 所以 Training 比 Pure Inference 更吃 Memory

Training forward不能所有 intermediate都立即丢掉。

因为 backward还需要它们。

这就是：

\[
\boxed{
\text{activation memory}
}
\]

的重要来源。

---

# 55. PyTorch 的 `save_for_backward`

自定义 `autograd.Function` 中，

forward可以：

```python
ctx.save_for_backward(...)
```

保存 backward需要的 tensors。

PyTorch也会为内置 operations管理相应的 saved tensors。

---

# 56. Memory-Time Tradeoff

可以选择：

> 保存更多 intermediate，backward直接使用。

或者：

> 少保存，在 backward时重新算。

后者就是 gradient/checkpointing 思想的重要基础。

---

# 57. Gradient Checkpointing

对很深网络，

不保存所有 activations，

只保留部分 checkpoint。

Backward时：

> 重新 forward某些区间恢复 intermediate。

于是：

\[
\boxed{
\text{more compute}
\leftrightarrow
\text{less memory}
}
\]

---

# 58. 这和 Backprop 原理不冲突

Chain Rule完全一样。

只是选择：

> local derivative需要的 forward values，是存起来还是重新算。

---

# 59. PyTorch Autograd 如何记录 Graph？

当某个 operation的至少一个输入：

```python
requires_grad=True
```

并处于 grad mode，

PyTorch会记录：

> 生成该 tensor的 operation history。

结果 tensor通常有：

```python
grad_fn
```

指向 backward graph中的 function节点。

---

# 60. 什么是 Leaf Tensor？

简化理解：

> 用户直接创建、不是由另一个可微 operation生成的 tensor。

Model parameters：

```python
nn.Parameter
```

通常是典型 leaf tensors。

---

# 61. 参数为什么默认 `requires_grad=True`？

`nn.Parameter`注册到 module后，

通常需要被 optimizer训练。

所以模型 weights：

\[
W,b
\]

是 autograd需要追踪的 leaf parameters。

---

# 62. Non-Leaf Tensor 是什么？

例如：

```python
z = x @ W.T + b
```

如果：

\[
W
\]

requires grad，

那么：

\[
z
\]

是 computation产生的中间 tensor。

它会有：

```python
grad_fn
```

所以是 non-leaf。

---

# 63. `.grad` 默认存在哪里？

PyTorch backward时，

默认主要把 gradient累积到：

> `requires_grad=True` 的 leaf tensors。

也就是典型 model parameters：

```python
W.grad
b.grad
```

---

# 64. 为什么中间 Activation 的 `.grad` 常是 None？

并不是因为：

> Backprop没有计算它的 gradient。

恰恰相反：

> backward通常必须计算中间梯度才能继续往前。

只是 PyTorch默认：

> 不把每个 non-leaf intermediate的 gradient永久保存在 `.grad` 字段。

这样节省 memory。

---

# 65. 想查看 Non-Leaf Gradient 怎么办？

可以：

```python
h.retain_grad()
```

然后：

```python
loss.backward()
```

再查看：

```python
h.grad
```

这对学习/debugging很有用。

---

# 66. `requires_grad=True` 到底是什么意思？

表示：

> 希望 autograd记录相关操作，以便之后能计算到这些 leaf tensors的 gradient。

但注意：

> `requires_grad=True` 不等于 `.grad` 当前已经有值。

只有 backward后：

> 才可能被填充。

---

# 67. `grad_fn` 又是什么？

它表示：

> 这个 non-leaf tensor是通过什么 recorded operation产生的。

例如：

```python
y = x * 2
```

可能看到某类 multiplication backward function。

这可以帮助理解 dynamic graph。

---

# 68. PyTorch 的 Graph 是 Dynamic 的

PyTorch eager autograd：

> Forward执行什么 operations，就记录什么 graph。

下一次 forward：

> 可以重新构建新的 graph。

所以 Python控制流：

```python
if ...
for ...
```

可以根据输入走不同路径。

---

# 69. 为什么一次 `backward()` 后 Graph 常被释放？

默认情况下，

为节省 memory：

> backward需要的 graph buffers会被释放。

如果想在同一个 graph上再次 backward，

通常需要：

```python
retain_graph=True
```

---

# 70. 为什么正常 Training 不需要 Retain Graph？

因为下一个 batch会：

> 重新做一次 forward，

建立新的 computation graph。

标准训练：

```python
forward
backward
step

forward
backward
step
```

每轮 graph都是新的。

---

# 71. `retain_graph=True` 不应该随便开

如果一直保留 graph：

> memory可能持续增加。

只有确实需要：

- multiple backward passes on same graph；
- 特定 higher-order / multi-loss流程；

才使用。

---

# 72. Multiple Loss 一定需要 Retain Graph 吗？

不一定。

最简单：

\[
L=L_1+L_2
\]

直接：

```python
loss = loss1 + loss2
loss.backward()
```

只需一次 backward。

---

# 73. 如果分开 Backward

```python
loss1.backward(retain_graph=True)
loss2.backward()
```

也可以在某些情况下使用。

Gradient会：

> accumulate。

但如果没有特殊理由，

先把 losses加起来：

> 通常更直接。

---

# 74. ACT 正是先组成一个 Total Loss

```python
loss =
    l1
    +
    kl_weight * kl
```

然后：

```python
loss.backward()
```

所以 Autograd一次反向 traversal：

> 自动把两条 loss gradient贡献合并。

---

# 75. `detach()` 是什么？

如果：

```python
y = x.detach()
```

得到：

\[
y
\]

与：

\[
x
\]

共享/关联数值语义，

但：

> 从 Autograd graph角度切断了产生 x 的历史。

后续使用：

\[
y
\]

不会把 gradient传回：

> x之前的 graph。

---

# 76. 计算图上的直觉

原本：

```text
a → b → c → L
```

如果：

```python
b_detached = b.detach()
```

然后：

```text
b_detached → c → L
```

那么 backward到：

\[
b_{\text{detached}}
\]

就不会继续穿回：

\[
a
\]

---

# 77. `detach()` 并不把数值设成0

这是常见误解。

它改变的是：

> gradient history。

不是：

> tensor value。

---

# 78. Stop-Gradient 的概念

很多论文写：

\[
sg(x)
\]

或：

> stop gradient。

PyTorch常通过：

```python
detach()
```

实现类似效果。

数值 forward继续使用，

但 backward：

> 不穿过这条路径。

---

# 79. `torch.no_grad()` 又是什么？

在：

```python
with torch.no_grad():
    ...
```

内部进行的 operations：

> 不被 autograd记录进 backward graph。

这常用于：

- inference；
- parameter update；
- 不需要梯度的预处理。

---

# 80. `no_grad()` 和 `detach()` 的区别

### `detach()`

针对：

> 一个 tensor / graph edge。

---

### `no_grad()`

针对：

> 一整个代码区域中的 operations。

二者都可用于阻止 gradient tracking，

但作用方式不同。

---

# 81. `model.eval()` 又不是这两个

这一点必须再强调。

```python
model.eval()
```

改变：

- Dropout；
- BatchNorm；

等 module的 train/eval behavior。

它：

\[
\boxed{
\text{不会自动关闭 autograd}
}
\]

---

# 82. 所以标准 Inference 常写

```python
model.eval()

with torch.no_grad():
    y = model(x)
```

或：

```python
torch.inference_mode()
```

分别处理：

- module behavior；
- gradient tracking。

---

# 83. ACT Validation 使用 `torch.inference_mode()`

官方 training loop：

```python
with torch.inference_mode():
    policy.eval()
    ...
```

因此 validation时：

1. Dropout关闭；
2. 不构建训练 autograd graph。

之后 training：

```python
policy.train()
```

恢复 Dropout等 train behavior。

---

# 84. 为什么 Validation 不需要 Backward？

因为 validation只想估计：

> 当前模型泛化 loss。

不更新 parameters。

所以没有必要保留：

> backward graph。

---

# 85. Non-Scalar Tensor 为什么不能直接 `.backward()`？

假设：

\[
y
\in
\mathbb R^m
\]

而不是 scalar。

“\(y\) 对 \(x\) 的 gradient”是什么？

实际上是：

\[
\boxed{
Jacobian
}
\]

不是单一 gradient vector。

所以你必须说明：

> 你想把 output directions怎样组合成一个 scalar-like objective。

---

# 86. PyTorch 需要一个 `gradient` Argument

如果：

```python
y.backward(v)
```

其中：

\[
v
\]

shape与：

\[
y
\]

相同，

PyTorch计算的本质是：

\[
\boxed{
J_y(x)^\top v
}
\]

在 column-gradient notation中。

---

# 87. Scalar Loss 为什么最方便？

如果：

\[
L
\]

只有一个元素，

初始 seed gradient自然：

\[
\frac{\partial L}{\partial L}=1
\]

所以：

```python
loss.backward()
```

无需额外指定。

Backward从：

\[
\boxed{
1
}
\]

开始向图中传播。

---

# 88. Backprop 的真正起点就是 1

这是一个很漂亮的事实。

因为：

\[
\boxed{
\frac{dL}{dL}=1
}
\]

所以 root节点的 upstream gradient：

\[
1
\]

然后每个 operation：

> 用 local derivative把它继续传下去。

---

# 89. 一个完整 Scalar Graph

```text
x
↓ f
a
↓ g
b
↓ h
L
```

Backward seed：

\[
\bar L=1
\]

其中：

\[
\bar b
=
\frac{\partial L}{\partial b}
\]

\[
\bar a
=
\frac{\partial L}{\partial a}
\]

\[
\bar x
=
\frac{\partial L}{\partial x}
\]

有些 autodiff文献用：

\[
\bar x
\]

表示 adjoint/cotangent。

---

# 90. 为什么叫 Upstream Gradient？

在 backward方向看，

Loss在“上游”。

当前节点收到：

> 后续所有计算已经汇总出的 Loss sensitivity。

所以叫：

> upstream gradient。

但不同教材的“upstream/downstream”措辞有时方向感不同，

最安全还是看：

\[
\boxed{
\frac{\partial L}{\partial \text{current output}}
}
\]

---

# 91. Local Gradient 是什么？

某 operation：

\[
y=f(x)
\]

local derivative：

\[
\boxed{
\frac{\partial y}{\partial x}
}
\]

它只描述：

> 当前 operation自身。

与最终 Loss是什么无关。

---

# 92. Global Gradient 是什么？

\[
\boxed{
\frac{\partial L}{\partial x}
}
\]

它整合：

> x到Loss之间所有后续 paths。

Backprop就是：

> 用 local rules高效构造 global gradients。

---

# 93. 为什么 Branch 会导致求和？

如果：

\[
x
\]

被：

\[
f(x)
\]

和：

\[
g(x)
\]

同时使用，

那么 \(x\) 对 Loss 的总影响：

> 是沿所有 downstream usages的 contribution总和。

Autograd engine会自动完成这个 accumulation。

---

# 94. 一个 Parameter 被重复使用也一样

例如 RNN weight在多个 time steps共享。

同一个：

\[
W
\]

在：

\[
t=1,2,\ldots,T
\]

反复出现。

总 gradient：

\[
\boxed{
\frac{\partial L}{\partial W}
=
\sum_t
\left.
\frac{\partial L}{\partial W}
\right|_t
}
\]

这就是 parameter sharing 下的梯度累积。

---

# 95. Transformer Position-Wise MLP 也共享参数

同一个：

\[
W_1
\]

被所有：

\[
N
\]

tokens使用。

所以：

\[
\nabla_{W_1}L
\]

会累积来自：

> 所有 token positions、所有 batch samples

的 contribution。

---

# 96. 这就是“参数共享”怎样被训练

共享不是：

> 只看一个 token。

而是：

> 一个 weight同时服务很多 positions，因此收到这些 positions共同的 training signal。

---

# 97. Batch Gradient 也是求和/平均形式

如果：

\[
L
=
\frac1B
\sum_{i=1}^{B}
L_i
\]

那么：

\[
\boxed{
\nabla_\theta L
=
\frac1B
\sum_i
\nabla_\theta L_i
}
\]

所以每个 mini-batch update：

> 汇总多个 samples的 gradient。

---

# 98. Mean vs Sum Loss 会改变 Gradient Scale

如果：

\[
L_{sum}
=
\sum_iL_i
\]

而：

\[
L_{mean}
=
\frac1B\sum_iL_i
\]

则：

\[
\nabla L_{mean}
=
\frac1B
\nabla L_{sum}
\]

方向相同，

scale不同。

这会和：

- learning rate；
- batch size；

产生关系。

---

# 99. ACT 的 L1 Masking 有一个实现细节

官方：

```python
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
```

所以：

> padded elements乘0。

然后对整个 tensor：

> `.mean()`。

---

# 100. 被 Mask 成 0 的 Element Gradient

对于某 padded position：

\[
mask=0
\]

则对应：

\[
0\times |a-\hat a|
\]

对：

\[
\hat a
\]

的 gradient：

\[
0
\]

所以 padded targets：

> 不通过 L1给 action prediction传 gradient。

---

# 101. 但 Denominator 仍然是 Entire Tensor Mean

由于代码先 mask再：

```python
.mean()
```

mean denominator仍按整个 tensor元素数量计算，

而不是只除：

> non-padding count。

这会让有效 L1 gradient scale：

> 随 padding比例有所变化。

这是实现细节，

不是 Backprop算法本身。

---

# 102. L1 的 Local Derivative

单个 element：

\[
\ell=
|\hat a-a|
\]

如果：

\[
\hat a>a
\]

则：

\[
\frac{\partial\ell}{\partial\hat a}=1
\]

如果：

\[
\hat a<a
\]

则：

\[
-1
\]

在完全相等点不可导，

framework采用subgradient convention。

所以 L1给 action head的是：

> signed error direction，

不是 MSE那样与误差 magnitude成比例。

---

# 103. 现在把 ACT 训练 Graph 画出来

简化：

```text
demonstration actions ───────────────────────────────┐
        │                                            │
        ▼                                            │
CVAE Encoder                                         │
        │                                            │
        ▼                                            │
      h_CLS                                          │
        │                                            │
        ▼                                            │
 latent_proj                                         │
        │                                            │
   ┌────┴────┐                                       │
   ▼         ▼                                       │
   μ       logvar                                    │
   │         │                                       │
   └────┬────┘                                       │
        ▼                                            │
reparameterization                                   │
        │                                            │
        ▼                                            │
        z                                            │
        │                                            │
        ▼                                            │
latent_out_proj                                      │
        │                                            │
        ├──────────────┐                             │
        │              │                             │
images → ResNet ───────┤                             │
qpos → projection ─────┤                             │
                       ▼                             │
             Policy Transformer                     │
                       │                             │
                       ▼                             │
                   action_head                       │
                       │                             │
                       ▼                             │
                     a_hat                           │
                       │                             │
                       └───────── compare ────────────┘
                                  │
                                  ▼
                                 L1

μ, logvar
   │
   ▼
  KL
   │
   ▼
 L_KL

Total:
L = L1 + β L_KL
```

---

# 104. `loss.backward()` 从哪里开始？

Total scalar：

\[
L
=
L_{L1}
+
\beta L_{KL}
\]

seed：

\[
\frac{\partial L}{\partial L}=1
\]

Addition节点把 gradient分别送给：

\[
L_{L1}
\]

和：

\[
L_{KL}
\]

---

# 105. 对 L1 Branch

\[
\frac{\partial L}{\partial L_{L1}}
=
1
\]

所以 L1 branch gradient未额外缩放。

---

# 106. 对 KL Branch

\[
\frac{\partial L}{\partial L_{KL}}
=
\beta
\]

所以 KL产生的所有 upstream gradients：

> 在进入共享上游 graph前被 \(\beta\) 缩放。

ACT canonical：

\[
\beta=10
\]

因此：

> KL gradient contribution被乘10。

但最终影响大小仍取决于实际 derivative magnitude。

---

# 107. β=10 不代表 KL Gradient 永远比 L1 大10倍

因为：

\[
\nabla L
=
\nabla L_1
+
10\nabla L_{KL}
\]

如果：

\[
\|\nabla L_{KL}\|
\]

本来很小，

乘10后仍可能比：

\[
\nabla L_1
\]

小。

所以 Loss coefficient只是：

> 线性缩放该 branch gradient。

不是最终 dominance保证。

---

# 108. L1 Gradient 首先训练谁？

\[
L1
\]

直接依赖：

\[
a_{hat}
\]

而：

\[
a_{hat}
=
action\_head(hs)
\]

所以最直接收到 gradient的是：

> `action_head`。

---

# 109. 然后 Gradient 进入 Decoder Hidden State

Linear backward：

\[
\nabla_{hs}L_1
=
W_{action}^\top
\nabla_{a_{hat}}L_1
\]

所以 decoder output：

\[
hs
\]

收到 gradient。

---

# 110. 然后穿过 Transformer Decoder

Decoder内部有：

- FFN；
- Cross-Attention；
- Self-Attention；
- Residual；
- LayerNorm。

每个 operation都有自己的 backward rule。

所以 gradient继续传播到：

- decoder parameters；
- action query embeddings；
- encoder memory；
- 等。

---

# 111. Cross-Attention 让 L1 进入 Policy Encoder

Decoder Cross-Attention读取：

\[
memory
\]

作为 K/V。

因此：

\[
a_{hat}
\]

依赖：

\[
memory
\]

所以：

\[
\boxed{
L1
\rightarrow
Decoder
\rightarrow
CrossAttention
\rightarrow
Policy\ Encoder
}
\]

存在可微路径。

---

# 112. Policy Encoder 再连接到 Images

Policy Encoder memory依赖：

> visual backbone features。

所以 L1 gradient继续回到：

- `input_proj`；
- ResNet backbone；

如果这些 parameters是 trainable。

---

# 113. L1 也回到 qpos Projection

Policy输入里有：

> current joint state token。

所以：

\[
L1
\]

也会训练：

```python
input_proj_robot_state
```

以及其后的 shared Transformer pathways。

---

# 114. L1 还会进入 Latent Input

Policy Transformer同时依赖：

\[
latent\_input
=
latent\_out\_proj(z)
\]

所以：

\[
L1
\]

会反向进入：

```python
latent_out_proj
```

---

# 115. 然后遇到随机采样 z

Training：

\[
z
=
\mu+\sigma\epsilon
\]

其中：

\[
\sigma
=
e^{\frac12\log\sigma^2}
\]

\[
\epsilon
\sim\mathcal N(0,I)
\]

在本次 forward中：

\[
\epsilon
\]

被当作采样得到的固定随机 tensor。

---

# 116. 为什么 L1 Gradient 能穿过 Sampling？

因为 reparameterization把：

\[
z
\]

写成：

> 对 \(\mu,\logvar\) 的 differentiable deterministic function，条件是当前 \(\epsilon\) 固定。

所以：

\[
\frac{\partial z}{\partial\mu}=1
\]

---

# 117. 对 logvar 的 Gradient

\[
\sigma
=
e^{\frac12\logvar}
\]

\[
z=
\mu+\sigma\epsilon
\]

所以：

\[
\frac{\partial z}{\partial\logvar}
=
\epsilon
\frac{\partial\sigma}{\partial\logvar}
\]

而：

\[
\frac{\partial\sigma}{\partial\logvar}
=
\frac12
e^{\frac12\logvar}
=
\frac12\sigma
\]

因此：

\[
\boxed{
\frac{\partial z}{\partial\logvar}
=
\frac12\sigma\epsilon
}
\]

---

# 118. 所以 Reconstruction/L1 直接训练 μ 和 logvar

通过：

\[
L1
\rightarrow
z
\rightarrow
\mu,\logvar
\]

CVAE encoder不仅收到：

> KL gradient，

也收到：

> reconstruction gradient。

这是理解 VAE/CVAE训练最重要的点之一。

---

# 119. 如果直接做不可重参数化的采样会怎样？

若 sampling operation被当成：

> parameter-dependent random black box，

普通 pathwise backprop无法直接把 sample derivative传给 distribution parameters。

Reparameterization正是：

> 把随机源移到独立的 \(\epsilon\) 上。

我们在 [Reparameterization Trick](../generative-models/reparameterization-trick.md) 已详细讲过。

---

# 120. L1 继续进入 latent_proj

\[
\mu,\logvar
\]

来自：

\[
latent\_info
=
latent\_proj(h_{CLS})
\]

所以：

> L1 gradient继续训练 `latent_proj`。

---

# 121. 再进入 CVAE Transformer Encoder

\[
h_{CLS}
\]

来自：

> CVAE Transformer Encoder。

因此 L1也会通过 latent path：

> 更新 CVAE Encoder parameters。

这很重要：

\[
\boxed{
\text{CVAE Encoder不是只靠 KL 学习}
}
\]

它同时要帮助 reconstruction。

---

# 122. L1 最后还能回到 Demonstration Action Projection

CVAE Encoder输入包括：

- current qpos；
- target action chunk。

所以通过 encoder graph，

L1 gradient会更新：

```python
encoder_action_proj
encoder_joint_proj
```

和 CVAE Transformer parameters。

---

# 123. 但 Demonstration Action Tensor 本身需要 Gradient 吗？

通常：

> 不需要。

Training target：

\[
actions
\]

是数据，不是可训练 parameter。

它通常：

```python
requires_grad=False
```

所以 Autograd会计算参数所需中间 gradients，

但不会把 optimizer gradient存到：

> dataset target。

---

# 124. Input 不需要 requires_grad，不代表 Model Parameters 无法训练

这是常见误区。

即使：

```python
x.requires_grad == False
```

只要：

\[
W.requires\_grad=True
\]

operation：

\[
y=Wx
\]

仍会记录足够信息来计算：

\[
\frac{\partial L}{\partial W}
\]

因为我们需要训练：

> W，

不需要训练 x。

---

# 125. KL Branch 又走哪里？

KL只直接依赖：

\[
\mu,\logvar
\]

官方：

\[
KL
=
-\frac12
\sum_j
\left(
1+\logvar_j
-\mu_j^2
-e^{\logvar_j}
\right)
\]

再做 batch reduction。

---

# 126. KL 对 μ 的 Derivative

单维：

\[
KL_j
=
-\frac12
(
1+\logvar_j-\mu_j^2-e^{\logvar_j}
)
\]

所以：

\[
\boxed{
\frac{\partial KL_j}{\partial\mu_j}
=
\mu_j
}
\]

---

# 127. KL 对 logvar

\[
\boxed{
\frac{\partial KL_j}{\partial\logvar_j}
=
\frac12
(
e^{\logvar_j}-1
)
}
\]

因此：

- \(\mu\)偏离0：
  > gradient拉回；
- variance偏离1：
  > gradient相应调整 logvar。

---

# 128. β 再把 KL Gradient 缩放

总 loss：

\[
L=L1+\beta KL
\]

所以：

\[
\boxed{
\left.
\frac{\partial L}{\partial\mu}
\right|_{KL}
=
\beta\mu
}
\]

以及：

\[
\boxed{
\left.
\frac{\partial L}{\partial\logvar}
\right|_{KL}
=
\frac{\beta}{2}
(
e^{\logvar}-1
)
}
\]

忽略 reduction scale。

---

# 129. μ/logvar 收到两条 Branch

所以真正：

\[
\boxed{
\nabla_\mu L
=
\nabla_\mu L1
+
\beta\nabla_\mu KL
}
\]

\[
\boxed{
\nabla_{\logvar}L
=
\nabla_{\logvar}L1
+
\beta\nabla_{\logvar}KL
}
\]

这就是 CVAE training的 tradeoff在 gradient层面的真实形式。

---

# 130. Reconstruction Branch 想让 z 编码有用信息

如果某种 latent variation能帮助：

\[
a_{hat}
\]

更接近 demonstration actions，

L1会推动：

> \(\mu,\logvar\) 产生更有利于 reconstruction的 latent samples。

---

# 131. KL Branch 想让 Posterior 靠近 Prior

KL则鼓励：

\[
q(z|x,y)
\]

靠近：

\[
\mathcal N(0,I)
\]

所以二者可能存在：

> gradient tension。

这就是 \(\beta\) 需要权衡的原因。

---

# 132. 这比“KL 让 μ=0, var=1”更准确

因为每次参数更新看到的是真正总 gradient：

\[
\boxed{
\nabla L_{recon}
+
\beta\nabla L_{KL}
}
\]

不是：

> KL单独决定 posterior。

所以某个样本：

\[
\mu\neq0
\]

完全正常。

---

# 133. KL Branch 会训练 Action Head 吗？

**不会直接训练。**

原因：

\[
KL
\]

只依赖：

\[
\mu,\logvar
\]

而它们在计算图上位于：

> Action Head 之前的另一 branch。

不存在：

\[
KL\rightarrow action\_head
\]

依赖路径。

所以：

\[
\boxed{
\frac{\partial KL}{\partial W_{action}}=0
}
\]

---

# 134. KL 会训练 Policy Decoder 吗？

同样：

> 不通过 KL branch。

KL在：

\[
\mu,\logvar
\]

处就结束。

Policy decoder是：

> \(z\) 之后的 downstream reconstruction branch。

所以 KL不依赖 decoder output。

---

# 135. 但 Decoder 可以间接影响未来 Encoder Learning 吗？

在同一个 instantaneous gradient graph中：

> KL本身不经过 decoder。

但总训练动态中：

- L1更新 decoder；
- L1也更新 encoder；
- KL更新 encoder；

这些参数长期共同变化。

所以“没有直接 gradient path”：

\[
\neq
\]

“训练过程中完全无相互影响”。

---

# 136. 计算图 Path 与长期 Optimization Coupling 要区分

当前 backward：

> 看当前图的 derivative paths。

多个 step之后：

> 参数彼此通过共享 loss/data dynamics共同变化。

这是两个不同层次。

---

# 137. ACT 的一个极好的 Graph 例子：`is_pad_head`

Model forward计算：

```python
a_hat =
    self.action_head(hs)

is_pad_hat =
    self.is_pad_head(hs)
```

两者都存在。

---

# 138. 但当前 Official Policy Loss 用了谁？

`policy.py` 中：

```python
a_hat, is_pad_hat, (mu, logvar) =
    self.model(...)
```

然后构造：

```python
l1 = ...
kl = ...
loss =
    l1 + kl_weight * kl
```

并没有把：

```python
is_pad_hat
```

放进 total loss。

---

# 139. 因此 `is_pad_head` 会怎样？

如果：

\[
is\_pad\_hat
\]

不再被其他 loss使用，

那么 total loss：

\[
L
\]

不依赖：

\[
W_{is\_pad}
\]

所以：

\[
\boxed{
\frac{\partial L}{\partial W_{is\_pad}}
=
0
}
\]

更具体在 Autograd中：

> 它可能没有收到这次 loss backward 的 gradient contribution。

---

# 140. 这说明“Module 存在”不等于“Module 被训练”

真正条件：

\[
\boxed{
\text{parameter must lie on a differentiable path to the loss}
}
\]

如果 output被计算出来但最终：

> 没参与 loss，

那一支 graph对本次 objective没有 gradient。

---

# 141. 同理：打印一个 Tensor 不会让它被训练

只有：

> Loss function依赖它

才会产生 gradient路径。

这也是设计 auxiliary heads时必须记住的。

---

# 142. 如果给 `is_pad_hat` 加 BCE Loss 呢？

例如：

\[
L
=
L1
+
\beta KL
+
\lambda L_{pad}
\]

其中：

\[
L_{pad}
=
BCE(
is\_pad\_hat,
is\_pad
)
\]

那么：

`is_pad_head` 会收到：

\[
\lambda\nabla L_{pad}
\]

并且这条 loss还会继续回到：

> shared decoder hidden states。

---

# 143. Shared Trunk 会收到所有 Head Loss 的和

如果：

\[
hs
\]

同时送到：

- action head；
- pad head；

且两边都有 loss，

那么：

\[
\boxed{
\nabla_{hs}L
=
\nabla_{hs}L_{action}
+
\nabla_{hs}L_{pad}
}
\]

这就是 multi-task learning最基本的 gradient merge。

---

# 144. Gradient Conflict 是什么？

如果两条 loss希望 shared parameter往不同方向走，

可能出现：

\[
\nabla L_1^\top
\nabla L_2<0
\]

即 gradient方向有冲突。

Multi-objective / multi-task optimization会研究如何处理这种情况。

ACT当前主要的两条 objective：

> reconstruction 与 KL

在 latent encoder参数上就可能存在这种 tradeoff。

---

# 145. 为什么 KL Weight 会影响 Representation？

因为它直接改变：

\[
\nabla_{\theta_{encoder}} L
\]

中 KL contribution比例。

所以：

\[
\beta
\]

不是只改变打印出来的 loss数值。

它改变：

> 实际 parameter update direction。

---

# 146. Loss Value Scale 与 Gradient Scale 不完全等同

一个 loss项数值很大：

> 不一定 gradient norm也同样大。

反之亦然。

训练真正推动参数的是：

\[
\boxed{
\nabla_\theta L
}
\]

所以分析 multi-loss模型时，

gradient norms有时比 raw loss values更有信息。

---

# 147. 为什么 `loss.backward()` 不会训练 Input Image 本身？

Image tensor通常：

```python
requires_grad=False
```

所以虽然 backward为了算 backbone parameter gradients会计算相关局部 quantities，

最终不会把：

\[
image.grad
\]

作为训练目标积累。

---

# 148. 如果故意设置 `image.requires_grad=True` 呢？

那么可以计算：

\[
\frac{\partial L}{\partial image}
\]

这就是很多：

- saliency；
- adversarial attack；
- input optimization；

方法的基础。

Backprop不仅能对 parameters求 gradient，

也能对 input求。

---

# 149. 所以 Backprop 并不天然只服务“训练 Weight”

它更一般地计算：

> scalar objective对 computation graph中 differentiable variables的 derivatives。

Goodfellow 等教材也强调：

> backprop可以计算的不只是 cost 对 parameters 的 gradient。

---

# 150. 为什么 Gradient-Based Interpretability 会用 Backprop？

例如：

\[
\nabla_xL
\]

或：

\[
\nabla_xscore
\]

衡量：

> output对 input pixels/features的局部敏感度。

Captum等解释工具大量利用这个机制。

---

# 151. 但 Gradient 不自动等于 Causal Explanation

Gradient是：

> 局部敏感度。

它可能受：

- saturation；
- scaling；
- model nonlinearities；

影响。

不能直接把：

\[
|\nabla_x|
\]

叫做完整因果解释。

---

# 152. Backprop 到 Attention 时发生什么？

Attention：

\[
A=
softmax(
QK^\top/\sqrt{d_k}
)
\]

\[
O=AV
\]

Loss对：

\[
O
\]

的 gradient会分成多个参数路径：

- V path；
- A path。

---

# 153. V Path

因为：

\[
O=AV
\]

所以 gradient会进入：

\[
V
\]

进而进入：

\[
W_V
\]

和 source hidden states。

---

# 154. Attention Weight Path

同一个：

\[
O=AV
\]

也给：

\[
A
\]

gradient。

然后：

\[
A=softmax(S)
\]

gradient穿过 Softmax进入：

\[
S
\]

---

# 155. 再穿过 Score

\[
S=
QK^\top/\sqrt{d_k}
\]

于是 gradient分别进入：

\[
Q
\]

和：

\[
K
\]

最终训练：

\[
W_Q,W_K
\]

---

# 156. 所以 Q/K/V Roles 也是 Backprop 塑造出来的

Loss没有单独标签：

```text
这个 vector 应该成为 good query
```

而是通过：

> 最终 task loss对 Attention输出的要求，

逐渐调整 Q/K/V projection matrices。

这就是我们前面说：

\[
\boxed{
Role
\leftarrow
computation\ graph
+
gradient
}
\]

的真正训练机制。

---

# 157. Backprop 到 Softmax 为什么会 Coupling？

Softmax：

\[
p_i
=
\frac{e^{z_i}}{\sum_je^{z_j}}
\]

一个 logit：

\[
z_j
\]

改变会影响：

> 所有 \(p_i\)。

所以 Softmax Jacobian有 off-diagonal terms：

\[
-p_ip_j
\]

Backward自然把一个 position的变化：

> 传播成对同一 row其他 logits的 coupled gradients。

---

# 158. Backprop 到 LayerNorm 为什么会 Coupling？

LayerNorm：

\[
\mu,\sigma^2
\]

由一个 token全部 features共同决定。

所以某个：

\[
x_j
\]

改变：

> 会影响其他 normalized features。

因此 backward中 features之间也相互耦合。

这与 ReLU的纯 element-wise backward不同。

---

# 159. Backprop 到 Dropout 呢？

Training中已经采样 mask：

\[
m
\]

inverted dropout：

\[
y=
\frac{m\odot x}{q}
\]

所以：

\[
\boxed{
\nabla_xL
=
\frac{m}{q}
\odot
\nabla_yL
}
\]

被 drop的位置：

\[
m=0
\]

gradient也是：

\[
0
\]

---

# 160. Eval 时 Dropout 是 Identity

所以：

\[
y=x
\]

local derivative：

\[
I
\]

但 evaluation通常：

> 根本不执行 backward。

---

# 161. Backprop 经过 Residual 时为什么更容易有直接路径？

\[
y=x+F(x)
\]

upstream：

\[
g_y
\]

addition backward会：

- 一份直接给 identity x；
- 一份给 F(x)。

所以至少有：

\[
\boxed{
g_y
}
\]

这个直接 contribution回到 x，

再加 F branch contribution。

---

# 162. Post-LN 时为什么又更复杂？

原始 Transformer：

\[
y=LN(x+F(x))
\]

upstream gradient先经过：

\[
J_{LN}^\top
\]

再在 residual addition处分叉。

所以 shortcut gradient并不是从最终 output：

> 完全 untouched地直接走回 x。

这也是 Pre/Post-LN优化行为差异的一部分。

---

# 163. Pre-LN

\[
y=x+F(LN(x))
\]

这里 outermost addition直接给：

\[
x
\]

一个 identity gradient branch：

\[
g_y
\]

因此 backward结构不同。

---

# 164. Backprop 其实把我们之前所有模块串起来了

之前分别学：

- Linear；
- ReLU；
- Softmax；
- Attention；
- Residual；
- LayerNorm；
- Dropout；
- CVAE。

Backprop告诉我们：

> **为什么这些模块能够作为一个整体联合训练。**

只要每个模块提供：

> 合适的 local derivative/VJP，

最终 Loss就可以把 gradient传播到它们前面。

---

# 165. “End-to-End Differentiable” 到底是什么意思？

大致指：

> 从最终 objective到我们想训练的前端参数之间存在可微/可处理的 computation path。

例如：

```text
image
↓
ResNet
↓
Transformer
↓
action
↓
L1
```

所以 L1可以：

> 直接训练视觉 backbone。

这就是 end-to-end训练的重要含义。

---

# 166. 只要 Graph 中有一个 `detach()` 会怎样？

那条 path在该处停止 gradient。

例如：

\[
h_{det}=detach(h)
\]

后续：

\[
L=f(h_{det})
\]

不会给产生：

\[
h
\]

的上游 parameters传 gradient。

---

# 167. 但如果 h 还有另一条未 detach 的 Loss Path 呢？

那上游仍可能：

> 从另一条路径收到 gradient。

再次体现：

> gradient按所有可微 paths求和。

---

# 168. Non-Differentiable Operation 会怎样？

有些 operation：

> framework定义 subgradient/convention。

例如：

- ReLU at 0；
- absolute value at 0。

有些真正 discrete operation：

- argmax；
- sampling discrete categories；

普通 pathwise gradient可能无法直接穿过。

这时需要：

- surrogate gradient；
- REINFORCE；
- straight-through estimator；
- continuous relaxation；

等其他方法。

---

# 169. 为什么 `argmax` 很难反传？

输出 index会在大部分小扰动下：

> 完全不变，

跨 boundary又突然跳变。

所以普通 derivative无法提供有用连续梯度。

这也是训练 classification为什么通常用：

> Softmax/CrossEntropy，

而不是对 argmax index直接求 loss。

---

# 170. ACT 为什么不在训练中做实际 Environment Rollout 再 Backprop？

ACT是 imitation learning。

Training loss直接比较：

\[
a_{hat}
\]

和 demonstration：

\[
a
\]

不需要把 action送进真实机器人 dynamics并对 environment求 gradient。

所以：

\[
\boxed{
\text{policy supervised loss is differentiable internally}
}
\]

---

# 171. 这和 Model-Based Differentiable Control 不一样

有些方法会有：

\[
a
\rightarrow
dynamics
\rightarrow
future\ state
\rightarrow
loss
\]

如果 dynamics model可微，

gradient可以进一步穿过 dynamics。

ACT canonical training没有这条路径。

---

# 172. Temporal Ensemble 参与 Backprop 吗？

不参与 canonical training。

Temporal Ensemble是：

> inference-time action aggregation。

Training loss针对：

> predicted action chunk。

所以：

\[
\boxed{
TE\text{ has no training gradient path}
}
\]

---

# 173. PID Robot Controller 参与 Backprop 吗？

也不参与 ACT policy training。

训练只使用 offline demonstrations。

实际 inference时 action target交给：

> low-level controller。

所以 controller不在：

\[
loss.backward()
\]

graph里。

---

# 174. 为什么把 Training Graph 和 Execution Graph 区分很重要？

模型训练：

> 哪些模块收到 gradient，

由 training computation graph决定。

推理时系统里出现：

- Temporal Ensemble；
- PID；
- robot environment；

不代表这些东西：

> 参与过 policy gradient training。

---

# 175. Optimizer 看不到 Computation Graph 吗？

Optimizer通常只拿：

> model parameters和它们的 `.grad`。

例如：

```python
optimizer.step()
```

它不需要重新理解：

> gradient是通过哪个 Attention、哪个 Loss算出来的。

Backprop已经把所有信息压缩成：

\[
\boxed{
parameter.grad
}
\]

---

# 176. Adam 做的不是 Backprop

Adam读取：

\[
g_t
=
\nabla_\theta L_t
\]

然后维护：

- first moment；
- second moment；

再更新 parameters。

所以：

\[
\boxed{
\text{Backprop computes }g_t
}
\]

\[
\boxed{
\text{Adam decides how to use }g_t
}
\]

---

# 177. Learning Rate 属于哪里？

属于：

> optimizer/update rule。

它不会改变 Chain Rule计算出来的 raw derivative定义。

但会改变：

> 参数实际走多远。

---

# 178. Gradient Clipping 又在哪里？

通常发生在：

```text
backward之后
optimizer.step之前
```

对已经算出的 gradient：

> 限制 norm/value。

这同样不是 Backprop本身。

---

# 179. Weight Decay 呢？

可能通过：

- optimizer decoupled weight decay；
- 或 objective中的 regularization项；

作用。

若作为 explicit loss：

\[
L+\lambda\|W\|^2
\]

则它通过 Backprop产生额外 gradient。

如果是 AdamW式 decoupled decay：

> update rule层面处理。

---

# 180. “Gradient Flow”是什么意思？

通常指：

> 梯度是否能够从 Loss有效传播到较早 parameters，以及 magnitude/conditioning如何。

会关注：

- vanishing；
- exploding；
- zero gradient；
- branch；
- saturation；
- normalization；
- residual。

---

# 181. Gradient Vanishing

深链：

\[
\frac{\partial L}{\partial x}
=
J_n^\top
J_{n-1}^\top
\cdots
J_1^\top
g
\]

如果很多 Jacobian的 effective singular values：

\[
<1
\]

梯度可能指数式衰减。

---

# 182. Gradient Exploding

如果连乘中 effective amplification：

\[
>1
\]

反复出现，

gradient norm可能迅速增大。

---

# 183. 所以 Chain Rule 既让 Deep Learning 可训练，也创造了 Deep Optimization 问题

同一个：

\[
\boxed{
\text{Jacobian product}
}
\]

机制，

既负责：

> 把 supervision传到早期 layers，

也可能导致：

- vanishing；
- exploding。

---

# 184. ReLU 为什么有帮助但不彻底解决？

active区：

\[
ReLU'=1
\]

不会额外缩小 gradient。

但：

- inactive区是0；
- weight matrices仍参与连乘。

所以仍需：

- initialization；
- residual；
- normalization；

等结构。

---

# 185. Residual 为什么重要？

因为：

\[
J=
I+J_F
\]

给 gradient一个 identity component，

改善非常深网络的优化路径。

这也是 Transformer能堆多层的重要结构之一。

---

# 186. LayerNorm 为什么又影响 Backprop？

因为它对 activation尺度和 Jacobian结构进行重参数化。

Pre/Post placement甚至决定：

> identity path是否需要先穿过 normalization Jacobian。

---

# 187. Backprop 是精确 Gradient 吗？

对当前浮点计算图与定义的 local backward rules而言：

> Autodiff计算的是解析 Chain Rule derivative，而不是 finite-difference approximation。

但现实有：

- floating point roundoff；
- nondifferentiable-point convention；
- stochastic sampling；
- mixed precision；

所以“精确”需理解为：

> algorithmic differentiation意义，而非无限精度实数计算。

---

# 188. Dropout 下的 Gradient 是什么？

每次 training forward随机采样一个 mask。

Backward计算：

> **当前这个 sampled stochastic computation graph 的 pathwise gradient。**

下一个 forward换 mask：

> gradient也会不同。

---

# 189. CVAE z Sampling 同理

每次：

\[
\epsilon
\]

不同。

Backward得到：

> 当前 \(\epsilon\) sample下的 reparameterized gradient estimate。

多次 stochastic batches/samples：

> 共同近似期望 objective 的 gradient。

---

# 190. 这就是 Stochastic Gradient 的另一个来源

不仅 mini-batch sampling会造成 stochasticity。

ACT training还可能有：

- CVAE epsilon；
- Dropout mask；
- random episode/start timestep。

所以每步 gradient：

> 是 noisy estimate。

---

# 191. Noise 并不意味着 Gradient 没有方向

如果 estimator合理，

长期平均上仍可以：

> 指向降低 expected objective的方向。

现代深度学习大量依赖这种 stochastic optimization。

---

# 192. `detach_dict` 为什么常用于 Logging？

ACT training会把 forward dict detach后存 history。

这是合理的：

> logging不需要保留整个 computation graph。

如果把每个 loss tensor连着 graph全部存进 Python list，

可能导致：

> graph references无法释放，memory增长。

---

# 193. 一个重要工程原则

当某 tensor只用于：

- 打印；
- history；
- metric；

不再需要 backward，

通常可以：

```python
tensor.detach()
```

或：

```python
tensor.item()
```

避免无意保留 graph。

---

# 194. `.item()` 和 `.detach()` 区别

### `.detach()`

仍是 tensor，

但与 graph断开。

### `.item()`

将单元素 tensor转成：

> Python number。

也自然不再保留 gradient graph。

---

# 195. 不要在 Loss 计算中乱用 `.item()`

例如：

```python
loss = loss1.item() + loss2
```

那么：

\[
loss1
\]

已经变成 Python数值，

它对参数的 gradient path：

> 被切断。

这会导致：

> loss1不再训练对应模型部分。

---

# 196. 同理 NumPy Conversion 也可能切 Graph

通常：

```python
tensor.detach().cpu().numpy()
```

用于：

> visualization/logging。

如果把 tensor转 NumPy，做计算再转回来：

> 普通 PyTorch Autograd不会自动知道 NumPy operations的 derivatives。

---

# 197. 所以 End-to-End Graph 必须保持在 Autograd 支持的 Operations 中

如果中间调用：

- NumPy；
- external simulator；
- custom C++ op without backward；
- discrete black box；

gradient可能断掉。

若希望可微：

> 必须提供可微实现或自定义 backward。

---

# 198. 自定义 `autograd.Function`

如果实现一个新 operation：

> 可以自己定义 forward和backward。

Backward接收：

> output upstream gradients，

返回：

> input gradients。

这正是 local VJP接口。

---

# 199. 为什么 Custom Backward 容易写错？

需要确保：

- shape；
- dtype；
- device；
- broadcasting reductions；
- saved tensors；
- nondifferentiable inputs；

都正确。

所以能用原生 PyTorch operations组合时：

> 通常更安全。

---

# 200. Broadcasting 的 Backward 为什么需要 Sum？

例：

\[
y=x+b
\]

其中：

\[
x:
[B,D]
\]

\[
b:
[D]
\]

Forward把 b broadcast到：

\[
B
\]

个 samples。

所以 backward：

\[
\boxed{
\nabla_bL
=
\sum_{i=1}^{B}
\nabla_{y_i}L
}
\]

这又是 gradient accumulation。

---

# 201. Linear Bias Gradient 就是典型 Broadcast Reduction

一个 bias：

\[
b_j
\]

被：

> 所有 batch samples、所有 token positions

共享使用。

所以其 gradient：

> 汇总这些 usages的 contribution。

---

# 202. 为什么 FFN Bias 能从所有 1202 Tokens 学习？

因为：

\[
b
\]

在每个 token forward中都被加一次。

Backward时：

> 1202 positions的 gradient contribution全部求和/平均进同一个 bias gradient。

---

# 203. Attention Projection Weight 同样如此

同一个：

\[
W_Q
\]

处理所有 tokens。

所以其 gradient来自：

> 所有 query positions、所有 batch examples。

---

# 204. 一个参数为什么有时 Gradient 恰好是0？

可能原因很多：

1. 不在 Loss path上；
2. ReLU branch inactive；
3. mask使 contribution为0；
4. derivatives数学上抵消；
5. saturation；
6. 当前 batch没有激活相关行为。

不能仅凭：

```python
grad == 0
```

立刻判断：

> 参数坏了。

---

# 205. `grad is None` 和 `grad == 0` 也不同

### `grad is None`

可能表示：

> parameter没有参与这次 backward path，或尚未 backward/grad被设None。

### `grad` 是全零 tensor

表示：

> graph路径存在并产生了一个数值为0的 gradient，或框架创建了零梯度。

Debug时要区分。

---

# 206. `is_pad_head` 是一个很适合检查 `grad is None` 的例子

如果 official total loss完全不依赖它，

其参数：

> 可能不会收到 gradient。

这与某个 ReLU参数收到“数值0 gradient”概念不同。

---

# 207. Freeze Parameter 又是什么？

如果：

```python
param.requires_grad_(False)
```

那么 autograd不需要为：

> 更新这个 leaf parameter

计算/保存对应 gradient。

常用于：

- frozen backbone；
- fine-tuning。

---

# 208. Freeze 之后下游还能训练吗？

可以。

假设：

```text
frozen backbone
↓
trainable head
```

backbone output作为数值输入给 head。

Head parameters仍然可以：

> 正常计算 gradient并更新。

只是 gradient不需要回到 frozen backbone parameters。

---

# 209. 如果只 Freeze Backbone Parameters，输入还可能需要 Gradient 吗？

如果另有可训练 upstream path或对 input求 gradient：

> 情况可以更复杂。

但普通 fine-tuning：

> backbone parameters关闭 requires_grad即可截断对这些 leaves的 parameter gradient需求。

---

# 210. ACT 如果 Freeze ResNet 会怎样？

L1仍训练：

- Transformer；
- action head；
- latent modules；

但不会更新：

> frozen ResNet parameters。

视觉 features仍作为 forward inputs参与 policy。

---

# 211. Why Backprop Is Efficient：Shared Subexpressions

Goodfellow 等教材强调：

> 直接展开 Chain Rule可能重复计算很多相同 subexpressions。

Backprop存储并复用 forward/intermediate derivative information，

避免：

> 指数级重复展开。

---

# 212. 一个链

\[
w
\rightarrow
x=f(w)
\rightarrow
y=f(x)
\rightarrow
z=f(y)
\]

Derivative：

\[
\frac{dz}{dw}
=
f'(y)
f'(x)
f'(w)
\]

如果符号展开：

\[
f'(f(f(w)))
f'(f(w))
f'(w)
\]

会重复出现：

\[
f(w)
\]

等 expressions。

Forward先保存：

\[
x,y
\]

Backward直接复用。

---

# 213. 所以 Forward 和 Backward 是一对

Forward不仅产生：

> prediction。

还产生：

> backward所需 context。

这也是为什么 training框架需要管理 computation history。

---

# 214. Backward Complexity 大致如何？

Goodfellow教材指出，

对典型 computational graph，

Backprop访问 graph edges并执行 local derivative products，

计算量与 forward：

> 通常是同一数量级，而不是每个参数单独重新跑一遍网络。

具体常数依赖 operation。

---

# 215. 这就是为什么百万参数也能训练

如果每个参数都单独 finite-difference：

> 一步训练需要百万次 forward。

Backprop：

> 一次 forward + 一次 reverse pass

就得到所有 parameter gradients。

这是神经网络训练可行的核心原因之一。

---

# 216. 历史上 1986 论文的重要性

Rumelhart、Hinton、Williams 1986：

**Learning representations by back-propagating errors**

展示了：

> 通过调整多层网络 connection weights降低 output与target差异，hidden units可以学习有用 task features。

这是 backprop训练多层 neural networks最经典、最有影响力的论文之一。

---

# 217. 但不要说“1986发明了 Chain Rule”

Chain Rule远早于神经网络。

Automatic differentiation、reverse accumulation、neural-network gradient方法也有更早历史。

所以更严谨：

\[
\boxed{
\text{1986 paper is a landmark popularization/application of backprop to learning hidden representations}
}
\]

而不是：

> 所有反向传播思想第一次出现于1986。

---

# 218. Common Misconception 1：Backprop 是把 Loss 数值倒着传

**错误。**

传播的是：

\[
\boxed{
\text{derivatives / gradient information}
}
\]

---

# 219. Common Misconception 2：Backprop 就是 Gradient Descent

**错误。**

Backprop：

> 算 gradient。

Gradient Descent / Adam：

> 用 gradient更新。

---

# 220. Common Misconception 3：`loss.backward()` 会直接改 Parameters

**错误。**

它填充/累积：

```python
param.grad
```

真正更新通常：

```python
optimizer.step()
```

---

# 221. Common Misconception 4：`optimizer.step()` 会自己算 Gradient

标准使用中：

> 不会。

它读取已经存在的：

```python
param.grad
```

---

# 222. Common Misconception 5：Backward 是把 Forward Operation 逆过来

**错误。**

它计算 local derivatives/VJPs。

---

# 223. Common Misconception 6：Operation 必须可逆才能 Backprop

**错误。**

ReLU不可逆，

仍可 backward。

---

# 224. Common Misconception 7：所有函数必须 Everywhere Differentiable

**错误。**

ReLU/L1等不可导点可以采用 subgradient/convention。

---

# 225. Common Misconception 8：Backprop 会构造整个 Jacobian Matrix

通常：

**不会。**

它计算：

\[
J^\top g
\]

这类 local products。

---

# 226. Common Misconception 9：Vector Output 可以直接 `backward()` 而无需说明方向

通常：

**不能。**

Non-scalar需要指定：

> output gradient / cotangent。

---

# 227. Common Misconception 10：Scalar Loss backward 没有初始 Gradient

其实 seed：

\[
\boxed{
\frac{\partial L}{\partial L}=1
}
\]

---

# 228. Common Misconception 11：一个节点多条 downstream paths 时随便选一条

**错误。**

所有 path gradient contributions要：

\[
\boxed{
\text{sum}
}
\]

---

# 229. Common Misconception 12：Residual Gradient 只来自 Shortcut

**错误。**

总 gradient包括：

\[
I
\]

path和：

\[
F
\]

branch contribution。

---

# 230. Common Misconception 13：`.grad` 里只有最后一次 backward结果

PyTorch默认：

> accumulate。

所以需要适当：

```python
zero_grad()
```

---

# 231. Common Misconception 14：Non-Leaf `.grad=None` 说明没有 Gradient 经过它

**错误。**

中间 gradient可能被计算并使用，只是默认不 retained。

---

# 232. Common Misconception 15：`requires_grad=True` 说明 Tensor 已经有 Gradient

**错误。**

只是要求追踪/计算。

---

# 233. Common Misconception 16：`model.eval()` 会关闭 Backprop

**错误。**

`eval()`不是 gradient mode。

---

# 234. Common Misconception 17：`torch.no_grad()` 等于 `model.eval()`

**错误。**

一个控制 autograd recording，

一个控制某些 module train/eval行为。

---

# 235. Common Misconception 18：`detach()` 会把 Tensor 变0

**错误。**

数值不变，

gradient history断开。

---

# 236. Common Misconception 19：`.item()` 可以安全放进 Loss 公式而不影响 Training

**错误。**

它会把 tensor变 Python scalar并切断那条 autograd路径。

---

# 237. Common Misconception 20：Input 不 requires_grad，模型就不能训练

**错误。**

Model parameters只要 requires_grad即可计算 parameter gradients。

---

# 238. Common Misconception 21：KL Loss 会直接训练 Action Head

**错误。**

没有 dependency path。

---

# 239. Common Misconception 22：L1 Loss 只训练 Decoder

**错误。**

它会通过 policy graph、latent reparameterization继续回到 CVAE encoder。

---

# 240. Common Misconception 23：Random Sampling z 会把 Gradient 完全切断

ACT使用：

\[
z=\mu+\sigma\epsilon
\]

reparameterization，

所以 pathwise gradient可传回：

\[
\mu,\logvar
\]

---

# 241. Common Misconception 24：KL 是唯一训练 μ/logvar 的 Loss

**错误。**

Reconstruction/L1也通过：

\[
z
\]

训练它们。

---

# 242. Common Misconception 25：β=10 表示 KL 对参数影响一定是 L1 的10倍

**错误。**

它只是把：

\[
\nabla KL
\]

乘10。

实际 gradient norms仍取决于两项自身 derivatives。

---

# 243. Common Misconception 26：Module 只要存在于模型里就一定被训练

**错误。**

必须连接到 Loss。

---

# 244. Common Misconception 27：ACT `is_pad_head` 算出来了，所以一定收到当前 Loss Gradient

当前 official `policy.py` total loss：

> 没有使用 `is_pad_hat`。

因此不能因为 forward算了它：

> 就假定它被这条 objective训练。

---

# 245. Common Misconception 28：Temporal Ensemble 参与 ACT Backprop

**错误。**

它是 inference-time aggregation。

---

# 246. Common Misconception 29：机器人 Environment/PID 在 ACT Loss Graph 中

**错误。**

Canonical ACT是 offline imitation learning。

---

# 247. Common Misconception 30：Backprop 只适用于 MLP

**错误。**

只要 operation有适当 differentiable backward rule，

同一机制可训练：

- CNN；
- Transformer；
- VAE；
- policy network；
- differentiable simulator；
- 等。

---

# 248. 一张图记住 Backprop

```text
FORWARD

x
│
▼
operation f
│
▼
y
│
▼
operation g
│
▼
z
│
▼
Loss L


BACKWARD

∂L/∂L = 1
│
▼
g backward:
upstream × local derivative
│
▼
∂L/∂y
│
▼
f backward:
upstream × local derivative
│
▼
∂L/∂x
```

---

# 249. 一张图记住 Branch Gradient

```text
          ┌→ branch A ──→ L
x ────────┤
          └→ branch B ──→ L
```

那么：

\[
\boxed{
\nabla_xL
=
\nabla_xL|_A
+
\nabla_xL|_B
}
\]

---

# 250. 一张图记住 Reverse-Mode AD

```text
Forward:
values flow →→→

Backward:
cotangents / gradients flow ←←←
```

每个 node只做：

\[
\boxed{
local\ VJP
}
\]

整个 graph自动组合成：

\[
\boxed{
\nabla_\theta L
}
\]

---

# 251. 一张图记住 ACT Gradient Flow

```text
                     demonstration actions
                       │             │
                       │             └─────────────┐
                       ▼                           │
                  CVAE Encoder                    │
                       │                           │
                       ▼                           │
                    h_CLS                          │
                       │                           │
                       ▼                           │
                 latent_proj                       │
                   │      │                        │
                   ▼      ▼                        │
                   μ    logvar                     │
                   │      │                        │
                   ├── KL ┴───────────────→ βKL    │
                   │                               │
                   ▼                               │
           reparameterization                      │
                   │                               │
                   ▼                               │
                   z                               │
                   │                               │
                   ▼                               │
           latent_out_proj                         │
                   │                               │
images → ResNet ───┼────────┐                      │
qpos → proj ───────┼──────┐ │                      │
                   ▼      ▼ ▼                      │
             Policy Transformer                    │
                   │                               │
                   ▼                               │
              action_head                          │
                   │                               │
                   ▼                               │
                 a_hat ───────── compare ──────────┘
                   │
                   ▼
                  L1

Total:
L = L1 + β KL
```

Backward：

```text
L1:
action_head
← decoder
← cross-attention
← policy encoder / ResNet / qpos projection
← latent_out_proj
← z
← μ, logvar
← latent_proj
← CVAE encoder

KL:
μ, logvar
← latent_proj
← CVAE encoder

At shared μ/logvar/encoder parameters:
gradients add.
```

---

# 252. ACT 最值得记住的 Gradient 结构

对于 CVAE encoder parameters：

\[
\theta_E
\]

有：

\[
\boxed{
\nabla_{\theta_E}L
=
\nabla_{\theta_E}L_{L1}
+
\beta
\nabla_{\theta_E}L_{KL}
}
\]

---

对于 Action Head：

\[
\theta_A
\]

只有：

\[
\boxed{
\nabla_{\theta_A}L
=
\nabla_{\theta_A}L_{L1}
}
\]

因为：

\[
KL
\]

不依赖 Action Head。

---

对于 Policy Transformer：

\[
\theta_P
\]

同样主要：

\[
\boxed{
\nabla_{\theta_P}L
=
\nabla_{\theta_P}L_{L1}
}
\]

KL不经过它。

---

# 253. 一句话真正理解 Backpropagation

> **Backpropagation 不是把“预测误差”这个数值从输出层原样倒着传，而是在 forward computation graph 上从 scalar loss 的种子梯度 \(\partial L/\partial L=1\) 开始，按反向拓扑顺序让每个 operation把收到的 upstream gradient与自己的 local Jacobian组合成对输入的 gradient；当一个变量通过多条路径影响 loss 时，这些 contribution自动相加。这样无需显式构造巨大 Jacobian，就能通过一系列局部 VJP 高效得到数百万参数的 \(\nabla_\theta L\)。**

---

# 254. 一句话理解为什么 Hidden Layer 能学东西

> **Hidden layer没有自己的人工标签也没有关系：只要它处在 output loss 的可微路径上，最终 loss 对 hidden activation 的敏感度就能通过 Chain Rule继续转化为对该层 weights 的敏感度，因此 supervision可以穿过很多中间模块间接塑造内部 representations；这就是 end-to-end representation learning 的数学基础。**

---

# 255. 一句话理解 ACT 的 Backprop

> **ACT 训练时把 L1 reconstruction 与 \(\beta KL\) 合成一个 scalar loss后只需一次 `loss.backward()`：L1从 action prediction沿 Action Head、Decoder、Cross-Attention、Policy Encoder、视觉 backbone与 latent branch反向传播，并通过 reparameterization继续进入 \(\mu/\logvar\) 和 CVAE Encoder；KL则从 \(\mu/\logvar\) 直接进入 latent projection与 CVAE Encoder，两条 gradient在共享 latent-encoder parameters处相加，因此 CVAE必须同时满足“帮助动作重建”和“靠近 prior”两个训练压力。**

---

# 256. 下一篇：Gradient Descent

现在我们已经解决：

\[
\boxed{
\text{Gradient 是怎样算出来的}
}
\]

但是还没有解决另一个问题：

> **有了 gradient，参数到底应该怎样更新？**

下一篇：

> **Gradient Descent：为什么沿负梯度方向走，Loss 就会下降？**

会从最基础：

\[
\theta_{t+1}
=
\theta_t
-
\eta\nabla_\theta L
\]

开始解释：

- gradient为什么是最陡上升方向；
- 为什么 negative gradient是局部最陡下降方向；
- Taylor expansion；
- learning rate；
- overshoot；
- convex vs non-convex；
- local minimum；
- saddle point；
- stochastic gradient；
- mini-batch；
- gradient noise；
- momentum；
- 为什么 Adam不是“Backprop的升级版”；
- ACT的 optimizer究竟接收了什么；
- backbone learning rate与main learning rate为什么可以不同。

---

## Primary Historical Source

David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams.

**Learning representations by back-propagating errors.**  
Nature 323, 533–536, 1986.

- Nature: https://www.nature.com/articles/323533a0
- DOI: https://doi.org/10.1038/323533a0

论文描述了一种通过调整 network connection weights，使实际输出与目标输出之间差异减小的 learning procedure，并强调：

> hidden units 可以通过这种训练形成对任务有用的内部 features / representations。

它是现代多层 neural network backpropagation最经典、最有影响力的历史论文之一。

历史上 Chain Rule、automatic differentiation 与 reverse accumulation具有更早来源，因此更严谨的说法是：

> **Rumelhart–Hinton–Williams 1986 是 backpropagation 用于训练 hidden representations 的 landmark work，而不是 Chain Rule 或所有 reverse-mode differentiation思想的最初发明。**

---

## Core Textbook Source

Ian Goodfellow, Yoshua Bengio, Aaron Courville.

**Deep Learning.**  
MIT Press, 2016.

Chapter 6 — Deep Feedforward Networks:

- https://www.deeplearningbook.org/contents/mlp.html
- https://www.deeplearningbook.org/

Section 6.5 系统介绍：

- computational graphs；
- Chain Rule；
- recursively applying Chain Rule；
- back-propagation；
- tensor/Jacobian notation；
- forward graph与backward graph；
- shared subexpression复用；
- backprop computation cost。

其核心定义非常重要：

> Back-propagation is an algorithm that computes the chain rule with a specific order of operations that is highly efficient.

本文的 computational-graph / chain-rule 主线以此为主要教材依据。

---

## PyTorch Autograd Primary Reference

PyTorch:

**Autograd mechanics**

- https://docs.pytorch.org/docs/stable/notes/autograd

PyTorch明确描述：

> autograd is a reverse automatic differentiation system。

Forward执行时会记录产生 tensors 的 operation graph；从 graph roots向 leaves反向追踪时，使用 Chain Rule自动计算 gradients。

文档同时解释：

- `requires_grad`；
- leaf / non-leaf tensors；
- `grad_fn`；
- `no_grad`；
- inference mode；
- `eval()` 与 gradient computation的区别。

---

## PyTorch `backward`

- https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html
- https://docs.pytorch.org/docs/stable/generated/torch.autograd.backward.html

官方文档说明：

- `.backward()` 使用 Chain Rule；
- scalar tensor可直接 backward；
- non-scalar tensor需要额外提供 `gradient`；
- backward会在 graph leaves累积 gradients；
- 因此调用前通常需要把 `.grad` 清零或设为 `None`。

对于 non-scalar outputs，

PyTorch计算的是 Jacobian与给定 gradient vector的乘积，而不是默认返回整个 Jacobian。

---

## PyTorch Detach / No-Grad

- Autograd mechanics:
  https://docs.pytorch.org/docs/stable/notes/autograd
- `detach`:
  https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html
- `requires_grad`:
  https://docs.pytorch.org/docs/stable/generated/torch.Tensor.requires_grad.html

重要区别：

### `requires_grad=False`

控制某些 leaf parameters/tensors是否参与 gradient tracking。

### `detach()`

从当前 graph中切断一个 tensor的 autograd history。

### `torch.no_grad()`

让一个代码区域的 operations不被记录到 backward graph。

### `model.eval()`

改变 Dropout/BatchNorm等 module behavior，

**不是关闭 Autograd。**

---

## ACT Primary Source

Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.

**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.**  
RSS 2023.

- arXiv: https://arxiv.org/abs/2304.13705
- PDF: https://arxiv.org/pdf/2304.13705

ACT使用 CVAE style objective，将 action-chunk reconstruction与 KL regularization结合。

本文实现细节以官方代码为最终依据。

---

## ACT Official Loss

Official repository:

https://github.com/tonyzhaozh/act

`policy.py`:

https://github.com/tonyzhaozh/act/blob/main/policy.py

当前 official `ACTPolicy`：

```python
a_hat, is_pad_hat, (mu, logvar) =
    self.model(
        qpos,
        image,
        env_state,
        actions,
        is_pad
    )

total_kld, _, _ =
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

loss_dict['l1'] = l1
loss_dict['kl'] = total_kld[0]

loss_dict['loss'] =
    loss_dict['l1']
    +
    loss_dict['kl']
    *
    self.kl_weight
```

因此实际优化：

\[
\boxed{
L
=
L_{L1}
+
\beta L_{KL}
}
\]

而不是论文 Algorithm 1 中简写的 MSE reconstruction。

---

## ACT Reparameterization

`detr/models/detr_vae.py`:

https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py

官方：

```python
def reparametrize(
    mu,
    logvar
):
    std =
        logvar.div(2).exp()

    eps =
        Variable(
            std.data
            .new(std.size())
            .normal_()
        )

    return (
        mu
        +
        std * eps
    )
```

数学：

\[
\boxed{
z
=
\mu
+
e^{\frac12\logvar}
\epsilon
}
\]

因此当前 sampled：

\[
\epsilon
\]

被当作独立随机 source，

而：

\[
z
\]

对：

\[
\mu,\logvar
\]

保持 differentiable path。

这使 L1 reconstruction gradient能够反向穿过 latent sample，进入 CVAE Encoder。

---

## ACT Training Loop

`imitate_episodes.py`:

https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py

官方训练核心：

```python
policy.train()

optimizer.zero_grad()

for batch_idx, data in enumerate(
    train_dataloader
):
    forward_dict =
        forward_pass(
            data,
            policy
        )

    loss =
        forward_dict['loss']

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()
```

这个顺序正好对应：

\[
\boxed{
Forward
\rightarrow
Loss
\rightarrow
Backprop
\rightarrow
Optimizer\ Step
\rightarrow
Clear\ Gradients
}
\]

Validation则：

```python
with torch.inference_mode():
    policy.eval()
```

因此不建立普通 training backward graph。

---

## ACT `is_pad_head` Gradient Caveat

Official `detr_vae.py`：

```python
a_hat =
    self.action_head(hs)

is_pad_hat =
    self.is_pad_head(hs)

return (
    a_hat,
    is_pad_hat,
    [mu, logvar]
)
```

但当前 official `policy.py` 构造 total loss时：

> 使用 `a_hat` 的 L1 和 `mu/logvar` 的 KL，

没有把：

```python
is_pad_hat
```

加入 loss。

因此从这条 official objective的 computation graph看：

\[
\boxed{
is\_pad\_head
\text{ 没有通过 }is\_pad\_hat\text{ 接到 total loss}
}
\]

这是理解 Backprop最直接的真实代码例子之一：

> **Forward里算出了一个 tensor，不代表产生它的参数就一定会被当前 loss训练。**

---

## 本文知识连接

### 数学

- Derivative
- Partial Derivative
- Gradient
- Chain Rule
- Jacobian
- Vector-Jacobian Product
- Computational Graph

### Deep Learning

- [Linear Layer](./linear-layer.md)
- [ReLU](./relu.md)
- [MLP](./mlp.md)
- Loss Function
- Gradient Descent
- Adam
- Automatic Differentiation
- Gradient Checkpointing
- [Residual Connection](./residual-connection.md)
- [Layer Normalization](./layer-normalization.md)
- [Dropout](./dropout.md)

### Transformer

- [Attention](./attention.md)
- [Softmax](./softmax.md)
- [Multi-Head Attention](./multi-head-attention.md)
- [Transformer Encoder](./transformer-encoder.md)
- [Transformer Decoder](./transformer-decoder.md)

### Generative Modeling

- [VAE](../generative-models/vae.md)
- [CVAE](../generative-models/cvae.md)
- [Reparameterization Trick](../generative-models/reparameterization-trick.md)
- [KL Divergence](../mathematics/kl-divergence.md)

### Robot Learning

- [ACT Architecture](../robot-learning/act/architecture.md)
- [CVAE in ACT](../robot-learning/act/cvae-in-act.md)
- [ACT Training](../robot-learning/act/training.md)
- [ACT Complete Data Flow](../robot-learning/act/complete-data-flow.md)

### 下一步

- Gradient Descent
