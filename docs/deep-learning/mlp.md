---
title: "MLP：Linear + ReLU 为什么就能组成真正的神经网络？"
description: "从单个 affine neuron 出发，完整构造 Multi-Layer Perceptron：hidden layer、activation、forward propagation、loss、backpropagation、gradient descent、representation learning、width/depth 与 universal approximation，并解释 Transformer FFN 为什么本质上是 position-wise two-layer MLP，以及 ACT 中它如何处理每个 512-D token。"
status: reviewed
pageType: concept
canonical: /deep-learning/mlp
updated: "2026-09-15"
---

# MLP：Linear + ReLU 为什么就能组成真正的神经网络？

到现在为止，我们已经分别理解了两个最基础的模块：

## Linear Layer

\[
\boxed{
z=Wx+b
}
\]

它可以学习：

> 一组 feature combinations。

但如果网络里只有 Linear / Affine transformations，

无论堆多少层：

\[
Linear
\rightarrow
Linear
\rightarrow
Linear
\]

最终都可以合并成：

\[
\boxed{
y=W'x+b'
}
\]

所以它仍然只是：

> 一个全局 affine mapping。

---

## ReLU

\[
\boxed{
ReLU(x)=\max(0,x)
}
\]

它没有 trainable parameters，

但会根据输入：

> 让不同 hidden units active / inactive。

于是：

\[
W_2ReLU(W_1x+b_1)+b_2
\]

不能再简单合并成一层 Linear。

---

现在我们终于可以把它们组合起来：

\[
\boxed{
Linear
\rightarrow
ReLU
\rightarrow
Linear
}
\]

这已经是一个真正的：

> **Multi-Layer Perceptron，MLP。**

但“MLP 是几层全连接网络”还远远不够。

真正值得理解的是：

> 为什么只是反复做“加权和 → 非线性 → 加权和”，模型就能够自动学习 representations？

> Hidden Layer 到底隐藏了什么？

> Forward propagation 到底在计算什么？

> Loss 是怎么从最终 output影响到第一层 weights 的？

> Backpropagation 是不是把“误差值”直接从后往前传？

> Universal Approximation 是不是说“一个隐藏层就够了，所以深度没意义”？

> Transformer 明明是 Attention 模型，为什么内部仍然大量使用 MLP？

这一篇会第一次把一个完整神经网络从：

\[
\boxed{
\text{输入}
}
\]

一路讲到：

\[
\boxed{
\text{参数更新}
}
\]

形成完整闭环。

---

# 1. 先从最简单的问题开始

假设我们有两个输入：

\[
x_1,\quad x_2
\]

想预测一个输出：

\[
y
\]

例如：

```text
x₁ = 温度
x₂ = 湿度

y = 某个连续目标值
```

最简单的模型：

\[
\boxed{
\hat y
=
w_1x_1+w_2x_2+b
}
\]

也就是：

> Linear / affine model。

---

# 2. 这个模型只能学一个平面

二维输入时：

\[
\hat y
=
w_1x_1+w_2x_2+b
\]

几何上是：

> 一个平面。

如果真实函数非常复杂：

- 有弯曲；
- 有多个 regime；
- 某些 feature interaction只在某些情况下生效；

一个固定平面：

> 很难表达。

---

# 3. 加 Hidden Units

我们不直接从：

\[
x
\]

预测：

\[
y
\]

而先生成一些中间 features：

\[
h_1,h_2,\ldots,h_m
\]

例如：

\[
h_1
=
ReLU(
w_{11}x_1+w_{12}x_2+b_1
)
\]

\[
h_2
=
ReLU(
w_{21}x_1+w_{22}x_2+b_2
)
\]

……

然后：

\[
\hat y
=
v_1h_1+
v_2h_2+
\cdots+
v_mh_m+
c
\]

---

# 4. Matrix 写法

输入：

\[
x\in\mathbb R^{d_{in}}
\]

第一层：

\[
\boxed{
z^{(1)}
=
W^{(1)}x+b^{(1)}
}
\]

activation：

\[
\boxed{
h^{(1)}
=
\phi(
z^{(1)}
)
}
\]

输出层：

\[
\boxed{
\hat y
=
W^{(2)}h^{(1)}
+
b^{(2)}
}
\]

其中：

\[
\phi
\]

可以是：

- ReLU；
- Sigmoid；
- Tanh；
- GELU；
- 等。

---

# 5. 这就是最基本的一 Hidden-Layer MLP

结构：

```text
Input
  │
  ▼
Linear
  │
  ▼
Nonlinear Activation
  │
  ▼
Linear
  │
  ▼
Output
```

数学：

\[
\boxed{
f(x)
=
W_2
\phi(
W_1x+b_1
)
+b_2
}
\]

---

# 6. 为什么叫 Multi-Layer Perceptron？

历史上：

> Perceptron 是早期人工神经网络模型。

现代 deep learning语境里：

> MLP通常指由多个 fully-connected / dense layers与 nonlinear activations组成的 feedforward network。

但要注意：

“Perceptron”这个词在不同历史/教材语境中可能指：

- Rosenblatt式 threshold perceptron；
- 单层 linear classifier；
- 更宽泛的 neuron model。

现代工程里：

\[
\boxed{
MLP
\approx
\text{stack of dense affine layers + nonlinear activations}
}
\]

是最常见用法。

---

# 7. MLP 和 Neural Network 是不是同义词？

不是完全同义。

MLP 是：

> neural network 的一种具体 architecture。

其他神经网络还有：

- CNN；
- RNN；
- Transformer；
- GNN；
- Diffusion-model U-Net；
- 等。

但很多复杂 architecture内部：

> 仍然包含 MLP blocks。

---

# 8. Transformer 里面也有 MLP

原始 Transformer 的 FFN：

\[
FFN(x)
=
W_2ReLU(W_1x+b_1)+b_2
\]

本质就是：

\[
\boxed{
\text{一个 two-layer MLP}
}
\]

只不过：

> 对每个 token position独立使用同一套参数。

所以也常叫：

> Position-wise MLP / Feed-Forward Network。

---

# 9. ACT 中也是一样

ACT：

\[
d_{\text{model}}=512
\]

\[
d_{ff}=3200
\]

所以每个 Transformer token：

\[
x\in\mathbb R^{512}
\]

都会经过：

\[
\boxed{
512
\rightarrow
3200
\rightarrow
ReLU
\rightarrow
512
}
\]

严格顺序是：

\[
512
\xrightarrow{Linear}
3200
\xrightarrow{ReLU}
3200
\xrightarrow{Dropout}
3200
\xrightarrow{Linear}
512
\]

---

# 10. MLP 最关键的概念不是“多层”

而是：

\[
\boxed{
\text{function composition}
}
\]

如果：

\[
f_1(x)
=
\phi(
W_1x+b_1
)
\]

\[
f_2(h)
=
\phi(
W_2h+b_2
)
\]

\[
f_3(h)
=
W_3h+b_3
\]

整个网络：

\[
\boxed{
f(x)
=
f_3(
f_2(
f_1(x)
)
)
}
\]

---

# 11. 每一层都在重新表示前一层

Raw input：

\[
x
\]

第一层变成：

\[
h^{(1)}
\]

第二层：

\[
h^{(2)}
\]

最后：

\[
\hat y
\]

所以：

\[
\boxed{
x
\rightarrow
h^{(1)}
\rightarrow
h^{(2)}
\rightarrow
\hat y
}
\]

---

# 12. Hidden Representation 是什么？

假设：

\[
h^{(1)}
=
ReLU(W_1x+b_1)
\]

它就是：

> 网络从原始 input自动构造出来的一组 intermediate features。

这些 features：

> 不需要人工预先命名。

它们只需要：

> 对最终 task有用。

---

# 13. 为什么叫 Hidden Layer？

因为训练数据只直接给我们：

- input：
  \[
  x
  \]
- target：
  \[
  y
  \]

并没有给：

\[
h
\]

一个明确 ground-truth label。

所以中间：

\[
h
\]

是模型内部产生的：

> hidden representation。

---

# 14. Hidden 不是“不可观察”

在代码里当然可以：

```python
print(h)
```

或 hook出来。

“Hidden”主要是指：

> 它不是训练数据直接监督的 observed target layer。

所以：

\[
\boxed{
\text{hidden}
\neq
\text{inaccessible}
}
\]

---

# 15. Hidden Feature 是谁设计的？

不是人手工指定：

```text
h1 = 身高
h2 = 年龄
h3 = 颜色
```

而是：

\[
W,b
\]

通过最终 loss的梯度被训练。

所以：

\[
\boxed{
\text{representation learning}
}
\]

意味着：

> 模型自己学习中间 feature transformation。

---

# 16. 为什么这是神经网络强大的关键？

传统 feature engineering可能需要人手工设计：

\[
\phi_1(x),
\phi_2(x),\ldots
\]

神经网络则让：

\[
W_1,W_2,\ldots
\]

直接从数据中学习：

> 什么 intermediate features有助于最终 prediction。

---

# 17. 一个非常简单的“特征重编码”例子

假设输入：

\[
x=
[x_1,x_2]
\]

第一 hidden neuron：

\[
h_1=
ReLU(x_1+x_2-1)
\]

它可能对：

> “两个 feature的总量是否超过某个 threshold”

敏感。

第二个：

\[
h_2=
ReLU(x_1-x_2)
\]

可能对：

> “x1是否显著大于x2”

敏感。

然后 output layer组合：

\[
h_1,h_2
\]

形成最终任务函数。

---

# 18. 这里没有人告诉 h1/h2 的语义

我们只是人为举例解释权重。

真正训练时：

\[
W_1,b_1
\]

由 optimizer学习。

所以隐藏特征的语义可能：

- distributed；
- entangled；
- polysemantic；
- difficult to name。

---

# 19. Representation Learning 不等于每个 Neuron 都有清晰语义

这是重要纠错。

模型可以用：

> 多个 neurons共同编码一个因素。

也可以：

> 一个 neuron参与多个不同特征。

所以：

\[
\boxed{
\text{learned representation}
\neq
\text{one neuron = one human concept}
}
\]

---

# 20. MLP 为什么叫 Feedforward Network？

因为信息在 forward computation中：

\[
x
\rightarrow
h_1
\rightarrow
h_2
\rightarrow
y
\]

沿一个有向无环计算图向前传。

没有：

> recurrent feedback loop。

---

# 21. Feedforward 和 Backprop 不矛盾

“Feedforward”描述：

> inference / forward computation graph。

Backpropagation描述：

> training时计算梯度的方法。

所以一个 feedforward network：

> 完全可以使用 backward gradient computation训练。

---

# 22. Forward Propagation 是什么？

Forward propagation：

> 给定 input和当前 parameters，从网络最前面一层层算到 output。

例如：

\[
z^{(1)}
=
W_1x+b_1
\]

\[
h^{(1)}
=
ReLU(z^{(1)})
\]

\[
z^{(2)}
=
W_2h^{(1)}+b_2
\]

\[
\hat y=z^{(2)}
\]

---

# 23. 然后才计算 Loss

假设 regression：

\[
L
=
\frac12(
\hat y-y
)^2
\]

其中：

\[
y
\]

是真实 target。

Loss回答：

> 当前 prediction有多差。

---

# 24. Loss 本身不会自动修改 Weight

这是非常基础但非常重要。

算出：

\[
L=10
\]

不会神奇地让：

\[
W
\]

改变。

还需要知道：

\[
\boxed{
\frac{\partial L}{\partial W}
}
\]

也就是：

> weight往哪个方向变化会让 loss改变多少。

---

# 25. 这就是 Backpropagation 的任务

Backpropagation的核心不是：

> “把 loss这个数字倒着传”。

而是：

\[
\boxed{
\text{高效计算计算图中所有参数的梯度}
}
\]

利用：

> Chain Rule。

---

# 26. 一个最小完整 Network

为了真正理解，

先考虑：

\[
x
\]

是 scalar。

第一层：

\[
z_1=w_1x+b_1
\]

ReLU：

\[
h=ReLU(z_1)
\]

第二层：

\[
\hat y=w_2h+b_2
\]

Loss：

\[
L=
\frac12(
\hat y-y
)^2
\]

---

# 27. Forward Example

取：

\[
x=2
\]

\[
y=5
\]

参数：

\[
w_1=1
\]

\[
b_1=0
\]

\[
w_2=2
\]

\[
b_2=0
\]

第一层：

\[
z_1=1(2)=2
\]

---

# 28. ReLU

因为：

\[
z_1=2>0
\]

所以：

\[
h=2
\]

---

# 29. Output

\[
\hat y
=
w_2h
=
2(2)
=
4
\]

---

# 30. Loss

\[
L
=
\frac12(4-5)^2
\]

\[
=
\frac12
\]

所以：

\[
\boxed{
L=0.5
}
\]

---

# 31. 现在我们想知道 w2 应该怎么改

Chain Rule：

\[
\frac{\partial L}{\partial w_2}
=
\frac{\partial L}{\partial \hat y}
\frac{\partial\hat y}{\partial w_2}
\]

---

# 32. 第一项

\[
L=
\frac12(\hat y-y)^2
\]

所以：

\[
\frac{\partial L}{\partial\hat y}
=
\hat y-y
\]

当前：

\[
4-5=-1
\]

---

# 33. 第二项

\[
\hat y=w_2h+b_2
\]

所以：

\[
\frac{\partial\hat y}{\partial w_2}
=
h
=
2
\]

因此：

\[
\boxed{
\frac{\partial L}{\partial w_2}
=
(-1)(2)
=
-2
}
\]

---

# 34. Gradient 为负是什么意思？

如果 gradient：

\[
-2
\]

说明：

> 稍微增加 \(w_2\)，loss会下降。

Gradient descent：

\[
w_2
\leftarrow
w_2-\eta
\frac{\partial L}{\partial w_2}
\]

例如：

\[
\eta=0.1
\]

则：

\[
w_2
\leftarrow
2-0.1(-2)
\]

\[
=
2.2
\]

---

# 35. 那 w1 怎么知道最终预测错了？

这才是 Backprop最有意思的地方。

\[
w_1
\]

离 loss很远。

路径：

\[
w_1
\rightarrow
z_1
\rightarrow
h
\rightarrow
\hat y
\rightarrow
L
\]

所以：

\[
\frac{\partial L}{\partial w_1}
=
\frac{\partial L}{\partial\hat y}
\frac{\partial\hat y}{\partial h}
\frac{\partial h}{\partial z_1}
\frac{\partial z_1}{\partial w_1}
\]

---

# 36. 一项一项算

我们已有：

\[
\frac{\partial L}{\partial\hat y}
=
-1
\]

---

\[
\hat y=w_2h
\]

所以：

\[
\frac{\partial\hat y}{\partial h}
=
w_2
=
2
\]

---

# 37. ReLU Gradient

当前：

\[
z_1=2>0
\]

所以：

\[
\frac{\partial h}{\partial z_1}
=
1
\]

---

# 38. 第一层

\[
z_1=w_1x+b_1
\]

所以：

\[
\frac{\partial z_1}{\partial w_1}
=
x
=
2
\]

---

# 39. 全部乘起来

\[
\frac{\partial L}{\partial w_1}
=
(-1)(2)(1)(2)
\]

\[
\boxed{
=-4
}
\]

因此第一层也收到：

> 来自最终 task loss的学习信号。

---

# 40. 这就是 End-to-End Learning

训练数据没有告诉第一层：

> “你应该学什么 feature。”

它只告诉网络：

\[
y=5
\]

但最终 loss通过 chain rule：

> 一层层计算每个参数对最终 error的影响。

于是：

\[
W_1
\]

也能被更新。

这就是：

\[
\boxed{
\text{end-to-end differentiable learning}
}
\]

---

# 41. 如果 ReLU 当时 Inactive 呢？

假设：

\[
z_1<0
\]

则：

\[
h=0
\]

同时：

\[
\frac{\partial h}{\partial z_1}=0
\]

所以：

\[
\frac{\partial L}{\partial w_1}
=
\cdots\times0\times\cdots
=
0
\]

---

# 42. 这就是 Activation 决定 Gradient Route

ReLU不仅改变 forward value。

它还改变：

> backward中的 Jacobian。

Active：

\[
1
\]

Inactive：

\[
0
\]

所以：

\[
\boxed{
\text{forward gating}
\leftrightarrow
\text{backward gating}
}
\]

---

# 43. Backpropagation 不等于 Gradient Descent

这两个概念必须严格区分。

## Backpropagation

计算：

\[
\boxed{
\nabla_\theta L
}
\]

也就是 gradients。

---

## Gradient Descent / Adam

利用 gradient更新参数：

\[
\boxed{
\theta
\leftarrow
\theta-\eta\nabla_\theta L
}
\]

或更复杂的 optimizer rule。

---

# 44. 所以训练流程是

```text
Forward
↓
Prediction
↓
Loss
↓
Backpropagation
↓
Gradients
↓
Optimizer Step
↓
Updated Parameters
```

而不是：

```text
Backprop = 参数更新
```

---

# 45. PyTorch 中也完全分开

概念上：

```python
pred = model(x)

loss = criterion(pred, y)

optimizer.zero_grad()

loss.backward()

optimizer.step()
```

---

其中：

```python
loss.backward()
```

主要：

> 计算并累积 gradients。

而：

```python
optimizer.step()
```

才真正：

> 修改 parameters。

---

# 46. `zero_grad()` 又是为什么？

PyTorch gradients默认：

> accumulate。

如果不清零，

上一 batch gradient可能和当前 batch叠加。

所以标准 training loop通常：

```python
optimizer.zero_grad()
```

然后：

```python
loss.backward()
```

---

# 47. 为什么设计成 Accumulate？

因为有时我们故意需要：

> gradient accumulation。

例如显存不够，

可以多个 micro-batches：

```text
backward
backward
backward
```

再统一：

```python
optimizer.step()
```

所以框架默认累积更灵活。

---

# 48. 一个完整 MLP 可以有很多 Hidden Layers

例如：

\[
h^{(1)}
=
ReLU(
W_1x+b_1
)
\]

\[
h^{(2)}
=
ReLU(
W_2h^{(1)}+b_2
)
\]

\[
h^{(3)}
=
ReLU(
W_3h^{(2)}+b_3
)
\]

\[
\hat y
=
W_4h^{(3)}+b_4
\]

---

# 49. 为什么叫 Deep Neural Network？

没有一个全领域统一的严格层数 threshold。

通常：

> 有多个 learned hidden transformations的 neural network

会被称为：

> deep network。

现代模型几十、几百层都很常见。

---

# 50. “几层 MLP”为什么经常数法不一样？

这是一个现实中的命名混乱。

比如：

```text
Linear1
ReLU
Linear2
```

有人叫：

> two-layer MLP

因为有：

\[
2
\]

个 parameterized Linear layers。

---

有人可能口语上说：

> one-hidden-layer MLP

因为只有：

\[
1
\]

个 hidden layer。

这两个说法可以描述：

> 同一个 architecture。

---

# 51. 所以最安全的描述方式

不要只说：

> “2层网络”。

而说：

\[
\boxed{
d_{in}
\rightarrow
d_{hidden}
\rightarrow
d_{out}
}
\]

并明确：

- 几个 hidden layers；
- 几个 Linear layers；
- activation放在哪里。

---

# 52. Transformer FFN 就是经典例子

原始 Transformer：

\[
512
\rightarrow
2048
\rightarrow
512
\]

有：

> 两个 Linear transformations。

中间：

> 一个 hidden layer / activation stage。

所以常叫：

> two-layer feed-forward network。

---

# 53. ACT：

\[
512
\rightarrow
3200
\rightarrow
512
\]

同样：

> two Linear layers + one ReLU hidden stage。

---

# 54. Hidden Layer 的 Width 是什么？

例如：

\[
512
\rightarrow
3200
\rightarrow
512
\]

hidden width：

\[
\boxed{
3200
}
\]

表示：

> 中间有 3200 个 scalar hidden units / channels。

---

# 55. Width 增加带来什么？

更多 hidden units意味着：

> 更多 learned affine responses：

\[
z_j=w_j^\top x+b_j
\]

以及更多：

> nonlinear activation boundaries。

所以更宽网络通常拥有：

> 更大的 representational capacity。

---

# 56. 但 Width 越大不一定永远越好

代价：

- 更多 parameters；
- 更多 FLOPs；
- 更多 memory；
- 可能更易 overfit；
- optimization特性变化。

所以：

\[
\boxed{
\text{capacity}
\neq
\text{guaranteed generalization}
}
\]

---

# 57. Depth 又增加什么？

更深：

\[
x
\rightarrow
h_1
\rightarrow
h_2
\rightarrow
h_3
\]

意味着：

> 后一层可以基于前一层已经提取出的 nonlinear features继续组合。

这允许：

> hierarchical composition。

---

# 58. 一个直观例子

第一层可能形成：

> 简单 feature combinations。

第二层：

> 组合第一层 features。

第三层：

> 再组合更高阶 patterns。

这是一种 useful intuition，

但不要自动认为：

> 每层一定对应某种固定人类层级语义。

---

# 59. Width vs Depth 不是简单互相替代

理论上：

> 很宽的浅网络具有强大的 approximation capacity。

但深度可以用：

> 不同的 function composition结构

更高效地表示某些 functions。

因此现代深度学习不仅追求：

> “能不能表示”。

还关注：

- parameter efficiency；
- optimization；
- inductive bias；
- generalization；
- computation。

---

# 60. Universal Approximation 到底说什么？

这是 MLP最容易被误解的理论之一。

Cybenko 1989 的经典结果表明：

> 某类只有一个 hidden layer、使用连续 sigmoidal nonlinearity的 feedforward networks，可以在 unit hypercube上以任意精度 uniformly approximate任意 continuous function。

关键字很多：

\[
\boxed{
\text{approximate}
}
\]

\[
\boxed{
\text{continuous function}
}
\]

\[
\boxed{
\text{compact-like domain}
}
\]

\[
\boxed{
\text{sufficiently many hidden units}
}
\]

---

# 61. 它不等于“一个 Hidden Layer 永远最好”

Universal Approximation只是表达能力的：

> existence theorem。

它说明：

> 存在某些 parameters可以逼近目标函数。

不说明：

1. 需要多少 hidden units；
2. 是否实际能找到这些 weights；
3. SGD是否容易训练；
4. 数据量是否足够；
5. 泛化是否好；
6. 深网络是否没有优势。

---

# 62. “存在”与“能训练出来”不是一回事

这和我们 Residual Connection文章里讲的一样：

\[
\boxed{
\text{Representability}
\neq
\text{Trainability}
}
\]

一个网络理论上能表示某函数，

优化器：

> 不一定容易找到。

---

# 63. Universal Approximation 也不是“网络能精确表示任何函数”

经典定理说的是：

> 对目标 function，可以把 approximation error做得任意小。

这不等于：

> 有限网络对所有函数都严格 exact equality。

---

# 64. 也不是“训练数据记住了就算 Universal Approximation”

逼近函数的数学命题：

> 和简单 memorization有限样本不同。

不要把：

> interpolation training samples

直接等价：

> uniform function approximation theorem。

---

# 65. Cybenko 原结果主要针对 Sigmoidal Nonlinearity

因此如果我们讨论：

> ReLU network universal approximation，

最好不要错误地说：

> “Cybenko 1989已经直接证明 ReLU版本。”

ReLU不是经典 bounded sigmoid。

后续理论扩展到更广 activation classes。

---

# 66. 更安全的结论

\[
\boxed{
\text{MLPs with suitable nonlinear activations have universal approximation properties under appropriate conditions.}
}
\]

但：

> universal approximation是 capacity theorem，不是 optimization guarantee。

---

# 67. 为什么 Nonlinearity 是定理中的关键？

如果 activation是 identity：

\[
\phi(x)=x
\]

那么无论多少 hidden units：

\[
W_2(W_1x+b_1)+b_2
\]

仍只是 affine。

所以：

\[
\boxed{
\text{nonlinearity is essential for general nonlinear approximation}
}
\]

---

# 68. 为什么 Hidden Layer 能学习 Basis Functions？

一 hidden-layer网络：

\[
f(x)
=
\sum_{j=1}^m
v_j
\phi(
w_j^\top x+b_j
)
+c
\]

可以看成：

> 先产生 \(m\) 个 nonlinear basis-like functions：

\[
\phi(
w_j^\top x+b_j
)
\]

再用：

\[
v_j
\]

线性组合它们。

---

# 69. 对 ReLU 来说

\[
\phi(
w_j^\top x+b_j
)
=
\max(
0,w_j^\top x+b_j
)
\]

每个 hidden unit：

> 提供一个 learned hinge feature。

很多 hinge features组合：

> 形成复杂 piecewise-affine mapping。

---

# 70. 这就是 MLP 的两阶段直觉

### Hidden Layer

学习：

\[
\boxed{
\text{useful nonlinear features}
}
\]

### Output Layer

学习：

\[
\boxed{
\text{how to combine/read out these features}
}
\]

---

# 71. 但深 MLP 不只是“一次 feature extraction + readout”

多 hidden layers：

\[
h_1
\rightarrow
h_2
\rightarrow
h_3
\]

每一层：

> 都在重编码前一层 representation。

所以：

\[
\boxed{
\text{iterative representation transformation}
}
\]

更准确。

---

# 72. MLP 的 Parameters 是什么？

两层 MLP：

\[
f(x)
=
W_2\phi(W_1x+b_1)+b_2
\]

参数：

\[
\boxed{
\theta=
\{W_1,b_1,W_2,b_2\}
}
\]

Activation如 ReLU：

> 没有 trainable parameter。

---

# 73. 参数量怎么算？

假设：

\[
d_{in}=512
\]

\[
d_h=3200
\]

\[
d_{out}=512
\]

第一层：

\[
W_1:
[3200,512]
\]

\[
b_1:
[3200]
\]

参数：

\[
3200\times512+3200
\]

\[
=
1,641,600
\]

---

# 74. 第二层

\[
W_2:
[512,3200]
\]

\[
b_2:
[512]
\]

参数：

\[
512\times3200+512
\]

\[
=
1,638,912
\]

---

# 75. 总参数

\[
1,641,600
+
1,638,912
\]

\[
\boxed{
=
3,280,512
}
\]

所以 ACT一个 FFN MLP：

> 约 3.28M parameters。

---

# 76. ReLU 自己贡献多少参数？

\[
\boxed{
0
}
\]

所以几乎所有参数都在：

> Linear layers。

但没有 ReLU，

这 3.28M parameters整体仍只能形成 affine map。

---

# 77. 这说明 Parameter Count 和 Function Class 是两回事

同样 3.28M weights：

如果没有 nonlinearity：

> 仍可以 collapse。

加入一个零参数 ReLU：

> function class立刻完全不同。

所以：

\[
\boxed{
\text{architecture operations matter as much as parameter count}
}
\]

---

# 78. 一个 Classification MLP

例如输入：

\[
x\in\mathbb R^{784}
\]

hidden：

\[
256
\]

10 classes。

可以：

\[
h=
ReLU(W_1x+b_1)
\]

\[
z=W_2h+b_2
\]

其中：

\[
z\in\mathbb R^{10}
\]

---

# 79. z 是什么？

不是 probability。

它是：

\[
\boxed{
\text{logits}
}
\]

然后：

\[
Softmax(z)
\]

可以变成 class probabilities。

训练时通常：

> CrossEntropyLoss直接接 logits。

---

# 80. 一个 Regression MLP

如果预测：

\[
y\in\mathbb R^3
\]

可以：

\[
h=
ReLU(W_1x+b_1)
\]

\[
\hat y=W_2h+b_2
\]

最后：

> 不一定需要 activation。

因为 regression outputs可能任意 real。

---

# 81. 所以 MLP 最后一层 Activation 取决于任务

### Multiclass classification

Linear logits + Softmax/CrossEntropy。

### Binary

Linear logit + Sigmoid/BCEWithLogits。

### Regression

often Linear output。

### Positive output

可能 Softplus / exponential parameterization。

---

# 82. MLP 不是固定要求“每层都 ReLU”

Hidden activations可以：

- ReLU；
- GELU；
- Tanh；
- Sigmoid；
- SiLU；
- GLU-style gates。

不同 architecture选择不同。

---

# 83. 为什么 Transformer FFN 常被叫 MLP？

因为典型现代写法：

```text
Linear
→ activation
→ Linear
```

符合最经典 MLP block。

只不过 Transformer中：

> 它不是整个模型。

而是：

> Attention block旁边的 per-token nonlinear subnetwork。

---

# 84. 一个 Transformer Layer 的宏观分工

可以再次压缩成：

\[
\boxed{
\text{Attention}
=
\text{token communication}
}
\]

\[
\boxed{
\text{MLP / FFN}
=
\text{per-token nonlinear feature computation}
}
\]

---

# 85. Attention Output 已经 Contextualized

Self-Attention后：

\[
x_i
\]

已经读取其他 tokens。

然后 MLP：

\[
MLP(x_i)
\]

对每一个 token：

> 独立做 nonlinear feature transformation。

所以两者配合：

```text
communicate
↓
compute locally
↓
communicate
↓
compute locally
...
```

---

# 86. 这也是 Transformer 为什么不是“只有 Attention”

即使论文叫：

> Attention Is All You Need，

architecture仍然有：

- Embeddings；
- Positional Encoding；
- MHA；
- FFN/MLP；
- Residual；
- LayerNorm；
- Dropout。

标题主要强调：

> 不再需要 RNN/CNN recurrence/convolution作为 sequence dependency backbone。

不是：

> 整个模型只有 Attention operation。

---

# 87. ACT 中 Policy Encoder 的 MLP 是 Position-Wise

假设 Encoder memory：

\[
X
\in
\mathbb R^{1202\times B\times512}
\]

FFN：

\[
512\rightarrow3200\rightarrow512
\]

对每个：

\[
X[s,b,:]
\]

单独运行同一 MLP。

---

# 88. 所以可以写

\[
\boxed{
Y_{s,b,:}
=
MLP(
X_{s,b,:}
)
}
\]

同一：

\[
W_1,W_2,b_1,b_2
\]

用于所有：

\[
s,b
\]

---

# 89. MLP 不改变 Sequence Length

输入：

\[
[1202,B,512]
\]

Linear1：

\[
[1202,B,3200]
\]

ReLU：

\[
[1202,B,3200]
\]

Linear2：

\[
\boxed{
[1202,B,512]
}
\]

所以：

\[
1202
\]

不变。

---

# 90. Decoder 同理

\[
[k,B,512]
\]

例如：

\[
[100,B,512]
\]

经过 FFN：

\[
[100,B,3200]
\]

然后：

\[
\boxed{
[100,B,512]
}
\]

100 action slots不变。

---

# 91. MLP 会让 Action Slots 互相交流吗？

不会。

如果 slot 0和slot 1需要交流：

> Decoder Self-Attention负责。

MLP只是：

> 分别加工每个 slot已经拥有的 contextual features。

---

# 92. 为什么参数共享很有用？

如果给每个 token position单独一套 MLP：

\[
W^{(1)}_{position}
\]

参数会随：

\[
N
\]

暴涨，

且不能自然泛化到不同 positions。

共享同一 MLP意味着：

> 学到的是 feature-space rule，而不是某个 position专属 rule。

---

# 93. 但 Positional 信息仍然可以影响 MLP

虽然 MLP没有显式 position parameter，

它接收到的 hidden vector：

> 已经可能包含 positional/contextual information。

所以同一个 MLP：

> 对不同 positions仍会产生不同 output。

---

# 94. 这和卷积 Weight Sharing 有一点类似

CNN：

> 同一个 kernel在不同 spatial locations共享。

Transformer FFN：

> 同一个 MLP在不同 token positions共享。

但两者 operation不同：

- CNN kernel会跨邻域；
- FFN只跨 feature dimension。

所以只是：

> parameter sharing的类比。

---

# 95. MLP 和 CNN 最大区别之一

Dense MLP：

\[
y=Wx+b
\]

每个 output feature通常连接所有 input features。

CNN：

> 利用 local connectivity与spatial weight sharing。

因此对图像：

> CNN有更强 inductive bias与效率。

---

# 96. MLP 能处理图像吗？

当然可以。

把 image flatten：

\[
H\times W\times C
\]

成 vector：

\[
x
\]

然后 MLP。

但：

- 参数量巨大；
- 不显式利用局部 spatial structure。

所以传统视觉中 CNN更合适。

现代 MLP-Mixer等架构又探索了：

> 如何用结构化 MLP高效处理图像。

---

# 97. MLP 能处理 Sequence 吗？

可以，

但朴素 fixed-size MLP如果直接 flatten整个 sequence：

> 参数通常和 sequence length绑定。

Transformer选择：

- Attention处理 token mixing；
- position-wise MLP处理 feature mixing；

是更结构化的设计。

---

# 98. MLP 中 “Perceptron” 与 Rosenblatt Perceptron 要不要严格区分？

要。

经典 Rosenblatt perceptron通常是：

> threshold-based binary classifier / learning algorithm。

现代 MLP：

> 使用 differentiable / subdifferentiable activations和 backprop训练多层网络。

历史上有关联，

但不是：

> 直接把单层 perceptron机械叠起来这么简单。

---

# 99. Perceptron 的经典限制

单层 linear decision boundary无法表示：

> XOR。

这是神经网络史上著名例子。

因为 XOR：

> 不是 linearly separable。

---

# 100. XOR 为什么重要？

输入：

\[
(0,0)\rightarrow0
\]

\[
(0,1)\rightarrow1
\]

\[
(1,0)\rightarrow1
\]

\[
(1,1)\rightarrow0
\]

不存在一条直线：

> 把正类和负类完全分开。

---

# 101. Hidden Layer 可以解决 XOR

Hidden units可以构造：

> 非线性 intermediate features。

然后 output layer再对这些 hidden features做 linear separation。

所以：

\[
\boxed{
\text{representation transformation can make a non-linearly-separable raw problem linearly readable}
}
\]

这是神经网络最经典的直觉之一。

---

# 102. 一个很重要的 Representation Learning 思维

Raw input space里：

> task可能很难。

模型不一定直接在 raw space里画复杂 boundary。

它可以先学：

\[
h=f(x)
\]

让 task在：

\[
h\text{-space}
\]

变得简单。

然后 final Linear head：

\[
Wh+b
\]

就够。

---

# 103. 这也是为什么深模型最后常常只有 Linear Head

复杂性已经被前面：

> representation network

吸收了。

最终：

> 简单 readout即可。

ACT也是：

\[
decoder\ hidden\ state
\rightarrow
Linear(512,14)
\]

输出 action。

---

# 104. Backprop 为什么叫 “Back-Propagation of Errors”？

1986 Rumelhart, Hinton, Williams 的著名工作标题：

> **Learning representations by back-propagating errors**

其核心思想是：

> 用 output error的信息通过 differentiable network向后计算 weight changes。

现代更精确的语言：

\[
\boxed{
\text{reverse-mode differentiation / gradient backpropagation through the computation graph}
}
\]

---

# 105. 但不要把 “Error” 当成一个数字原样往回传

例如 output error：

\[
\hat y-y
\]

到了第一层时，

它已经被沿路径乘上：

- 后续 weights；
- activation derivatives；
- normalization Jacobians；
- 等。

所以第一层收到的是：

\[
\boxed{
\frac{\partial L}{\partial W_1}
}
\]

而不是简单复制：

\[
\hat y-y
\]

---

# 106. 一个 Vector MLP 的 Backprop

两层：

\[
z_1=W_1x+b_1
\]

\[
h=\phi(z_1)
\]

\[
\hat y=W_2h+b_2
\]

loss：

\[
L(\hat y,y)
\]

---

# 107. 从 Output 开始

定义：

\[
\delta_2
=
\frac{\partial L}{\partial \hat y}
\]

---

# 108. 第二层 Parameter Gradient

\[
\hat y=W_2h+b_2
\]

所以：

\[
\boxed{
\frac{\partial L}{\partial W_2}
=
\delta_2h^\top
}
\]

以及：

\[
\boxed{
\frac{\partial L}{\partial b_2}
=
\delta_2
}
\]

---

# 109. 把 Gradient 传给 Hidden Representation

\[
\boxed{
\frac{\partial L}{\partial h}
=
W_2^\top\delta_2
}
\]

这就是：

> output loss对 hidden features的敏感度。

---

# 110. 穿过 Activation

\[
h=\phi(z_1)
\]

elementwise时：

\[
\boxed{
\delta_1
=
\frac{\partial L}{\partial z_1}
=
\left(
W_2^\top\delta_2
\right)
\odot
\phi'(z_1)
}
\]

---

# 111. 第一层 Gradient

\[
\boxed{
\frac{\partial L}{\partial W_1}
=
\delta_1x^\top
}
\]

\[
\boxed{
\frac{\partial L}{\partial b_1}
=
\delta_1
}
\]

这就是最经典两层 MLP backprop equation。

---

# 112. ReLU 时

\[
\phi'(z_1)
=
1[z_1>0]
\]

所以：

\[
\boxed{
\delta_1
=
(
W_2^\top\delta_2
)
\odot
1[z_1>0]
}
\]

Inactive units：

> gradient被 gate成0。

---

# 113. 这说明 Backprop 和 Forward Activation Pattern 紧密相关

Forward：

\[
1[z_j>0]
\]

决定：

> 哪些 hidden units参与 output。

Backward：

同样这个 mask决定：

> 哪些 hidden units接收 gradient。

---

# 114. 自动微分做的就是这些 Chain Rule

现代 PyTorch：

```python
loss.backward()
```

不需要我们手动写这些 derivatives。

但它底层本质：

> 沿 computation graph使用局部 derivative应用 chain rule。

---

# 115. Autograd 不等于“AI自己理解数学”

每个 primitive operation都有：

> 已知 backward derivative rule。

例如：

- matrix multiplication；
- addition；
- ReLU；
- Softmax；
- LayerNorm。

Autograd系统按 graph组合这些局部规则。

---

# 116. 为什么 Backprop 高效？

如果有：

\[
P
\]

个 parameters，

最笨的方法可以逐个 parameter：

> perturb一下、重新 forward，

计算 finite difference。

那会极其昂贵。

Backprop / reverse-mode AD：

> 一次反向遍历就可以高效获得大量参数对一个 scalar loss的梯度。

这正适合 neural networks：

\[
\text{many parameters}
\rightarrow
\text{one scalar loss}
\]

---

# 117. Numerical Gradient 与 Backprop 不同

Finite difference：

\[
\frac{\partial L}{\partial\theta_i}
\approx
\frac{
L(\theta_i+\epsilon)-L(\theta_i-\epsilon)
}{
2\epsilon
}
\]

通常只用于：

> gradient checking。

训练大网络不使用这种方法，

因为成本太高且有数值误差。

---

# 118. Training Loop 真正发生什么？

每个 batch：

### Step 1

取数据：

\[
(x,y)
\]

### Step 2

Forward：

\[
\hat y=f_\theta(x)
\]

### Step 3

Loss：

\[
L(\hat y,y)
\]

### Step 4

Backward：

\[
\nabla_\theta L
\]

### Step 5

Optimizer：

\[
\theta\leftarrow Update(\theta,\nabla_\theta L)
\]

然后下一个 batch。

---

# 119. 为什么叫 Epoch？

如果训练 dataset有：

\[
N
\]

个 samples，

模型把：

> 整个训练集大致看过一遍

通常称：

\[
\boxed{
1\ epoch
}
\]

---

# 120. Batch Size 是什么？

一次 parameter update使用：

\[
B
\]

个 samples。

例如：

\[
B=32
\]

则把32个样本的 loss / gradient结合后：

> 更新一次参数。

---

# 121. Full Batch / Mini-Batch / SGD

### Full Batch

每次使用：

> 整个 dataset。

### Stochastic

理论最窄意义：

> 每次1 sample。

### Mini-Batch

现代常用：

> 每次一小批 samples。

实际工程中“SGD training”常宽泛包括：

> mini-batch stochastic optimization。

---

# 122. 为什么 Mini-Batch？

平衡：

- compute parallelism；
- gradient noise；
- memory；
- update frequency。

GPU特别适合：

> batch matrix operations。

---

# 123. MLP 的 Loss 由 Task 决定

MLP architecture：

> 不自动决定 loss。

Regression可能：

\[
MSE
\]

Classification：

\[
CrossEntropy
\]

Binary：

\[
BCE
\]

ACT：

> L1 reconstruction + KL。

所以：

\[
\boxed{
\text{network function}
\neq
\text{training objective}
}
\]

---

# 124. 同一个 MLP 可以完成不同任务

只要改变：

- output dimension；
- target；
- loss；

同一个基本 MLP framework可以用于：

- classification；
- regression；
- representation learning；
- generative models；
- policy networks。

---

# 125. 为什么 Task Loss 能塑造 Hidden Representation？

因为：

\[
\frac{\partial L}{\partial W_1}
\]

依赖：

> 最终 output loss。

所以 hidden layer没有 direct label，

仍被：

> downstream task pressure

训练。

这就是 representation learning的核心机制。

---

# 126. 如果加 Auxiliary Loss 会怎样？

某个 hidden representation：

\[
h
\]

如果额外接一个 auxiliary head与loss，

那么：

\[
h
\]

会同时收到：

> 多个 objectives的 gradients。

这会改变其 learned representation。

---

# 127. ACT CVAE 就是一个很好的例子

Policy decoder最终 action loss：

> 塑造大量 network parameters。

CVAE latent encoder还同时受到：

\[
KL
\]

约束。

所以相同 hidden computations：

> 可能同时受到 reconstruction与regularization signals。

---

# 128. MLP 的 Hidden Dimension 是人为设计还是学习出来？

Width：

\[
3200
\]

这种 dimension：

> 是 hyperparameter，由人设定。

但每个 hidden unit的：

\[
W,b
\]

是：

> 训练学习。

所以：

\[
\boxed{
\text{architecture size is designed;}
\quad
\text{feature functions are learned.}
}
\]

---

# 129. 为什么不让模型自己决定 Hidden Width？

可以研究：

- pruning；
- neural architecture search；
- sparsity；
- dynamic width；
- MoE。

但标准 MLP：

> width固定。

模型只学参数值。

---

# 130. MLP 是否每个 Hidden Unit 都必须连到所有 Input Features？

经典 dense MLP：

> 是。

这就是：

> fully connected。

但可以设计：

- sparse MLP；
- grouped linear；
- convolution；
- low-rank layers。

所以“MLP”通常默认 dense connectivity，

但神经网络不必都 dense。

---

# 131. 为什么 Dense MLP 参数容易很多？

如果：

\[
d_{in}=10000
\]

\[
d_h=10000
\]

一个 weight matrix：

\[
10000^2
=
100M
\]

weights。

所以对高维 structured data，

architecture通常利用结构降低参数。

---

# 132. Transformer FFN 为什么反而愿意很宽？

因为 token hidden width：

\[
d_{\text{model}}
\]

已经是抽象 representation。

FFN只对：

> 每个 token独立。

所以宽 MLP提供强大的 feature transformation能力，

而不需要 flatten整个 sequence。

---

# 133. MLP 和 Attention 的参数依赖差异

Position-wise MLP：

参数量：

> 与 sequence length无关。

Attention Q/K/V/O parameters：

> 也主要与 hidden dimension有关。

但 Attention activation computation：

\[
N^2
\]

随 sequence length平方增长。

MLP activation compute：

\[
O(Nd_{\text{model}}d_{ff})
\]

对 \(N\) 线性增长。

---

# 134. ACT 1202 Token Encoder 中，FFN Compute 很可观

每个 token：

\[
512\rightarrow3200\rightarrow512
\]

大约两次大 matrix multiply。

1202 tokens全部做。

所以即使 Attention很显眼，

FFN也可能占据：

> 大量参数和 FLOPs。

---

# 135. 为什么现代 Transformer FFN 经常比 Attention 参数更多？

标准 MHA major weights约：

\[
4d_{\text{model}}^2
\]

FFN：

\[
2d_{\text{model}}d_{ff}
\]

如果：

\[
d_{ff}\approx4d_{\text{model}}
\]

那么 FFN：

\[
\approx8d_{\text{model}}^2
\]

比一组 MHA：

\[
4d_{\text{model}}^2
\]

更多。

ACT：

\[
d_{ff}=3200
\]

相对于512更宽，

差距更明显。

---

# 136. 这就是为什么“大模型=Attention参数”是错误的

Transformer能力来自整个 block：

- Attention；
- MLP；
- residual；
- norm；
- embeddings；
- depth。

MLP是其中非常大的 component。

---

# 137. MLP 的 Initialization 为什么重要？

深网络每层：

\[
h_{l+1}=\phi(W_lh_l+b_l)
\]

如果 weights太大：

> activations/gradients可能爆炸。

太小：

> 信号可能衰减。

所以 initialization会显著影响：

> 训练是否容易。

---

# 138. Glorot & Bengio 2010 为什么重要？

他们研究了：

> deep feedforward networks从random initialization训练困难的问题。

论文分析：

- activation saturation；
- activation/gradient across layers；
- layer Jacobian singular values；

并提出后来被称为：

> Xavier / Glorot initialization

的初始化方法。

---

# 139. 这说明 MLP 不是“有 Backprop 就一定好训”

理论上 gradient可算：

\[
\neq
\]

实际深网络优化稳定。

训练还依赖：

- initialization；
- activation；
- normalization；
- residual；
- optimizer；
- learning rate。

---

# 140. 为什么后来需要 Residual Connection？

普通深 MLP/CNN：

> 深度增加可能越来越难优化。

ResNet通过：

\[
x+F(x)
\]

提供 identity path，

让深网络更易训练。

Transformer同样大量依赖 residual。

---

# 141. 所以现代网络不是纯 MLP Stack

经典 MLP是理解 foundation。

现代 architecture会在它之上加入：

- convolution；
- attention；
- residual；
- normalization；
- gating；
- routing。

但这些复杂模型仍大量依赖：

> MLP式 nonlinear feature transformations。

---

# 142. MLP 是否可以没有 Bias？

可以：

\[
h=\phi(Wx)
\]

但 bias通常增加：

> boundary translation flexibility。

ReLU时尤其：

\[
w^\top x+b=0
\]

bias决定 activation hyperplane偏移。

---

# 143. 如果所有 ReLU 层没有 Bias 会怎样？

所有第一层 activation boundaries：

\[
w^\top x=0
\]

都经过 origin。

网络仍然可以很有表达能力，

但 geometry受到额外约束。

---

# 144. MLP 中 Bias 是否每个 Token 一套？

在 Transformer FFN：

> 不是。

Linear的：

\[
b_1\in\mathbb R^{3200}
\]

\[
b_2\in\mathbb R^{512}
\]

对所有 tokens共享。

---

# 145. 但不同 Token 输出不同

因为：

\[
x_i
\]

不同。

所以：

\[
W_1x_i+b_1
\]

不同。

共享参数不是：

> 共享 activation。

---

# 146. Batch 中也共享参数

所有 samples：

\[
x^{(1)},\ldots,x^{(B)}
\]

使用同一个：

\[
W,b
\]

这是 machine learning generalization的基本前提之一：

> 学同一个 function，应用于不同 examples。

---

# 147. 如果每个 Training Sample 一套独立参数会怎样？

模型很容易：

> 直接 memorise sample-specific mapping，

但无法自然应用到新样本。

共享 function parameters：

> 强迫模型寻找跨样本规律。

---

# 148. MLP 的 Output 是 Deterministic 吗？

如果没有：

- Dropout；
- stochastic latent sampling；
- other randomness；

固定：

\[
x,\theta
\]

则：

\[
f_\theta(x)
\]

deterministic。

---

# 149. Dropout 会让 Training MLP Stochastic

如果：

\[
h=
Dropout(
ReLU(
W_1x+b_1
)
)
\]

训练时 mask随机。

Eval：

> Dropout关闭。

---

# 150. ACT FFN 正是这样

训练：

\[
Linear
\rightarrow
ReLU
\rightarrow
Dropout
\rightarrow
Linear
\]

所以同一 token和weights：

> training forward可能因 Dropout略不同。

Inference：

> Dropout关闭。

---

# 151. MLP 本身是不是 Probabilistic Model？

不是必然。

普通 MLP只是：

> deterministic parametric function。

如果它输出：

- probability distribution parameters；
- logits；
- mean/variance；

才可以成为 probabilistic model的一部分。

---

# 152. VAE 中的 Encoder MLP 也是这个道理

一个 MLP可能输出：

\[
\mu,\log\sigma^2
\]

但 MLP本身：

> 不等于 VAE。

是后续 probabilistic interpretation与loss：

> 让这些 outputs成为 distribution parameters。

---

# 153. MLP Output “语义”仍然来自 Computation Graph

和上一篇 Linear完全一致。

例如：

\[
MLP:512\rightarrow256\rightarrow32
\]

这32维可以被训练成：

- embedding；
- action；
- latent mean；
- logits；

取决于：

> downstream use与loss。

---

# 154. 一个最小 PyTorch MLP

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)
```

数学：

\[
\boxed{
f(x)
=
W_2ReLU(W_1x+b_1)+b_2
}
\]

---

# 155. 训练一个 Regression MLP

概念代码：

```python
model = MLP()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

criterion = nn.MSELoss()

for x, y in loader:
    pred = model(x)

    loss = criterion(
        pred,
        y
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
```

---

# 156. 第一次训练前 W 是“知识”吗？

不是我们通常说的 learned task knowledge。

它只是：

> random initialization。

第一次 forward甚至可能很差。

---

# 157. 学习发生在哪里？

重复：

```text
prediction
↓
loss
↓
gradient
↓
parameter update
```

以后：

\[
W
\]

逐渐调整。

所以：

\[
\boxed{
\text{trained model}
=
\text{architecture + learned parameters}
}
\]

---

# 158. Architecture 与 Parameters 必须区分

Architecture：

```text
2 → 8 → 1
ReLU
```

参数：

\[
W_1,b_1,W_2,b_2
\]

很多不同 parameter settings：

> 共用同一个 architecture。

---

# 159. Hyperparameters 又是什么？

例如：

- hidden width = 8；
- number of layers = 2；
- learning rate；
- batch size；
- activation choice。

这些通常：

> 不由普通 gradient descent直接训练。

而由设计/搜索确定。

---

# 160. Parameter vs Hyperparameter

### Parameter

\[
W,b
\]

通过 training gradient学习。

### Hyperparameter

例如：

\[
d_h=3200
\]

人为设定。

这是非常基础的 distinction。

---

# 161. Loss 下降意味着 Hidden Representation 一定“更有意义”吗？

只意味着：

> 对当前 objective更有用。

不保证：

- 更可解释；
- 更符合人类语义；
- 更因果；
- 更公平；
- 更robust。

模型优化的是：

> 给定 objective。

---

# 162. 所以 Objective Design 非常重要

如果 loss只奖励：

> 某种 shortcut behavior，

模型完全可能学 shortcut。

MLP本身不会自动选择：

> “人类希望的正确推理方式”。

---

# 163. 为什么 Training Data 也决定 Representation？

Gradient来自：

> 当前 samples。

如果 dataset存在：

- bias；
- spurious correlation；
- missing cases；

模型会根据这些数据规律塑造 hidden features。

所以：

\[
\boxed{
\text{learned representation}
=
\text{architecture + data + objective + optimization}
}
\]

---

# 164. MLP 为什么能 Overfit？

如果参数很多，

可以学习非常复杂的函数，

甚至：

> 拟合训练数据里的噪声。

所以训练 loss很低：

> 不等于 test性能好。

---

# 165. Overfitting 的基本形式

训练：

\[
L_{train}
\downarrow
\]

但 validation/test：

> 不继续改善甚至变差。

这表示模型：

> 对训练数据规律拟合得太具体。

---

# 166. Regularization 如何帮助？

常见：

- weight decay；
- dropout；
- data augmentation；
- early stopping；
- smaller model；
- more data。

它们以不同方式：

> 控制 effective fitting behavior。

---

# 167. ACT FFN 中 Dropout 就是一种 Regularization

hidden：

\[
3200
\]

ReLU后：

> training随机 drop部分 activations。

这迫使 MLP：

> 不过度依赖固定 hidden feature combination。

---

# 168. MLP 是否需要 Normalization？

不是定义上的必须。

最经典 MLP可以只有：

- Linear；
- Activation。

但深网络训练中常加入：

- BatchNorm；
- LayerNorm；
- RMSNorm。

Transformer则把：

> LayerNorm放在 MLP block周围。

---

# 169. ACT Post-LN FFN 完整结构

设 contextual token：

\[
x
\]

来自上一 sub-layer norm。

FFN：

\[
f=
Linear_2(
Dropout(
ReLU(
Linear_1(x)
)
)
)
\]

然后：

\[
r=
x+
Dropout(f)
\]

最后：

\[
\boxed{
y=LayerNorm(r)
}
\]

---

# 170. 所以 ACT MLP 本身不是整个 FFN Block 的全部

严格区分：

### MLP Core

\[
Linear_1
\rightarrow
ReLU
\rightarrow
Dropout
\rightarrow
Linear_2
\]

### Transformer FFN Sublayer

还包括：

- residual-output dropout；
- residual addition；
- LayerNorm placement。

---

# 171. 为什么这个 distinction 有用？

读代码时：

```python
linear1
activation
dropout
linear2
```

是：

> nonlinear feedforward computation。

而：

```python
src + dropout2(src2)
norm2
```

属于：

> Transformer sublayer wrapper。

不要把所有东西都叫 MLP。

---

# 172. ACT Decoder 一层有几个 MLP？

每个 decoder layer：

> 一个 FFN/MLP core。

7 layers：

\[
7
\]

个不同 MLPs，

参数不共享。

---

# 173. Encoder 呢？

Policy Encoder：

\[
4
\]

层，

每层一个独立 MLP。

CVAE Encoder：

\[
4
\]

层，

也每层一个独立 MLP。

---

# 174. 它们的 Architecture 可以相同，Parameters 不同

每个可能都是：

\[
512\rightarrow3200\rightarrow512
\]

但：

\[
W_1^{layer1}
\neq
W_1^{layer2}
\]

一般成立。

---

# 175. 为什么不共享所有 Layer 的 MLP？

可以设计共享式 architecture，

例如 Universal Transformer等思路。

但 original Transformer / ACT：

> layers使用独立 parameters。

这提高：

> layer-specific transformation capacity。

---

# 176. MLP 是否“记忆”了训练数据？

Parameters确实编码训练过程中学到的信息。

但“记忆”可能指：

- generalizable pattern；
- exact memorization；
- distributed feature structure。

不能仅凭：

> 参数很多

就判断它以何种方式记忆。

---

# 177. Deep Learning 为什么叫 “Representation Learning”？

因为：

> 不只是学习 final output weights。

中间所有：

\[
W_l
\]

也被 task loss更新，

所以每一层：

> 都在学习数据的内部 representation。

---

# 178. “Hidden Representation”不是额外存进数据库的 Feature

它是：

> forward时根据 input动态算出来的 activation。

例如：

\[
h(x)
=
ReLU(Wx+b)
\]

换一个 input：

\[
x'
\]

就得到：

\[
h(x')
\]

---

# 179. Parameters 与 Representation 再区分一次

Parameters：

\[
W,b
\]

在 inference中固定。

Representation / activation：

\[
h(x)
\]

随 input变化。

这是读神经网络时极其重要的 distinction。

---

# 180. Attention Weight 也属于 Activation，不是 Parameter

同理：

\[
A(X)
\]

每个 sequence动态生成。

Q/K/V projection matrices：

\[
W_Q,W_K,W_V
\]

才是 learned parameters。

---

# 181. MLP 中 Activation 也不是 Parameter

\[
h
=
ReLU(Wx+b)
\]

当前：

\[
h
\]

是动态 activation。

\[
W,b
\]

是 parameters。

---

# 182. 为什么这对模型解释很重要？

如果你说：

> “模型学了一个 h=0.7”

不准确。

0.7可能只是：

> 某个样本的一次 activation。

真正长期储存在模型中的：

> 是 parameters。

---

# 183. MLP Training 是否等于记住一个 Lookup Table？

不是一般情况。

Dense MLP定义的是：

\[
f_\theta(x)
\]

一个 continuous / piecewise-continuous mapping。

它对没见过的：

\[
x
\]

也会根据 learned function产生 output。

是否 generalize好：

> 是另一问题。

---

# 184. 为什么 MLP 可以 Interpolate？

训练点之间，

ReLU network：

> 通过 learned piecewise-affine function自然定义中间区域输出。

这和纯 sample lookup不同。

---

# 185. 但高维空间的 Generalization 非常复杂

不能只用：

> “插值”

简单解释现代 neural networks全部行为。

Generalization涉及：

- data manifold；
- inductive bias；
- optimization；
- scale；
- regularization。

这里先不展开。

---

# 186. MLP 的 Function 是连续的吗？

如果 activation是 ReLU，

Linear和ReLU都连续，

所以有限 ReLU MLP：

\[
\boxed{
\text{continuous piecewise-affine}
}
\]

---

# 187. Sigmoid/Tanh MLP 呢？

这些 activation平滑，

所以 network通常：

> smooth nonlinear function。

不同 activation产生：

> 不同 function geometry和gradient properties。

---

# 188. MLP 的 Jacobian 会随 Input 改变

ReLU network：

\[
f(x)
=
W_2ReLU(W_1x+b_1)+b_2
\]

在非边界点：

\[
\boxed{
J_f(x)
=
W_2D(x)W_1
}
\]

其中：

\[
D(x)
\]

由 active units决定。

---

# 189. 这和 Affine Model 最大区别

Affine：

\[
J_f(x)=W
\]

到处一样。

MLP：

\[
J_f(x)
\]

随 input region改变。

所以：

> 不同 input可能具有不同 local sensitivity。

---

# 190. 更深 Network 的 Jacobian

例如：

\[
h_1=\phi(W_1x)
\]

\[
h_2=\phi(W_2h_1)
\]

\[
y=W_3h_2
\]

Jacobain：

\[
\boxed{
J
=
W_3D_2W_2D_1W_1
}
\]

ReLU masks：

\[
D_1,D_2
\]

随 input变化。

---

# 191. 这也解释了 Gradient Vanishing / Exploding 的来源之一

Backward中会出现大量：

- weight matrices；
- activation Jacobians；

的连乘。

如果整体 singular values过小：

> gradient衰减。

过大：

> gradient放大。

这也是深网络训练研究的核心问题。

---

# 192. ReLU 不是唯一影响 Gradient 的因素

还有：

- weights；
- initialization；
- LayerNorm；
- Residual；
- optimizer。

所以深网络 gradient dynamics：

> 是整个 architecture的共同结果。

---

# 193. 为什么 Residual 改变这个连乘？

纯 stack：

\[
J_l
\]

不断相乘。

Residual：

\[
x_{l+1}=x_l+F_l(x_l)
\]

Jacobian：

\[
I+J_{F_l}
\]

多出 identity component。

这就是 Residual文章里讲的优化优势之一。

---

# 194. Transformer 其实是 “Attention + MLP + Residual” 的反复组合

可以写成一个高层图：

```text
representation
↓
Attention communication
↓
Residual / Norm
↓
MLP computation
↓
Residual / Norm
↓
next layer
```

---

# 195. ACT 也完全遵循这个模式

Policy Encoder：

```text
1202 observation tokens
↓
Multi-Head Self-Attention
↓
Residual + LayerNorm
↓
Position-wise MLP
↓
Residual + LayerNorm
```

重复：

\[
4
\]

次。

---

# 196. Decoder：

```text
action slots
↓
Self-Attention
↓
Cross-Attention to observation memory
↓
MLP
↓
next decoder layer
```

每个 sublayer都有 residual/norm wrapper。

---

# 197. 为什么 Decoder Cross-Attention 后还需要 MLP？

Cross-Attention解决：

> 从 observation memory读什么。

读回来后：

> 这些 information还需要在每个 action slot内部做 nonlinear feature processing。

MLP负责：

> 重新组合这些 contextual features。

---

# 198. 可以不用 MLP 吗？

当然可以设计 ablation。

但那会降低 architecture capacity。

Original Transformer的标准 block：

> Attention + FFN都属于核心组件。

不能理解成：

> MLP只是可有可无的小尾巴。

---

# 199. 为什么 MLP 不需要看到 Raw Image？

ACT中 raw images先：

> ResNet backbone → visual tokens → Transformer。

MLP处理的是：

\[
512\text{-D hidden representation}
\]

不是原始：

\[
480\times640\times3
\]

pixels。

---

# 200. 这体现 Hierarchical Processing

不同模块各自处理：

- ResNet：视觉 feature extraction；
- Attention：跨 token/context通信；
- MLP：hidden feature transformation；
- Action head：readout成 robot action。

---

# 201. 一个完整 ACT Action Slot 到 MLP

假设 Cross-Attention之后某个 action slot：

\[
x_i\in\mathbb R^{512}
\]

Linear1：

\[
z_i=W_1x_i+b_1
\]

\[
z_i\in\mathbb R^{3200}
\]

---

# 202. ReLU

\[
h_i=
ReLU(z_i)
\]

\[
h_i\in\mathbb R^{3200}
\]

产生：

> 当前 action slot特有的 activation pattern。

---

# 203. Dropout

训练：

\[
\tilde h_i=
Dropout(h_i)
\]

推理：

\[
\tilde h_i=h_i
\]

---

# 204. Linear2

\[
f_i=
W_2\tilde h_i+b_2
\]

\[
f_i\in\mathbb R^{512}
\]

这就是：

> FFN residual update。

---

# 205. Residual

\[
r_i=x_i+Dropout(f_i)
\]

---

# 206. LayerNorm

Post-LN canonical ACT：

\[
\boxed{
y_i=LN(r_i)
}
\]

然后：

> 进入下一 Transformer stage。

---

# 207. 所以 MLP 不直接输出 Robot Action

它输出：

\[
512
\]

维 hidden update。

最终所有 decoder layers结束后：

\[
512\rightarrow14
\]

action head才真正生成 joint targets。

---

# 208. 这也解释为什么 Hidden Layer 没有直接 Ground Truth

没有 target说：

> FFN第1734维应该等于0.6。

它只通过最终：

> action loss

被间接训练。

---

# 209. Common Misconception 1：MLP 就是 Linear Regression 堆很多层

**错误。**

没有 nonlinear activation时：

> 确实会退化成一个 affine map。

真正 MLP表达力来自：

\[
\boxed{
\text{Affine + Nonlinearity Composition}
}
\]

---

# 210. Common Misconception 2：层数越多一定越强

表达 capacity通常会变，

但实际性能还取决于：

- optimization；
- data；
- regularization；
- architecture。

更深也可能：

> 更难训练。

---

# 211. Common Misconception 3：Hidden Layer 就是“模型的隐藏知识库”

**不准确。**

Hidden activation：

> 是当前 input动态产生的 representation。

长期存储在模型中的是：

> parameters。

---

# 212. Common Misconception 4：一个 Hidden Neuron一定对应一个 Concept

**不保证。**

Representations可能高度 distributed。

---

# 213. Common Misconception 5：Backprop 就是把预测误差数字倒着传

**错误。**

传的是：

> 通过 chain rule计算出的 gradients / adjoints。

---

# 214. Common Misconception 6：Backprop 会直接修改 Parameters

**错误。**

Backprop算：

\[
\nabla_\theta L
\]

Optimizer才更新。

---

# 215. Common Misconception 7：`loss.backward()` 就等于 `optimizer.step()`

**错误。**

两者必须严格区分。

---

# 216. Common Misconception 8：Forward Propagation 只发生在 Inference

**错误。**

Training也先 forward，

才能得到 prediction和loss。

---

# 217. Common Misconception 9：训练时只有 Backward，没有 Forward

**错误。**

完整训练：

\[
Forward
\rightarrow
Loss
\rightarrow
Backward
\rightarrow
Update
\]

---

# 218. Common Misconception 10：Universal Approximation 说明一个 Hidden Layer 足够，所以 Deep Learning 没意义

**错误。**

定理是：

> existence / approximation capacity result。

不讨论 practical efficiency、optimization等全部问题。

---

# 219. Common Misconception 11：Universal Approximation 说明 SGD 一定找到那个解

**错误。**

Existence：

\[
\neq
\]

optimization guarantee。

---

# 220. Common Misconception 12：Universal Approximation 说明有限网络精确等于任何函数

**错误。**

经典结论讨论：

> arbitrarily good approximation under conditions。

---

# 221. Common Misconception 13：Cybenko 1989 直接证明了任意 ReLU MLP 定理

**不准确。**

经典 Cybenko结果主要针对：

> continuous sigmoidal nonlinearities。

ReLU universal approximation有后续更一般理论。

---

# 222. Common Misconception 14：MLP 的 Output 必须做 Softmax

**错误。**

取决于任务。

Regression常直接 Linear output。

---

# 223. Common Misconception 15：所有 Hidden Layers 都必须 ReLU

**错误。**

Activation是 architecture choice。

---

# 224. Common Misconception 16：MLP 只能做 Classification

**错误。**

它可以用于几乎任何 vector-to-vector differentiable modeling任务。

---

# 225. Common Misconception 17：MLP 不能处理 Batch

**错误。**

Linear天然对 leading dimensions并行处理。

---

# 226. Common Misconception 18：Transformer FFN 不是 MLP，因为它在 Transformer 里面

**错误。**

它本质就是：

> position-wise two-layer MLP / feed-forward network。

---

# 227. Common Misconception 19：ACT 的 3200 是 3200 个 Tokens

**错误。**

它是：

> 每个 token内部的 hidden feature dimension。

---

# 228. Common Misconception 20：FFN 会让 ACT 的 1202 Tokens 互相交流

**错误。**

跨 token交流主要由 Attention。

---

# 229. Common Misconception 21：共享 MLP 参数意味着所有 Tokens 输出一样

**错误。**

Input不同，

activation pattern不同。

---

# 230. Common Misconception 22：一个 MLP Layer 的参数会随 Input 改变

**标准 MLP不会。**

Weights固定，

activations动态。

---

# 231. Common Misconception 23：Hidden Representation 是 Parameter

**错误。**

Representation：

> activation。

Parameter：

> \(W,b\)。

---

# 232. Common Misconception 24：参数越多必然 Generalization 越好

**错误。**

Capacity增大也可能 overfit。

---

# 233. Common Misconception 25：训练 Loss 越低就说明模型理解得越正确

**错误。**

它只说明：

> 对训练 objective拟合得好。

---

# 234. Common Misconception 26：ReLU MLP 是全局 Linear，因为每个区域都是 Linear

**错误。**

跨区域 rule改变，

所以全局 nonlinear。

---

# 235. Common Misconception 27：MLP 的所有 Layer 可以一次矩阵乘完成

如果中间有 nonlinear activation：

> 一般不能合并。

---

# 236. Common Misconception 28：Two-Layer MLP 一定有两个 Hidden Layers

**错误。**

常见命名是：

> 两个 Linear layers、一个 hidden stage。

---

# 237. Common Misconception 29：One-Hidden-Layer 与 Two-Layer Network 一定冲突

不一定。

可能是：

> 同一个 network的两种计数习惯。

---

# 238. Common Misconception 30：MLP 是过时结构，Transformer 已经不用了

**完全错误。**

现代 Transformer核心 block仍含巨大 MLP/FFN。

---

# 239. 一张图理解最基本 MLP

```text
Input x
  │
  ▼
┌─────────────────┐
│ Linear W₁,b₁    │
└─────────────────┘
  │
  ▼
z₁
  │
  ▼
┌─────────────────┐
│ ReLU            │
└─────────────────┘
  │
  ▼
Hidden h
  │
  ▼
┌─────────────────┐
│ Linear W₂,b₂    │
└─────────────────┘
  │
  ▼
Prediction ŷ
```

数学：

\[
\boxed{
\hat y
=
W_2ReLU(W_1x+b_1)+b_2
}
\]

---

# 240. 一张图理解 Training

```text
x
│
▼
MLP
│
▼
ŷ
│
├──────── compare with target y
│
▼
Loss L
│
▼
Backpropagation
│
▼
∂L/∂W₁
∂L/∂b₁
∂L/∂W₂
∂L/∂b₂
│
▼
Optimizer
│
▼
Updated parameters
```

---

# 241. 一张图理解 Representation Learning

```text
Raw input x
    │
    ▼
learned affine transform
    │
    ▼
nonlinear gating
    │
    ▼
hidden representation h₁
    │
    ▼
learned transform
    │
    ▼
hidden representation h₂
    │
    ▼
simple readout
    │
    ▼
task output
```

模型的重点不只是：

> “最后预测什么”。

也在不断学习：

> “怎样重新表示输入，使最终任务变容易”。

---

# 242. 一张图理解 ACT 中的 MLP

```text
one contextual ACT token
x ∈ R^512
       │
       ▼
Linear 512 → 3200
       │
       ▼
3200 learned pre-activation features
       │
       ▼
ReLU
       │
       ▼
input-dependent active feature subset
       │
       ▼
Dropout (training only)
       │
       ▼
Linear 3200 → 512
       │
       ▼
FFN update
       │
       ▼
Residual + LayerNorm
```

这套 MLP：

> 对每个 token共享参数、独立执行。

---

# 243. 如果只记住一个公式

最经典 MLP：

\[
\boxed{
f_\theta(x)
=
W_2
\phi(
W_1x+b_1
)
+
b_2
}
\]

其中：

\[
\theta=
\{
W_1,b_1,W_2,b_2
\}
\]

Activation：

\[
\phi
\]

让两层 Linear：

> 无法合并。

---

# 244. 如果只记住一个 Training 公式

训练的目标：

\[
\boxed{
\theta^*
=
\arg\min_\theta
\mathbb E_{(x,y)}
[
L(
f_\theta(x),y
)
]
}
\]

实际用 finite dataset / mini-batches近似。

---

# 245. 如果只记住一个 Backprop 思想

对于复合函数：

\[
L(
f_3(
f_2(
f_1(x)
)
)
)
\]

Chain Rule让我们从后往前计算：

\[
\boxed{
\frac{\partial L}{\partial \theta_l}
}
\]

每层参数都能收到：

> 最终 task objective的学习信号。

---

# 246. 一句话真正理解 MLP

> **MLP 不是简单“堆很多神经元”，而是把多个 learned affine transformations 与 nonlinear activations复合成一个可训练的函数：Linear layers学习怎样重新组合当前 features，activation让不同输入进入不同计算区域，从而形成 input-dependent nonlinear mappings；最终 task loss再通过 backpropagation 的 chain rule把梯度传到所有中间层，使 hidden representations无需人工标签就能逐步被塑造成对任务有用的内部坐标。**

---

# 247. 一句话理解为什么 MLP 能学 Representation

> **第一层并不知道自己应该提取什么 feature，第二层也没有隐藏层 ground truth；它们之所以逐渐形成有用表示，是因为最终 prediction error对每个中间 activation都有可微依赖，backprop能够计算“如果这个 hidden feature改变一点，最终 loss会怎样变化”，再继续把这种敏感度转换成每个 weight的梯度，因此 supervision可以跨多层间接塑造内部 representation。**

---

# 248. 一句话连接 Transformer

> **Transformer 并没有取代 MLP，而是把 MLP放进每一个 block：Attention负责让 tokens彼此交换信息，position-wise MLP则对每一个已经 contextualized 的 token执行 \(d_{\text{model}}\rightarrow d_{ff}\rightarrow d_{\text{model}}\) 的 nonlinear feature transformation；因此 Transformer可以理解为 communication 与 computation 两种机制交替堆叠，而 FFN/MLP正是其中主要的 per-token computation engine。**

---

# 249. 一句话连接 ACT

> **ACT 中每个 512-D visual/joint/latent memory token与每个 512-D action-slot representation都会反复进入 \(512\rightarrow3200\rightarrow ReLU\rightarrow512\) 的 MLP；这些 MLP不负责跨 token通信，而是根据每个 token当前 contextual state激活不同的3200维中间 feature subset，再把它们组合成512维 residual update，因此即使所有 positions共享同一 FFN参数，不同观察位置和未来动作位置仍能执行完全不同的有效 nonlinear computation。**

---

# 250. 下一篇：Backpropagation

现在我们已经第一次看到了完整训练闭环：

\[
Forward
\rightarrow
Loss
\rightarrow
Backward
\rightarrow
Optimizer
\]

但这一篇只把 Backprop讲到了：

> 足够理解 MLP训练。

下一步应该把它单独拆成 canonical page：

> **[Backpropagation：Loss 到底怎样一路传回第一层？](./backpropagation.md)**

会从：

\[
\frac{dL}{dw}
=
\frac{dL}{dy}
\frac{dy}{dw}
\]

开始，完整讲：

- computational graph；
- local derivative；
- chain rule；
- upstream gradient；
- downstream gradient；
- vector-Jacobian product；
- reverse-mode automatic differentiation；
- 为什么 backward不是“反向运行网络”；
- 为什么 forward要保存 activations；
- gradient accumulation；
- detach；
- `requires_grad`；
- `loss.backward()`；
- `torch.no_grad()`；
- 为什么 scalar loss对百万参数特别适合 reverse mode；
- MLP完整手算一次 backprop；
- Attention / Softmax / LayerNorm的gradient如何被同一机制统一起来；
- ACT 的 L1 + KL 怎样同时把 gradient送入 decoder、policy encoder、CVAE encoder以及 \(\mu/\log\sigma^2\) heads。

---

## Core Textbook Source：Deep Feedforward Networks

Ian Goodfellow, Yoshua Bengio, Aaron Courville.

**Deep Learning.**  
MIT Press, 2016.

- Book: https://www.deeplearningbook.org/
- Chapter 6: Deep Feedforward Networks

Chapter 6 是现代 feedforward neural networks / MLP 的经典教材来源之一。

核心框架包括：

- feedforward networks定义从 input到 output的 mapping；
- intermediate hidden units构成 internal representations；
- network通过 compositions of functions构造；
- training通过 gradient-based learning和 back-propagation计算所需 derivatives；
- output unit / loss设计依赖 task。

本文以这一现代定义作为 MLP 主线。

---

## Historical Backpropagation Source

David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams.

**Learning representations by back-propagating errors.**  
Nature 323, 533–536, 1986.

- DOI: https://doi.org/10.1038/323533a0
- Geoffrey Hinton publication page: https://www.cs.toronto.edu/~hinton/papers.html

这篇经典工作展示了：

> output error information可以通过多层 differentiable network反向计算 connection-weight changes，从而训练 hidden units学习内部 representations。

历史上 Backprop还有更早的发展来源，

因此不应简单说：

> “1986论文发明了所有 Backprop思想”。

更准确：

> Rumelhart–Hinton–Williams 1986是使多层神经网络反向传播训练广为人知的关键经典工作之一。

---

## Universal Approximation Source

George Cybenko.

**Approximation by Superpositions of a Sigmoidal Function.**  
Mathematics of Control, Signals and Systems, 1989.

- DOI: https://doi.org/10.1007/BF02551274
- Springer: https://link.springer.com/article/10.1007/BF02551274

论文证明：

> finite linear combinations of compositions of a fixed univariate nonlinearity with affine functionals，在适当条件下可以 uniformly approximate unit hypercube上 continuous functions。

对 single-hidden-layer networks，

论文特别讨论：

> continuous sigmoidal nonlinearities。

因此本文只把 Cybenko结果用于说明：

\[
\boxed{
\text{shallow nonlinear feedforward networks can have universal approximation power}
}
\]

而不错误声称：

> 该1989 theorem直接就是现代 ReLU network的全部 universal approximation理论。

---

## Deep-Network Optimization Background

Xavier Glorot, Yoshua Bengio.

**Understanding the difficulty of training deep feedforward neural networks.**  
AISTATS 2010.

- PMLR: https://proceedings.mlr.press/v9/glorot10a.html
- PDF: https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

论文研究：

- deep feedforward networks从random initialization训练困难；
- nonlinear activation saturation；
- activations / gradients across layers；
- Jacobian singular-value behavior；
- initialization对 convergence的影响。

并提出后来广泛称为：

> Xavier / Glorot initialization

的初始化方案。

它说明：

\[
\boxed{
\text{having a differentiable MLP and backprop is not enough to guarantee easy deep optimization}
}
\]

---

## ReLU Source

Xavier Glorot, Antoine Bordes, Yoshua Bengio.

**Deep Sparse Rectifier Neural Networks.**  
AISTATS 2011.

- PMLR: https://proceedings.mlr.press/v15/glorot11a.html
- PDF: https://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf

ReLU / rectifier nonlinearities使 deep feedforward networks形成：

> sparse, nonlinear hidden representations。

本文 MLP使用 ReLU作为主要 pedagogical activation，

与 ACT canonical Transformer FFN保持一致。

---

## Transformer Primary Source

Ashish Vaswani et al.

**Attention Is All You Need.**  
NeurIPS 2017.

- arXiv: https://arxiv.org/abs/1706.03762
- PDF: https://arxiv.org/pdf/1706.03762

Section 3.3：

\[
\boxed{
FFN(x)
=
\max(
0,xW_1+b_1
)
W_2+b_2
}
\]

原始 Base Transformer：

\[
d_{\text{model}}=512
\]

\[
d_{ff}=2048
\]

因此其每个 position-wise FFN：

> 本质上就是共享参数的 two-layer ReLU MLP。

---

## ACT Primary Source

Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.

**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.**  
RSS 2023.

- arXiv: https://arxiv.org/abs/2304.13705
- PDF: https://arxiv.org/pdf/2304.13705

ACT Table III：

\[
d_{\text{model}}=512
\]

\[
d_{ff}=3200
\]

所以 ACT Transformer FFN：

\[
\boxed{
512
\rightarrow
3200
\rightarrow
ReLU
\rightarrow
512
}
\]

并配合：

- dropout；
- residual；
- LayerNorm。

---

## ACT Official Implementation

Repository:

https://github.com/tonyzhaozh/act

Transformer:

https://github.com/tonyzhaozh/act/blob/main/detr/models/transformer.py

Encoder / Decoder FFN核心：

```python
self.linear1 =
    nn.Linear(
        d_model,
        dim_feedforward
    )

self.dropout =
    nn.Dropout(
        dropout
    )

self.linear2 =
    nn.Linear(
        dim_feedforward,
        d_model
    )
```

forward：

```python
src2 =
    self.linear2(
        self.dropout(
            self.activation(
                self.linear1(src)
            )
        )
    )
```

canonical activation：

```python
activation="relu"
```

因此核心就是：

\[
\boxed{
MLP(x)
=
Linear_2(
Dropout(
ReLU(
Linear_1(x)
)
)
)
}
\]

---

## 本文知识连接

### 数学

- Function Composition
- Derivative
- Partial Derivative
- Gradient
- Chain Rule
- Jacobian
- Matrix Multiplication

### Deep Learning

- [Linear Layer](./linear-layer.md)
- [ReLU](./relu.md)
- Activation Function
- [Backpropagation](./backpropagation.md)
- Gradient Descent
- Loss Function
- [Feed-Forward Network](./feed-forward-network.md)
- Xavier Initialization
- Overfitting
- Regularization
- [Dropout](./dropout.md)

### Transformer

- [Transformer](./transformer.md)
- [Attention](./attention.md)
- [Transformer Encoder](./transformer-encoder.md)
- [Transformer Decoder](./transformer-decoder.md)
- [Feed-Forward Network](./feed-forward-network.md)
- [Residual Connection](./residual-connection.md)
- [Layer Normalization](./layer-normalization.md)

### Robot Learning

- [ACT Architecture](../robot-learning/act/architecture.md)
- [ACT Training](../robot-learning/act/training.md)
- [ACT Inference](../robot-learning/act/inference.md)
- [ACT Complete Data Flow](../robot-learning/act/complete-data-flow.md)

### 下一步

- [Backpropagation](./backpropagation.md)
