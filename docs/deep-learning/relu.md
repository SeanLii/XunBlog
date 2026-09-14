---
title: "ReLU：为什么只做 max(0,x)，就能让神经网络学非线性？"
description: "从线性函数的定义出发，严格理解 ReLU 为什么是非线性但又是分段线性的、它如何通过 learned hyperplanes 让 MLP 对不同输入区域使用不同 affine rules、梯度如何传播、为什么会产生稀疏激活与 dying ReLU，并连接到 Transformer / ACT 的 512→3200→512 FFN。"
status: reviewed
pageType: concept
canonical: /deep-learning/relu
updated: "2026-09-15"
---

# ReLU：为什么只做 \(\max(0,x)\)，就能让神经网络学非线性？

上一篇 [Linear Layer](./linear-layer.md) 里，我们得到一个非常重要的结论：

如果网络只有：

\[
Linear
\rightarrow
Linear
\rightarrow
Linear
\]

那么不管堆多少层，

最终都可以合并成：

\[
\boxed{
y=Wx+b
}
\]

也就是一个 affine transformation。

这会产生一个非常直接的问题：

> **既然 Linear Layer 这么受限制，神经网络到底是从哪里获得复杂非线性能力的？**

在 ACT 的 Transformer FFN 中，答案看起来简单得有些离谱：

\[
\boxed{
ReLU(x)=\max(0,x)
}
\]

就这么一个函数：

- 输入负数 → 变成 0；
- 输入正数 → 原样保留。

原始 Transformer 的 FFN：

\[
\boxed{
FFN(x)
=
\max(0,xW_1+b_1)W_2+b_2
}
\]

ACT 也是同一结构，只是 hidden width 从原始 Transformer 的：

\[
2048
\]

变成：

\[
3200
\]

于是一个非常值得认真理解的问题出现了：

> **为什么只是在 0 的地方“折一下”，就足以让神经网络从单一 affine mapping 变成可以表达复杂函数的深度模型？**

这一篇会完整回答：

1. 什么叫 Linear / Nonlinear？
2. ReLU 为什么严格来说是 nonlinear？
3. 为什么它同时又叫 piecewise linear？
4. “非线性”和“分段线性”为什么不矛盾？
5. 一个 ReLU neuron 到底把输入空间切成了什么？
6. 为什么：
   \[
   w^\top x+b=0
   \]
   是一条 activation boundary？
7. 多个 ReLU neurons 如何把空间切成很多 regions？
8. 为什么每个 region 内整个 network 又是 affine 的？
9. 为什么这仍然能逼近复杂曲线？
10. ReLU 的 derivative 是什么？
11. \(x=0\) 不可导，神经网络为什么还能训练？
12. PyTorch 在 \(x=0\) 用什么 gradient？
13. ReLU 为什么比 sigmoid/tanh 更不容易在正区间饱和？
14. 什么是 dying ReLU？
15. “ReLU 解决梯度消失”为什么是过度简化？
16. 为什么 ReLU 会产生很多精确的 0？
17. 为什么 sparse activation 不等于 sparse parameters？
18. ACT 的 \(512\rightarrow3200\rightarrow ReLU\rightarrow512\) 到底发生了什么？
19. 为什么 ACT 的 ReLU 不会让最终 action 只能是正数？
20. ReLU、Leaky ReLU、GELU 又是什么关系？

---

# 1. ReLU 的定义

Rectified Linear Unit：

\[
\boxed{
ReLU(x)=\max(0,x)
}
\]

分段写：

\[
\boxed{
ReLU(x)
=
\begin{cases}
0,&x\le0\\
x,&x>0
\end{cases}
}
\]

图像非常简单：

```text
y
↑
│              /
│            /
│          /
│        /
│      /
│_____/____________→ x
     0
```

左边：

\[
x<0
\]

全部压成：

\[
0
\]

右边：

\[
x>0
\]

保持：

\[
y=x
\]

---

# 2. PyTorch ReLU 就是这个函数

官方：

```python
nn.ReLU()
```

逐元素执行：

\[
\boxed{
ReLU(x)=\max(0,x)
}
\]

输入 shape：

\[
(*)
\]

输出 shape：

\[
(*)
\]

完全不改变 tensor shape。

所以：

\[
[B,N,3200]
\]

经过 ReLU后仍然：

\[
\boxed{
[B,N,3200]
}
\]

只是数值发生变化。

---

# 3. ReLU 是逐 Element 的

如果：

\[
x=
[-2,\,0.5,\,-1,\,3]
\]

那么：

\[
ReLU(x)
=
[0,\,0.5,\,0,\,3]
\]

它不会：

- 求平均；
- 做矩阵乘；
- 混合 feature dimensions。

每个元素：

\[
x_i
\]

单独执行：

\[
\max(0,x_i)
\]

---

# 4. 为什么这个函数叫“Rectified”？

“Rectify”可以理解成：

> 把负值截掉 / 整流。

电子学里的 rectifier 会让信号主要只在某一方向通过。

ReLU也类似：

\[
x\le0
\]

被压到：

\[
0
\]

而正半轴通过。

所以名字：

> Rectified Linear Unit。

---

# 5. 为什么它又叫“Linear Unit”？

因为在 active region：

\[
x>0
\]

它就是：

\[
y=x
\]

也就是一条直线。

在 inactive region：

\[
x<0
\]

则：

\[
y=0
\]

也是一条水平直线。

因此 ReLU 是：

\[
\boxed{
\text{piecewise linear}
}
\]

---

# 6. 但它为什么又是 Nonlinear？

这里是最关键的数学区分。

严格 linear function：

\[
f
\]

必须满足：

\[
f(x+y)=f(x)+f(y)
\]

和：

\[
f(cx)=cf(x)
\]

对所有允许的：

\[
x,y,c
\]

成立。

---

# 7. ReLU 不满足 Additivity

取：

\[
x=1
\]

\[
y=-1
\]

则：

\[
ReLU(x+y)
=
ReLU(0)
=
0
\]

但是：

\[
ReLU(1)+ReLU(-1)
=
1+0
=
1
\]

所以：

\[
\boxed{
ReLU(x+y)
\neq
ReLU(x)+ReLU(y)
}
\]

因此它不是 linear transformation。

---

# 8. ReLU 也不满足所有 Scalar Homogeneity

取：

\[
x=1
\]

\[
c=-1
\]

左边：

\[
ReLU(cx)
=
ReLU(-1)
=
0
\]

右边：

\[
cReLU(x)
=
-1(1)
=
-1
\]

所以：

\[
\boxed{
ReLU(cx)
\neq
cReLU(x)
}
\]

对负 scalar不成立。

---

# 9. 但 ReLU 有 Positive Homogeneity

如果：

\[
c\ge0
\]

那么：

\[
\boxed{
ReLU(cx)=cReLU(x)
}
\]

例如：

\[
ReLU(3x)
=
3ReLU(x)
\]

当：

\[
3>0
\]

这叫：

> positive homogeneity。

它是 ReLU一个很重要的数学性质，

但：

> **positive homogeneous 不等于 linear。**

---

# 10. “Nonlinear”与“Piecewise Linear”完全不矛盾

ReLU整个函数：

\[
\max(0,x)
\]

不是一条全局直线。

但它可以分成两个区域：

### Region 1

\[
x<0
\]

\[
ReLU(x)=0
\]

Linear/Affine。

### Region 2

\[
x>0
\]

\[
ReLU(x)=x
\]

Linear。

---

所以：

\[
\boxed{
\text{globally nonlinear}
}
\]

同时：

\[
\boxed{
\text{locally piecewise linear}
}
\]

完全可以同时成立。

---

# 11. 真正的 Nonlinearity 就发生在“折点”

ReLU 的特殊点：

\[
x=0
\]

在左边 slope：

\[
0
\]

在右边 slope：

\[
1
\]

也就是说：

> 经过 0 以后，函数使用了不同的线性规则。

这一个：

\[
\boxed{
\text{kink / hinge}
}
\]

就是最简单的非线性来源。

---

# 12. 为什么一个折点这么重要？

因为一个 affine function：

\[
y=ax+b
\]

整个实数轴上只有：

> 同一个 slope。

而：

\[
ReLU(x)
\]

允许：

```text
左边 slope = 0
右边 slope = 1
```

所以：

> 不同 input region 使用不同 linear rule。

这就是神经网络非线性表达能力的最小核心。

---

# 13. 先看一个 ReLU Neuron

一个神经元通常先计算 pre-activation：

\[
z=w^\top x+b
\]

再 ReLU：

\[
\boxed{
h=
ReLU(z)
=
\max(0,w^\top x+b)
}
\]

---

# 14. 什么时候这个 Neuron Active？

如果：

\[
w^\top x+b>0
\]

那么：

\[
h=w^\top x+b
\]

这个 neuron：

> active。

---

# 15. 什么时候 Inactive？

如果：

\[
w^\top x+b\le0
\]

则：

\[
h=0
\]

这个 neuron：

> inactive。

---

# 16. Active / Inactive 的边界是什么？

正好是：

\[
\boxed{
w^\top x+b=0
}
\]

这在高维空间中是一个：

> hyperplane。

---

# 17. 在 2D 中就是一条直线

如果：

\[
x=
[x_1,x_2]
\]

那么：

\[
w_1x_1+w_2x_2+b=0
\]

是一条直线。

这条线把平面分成：

\[
w^\top x+b>0
\]

和：

\[
w^\top x+b<0
\]

两边。

---

# 18. 一边 Neuron 工作，一边 Neuron 为 0

例如：

\[
h=
ReLU(x_1+x_2-1)
\]

边界：

\[
x_1+x_2=1
\]

如果：

\[
x_1+x_2>1
\]

则：

\[
h=x_1+x_2-1
\]

如果：

\[
x_1+x_2\le1
\]

则：

\[
h=0
\]

所以这个 neuron对 input space实行：

> **条件性的 affine response。**

---

# 19. 这就是 ReLU 神经元真正“加条件”的地方

没有 ReLU：

\[
h=w^\top x+b
\]

无论 input在哪里：

> 都使用同一个 affine rule。

有 ReLU：

\[
h=
\begin{cases}
0,&w^\top x+b\le0\\
w^\top x+b,&w^\top x+b>0
\end{cases}
\]

于是：

\[
\boxed{
\text{the effective rule depends on the input region}
}
\]

---

# 20. 多个 ReLU Neurons 会怎样？

假设 hidden layer有：

\[
m
\]

个 ReLU units：

\[
h_j
=
ReLU(w_j^\top x+b_j)
\]

每一个 neuron都产生一条：

\[
w_j^\top x+b_j=0
\]

hyperplane。

---

# 21. 这些 Hyperplanes 会一起切分 Input Space

例如 2D 中三条不同直线：

```text
        \   /
---------\-/-------
          X
---------/-\-------
        /   \
```

会把平面切成：

> 多个不同 regions。

在每个 region里，

每个 ReLU neuron都有一个固定状态：

```text
active / inactive
```

---

# 22. 一个 Activation Pattern

假设有4个 ReLU units。

某个 input region中可能：

```text
Neuron 1: active
Neuron 2: inactive
Neuron 3: active
Neuron 4: inactive
```

写成：

\[
\boxed{
[1,0,1,0]
}
\]

另一个 region可能：

\[
[1,1,0,0]
\]

---

# 23. 为什么 Activation Pattern 很重要？

因为一旦 active/inactive pattern固定，

每个 ReLU都知道：

- active：
  \[
  ReLU(z)=z
  \]
- inactive：
  \[
  ReLU(z)=0
  \]

所以在这个 region里：

> 所有 ReLU都退化成固定 linear selection。

因此整个 network：

\[
\boxed{
\text{在单个 region 内是 affine / linear-piece}
}
\]

---

# 24. 但跨 Region 时 Rule 会改变

当 input跨过某条：

\[
w_j^\top x+b_j=0
\]

hyperplane，

第 \(j\) 个 ReLU：

> 从 inactive切换成 active，

或者反过来。

于是整个 network的 effective affine function：

> 发生改变。

所以整体是：

\[
\boxed{
\text{piecewise affine}
}
\]

---

# 25. 这就是 ReLU Network 的核心几何

可以把一个 ReLU网络想成：

> 用 learned hyperplanes把输入/hidden space切成许多区域。

每个区域内部：

\[
y=A_rx+c_r
\]

使用一套 affine rule。

不同 region：

\[
r
\]

拥有不同：

\[
A_r,c_r
\]

---

# 26. 为什么这足以表达“曲线”？

因为很多小线段拼起来：

> 可以逼近曲线。

例如圆弧：

> 可以用很多短直线段近似。

一个 nonlinear function：

> 也可以用大量 piecewise-linear segments近似。

所以 ReLU网络虽然每个局部 region都很简单，

组合起来可以形成：

> 非常复杂的 global function。

---

# 27. 一个真正能手算的 1D 例子

构造：

\[
\boxed{
f(x)
=
ReLU(x)
-
2ReLU(x-1)
+
ReLU(x-2)
}
\]

我们看看它到底是什么。

---

# 28. Region 1：\(x\le0\)

三个 ReLU：

\[
ReLU(x)=0
\]

\[
ReLU(x-1)=0
\]

\[
ReLU(x-2)=0
\]

所以：

\[
\boxed{
f(x)=0
}
\]

---

# 29. Region 2：\(0<x\le1\)

此时：

\[
ReLU(x)=x
\]

但：

\[
ReLU(x-1)=0
\]

\[
ReLU(x-2)=0
\]

所以：

\[
\boxed{
f(x)=x
}
\]

---

# 30. Region 3：\(1<x\le2\)

此时：

\[
ReLU(x)=x
\]

\[
ReLU(x-1)=x-1
\]

\[
ReLU(x-2)=0
\]

所以：

\[
f(x)
=
x-2(x-1)
\]

\[
=
x-2x+2
\]

\[
\boxed{
f(x)=2-x
}
\]

---

# 31. Region 4：\(x>2\)

三个都 active：

\[
f(x)
=
x
-
2(x-1)
+
(x-2)
\]

展开：

\[
x-2x+2+x-2
\]

\[
\boxed{
f(x)=0
}
\]

---

# 32. 所以完整函数

\[
\boxed{
f(x)
=
\begin{cases}
0,&x\le0\\
x,&0<x\le1\\
2-x,&1<x\le2\\
0,&x>2
\end{cases}
}
\]

图像：

```text
y
↑
1        /\
        /  \
       /    \
0 ____/      \________→ x
     0   1    2
```

---

# 33. 三个 ReLU 已经造出了一个“山峰”

单个 affine function：

> 永远做不到这种先上升、再下降、再归零的形状。

但三个 ReLU basis functions：

\[
ReLU(x-c)
\]

组合以后就做到了。

这就是：

\[
\boxed{
\text{piecewise-linear basis construction}
}
\]

---

# 34. ReLU 可以被理解成一个 Hinge Basis

\[
ReLU(x-c)
=
\max(0,x-c)
\]

在：

\[
x=c
\]

之前为：

\[
0
\]

之后增加 slope：

\[
1
\]

所以它像一个：

> 从某个 breakpoint开始生效的 hinge function。

多个不同：

\[
c
\]

的 ReLU组合，

可以控制不同区间的 slope。

---

# 35. 这就是 1D 中 ReLU Network 的直观本质

每加一个 ReLU：

> 就可能增加一个新的 breakpoint。

然后后续 Linear Layer：

> 决定每个 breakpoint之后 slope怎样改变。

所以很多 ReLU组合可以形成：

> 很复杂的折线函数。

---

# 36. 高维就是把“折点”推广成“折面”

1D：

\[
wx+b=0
\]

是一个点。

2D：

> 一条线。

3D：

> 一个平面。

更高维：

> hyperplane。

所以高维 ReLU network：

> 用大量 learned hyperplanes切分空间。

---

# 37. Deep ReLU Network 为什么比单层更复杂？

第一层先切出 regions。

第二层看到的是：

> 第一层经过折叠/门控后的 representation。

它又可以用新的 hyperplanes：

> 在已经变换过的空间中继续切。

一层层复合后：

> region structure可以变得非常复杂。

---

# 38. 不要简单说“L 层就只有 L 个折点”

错误。

每层可能有大量 neurons，

并且后层 boundary作用于：

> 前层 nonlinear representation。

所以 region数量可以随着：

- width；
- depth；

快速增长。

精确 upper/lower bounds是独立理论研究主题。

---

# 39. 我们当前需要记住的不是 Region 数量公式

而是：

\[
\boxed{
\text{ReLU turns one global affine rule into input-dependent piecewise affine rules.}
}
\]

这就是它提供非线性表达能力的核心。

---

# 40. 回到一个 MLP

假设：

\[
h=
ReLU(W_1x+b_1)
\]

然后：

\[
y=W_2h+b_2
\]

即：

\[
\boxed{
y=
W_2ReLU(W_1x+b_1)+b_2
}
\]

---

# 41. 如果没有 ReLU

\[
y=
W_2(W_1x+b_1)+b_2
\]

可以合并：

\[
y=W'x+b'
\]

全局只有一个 affine rule。

---

# 42. 有 ReLU 后

定义 active mask：

\[
D(x)
=
diag(
m_1(x),\ldots,m_m(x)
)
\]

其中：

\[
m_j(x)
=
\begin{cases}
1,&z_j>0\\
0,&z_j\le0
\end{cases}
\]

那么在某个固定 activation region内：

\[
ReLU(z)=D z
\]

---

# 43. 于是该 Region 内

\[
y
=
W_2D(W_1x+b_1)+b_2
\]

展开：

\[
\boxed{
y
=
W_2DW_1x
+
W_2Db_1
+
b_2
}
\]

这就是一个 affine function。

---

# 44. 关键是 D 取决于 Input

不同：

\[
x
\]

可能产生不同：

\[
D(x)
\]

所以虽然每个 region：

\[
y=A_rx+c_r
\]

但：

\[
A_r,c_r
\]

随着 region改变。

这就是整个 network非线性的来源。

---

# 45. 这个公式非常重要

ReLU Network可以理解成：

\[
\boxed{
\text{input chooses an activation mask}
}
\]

然后：

\[
\boxed{
\text{activation mask chooses an effective affine map}
}
\]

注意：

> mask不是离散模型另外预测出来的变量。

它就是由：

\[
W_1x+b_1>0
\]

自然决定。

---

# 46. 为什么可以说 ReLU 是 Gate？

因为：

\[
ReLU(z)
=
z\cdot 1[z>0]
\]

除了 \(z=0\) 边界约定。

所以它相当于：

> 让 pre-activation自己决定这条 feature channel是否通过。

---

# 47. 但和 Sigmoid Gate 不一样

Sigmoid：

\[
\sigma(z)
\in(0,1)
\]

是：

> soft gate。

ReLU：

\[
1[z>0]
\]

从 derivative / region角度更像：

> hard on/off boundary，

但 active时保留：

\[
z
\]

的实际 magnitude。

所以：

\[
\boxed{
ReLU
\neq
binary\ gate
}
\]

它的 output不是只有：

\[
0/1
\]

而是：

\[
0
\]

或：

\[
z
\]

---

# 48. ReLU 的 Derivative

对：

\[
x<0
\]

\[
ReLU(x)=0
\]

所以：

\[
\boxed{
ReLU'(x)=0
}
\]

---

对：

\[
x>0
\]

\[
ReLU(x)=x
\]

所以：

\[
\boxed{
ReLU'(x)=1
}
\]

---

# 49. 在 x=0 呢？

左导数：

\[
0
\]

右导数：

\[
1
\]

不相等。

所以数学上：

\[
\boxed{
ReLU'(0)
\text{ 不存在}
}
\]

---

# 50. 那神经网络怎么 Backprop？

因为：

> 恰好落在一个不可导点，并不会让整个优化过程不可用。

自动微分框架可以在不可导点：

> 选择一个 subgradient / convention。

---

# 51. PyTorch 在 ReLU(0) 用什么 Gradient？

PyTorch的 ReLU backward等价于：

\[
\boxed{
\frac{\partial ReLU(x)}{\partial x}
=
\begin{cases}
0,&x\le0\\
1,&x>0
\end{cases}
}
\]

也就是在：

\[
x=0
\]

选择：

\[
\boxed{
0
}
\]

作为 backward gradient。

---

# 52. 为什么 0 是合理的 Subgradient？

ReLU：

\[
f(x)=\max(0,x)
\]

是 convex function。

在：

\[
x=0
\]

subdifferential：

\[
[0,1]
\]

里面任意值都可以是 convex-analysis意义的 subgradient。

PyTorch autograd文档对局部 convex、不可导函数的规则之一是：

> 使用 minimum-norm subgradient。

对：

\[
[0,1]
\]

minimum norm是：

\[
0
\]

---

# 53. “不可导”不等于“不能训练”

很多优化目标本来就可能有：

- kink；
- absolute value；
- max；
- piecewise structure。

Gradient-based optimization可以通过：

- subgradient；
- chosen convention；

处理这些位置。

所以：

\[
\boxed{
\text{differentiable everywhere}
}
\]

不是现代神经网络可训练的必要条件。

---

# 54. 但 x=0 附近会不会有问题？

通常不会因为“恰好不可导”本身成为主要问题。

更实际的问题是：

> 负区间 derivative整个都是 0。

这会带来：

> dying ReLU。

---

# 55. 什么是 Dying ReLU？

一个 ReLU neuron：

\[
h=
ReLU(w^\top x+b)
\]

如果对于当前训练数据长期都有：

\[
w^\top x+b<0
\]

那么：

\[
h=0
\]

并且：

\[
\frac{\partial h}{\partial z}=0
\]

---

# 56. 于是这条局部 Gradient Path 被切断

如果 downstream gradient：

\[
\frac{\partial L}{\partial h}
\]

存在，

但：

\[
\frac{\partial h}{\partial z}=0
\]

那么：

\[
\frac{\partial L}{\partial z}
=
\frac{\partial L}{\partial h}
\frac{\partial h}{\partial z}
=
0
\]

---

# 57. Weight Gradient 也变成 0

\[
z=w^\top x+b
\]

所以：

\[
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial z}x
\]

如果：

\[
\frac{\partial L}{\partial z}=0
\]

则：

\[
\boxed{
\frac{\partial L}{\partial w}=0
}
\]

bias同样：

\[
\boxed{
\frac{\partial L}{\partial b}=0
}
\]

对这个 example / path。

---

# 58. 如果对所有训练样本都 Inactive 呢？

那么这个 neuron可能长期收不到来自该 ReLU路径的梯度，

于是表现得像：

> “死掉了”。

这就是：

\[
\boxed{
\text{dying ReLU}
}
\]

---

# 59. Dying 是否一定永久不可恢复？

不要说得过度绝对。

在复杂网络里：

- upstream representation可能变化；
- optimizer其他机制可能改变参数环境；
- normalization会改变输入分布；
- weight decay等也可能影响参数。

所以更准确：

> 如果一个 ReLU unit对相关输入长期处于负 pre-activation区，它的本地 ReLU gradient为0，会严重阻碍通过该路径重新激活。

---

# 60. 为什么大 Learning Rate 可能增加 Dying ReLU 风险？

一次很大的 update可能把：

\[
w,b
\]

推到一个区域，

使大量 training inputs：

\[
w^\top x+b<0
\]

之后该 neuron就很难通过自身 task gradient回来。

所以：

- initialization；
- learning rate；
- normalization；

都会影响 ReLU activation health。

---

# 61. Leaky ReLU 是怎么处理这个问题的？

定义：

\[
\boxed{
LeakyReLU(x)
=
\begin{cases}
\alpha x,&x<0\\
x,&x\ge0
\end{cases}
}
\]

其中：

\[
\alpha>0
\]

通常比较小。

---

# 62. 它的负区间 Gradient 不再是 0

\[
\boxed{
LeakyReLU'(x)
=
\alpha
}
\]

当：

\[
x<0
\]

所以即使 neuron在负区间，

仍有：

> 一条小 gradient path。

---

# 63. 但 ACT Canonical FFN 用的不是 Leaky ReLU

ACT / DETR-style Transformer：

```python
activation="relu"
```

默认 activation helper返回：

```python
F.relu
```

所以 ACT canonical FFN是：

\[
\boxed{
ReLU
}
\]

不是：

- LeakyReLU；
- GELU；
- SiLU。

---

# 64. ReLU 和 Sigmoid 的 Gradient 差别

Sigmoid：

\[
\sigma(x)
=
\frac1{1+e^{-x}}
\]

derivative：

\[
\boxed{
\sigma'(x)
=
\sigma(x)(1-\sigma(x))
}
\]

最大值只有：

\[
0.25
\]

---

# 65. Sigmoid 在大正/负输入都会 Saturate

当：

\[
x\gg0
\]

\[
\sigma(x)\approx1
\]

\[
\sigma'(x)\approx0
\]

当：

\[
x\ll0
\]

\[
\sigma(x)\approx0
\]

\[
\sigma'(x)\approx0
\]

所以两侧都可能：

> gradient非常小。

---

# 66. ReLU 正区间不 Saturate

当：

\[
x>0
\]

无论：

\[
x=0.1
\]

还是：

\[
x=100
\]

都有：

\[
\boxed{
ReLU'(x)=1
}
\]

所以正区间不存在 Sigmoid那种：

> 输入越大 derivative越接近0

的 saturation。

---

# 67. 这为什么有助于深网络优化？

如果一条 active path上的 ReLU derivative：

\[
=1
\]

它不会额外乘一个：

\[
<1
\]

的 activation derivative。

相较 sigmoid/tanh饱和区：

> gradient可以更直接传播。

---

# 68. 但“ReLU 解决了 Vanishing Gradient”仍然过度简化

因为：

1. 负区间：
   \[
   derivative=0
   \]
2. weight matrices的 Jacobians仍会相乘；
3. normalization、depth、initialization都影响 gradient；
4. active pattern可能改变。

所以更准确：

\[
\boxed{
\text{ReLU avoids positive-side saturation and often improves gradient propagation,}
}
\]

而不是：

\[
\boxed{
\text{ReLU mathematically eliminates vanishing gradients.}
}
\]

---

# 69. Tanh 又怎样？

\[
tanh(x)
\in(-1,1)
\]

derivative：

\[
\boxed{
1-tanh^2(x)
}
\]

当：

\[
|x|
\]

很大，

\[
|tanh(x)|\approx1
\]

所以：

\[
tanh'(x)\approx0
\]

也会 saturation。

---

# 70. ReLU 的计算非常便宜

核心：

\[
\max(0,x)
\]

相比：

- sigmoid的 exponential；
- tanh的 exponential-like计算；

通常非常简单。

这也是 ReLU长期广泛使用的重要工程优势之一。

---

# 71. 但现代硬件下不能只用“便宜”解释一切

实际模型速度还取决于：

- fused kernels；
- memory bandwidth；
- matrix multiplication；
- precision；
- compiler。

而 Transformer主要 FLOPs常来自：

> Linear/GEMM。

所以 ReLU便宜是一点，

不是整个模型快慢的唯一决定因素。

---

# 72. ReLU 为什么会产生 Sparse Activation？

因为所有：

\[
z\le0
\]

都变成：

\[
0
\]

所以一个 hidden vector可能：

\[
[2.1,-0.4,0.8,-3.0,1.2]
\]

经过 ReLU：

\[
[2.1,0,0.8,0,1.2]
\]

出现精确的 zeros。

---

# 73. Glorot 等 2011 特别强调这一点

**Deep Sparse Rectifier Neural Networks** 指出 rectifying neurons：

> 会产生带有真实 zeros 的 sparse representations。

这是 ReLU早期吸引人的性质之一。

---

# 74. Sparse Activation 不等于 Sparse Weight

这是必须区分的。

ReLU可能产生：

\[
h_j=0
\]

表示：

> 当前 input下这个 activation为0。

但 weight matrix：

\[
W
\]

仍然可能是完全 dense 的。

所以：

\[
\boxed{
\text{activation sparsity}
\neq
\text{parameter sparsity}
}
\]

---

# 75. 也不等于模型自动加速

即使 ReLU output有很多0，

后续：

```python
nn.Linear
```

通常仍然执行：

> dense matrix multiplication。

除非 hardware / kernel专门利用 activation sparsity。

所以：

\[
\boxed{
\text{many zeros}
\not\Rightarrow
\text{automatic FLOP reduction}
}
\]

---

# 76. Sparse Activation 有什么 Representation Intuition？

某个 input只激活：

> 一部分 hidden units。

另一个 input可能激活：

> 另一部分。

所以 network可以用：

\[
\boxed{
\text{不同 activation subsets}
}
\]

表示不同 input regimes。

这和前面：

> activation pattern决定 piecewise-affine region

是同一件事的两个视角。

---

# 77. “Sparse”与“Conditional Computation”有点像吗？

作为直觉：

> ReLU确实让某些 channels在某些 input上不参与输出。

但标准 dense implementation仍会：

> 先算 pre-activation，再做 ReLU。

所以它不是 MoE那种：

> 真正只执行被选 expert

的结构性 conditional compute。

不要混淆。

---

# 78. ReLU 为什么没有 Upper Bound？

\[
x>0
\]

时：

\[
ReLU(x)=x
\]

所以：

\[
x\rightarrow+\infty
\]

输出也：

\[
\rightarrow+\infty
\]

它不是 bounded activation。

---

# 79. 这既是优点也可能是风险

优点：

> 不会像 sigmoid/tanh在正方向饱和。

风险：

> activation magnitude可能很大。

所以模型还依赖：

- initialization；
- normalization；
- optimizer；
- residual design；

控制数值尺度。

---

# 80. ReLU 不是“归一化”函数

它不会：

- 控制 mean；
- 控制 variance；
- 把 norm固定；
- 把值压到 [0,1]。

所以：

\[
\boxed{
ReLU
\neq
LayerNorm
}
\]

二者功能完全不同。

---

# 81. ReLU 也不是 Probability Function

输出：

\[
0,2,100,\ldots
\]

都可能。

不要求：

\[
\sum_i y_i=1
\]

所以：

\[
\boxed{
ReLU
\neq
Softmax
}
\]

---

# 82. ReLU 也不是 Binary Threshold

Binary threshold：

\[
1[z>0]
\]

输出：

\[
0/1
\]

ReLU：

\[
z1[z>0]
\]

输出：

> 正值的 magnitude仍保留。

所以它同时编码：

1. 是否 active；
2. active时有多强。

---

# 83. ReLU 为什么保留 Relative Intensity？

Nair & Hinton 2010 的 early rectified-unit工作强调：

> 与 binary units相比，rectified linear units可以在多层 feature detectors中保留 relative intensity information。

直觉上：

binary：

\[
z>0\Rightarrow1
\]

那么：

\[
1
\]

和：

\[
100
\]

都可能只变成：

\[
1
\]

而 ReLU：

\[
1\rightarrow1
\]

\[
100\rightarrow100
\]

仍保留幅度差。

---

# 84. 但 Nair & Hinton 2010 的具体模型背景要注意

那篇论文讨论的是：

> Restricted Boltzmann Machines 中的 noisy rectified linear units。

它是 ReLU历史上的重要工作，

但不要把里面每个 stochastic-unit细节：

> 直接等同现代 deterministic `nn.ReLU` feedforward layer。

现代 canonical ReLU定义仍：

\[
\max(0,x)
\]

---

# 85. Glorot et al. 2011 更直接连接 Deep Feedforward Networks

该论文研究：

> rectifier nonlinearities用于深层监督网络。

其 abstract明确指出：

- hard non-linearity；
- zero点不可导；
- true zero sparse representations；
- deep rectifier networks可在纯监督任务中取得强表现。

这是理解现代 ReLU deep networks的重要早期来源。

---

# 86. ReLU 的 Convexity

函数：

\[
f(x)=\max(0,x)
\]

是 convex。

因为它可以看成两个 affine functions：

\[
0
\]

和：

\[
x
\]

的 pointwise maximum：

\[
\boxed{
f(x)=\max(f_1(x),f_2(x))
}
\]

pointwise maximum of convex functions仍 convex。

---

# 87. 但 ReLU Network 整体通常不 Convex

虽然单个 ReLU对其 scalar input是 convex，

但经过：

- negative output weights；
- multiple layers；
- composition；

整个 neural network：

> 一般不是 input或parameters上的简单 convex function。

所以：

\[
\boxed{
\text{ReLU convex}
\not\Rightarrow
\text{training objective convex}
}
\]

---

# 88. 训练 ReLU Network 仍然是 Nonconvex Optimization

例如两层网络：

\[
y=W_2ReLU(W_1x+b_1)+b_2
\]

对：

\[
W_1,W_2
\]

联合优化：

> 一般高度 nonconvex。

所以 ReLU本身简单：

> 不意味着深网优化变成线性回归。

---

# 89. ReLU 是否可逆？

不可以。

所有：

\[
x\le0
\]

都映射到：

\[
0
\]

例如：

\[
-1,-5,-100
\]

全部：

\[
\rightarrow0
\]

所以：

\[
\boxed{
ReLU
\text{ loses negative-side magnitude information}
}
\]

---

# 90. 那网络为什么敢丢负值？

因为：

> pre-activation的正负本身是 learned coordinate system中的状态。

Weight和bias可以学习：

> 哪些 feature combinations应该进入 active side。

而且一个 layer通常有很多 units，

不同 units可以编码互补信息。

---

# 91. 如果真的需要负侧信息怎么办？

网络可以使用：

- 另一个 neuron；
- different weights；
- LeakyReLU；
- GELU；
- other activations。

例如两个 units：

\[
ReLU(x)
\]

和：

\[
ReLU(-x)
\]

一起就能保留正负两侧 magnitude。

---

# 92. 一个漂亮的恒等式

\[
\boxed{
x
=
ReLU(x)-ReLU(-x)
}
\]

验证：

如果：

\[
x>0
\]

右边：

\[
x-0=x
\]

如果：

\[
x<0
\]

右边：

\[
0-(-x)=x
\]

所以 ReLU虽然单独截断负值，

多个 units组合可以重新表达 signed quantity。

---

# 93. Absolute Value 也可以由 ReLU 表示

\[
\boxed{
|x|
=
ReLU(x)+ReLU(-x)
}
\]

所以两个非常简单的 ReLU units：

> 就能构造一个新的 nonlinear function。

---

# 94. 这再次展示“简单 Primitive + Combination”

单个 ReLU：

> 非常简单。

组合起来：

\[
\boxed{
\text{can build richer piecewise-linear functions}
}
\]

神经网络复杂性主要来自：

> 大量这样的简单模块复合。

---

# 95. ReLU 前的 Bias 为什么很重要？

Neuron：

\[
ReLU(w^\top x+b)
\]

边界：

\[
w^\top x+b=0
\]

bias：

\[
b
\]

允许 hyperplane：

> 在空间中平移。

---

# 96. 如果没有 Bias

边界：

\[
w^\top x=0
\]

必须经过：

> origin。

所有 ReLU hinges都被限制通过原点。

这会限制：

> region boundary的位置灵活性。

---

# 97. 1D 里更明显

无 bias：

\[
ReLU(wx)
\]

kink只能在：

\[
x=0
\]

---

有 bias：

\[
ReLU(wx+b)
\]

kink：

\[
wx+b=0
\]

即：

\[
\boxed{
x=-b/w
}
\]

可以被学习到任意位置。

---

# 98. 所以 Bias 决定 Breakpoint 的 Translation

Weight：

\[
w
\]

控制：

- boundary orientation；
- scaling。

Bias：

\[
b
\]

控制：

> boundary offset。

高维也是同样。

---

# 99. 为什么 ReLU 网络被称为 Piecewise-Affine 更严格？

因为 Linear Layer通常有：

\[
+b
\]

所以每个 region内是：

\[
Ax+c
\]

即：

> affine。

如果所有 bias都为0，

则更接近：

> piecewise linear / positively homogeneous。

工程和文献中常宽泛说：

> piecewise linear network。

严格数学上有 bias时：

\[
\boxed{
\text{piecewise affine}
}
\]

更准确。

---

# 100. 为什么我们仍然常说 Piecewise Linear ReLU Network？

这是领域中的常见简写。

“linear region”通常也常把：

> affine piece

包括进去。

在严谨文章里最好注明：

> with biases, each region is affine。

---

# 101. ReLU Layer 对 Vector 的 Jacobian

设：

\[
h=ReLU(z)
\]

逐元素。

定义：

\[
D(z)
=
diag(
1[z_1>0],
\ldots,
1[z_m>0]
)
\]

忽略 zero边界。

那么：

\[
\boxed{
\frac{\partial h}{\partial z}
=
D(z)
}
\]

---

# 102. Linear + ReLU 的 Jacobian

\[
z=Wx+b
\]

\[
h=ReLU(z)
\]

所以：

\[
\boxed{
\frac{\partial h}{\partial x}
=
D(z)W
}
\]

这说明：

> input决定哪些 rows / feature paths active。

---

# 103. 两层 MLP 的 Jacobian

\[
y=W_2ReLU(W_1x+b_1)+b_2
\]

在一个固定 activation region内：

\[
\boxed{
J
=
W_2DW_1
}
\]

所以不同：

\[
D
\]

意味着不同：

\[
J
\]

也就是：

> 不同 input region具有不同 local linear transformation。

---

# 104. 这是理解 Nonlinearity 的一个非常严谨方式

Affine network：

\[
J=W
\]

到处一样。

ReLU network：

\[
J(x)=W_2D(x)W_1
\]

会随：

\[
x
\]

改变。

因此：

\[
\boxed{
\text{input-dependent Jacobian}
}
\]

就是它非线性结构的一个体现。

---

# 105. 为什么单层 ReLU 已经是 Nonlinear？

因为：

\[
D(x)
\]

会随着 input跨 boundary改变。

不需要很多层才“开始非线性”。

一层：

\[
ReLU(Wx+b)
\]

已经是 nonlinear map。

---

# 106. Deep Network 增加的是什么？

深度让：

> 后续 boundaries建立在前面已经非线性变换后的 coordinates上。

所以可以构造：

> 更复杂的 hierarchical piecewise-affine maps。

---

# 107. ReLU 会让 Output 全部非负吗？

**ReLU layer本身：会。**

\[
h=ReLU(z)
\]

所以：

\[
h_i\ge0
\]

---

# 108. 但整个 FFN Output 不一定非负

Transformer FFN：

\[
h=
ReLU(xW_1+b_1)
\]

然后：

\[
y=hW_2+b_2
\]

第二个 Linear：

> 可以有正负 weights和bias。

所以：

\[
\boxed{
y
\text{ 可以正也可以负}
}
\]

---

# 109. 这对 ACT 非常重要

ACT FFN：

\[
512
\rightarrow
3200
\rightarrow
ReLU
\rightarrow
512
\]

虽然：

\[
3200\text{-D hidden}
\]

经过 ReLU后非负，

但第二 Linear：

\[
3200\rightarrow512
\]

输出：

> 完全可以有负值。

---

# 110. 所以 ACT Action 不会因为 FFN 用 ReLU 而只能是正数

最终：

\[
action\_head:
512\rightarrow14
\]

也是 Linear。

所以最终 action coordinates：

> 可以正也可以负。

这是一个很常见的误解。

---

# 111. ReLU 在 ACT FFN 的准确位置

官方 Transformer：

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

默认：

```python
activation="relu"
```

因此：

```text
src [512]
↓
Linear1
↓
3200-D pre-activation
↓
ReLU
↓
3200-D nonnegative activation
↓
Dropout
↓
Linear2
↓
512-D signed residual update
```

---

# 112. Encoder 每个 Token 都独立做这个 ReLU

Policy Encoder：

\[
[1202,B,512]
\]

Linear1：

\[
[1202,B,3200]
\]

ReLU：

\[
\boxed{
[1202,B,3200]
}
\]

shape不变。

每个：

- latent token；
- joint token；
- visual token；

各自拥有自己的 3200-D activation pattern。

---

# 113. 同一个 FFN 参数，不同 Token 会有不同 Active Units

因为输入：

\[
x_i
\]

不同。

虽然：

\[
W_1,b_1
\]

共享，

但：

\[
z_i=W_1x_i+b_1
\]

不同。

所以：

\[
D(x_i)
\]

也不同。

---

# 114. 这就是 Position-Wise FFN 的 Nonlinear Power

所有 positions共享：

> 同一套 feature detectors。

但每个 token：

> 激活不同 subset。

所以：

\[
\boxed{
\text{shared rules}
+
\text{input-dependent activation pattern}
}
\]

产生不同 local computation。

---

# 115. Decoder Action Slots 同理

\[
[k,B,512]
\]

经过：

\[
Linear1
\]

得到：

\[
[k,B,3200]
\]

每一个 future action slot：

> 自己决定哪一部分 3200 features active。

所以不同 future positions：

> 即使使用相同 FFN parameters，

也会执行不同 effective affine map。

---

# 116. 这就是为什么“FFN 对所有 Token 一样”不等于“所有 Token 输出一样”

函数参数：

> 相同。

输入：

> 不同。

activation pattern：

> 不同。

输出自然不同。

---

# 117. ReLU 是否跨 Token？

不会。

它只是：

\[
elementwise
\]

所以 token communication：

> 仍然由 Attention负责。

ReLU作用于：

> 每个 token内部的 feature responses。

---

# 118. Attention 已经有 Softmax 非线性，为什么还需要 ReLU？

这是前面 FFN文章里的关键问题。

Softmax非线性主要参与：

> token-to-token routing。

ReLU则在：

> token内部 wide feature transformation

中提供 input-dependent gating。

二者作用位置不同。

---

# 119. 所以不能说 ReLU 是 Transformer 唯一 Nonlinearity

Transformer还有：

- Softmax；
- LayerNorm；
- 其他 operations。

更准确：

\[
\boxed{
\text{ReLU is the explicit activation nonlinearity inside the original FFN.}
}
\]

---

# 120. 原始 Transformer 明确用了 ReLU

Section 3.3：

\[
\boxed{
FFN(x)
=
\max(0,xW_1+b_1)W_2+b_2
}
\]

Base：

\[
d_{\text{model}}=512
\]

\[
d_{ff}=2048
\]

所以：

\[
512
\rightarrow
2048
\rightarrow
ReLU
\rightarrow
512
\]

---

# 121. ACT 改了 Width，不改这个基本 Activation 结构

ACT canonical：

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
ReLU
\rightarrow
512
}
\]

---

# 122. ACT 代码其实支持别的 Activation

DETR-derived helper：

```python
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu

    if activation == "gelu":
        return F.gelu

    if activation == "glu":
        return F.glu
```

所以 framework代码可选：

- ReLU；
- GELU；
- GLU。

但 constructor默认：

```python
activation="relu"
```

因此 canonical ACT：

\[
\boxed{
ReLU
}
\]

---

# 123. ReLU vs GELU：最基础区别

ReLU：

\[
\boxed{
\max(0,x)
}
\]

负数：

> 硬截成0。

---

GELU：

常写：

\[
\boxed{
GELU(x)=x\Phi(x)
}
\]

其中：

\[
\Phi
\]

是标准正态 CDF。

它是：

> 更平滑的 input-dependent gating。

---

# 124. GELU 的负区间不一定完全为 0

例如小负值：

> 可能保留一部分负 output。

所以它没有 ReLU那种：

> 所有负 pre-activation精确变0

的 hard sparsity。

---

# 125. 为什么现代 Transformer 常用 GELU / SiLU / SwiGLU？

后续模型发现：

- smoother activations；
- gated FFN；

在很多大模型设置中表现很好。

但这是：

> Transformer架构后续演化。

不能反过来说：

> 2017 Transformer用了 GELU。

---

# 126. ACT 也不能因为“现代模型都用 GELU”就写成 GELU

Official ACT / DETR-derived default：

\[
\boxed{
ReLU
}
\]

所以学习 ACT canonical implementation时：

> 必须以 ReLU为准。

---

# 127. ReLU 和 GELU 哪个“理论上更好”？

没有一个普遍定理说明：

> 所有任务都必须 GELU胜过 ReLU。

实际效果取决于：

- architecture；
- scale；
- optimization；
- data；
- task。

所以激活函数是：

> model design choice。

---

# 128. ReLU 的一个非常漂亮性质：连续

虽然：

\[
x=0
\]

不可导，

但函数本身连续。

因为：

\[
\lim_{x\to0^-}ReLU(x)=0
\]

\[
\lim_{x\to0^+}ReLU(x)=0
\]

且：

\[
ReLU(0)=0
\]

所以：

\[
\boxed{
ReLU\text{ continuous but not differentiable at }0
}
\]

---

# 129. Continuous 与 Differentiable 不一样

连续：

> 函数没有跳跃。

可导：

> 局部 slope极限唯一存在。

ReLU在0：

- 没有跳；
- 但左 slope 0、右 slope 1。

所以：

> 连续，但不可导。

---

# 130. 为什么这比 Step Function 更友好？

Heaviside step：

\[
H(x)=
\begin{cases}
0,&x<0\\
1,&x\ge0
\end{cases}
\]

几乎处处 derivative：

\[
0
\]

所以 gradient-based learning很困难。

ReLU active side：

\[
derivative=1
\]

因此保留了非常有用的 gradient path。

---

# 131. ReLU 和 Binary Neuron 的关键差异

Binary threshold只表达：

> on/off。

ReLU表达：

> on/off + positive magnitude。

所以它同时具有：

- gating；
- intensity coding。

这是其简单却有效的重要原因之一。

---

# 132. ReLU 的 Expected Activation 会改变 Distribution

如果 pre-activation大致对称于0，

ReLU会：

> 把负半边压成0。

所以输出：

- 有一个0 mass；
- 正侧保留。

因此 distribution通常：

> strongly non-Gaussian。

这再次说明：

> 深层 hidden activations完全不需要服从正态分布。

---

# 133. LayerNorm 后接 ReLU 会发生什么？

如果输入 ReLU前经过某种 normalization，

仍然可能有：

- 正值；
- 负值。

ReLU再把负值设0。

所以 normalization与activation：

> 完全是两个步骤。

---

# 134. ACT Post-LN FFN 的顺序

Encoder：

```text
上一 sublayer
↓
Residual + LayerNorm
↓
Linear1
↓
ReLU
↓
Dropout
↓
Linear2
↓
Residual + LayerNorm
```

所以 ReLU输入来自：

> normalized contextual representation经过Linear1后的 pre-activation。

---

# 135. Pre-LN 时呢？

Pre-LN：

```text
LayerNorm
↓
Linear1
↓
ReLU
↓
Dropout
↓
Linear2
↓
Residual Add
```

ReLU本身公式不变。

改变的是：

> normalization / residual placement。

---

# 136. ReLU 是否需要自己的 Parameters？

没有。

标准 ReLU：

\[
max(0,x)
\]

没有 trainable weights。

所以：

```python
nn.ReLU()
```

参数量：

\[
\boxed{
0
}
\]

---

# 137. 那它为什么这么重要？

因为：

> 参数数量不等于 architecture importance。

ReLU没有一个 trainable parameter，

却改变了：

\[
\boxed{
\text{整个 function class}
}
\]

没有它：

> 多层 affine network仍是一层 affine。

有它：

> network变成 input-dependent piecewise-affine system。

---

# 138. 这是一个很漂亮的深度学习事实

复杂能力不一定来自：

> 一个模块自己有很多参数。

有时一个零参数 operation：

\[
\boxed{
max(0,x)
}
\]

就能让前后数百万 parameters：

> 无法再被整体合并成一层。

---

# 139. ReLU 为什么不能放在任何地方都一样？

因为 function composition顺序重要。

例如：

\[
ReLU(Wx)
\]

和：

\[
WReLU(x)
\]

一般：

\[
\boxed{
ReLU(Wx)\neq WReLU(x)
}
\]

---

# 140. 一个例子

设：

\[
x=
\begin{bmatrix}
1\\
-1
\end{bmatrix}
\]

\[
W=
\begin{bmatrix}
1&1
\end{bmatrix}
\]

先 Linear：

\[
Wx=0
\]

再 ReLU：

\[
0
\]

---

先 ReLU：

\[
ReLU(x)
=
\begin{bmatrix}
1\\
0
\end{bmatrix}
\]

再 Linear：

\[
WReLU(x)=1
\]

所以：

\[
\boxed{
0\neq1
}
\]

顺序不能交换。

---

# 141. 为什么不能把 ReLU 推进 Matrix 里？

因为 ReLU不是 linear operator。

对于一般：

\[
W
\]

没有：

\[
ReLU(Wx)=WReLU(x)
\]

因此不能像两层 Linear那样：

> 通过矩阵乘法合并掉。

---

# 142. 这就是为什么它阻止 Layer Collapse

\[
W_2ReLU(W_1x+b_1)+b_2
\]

ReLU卡在中间，

使：

\[
W_2W_1
\]

不能直接替代整个函数。

这正是 FFN深度有意义的原因。

---

# 143. ReLU 是否保持输入顺序？

对于 scalar：

如果：

\[
x_1<x_2
\]

则：

\[
ReLU(x_1)\le ReLU(x_2)
\]

所以它是：

> monotonic nondecreasing。

但很多负值：

\[
-100<-1
\]

都映射：

\[
0=0
\]

所以并非严格 monotonic。

---

# 144. 为什么 Monotonic 仍能构造复杂 Non-Monotonic Network？

因为 ReLU后还有：

> negative output weights。

例如前面：

\[
ReLU(x)-2ReLU(x-1)+ReLU(x-2)
\]

整体就先升后降。

所以单个 activation monotonic：

> 不限制网络整体必须 monotonic。

---

# 145. ReLU 的 Lipschitz Constant

scalar ReLU满足：

\[
|ReLU(x)-ReLU(y)|
\le
|x-y|
\]

所以它是：

\[
\boxed{
1\text{-Lipschitz}
}
\]

因为 slope只有：

\[
0
\]

或：

\[
1
\]

---

# 146. 但整个 ReLU Network 不一定 1-Lipschitz

Linear weights可能有：

> 大 operator norm。

例如：

\[
100x
\]

后接 ReLU，

整体变化率可以远大于1。

所以：

\[
\boxed{
\text{ReLU itself 1-Lipschitz}
\not\Rightarrow
\text{network 1-Lipschitz}
}
\]

---

# 147. 为什么这类数学性质值得知道？

它帮助你区分：

> activation本身的性质

和：

> 完整网络的性质。

很多错误解释都来自：

> 把局部模块性质直接扩大成整个模型保证。

---

# 148. ReLU Output Mean 会是 0 吗？

不保证。

事实上 ReLU output：

\[
\ge0
\]

所以只要有正 activation，

mean通常：

\[
>0
\]

这也是为什么：

> normalization placement和initialization会影响 activation statistics。

---

# 149. ReLU 是否 Zero-Centered？

不是。

输出只能：

\[
\ge0
\]

所以 hidden activations不以0对称。

Tanh则：

\[
(-1,1)
\]

可以产生正负 outputs。

这是不同 activation的统计差异。

---

# 150. 但 ReLU 网络照样可以输出 Negative Values

再次强调：

后续 Linear：

\[
W_2h+b_2
\]

可产生：

\[
<0
\]

所以：

> hidden activation nonnegative

不等于：

> model output nonnegative。

---

# 151. ReLU 对概率模型有特殊概率解释吗？

标准 ReLU：

> 没有必要的概率解释。

它只是 deterministic activation。

不要因为它出现在：

- VAE；
- Transformer；

就给它附加：

> probability distribution语义。

---

# 152. ReLU 与 CVAE 的 Sampling 没关系

CVAE随机性来自：

\[
z=\mu+\sigma\epsilon
\]

ReLU则：

\[
max(0,x)
\]

完全 deterministic。

所以训练时：

- Dropout可能随机；
- z sampling可能随机；
- ReLU不随机。

---

# 153. ReLU 是否训练时和推理时不一样？

标准 ReLU：

> 完全一样。

不像 Dropout：

- train随机；
- eval identity。

ReLU在两种 mode下都：

\[
\boxed{
max(0,x)
}
\]

---

# 154. `model.eval()` 不会关闭 ReLU

它仍正常执行。

所以 ACT inference：

- Dropout关闭；
- ReLU继续工作；
- LayerNorm继续工作；
- Attention继续工作。

---

# 155. ReLU 的 Inplace 参数是什么？

PyTorch：

```python
nn.ReLU(
    inplace=False
)
```

如果：

```python
inplace=True
```

可以尝试直接修改 input tensor storage。

这可能节省一些 memory，

但需要小心 autograd / tensor reuse。

---

# 156. Canonical 理论不依赖 Inplace

无论：

```python
inplace=False
```

还是：

```python
True
```

数学函数仍：

\[
\max(0,x)
\]

所以理论文章只需把 inplace当：

> implementation detail。

---

# 157. ReLU 是否对整个 Vector 求 max？

不是。

\[
ReLU(x)
\]

对 vector表示：

\[
[
max(0,x_1),
\ldots,
max(0,x_n)
]
\]

不是：

\[
max(0,x_1,\ldots,x_n)
\]

不要和 max pooling混淆。

---

# 158. ReLU 和 Max Pooling 都有 max，但功能不同

ReLU：

\[
max(0,x_i)
\]

把一个 scalar与：

\[
0
\]

比较。

Max Pooling：

> 在一组 spatial / temporal elements中选最大值。

所以：

\[
\boxed{
ReLU
\neq
MaxPool
}
\]

---

# 159. 为什么 ReLU 的 Boundary 是 Learned？

ReLU本身 kink固定在：

\[
z=0
\]

但：

\[
z=w^\top x+b
\]

中的：

\[
w,b
\]

是 learned。

所以在原始 input / hidden space中，

boundary：

\[
w^\top x+b=0
\]

位置与方向：

> 都由训练学习。

---

# 160. 这是一个很重要的视角

虽然 activation函数自身永远：

\[
\max(0,z)
\]

不变，

但前面的 Linear Layer在学习：

> “应该让哪些 inputs落到正区间，哪些落到负区间。”

所以真正的 flexibility来自：

\[
\boxed{
\text{learned affine boundary}
+
\text{fixed nonlinear gate}
}
\]

---

# 161. 训练就是在移动这些 Hyperplanes

Weight：

\[
w
\]

变化：

> 旋转 /改变 boundary orientation。

Bias：

\[
b
\]

变化：

> 平移 boundary。

所以 optimizer逐步学习：

> 如何划分 representation space。

---

# 162. 多个 Layers 则不断重新划分表示空间

第一层：

> 切 input space。

第二层：

> 对第一层产生的 piecewise representation继续切。

所以深度允许：

> 逐层重塑 decision geometry。

---

# 163. 这和 Decision Tree 有点像吗？

可以做有限类比：

Decision Tree：

> 明确使用离散条件分支。

ReLU network：

> 用连续 learned hyperplanes隐式决定 activation regimes。

但它并不是一棵显式 tree，

regions共享参数结构，

并通过 gradient端到端训练。

所以只是：

> “不同区域使用不同规则”

这一点相似。

---

# 164. 为什么 ReLU Network 仍是 Continuous？

每个 ReLU连续，

Affine连续。

连续函数的 composition仍连续。

所以标准 ReLU MLP：

\[
\boxed{
\text{continuous piecewise-affine}
}
\]

除非其他模块引入 discontinuity。

---

# 165. 所以它不能精确表示 Jump Discontinuity 吗？

有限标准 ReLU network本身是连续的，

所以不能在实数域精确形成真正跳跃：

```text
0 → suddenly 1
```

但可以用非常陡的 piecewise-linear transition：

> 任意逼近某些 discontinuous behavior，在合适误差/区域意义下。

这属于更深入 approximation theory。

---

# 166. ReLU 与 Universal Approximation

足够宽的 non-polynomial activation network具有强大的函数逼近能力。

ReLU属于现代 universal approximation理论中常见 activation。

但这不是：

> “一个 ReLU neuron就能表示任意函数。”

需要：

- 足够 width/depth；
- suitable parameters；
- compact-domain/error条件。

所以不要把 universal approximation当成：

> training一定成功。

---

# 167. Expressivity 和 Optimization 仍然不同

网络能表示某个函数：

\[
\neq
\]

optimizer一定找到它。

这和我们 Residual Connection文章里的：

> representability vs trainability

是同一个重要区别。

---

# 168. 为什么 ReLU 容易结合 Residual Network？

Residual branch：

\[
F(x)
\]

可以由：

> Linear/Conv + ReLU

等组成。

ReLU负责 nonlinear feature gating，

shortcut：

\[
x
\]

负责 identity path。

二者是现代深网非常经典的组合。

---

# 169. ACT 的 ResNet18 Backbone 也大量使用 ReLU

ACT图像 backbone是：

> ResNet18。

经典 ResNet basic blocks中也广泛使用 ReLU。

所以 ACT里 ReLU并不只出现在 Transformer FFN。

---

# 170. 但 Backbone ReLU 和 Transformer FFN ReLU 参数共享吗？

当然不共享。

ReLU自己没有参数，

但前后的：

- convolution weights；
- Linear weights；

完全不同。

“都使用 ReLU”只是：

> activation function相同。

---

# 171. 为什么 ResNet18 和 Transformer 都能用同一个 ReLU？

因为 ReLU只看：

> scalar activation value。

它不关心这个数来自：

- image convolution；
- Transformer FFN；
- MLP；
- robot state。

所以它是：

> modality-agnostic primitive。

---

# 172. ReLU 会知道这个 Feature 是 Joint 还是 Image 吗？

不会。

和 Linear Layer一样：

> 它只执行数学函数。

语义来自：

> surrounding model/data。

---

# 173. 一个 ACT FFN Neuron 的具体公式

某个 token representation：

\[
x\in\mathbb R^{512}
\]

第一 Linear的第 \(j\) 个 hidden unit：

\[
z_j
=
w_j^\top x+b_j
\]

其中：

\[
w_j\in\mathbb R^{512}
\]

然后：

\[
\boxed{
h_j=
max(0,z_j)
}
\]

---

# 174. ACT 有 3200 个这样的 Hidden Units

所以：

\[
j=1,\ldots,3200
\]

每一个定义一个：

\[
w_j^\top x+b_j=0
\]

boundary。

对当前 token，

会产生一个：

\[
3200
\]

维 activation pattern：

\[
h
=
[
h_1,\ldots,h_{3200}
]
\]

---

# 175. 一个 Token 实际选择了一个 Active Set

定义：

\[
\mathcal A(x)
=
\{
j:
w_j^\top x+b_j>0
\}
\]

那么只有：

\[
j\in\mathcal A(x)
\]

的 units：

> 输出非零。

所以当前 token的 FFN computation可以理解成：

> 激活了一组 learned feature detectors。

---

# 176. 第二 Linear 把这些 Active Features 写回512-D

\[
y
=
W_2h+b_2
\]

也就是：

\[
y
=
\sum_{j=1}^{3200}
h_jv_j+b_2
\]

其中：

\[
v_j
\]

可以理解为：

> hidden unit \(j\) 对 residual-space output的 learned contribution direction。

---

# 177. 所以一个 ReLU Hidden Unit 可以怎样理解？

第一层 row：

\[
w_j
\]

决定：

> 在什么 input pattern下被激活，以及激活强度。

第二层对应 column：

\[
v_j
\]

决定：

> 激活后往 output representation写什么方向。

这是一个很有用的 MLP intuition。

---

# 178. 但不要把每个 Hidden Unit强行命名成概念

例如：

> “unit 1734就是抓杯子 feature”

没有实验不能这样说。

真实 representations可能：

- distributed；
- polysemantic；
- redundant。

这个 detector/write-direction只是数学结构直觉。

---

# 179. ReLU 在 ACT Encoder 和 Decoder作用不同吗？

公式完全相同：

\[
max(0,x)
\]

但输入 representation不同。

Encoder：

> observation-memory token。

Decoder：

> action-slot hidden state。

所以 learned：

\[
W_1,W_2
\]

不同 layer/module的参数不同，

从而 ReLU gating boundaries也不同。

---

# 180. ReLU 参数虽然为0，但 Boundary 是 Learned 的

这是一个容易误会的点。

ReLU自身：

> 无 trainable parameters。

但：

\[
ReLU(Wx+b)
\]

的 boundary由：

\[
W,b
\]

决定。

所以 network仍然学习：

> ReLU何时打开。

只是学习发生在：

> activation前的 Linear Layer。

---

# 181. 为什么不用直接学习 Threshold？

Bias：

\[
b
\]

已经控制 threshold。

1D：

\[
ReLU(wx+b)
\]

threshold：

\[
x=-b/w
\]

因此：

> 不需要 ReLU本身再有一个 threshold parameter。

---

# 182. PReLU 又是什么？

Parametric ReLU：

\[
f(x)
=
\begin{cases}
ax,&x<0\\
x,&x\ge0
\end{cases}
\]

其中：

\[
a
\]

可以是 learned parameter。

这让 negative slope：

> 由训练决定。

---

# 183. ACT Canonical 不是 PReLU

所以不要看到 “ReLU family” 就混成一个。

ACT用：

\[
\boxed{
F.relu
}
\]

经典 hard-zero negative branch。

---

# 184. ReLU 会不会改变 Feature Dimension？

不会。

\[
[3200]
\rightarrow
[3200]
\]

dimension change由：

> Linear Layer

完成。

ReLU只改变数值。

---

# 185. 为什么常把 `Linear + ReLU` 当成一个 Feature Layer？

因为：

- Linear创造新的 weighted combinations；
- ReLU根据输入选择 active combinations。

二者配合才形成：

> nonlinear feature transformation。

单独 ReLU没有 learned directions，

单独 Linear没有 nonlinear gating。

---

# 186. 这也是 MLP 最基本的单元

典型：

\[
\boxed{
h=
ReLU(Wx+b)
}
\]

然后堆叠：

\[
h_2=
ReLU(W_2h_1+b_2)
\]

最终：

\[
y=W_3h_2+b_3
\]

这就是最经典的 feedforward neural network形式之一。

---

# 187. 为什么最后一层常没有 ReLU？

因为 output semantics可能需要：

- negative values；
- logits；
- unrestricted regression。

所以通常 hidden layers用 activation，

最终 output layer根据任务选择：

- Linear；
- Sigmoid；
- Softmax；
- other transform。

---

# 188. ACT FFN 第二 Linear 后为什么不马上再 ReLU？

因为 Transformer定义：

\[
FFN(x)
=
Linear_2(
ReLU(
Linear_1(x)
)
)
\]

然后 output成为：

> signed residual update。

如果第二 Linear后也ReLU，

会限制：

\[
F(x)\ge0
\]

逐 feature，

改变 residual update function。

---

# 189. 这就是为什么 ReLU Placement 很重要

标准 FFN：

```text
Linear1
→ ReLU
→ Linear2
```

不是：

```text
Linear1
→ ReLU
→ Linear2
→ ReLU
```

原始 Transformer公式明确如此。

---

# 190. Dropout 又在什么位置？

ACT：

```text
Linear1
→ ReLU
→ Dropout
→ Linear2
→ Residual Dropout
→ Add
```

所以 ReLU负责：

> nonlinear gating。

Dropout负责：

> stochastic regularization。

不要把两者功能混起来。

---

# 191. ReLU 的 Output 可以作为 “Mask” 吗？

它自然产生 zeros，

但 output还有正 magnitude。

如果只关心 active/inactive pattern，

可以定义：

\[
m_i=1[z_i>0]
\]

但 model实际 forward使用的是：

\[
z_i m_i
\]

而不是仅：

\[
m_i
\]

---

# 192. ReLU 和 Dropout 都会产生0，但原因不同

### ReLU 0

由 input content决定：

\[
z\le0
\]

deterministic。

### Dropout 0

由 random Bernoulli mask决定。

所以：

\[
\boxed{
\text{ReLU zero}
=
\text{content-dependent}
}
\]

\[
\boxed{
\text{Dropout zero}
=
\text{training stochastic}
}
\]

---

# 193. Eval 时这两种0有什么区别？

Dropout：

> eval关闭，不再产生随机0。

ReLU：

> eval仍然根据 pre-activation产生0。

所以 inference里的 sparse activation：

> 主要可能来自 ReLU等 activation，而不是 Dropout。

---

# 194. ReLU 与 Attention Mask 的0也不同

Attention mask：

> 某些 routing connection非法。

ReLU：

> 某个 feature response处于 inactive side。

三个“0”：

- ReLU；
- Dropout；
- Attention Mask；

数学来源完全不同。

---

# 195. ReLU 和 Softmax 的 Competition 不同

Softmax：

> 所有 positions通过 denominator互相竞争。

一个 logit变大：

> 会降低其他 weights的相对份额。

ReLU：

> 每个 element独立 threshold。

一个 unit变大：

> 不会因为 ReLU本身强迫另一个 unit变小。

所以：

\[
\boxed{
ReLU
\text{ has no sum-to-one competition}
}
\]

---

# 196. 为什么 ReLU 很适合 Wide FFN？

Wide hidden layer：

\[
d_{ff}=3200
\]

可以创建很多:

\[
w_j^\top x+b_j
\]

candidate features。

ReLU随后让不同 inputs：

> 选择不同 subset。

这提供大量：

> conditional feature combinations。

---

# 197. 但“3200 ReLU = 2^{3200} Regions”能直接这么说吗？

不能。

虽然理论 activation patterns有：

\[
2^{3200}
\]

种 bit combinations，

但并不是所有 patterns都一定由：

\[
512\text{-D}
\]

input和特定 hyperplanes实现。

实际可达 region数量受：

- input dimension；
- weights；
- architecture；

约束。

所以不要用：

\[
2^{3200}
\]

直接声称 ACT一定有那么多 regions。

---

# 198. 正确说法

3200 ReLU units：

> 提供大量 possible activation boundaries和 patterns，

从而显著丰富 piecewise-affine capacity。

具体可达 region数量：

> 依赖参数和 geometry。

---

# 199. ReLU 是否一定让一半 Units 为0？

不一定。

如果 pre-activation distribution正负对称且 threshold在0，

可能粗略接近一半。

但训练后：

- bias；
- weight；
- LayerNorm；
- data；

会改变 distribution。

所以：

\[
\boxed{
\text{ReLU sparsity is learned/data-dependent, not fixed at 50\%.}
}
\]

---

# 200. 为什么 Bias 可能控制 Sparsity？

如果：

\[
b
\]

很正，

更多 inputs满足：

\[
w^\top x+b>0
\]

unit更常 active。

如果：

\[
b
\]

很负，

更常 inactive。

所以 activation sparsity：

> 与 learned bias直接相关。

---

# 201. ReLU 后的 “True Zero” 有什么特别？

Sigmoid：

\[
\sigma(x)
\]

有限 input时通常：

\[
>0
\]

Tanh：

> 也通常不是大量精确0。

ReLU则整个负半轴：

\[
\rightarrow0
\]

所以会产生：

> exact zeros，

这正是 Glorot et al. 2011强调的 sparse representation性质之一。

---

# 202. 但 Exact Zero 也带来 Dying Risk

同一个 hard cutoff：

\[
x\le0\rightarrow0
\]

既带来：

> sparse activations，

也意味着：

> negative side gradient=0。

所以 ReLU的优点和缺点：

> 来自同一个 hard rectification机制。

---

# 203. GELU 为什么更“软”？

GELU不会在：

\[
x=0
\]

突然从 slope 0切成 slope 1。

它平滑过渡，

并允许部分小负值传递。

所以：

> hard sparse gating减少，

但 gradient更平滑。

---

# 204. 为什么 ReLU 仍然值得单独学？

因为它是理解：

- MLP；
- CNN；
- ResNet；
- original Transformer；
- ACT；

最经典 activation之一。

而且它让：

> “深度为什么有意义”

这个问题可以被非常清楚地数学解释。

---

# 205. 常见误解一：ReLU 是 Linear Function，因为名字里有 Linear

**错误。**

它不满足 linearity。

---

# 206. 常见误解二：ReLU 是 Nonlinear，所以它不能 Piecewise Linear

**错误。**

它是：

\[
\boxed{
\text{globally nonlinear, piecewise linear}
}
\]

---

# 207. 常见误解三：ReLU 的非线性来自正区间弯曲

**错误。**

正区间就是直线：

\[
y=x
\]

关键是：

> 0处 slope发生切换。

---

# 208. 常见误解四：一个 ReLU 会产生一个曲线

更准确：

> 一个 ReLU产生一个 hinge / kink。

很多 hinges组合才能形成复杂折线结构。

---

# 209. 常见误解五：ReLU 本身有 Trainable Weight

**错误。**

标准 ReLU参数量：

\[
0
\]

---

# 210. 常见误解六：ReLU Boundary 固定在原始 Input 的0

**错误。**

ReLU作用于：

\[
z=w^\top x+b
\]

所以 input-space boundary：

\[
w^\top x+b=0
\]

是 learned。

---

# 211. 常见误解七：Bias 不影响 ReLU 激活区域

**错误。**

bias直接平移 activation boundary。

---

# 212. 常见误解八：ReLU 后所有 Network Outputs 都不能为负

**错误。**

后续 Linear可以重新产生负值。

---

# 213. 常见误解九：ACT 使用 ReLU，所以机器人动作只能为正

**错误。**

ReLU只在 FFN hidden stage。

最终 action head是 unrestricted Linear output。

---

# 214. 常见误解十：ReLU 完全解决 Vanishing Gradient

**错误。**

正区间不饱和是优势，

但负区间 gradient=0，

深层 Jacobian仍可能产生 gradient问题。

---

# 215. 常见误解十一：ReLU 在0有 Derivative=0 是数学定义

严格数学上：

\[
ReLU'(0)
\]

不存在。

PyTorch只是选择：

\[
0
\]

作为 backward convention/subgradient。

---

# 216. 常见误解十二：不可导点意味着 SGD不能训练 ReLU

**错误。**

Subgradient / autodiff conventions足以处理这种 piecewise-smooth函数。

---

# 217. 常见误解十三：Dying ReLU 是指 Weight 被删除了

**错误。**

参数仍然存在。

只是 neuron长期处于 inactive区，

本地 gradient为0。

---

# 218. 常见误解十四：ReLU 的0和 Dropout 的0一样

**错误。**

一个 deterministic content-dependent，

一个 stochastic regularization。

---

# 219. 常见误解十五：Sparse Activation 等于 Sparse Parameters

**错误。**

Weights仍可能完全 dense。

---

# 220. 常见误解十六：很多0会自动减少 GPU FLOPs

标准 dense kernels：

> 通常不会自动利用这些0。

---

# 221. 常见误解十七：ReLU 会改变 Tensor Shape

**错误。**

element-wise：

\[
(*)\rightarrow(*)
\]

---

# 222. 常见误解十八：ReLU 会让不同 Tokens 相互通信

**错误。**

它不跨 token。

---

# 223. 常见误解十九：ReLU 和 Max Pooling 是一回事

**错误。**

ReLU只拿每个 scalar与0比较。

---

# 224. 常见误解二十：ReLU 和 Softmax 都是“归一化”

**错误。**

ReLU不做 normalization。

---

# 225. 常见误解二十一：ReLU 输出是 Probability

**错误。**

可以：

\[
>1
\]

也不 sum-to-one。

---

# 226. 常见误解二十二：Linear→ReLU 之后还是一个 Linear Layer

**错误。**

ReLU让 active mask随 input变化。

---

# 227. 常见误解二十三：ReLU→Linear 可以和前面的 Linear 合并

**一般不能。**

非线性阻止矩阵合并。

---

# 228. 常见误解二十四：3200 个 ACT ReLU Units 必定产生 \(2^{3200}\) 可达区域

**错误。**

可达 activation patterns受 input dimension和geometry约束。

---

# 229. 常见误解二十五：每个 ReLU Hidden Unit都有清晰人类语义

**不保证。**

需要 empirical interpretability证据。

---

# 230. 常见误解二十六：ReLU 一定比 GELU 更差，因为它更旧

**错误。**

activation choice依赖具体模型与训练条件。

---

# 231. 常见误解二十七：ACT 支持 GELU，所以 canonical ACT 就用了 GELU

**错误。**

默认：

\[
\boxed{
activation="relu"
}
\]

---

# 232. 常见误解二十八：ReLU 训练和推理行为不同

**错误。**

标准 ReLU两者完全一样。

---

# 233. 常见误解二十九：`model.eval()` 会关闭 ReLU

**错误。**

它继续执行：

\[
max(0,x)
\]

---

# 234. 常见误解三十：ReLU 为0说明这个 Feature 永远没用

**错误。**

只说明：

> 当前 input下该 pre-activation没有通过这个 ReLU gate。

别的 input可能 active。

---

# 235. 用一张图理解 Linear vs ReLU

```text
Linear:

y
↑
│       /
│      /
│     /
│    /
│___/____________→ x

整个空间同一 slope
```

ReLU：

```text
y
↑
│           /
│          /
│         /
│        /
│_______/_________→ x
        0

左边 slope = 0
右边 slope = 1
```

关键不是曲线，

而是：

\[
\boxed{
\text{slope depends on region}
}
\]

---

# 236. 用一张图理解 ReLU Neuron

```text
Input x
   │
   ▼
z = wᵀx + b
   │
   ├── z ≤ 0 ──→ output 0
   │
   └── z > 0 ──→ output z
```

boundary：

\[
\boxed{
w^\top x+b=0
}
\]

由：

\[
w,b
\]

学习。

---

# 237. 用一张图理解 ReLU MLP

```text
x
│
▼
Linear W₁,b₁
│
▼
many pre-activations z₁...zₘ
│
▼
ReLU
│
├─ some active
├─ some zero
└─ activation pattern depends on x
│
▼
Linear W₂,b₂
│
▼
y
```

这就是：

\[
\boxed{
y=W_2ReLU(W_1x+b_1)+b_2
}
\]

---

# 238. 用一张图理解 ACT FFN

```text
one ACT token
x ∈ R^512
    │
    ▼
Linear 512 → 3200
    │
    ▼
z ∈ R^3200
    │
    ▼
ReLU element-wise
    │
    ├── negative features → 0
    └── positive features → keep magnitude
    │
    ▼
Dropout (training only)
    │
    ▼
Linear 3200 → 512
    │
    ▼
signed FFN update
    │
    ▼
Residual Add + LayerNorm
```

---

# 239. 一句话真正理解 ReLU

> **ReLU 的真正作用不是“把负数变成0”这么简单，而是让前面的 learned affine response \(w^\top x+b\) 成为一个 input-dependent gate：hyperplane \(w^\top x+b=0\) 把 representation space分成 active 与 inactive 两侧，因此不同输入会激活不同 hidden units、选择不同的 effective affine transformation。单个 ReLU只制造一个 hinge，但大量 ReLU与多层 Linear组合后会把空间切成大量 piecewise-affine regions，使整个网络能够用许多局部简单规则拼成复杂的全局 nonlinear function。**

---

# 240. 一句话理解“为什么没有 ReLU 就不行”

> **任意数量的纯 Linear/Affine layers都可以代数合并成一个 \(Wx+b\)，因此深度本身不会增加函数类别；插入 ReLU 后，active mask \(D(x)\) 会随输入变化，使网络 Jacobian从固定的 \(W\) 变成类似 \(W_2D(x)W_1\) 的 input-dependent mapping，因此不同输入区域使用不同 affine rules，深层组合才真正获得 nonlinear expressive power。**

---

# 241. 一句话连接 ACT

> **ACT 的 Transformer FFN 对每一个 512-D observation/action hidden token先通过 `Linear(512,3200)`构造3200个 learned pre-activation feature responses，再用 ReLU按当前 token内容选择 active subset，随后 `Linear(3200,512)`把这些被激活的 features重新组合成 signed residual update；因此3200维并不是3200个新 token，而是3200个可被 input-dependent ReLU gates动态启用或关闭的 feature channels。**

---

# 242. 下一篇：MLP

现在我们已经分别真正理解：

\[
Linear
\]

和：

\[
ReLU
\]

下一步就可以把它们组成深度学习最基础的完整网络：

\[
\boxed{
MLP
}
\]

下一篇：

> **[MLP：Linear + ReLU 为什么就能组成真正的神经网络？](./mlp.md)**

会完整解释：

- Perceptron、Dense Layer、MLP之间是什么关系；
- input layer / hidden layer / output layer是什么意思；
- 为什么“层数”在不同文献里会有不同计数；
- forward propagation；
- hidden representation；
- activation；
- loss；
- backpropagation；
- chain rule如何穿过多层；
- 一个完整小型 regression/classification网络怎样训练；
- Universal Approximation应该如何正确理解；
- width vs depth；
- 为什么 Transformer FFN本质是一个 position-wise two-layer MLP；
- 为什么 ACT Transformer虽然核心是 Attention，但内部依然大量依赖 MLP。

---

## Primary Historical Source：Rectified Linear Units

Vinod Nair, Geoffrey E. Hinton.

**Rectified Linear Units Improve Restricted Boltzmann Machines.**  
ICML 2010.

- Paper: https://icml.cc/2010/papers/432.pdf
- Metadata: https://mlanthology.org/icml/2010/nair2010icml-rectified/

这篇 early rectified-unit工作研究：

> noisy rectified linear units in Restricted Boltzmann Machines。

论文特别指出，相比 binary units，rectified linear units可以更好地保留：

> relative intensity information

在多层 feature detectors中的传播。

需要注意：

> 该论文具体讨论的是 RBM / stochastic rectified-unit formulation，不应把其全部 probabilistic细节直接等同于今天 deterministic `nn.ReLU()`。

---

## Primary Deep-Network Source

Xavier Glorot, Antoine Bordes, Yoshua Bengio.

**Deep Sparse Rectifier Neural Networks.**  
AISTATS 2011.

- PMLR: https://proceedings.mlr.press/v15/glorot11a.html
- PDF: https://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf

论文明确讨论 rectifying nonlinearities用于 deep networks，并强调：

- hard non-linearity；
- \(0\) 点 non-differentiability；
- exact zeros；
- sparse representations；
- deep rectifier networks在监督学习任务中的有效性。

本文关于：

\[
\boxed{
\text{hard rectification}
}
\]

和：

\[
\boxed{
\text{true-zero sparse activations}
}
\]

的历史背景主要以此为依据。

---

## PyTorch Primary Implementation Reference

PyTorch `torch.nn.ReLU`:

https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html

官方定义：

\[
\boxed{
ReLU(x)=\max(0,x)
}
\]

并明确：

- element-wise；
- input/output shape相同；
- `inplace=False`为默认。

---

## PyTorch Autograd：不可导点

PyTorch Autograd Mechanics:

https://docs.pytorch.org/docs/stable/notes/autograd.html

PyTorch文档说明：

> 对局部 convex但不可导的 elementary function，会使用 minimum-norm subgradient这一规则。

ReLU在：

\[
x=0
\]

数学上不可导。

其 subgradient set：

\[
[0,1]
\]

PyTorch `F.relu` backward在：

\[
x=0
\]

采用：

\[
\boxed{
0
}
\]

这一 convention。

相关实现逻辑可在 PyTorch source中看到 ReLU backward只对：

\[
x>0
\]

传递 gradient。

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
\max(0,xW_1+b_1)W_2+b_2
}
\]

原始 Base Transformer：

\[
d_{\text{model}}=512
\]

\[
d_{ff}=2048
\]

因此：

\[
\boxed{
512
\rightarrow
2048
\rightarrow
ReLU
\rightarrow
512
}
\]

论文明确说：

> FFN consists of two linear transformations with a ReLU activation in between。

---

## ACT Primary Source

Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.

**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.**  
RSS 2023.

- arXiv: https://arxiv.org/abs/2304.13705
- PDF: https://arxiv.org/pdf/2304.13705

ACT Table III给出：

\[
\boxed{
hidden\ dimension=512
}
\]

\[
\boxed{
feedforward\ dimension=3200
}
\]

所以 ACT FFN：

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

---

## ACT Official Implementation

Repository:

https://github.com/tonyzhaozh/act

Transformer source:

https://github.com/tonyzhaozh/act/blob/main/detr/models/transformer.py

DETR-style Transformer constructor默认：

```python
activation="relu"
```

FFN：

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

activation helper：

```python
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu

    if activation == "gelu":
        return F.gelu

    if activation == "glu":
        return F.glu
```

所以通用 implementation支持多个 activation，

但 canonical default：

\[
\boxed{
ReLU
}
\]

---

## 本文知识连接

### 数学

- Linear Function
- Affine Transformation
- Piecewise Linear Function
- Hyperplane
- Derivative
- Gradient
- Jacobian
- Convex Function
- Subgradient

### Deep Learning

- [Linear Layer](./linear-layer.md)
- [MLP](./mlp.md)
- [Feed-Forward Network](./feed-forward-network.md)
- Sigmoid
- Tanh
- GELU
- Leaky ReLU
- PReLU
- Activation Function

### Transformer

- [Transformer](./transformer.md)
- [Feed-Forward Network](./feed-forward-network.md)
- [Residual Connection](./residual-connection.md)
- [Layer Normalization](./layer-normalization.md)
- [Dropout](./dropout.md)

### Robot Learning

- [ACT Architecture](../robot-learning/act/architecture.md)
- [ACT Training](../robot-learning/act/training.md)
- [ACT Complete Data Flow](../robot-learning/act/complete-data-flow.md)

### 下一步

- [MLP](./mlp.md)
