---
title: "Layer Normalization：为什么 Transformer 每一层都要重新“标准化”？"
description: "从均值、方差和 z-score 标准化出发，严格理解 LayerNorm 对 Transformer hidden features 做了什么、为什么它不等于标准正态分布、γ/β 与 ε 的作用、与 BatchNorm 的根本区别，以及 Post-LN / Pre-LN 和 ACT 中 nn.LayerNorm(512) 的真实计算轴。"
status: reviewed
pageType: concept
canonical: /deep-learning/layer-normalization
updated: "2026-09-15"
---

# Layer Normalization：为什么 Transformer 每一层都要重新“标准化”？

上一篇 [Residual Connection](./residual-connection.md) 里，我们已经知道 Transformer 并不是简单：

\[
x\rightarrow F(x)
\]

而是先做：

\[
x+F(x)
\]

但原始 Transformer 真正使用的公式还多了一步：

\[
\boxed{
\operatorname{LayerNorm}
\left(
x+F(x)
\right)
}
\]

也就是说：

> Attention 算完，加回 residual，**还不结束**。

还要做一次：

> **Layer Normalization。**

为什么？

为什么一条已经训练好的 hidden vector 还要不断：

1. 减均值；
2. 除标准差；
3. 再乘一个 \(\gamma\)；
4. 再加一个 \(\beta\)？

更容易产生困惑的是：

如果 LayerNorm 做完以后：

\[
mean\approx0
\]

\[
variance\approx1
\]

那它是不是在强迫 neural representation：

> 服从标准正态分布：

\[
\mathcal N(0,1)
\]

？

**不是。**

这正是这一篇首先要解决的误区。

我们会完整回答：

1. LayerNorm 到底在 normalize 什么？
2. 对 `[B,N,D]` 的哪个轴求均值和方差？
3. 为什么 `nn.LayerNorm(512)` 不会把 1202 个 ACT tokens 混在一起？
4. 为什么减均值之后还要除标准差？
5. 均值 0、方差 1 为什么不等于标准正态分布？
6. \(\epsilon\) 为什么存在？
7. 为什么严格来说 normalize 后的 variance 不一定恰好等于 1？
8. \(\gamma\) 和 \(\beta\) 为什么不会让 normalization“白做”？
9. LayerNorm 和 BatchNorm 到底差在哪里？
10. 为什么 LayerNorm 训练和推理用同一套 statistics？
11. 原始 Transformer 为什么是 Post-LN？
12. Pre-LN 为什么会改变 gradient path？
13. ACT 中的 `nn.LayerNorm(512)` 实际对 `[1202,B,512]` 做了什么？

---

# 1. 先从一个 Hidden Vector 开始

假设 Transformer 中某个 token 的 hidden representation 是：

\[
x=
[
x_1,x_2,\ldots,x_D
]
\]

例如：

\[
D=512
\]

所以：

\[
x\in\mathbb R^{512}
\]

LayerNorm首先看的是：

> **这个 token 自己的 512 个 features。**

它不是先看：

- 其他 batch samples；
- 其他 tokens；
- 整个训练数据集。

先计算这个 vector 自己的 mean：

\[
\boxed{
\mu
=
\frac{1}{D}
\sum_{j=1}^{D}x_j
}
\]

然后计算 variance：

\[
\boxed{
\sigma^2
=
\frac{1}{D}
\sum_{j=1}^{D}
(x_j-\mu)^2
}
\]

再标准化：

\[
\boxed{
\hat x_j
=
\frac{
x_j-\mu
}{
\sqrt{\sigma^2+\epsilon}
}
}
\]

最后再做 learnable affine transformation：

\[
\boxed{
y_j
=
\gamma_j\hat x_j+\beta_j
}
\]

这就是 Transformer 中最常见 LayerNorm 的完整骨架。

---

# 2. LayerNorm 的四步

可以把它拆成：

```text
x
↓
减去当前 token 的 feature mean
↓
除以当前 token 的 feature standard deviation
↓
乘 learned γ
↓
加 learned β
↓
y
```

数学：

\[
\boxed{
x
\rightarrow
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
\rightarrow
\gamma\odot\hat x+\beta
}
\]

---

# 3. 一个 4 维手算例子

假设：

\[
x=[1,2,3,4]
\]

维度：

\[
D=4
\]

均值：

\[
\mu
=
\frac{1+2+3+4}{4}
=
2.5
\]

---

# 4. 减均值

\[
x-\mu
=
[-1.5,-0.5,0.5,1.5]
\]

现在 mean：

\[
\frac{-1.5-0.5+0.5+1.5}{4}
=
0
\]

所以：

\[
\boxed{
\text{减均值负责把中心移到 0}
}
\]

---

# 5. 计算方差

使用 LayerNorm / PyTorch 常见的 population-style estimator：

\[
\sigma^2
=
\frac{
(-1.5)^2+
(-0.5)^2+
0.5^2+
1.5^2
}{4}
\]

\[
=
\frac{
2.25+0.25+0.25+2.25
}{4}
\]

\[
=
1.25
\]

所以：

\[
\sigma
=
\sqrt{1.25}
\approx
1.118
\]

先暂时忽略很小的：

\[
\epsilon
\]

---

# 6. 再除标准差

得到：

\[
\hat x
\approx
[
-1.342,
-0.447,
0.447,
1.342
]
\]

它的 mean：

\[
0
\]

variance：

\[
1
\]

所以“减均值、除标准差”的作用非常明确：

\[
\boxed{
\text{消除共同平移}
+
\text{统一整体尺度}
}
\]

---

# 7. 为什么减均值之后还必须除标准差？

这是非常核心的一步。

假设两个 token：

\[
x^{(A)}
=
[1,2,3,4]
\]

另一个：

\[
x^{(B)}
=
[100,200,300,400]
\]

二者 pattern其实完全相同：

\[
x^{(B)}
=
100x^{(A)}
\]

---

仅减均值：

\[
x^{(A)}-\mu_A
=
[-1.5,-0.5,0.5,1.5]
\]

但：

\[
x^{(B)}-\mu_B
=
[-150,-50,50,150]
\]

虽然它们现在都：

\[
mean=0
\]

但尺度仍差：

\[
100\times
\]

所以：

\[
\boxed{
\text{减均值只消除中心差异，不消除尺度差异}
}
\]

---

# 8. 除标准差之后

A：

\[
\frac{x^{(A)}-\mu_A}{\sigma_A}
\]

B：

\[
\frac{x^{(B)}-\mu_B}{\sigma_B}
\]

因为：

\[
\sigma_B=100\sigma_A
\]

所以两者得到同样 normalized pattern：

\[
\boxed{
\hat x^{(A)}
=
\hat x^{(B)}
}
\]

忽略 \(\epsilon\)。

这就是除标准差的意义：

> 把“偏离自己均值多少”转换成“偏离自己多少个标准差”。

---

# 9. 这和 Z-Score 是不是很像？

是。

经典 z-score：

\[
z
=
\frac{x-\mu}{\sigma}
\]

LayerNorm 的核心 normalization step：

\[
\hat x
=
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
\]

数学形式本质上就是：

> **对当前 normalization group 做 z-score-like standardization。**

但要注意：

LayerNorm 后面还有：

\[
\gamma,\beta
\]

并且：

> 这里的“样本集合”是一个 hidden vector 中被指定 normalization 的 feature dimensions。

所以不要把普通数据预处理里的 z-score 和神经网络 LayerNorm完全当成同一个操作场景。

---

# 10. 最重要的纠错：Mean=0、Variance=1 不等于标准正态分布

标准正态分布的定义是：

\[
\boxed{
X\sim\mathcal N(0,1)
}
\]

它不仅规定：

\[
E[X]=0
\]

\[
Var(X)=1
\]

还规定：

> 整个概率密度函数必须具有 Gaussian bell shape。

仅仅 mean 和 variance相同：

> 不足以确定 distribution shape。

---

# 11. 一个最直接的反例

考虑四个数：

\[
[-1,-1,1,1]
\]

均值：

\[
0
\]

variance：

\[
1
\]

但这个经验分布只有：

\[
-1
\]

和：

\[
+1
\]

两个值。

它显然不是连续的：

\[
\mathcal N(0,1)
\]

bell curve。

所以：

\[
\boxed{
mean=0,\ variance=1
\not\Rightarrow
Gaussian
}
\]

---

# 12. 标准化和标准正态化不是同一件事

严格区分：

### Standardization

把数据变成：

\[
mean\approx0,\qquad variance\approx1
\]

---

### Standard Normal Distribution

要求随机变量分布：

\[
\boxed{
\mathcal N(0,1)
}
\]

LayerNorm做的是：

> normalization / standardization operation。

它没有执行：

> “把任意 distribution 映射成 Gaussian distribution”

这种 distribution transformation。

---

# 13. 为什么很多人会误会？

因为：

\[
\frac{x-\mu}{\sigma}
\]

也出现在把 Gaussian random variable：

\[
X\sim\mathcal N(\mu,\sigma^2)
\]

标准化成：

\[
Z\sim\mathcal N(0,1)
\]

的推导中。

但是那个结论成立的前提是：

\[
\boxed{
X\text{ 本来就是 Gaussian}
}
\]

因为 Gaussian 在 affine transformation 下仍然是 Gaussian。

LayerNorm中的 hidden features：

> 并没有被假定为一个 Gaussian random sample。

所以不能跳到：

\[
\mathcal N(0,1)
\]

---

# 14. 一个非常重要的逻辑顺序

如果：

\[
X\sim\mathcal N(\mu,\sigma^2)
\]

那么：

\[
Z=
\frac{X-\mu}{\sigma}
\]

确实：

\[
Z\sim\mathcal N(0,1)
\]

但：

\[
\boxed{
\frac{x-\mu}{\sigma}
\text{ 有 mean 0 / variance 1}
}
\]

不代表：

\[
\boxed{
x\text{ 必须是 Gaussian}
}
\]

这是两个完全不同的命题。

---

# 15. LayerNorm 到底是对谁求均值？

现在进入 Transformer最关键的 shape问题。

假设：

\[
X
\in
\mathbb R^{B\times N\times D}
\]

其中：

- \(B\)：batch size；
- \(N\)：token count；
- \(D\)：hidden dimension。

使用：

```python
nn.LayerNorm(D)
```

PyTorch会对：

\[
\boxed{
\text{最后一个维度 }D
}
\]

做 normalization。

所以对每个：

\[
(b,n)
\]

分别有：

\[
\mu_{b,n}
=
\frac1D
\sum_{d=1}^{D}
X_{b,n,d}
\]

---

# 16. 因此一个 Token 一套 Statistics

例如：

\[
B=2
\]

\[
N=3
\]

\[
D=4
\]

总共有：

\[
2\times3=6
\]

个 token vectors。

LayerNorm(4) 会为每一个 vector：

\[
X_{b,n,:}
\]

单独计算：

- 一个 mean；
- 一个 variance。

不是：

> 六个 token混在一起算一个 mean。

---

# 17. Shape 图

输入：

```text
Batch 0:
    Token 0 → [x x x x] → 自己算 μ₀₀, σ²₀₀
    Token 1 → [x x x x] → 自己算 μ₀₁, σ²₀₁
    Token 2 → [x x x x] → 自己算 μ₀₂, σ²₀₂

Batch 1:
    Token 0 → [x x x x] → 自己算 μ₁₀, σ²₁₀
    Token 1 → [x x x x] → 自己算 μ₁₁, σ²₁₁
    Token 2 → [x x x x] → 自己算 μ₁₂, σ²₁₂
```

所以：

\[
\boxed{
LayerNorm(D)
\text{ 不跨 token、不跨 batch 统计}
}
\]

---

# 18. 但说“LayerNorm 不混 Feature”也是错的

注意一个微妙点。

虽然 LayerNorm不跨 token，

但同一个 token的 features：

> **并不是独立处理。**

因为：

\[
\mu
=
\frac1D
\sum_jx_j
\]

以及：

\[
\sigma^2
=
\frac1D
\sum_j(x_j-\mu)^2
\]

都依赖所有：

\[
D
\]

个 features。

所以某一个：

\[
x_j
\]

改变，

会影响：

- mean；
- variance；

进而影响其他：

\[
\hat x_i
\]

---

# 19. 因此 LayerNorm 的准确描述

\[
\boxed{
\text{不跨 Tokens}
}
\]

但：

\[
\boxed{
\text{会在一个 Token 的 Feature Dimensions 内耦合统计量}
}
\]

这比简单说：

> “LayerNorm 是逐 feature 操作”

严谨得多。

---

# 20. 为什么叫 Layer Normalization？

原始 LayerNorm论文把 BatchNorm的思想“转置”到单个 training case：

BatchNorm：

> 对 mini-batch 中某个 neuron / feature 的 activations算 statistics。

LayerNorm：

> 对一个 training case中同一 layer的 hidden units算 statistics。

这就是名称来源。

在 Transformer语境中，

一个 token 的：

\[
D
\]

维 hidden state正好构成自然 normalization group。

---

# 21. 原始 LayerNorm 论文为什么想摆脱 Batch Statistics？

LayerNorm论文指出 BatchNorm依赖：

> mini-batch statistics。

这带来几个问题：

- 结果依赖 batch size；
- sequence/RNN使用不自然；
- train/test计算需要处理 batch statistics和population statistics差异。

LayerNorm则对：

> 单个 case内部 hidden units

求 statistics。

因此：

\[
\boxed{
\text{不依赖其他 batch samples}
}
\]

---

# 22. BatchNorm 到底怎样算？

以一个简单 feature vector模型为例。

假设 batch：

\[
x^{(1)},x^{(2)},\ldots,x^{(B)}
\]

某个 feature dimension：

\[
d
\]

BatchNorm主要会沿 batch方向收集：

\[
x_d^{(1)},
x_d^{(2)},\ldots,x_d^{(B)}
\]

算：

\[
\mu_d
\]

和：

\[
\sigma_d^2
\]

所以一个 sample的 normalization：

> 依赖同一 batch其他 samples。

---

# 23. LayerNorm 刚好换了方向

对于一个 sample / token：

\[
x=
[x_1,\ldots,x_D]
\]

LayerNorm沿：

\[
D
\]

features算：

\[
\mu,\sigma^2
\]

所以：

```text
BatchNorm:
固定 feature，看很多样本

LayerNorm:
固定样本/token，看很多 features
```

这是最核心区别之一。

---

# 24. 一张二维矩阵理解

假设：

\[
X
=
\begin{bmatrix}
x_{11}&x_{12}&x_{13}\\
x_{21}&x_{22}&x_{23}\\
x_{31}&x_{32}&x_{33}
\end{bmatrix}
\]

rows：

> samples。

columns：

> features。

---

BatchNorm高层上：

> 每一列算 statistics。

```text
↓   ↓   ↓
feature-wise across samples
```

---

LayerNorm：

> 每一行算 statistics。

```text
→ → →
within one sample across features
```

这是 LayerNorm论文最经典的区别直觉。

---

# 25. Transformer 为什么更偏爱 LayerNorm？

Transformer sequence具有：

- 可变长度；
- token-wise representations；
- 可能小 batch；
- autoregressive inference；
- training/inference sequence情况不同。

LayerNorm不依赖：

> mini-batch其他样本。

因此更自然地作用于：

\[
\text{每个 token hidden state}
\]

这也是它在 Transformer中成为基础组件的重要原因。

---

# 26. BatchNorm 训练和推理为什么通常不同？

标准 BatchNorm训练时使用：

> 当前 mini-batch statistics。

Inference时通常不能依赖：

> “刚好和当前样本一起进来的随机其他样本”

所以会使用训练过程中维护的：

> running mean / running variance。

因此：

\[
\boxed{
BatchNorm:
train\ computation
\neq
eval\ computation
}
\]

典型实现如此。

---

# 27. LayerNorm 为什么 Train/Eval 一样？

LayerNorm每个输入自己就可以计算：

\[
\mu(x)
\]

和：

\[
\sigma^2(x)
\]

不需要 population estimate。

所以训练：

\[
LN(x)
\]

推理：

\[
LN(x)
\]

都是使用：

> 当前这个 input的 statistics。

LayerNorm原论文明确强调：

> train 和 test执行相同 computation。

PyTorch `LayerNorm` 文档也明确说：

> statistics are computed from input data in both training and evaluation modes。

---

# 28. 所以 LayerNorm 没有 Running Mean / Running Variance

标准：

```python
nn.LayerNorm(512)
```

不像 BatchNorm那样维护：

```text
running_mean
running_var
```

因为不需要。

这也意味着：

> `model.eval()` 不会把 LayerNorm切换成“另一套历史统计”。

它还是当前输入自己算 mean/variance。

---

# 29. `model.eval()` 对 LayerNorm 什么都不改变吗？

就 normalization statistics而言：

> 基本是同一种计算。

但整个模型里其他模块，例如：

- Dropout；

会受 `eval()`影响。

所以不能说：

> `model.eval()` 对包含 LayerNorm的 Transformer完全没作用。

只是：

\[
\boxed{
LayerNorm\ itself
\text{ does not switch to running statistics}
}
\]

---

# 30. \(\epsilon\) 是什么？

真实公式：

\[
\hat x_j
=
\frac{
x_j-\mu
}{
\sqrt{\sigma^2+\epsilon}
}
\]

为什么不是：

\[
\frac{x_j-\mu}{\sigma}
\]

？

因为如果：

\[
\sigma^2=0
\]

就会除以：

\[
0
\]

产生数值问题。

---

# 31. 什么时候 Variance 会等于 0？

例如：

\[
x=[5,5,5,5]
\]

均值：

\[
\mu=5
\]

每一维：

\[
x_j-\mu=0
\]

所以：

\[
\sigma^2=0
\]

如果直接：

\[
\frac0{0}
\]

未定义。

加入：

\[
\epsilon>0
\]

后：

\[
\sqrt{0+\epsilon}
\]

非零。

---

# 32. PyTorch 默认 ε

当前 PyTorch：

```python
nn.LayerNorm(
    normalized_shape,
    eps=1e-5
)
```

所以默认：

\[
\boxed{
\epsilon=10^{-5}
}
\]

ACT代码：

```python
nn.LayerNorm(d_model)
```

没有手动修改 `eps`，

因此在对应 PyTorch默认设置下使用：

\[
10^{-5}
\]

---

# 33. ε 只是为了避免除 0 吗？

主要目的是：

> numerical stability。

即使 variance不是严格 0，

如果它非常小：

\[
\sigma^2\approx0
\]

那么：

\[
1/\sqrt{\sigma^2}
\]

会非常大，

导致数值敏感。

加入：

\[
\epsilon
\]

给 denominator提供下界。

---

# 34. 一个经常被忽略的数学细节

很多教程说 LayerNorm标准化后：

\[
variance=1
\]

如果没有：

\[
\epsilon
\]

确实如此。

但实际：

\[
\hat x
=
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
\]

所以：

\[
Var(\hat x)
=
\frac{
\sigma^2
}{
\sigma^2+\epsilon
}
\]

因此：

\[
\boxed{
Var(\hat x)<1
}
\]

只要：

\[
\epsilon>0
\]

且 variance有限。

---

# 35. 为什么平时仍说“单位方差”？

因为如果：

\[
\sigma^2\gg\epsilon
\]

例如：

\[
\sigma^2=1
\]

\[
\epsilon=10^{-5}
\]

那么：

\[
\frac1{1.00001}
\approx
0.99999
\]

几乎就是：

\[
1
\]

所以教学中说：

> approximately unit variance

通常没问题。

但严格数学上不一定精确 1。

---

# 36. PyTorch 的 Variance 用 N 还是 N-1？

当前 PyTorch `LayerNorm` 文档明确写：

> variance uses the biased estimator，等价于 `torch.var(..., correction=0)`。

所以：

\[
\boxed{
\sigma^2
=
\frac1D
\sum_{j=1}^D
(x_j-\mu)^2
}
\]

不是统计学 sample variance：

\[
\frac1{D-1}
\sum_j(\cdots)
\]

---

# 37. 为什么这里不追求无偏方差估计？

LayerNorm的目的不是：

> 从少量 observations无偏估计某个未知 population variance。

它是：

> 神经网络里的 deterministic normalization transformation。

所以使用：

\[
1/D
\]

非常自然。

PyTorch也明确采用这种定义。

---

# 38. \(\gamma\) 和 \(\beta\) 是什么？

标准化之后：

\[
\hat x
\]

再做：

\[
\boxed{
y=
\gamma\odot\hat x+\beta
}
\]

其中：

\[
\gamma
\]

叫：

- gain；
- scale；
- weight；

\[
\beta
\]

叫：

- bias；
- shift。

二者都是 trainable。

---

# 39. `LayerNorm(512)` 有多少 γ 和 β？

如果：

\[
D=512
\]

那么：

\[
\gamma\in\mathbb R^{512}
\]

\[
\beta\in\mathbb R^{512}
\]

所以 trainable parameters：

\[
512+512
=
\boxed{
1024
}
\]

默认初始化通常：

\[
\gamma=1
\]

\[
\beta=0
\]

因此刚开始 LayerNorm基本就是纯标准化。

---

# 40. 为什么标准化完还要让模型重新 Scale 和 Shift？

这看起来很反直觉。

我们刚刚费力把 representation变成：

\[
mean\approx0
\]

\[
variance\approx1
\]

然后又：

\[
\gamma\hat x+\beta
\]

把尺度和中心改掉。

是不是白做？

不是。

---

# 41. Normalization 的目标不是“永远强制输出只能均值 0 方差 1”

真正目的更接近：

> 给网络提供一个稳定、标准化的中间坐标系。

然后：

\[
\gamma,\beta
\]

允许模型学习：

> 每个 feature最终需要什么尺度和偏移。

所以：

\[
\boxed{
\text{normalize first}
\rightarrow
\text{learn useful affine calibration}
}
\]

---

# 42. 如果 γ=σ、β=μ，不就能恢复原数据？

对于某个固定 sample，

如果：

\[
\gamma,\beta
\]

能根据当前 sample动态设置成其：

\[
\sigma,\mu
\]

理论上可以恢复。

但 LayerNorm中的：

\[
\gamma,\beta
\]

是：

> 全局 learned parameters，

不是每个 sample自己的 \(\mu,\sigma\)。

所以不能简单说：

> “LayerNorm做了等价于没做。”

它仍然移除了每个 sample的某些 common shift/scale variation。

---

# 43. γ 和 β 是每个 Token 单独的吗？

不是。

同一个 LayerNorm module里的：

\[
\gamma_d,\beta_d
\]

会共享给：

- 所有 batch samples；
- 所有 token positions。

例如：

```python
nn.LayerNorm(512)
```

有一套：

\[
512
\]

维 gamma/beta。

不是：

\[
1202\times512
\]

套参数。

---

# 44. 为什么这和 FFN 的 Position-Wise 参数共享很像？

同样体现：

> Transformer不同 positions使用相同 feature-space rule。

LayerNorm statistics：

> 每个 token自己算。

但 learned：

\[
\gamma,\beta
\]

在 positions之间共享。

所以：

```text
statistics:
token-specific

learned affine parameters:
shared across tokens
```

---

# 45. 一个非常重要的区别

LayerNorm中的：

\[
\mu,\sigma^2
\]

不是 trainable parameters。

它们是：

> 当前 forward 根据 input动态算出来的 numbers。

而：

\[
\gamma,\beta
\]

才是：

> optimizer更新的 trainable parameters。

不要混淆。

---

# 46. Backprop 会经过 Mean 和 Variance 吗？

会。

\[
\mu(x)
\]

和：

\[
\sigma^2(x)
\]

都是 input的 differentiable functions。

所以 gradient不是简单：

\[
1/\sigma
\]

乘回来。

一个 feature改变会影响：

- 自己；
- mean；
- variance；
- 进而其他 normalized features。

---

# 47. LayerNorm 的 Jacobian

对一个 token：

\[
x\in\mathbb R^D
\]

定义：

\[
\bar x_i=x_i-\mu
\]

\[
s=
\sqrt{
\sigma^2+\epsilon
}
\]

\[
\hat x_i=\frac{\bar x_i}{s}
\]

则：

\[
\boxed{
\frac{
\partial \hat x_i
}{
\partial x_j
}
=
\frac{
\delta_{ij}-1/D
}{s}
-
\frac{
\bar x_i\bar x_j
}{
D s^3
}
}
\]

其中：

\[
\delta_{ij}
\]

是 Kronecker delta。

---

# 48. 这个公式告诉我们什么？

即使：

\[
i\neq j
\]

通常：

\[
\frac{
\partial\hat x_i
}{
\partial x_j
}
\neq0
\]

为什么？

因为：

\[
x_j
\]

会改变：

- mean；
- variance；

于是影响：

\[
\hat x_i
\]

所以再次强调：

\[
\boxed{
LayerNorm不跨 Token，
但会耦合一个 Token 内的 Features
}
\]

---

# 49. 加上 γ 后

\[
y_i=\gamma_i\hat x_i+\beta_i
\]

所以：

\[
\boxed{
\frac{
\partial y_i
}{
\partial x_j
}
=
\gamma_i
\left[
\frac{
\delta_{ij}-1/D
}{s}
-
\frac{
\bar x_i\bar x_j
}{
Ds^3
}
\right]
}
\]

这就是为什么在上一篇 Residual文章里说：

> Post-LN residual gradient path不能简单当作纯 identity。

---

# 50. LayerNorm 对整体平移有什么性质？

假设对一个 token所有 features都加同一个常数：

\[
x'=x+c\mathbf1
\]

新的均值：

\[
\mu'=\mu+c
\]

所以：

\[
x'-\mu'
=
x+c-(\mu+c)
=
x-\mu
\]

variance也不变。

因此：

\[
\boxed{
LN\text{ 的标准化部分对共同 feature shift 不敏感}
}
\]

---

# 51. 一个例子

\[
x=[1,2,3,4]
\]

和：

\[
x'=[101,102,103,104]
\]

虽然绝对值差：

\[
100
\]

但减各自 mean之后：

\[
[-1.5,-0.5,0.5,1.5]
\]

完全相同。

所以标准化结果相同，

忽略后续共享 \(\gamma,\beta\)。

---

# 52. LayerNorm 对整体 Scale 又怎样？

假设：

\[
x'=ax
\]

均值：

\[
\mu'=a\mu
\]

variance：

\[
\sigma'^2=a^2\sigma^2
\]

如果：

\[
\epsilon=0
\]

且：

\[
a>0
\]

则：

\[
\frac{
ax-a\mu
}{
|a|\sigma
}
=
\frac{x-\mu}{\sigma}
\]

所以 positive common scale被消除。

---

# 53. 如果 a<0 呢？

则：

\[
|a|=-a
\]

所以 normalized representation整体翻号：

\[
\hat x'
=
-\hat x
\]

因此不是对 negative scaling完全 invariant。

更准确：

> 对正的全局 scale近似 invariant。

---

# 54. ε 会破坏精确 Scale Invariance

实际：

\[
\frac{
a(x-\mu)
}{
\sqrt{
a^2\sigma^2+\epsilon
}
}
\]

不能总是严格化成：

\[
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
\]

因为：

\[
\epsilon
\]

没有随：

\[
a^2
\]

缩放。

当：

\[
\sigma^2\gg\epsilon
\]

时差异很小。

所以严格写：

\[
\boxed{
\text{approximately scale-invariant when }\epsilon\text{ is negligible}
}
\]

---

# 55. LayerNorm 会丢失信息吗？

纯 normalization：

\[
x
\rightarrow
\hat x
\]

会丢掉至少一些关于：

- 全体 features的共同 offset；
- 全体 features的整体 positive scale；

的信息。

因为很多不同：

\[
x
\]

可能映射到同一个：

\[
\hat x
\]

所以 normalization不是一般意义下可逆映射。

---

# 56. 那为什么网络敢丢这些信息？

因为 architecture认为：

> 这些全局 shift / scale variation往往不是每层表示中最重要的 task information，稳定 feature geometry更有价值。

并且：

- residual structure；
- learned \(\gamma,\beta\)；
- surrounding layers；

共同适应这一 normalization。

但不能说：

> LayerNorm是无损操作。

它不是。

---

# 57. LayerNorm 为什么能稳定训练？

这里需要谨慎。

LayerNorm原论文的核心目标是：

> 稳定 hidden-state dynamics、减少不同训练样本/时间步中 activation scale变化带来的优化困难。

它对每个 case自己标准化，

可以控制 hidden activations的中心与尺度。

但现代对 normalization为何有效有很多更深入解释。

所以不要把效果简化成唯一一句：

> “因为防止数值爆炸。”

更准确是：

\[
\boxed{
\text{它重参数化并控制中间表示的统计尺度，从而改变优化动态}
}
\]

---

# 58. LayerNorm 会不会保证每层数值绝不爆炸？

不保证。

因为最终还有：

\[
\gamma,\beta
\]

网络也可能学习大权重。

整个系统还受：

- residual；
- initialization；
- optimizer；
- learning rate；
- depth；

影响。

Normalization显著帮助训练稳定，

但不是绝对数值安全证明。

---

# 59. 为什么 Transformer 原论文把 LayerNorm 放 Residual 后？

原始 2017 Transformer规定：

\[
\boxed{
LayerNorm(
x+
Sublayer(x)
)
}
\]

即：

> Post-LN。

Encoder每层两次。

Decoder每层三次。

这就是原始 architecture definition。

---

# 60. Original Encoder Post-LN

第一 sub-layer：

\[
A=MHA(x)
\]

\[
\boxed{
h=
LN_1(
x+
Dropout(A)
)
}
\]

第二：

\[
F=FFN(h)
\]

\[
\boxed{
y=
LN_2(
h+
Dropout(F)
)
}
\]

所以每次 residual update后：

> 立即 normalization。

---

# 61. Original Decoder Post-LN

依次：

\[
\boxed{
h_1
=
LN_1(
x+
SelfAttention(x)
)
}
\]

\[
\boxed{
h_2
=
LN_2(
h_1+
CrossAttention(h_1,M)
)
}
\]

\[
\boxed{
h_3
=
LN_3(
h_2+
FFN(h_2)
)
}
\]

省略 dropout。

---

# 62. 为什么后来出现 Pre-LN？

后续研究发现 LayerNorm placement：

> 会显著影响深 Transformer 的 optimization。

Pre-LN：

\[
\boxed{
y=
x+
F(
LN(x)
)
}
\]

把 normalization移到 sub-layer之前。

---

# 63. Post-LN 和 Pre-LN 再做一次严格比较

### Post-LN

\[
\boxed{
y=LN(x+F(x))
}
\]

### Pre-LN

\[
\boxed{
y=x+F(LN(x))
}
\]

两者绝不等价。

因为：

\[
LN
\]

不是固定 linear function。

---

# 64. Gradient Path 的差异

Post-LN：

设：

\[
z=x+F(x)
\]

则：

\[
\boxed{
\frac{\partial y}{\partial x}
=
J_{LN}(z)
(
I+J_F
)
}
\]

所以 shortcut gradient仍要经过：

\[
J_{LN}
\]

---

Pre-LN：

\[
y=x+F(LN(x))
\]

所以：

\[
\boxed{
\frac{\partial y}{\partial x}
=
I+
J_F
J_{LN}
}
\]

identity term：

\[
I
\]

直接位于最外层。

---

# 65. 为什么这个 Difference 很重要？

Xiong 等 2020 对 LayerNorm placement进行了理论分析。

他们指出原始 Post-LN Transformer在初始化时：

> 靠近 output layer 的参数可能有较大的 expected gradients，

这使 learning-rate warm-up对稳定训练很重要。

他们分析的 Pre-LN结构则具有：

> 更良好的 initialization gradient behavior。

这说明：

\[
\boxed{
\text{LayerNorm不只是“数值标准化位置随便放哪里都一样”}
}
\]

它直接改变 optimization path。

---

# 66. Pre-LN 一定比 Post-LN 更好吗？

不能这么绝对。

Pre-LN在深层 optimization上有很多实践优势，

但最终性能、architecture和训练 recipe之间关系复杂。

不同现代模型还使用：

- RMSNorm；
- Sandwich Norm；
- DeepNorm；
- ScaleNorm；
- various residual scaling。

所以：

> “Pre-LN永远优于Post-LN”

不是可靠的普遍定理。

---

# 67. 但历史事实必须记住

\[
\boxed{
\text{Original Transformer = Post-LN}
}
\]

不是：

\[
Pre\text{-}LN
\]

很多现代代码默认已经不同，

所以读论文时不要拿现代习惯反推 2017 architecture。

---

# 68. LayerNorm 和 RMSNorm 有什么区别？

LayerNorm：

\[
\boxed{
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
}
\]

既：

- center；
- scale。

RMSNorm则主要根据 root-mean-square：

> 做尺度规范，

通常不减 feature mean。

所以两者不是一个东西。

RMSNorm在现代 LLM里很常见，

但原始 Transformer和 ACT canonical code用的是：

\[
\boxed{
LayerNorm}
\]

---

# 69. 为什么先不把 RMSNorm混进主线？

因为我们现在建立 canonical Transformer。

先理解：

\[
\mu,\sigma^2,\gamma,\beta
\]

的 LayerNorm。

之后再单独写：

> **RMSNorm：为什么不减均值也可以？**

这样知识图更清楚。

---

# 70. ACT 中到底有几个 LayerNorm？

Policy Encoder每个 layer：

\[
2
\]

个：

```python
self.norm1
self.norm2
```

Policy Decoder每个 layer：

\[
3
\]

个：

```python
self.norm1
self.norm2
self.norm3
```

另外 Decoder stack还配置一个：

```python
decoder_norm = nn.LayerNorm(d_model)
```

用于 stack output / intermediate handling。

如果开启 Pre-LN，

Encoder stack也会配置 final：

```python
encoder_norm
```

---

# 71. ACT Encoder 的 nn.LayerNorm(512)

官方：

```python
self.norm1 =
    nn.LayerNorm(d_model)

self.norm2 =
    nn.LayerNorm(d_model)
```

其中：

\[
d_{\text{model}}=512
\]

所以：

```python
nn.LayerNorm(512)
```

---

# 72. ACT Tensor 常是 [S,B,E]

PyTorch `MultiheadAttention` 默认 convention通常是：

\[
[S,B,E]
\]

例如 Policy Encoder：

\[
\boxed{
[1202,B,512]
}
\]

这里：

- 1202：sequence；
- B：batch；
- 512：embedding/features。

`nn.LayerNorm(512)` 看：

> 最后一维。

所以：

\[
\boxed{
\text{对每个 }(sequence\ position,batch\ item)
\text{ 的 512 features 单独 normalize}
}
\]

---

# 73. 它绝不会沿 1202 算 Mean

这是一个非常重要的 ACT shape理解。

对于：

\[
X\in\mathbb R^{1202\times B\times512}
\]

LayerNorm不是：

\[
\mu
=
\frac1{1202}
\sum_{s}X_s
\]

而是：

\[
\boxed{
\mu_{s,b}
=
\frac1{512}
\sum_{d=1}^{512}
X_{s,b,d}
}
\]

所以：

> visual token 17不会和 visual token 900一起算 LayerNorm mean。

---

# 74. joint token 和 visual token 也不共享 Statistics

虽然它们在同一个 sequence：

```text
latent
joint
visual₁
...
visual₁₂₀₀
```

但 LayerNorm对每一个 token分别计算：

\[
\mu,\sigma^2
\]

所以：

- latent token一套；
- joint token一套；
- 每个 visual token各一套。

它们只共享：

\[
\gamma,\beta
\]

parameters。

---

# 75. ACT Decoder 同理

Decoder hidden：

\[
[k,B,512]
\]

例如：

\[
k=100
\]

LayerNorm：

\[
\boxed{
\text{100 个 action slots各自独立对 512 features normalize}
}
\]

不会把 100 个 future action slots混在一起求 mean。

---

# 76. 这意味着 LayerNorm 不会直接做 Action Smoothing

因为它不跨：

\[
k
\]

这个时间/slot axis。

它只 normalize每个 slot内部 feature vector。

Action slots之间的 interaction发生在：

> Decoder Self-Attention。

所以：

\[
\boxed{
LayerNorm
\neq
Temporal smoothing
}
\]

---

# 77. CVAE Encoder 也是同样

CVAE sequence例如：

\[
[CLS],qpos,a_0,\ldots,a_{99}
\]

shape内部通常最终整理为：

\[
[102,B,512]
\]

LayerNorm对：

- `[CLS]`；
- qpos；
- 每个 action token；

分别沿 512 features normalize。

所以 `[CLS]`统计量不会和：

\[
a_{37}
\]

混起来。

---

# 78. 为什么不同 Token 可以共享同一 γ、β？

因为所有 token已经被投影到：

\[
512
\]

维统一 hidden space。

模型学习：

> feature dimension \(d\) 在这个 hidden space中应该有什么全局 scale / offset。

而每个 token自己的 activation中心和尺度：

> 由它自己的 \(\mu,\sigma^2\) 处理。

---

# 79. γ/β 会不会破坏 Token Independence？

不会跨 token。

\[
y_{s,b,d}
=
\gamma_d
\hat x_{s,b,d}
+
\beta_d
\]

只用当前：

\[
s,b,d
\]

normalized value和共享 parameter。

不会读取其他：

\[
s'
\]

token的 activation。

---

# 80. ACT 当前默认 Post-LN

官方 `TransformerEncoderLayer.forward(...)`：

```python
if self.normalize_before:
    return self.forward_pre(...)

return self.forward_post(...)
```

而 constructor默认：

```python
normalize_before=False
```

因此 canonical released default：

\[
\boxed{
Post\text{-}LN
}
\]

---

# 81. ACT Encoder Post-LN Code

核心：

```python
src2 =
    self.self_attn(...)[0]

src =
    src +
    self.dropout1(src2)

src =
    self.norm1(src)
```

然后：

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

完全对应：

\[
\boxed{
LN(
x+
Sublayer(x)
)
}
\]

---

# 82. ACT Decoder Post-LN Code

Self-Attention：

\[
tgt
\rightarrow
tgt+self\_attn
\rightarrow
norm1
\]

Cross-Attention：

\[
tgt
\rightarrow
tgt+cross\_attn
\rightarrow
norm2
\]

FFN：

\[
tgt
\rightarrow
tgt+ffn
\rightarrow
norm3
\]

所以一个 Decoder layer对每个 action slot做：

\[
3
\]

次 LayerNorm。

---

# 83. Pre-LN Code 又是什么？

Encoder：

```python
src2 =
    self.norm1(src)

src2 =
    self.self_attn(... value=src2)[0]

src =
    src +
    self.dropout1(src2)
```

FFN同样：

```python
src2 =
    self.norm2(src)

src2 =
    FFN(src2)

src =
    src +
    dropout(src2)
```

所以：

\[
\boxed{
LN
\rightarrow
Sublayer
\rightarrow
Residual Add
}
\]

---

# 84. ACT 默认 LayerNorm ε 是多少？

ACT调用：

```python
nn.LayerNorm(d_model)
```

没有显式设置：

```python
eps
```

PyTorch当前默认：

\[
\boxed{
eps=10^{-5}
}
\]

因此一个 ACT token：

\[
x\in\mathbb R^{512}
\]

实际标准化分母：

\[
\boxed{
\sqrt{
\sigma^2+10^{-5}
}
}
\]

---

# 85. ACT LayerNorm 的 Variance Estimator

PyTorch文档：

> variance采用 biased estimator，等价 `torch.var(input, correction=0)`。

所以对 512 features：

\[
\boxed{
\sigma^2
=
\frac1{512}
\sum_{d=1}^{512}
(x_d-\mu)^2
}
\]

不是：

\[
1/511
\]

---

# 86. 这和训练数据统计量完全不同

ACT还会对 qpos/actions做 dataset normalization。

例如训练 preprocessing可能使用数据集的：

- qpos mean；
- qpos std；
- action mean；
- action std。

那个 normalization作用于：

> 输入/输出物理变量数据。

LayerNorm则作用于：

> 模型内部 hidden representation。

二者不要混。

---

# 87. Dataset Normalization vs LayerNorm

### Dataset Normalization

统计量可能来自：

> 整个训练 dataset。

例如：

\[
q_{norm}
=
\frac{q-\mu_{dataset}}{\sigma_{dataset}}
\]

---

### LayerNorm

每次 forward，

对当前 token：

\[
x_{s,b,:}
\]

自己实时算：

\[
\mu_{s,b},\sigma^2_{s,b}
\]

所以：

\[
\boxed{
\text{data preprocessing normalization}
\neq
\text{neural LayerNorm}
}
\]

---

# 88. 两种标准化为什么可以同时存在？

因为它们稳定的是不同层级。

Dataset normalization：

> 让物理输入输出尺度更合适。

LayerNorm：

> 控制深层网络内部动态 hidden features。

一个发生在：

> model外/入口出口附近。

一个发生在：

> Transformer每层内部。

---

# 89. ImageNet Normalize 又是第三件事

ACT image输入还会做类似：

\[
(image-\mu_{RGB})/\sigma_{RGB}
\]

那是为了：

> 匹配 vision backbone / ImageNet预训练常用输入尺度。

这同样不是 LayerNorm。

所以 ACT里至少要区分：

1. image channel normalization；
2. qpos/action dataset standardization；
3. Transformer LayerNorm。

---

# 90. 为什么这三个都叫 Normalization 很烦？

因为“normalization”在机器学习里是一个宽泛词，

可能表示：

- min-max scaling；
- z-score standardization；
- BatchNorm；
- LayerNorm；
- image channel normalization；
- vector norm normalization。

所以看到：

> normalize

一定问：

\[
\boxed{
\text{对谁？沿哪个轴？统计量从哪里来？什么时候计算？}
}
\]

比背名字更可靠。

---

# 91. LayerNorm 的四个必问问题

以后看到任意 LayerNorm实现，问：

### 1. Normalized shape 是什么？

例如：

\[
512
\]

---

### 2. Statistics 沿哪些 axis 算？

PyTorch：

> normalized_shape对应的最后 D 个 dimensions。

---

### 3. γ/β 是否 learned？

PyTorch默认：

> 是。

---

### 4. Norm 在 Residual 前还是后？

决定：

> Pre-LN / Post-LN optimization structure。

---

# 92. 为什么 LayerNorm 对 Batch Size=1 也能正常工作？

因为 statistics不依赖：

> batch里的其他样本。

即使：

\[
B=1
\]

每个 token仍然有：

\[
512
\]

个 features可以算：

\[
\mu,\sigma^2
\]

所以 LayerNorm天然适合：

- small batch；
- online inference；
- one-sample robot control。

---

# 93. 这对 Robotics 很自然

机器人 inference时通常可能一次处理：

\[
B=1
\]

当前 observation。

如果 normalization依赖：

> “这一批其他机器人 observations”

就会很不方便。

LayerNorm完全不需要这种 batch context。

因此 ACT推理时：

\[
B=1
\]

也没有问题。

---

# 94. LayerNorm 对 Sequence Length 变化敏感吗？

`LayerNorm(512)`不关心：

\[
N
\]

是多少。

因为它只看：

> 最后一维 512。

所以理论上：

\[
N=10
\]

\[
N=100
\]

\[
N=1202
\]

同一个 LayerNorm module都可以处理。

这和 Position-wise FFN一样具有 length-independent parameter sharing。

---

# 95. 为什么 LayerNorm 不需要知道哪个 Token 是 Padding？

因为它对每个 token自己算。

PAD token可以被 normalize，

并不直接影响其他 token的 LayerNorm statistics。

真正要阻止 PAD影响真实 token：

> 在 Attention中使用 padding mask。

所以：

\[
\boxed{
PaddingMask
\neq
LayerNorm
}
\]

---

# 96. 如果 PAD Token经过 LayerNorm变成非零，会有问题吗？

只要 Attention / loss正确 mask，

通常不必要求 PAD representation永远为零。

因为关键是：

> 它不能向真实 tokens传播错误信息。

LayerNorm会处理它自己，

但不跨 token传播。

---

# 97. LayerNorm 会不会改变 Attention Scores？

会，间接取决于 architecture。

Pre-LN中：

\[
LN(x)
\]

直接作为 Attention输入，

因此直接影响：

\[
Q,K,V
\]

Post-LN中：

> 前一个 sub-layer output已经被 LN，

下一 sub-layer自然使用 normalized representation。

所以无论哪种，

LayerNorm都会影响 subsequent Q/K/V geometry。

---

# 98. 为什么 Scale 对 Dot Product Attention 很重要？

Attention score：

\[
q^\top k
\]

会受：

- direction；
- magnitude；

共同影响。

如果 hidden representation尺度在层间剧烈漂移，

Q/K magnitude也可能漂移。

LayerNorm通过控制 token feature scale：

> 帮助后续 projections工作在更可控的 activation regime。

但不要说：

> LayerNorm让所有 Q/K norm完全相同。

Linear projections和 learned gamma会改变它们。

---

# 99. LayerNorm 与 Attention 的 \(\sqrt{d_k}\) Scaling 是同一件事吗？

不是。

\[
1/\sqrt{d_k}
\]

是 Scaled Dot-Product Attention中对：

\[
q^\top k
\]

logit magnitude的维度缩放。

LayerNorm：

> normalize hidden features。

二者都与数值尺度有关，

但作用位置、数学对象完全不同。

---

# 100. LayerNorm 会不会让所有 Token 的 Norm 一样？

在纯标准化阶段，

如果：

\[
\epsilon=0
\]

且 variance按：

\[
1/D
\]

计算，

则：

\[
\frac1D
\sum_d
\hat x_d^2
=
1
\]

且 mean 0，

所以：

\[
\|\hat x\|_2^2=D
\]

也就是 norm：

\[
\sqrt D
\]

相同。

---

# 101. 但真实 LayerNorm 最终不保证相同 Norm

因为：

1. \(\epsilon>0\)；
2. 每维有不同：
   \[
   \gamma_d
   \]
3. 加：
   \[
   \beta_d
   \]

所以最终：

\[
y
\]

的 Euclidean norm可以随 token变化。

因此不要说：

> LayerNorm强制所有 token长度一样。

---

# 102. LayerNorm 的 Standardized Vector 有什么约束？

忽略 \(\epsilon\)：

\[
\sum_d\hat x_d=0
\]

并且：

\[
\frac1D
\sum_d\hat x_d^2=1
\]

所以：

\[
\sum_d\hat x_d^2=D
\]

即 standardized vector位于：

- 与 all-ones vector正交的 hyperplane；
- 同时位于固定-radius sphere；

的交集上。

这是一个有趣的几何解释。

---

# 103. 为什么 Mean=0 意味着和 Ones Vector 正交？

定义：

\[
\mathbf1=[1,\ldots,1]
\]

则：

\[
\hat x^\top\mathbf1
=
\sum_d\hat x_d
\]

mean 0意味着：

\[
\sum_d\hat x_d=0
\]

所以：

\[
\boxed{
\hat x\perp\mathbf1
}
\]

LayerNorm的 centering实际上去掉了：

> all-features-common direction。

---

# 104. 除标准差的几何作用

center后 vector：

\[
x-\mu\mathbf1
\]

除：

\[
\sigma
\]

把它缩放到一个固定 RMS尺度。

所以纯 LayerNorm标准化部分可以直觉化：

```text
先投影掉共同 offset direction
↓
再调整整体 feature scale
```

然后：

\[
\gamma,\beta
\]

再做 learnable coordinate-wise affine transform。

---

# 105. 这种几何解释为什么有用？

它帮助我们理解：

LayerNorm不是：

> “把每一个 feature单独变成标准正态。”

而是对整个：

\[
D
\]

维 vector做一个耦合变换。

先用整个 vector的 center和scale重新定坐标。

---

# 106. LayerNorm 会把 Feature Correlation 消掉吗？

不会。

它只控制：

- 一阶中心；
- 整体二阶尺度。

它不会让 covariance matrix变成 identity。

也不会保证 features独立。

所以：

\[
\boxed{
LayerNorm
\neq
Whitening
}
\]

---

# 107. Whitening 是什么？

Whitening通常希望：

\[
Cov(z)=I
\]

除了每维 variance外，

还消除 cross-feature correlations。

LayerNorm没有计算完整：

\[
D\times D
\]

covariance matrix。

所以它远比 whitening简单。

---

# 108. LayerNorm 会把每个 Feature 单独 Variance 变 1 吗？

不是这个意思。

对单个 token，

它沿 512 features计算一个共享：

\[
\sigma^2
\]

然后所有 features一起除同一个：

\[
\sqrt{\sigma^2+\epsilon}
\]

不是为每一个 feature：

\[
d
\]

单独算一个自己的 variance。

这一点很重要。

---

# 109. 那 γ 为什么是每维一个？

Normalization statistics共享：

\[
\mu,\sigma
\]

但最终模型允许：

> 不同 hidden dimensions拥有不同 learned scale和shift。

所以：

\[
\gamma_d,\beta_d
\]

是 per-feature。

这是：

> normalization group statistics共享 + affine parameters per coordinate

的组合。

---

# 110. 一个 3-D γ/β 例子

假设 normalize后：

\[
\hat x=
[-1,0,1]
\]

learned：

\[
\gamma=
[2,0.5,3]
\]

\[
\beta=
[1,-1,0]
\]

则：

\[
y=
[
2(-1)+1,
0.5(0)-1,
3(1)+0
]
\]

\[
=
[-1,-1,3]
\]

显然：

> 最终 y 的 mean不再 0，

variance也不再 1。

所以：

\[
\boxed{
\text{LayerNorm output}
\neq
\text{necessarily zero-mean/unit-variance}
}
\]

真正 standardized的是：

> affine前的 \(\hat x\)。

---

# 111. 那为什么名字还叫 Normalization Layer？

因为它的核心内部操作确实执行 normalization，

然后附加 trainable affine transform。

BatchNorm也有类似：

\[
\gamma,\beta
\]

设计。

这让网络既能获得 normalized optimization properties，

又保留可学习的表示尺度。

---

# 112. γ=0 会怎样？

如果某个 dimension：

\[
\gamma_d=0
\]

则：

\[
y_d=\beta_d
\]

该 dimension对当前 normalized input不再敏感。

所以 learned gain甚至可以：

> 关闭某个 normalized feature方向。

---

# 113. γ 可以是负数吗？

可以。

它是 unconstrained learned parameter。

如果：

\[
\gamma_d<0
\]

该 feature方向会翻转。

所以 \(\gamma\) 不只是“放大倍数”。

它可以：

- 放大；
- 缩小；
- 关闭；
- 翻转。

---

# 114. β 的作用是什么？

\[
\beta_d
\]

提供 learned offset。

如果没有 beta，

normalized representation始终围绕某种 zero-centered基准。

有 beta后，

网络可以为每个 hidden dimension设定：

> task-preferred baseline activation。

---

# 115. LayerNorm 的参数量为什么这么少？

对于：

\[
D=512
\]

只有：

\[
1024
\]

trainable parameters。

相比 ACT FFN约：

\[
3.28M
\]

参数，

LayerNorm参数量极小。

但它对训练 dynamics影响可以非常大。

所以：

\[
\boxed{
\text{parameter count}
\neq
\text{architectural importance}
}
\]

---

# 116. LayerNorm 的计算量也不大

每个 token：

- 求 mean；
- 求 variance；
- normalize；
- affine。

复杂度大约：

\[
O(D)
\]

比：

- Attention矩阵；
- 大 FFN矩阵乘；

轻得多。

但现代大模型里 normalization仍然是非常关键的 kernel。

---

# 117. 为什么 LayerNorm 在 Mixed Precision 时也要小心？

mean/variance涉及 reduction。

低精度可能导致：

> numerical accuracy问题。

现代框架/kernel会做各种稳定处理。

但 canonical LayerNorm数学公式不依赖特定 precision。

这里不展开硬件实现细节。

---

# 118. 常见误解一：LayerNorm 把 Hidden State 变成标准正态分布

**错误。**

它只标准化某些 moments。

不规定 distribution shape。

---

# 119. 常见误解二：Mean=0、Variance=1 就等于 N(0,1)

**错误。**

很多非 Gaussian分布也有：

\[
mean=0,\ variance=1
\]

---

# 120. 常见误解三：LayerNorm 对整个 Batch 求 Mean

**错误。**

`LayerNorm(512)`对每个 input position自己的 512 features求 statistics。

---

# 121. 常见误解四：ACT 的 1202 Tokens 一起算一个 Mean

**错误。**

每个：

\[
[s,b,:]
\]

512-D vector单独算。

---

# 122. 常见误解五：LayerNorm 完全逐 Feature 独立

**错误。**

features通过共享的：

\[
\mu,\sigma^2
\]

相互耦合。

---

# 123. 常见误解六：LayerNorm 会让 Token 互相交流

**错误。**

它不跨 token。

token communication主要由 Attention负责。

---

# 124. 常见误解七：除标准差只是为了防止除 0

**错误。**

除标准差的核心作用是：

> 统一尺度。

\(\epsilon\) 才主要负责数值稳定。

---

# 125. 常见误解八：ε 越大越好，因为更稳定

**错误。**

过大的：

\[
\epsilon
\]

会显著改变 normalization尺度，

使：

\[
Var(\hat x)
=
\sigma^2/(\sigma^2+\epsilon)
\]

偏离 1。

它是稳定性与准确标准化之间的数值设计参数。

---

# 126. 常见误解九：Normalize 后 Variance 永远精确为 1

真实实现有：

\[
\epsilon
\]

所以严格不是。

---

# 127. 常见误解十：PyTorch LayerNorm 用 D-1 无偏方差

**错误。**

当前 PyTorch文档明确：

\[
\boxed{
correction=0
}
\]

即 denominator为：

\[
D
\]

---

# 128. 常见误解十一：γ 和 β 把 normalization完全取消了

**错误。**

它们是跨样本共享的 learned parameters，

不能恢复每个 sample原本被移除的独立 mean/scale。

---

# 129. 常见误解十二：γ/β 是当前 Token 动态算出来的

**错误。**

它们是 trainable model parameters。

动态计算的是：

\[
\mu,\sigma^2
\]

---

# 130. 常见误解十三：LayerNorm 推理时用 Running Mean

**错误。**

它仍然根据当前 input实时计算 statistics。

---

# 131. 常见误解十四：`model.eval()` 会冻结 LayerNorm 到训练统计量

**错误。**

标准 LayerNorm没有 BatchNorm式 running stats。

---

# 132. 常见误解十五：LayerNorm 和 BatchNorm 只是名字不同

**错误。**

统计轴和 train/eval behavior本质不同。

---

# 133. 常见误解十六：LayerNorm 和 Dataset Z-Score 是同一套 Statistics

**错误。**

Dataset standardization用训练数据统计量。

LayerNorm每个 hidden token实时算 statistics。

---

# 134. 常见误解十七：ImageNet Normalize 就是 LayerNorm

**错误。**

RGB channel preprocessing和 Transformer hidden-state normalization不是同一层级。

---

# 135. 常见误解十八：LayerNorm 会 Whitening

**错误。**

它不消除完整 feature covariance。

---

# 136. 常见误解十九：Post-LN 和 Pre-LN 只是写法不同

**错误。**

它们的 forward function和 gradient Jacobian都不同。

---

# 137. 常见误解二十：Original Transformer 是 Pre-LN

**错误。**

原始 2017：

\[
\boxed{
Post\text{-}LN
}
\]

---

# 138. 常见误解二十一：ACT 默认是 Pre-LN

当前官方默认：

\[
normalize\_before=False
\]

所以：

\[
\boxed{
Post\text{-}LN
}
\]

---

# 139. 常见误解二十二：LayerNorm 的作用就是防梯度爆炸

**过度简化。**

它改变 hidden-state statistics和optimization geometry，

作用不能缩成一个单一现象。

---

# 140. 常见误解二十三：LayerNorm 是可逆的

纯 normalization会移除 sample-specific shift / scale信息，

通常不是一一映射。

---

# 141. 常见误解二十四：LayerNorm 会让不同 Modality 变成同一种语义

**错误。**

它统一某些统计尺度，

不抹去所有 content/semantic differences。

visual、joint、latent tokens仍有不同 representations。

---

# 142. 常见误解二十五：LayerNorm 能代替 Residual

**错误。**

Residual和Normalization解决不同结构问题。

Transformer把它们组合使用。

---

# 143. LayerNorm vs BatchNorm 总结表

| 问题 | LayerNorm | BatchNorm |
|---|---|---|
| Statistics 主要来自哪里？ | 单个 sample/token 内 features | mini-batch 中同 feature/channel |
| 依赖 batch size？ | 基本不依赖 | 会依赖训练 batch statistics |
| Train / Eval statistics | 当前 input，基本同样计算 | 典型实现 train 用 batch，eval 用 running stats |
| Transformer 常用？ | 是 | canonical Transformer 不使用 |
| 可变 sequence / B=1 | 很自然 | 通常更麻烦 |
| Learnable scale/bias | 有 | 有 |
| 变成 Gaussian？ | 否 | 否 |

---

# 144. Transformer 中一张最重要的图

```text
Token x ∈ R^512
      │
      ├─────────────── Residual path ───────┐
      │                                     │
      ▼                                     │
Sublayer(x)                                 │
      │                                     │
      ▼                                     │
   Dropout                                  │
      │                                     │
      └──────────────────────→  Add  ←──────┘
                                  │
                                  ▼
                           LayerNorm(512)
                                  │
                        ┌─────────┴─────────┐
                        │                   │
                    compute μ          compute σ²
                 across 512 dims      across 512 dims
                        │                   │
                        └─────────┬─────────┘
                                  ▼
                         (x-μ)/sqrt(σ²+ε)
                                  │
                                  ▼
                              γ ⊙ · + β
                                  │
                                  ▼
                             next hidden
```

这是原始 Post-LN Transformer。

---

# 145. ACT Policy Encoder 的真实 Axis

输入内部 convention：

\[
\boxed{
[1202,B,512]
}
\]

于是：

```python
nn.LayerNorm(512)
```

等价于对每一个：

```text
token_position s
batch_item b
```

取：

```text
X[s, b, :]
```

这一个 512-D vector，

单独执行：

\[
\mu_{s,b}
\]

\[
\sigma^2_{s,b}
\]

---

# 146. ACT Decoder 的真实 Axis

\[
\boxed{
[k,B,512]
}
\]

例如：

\[
[100,B,512]
\]

每个 action slot：

\[
X[i,b,:]
\]

自己 normalize。

所以：

\[
\boxed{
\text{LayerNorm不会把未来 100 个动作 Slot 平均到一起}
}
\]

---

# 147. 一个 ACT Token 的完整公式

假设某个 Policy Encoder token在某一层 residual addition之后：

\[
r\in\mathbb R^{512}
\]

计算：

\[
\mu
=
\frac1{512}
\sum_{d=1}^{512}r_d
\]

\[
\sigma^2
=
\frac1{512}
\sum_{d=1}^{512}
(r_d-\mu)^2
\]

然后：

\[
\hat r_d
=
\frac{
r_d-\mu
}{
\sqrt{
\sigma^2+10^{-5}
}
}
\]

最后：

\[
\boxed{
y_d=
\gamma_d\hat r_d+\beta_d
}
\]

其中：

\[
\gamma,\beta\in\mathbb R^{512}
\]

这就是 `nn.LayerNorm(512)` 对这个 token真正做的事。

---

# 148. 为什么它适合 ACT 的多模态 Hidden Space？

ACT把：

- image；
- joint；
- latent；

都投影到：

\[
512
\]

维 Transformer space。

不同 modality产生的 hidden activations可能有不同数值模式。

LayerNorm提供统一的 per-token normalization规则，

让后续 Transformer layers处理更受控的 hidden scale。

但它不会：

> 自动让不同 modality分布完全一致。

那不是 LayerNorm的保证。

---

# 149. 为什么 joint token 不会被 image tokens 的大数值“拖着一起归一化”？

因为 statistics不跨 token。

joint token自己：

\[
512
\]

维算自己的 mean/std。

visual token自己算自己的。

所以一个 visual token magnitude很大：

> 不会直接改变 joint token的 LayerNorm statistics。

这正是 LayerNorm和跨-token normalization不同的地方。

---

# 150. 为什么 LayerNorm 后 Attention 仍然能区分 Token 强弱？

因为：

1. normalized pattern仍然不同；
2. learned \(\gamma,\beta\)重新标定 features；
3. Q/K/V projections重新构造 magnitudes；
4. positional information存在；
5. residual/context深度不同。

所以消除一个 token自己的 common shift/scale：

> 不等于让所有 tokens相同。

---

# 151. 一个重要的“尺度信息”细节

LayerNorm会弱化：

> “这个 token所有 hidden features整体都放大了 100 倍”

这种信息。

因为正 scale大体会被 normalization消掉。

如果模型真的需要表示某种 confidence magnitude，

它必须通过：

> normalization后仍可保留的方向/pattern，或其他 architecture paths

来编码。

这也是 normalization改变 representation geometry的真实含义。

---

# 152. 为什么现代研究会对 Normalization 设计很敏感？

因为 normalization不是一个纯辅助数值 trick。

它决定：

- 什么信息被 scale-normalized；
- residual path怎样传播；
- gradient怎样通过；
- hidden representation处在哪种几何空间。

因此改变：

- LayerNorm；
- RMSNorm；
- Pre/Post placement；

可能对训练产生非常大影响。

---

# 153. LayerNorm 会不会改变方向？

会。

减均值：

\[
x-\mu\mathbf1
\]

本身就可能改变原 vector相对于原点的方向。

除 scalar standard deviation不会进一步改变 centered vector方向，

但 gamma如果 per-coordinate不同：

\[
\gamma\odot\hat x
\]

又会改变方向。

所以 LayerNorm不是：

> 只改变 vector length。

---

# 154. 它和 L2 Normalization 完全不同

L2 normalization：

\[
\boxed{
\frac{x}{\|x\|_2}
}
\]

主要把 vector norm变成 1。

不会先减 feature mean。

LayerNorm：

\[
\boxed{
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
}
\]

先 center，

再按 RMS-like feature spread scale。

所以：

\[
\boxed{
LayerNorm
\neq
L2Norm
}
\]

---

# 155. 为什么两者都可能让 Norm 受控，却不是一个东西？

L2Norm保留：

> 从原点看的 direction。

LayerNorm先去掉：

> all-ones/common-offset component。

所以它改变的几何结构不同。

---

# 156. LayerNorm 和 Softmax 也不要混

Softmax：

\[
p_i=
\frac{e^{z_i}}{\sum_je^{z_j}}
\]

输出：

- positive；
- sum=1。

LayerNorm：

> 输出可正可负，
> 不要求 sum=1。

即使 mean在 affine前约为0，

它也完全不是 probability normalization。

---

# 157. “Normalization”这个词必须问目的

Softmax normalization：

> 把 scores变成 probability-like weights。

LayerNorm：

> 规范 hidden feature statistics。

Image normalization：

> 规范 input channels。

Vector normalization：

> 规范 norm。

所以不要因为都叫：

> normalization

就把数学目标混为一谈。

---

# 158. LayerNorm 与我们之前的标准正态分布知识怎么连接？

最核心的桥：

\[
z=\frac{x-\mu}{\sigma}
\]

是一种：

> 标准化坐标。

它表达：

> 当前值相对自身 group center偏离多少个标准差。

但只有当原随机变量本来 Gaussian时，

这个 affine transformation才把 distribution变成：

\[
\mathcal N(0,1)
\]

LayerNorm没有这个 Gaussian前提。

所以：

\[
\boxed{
\text{LayerNorm借用了标准化数学}
\neq
\text{LayerNorm假设 Hidden State是正态分布}
}
\]

---

# 159. 如果只记住一件事

请记：

\[
\boxed{
\operatorname{LayerNorm}(x)
=
\gamma\odot
\frac{
x-\operatorname{mean}(x)
}{
\sqrt{
\operatorname{var}(x)+\epsilon
}
}
+\beta
}
\]

在 Transformer `LayerNorm(D)` 中，

这个：

\[
mean,\ variance
\]

通常是：

> **每一个 token自己沿 hidden feature dimension \(D\) 计算。**

它不跨：

- batch；
- sequence positions。

---

# 160. 第二件必须记住的事

\[
\boxed{
\text{Zero mean + unit variance}
\neq
\text{standard normal distribution}
}
\]

LayerNorm不是 Gaussianizer。

---

# 161. 第三件必须记住的事

\[
\boxed{
\gamma,\beta
}
\]

意味着最终 LayerNorm output：

> 并不要求 mean=0、variance=1。

标准化只是中间步骤。

---

# 162. 第四件必须记住的事

Original Transformer：

\[
\boxed{
Post\text{-}LN:
\quad
LN(
x+F(x)
)
}
\]

现代许多模型：

\[
\boxed{
Pre\text{-}LN:
\quad
x+F(LN(x))
}
\]

这两个 gradient structure不同。

---

# 163. 第五件必须记住的事：ACT

对于：

\[
[1202,B,512]
\]

或：

\[
[k,B,512]
\]

ACT的：

```python
nn.LayerNorm(512)
```

永远看最后的：

\[
512
\]

hidden features。

因此：

\[
\boxed{
\text{每个 observation token / action slot 独立 normalize}
}
\]

而不是把所有 tokens混在一起。

---

# 164. 一句话真正理解 Layer Normalization

> **Layer Normalization 是一种针对单个 hidden representation 的动态标准化：它用这个 representation 自己的一组 feature values 计算均值和方差，去掉共同偏移并规范整体 feature scale，再用共享的可学习 \(\gamma,\beta\) 恢复模型所需的逐维尺度与偏移；因此它不依赖 mini-batch，也不会把 representation 强迫成 Gaussian，而是在保持 token 独立的同时重塑每个 token 内部的 feature geometry 与优化尺度。**

---

# 165. 一句话真正理解 Transformer 中的 LayerNorm

> **Transformer 中 LayerNorm 的作用不是“把每一层变成标准正态分布”，而是让每个 token 的 hidden features 在进入后续 Attention / FFN computation 时处于受控、可学习的统计尺度；而它放在 residual 前还是后，会直接改变 residual stream与 gradient的传播方式，因此 Post-LN / Pre-LN 是真正的 architecture区别，而不是代码风格差异。**

---

# 166. 一句话连接 ACT

> **ACT 对每一个 512-D visual、joint、latent memory token以及每一个 512-D action-query hidden slot分别执行 LayerNorm；`nn.LayerNorm(512)` 不会沿 1202-token observation sequence或 \(k\)-step action sequence求统计量，而只对每个 token自身的 512 个 hidden features实时计算 mean/variance，因此它既适合多模态 Transformer，也适合 batch size 1 的机器人在线推理。**

---

# 167. 下一篇：Dropout

现在 Transformer Layer中：

- Attention；
- FFN；
- Residual；
- LayerNorm；

都已经拆开。

还剩一个反复出现在官方代码里的：

```python
Dropout
```

下一篇：

> **[Dropout：训练时随机删神经元，为什么推理时反而全部使用？](./dropout.md)**

会详细解释：

- Bernoulli mask到底是什么；
- 为什么训练时随机设 0 是 regularization；
- 为什么 PyTorch 使用 inverted dropout；
- 为什么训练时保留值要除以 \(1-p\)；
- 为什么这样可以让 train / eval 的 expectation对齐；
- Dropout为什么不是“永久删除 neuron”；
- Attention dropout、FFN dropout、Residual dropout有什么区别；
- `model.train()` / `model.eval()`到底改变什么；
- ACT Table III 的 dropout 0.1具体进入哪些 Transformer模块；
- 为什么 ACT inference中的 Dropout会关闭，而 LayerNorm仍继续使用当前输入统计量。

---

## Primary Source：Layer Normalization

Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton.

**Layer Normalization.**  
2016.

- arXiv: https://arxiv.org/abs/1607.06450
- DOI: https://doi.org/10.48550/arXiv.1607.06450

论文核心事实：

- Batch Normalization使用 mini-batch中 activations 的统计量；
- Layer Normalization改为对单个 training case中一层的 hidden units计算 mean / variance；
- 每个 hidden unit具有 adaptive gain 与 bias；
- LayerNorm不依赖 mini-batch size；
- train 与 test使用同样的 normalization computation；
- 论文尤其强调它很自然地适用于 recurrent / sequence models。

---

## Background：Batch Normalization

Sergey Ioffe, Christian Szegedy.

**Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.**  
ICML 2015.

- arXiv: https://arxiv.org/abs/1502.03167
- DOI: https://doi.org/10.48550/arXiv.1502.03167

BatchNorm的核心区别是：

> normalization statistics依赖 mini-batch中的样本集合。

这与 LayerNorm：

> 单个 sample内部计算 statistics

形成根本对比。

---

## Transformer Primary Source

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,  
Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin.

**Attention Is All You Need.**  
NeurIPS 2017.

- arXiv: https://arxiv.org/abs/1706.03762
- DOI: https://doi.org/10.48550/arXiv.1706.03762

Section 3.1 明确规定：

\[
\boxed{
LayerNorm(
x+
Sublayer(x)
)
}
\]

即：

> original Transformer uses Post-LN。

Encoder每层：

- Self-Attention Add & Norm；
- FFN Add & Norm。

Decoder每层：

- Self-Attention Add & Norm；
- Cross-Attention Add & Norm；
- FFN Add & Norm。

---

## Pre-LN / Post-LN Background

Ruibin Xiong et al.

**On Layer Normalization in the Transformer Architecture.**  
ICML 2020.

- arXiv: https://arxiv.org/abs/2002.04745

该工作分析：

### Post-LN

\[
\boxed{
LN(
x+F(x)
)
}
\]

与：

### Pre-LN

\[
\boxed{
x+F(
LN(x)
)
}
\]

在 initialization gradient behavior上的差异。

论文指出原始 Post-LN Transformer靠近 output的梯度在初始化时可能较大，而 Pre-LN具有更良好的初始化梯度行为，从而解释 learning-rate warm-up与 LayerNorm placement之间的重要关系。

---

## PyTorch Implementation Reference

PyTorch `torch.nn.LayerNorm`:

https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

当前 API：

```python
nn.LayerNorm(
    normalized_shape,
    eps=1e-5,
    elementwise_affine=True
)
```

PyTorch文档明确：

- mean/std在 `normalized_shape` 对应的最后若干 dimensions上计算；
- `LayerNorm(512)` 因而 normalize最后一个 512-D feature dimension；
- variance使用 biased estimator：
  \[
  correction=0
  \]
- 默认：
  \[
  \epsilon=10^{-5}
  \]
- \(\gamma,\beta\) 默认可学习；
- train / eval均使用当前 input statistics。

---

## ACT Official Implementation

Official repository:

https://github.com/tonyzhaozh/act

ACT `detr/models/transformer.py` 继承自 DETR-style Transformer。

Encoder layer：

```python
self.norm1 =
    nn.LayerNorm(d_model)

self.norm2 =
    nn.LayerNorm(d_model)
```

Decoder layer：

```python
self.norm1 =
    nn.LayerNorm(d_model)

self.norm2 =
    nn.LayerNorm(d_model)

self.norm3 =
    nn.LayerNorm(d_model)
```

在 ACT canonical hidden dimension：

\[
d_{\text{model}}=512
\]

因此等价：

```python
nn.LayerNorm(512)
```

---

### ACT Post-LN Default

官方实现：

```python
if self.normalize_before:
    return self.forward_pre(...)

return self.forward_post(...)
```

而默认：

```python
normalize_before=False
```

所以 released default是：

\[
\boxed{
Post\text{-}LN
}
\]

Post-LN Encoder：

```python
src =
    src +
    self.dropout1(src2)

src =
    self.norm1(src)
```

即：

\[
\boxed{
LN(
x+
Sublayer(x)
)
}
\]

---

### ACT Shape Meaning

当 Policy Encoder representation为：

\[
[1202,B,512]
\]

`LayerNorm(512)` 对每一个：

\[
X[s,b,:]
\]

独立计算：

\[
\mu_{s,b}
=
\frac1{512}
\sum_dX_{s,b,d}
\]

与：

\[
\sigma^2_{s,b}
=
\frac1{512}
\sum_d
(X_{s,b,d}-\mu_{s,b})^2
\]

Decoder：

\[
[k,B,512]
\]

同理。

所以：

\[
\boxed{
\text{LayerNorm never averages ACT's 1202 observation tokens or }k\text{ action slots together}
}
\]

---

## 本文知识连接

### 数学前置

- Mean
- Variance
- Standard Deviation
- Standardization
- [Normal Distribution](../mathematics/normal-distribution.md)
- Standard Normal Distribution
- Jacobian
- Kronecker Delta

### Deep Learning

- Batch Normalization
- RMSNorm
- [Residual Connection](./residual-connection.md)
- [Dropout](./dropout.md)

### Transformer

- [Transformer Encoder](./transformer-encoder.md)
- [Transformer Decoder](./transformer-decoder.md)
- [Feed-Forward Network](./feed-forward-network.md)
- Pre-LN vs Post-LN

### Robot Learning

- [ACT Architecture](../robot-learning/act/architecture.md)
- [ACT Training](../robot-learning/act/training.md)
- [ACT Complete Data Flow](../robot-learning/act/complete-data-flow.md)

### 下一步

- [Dropout](./dropout.md)
