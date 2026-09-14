---
title: "Softmax：为什么指数归一化能把分数变成 Attention Weight？"
description: "从 raw scores 到 normalized weights，严格理解 Softmax 的指数、归一化、平移不变性、相对差值、temperature、梯度、饱和与数值稳定，并连接 Transformer Attention。"
status: reviewed
pageType: concept
canonical: /deep-learning/softmax
updated: "2026-09-15"
---

# Softmax：为什么指数归一化能把分数变成 Attention Weight？

在 [Dot Product](./dot-product.md) 中，我们已经知道 Transformer 会先计算：

$$
QK^\top
$$

得到一组：

> Query–Key compatibility scores。

例如当前 Query 对三个 Keys 的分数是：

$$
[2,\;1,\;0]
$$

但这里立刻出现一个问题：

> **为什么不能直接拿 $[2,1,0]$ 当 Attention Weight？**

如果直接加权：

$$
2v_1+1v_2+0v_3
$$

似乎也能工作。

那为什么 Transformer 一定要再做：

$$
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)
$$

把它变成：

$$
[0.665,\;0.245,\;0.090]
$$

这样的权重？

进一步：

> 为什么 Softmax 要用指数函数？

为什么是：

$$
e^{z_i}
$$

而不是：

$$
z_i^2
$$

或者：

$$
|z_i|
$$

？

为什么统一加一个常数：

$$
[2,1,0]
\rightarrow
[102,101,100]
$$

Softmax 完全不变？

为什么把所有 logits 乘大：

$$
[2,1,0]
\rightarrow
[20,10,0]
$$

结果却会变得极度尖锐？

为什么 Transformer 原论文又特别担心：

> dot product 太大把 Softmax 推进“极小梯度区域”？

这一篇把 Softmax 从头拆开。

---

## 1. 先明确：Softmax 的输入不是概率

假设模型输出：

$$
z=
[z_1,z_2,\ldots,z_n]
$$

这些：

$$
z_i
$$

通常叫：

> **logits**

或者：

> raw scores。

它们可以是任意实数：

$$
[-4.2,\;0.7,\;13.5]
$$

并不要求：

$$
z_i\ge0
$$

也不要求：

$$
\sum_i z_i=1
$$

所以：

> logits 本身不是概率，也不是 normalized weights。

Softmax 的任务就是：

$$
\boxed{
\mathbb R^n
\rightarrow
\text{一组非负且和为 1 的权重}
}
$$

---

## 2. Softmax 的定义

对于：

$$
z\in\mathbb R^n
$$

Softmax 第 $i$ 个输出：

$$
\boxed{
p_i
=
\operatorname{softmax}(z)_i
=
\frac{
e^{z_i}
}{
\sum_{j=1}^{n}e^{z_j}
}
}
$$

立刻有：

$$
p_i>0
$$

以及：

$$
\boxed{
\sum_i p_i=1
}
$$

因此：

$$
p
$$

具有 probability-vector 的数学形式。

在分类中，它常被建模为类别概率。

在 Attention 中，更准确地说通常是：

> **normalized attention weights。**

---

## 3. Softmax 做的其实是两步

不要把整个公式看成一团。

先：

$$
\boxed{
u_i=e^{z_i}
}
$$

把任意实数 score 转成：

> 正数。

然后：

$$
\boxed{
p_i=
\frac{u_i}{\sum_j u_j}
}
$$

做归一化。

所以：

```text
raw logits
↓
exponential
↓
positive scores
↓
divide by total
↓
normalized weights
```

这就是 Softmax 的基本结构。

---

## 4. 为什么不能直接用 z_i / Σz_j？

假设：

$$
z=[2,1,0]
$$

直接除总和：

$$
\frac{z}{\sum z}
=
\left[
\frac23,
\frac13,
0
\right]
$$

看起来似乎也可以。

但换成：

$$
z=[2,1,-1]
$$

总和：

$$
2
$$

直接归一化得到：

$$
[1,\;0.5,\;-0.5]
$$

出现负 weight。

如果：

$$
z=[2,-2,0]
$$

总和甚至是：

$$
0
$$

无法除。

所以 raw-score normalization：

$$
z_i/\sum_jz_j
$$

不能稳定把任意：

$$
\mathbb R^n
$$

映射到 positive simplex。

---

## 5. 为什么指数函数很适合？

指数：

$$
e^x
$$

有几个非常适合 Softmax 的性质。

---

### 性质一：永远为正

对于任意：

$$
x\in\mathbb R
$$

都有：

$$
e^x>0
$$

所以无论 logit 是：

$$
-100
$$

还是：

$$
100
$$

指数后都可以作为正权重基础。

---

### 性质二：单调递增

如果：

$$
z_i>z_j
$$

那么：

$$
e^{z_i}>e^{z_j}
$$

所以 Softmax 不会改变排序。

logit 最大的项：

> Softmax 后仍然最大。

---

### 性质三：差值变成比值

这是最关键的一条。

Softmax 两项的比值：

$$
\frac{p_i}{p_j}
=
\frac{
e^{z_i}
}{
e^{z_j}
}
$$

分母 normalization constant 抵消：

$$
\boxed{
\frac{p_i}{p_j}
=
e^{z_i-z_j}
}
$$

所以：

> 两个 weight 的相对比例，只由两个 logits 的差值决定。

这使 logits 很自然地可以被理解为：

> **log-relative weight / log-odds-like scores。**

---

## 6. 一个例子：差 1 分意味着什么？

假设：

$$
z_1-z_2=1
$$

那么：

$$
\frac{p_1}{p_2}
=
e^1
\approx2.718
$$

也就是说：

> logit 高 1，并不是 weight 高 1。

而是：

> 未归一化 weight 大约是另一个的 2.718 倍。

如果差：

$$
2
$$

则：

$$
e^2\approx7.389
$$

所以指数会把：

> **加法 score difference**

变成：

> **乘法 weight ratio。**

这是 Softmax 的核心结构之一。

---

## 7. 为什么说 Softmax 只关心“相对差值”？

考虑给所有 logits 加同一个常数：

$$
z_i'=z_i+c
$$

Softmax：

$$
p_i'
=
\frac{
e^{z_i+c}
}{
\sum_j e^{z_j+c}
}
$$

利用：

$$
e^{z_i+c}
=
e^c e^{z_i}
$$

得到：

$$
p_i'
=
\frac{
e^c e^{z_i}
}{
e^c\sum_j e^{z_j}
}
$$

共同因子：

$$
e^c
$$

约掉：

$$
\boxed{
p_i'=p_i
}
$$

所以：

$$
\boxed{
\operatorname{softmax}(z+c\mathbf1)
=
\operatorname{softmax}(z)
}
$$

---

## 8. 这叫 Shift Invariance

例如：

$$
[2,1,0]
$$

和：

$$
[102,101,100]
$$

Softmax 完全相同。

因为两者之间的差值关系：

$$
2-1=1
$$

$$
1-0=1
$$

没有改变。

所以 Softmax 不在乎：

> 整组 logits 的绝对零点在哪里。

它在乎：

> **彼此之间差多少。**

---

## 9. 这为什么很适合 Attention？

Attention 中：

$$
s_j
=
\frac{
q^\top k_j
}{
\sqrt{d_k}
}
$$

真正重要的是：

> 对当前 query 而言，哪个 key 比哪个 key 更匹配。

如果所有 keys 的 scores 一起多：

$$
100
$$

匹配排序和相对差值都没变。

我们当然不希望 Attention 突然产生不同结果。

Softmax 的 shift invariance 正好符合：

> **相对竞争**

这一需求。

---

## 10. Softmax 的输出为什么和为 1？

定义：

$$
p_i=
\frac{
e^{z_i}
}{
\sum_j e^{z_j}
}
$$

所以：

$$
\sum_i p_i
=
\sum_i
\frac{
e^{z_i}
}{
\sum_j e^{z_j}
}
$$

分母相同：

$$
=
\frac{
\sum_i e^{z_i}
}{
\sum_j e^{z_j}
}
=
1
$$

于是：

$$
p
$$

落在 probability simplex：

$$
\boxed{
p_i>0,\qquad
\sum_i p_i=1
}
$$

---

## 11. 什么是 Probability Simplex？

对于 3 个类别：

$$
p_1,p_2,p_3
$$

满足：

$$
p_i\ge0
$$

和：

$$
p_1+p_2+p_3=1
$$

所有合法向量形成一个二维三角形区域。

更一般地，

Softmax 把：

$$
\mathbb R^n
$$

映射到：

$$
(n-1)
$$

维 probability simplex 的内部。

为什么只有：

$$
n-1
$$

个自由度？

因为最后一个概率由：

$$
p_n=1-\sum_{i=1}^{n-1}p_i
$$

决定。

---

## 12. Softmax 不是“一般归一化”的唯一方法

必须严格说明：

> **Softmax 不是唯一能把 scores 变成非负且和为 1 权重的方法。**

例如也可以设计：

$$
p_i=
\frac{
z_i^2
}{
\sum_j z_j^2
}
$$

只要不是全零。

或者先 ReLU：

$$
p_i=
\frac{
\max(0,z_i)
}{
\sum_j\max(0,z_j)
}
$$

也能得到某种 normalized weights。

所以：

> Softmax 不是由“权重和为 1”这一要求唯一推导出来的。

它之所以被广泛使用，

是因为 exponential normalization 具有非常好的：

- 单调性；
- 可微性；
- log-space 解释；
- relative-ratio structure；
- maximum-likelihood / categorical modeling compatibility；
- 优化性质。

---

## 13. Softmax 和 Log Probability 的关系

定义：

$$
p_i=
\frac{
e^{z_i}
}{
\sum_j e^{z_j}
}
$$

取 log：

$$
\log p_i
=
\log e^{z_i}
-
\log
\sum_j e^{z_j}
$$

所以：

$$
\boxed{
\log p_i
=
z_i
-
\log\sum_j e^{z_j}
}
$$

第二项：

$$
\operatorname{LSE}(z)
=
\log\sum_j e^{z_j}
$$

叫：

> **LogSumExp**

因此：

$$
\boxed{
\log p_i
=
z_i-\operatorname{LSE}(z)
}
$$

---

## 14. 为什么 Logits 这个名字很合理？

可以重新排列：

$$
z_i
=
\log p_i
+
\operatorname{LSE}(z)
$$

也就是说所有 logits 共享一个 additive normalizing constant。

而两项差：

$$
z_i-z_j
$$

对应：

$$
\log p_i-\log p_j
$$

即：

$$
\boxed{
z_i-z_j
=
\log
\frac{p_i}{p_j}
}
$$

因此：

> logit difference 就是 log probability ratio。

这解释了为什么：

> Softmax 的输入可以理解成 unnormalized log probabilities。

Goodfellow、Bengio、Courville 的《Deep Learning》正是用：

> unnormalized log probabilities

来引出 Softmax。

---

## 15. 为什么 Exponential 和 Maximum Likelihood 很自然？

如果：

$$
z_i
$$

表示 unnormalized log probability，

那么：

$$
e^{z_i}
$$

自然恢复成：

> positive unnormalized probability mass。

然后再除：

$$
\sum_j e^{z_j}
$$

得到合法 categorical distribution。

所以：

```text
log-score
↓ exp
positive unnormalized mass
↓ normalize
probability
```

是一个非常自然的概率建模流程。

---

## 16. Softmax 在分类里可以是真正的概率模型参数

多分类任务中可以定义：

$$
P(Y=i\mid x)
=
\operatorname{softmax}(z(x))_i
$$

这里 Softmax 输出明确就是：

> 模型定义的 categorical conditional distribution。

然后用 maximum likelihood / cross-entropy 训练。

这种语境下说：

> “Softmax 输出概率”

是严格合理的。

---

## 17. Attention 里为什么更推荐叫 Weight？

Transformer 中：

$$
A_{ij}
=
\operatorname{softmax}_j(
s_{ij}
)
$$

每一行也：

$$
A_{ij}\ge0
$$

且：

$$
\sum_jA_{ij}=1
$$

所以数学形式像 probability distribution。

但它主要表示：

> 当前 query 对不同 values 的 normalized aggregation coefficients。

除非模型另外赋予 probabilistic semantics，

最好称：

> **attention weights**

而不是直接断言：

> “这是 key $j$ 是正确答案的真实概率。”

---

## 18. Equal Logits 会发生什么？

如果：

$$
z_1=z_2=\cdots=z_n=c
$$

那么：

$$
e^{z_i}=e^c
$$

所以：

$$
p_i=
\frac{e^c}{ne^c}
=
\frac1n
$$

因此：

$$
\boxed{
\text{equal logits}
\Rightarrow
\text{uniform distribution}
}
$$

在 Attention 中意味着：

> 如果当前 query 对所有 keys 的 compatibility 完全一样，Softmax 会平均读取所有 values。

---

## 19. 最大 Logit 一定有最大 Weight

因为 exponential 是严格单调递增：

$$
z_i>z_j
\Rightarrow
e^{z_i}>e^{z_j}
$$

共同分母不会改变排序：

$$
\boxed{
z_i>z_j
\Rightarrow
p_i>p_j
}
$$

所以 Softmax 保留 logits 的 ranking。

---

## 20. 但 Softmax 不保留线性比例

例如：

$$
z=[2,1]
$$

不是说：

$$
p_1=2p_2
$$

而是：

$$
\frac{p_1}{p_2}
=
e^{2-1}
=
e
$$

所以 Softmax 的比例关系发生在：

> exponential space。

---

## 21. 为什么叫 “Soft Max”？

考虑：

$$
z=[2,1,0]
$$

最大值位置是第一个。

Softmax：

$$
[0.665,0.245,0.090]
$$

不是 hard：

$$
[1,0,0]
$$

但已经明显偏向最大项。

所以可以理解为：

> 一个平滑、可微的“偏向最大值”机制。

它不会像：

$$
\arg\max
$$

那样只保留一个离散 index。

所以叫：

> **soft-max**

---

## 22. Softmax 本身不等于 max 的数值近似

这里要小心。

Softmax 输出的是：

> 一组权重 / distribution。

而：

$$
\max_i z_i
$$

输出一个 scalar。

真正更接近 max 的平滑 scalar approximation 是：

$$
\boxed{
\operatorname{LSE}(z)
=
\log\sum_i e^{z_i}
}
$$

LogSumExp 与 max 的关系：

$$
\max_i z_i
\le
\operatorname{LSE}(z)
\le
\max_i z_i+\log n
$$

所以：

> Softmax 和 LogSumExp 密切相关，但不是同一个函数。

---

## 23. Temperature 是什么？

更一般的 Softmax：

$$
\boxed{
p_i(T)
=
\frac{
e^{z_i/T}
}{
\sum_j e^{z_j/T}
}
}
$$

其中：

$$
T>0
$$

叫：

> **temperature**

---

## 24. T 很小时发生什么？

如果：

$$
T<1
$$

那么 logits 被：

$$
1/T
$$

放大。

例如：

$$
z=[2,1,0]
$$

若：

$$
T=0.1
$$

等价于：

$$
[20,10,0]
$$

Softmax 会非常尖锐：

$$
p_{\max}\approx1
$$

因此：

$$
\boxed{
T\downarrow
\Rightarrow
\text{distribution sharper}
}
$$

---

## 25. T 很大时发生什么？

如果：

$$
T\gg1
$$

logits differences 被压小。

例如：

$$
[2,1,0]
$$

除：

$$
100
$$

变成：

$$
[0.02,0.01,0]
$$

指数几乎都：

$$
\approx1
$$

Softmax 接近：

$$
[1/3,1/3,1/3]
$$

因此：

$$
\boxed{
T\uparrow
\Rightarrow
\text{distribution flatter}
}
$$

---

## 26. 极限情况

如果最大 logit 唯一：

$$
T\rightarrow0^+
$$

则：

$$
\operatorname{softmax}(z/T)
$$

趋向：

> one-hot argmax。

而：

$$
T\rightarrow\infty
$$

则：

$$
p_i\rightarrow\frac1n
$$

趋向 uniform。

因此 Temperature 控制：

> soft selection 到 hard selection 的程度。

---

## 27. Transformer 的 1/√d_k 是 Temperature 吗？

形式上：

$$
\operatorname{softmax}
\left(
\frac{
QK^\top
}{
\sqrt{d_k}
}
\right)
$$

确实类似于：

$$
T=\sqrt{d_k}
$$

的温度缩放。

但更准确的说法是：

> Transformer 使用固定 scaling factor $1/\sqrt{d_k}$ 来稳定 dot-product logits 的尺度。

它不是：

> 一个用户为了采样 diversity 任意调节的 temperature hyperparameter。

数学形式类似，

设计目的不同。

---

## 28. 为什么 Logit Scale 会影响 Attention Sharpness？

Softmax 不对：

> 乘法缩放

保持不变。

例如：

$$
z=[2,1,0]
$$

Softmax：

$$
\approx
[0.665,0.245,0.090]
$$

如果乘：

$$
10
$$

：

$$
[20,10,0]
$$

则几乎：

$$
[1,0,0]
$$

所以：

$$
\boxed{
\operatorname{softmax}(cz)
\neq
\operatorname{softmax}(z)
}
$$

一般当：

$$
c\neq1
$$

---

## 29. 为什么加法不影响，乘法会影响？

加常数：

$$
z_i+c
$$

只给所有：

$$
e^{z_i}
$$

共同乘：

$$
e^c
$$

归一化时抵消。

乘常数：

$$
cz_i
$$

会改变 pairwise difference：

$$
cz_i-cz_j
=
c(z_i-z_j)
$$

所以 relative ratio：

$$
\frac{p_i}{p_j}
=
e^{c(z_i-z_j)}
$$

改变。

因此：

> Softmax 只对 global shift 不敏感，对 scale 非常敏感。

---

## 30. 这和 Attention Scaling 为什么完全连起来了？

Attention raw logits：

$$
q^\top k
$$

维度：

$$
d_k
$$

变大时典型 magnitude 会增长。

如果不 scale：

> score differences 也可能整体变大。

Softmax 会越来越尖锐。

一旦过度尖锐，

梯度可能非常小。

所以 Transformer 用：

$$
\boxed{
1/\sqrt{d_k}
}
$$

把 logits 保持在更合理尺度。

---

## 31. Softmax 的导数到底是什么？

设：

$$
p_i
=
\frac{
e^{z_i}
}{
\sum_k e^{z_k}
}
$$

我们需要计算：

$$
\frac{\partial p_i}{\partial z_j}
$$

结果是：

$$
\boxed{
\frac{\partial p_i}{\partial z_j}
=
p_i
(
\delta_{ij}-p_j
)
}
$$

其中：

$$
\delta_{ij}
=
\begin{cases}
1,&i=j\\
0,&i\neq j
\end{cases}
$$

---

## 32. 对自己的 Logit 求导

如果：

$$
i=j
$$

那么：

$$
\boxed{
\frac{\partial p_i}{\partial z_i}
=
p_i(1-p_i)
}
$$

所以提高：

$$
z_i
$$

会提高：

$$
p_i
$$

只要：

$$
0<p_i<1
$$

导数为正。

---

## 33. 对别人的 Logit 求导

如果：

$$
i\neq j
$$

那么：

$$
\boxed{
\frac{\partial p_i}{\partial z_j}
=
-p_ip_j
}
$$

为负。

这意味着：

> 提高一个类别/Key 的 logit，会压低其他项的 normalized weight。

这正是 Softmax 的：

> **competition coupling。**

---

## 34. Softmax Jacobian

把所有偏导写成矩阵：

$$
\boxed{
J
=
\operatorname{diag}(p)
-
pp^\top
}
$$

这说明 Softmax 输出不是：

> 每一维独立变化。

每个 logit 都会影响整行 distribution。

这和 Sigmoid 有根本差异。

---

## 35. Softmax 为什么会 Saturate？

如果某一项：

$$
p_i\approx1
$$

那么：

$$
p_i(1-p_i)
\approx0
$$

其他：

$$
p_j\approx0
$$

相关偏导：

$$
p_j(1-p_j)
$$

也很小。

所以 distribution 极端尖锐时，

Softmax 本身的很多局部 derivatives 会变小。

这就是：

> saturation。

---

## 36. Transformer 原论文为什么担心这一点？

《Attention Is All You Need》指出：

当：

$$
d_k
$$

较大时，

dot products magnitude 可能较大，

从而把 Softmax 推到：

> extremely small gradients 的区域。

因此缩放：

$$
\frac1{\sqrt{d_k}}
$$

来缓解。

这就是 QK scaling 与 Softmax gradient 之间的直接连接。

---

## 37. 但“Softmax Saturation = 模型完全学不动”也不总正确

需要区分具体 loss 和 computation graph。

例如分类中：

> Softmax + Cross-Entropy

组合后对 logits 的 gradient 非常简洁：

$$
\boxed{
\frac{\partial L}{\partial z_i}
=
p_i-y_i
}
$$

所以如果模型非常自信但预测错误：

$$
p_{\text{wrong}}\approx1
$$

gradient 仍然可以很大。

Goodfellow 等人的教材也强调：

> 合理的 log-likelihood objective 能缓解把 Softmax 单独看成饱和激活时的一些训练问题。

---

## 38. Attention 中情况为什么不同？

Attention Softmax 通常不是最终分类输出。

它位于网络中间：

$$
scores
\rightarrow
softmax
\rightarrow
weighted\ values
\rightarrow
后续网络
\rightarrow
loss
$$

如果 attention distribution 极端饱和，

某些 score adjustments 的梯度可能变得非常弱。

这就是原 Transformer 在 Scaled Dot-Product Attention 中明确加入 scaling 的背景。

---

## 39. Cross-Entropy 为什么和 Softmax 配合得这么好？

假设正确类别：

$$
y
$$

是 one-hot。

Cross-entropy：

$$
L
=
-\sum_i y_i\log p_i
$$

如果正确类别是：

$$
c
$$

则：

$$
L=-\log p_c
$$

而：

$$
\log p_c
=
z_c-\log\sum_j e^{z_j}
$$

所以：

$$
L
=
-z_c
+
\log\sum_j e^{z_j}
$$

求导得到：

$$
\boxed{
\nabla_zL=p-y
}
$$

非常干净。

所以：

> Softmax + log-likelihood/cross-entropy

是一个非常自然的组合。

---

## 40. Softmax 和 Sigmoid 有什么根本区别？

Sigmoid：

$$
\boxed{
\sigma(z)
=
\frac1{1+e^{-z}}
}
$$

对每一个 logit：

> 独立地映射到 $(0,1)$。

Softmax：

$$
\boxed{
p_i
=
\frac{e^{z_i}}{\sum_j e^{z_j}}
}
$$

不同 dimensions：

> 相互耦合、共同竞争。

---

## 41. 一个例子

三个 logits：

$$
[2,1,0]
$$

分别 Sigmoid：

$$
[
\sigma(2),
\sigma(1),
\sigma(0)
]
$$

大约：

$$
[0.881,0.731,0.5]
$$

总和：

$$
2.112
$$

不等于：

$$
1
$$

因为每一项独立。

Softmax：

$$
[0.665,0.245,0.090]
$$

总和：

$$
1
$$

所以 Softmax 表示：

> 候选之间的相对分配。

---

## 42. 什么时候更适合 Sigmoid？

如果多个标签可以同时独立成立：

```text
这张图片里有：
cat = yes
dog = yes
car = no
```

通常适合多个 independent sigmoids。

因为：

> cat 成立并不排斥 dog 成立。

---

## 43. 什么时候更适合 Softmax？

如果候选是：

> 相互竞争的一组选项，

例如单标签分类：

```text
cat
dog
car
```

只选一个类别，

Softmax 很自然。

Attention 虽然不是分类，

但对于一个 query：

> 它要在所有 allowed keys 之间分配总量为 1 的读取权重。

所以 Softmax 也很合适。

---

## 44. Attention 为什么不用每个 Key 一个 Sigmoid？

理论上可以设计 sigmoid attention/gating。

但如果每个：

$$
\alpha_j
=
\sigma(s_j)
$$

独立，

则：

$$
\sum_j\alpha_j
$$

不固定。

所有 keys 甚至可以同时：

$$
\approx1
$$

那么 output magnitude 会随着：

> candidate 数量和 active gates 数量

发生变化。

Softmax 则给每个 query 一个规范化的竞争式 distribution：

$$
\sum_j\alpha_j=1
$$

使 weighted aggregation 尺度更稳定、更容易解释。

这不是说 Sigmoid Attention 不存在，

而是解释标准 Transformer 为什么选择 row-wise Softmax。

---

## 45. Softmax 是一种“竞争机制”

因为：

$$
\sum_jp_j=1
$$

所以某一个：

$$
p_i
$$

增加，

通常意味着其他项的总份额减少。

从 derivative：

$$
\frac{\partial p_i}{\partial z_j}
=
-p_ip_j
\qquad(i\neq j)
$$

也能直接看见：

> 不同候选互相竞争。

在 Attention 中：

> 一个 key 获得更多 attention mass，会压缩其他 keys 的相对份额。

---

## 46. 为什么所有 Weight 都严格大于 0？

普通 Softmax：

$$
e^{z_i}>0
$$

所以对于有限：

$$
z_i
$$

都有：

$$
p_i>0
$$

因此它不会产生严格的：

$$
0
$$

权重。

只会：

> 非常接近 0。

例外是：

> Attention Mask。

我们可以在 Softmax 前把某个 logit 设为：

$$
-\infty
$$

于是：

$$
e^{-\infty}=0
$$

最终 weight 精确为：

$$
0
$$

---

## 47. 这就是 Causal Mask 的数学基础

假设 scores：

$$
[2,1,0]
$$

但第三个 key 不允许访问。

加 mask：

$$
[2,1,-\infty]
$$

Softmax：

$$
\left[
\frac{e^2}{e^2+e^1},
\frac{e^1}{e^2+e^1},
0
\right]
$$

于是被 mask 的位置完全不参与 weighted sum。

所以：

> Mask 不是 Softmax 之后把结果随便改成 0。

标准做法通常是在：

> Softmax 前修改 logits。

---

## 48. 为什么 -∞ 是完美 Mask？

因为：

$$
e^{-\infty}=0
$$

所以：

$$
p_{\text{masked}}=0
$$

而且剩余未 mask 的项会自动重新归一化为：

$$
1
$$

这正是我们需要的行为。

---

## 49. 数值稳定问题：直接 exp(z) 可能溢出

假设：

$$
z=[1000,999,998]
$$

数学上 Softmax 完全正常。

但计算机中：

$$
e^{1000}
$$

可能 overflow。

如果变成：

$$
\infty
$$

后续：

$$
\frac{\infty}{\infty}
$$

可能得到：

> NaN。

所以实际代码绝不能天真地直接：

```python
np.exp(z) / np.exp(z).sum()
```

---

## 50. Shift Invariance 给了我们一个完美技巧

我们知道：

$$
\operatorname{softmax}(z)
=
\operatorname{softmax}(z-c)
$$

可以选：

$$
c=\max_i z_i
$$

于是：

$$
\boxed{
\operatorname{softmax}(z)
=
\operatorname{softmax}
(
z-\max(z)
)
}
$$

最大的 logit 被变成：

$$
0
$$

其余：

$$
\le0
$$

所以最大的 exponential：

$$
e^0=1
$$

不会 overflow。

---

## 51. 数值例子

原 logits：

$$
[1000,999,998]
$$

减最大值：

$$
[0,-1,-2]
$$

Softmax 完全相同。

指数：

$$
[1,e^{-1},e^{-2}]
$$

约：

$$
[1,0.368,0.135]
$$

再归一化。

数值非常稳定。

---

## 52. 所以稳定 Softmax 的标准实现

概念代码：

```python
def softmax(z):
    z = z - z.max()
    exp_z = exp(z)
    return exp_z / exp_z.sum()
```

对于 batch / matrix：

> 沿需要 Softmax 的 dimension 分别减该维最大值。

在 Attention 中：

> 每个 query row 各自减自己的 row maximum。

---

## 53. 为什么不能整个 Matrix 只减一个全局最大值？

数学上如果给所有 logits 加/减相同常数，

仍然不会改变 Softmax。

但 row-wise Attention 的 Softmax 是：

> 每一行独立的 distribution。

最自然和数值稳定的做法是：

$$
z_{i,:}
-
\max_j z_{ij}
$$

每一行单独处理。

这样每个 row 至少有一个：

$$
0
$$

---

## 54. Attention Softmax 到底在哪个维度？

如果：

$$
S=QK^\top
$$

shape：

$$
[n_q,n_k]
$$

第：

$$
i
$$

行：

$$
S_{i,:}
$$

是 query $i$ 对所有 keys 的 scores。

所以 Softmax 沿：

$$
\boxed{
\text{key dimension}
}
$$

即：

$$
j
$$

做。

得到：

$$
\sum_jA_{ij}=1
$$

对每个 query 独立成立。

---

## 55. 一个 2 × 3 Attention Score Matrix

假设：

$$
S=
\begin{bmatrix}
2&1&0\\
0&0&0
\end{bmatrix}
$$

row-wise Softmax：

第一行：

$$
\approx
[0.665,0.245,0.090]
$$

第二行：

$$
[1/3,1/3,1/3]
$$

所以：

$$
A=
\begin{bmatrix}
0.665&0.245&0.090\\
0.333&0.333&0.333
\end{bmatrix}
$$

表示：

> 第一个 query 偏向第一个 key；

> 第二个 query 认为三个 keys 同等匹配。

---

## 56. 为什么不是 Column-Wise Softmax？

Column-wise 会让：

> 所有 queries 竞争同一个 key 的总 attention mass。

但标准 Attention 想表达的是：

> 对每个 query，它应该怎样在所有 available keys 中分配读取权重。

所以 normalization 应该固定 query，遍历 keys。

即：

> row-wise。

---

## 57. Softmax 和 Weighted Sum 如何连接？

得到：

$$
A_{ij}
$$

后：

$$
O=AV
$$

对于 query $i$：

$$
\boxed{
o_i
=
\sum_jA_{ij}v_j
}
$$

因为：

$$
A_{ij}\ge0
$$

且：

$$
\sum_jA_{ij}=1
$$

所以：

$$
o_i
$$

是 Values 的 convex combination。

这给 Attention output 一个相对稳定的 aggregation scale。

---

## 58. 为什么 Convex Combination 是个好性质？

如果 Values：

$$
v_j
$$

都有类似 magnitude，

normalized weights 不会仅仅因为：

> memory 有更多 token

就让 sum magnitude 线性增长。

对比未归一化：

$$
\sum_j s_jv_j
$$

其 scale 更容易受：

- score magnitude；
- sequence length；

直接影响。

Softmax 把：

> relative preference

和：

> aggregate normalization

结合起来。

---

## 59. 但 Multi-Head + W_O 后就不再只是 Convex Hull

单个 attention head：

$$
AV
$$

确实逐 query 是 Values 的 convex combination。

但后面：

$$
\operatorname{Concat}(head_1,\ldots,head_h)W_O
$$

有 learned linear output projection。

再加：

- residual；
- FFN；

整个 Transformer layer 并不受限于：

> 原 Values 的 convex hull。

所以不要把“Attention 是 weighted average”过度解释成：

> Transformer 只能平均已有信息。

---

## 60. Softmax 为什么会放大差异？

假设：

$$
z_1-z_2=1
$$

weight ratio：

$$
e
$$

如果差：

$$
5
$$

ratio：

$$
e^5
\approx148.4
$$

所以 logit difference 线性增加，

weight ratio 指数增加。

这让 Softmax 能把：

> moderate score advantage

变成：

> very strong preference。

这也是它既强大又需要 scale control 的原因。

---

## 61. 为什么这种“放大”不是永远越强越好？

如果所有 attention distributions 都极度尖锐：

```text
一个 key ≈ 1
其他 key ≈ 0
```

模型几乎变成 hard routing。

这可能：

- 降低信息混合；
- 让 gradient 更集中/更弱；
- 对 score perturbation 更敏感。

所以 Attention 需要一个合适的 logit scale。

Transformer 的：

$$
1/\sqrt{d_k}
$$

正是为了避免 dimension 增大时无意中让 Softmax越来越尖锐。

---

## 62. Softmax 的 Entropy

一个 distribution：

$$
p
$$

的 entropy：

$$
H(p)
=
-\sum_i p_i\log p_i
$$

如果 Softmax 很平：

$$
p_i\approx1/n
$$

entropy 高。

如果很尖：

$$
p_{\max}\approx1
$$

entropy 低。

因此 Temperature / logit scale 也可以理解为：

> 控制 Softmax distribution entropy。

这在：

- knowledge distillation；
- sampling；
- calibration；

等场景里很重要。

---

## 63. Temperature 和 Knowledge Distillation

在 knowledge distillation 中常使用较高：

$$
T
$$

让 teacher Softmax distribution 更平滑。

这样不仅告诉 student：

> 正确类别是谁，

还暴露：

> 其他类别之间的相对关系。

例如：

```text
cat 0.6
dog 0.3
car 0.1
```

比：

```text
cat 0.999
dog 0.001
car ~0
```

保留更多“dark knowledge”。

这是 Temperature 的另一个经典应用。

但与 Transformer $1/\sqrt{d_k}$ 的设计目的不要混淆。

---

## 64. 为什么 Softmax 输入叫 Logits？

对于二分类 Logistic Regression，

logit 通常和：

$$
\log\frac{p}{1-p}
$$

相关。

Softmax 是多分类 Logistic / Log-Linear 模型的自然推广。

对两类 Softmax：

$$
p_1
=
\frac{e^{z_1}}{e^{z_1}+e^{z_2}}
$$

设：

$$
z_2=0
$$

则：

$$
p_1
=
\frac{e^{z_1}}{e^{z_1}+1}
=
\sigma(z_1)
$$

所以：

> Sigmoid 可以看成两类 Softmax 的一种特殊参数化。

---

## 65. 两类 Softmax 怎样化成 Sigmoid？

更一般：

$$
p_1
=
\frac{
e^{z_1}
}{
e^{z_1}+e^{z_2}
}
$$

上下同除：

$$
e^{z_2}
$$

得到：

$$
p_1
=
\frac{
e^{z_1-z_2}
}{
e^{z_1-z_2}+1
}
$$

所以：

$$
\boxed{
p_1
=
\sigma(z_1-z_2)
}
$$

这再次说明：

> Softmax 只关心 logit differences。

---

## 66. Softmax 是不是可逆？

不是一一可逆。

因为：

$$
z
$$

和：

$$
z+c\mathbf1
$$

会产生完全相同 Softmax。

所以从：

$$
p
$$

无法恢复 logits 的绝对 offset。

只能恢复：

> logit differences。

例如：

$$
z_i-z_j
=
\log\frac{p_i}{p_j}
$$

因此 Softmax 有一维 redundancy：

> global additive shift。

---

## 67. 为什么这和 n-1 自由度对应？

$$
n
$$

个 logits，

但整体加常数不改变结果。

所以有效自由度：

$$
n-1
$$

恰好和 categorical probability vector：

$$
\sum_i p_i=1
$$

所拥有的：

$$
n-1
$$

自由度对应。

这是 Softmax 几何上非常漂亮的地方。

---

## 68. Softmax 的 Historical Note

“Softmax”这个名称通常归功于 John S. Bridle 在 1990 年对 neural-network classification outputs 的讨论。

他把它描述为：

> normalized exponential 的 multi-input logistic generalization。

但 exponential-normalization 形式本身属于更广泛、更早的统计建模思想。

因此历史上更准确的表述是：

> Bridle 1990 是“softmax”这一术语和神经网络分类语境的重要早期来源之一，

而不是：

> “1990 年以前数学上从未出现指数归一化。”

---

## 69. Softmax 为什么适合 Attention，而不是“理论上唯一正确”？

把前面的性质放在一起：

1. 任意 real scores 都能映射成 positive weights；
2. weights 自动和为 1；
3. 排序不变；
4. relative ratios 由 score differences 控制；
5. global shift 不影响结果；
6. 完全可微；
7. 可以通过 scale/temperature 控制 sharpness；
8. matrix implementation 高效；
9. 和 masking 天然兼容。

所以 Softmax 对 Attention 非常合适。

但：

$$
\boxed{
\text{useful and elegant}
\neq
\text{mathematically unique}
}
$$

现代研究确实存在：

- sparsemax；
- entmax；
- sigmoid attention；
- kernel attention；

等其他 normalization / routing 形式。

---

## 70. 一个完整 Attention 数值例子

假设一个 Query 对 3 个 Keys 的 scaled scores：

$$
z=[2,1,0]
$$

---

### Step 1：Exponential

$$
e^z
=
[
e^2,
e^1,
e^0
]
$$

约：

$$
[7.389,2.718,1]
$$

---

### Step 2：Sum

$$
Z
=
7.389+2.718+1
=
11.107
$$

---

### Step 3：Normalize

$$
p_1
=
7.389/11.107
\approx0.665
$$

$$
p_2
\approx0.245
$$

$$
p_3
\approx0.090
$$

所以：

$$
\boxed{
A=[0.665,0.245,0.090]
}
$$

---

### Step 4：Weighted Values

若：

$$
v_1=[1,0]
$$

$$
v_2=[0,1]
$$

$$
v_3=[1,1]
$$

那么：

$$
o
=
0.665v_1
+
0.245v_2
+
0.090v_3
$$

$$
=
[0.755,\;0.335]
$$

这就是完整：

$$
\text{scores}
\rightarrow
\text{Softmax}
\rightarrow
\text{Attention output}
$$

---

## 71. 如果给全部 Scores 加 100

变成：

$$
[102,101,100]
$$

Softmax 仍：

$$
[0.665,0.245,0.090]
$$

因为：

$$
\boxed{
\text{relative differences unchanged}
}
$$

---

## 72. 如果把全部 Scores 乘 10

变成：

$$
[20,10,0]
$$

Softmax 几乎：

$$
[0.99995,\;0.000045,\;\text{very small}]
$$

所以：

$$
\boxed{
\text{scale controls sharpness}
}
$$

这就是为什么 Attention logits 的尺度绝不能忽略。

---

## 73. 如果全部 Scores 都相同

$$
[5,5,5]
$$

Softmax：

$$
[1/3,1/3,1/3]
$$

所以：

> 没有相对 preference 时，Attention 自动退化成均匀平均。

---

## 74. 如果某个 Score 是 -∞

$$
[2,1,-\infty]
$$

Softmax：

$$
[\approx0.731,\approx0.269,0]
$$

所以：

> Mask 可以彻底禁止某个 Value 被读取。

---

## 75. ACT 中 Softmax 在哪里？

ACT 使用 Transformer。

因此：

#### CVAE Encoder Self-Attention

对于 `[CLS]`、joint、action tokens：

$$
A
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)
$$

决定 token 间信息聚合。

---

#### Policy Encoder Self-Attention

visual / joint / latent tokens 之间同样使用 Softmax attention weights。

---

#### Policy Decoder Cross-Attention

action decoder query 对 observation memory：

$$
QK^\top
$$

得到 scores，

再 Softmax。

所以不同 future action slots 可以对 observation memory 产生不同 normalized reading patterns。

---

## 76. ACT 的 Temporal Ensemble 再次不是 Softmax Attention

Temporal Ensemble 权重：

$$
w_i=e^{-mi}
$$

然后：

$$
\alpha_i=
\frac{w_i}{\sum_jw_j}
$$

从形式上看：

$$
\alpha_i
=
\frac{
e^{-mi}
}{
\sum_j e^{-mj}
}
$$

它其实也可以写成：

$$
\operatorname{softmax}(
[0,-m,-2m,\ldots]
)
$$

这是一个很有意思的数学观察。

但是它的 logits：

$$
-mi
$$

不是网络根据 Q/K content 学出来的。

它们是：

> 人工定义的 prediction-age scores。

所以：

```text
Transformer Attention:
learned content-dependent logits

Temporal Ensemble:
hand-designed time-dependent logits
```

两者仍然不是同一个机制。

---

## 77. 这个联系为什么值得知道？

因为它揭示：

> Softmax 本身只是一个“把 relative scores 变成 normalized weights”的通用函数。

这些 scores 可以来自：

- class logits；
- QK dot products；
- manually designed energies；
- distances；
- other models。

Softmax 不关心：

> score 是怎么来的。

它只负责：

$$
\boxed{
\text{scores}
\rightarrow
\text{normalized relative weights}
}
$$

---

## 78. Energy-Based 视角

有时概率模型写：

$$
p_i
=
\frac{
e^{-E_i}
}{
\sum_j e^{-E_j}
}
$$

其中：

$$
E_i
$$

叫 energy。

energy 越低：

$$
e^{-E_i}
$$

越大。

这和 Softmax logits：

$$
z_i=-E_i
$$

完全一致。

所以 Softmax 也可以理解为：

> normalized exponential of negative energies。

这与统计物理中的 Gibbs / Boltzmann distribution 形式密切相关。

---

## 79. 为什么 Softmax 对 Score Difference 的解释这么自然？

因为：

$$
\log
\frac{p_i}{p_j}
=
z_i-z_j
$$

也就是说：

> logits 的差值直接等于 log weight ratio。

所以模型只要学：

> “A 应该比 B 更偏好多少 log-units”

Softmax 就自动把它转成：

> multiplicative relative preference。

这种结构很适合 ranking / competition。

---

## 80. Softmax 并不会让最大的项一定接近 1

例如：

$$
[0.1,0.09,0.08]
$$

虽然第一个最大，

但三者非常接近。

Softmax 会比较平。

所以：

> “最大 logit”只决定谁第一。

而：

> “和其他 logits 差多少”决定有多自信 / 多尖锐。

这在 Attention 中也一样。

---

## 81. 为什么绝对 Logit 很大也不代表非常自信？

例如：

$$
[1000,999.99,999.98]
$$

虽然数字都极大，

差值却只有：

$$
[0,-0.01,-0.02]
$$

Softmax 仍然接近均匀。

所以：

> confidence / attention sharpness 取决于 relative differences，不是 absolute magnitude offset。

---

## 82. 但为什么 Transformer 又说 Dot Product Magnitude 大会导致 Softmax Saturation？

因为 Transformer 担心的不是：

> 所有 logits 一起加大常数。

而是：

> dot products 的 variance / pairwise differences 随 $d_k$ 增大。

如果整个 score distribution scale 增大：

$$
z
\rightarrow
cz
$$

pairwise differences 也被：

$$
c
$$

放大。

这会让 Softmax 更尖锐。

所以没有矛盾。

---

## 83. Shift 和 Scale 必须严格区分

### Shift

$$
z\rightarrow z+c
$$

Softmax：

$$
\boxed{\text{不变}}
$$

---

### Scale

$$
z\rightarrow cz
$$

Softmax：

$$
\boxed{\text{改变}}
$$

这两个性质是理解 Softmax 最关键的一组对比。

---

## 84. Softmax Numerical Stability 和 Shift Invariance 是同一个性质的工程应用

因为：

$$
softmax(z)=softmax(z-\max z)
$$

我们可以自由改变 absolute offset，

把最大 logit 移到：

$$
0
$$

而不改变数学结果。

所以：

> 一个纯数学不变性，直接带来了一个重要的工程稳定技巧。

这是非常漂亮的例子。

---

## 85. 常见误解一：Softmax 是把每个数除以总和

**错误。**

先要：

$$
e^{z_i}
$$

再归一化。

---

## 86. 常见误解二：Softmax 输入必须是正数

**错误。**

logits 可以是任意实数。

指数会把它们变正。

---

## 87. 常见误解三：Softmax 输出一定是真实概率

**不一定。**

分类概率模型中可以定义为概率。

Attention 中更准确是 normalized weights。

---

## 88. 常见误解四：Softmax 是唯一能生成概率分布的方法

**错误。**

它是非常常用且有良好性质的一种 normalized exponential mapping。

---

## 89. 常见误解五：Softmax 只关心哪个 Logit 最大

**错误。**

它还关心：

> 最大项和其他项差多少。

---

## 90. 常见误解六：所有 Logits 加 100 会让模型更自信

**错误。**

Softmax 完全不变。

---

## 91. 常见误解七：所有 Logits 乘 100 也不影响

**错误。**

会使 distribution 通常更尖锐。

---

## 92. 常见误解八：Scaled Dot-Product 的 √d_k 是为了让 Softmax 和为 1

**错误。**

和为 1 是 Softmax normalization 的结果。

Scaling 是为了控制 logits scale。

---

## 93. 常见误解九：Sigmoid 和 Softmax 都输出 0–1，所以一样

**错误。**

Sigmoid dimensions 通常独立。

Softmax dimensions 竞争并和为 1。

---

## 94. 常见误解十：Softmax 每一维的导数只和自己有关

**错误。**

$$
\frac{\partial p_i}{\partial z_j}
=
-p_ip_j
$$

对：

$$
i\neq j
$$

也非零。

所以 outputs 强耦合。

---

## 95. 常见误解十一：Softmax Saturation 意味着 Cross-Entropy 一定完全没梯度

**错误。**

Softmax + cross-entropy 对 logits：

$$
\nabla_zL=p-y
$$

组合后有重要简化。

需要看完整 objective，而不是只看 Softmax 单层 derivative。

---

## 96. 常见误解十二：减 max 是一种近似

**错误。**

$$
softmax(z-\max z)
$$

与：

$$
softmax(z)
$$

在精确数学上完全相同。

只是数值实现更稳定。

---

## 97. 常见误解十三：Attention Softmax 是对整个 Matrix 一次归一化

**错误。**

标准 Attention 对每个 query：

> 沿 key dimension 单独做 Softmax。

---

## 98. 常见误解十四：Temporal Ensemble 使用 Softmax，所以就是 Transformer Attention

即使把其指数归一化写成 Softmax 形式，

两者仍不同。

关键不在 Softmax，

而在：

> logits 是怎样生成的。

Transformer：

$$
q^\top k
$$

learned + input dependent。

Temporal Ensemble：

$$
-mi
$$

hand-designed + age dependent。

---

## 99. 用四条性质记住 Softmax

### 1. Positivity

$$
\boxed{
p_i>0
}
$$

---

### 2. Normalization

$$
\boxed{
\sum_i p_i=1
}
$$

---

### 3. Shift Invariance

$$
\boxed{
softmax(z+c)=softmax(z)
}
$$

---

### 4. Ratio by Differences

$$
\boxed{
\frac{p_i}{p_j}
=
e^{z_i-z_j}
}
$$

这四条已经解释了 Softmax 大部分核心行为。

---

## 100. 再加一条：Scale Controls Sharpness

$$
\boxed{
softmax(cz)
}
$$

随着：

$$
c
$$

增大通常更尖锐。

等价 temperature：

$$
T=1/c
$$

越小越尖锐。

这就是 Transformer Attention scaling 的入口。

---

## 101. 用一个公式理解 Softmax 梯度

$$
\boxed{
\frac{\partial p_i}{\partial z_j}
=
p_i(\delta_{ij}-p_j)
}
$$

它告诉我们：

- 自己的 logit 上升 → 自己概率上升；
- 别人的 logit 上升 → 自己概率下降；
- 所有候选存在竞争；
- 极端饱和时很多局部 gradient 会变小。

---

## 102. 一句话真正理解 Softmax

> **Softmax 把一组任意实数 logits 先指数化成正的相对质量，再除以总质量形成和为 1 的权重；指数结构使两个权重的比值只由对应 logits 的差值决定，因此模型可以在不关心绝对 score 零点的情况下表达候选之间的相对偏好，而 logit scale 又决定这种偏好最终是平滑还是尖锐。**

在 Attention 中可以进一步读成：

```text
QKᵀ
给出“谁比谁更匹配”

1 / √d_k
控制这些差异的尺度

Softmax
把相对匹配程度
变成每个 Query 的读取权重

V
提供真正被读取的信息
```

所以整个公式：

$$
\boxed{
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
}
$$

现在每一部分都有了明确数学意义。

---

## 103. 下一步

到这里，Scaled Dot-Product Attention 公式的三个数学核心已经分别拆开：

```text
Q / K / V
↓
Dot Product
↓
Scale
↓
Softmax
↓
Weighted Sum
```

下一篇最自然的是把它们重新组装：

> **[Self-Attention：一个 Token 到底怎样读取整条 Sequence？](./self-attention.md)**

这篇会用一组非常小的 token vectors，从：

$$
X
$$

开始真正计算：

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

然后：

$$
QK^\top
$$

→ scale → mask（如果有）→ row-wise Softmax → $AV$。

并解释：

- 为什么输出 token 数量不变；
- 为什么每个 token 得到不同 context；
- 为什么 Self-Attention 没有 position information 就是 permutation-equivariant；
- 为什么 residual 之后不会“丢掉自己”；
- 为什么 ACT policy encoder 可以用 Self-Attention 融合 1202 个 tokens。

---

### Mathematical / Deep Learning Source

Ian Goodfellow, Yoshua Bengio, Aaron Courville.  
**Deep Learning.** MIT Press, 2016.

Online book:

https://www.deeplearningbook.org/

Chapter 6 给出 Softmax：

$$
\boxed{
softmax(z)_i
=
\frac{
\exp(z_i)
}{
\sum_j\exp(z_j)
}
}
$$

并从：

> unnormalized log probabilities

的角度解释 exponential + normalization。

教材还明确给出：

$$
softmax(z)
=
softmax(z+c)
$$

以及数值稳定形式：

$$
\boxed{
softmax(z)
=
softmax(
z-\max_i z_i
)
}
$$

说明 subtract-max 不改变数学结果，同时避免 exponential overflow。

---

### Historical Note

John S. Bridle.  
**Probabilistic Interpretation of Feedforward Classification Network Outputs, with Relationships to Statistical Pattern Recognition.**  
1990.

该工作把 normalized exponential 描述为：

> softmax multi-input generalization of the logistic nonlinearity，

是 “softmax” 术语在 neural-network classification 语境中的重要早期来源。

DOI:

10.1007/978-3-642-76153-9_28

---

### Transformer Primary Source

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,  
Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin.

**Attention Is All You Need.**  
NIPS 2017.

- arXiv: https://arxiv.org/abs/1706.03762
- PDF: https://arxiv.org/pdf/1706.03762

Section 3.2.1 定义：

$$
\boxed{
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{
QK^\top
}{
\sqrt{d_k}
}
\right)V
}
$$

原论文明确指出：

- Query 与 Keys 先计算 dot products；
- 除以 $\sqrt{d_k}$；
- 再 Softmax 得到 Values 的权重；
- 大 $d_k$ 下 unscaled dot products 可能 magnitude 过大，把 Softmax 推入 extremely small gradients 区域。

---

### 本文知识连接

#### 数学前置

- Exponent
- Logarithm
- Probability Distribution
- Weighted Average

#### 数学延伸

- LogSumExp
- Categorical Distribution
- Entropy
- Gradient & Chain Rule

#### Deep Learning

- [Attention](./attention.md)
- [Query / Key / Value](./qkv.md)
- [Transformer](./transformer.md)
- [Self-Attention](./self-attention.md)
- [Cross-Attention](./cross-attention.md)
- [Causal Mask](./causal-mask.md)
- Cross-Entropy
- Sigmoid
- Temperature

#### 数学连接

- [Dot Product](./dot-product.md)

#### Robot Learning

- [ACT Architecture](../robot-learning/act/architecture.md)
- [Temporal Ensemble](../robot-learning/act/temporal-ensemble.md)

#### 下一步

- [Self-Attention](./self-attention.md)
