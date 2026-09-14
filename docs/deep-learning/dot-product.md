---
title: "Dot Product：为什么两个向量点乘就能衡量匹配？"
description: "从逐维乘积、长度、夹角与投影出发，推导 a·b = ||a||||b||cosθ，解释点积的几何意义、与 cosine similarity 的区别，以及 Transformer 为什么使用 QKᵀ 作为 Attention compatibility score。"
status: reviewed
pageType: concept
canonical: /deep-learning/dot-product
updated: "2026-09-15"
---

# Dot Product：为什么两个向量点乘就能衡量匹配？

在 [Query / Key / Value](./qkv.md) 中，我们已经知道 Transformer 会计算：

\[
q_i^\top k_j
\]

来衡量一个 Query 和一个 Key 的：

> compatibility。

但这里有一个非常基础、也非常值得追问的问题：

> **为什么两个向量只要“逐维相乘再相加”，就能表示它们匹不匹配？**

点积定义看起来非常朴素：

\[
\boxed{
\mathbf a\cdot\mathbf b
=
a_1b_1+a_2b_2+\cdots+a_nb_n
}
\]

例如：

\[
\mathbf a=[1,2]
\]

\[
\mathbf b=[3,4]
\]

那么：

\[
\mathbf a\cdot\mathbf b
=
1\times3+2\times4
=
11
\]

但为什么：

\[
11
\]

这个数字有几何意义？

为什么同方向的向量通常点积为正？

为什么正交：

\[
\mathbf a\perp\mathbf b
\]

时点积恰好：

\[
0
\]

为什么反方向会变成负数？

更重要的是：

> **为什么 Transformer 可以拿这个运算判断 Query 和 Key 是否“匹配”？**

要回答这些问题，不能只背：

\[
\mathbf a\cdot\mathbf b
=
\|\mathbf a\|
\|\mathbf b\|
\cos\theta
\]

而要真正理解这个公式为什么成立，以及它告诉了我们什么。

---

# 1. 先从最简单的情况开始：一维

假设一条数轴上有两个一维向量：

\[
a=3
\]

\[
b=2
\]

它们的乘积：

\[
ab=6
\]

两者都是正数，表示：

> 它们指向同一方向。

如果：

\[
a=3
\]

\[
b=-2
\]

那么：

\[
ab=-6
\]

表示：

> 它们方向相反。

如果其中一个：

\[
b=0
\]

那么：

\[
ab=0
\]

也就没有方向贡献。

所以在一维空间中，普通乘法已经同时编码了两件事：

1. **大小**；
2. **方向是否一致**。

这其实已经是点积最原始的影子。

---

# 2. 二维后为什么变成“逐维乘积再求和”？

现在：

\[
\mathbf a=
[a_1,a_2]
\]

\[
\mathbf b=
[b_1,b_2]
\]

两个坐标维度可以分别看成：

- x 方向；
- y 方向。

那么：

\[
a_1b_1
\]

衡量它们在 x 方向上的一致贡献，

而：

\[
a_2b_2
\]

衡量它们在 y 方向上的一致贡献。

将两个方向贡献相加：

\[
\boxed{
\mathbf a\cdot\mathbf b
=
a_1b_1+a_2b_2
}
\]

就得到整个二维空间中的综合结果。

推广到：

\[
n
\]

维：

\[
\boxed{
\mathbf a\cdot\mathbf b
=
\sum_{i=1}^{n}a_ib_i
}
\]

这就是标准欧氏空间中的 Dot Product，也叫：

> **Inner Product 的一种具体形式。**

---

# 3. 但“逐维相乘再相加”还没有解释几何意义

现在举三个例子。

---

## 情况 A：完全同方向

\[
\mathbf a=[1,0]
\]

\[
\mathbf b=[2,0]
\]

则：

\[
\mathbf a\cdot\mathbf b
=
1\times2+0\times0
=
2
\]

正数。

---

## 情况 B：垂直

\[
\mathbf a=[1,0]
\]

\[
\mathbf b=[0,2]
\]

则：

\[
\mathbf a\cdot\mathbf b
=
1\times0+0\times2
=
0
\]

---

## 情况 C：完全反方向

\[
\mathbf a=[1,0]
\]

\[
\mathbf b=[-2,0]
\]

则：

\[
\mathbf a\cdot\mathbf b
=
-2
\]

我们已经看见：

```text
同方向  → 正
垂直    → 0
反方向  → 负
```

这显然不像巧合。

真正原因就在：

\[
\boxed{
\mathbf a\cdot\mathbf b
=
\|\mathbf a\|
\|\mathbf b\|
\cos\theta
}
\]

---

# 4. 向量长度是什么？

向量：

\[
\mathbf a=[a_1,a_2]
\]

可以看成从原点：

\[
(0,0)
\]

指向：

\[
(a_1,a_2)
\]

的箭头。

根据勾股定理：

\[
\boxed{
\|\mathbf a\|
=
\sqrt{
a_1^2+a_2^2
}
}
\]

更一般地：

\[
\boxed{
\|\mathbf a\|
=
\sqrt{
\sum_i a_i^2
}
}
\]

注意：

\[
\mathbf a\cdot\mathbf a
=
\sum_i a_i^2
\]

因此：

\[
\boxed{
\mathbf a\cdot\mathbf a
=
\|\mathbf a\|^2
}
\]

所以一个向量和自己点积：

> 就是它长度的平方。

这是 Dot Product 最基础的性质之一。

---

# 5. 为什么 a·b 和夹角有关？

我们现在来真正推导：

\[
\mathbf a\cdot\mathbf b
=
\|\mathbf a\|
\|\mathbf b\|
\cos\theta
\]

考虑两个向量：

\[
\mathbf a
\]

和：

\[
\mathbf b
\]

它们从同一个原点出发，

夹角为：

\[
\theta
\]

它们的端点之间的向量是：

\[
\mathbf a-\mathbf b
\]

所以这条边长度：

\[
\|\mathbf a-\mathbf b\|
\]

---

# 6. 用余弦定理表达这条边

由余弦定理：

\[
\boxed{
\|\mathbf a-\mathbf b\|^2
=
\|\mathbf a\|^2
+
\|\mathbf b\|^2
-
2
\|\mathbf a\|
\|\mathbf b\|
\cos\theta
}
\]

这是纯几何结果。

---

# 7. 再用坐标展开同一个长度

另一方面：

\[
\|\mathbf a-\mathbf b\|^2
=
(\mathbf a-\mathbf b)
\cdot
(\mathbf a-\mathbf b)
\]

展开：

\[
=
\mathbf a\cdot\mathbf a
-
\mathbf a\cdot\mathbf b
-
\mathbf b\cdot\mathbf a
+
\mathbf b\cdot\mathbf b
\]

Dot Product 满足交换律：

\[
\mathbf a\cdot\mathbf b
=
\mathbf b\cdot\mathbf a
\]

所以：

\[
\|\mathbf a-\mathbf b\|^2
=
\|\mathbf a\|^2
+
\|\mathbf b\|^2
-
2\mathbf a\cdot\mathbf b
\]

---

# 8. 两个表达式描述的是同一个量

刚才几何余弦定理给：

\[
\|\mathbf a-\mathbf b\|^2
=
\|\mathbf a\|^2
+
\|\mathbf b\|^2
-
2\|\mathbf a\|\|\mathbf b\|\cos\theta
\]

坐标展开给：

\[
\|\mathbf a-\mathbf b\|^2
=
\|\mathbf a\|^2
+
\|\mathbf b\|^2
-
2\mathbf a\cdot\mathbf b
\]

两式相等。

去掉共同项：

\[
-2\mathbf a\cdot\mathbf b
=
-2\|\mathbf a\|\|\mathbf b\|\cos\theta
\]

所以：

\[
\boxed{
\mathbf a\cdot\mathbf b
=
\|\mathbf a\|
\|\mathbf b\|
\cos\theta
}
\]

这就是 Dot Product 的坐标定义与几何意义之间的桥梁。

---

# 9. 这个公式真正告诉了我们什么？

\[
\mathbf a\cdot\mathbf b
=
\|\mathbf a\|
\|\mathbf b\|
\cos\theta
\]

说明点积同时受三件事影响：

1. \(\mathbf a\) 的长度；
2. \(\mathbf b\) 的长度；
3. 两者夹角。

也就是说：

\[
\boxed{
\text{Dot Product}
=
\text{Magnitude Factor}
\times
\text{Directional Alignment}
}
\]

其中：

\[
\cos\theta
\]

负责方向关系。

---

# 10. 为什么同方向 Dot Product 最大？

如果：

\[
\theta=0
\]

则：

\[
\cos0=1
\]

所以：

\[
\boxed{
\mathbf a\cdot\mathbf b
=
\|\mathbf a\|
\|\mathbf b\|
}
\]

在两个向量长度固定的情况下，

这是能取得的最大点积。

所以：

> 同方向意味着最强正 alignment。

---

# 11. 为什么垂直时 Dot Product = 0？

如果：

\[
\theta=90^\circ
\]

则：

\[
\cos90^\circ=0
\]

所以：

\[
\boxed{
\mathbf a\cdot\mathbf b=0
}
\]

这就是为什么在欧氏空间中：

\[
\mathbf a\perp\mathbf b
\]

等价于：

\[
\mathbf a\cdot\mathbf b=0
\]

对于非零向量成立。

这种关系叫：

> **Orthogonality**

也就是正交。

---

# 12. 为什么反方向时是负数？

如果：

\[
\theta=180^\circ
\]

则：

\[
\cos180^\circ=-1
\]

所以：

\[
\boxed{
\mathbf a\cdot\mathbf b
=
-\|\mathbf a\|\|\mathbf b\|
}
\]

这就是固定长度下最小的点积。

因此：

```text
夹角 < 90°
→ cosθ > 0
→ dot product > 0

夹角 = 90°
→ cosθ = 0
→ dot product = 0

夹角 > 90°
→ cosθ < 0
→ dot product < 0
```

---

# 13. 所以 Dot Product 可以衡量“方向一致程度”

对于固定长度的两个向量：

\[
\|\mathbf a\|,\|\mathbf b\|
\]

不变，

点积大小完全由：

\[
\cos\theta
\]

决定。

所以可以说：

> **Dot Product 能反映两个向量的 directional alignment。**

但这里有一个关键限定：

\[
\boxed{
\text{fixed magnitude}
}
\]

因为实际点积还会受到向量长度影响。

---

# 14. Dot Product 不是纯粹的“角度相似度”

例如：

\[
\mathbf a=[1,0]
\]

\[
\mathbf b=[1,0]
\]

点积：

\[
1
\]

但：

\[
\mathbf c=[100,0]
\]

与：

\[
\mathbf b=[1,0]
\]

完全同方向，

点积：

\[
100
\]

两组夹角都是：

\[
0^\circ
\]

但点积差了 100 倍。

所以：

> Dot Product 不仅看方向，也看 magnitude。

这正是它和 Cosine Similarity 最大的区别。

---

# 15. Cosine Similarity 是什么？

从：

\[
\mathbf a\cdot\mathbf b
=
\|\mathbf a\|
\|\mathbf b\|
\cos\theta
\]

整理：

\[
\boxed{
\cos\theta
=
\frac{
\mathbf a\cdot\mathbf b
}{
\|\mathbf a\|\|\mathbf b\|
}
}
\]

这就是：

> **Cosine Similarity**

对于非零向量：

\[
\boxed{
\operatorname{cos\_sim}(\mathbf a,\mathbf b)
=
\frac{
\mathbf a^\top\mathbf b
}{
\|\mathbf a\|\|\mathbf b\|
}
}
\]

它把两个向量先按长度归一化。

因此只看：

> 方向。

---

# 16. Dot Product vs Cosine Similarity

可以直接比较。

## Dot Product

\[
\mathbf a^\top\mathbf b
\]

依赖：

- direction；
- magnitude。

---

## Cosine Similarity

\[
\frac{
\mathbf a^\top\mathbf b
}{
\|\mathbf a\|\|\mathbf b\|
}
\]

只看：

- direction。

所以：

\[
\boxed{
\text{Dot Product}
=
\text{Cosine Similarity}
\times
\|\mathbf a\|
\|\mathbf b\|
}
\]

---

# 17. 一个具体例子

令：

\[
\mathbf q=[1,0]
\]

两个 keys：

\[
\mathbf k_1=[1,0]
\]

\[
\mathbf k_2=[10,0]
\]

两者都完全和：

\[
\mathbf q
\]

同方向。

Cosine Similarity：

\[
\cos(q,k_1)=1
\]

\[
\cos(q,k_2)=1
\]

但 Dot Product：

\[
q^\top k_1=1
\]

\[
q^\top k_2=10
\]

所以 Dot Product 会明显偏向：

\[
k_2
\]

这说明：

> **向量 norm 也会影响 attention score。**

---

# 18. 这是不是 Dot Product 的缺点？

不一定。

如果我们的目标确实只想衡量：

> 纯方向相似度，

那么 cosine similarity 更直接。

但 Transformer 并没有强制：

> Query / Key norm 必须被删除。

因为 learned model 可以利用：

\[
\|q\|
\]

和：

\[
\|k\|
\]

携带信息。

所以 dot product 保留 magnitude 是：

> 一种模型自由度。

它不天然比 cosine 更正确或错误。

---

# 19. 为什么 Transformer 用 Dot Product，而不是 Cosine Similarity？

《Attention Is All You Need》并没有论证：

> dot product 是唯一理论最优 compatibility function。

原论文主要比较：

- additive attention；
- dot-product attention。

并指出 dot product 可以通过高度优化的矩阵乘法高效实现。

所有 queries 和 keys 的 pairwise scores：

\[
\boxed{
QK^\top
}
\]

可以一次 GEMM 完成。

这对 GPU/TPU 非常友好。

再加：

\[
1/\sqrt{d_k}
\]

控制 score scale。

所以 Scaled Dot-Product Attention 是：

> **高效、简单、可学习且实验有效的设计。**

不是数学上唯一可能的 attention score。

---

# 20. Dot Product 还有另一种几何理解：Projection

这是理解 Attention 匹配特别有用的视角。

考虑：

\[
\mathbf a
\]

在：

\[
\mathbf b
\]

方向上的标量投影。

单位向量：

\[
\hat{\mathbf b}
=
\frac{
\mathbf b
}{
\|\mathbf b\|
}
\]

那么：

\[
\mathbf a
\]

在 \(\mathbf b\) 方向上的 scalar projection 是：

\[
\boxed{
\mathbf a\cdot\hat{\mathbf b}
}
\]

展开：

\[
=
\|\mathbf a\|\cos\theta
\]

所以：

\[
\boxed{
\mathbf a\cdot\mathbf b
=
\|\mathbf b\|
\times
\operatorname{comp}_{\mathbf b}(\mathbf a)
}
\]

也就是说：

> Dot Product 可以理解为“一个向量在另一个方向上的投影量，再乘另一个向量长度”。

---

# 21. 为什么 Projection 视角很有用？

假设：

\[
\mathbf q
\]

是 Query。

某个 Key：

\[
\mathbf k
\]

如果和 Query 方向高度一致，

那么：

\[
q
\]

在 \(k\) 方向上的投影就大。

于是：

\[
q^\top k
\]

也大。

所以从几何上：

> 高 dot-product compatibility 表示两个 learned representations 在 matching space 中有较强方向一致性，同时还受两者长度调制。

---

# 22. 但 Attention 的 Matching Space 不是物理空间

必须强调。

Q/K 向量中的坐标：

\[
q_1,q_2,\ldots,q_{d_k}
\]

通常没有：

```text
x 轴
y 轴
z 轴
```

这种人类可见空间意义。

它们是：

> learned feature dimensions。

所以夹角：

\[
\theta
\]

也不是物理世界角度。

它是：

> learned representation space 中的几何关系。

---

# 23. 为什么神经网络能让“该匹配的向量”方向变得合适？

因为：

\[
q=xW_Q
\]

\[
k=xW_K
\]

其中：

\[
W_Q,W_K
\]

可训练。

如果某个 Query 应该关注某个 Key，

但目前：

\[
q^\top k
\]

太小，

最终 task loss 会通过 gradient 调整：

\[
W_Q,W_K
\]

从而改变：

- q 的方向；
- q 的长度；
- k 的方向；
- k 的长度。

长期训练后，matching geometry 逐渐形成。

详细机制见：

- [Query / Key / Value](./qkv.md)

---

# 24. 一个简单二维“学习匹配空间”的例子

假设某个 Query 当前：

\[
q=[1,0]
\]

应该匹配 Key A：

\[
k_A=[0,1]
\]

但现在：

\[
q^\top k_A=0
\]

完全正交。

如果任务 loss 希望两者发生更强 attention，

训练可以改变：

\[
W_Q
\]

或：

\[
W_K
\]

让未来表示变成例如：

\[
q'=[1,1]
\]

\[
k_A'=[1,1]
\]

于是：

\[
q'^\top k_A'=2
\]

compatibility 提高。

这里并不是模型“理解了角度定理”。

只是 gradient 在参数空间中推动 representation geometry 朝有利方向变化。

---

# 25. Dot Product 为什么是双线性的？

点积满足：

\[
(\alpha \mathbf a+\beta\mathbf b)\cdot\mathbf c
=
\alpha(\mathbf a\cdot\mathbf c)
+
\beta(\mathbf b\cdot\mathbf c)
\]

同样对第二个变量也线性。

这种性质叫：

> **Bilinearity**

它意味着 Dot Product 和 Linear Projection 配合非常自然。

因为：

\[
Q=XW_Q
\]

\[
K=XW_K
\]

之后：

\[
QK^\top
\]

整个过程仍然可以高效用矩阵代数表示。

---

# 26. Dot Product 为什么和矩阵乘法天然连接？

设：

\[
Q=
\begin{bmatrix}
q_1^\top\\
q_2^\top\\
\vdots\\
q_n^\top
\end{bmatrix}
\]

\[
K=
\begin{bmatrix}
k_1^\top\\
k_2^\top\\
\vdots\\
k_m^\top
\end{bmatrix}
\]

那么：

\[
K^\top
\]

的第 \(j\) 列正是：

\[
k_j
\]

所以：

\[
(QK^\top)_{ij}
=
q_i^\top k_j
\]

于是一个矩阵乘法：

\[
\boxed{
QK^\top
}
\]

就同时计算所有：

\[
n\times m
\]

个 Query–Key dot products。

这正是 Transformer 高效实现 Attention 的关键。

---

# 27. Shape Example

假设：

\[
Q\in\mathbb R^{4\times3}
\]

也就是：

- 4 个 queries；
- 每个 query 3 维。

\[
K\in\mathbb R^{6\times3}
\]

也就是：

- 6 个 keys；
- 每个 key 3 维。

那么：

\[
K^\top
\in
\mathbb R^{3\times6}
\]

所以：

\[
\boxed{
QK^\top
\in
\mathbb R^{4\times6}
}
\]

每个：

\[
(i,j)
\]

元素都是：

\[
q_i^\top k_j
\]

即：

> 第 \(i\) 个 query 和第 \(j\) 个 key 的 compatibility score。

---

# 28. 为什么 Q 和 K 维度必须一样？

因为：

\[
q^\top k
\]

需要逐维配对：

\[
q_1k_1+q_2k_2+\cdots
\]

所以两者必须存在相同：

\[
d_k
\]

matching dimension。

这就是为什么 Transformer 原论文说：

> queries 和 keys 的维度是 \(d_k\)。

Value：

\[
v
\]

则可以有另一个维度：

\[
d_v
\]

因为它不参与 QK dot product。

---

# 29. 为什么 Value 不要求 d_v = d_k？

Attention weights：

\[
A
\in\mathbb R^{n_q\times n_k}
\]

只需要和：

\[
V\in\mathbb R^{n_k\times d_v}
\]

相乘。

得到：

\[
O\in\mathbb R^{n_q\times d_v}
\]

因此：

\[
d_v
\]

可以和：

\[
d_k
\]

不同。

原始 Transformer 每个 head 恰好使用：

\[
d_k=d_v=64
\]

但这是 architecture choice，

不是 Attention 的数学必然要求。

---

# 30. 为什么 Dot Product 可能随 Dimension 变大？

假设：

\[
q_i,k_i
\]

各维：

- independent；
- mean 0；
- variance 1。

点积：

\[
q^\top k
=
\sum_{r=1}^{d_k}q_rk_r
\]

每一项：

\[
q_rk_r
\]

均值大致：

\[
0
\]

方差大致：

\[
1
\]

独立相加后：

\[
\operatorname{Var}(q^\top k)
=
d_k
\]

因此标准差：

\[
\boxed{
\sqrt{d_k}
}
\]

这就是为什么 dimension 越大，

raw dot-product logits 典型 magnitude 会越大。

---

# 31. 为什么 Attention 除以 \(\sqrt{d_k}\) 而不是 \(d_k\)？

如果 raw dot product 的 variance：

\[
d_k
\]

那么除以：

\[
\sqrt{d_k}
\]

之后：

\[
\operatorname{Var}
\left(
\frac{
q^\top k
}{
\sqrt{d_k}
}
\right)
=
\frac{
d_k
}{
d_k
}
=
1
\]

大致恢复到：

> dimension-independent 的常数量级。

如果直接除：

\[
d_k
\]

则 variance 会变成：

\[
1/d_k
\]

可能又压得过小。

所以：

\[
\boxed{
1/\sqrt{d_k}
}
\]

和方差尺度正好匹配。

---

# 32. 一个具体 Attention Score 例子

假设：

\[
d_k=4
\]

Query：

\[
q=[1,1,0,0]
\]

三个 Keys：

\[
k_1=[1,1,0,0]
\]

\[
k_2=[1,-1,0,0]
\]

\[
k_3=[0,0,1,1]
\]

则：

\[
q^\top k_1=2
\]

\[
q^\top k_2=0
\]

\[
q^\top k_3=0
\]

说明在这个 matching space 中：

\[
k_1
\]

最匹配。

Scale：

\[
\sqrt{d_k}=2
\]

所以 logits：

\[
[1,0,0]
\]

Softmax：

\[
\approx
[0.576,0.212,0.212]
\]

然后这些权重才用于 Values。

---

# 33. 为什么点积为 0 不代表“两个 Token 完全无关”？

非常重要。

在某一个 head、某一个 layer 的 learned Q/K space 中：

\[
q^\top k=0
\]

只表示：

> 当前这两个 matching vectors 正交。

它不意味着：

- 两个原 token 没语义关系；
- 其他 attention head 也不关注；
- 其他 layer 也不关注；
- 整个模型认为它们无关。

所以 Dot Product score 是：

> 局部计算图中的 compatibility。

不是全局语义判决。

---

# 34. 为什么负 Dot Product 也不等于“敌对语义”？

如果：

\[
q^\top k<0
\]

只表示它们在当前 matching space 中形成钝角关系。

经过 Softmax：

\[
e^{q^\top k}
\]

仍然是正数。

所以这个 key 仍可能获得非零 weight。

负 score 的意义只是：

> 相对于其他更高 score 的 keys，当前匹配较弱。

不是自然语言上的“反义词”。

---

# 35. Dot Product 是无界的吗？

是。

如果把：

\[
q
\]

放大：

\[
q'=100q
\]

那么：

\[
q'^\top k
=
100(q^\top k)
\]

所以 Dot Product 没有固定范围。

它可以：

\[
-\infty
\]

到：

\[
+\infty
\]

理论上无界。

而 Cosine Similarity：

\[
\in[-1,1]
\]

这又是二者的重要区别。

---

# 36. 为什么 Attention 允许无界 Score？

因为 raw score 后面还有：

\[
\operatorname{softmax}
\]

将它们转换成：

\[
[0,1]
\]

范围内归一化权重。

所以 raw logits 本来就不需要是概率。

事实上很多机器学习模型都喜欢：

> 先产生无界 logits，再 Softmax。

---

# 37. Dot Product 和 Euclidean Distance 有什么关系？

对两个向量：

\[
\|\mathbf a-\mathbf b\|^2
\]

展开：

\[
=
\|\mathbf a\|^2
+
\|\mathbf b\|^2
-
2\mathbf a^\top\mathbf b
\]

所以：

\[
\boxed{
\mathbf a^\top\mathbf b
=
\frac{
\|\mathbf a\|^2
+
\|\mathbf b\|^2
-
\|\mathbf a-\mathbf b\|^2
}{2}
}
\]

这说明：

> Dot Product、向量长度和 Euclidean Distance 并不是彼此无关的概念。

如果：

\[
\|\mathbf a\|
\]

和：

\[
\|\mathbf b\|
\]

固定，

那么：

> Dot Product 越大 \(\Leftrightarrow\) Euclidean Distance 越小。

---

# 38. 但如果 Norm 不固定，这个等价就不存在

例如：

\[
a=[1,0]
\]

\[
b=[1,0]
\]

距离：

\[
0
\]

dot：

\[
1
\]

而：

\[
c=[100,0]
\]

和：

\[
a
\]

方向一致，

dot：

\[
100
\]

但距离：

\[
99
\]

非常大。

所以不能简单说：

> “Dot Product 大 = Euclidean Distance 小。”

只有在 norm 被固定或控制时才更接近这种关系。

---

# 39. 如果向量都归一化成 Unit Vector，会发生什么？

如果：

\[
\|\mathbf a\|=\|\mathbf b\|=1
\]

那么：

\[
\boxed{
\mathbf a^\top\mathbf b
=
\cos\theta
}
\]

此时：

> Dot Product = Cosine Similarity。

并且：

\[
\|\mathbf a-\mathbf b\|^2
=
2-2\mathbf a^\top\mathbf b
\]

所以在单位球面上：

- dot product；
- cosine similarity；
- Euclidean distance；

彼此可以直接转换。

---

# 40. Transformer 为什么不先把 Q/K Normalize 成 Unit Vector？

原始 Transformer 没这么做。

它让：

\[
q,k
\]

的 norm 也可以参与 compatibility。

这意味着模型可以通过：

- direction；
- magnitude；

共同控制 attention logits。

是否使用 Q/K normalization 是后续架构可以另外设计的选择。

原始 Scaled Dot-Product Attention 采用：

\[
\frac{q^\top k}{\sqrt{d_k}}
\]

而不是 cosine similarity。

---

# 41. Dot Product 的“相似度”解释必须加限定

所以更严谨的说法不是：

> “Dot Product 就是相似度。”

而是：

> **Dot Product 是一种结合向量方向一致性与 magnitude 的双线性 compatibility measure；当向量 norm 固定时，它与 cosine similarity 单调等价。**

在 Attention 中：

> 由于 Q/K 是 learned projections，模型会学习一个让 dot product 对任务关系有用的 matching space。

这个说法更准确。

---

# 42. Dot Product 和 Projection 的关系再推一步

向量：

\[
\mathbf a
\]

在：

\[
\mathbf b
\]

方向上的向量投影：

\[
\operatorname{proj}_{\mathbf b}(\mathbf a)
=
\frac{
\mathbf a\cdot\mathbf b
}{
\|\mathbf b\|^2
}
\mathbf b
\]

为什么？

先求 scalar projection：

\[
\operatorname{comp}_{\mathbf b}(\mathbf a)
=
\frac{
\mathbf a\cdot\mathbf b
}{
\|\mathbf b\|
}
\]

再乘 \(\mathbf b\) 的 unit vector：

\[
\frac{
\mathbf b
}{
\|\mathbf b\|
}
\]

得到：

\[
\boxed{
\operatorname{proj}_{\mathbf b}(\mathbf a)
=
\frac{
\mathbf a\cdot\mathbf b
}{
\|\mathbf b\|^2
}
\mathbf b
}
\]

所以 Dot Product 直接决定：

> 一个向量沿另一个方向到底有多少成分。

---

# 43. 为什么正交特别重要？

如果：

\[
\mathbf a\cdot\mathbf b=0
\]

那么：

\[
\mathbf a
\]

在：

\[
\mathbf b
\]

方向的 projection：

\[
0
\]

也就是：

> \(\mathbf a\) 在 \(\mathbf b\) 方向上没有任何分量。

这就是正交在几何上的真正含义。

所以 Dot Product 不只是一个“相似度函数”。

它还是：

> 长度、角度、正交、投影

这些欧氏几何概念的核心连接器。

---

# 44. 为什么“逐维乘积再求和”恰好具有这些漂亮性质？

在标准笛卡尔坐标系里：

\[
\mathbf e_1,\ldots,\mathbf e_n
\]

是一组 orthonormal basis。

即：

\[
\mathbf e_i\cdot\mathbf e_j
=
\begin{cases}
1,&i=j\\
0,&i\neq j
\end{cases}
\]

任意：

\[
\mathbf a
=
\sum_i a_i\mathbf e_i
\]

\[
\mathbf b
=
\sum_j b_j\mathbf e_j
\]

利用双线性：

\[
\mathbf a\cdot\mathbf b
=
\sum_i\sum_j
a_ib_j
(
\mathbf e_i\cdot\mathbf e_j
)
\]

因为不同 basis directions 正交：

\[
i\neq j
\Rightarrow
\mathbf e_i\cdot\mathbf e_j=0
\]

只剩：

\[
\boxed{
\sum_i a_ib_i
}
\]

所以“逐维乘积再求和”并不是随便发明的规则。

它正是：

> 在 orthonormal coordinates 中计算 Euclidean inner product 的自然表达。

---

# 45. Dot Product 和 Inner Product 是一回事吗？

在：

\[
\mathbb R^n
\]

标准坐标中，

我们通常把：

\[
\mathbf a^\top\mathbf b
\]

叫：

- dot product；
- standard inner product。

但更一般的 Inner Product 可以定义成其他形式。

例如：

\[
\langle a,b\rangle_M
=
a^\top M b
\]

其中：

\[
M
\]

满足适当的正定条件。

所以：

> Dot Product 是标准欧氏空间最常用的一种 inner product。

这也是为什么数学教材有时使用：

\[
\langle a,b\rangle
\]

而 AI 代码里更常看到：

\[
a^\top b
\]

---

# 46. 为什么 Attention 不直接学习一个 Bilinear Matrix？

其实可以。

例如 compatibility：

\[
q^\top Wk
\]

就是一种 bilinear score。

但 Transformer 已经先做：

\[
q=xW_Q
\]

\[
k=xW_K
\]

所以：

\[
q^\top k
\]

本身已经包含 learned projection。

展开：

\[
(x_iW_Q)(x_jW_K)^\top
\]

等价于：

\[
x_i
W_QW_K^\top
x_j^\top
\]

因此 Q/K projection + dot product 已经构成了一种：

> learned bilinear interaction。

---

# 47. 这一点非常重要

有时会觉得：

> “Dot Product 太简单，怎么可能学复杂关系？”

但实际上 score：

\[
q_i^\top k_j
\]

里面：

\[
q_i=x_iW_Q
\]

\[
k_j=x_jW_K
\]

所以：

\[
\boxed{
q_i^\top k_j
=
x_iW_QW_K^\top x_j^\top
}
\]

真正 learnable 的并不只是：

> 一个裸点积。

前面还有两套高维 learned transformations。

因此模型可以学习非常复杂的 matching geometry。

---

# 48. Multi-Head 又进一步增强这种自由度

第 \(h\) 个 head：

\[
q_i^{(h)}
=
x_iW_Q^{(h)}
\]

\[
k_j^{(h)}
=
x_jW_K^{(h)}
\]

所以：

\[
score_{ij}^{(h)}
=
x_i
W_Q^{(h)}
W_K^{(h)\top}
x_j^\top
\]

不同 heads 有不同：

\[
W_Q^{(h)}W_K^{(h)\top}
\]

也就是不同 matching geometry。

因此 Multi-Head Attention 可以同时学习多种不同关系。

---

# 49. 一个 ACT 例子

ACT policy decoder 中某个 future action query representation：

\[
x_{\text{action}}
\]

经过：

\[
W_Q
\]

得到：

\[
q
\]

observation memory 中一个 wrist-camera token：

\[
m_{\text{wrist}}
\]

经过：

\[
W_K
\]

得到：

\[
k_{\text{wrist}}
\]

score：

\[
\frac{
q^\top k_{\text{wrist}}
}{
\sqrt{d_k}
}
\]

如果这个 score 相对其他 memory tokens 高，

Softmax 后：

> wrist-camera token 会获得更大的 attention weight。

然后真正传入 decoder 的内容来自：

\[
v_{\text{wrist}}
=
m_{\text{wrist}}W_V
\]

所以：

```text
Dot Product
决定“这个 memory token 和当前 action query 有多匹配”

Value
决定“如果匹配，它到底传递什么”
```

---

# 50. 为什么 ACT 里的 Dot Product 不代表“两个物理向量方向相似”？

因为：

\[
q
\]

和：

\[
k
\]

不是：

- 机械臂位置向量；
- 三维空间坐标；
- 速度方向。

它们是：

> learned hidden features。

所以：

\[
q^\top k
\]

是在：

> neural representation space

中做几何匹配。

这也是深度学习里“向量空间”的一个重要观念：

> 几何关系可以存在于抽象 feature space，而不只存在于真实物理空间。

---

# 51. 为什么 Word Embedding 也常用 Dot Product / Cosine？

因为 embedding 训练往往会让：

> 有某种统计或语义关系的对象在 representation space 中形成可利用的几何结构。

例如：

\[
w_1^\top w_2
\]

或 cosine similarity 可以作为一种 relation measure。

但再次强调：

> Attention 中 Q/K space 是专门为当前 layer/head 的匹配任务学习的，不应直接等同于静态 word embedding semantic space。

---

# 52. 点积可以为 0，但两个向量都不为 0

例如：

\[
a=[1,1]
\]

\[
b=[1,-1]
\]

则：

\[
a^\top b
=
1-1
=
0
\]

但：

\[
a\neq0
\]

\[
b\neq0
\]

这说明：

> “点积为 0”不意味着某个向量没有信息。

它表示：

> 两个向量在当前 Euclidean inner-product geometry 下正交。

---

# 53. 零向量是一个特殊情况

如果：

\[
a=0
\]

则对任意：

\[
b
\]

都有：

\[
a^\top b=0
\]

但零向量没有定义良好的方向，

因为：

\[
\|a\|=0
\]

所以：

\[
\frac{
a^\top b
}{
\|a\|\|b\|
}
\]

无法计算。

因此 cosine similarity 对零向量没有定义。

Dot Product 则仍然定义为：

\[
0
\]

---

# 54. Cauchy–Schwarz Inequality

点积满足一个非常重要的不等式：

\[
\boxed{
|\mathbf a\cdot\mathbf b|
\le
\|\mathbf a\|
\|\mathbf b\|
}
\]

这叫：

> **Cauchy–Schwarz Inequality**

结合：

\[
\mathbf a\cdot\mathbf b
=
\|\mathbf a\|
\|\mathbf b\|
\cos\theta
\]

就对应：

\[
|\cos\theta|\le1
\]

等号成立当且仅当两个非零向量线性相关，也就是：

> 同方向或反方向。

---

# 55. 为什么这个不等式对 Cosine Similarity 很重要？

因为：

\[
\frac{
\mathbf a\cdot\mathbf b
}{
\|\mathbf a\|\|\mathbf b\|
}
\]

由 Cauchy–Schwarz 保证落在：

\[
[-1,1]
\]

所以 cosine similarity 有清晰固定范围：

\[
-1
\]

完全反方向，

\[
0
\]

正交，

\[
1
\]

完全同方向。

---

# 56. Attention 为什么不把 Dot Product 直接解释成概率？

因为：

\[
q^\top k
\]

可以是：

- 负数；
- 大于 1；
- 任意实数。

它只是：

> logit / compatibility score。

然后 Softmax：

\[
\boxed{
\alpha_j
=
\frac{
\exp(s_j)
}{
\sum_r\exp(s_r)
}
}
\]

才把这些 scores 转成 normalized weights。

所以在 Attention 中：

```text
Dot Product
→ raw compatibility

Softmax
→ normalized attention weights
```

二者必须区分。

---

# 57. 为什么 Dot Product 大不等于 Weight 一定大很多？

因为 Attention weight 取决于：

> 相对 scores。

例如：

\[
[100,99,98]
\]

与：

\[
[2,1,0]
\]

虽然第一组绝对值大很多，

它们之间的差：

\[
[0,-1,-2]
\]

是相同的。

由于 Softmax 对统一加常数不变：

\[
\operatorname{softmax}([100,99,98])
=
\operatorname{softmax}([2,1,0])
\]

所以 Attention 最终更关心：

> score differences。

---

# 58. 为什么 Scale 会影响 Weight Sharpness？

如果 scores：

\[
[2,1,0]
\]

Softmax 大约：

\[
[0.665,0.245,0.090]
\]

如果全部乘 10：

\[
[20,10,0]
\]

Softmax 会极度偏向第一个。

所以 Dot Product 的整体 scale 会改变：

> attention distribution 的尖锐程度。

这正是：

\[
1/\sqrt{d_k}
\]

非常重要的原因。

---

# 59. Dot Product 和 Linear Regression 里的 xᵀw 有关系吗？

有。

线性模型：

\[
y=w^\top x+b
\]

里面：

\[
w^\top x
\]

本质也是 Dot Product。

可以理解为：

> 输入 \(x\) 在权重向量 \(w\) 所定义方向上的响应。

所以 Dot Product 并不是 Transformer 特殊发明。

它是线性代数和机器学习里最基础的操作之一。

Attention 只是把它用于：

> Query–Key compatibility。

---

# 60. 神经网络里的 Logit 也常是 Dot Product

分类模型最后可能有：

\[
z_c=w_c^\top h+b_c
\]

每个 class：

\[
c
\]

有一个 weight vector：

\[
w_c
\]

hidden representation：

\[
h
\]

与哪个 class vector 点积更大，

对应 logit 往往更大。

从这个角度看：

> Attention 的 Query–Key matching 和 Linear classifier 都在利用 learned vector geometry。

---

# 61. 为什么这对理解 AI 很重要？

现代深度学习中大量“语义”并不是以：

```text
if object == cat:
    ...
```

这种显式规则存储。

而是被编码进高维 representation geometry：

- direction；
- subspace；
- distance；
- dot product；
- norm。

所以理解 Dot Product，

其实是在理解一个非常普遍的 AI 模式：

> **把抽象关系转换成向量几何，再用简单矩阵运算处理。**

---

# 62. Dot Product 的五个核心视角

可以把它从五个角度同时理解。

---

## 视角 1：坐标计算

\[
\boxed{
a^\top b
=
\sum_i a_ib_i
}
\]

---

## 视角 2：长度与夹角

\[
\boxed{
a^\top b
=
\|a\|\|b\|\cos\theta
}
\]

---

## 视角 3：Projection

\[
\boxed{
\operatorname{comp}_b(a)
=
\frac{
a^\top b
}{
\|b\|
}
}
\]

---

## 视角 4：Orthogonality

\[
\boxed{
a^\top b=0
\iff
a\perp b
}
\]

对非零欧氏向量。

---

## 视角 5：Learned Compatibility

在 Attention：

\[
\boxed{
score(q,k)
=
q^\top k
}
\]

Q/K 的 learned projection 让这种几何关系适配最终任务。

---

# 63. 常见误解一：Dot Product 就是两个向量的长度相乘

**错误。**

还有：

\[
\cos\theta
\]

因子。

只有同方向时：

\[
a^\top b
=
\|a\|\|b\|
\]

---

# 64. 常见误解二：Dot Product 大就表示夹角一定小

**不一定。**

因为 magnitude 也影响结果。

一个 norm 非常大的向量即使夹角更差，

仍可能得到更大的 Dot Product。

如果只想比较角度，

应该看：

> cosine similarity。

---

# 65. 常见误解三：Dot Product = Cosine Similarity

**错误。**

除非：

\[
\|a\|=\|b\|=1
\]

或者先做 L2 normalization。

---

# 66. 常见误解四：Dot Product 为 0 说明两个向量之一是 0

**错误。**

两个非零正交向量也有：

\[
a^\top b=0
\]

---

# 67. 常见误解五：负 Dot Product 表示“两个 Token 语义相反”

**错误。**

只表示：

> 当前 learned matching space 中方向关系产生负 compatibility。

---

# 68. 常见误解六：Transformer 用 Dot Product 是因为它天然理解语义

**错误。**

真正的语义 / task relation 是：

\[
W_Q,W_K
\]

通过训练把 hidden representations 映射到合适 matching geometry 后形成的。

---

# 69. 常见误解七：除 \(\sqrt{d_k}\) 是为了 Normalize 向量长度到 1

**错误。**

它不是 L2 normalization。

它只是整体 scale logits：

\[
q^\top k
\rightarrow
\frac{
q^\top k
}{
\sqrt{d_k}
}
\]

Q/K norm 仍然可以不同。

---

# 70. 常见误解八：Scale 后就变成 Cosine Similarity

**错误。**

Cosine：

\[
\frac{
q^\top k
}{
\|q\|\|k\|
}
\]

Scaled Dot Product：

\[
\frac{
q^\top k
}{
\sqrt{d_k}
}
\]

分母完全不同。

---

# 71. 常见误解九：QKᵀ 每一行就是概率

**错误。**

它只是 logits。

必须经过：

\[
Softmax
\]

才得到 normalized attention weights。

---

# 72. 常见误解十：Dot Product 大 = 两个原始 Token 很相似

**错误。**

Attention 比较的是：

\[
q_i=x_iW_Q
\]

和：

\[
k_j=x_jW_K
\]

不是直接比较原始 tokens。

而且它学习的是：

> task-specific compatibility。

---

# 73. 用三个例子记住方向关系

假设：

\[
a=[1,0]
\]

---

### 同方向

\[
b=[2,0]
\]

\[
a^\top b=2
\]

---

### 正交

\[
b=[0,2]
\]

\[
a^\top b=0
\]

---

### 反方向

\[
b=[-2,0]
\]

\[
a^\top b=-2
\]

对应：

\[
\cos0^\circ=1
\]

\[
\cos90^\circ=0
\]

\[
\cos180^\circ=-1
\]

---

# 74. 用一个公式记住 Dot Product 的真正意义

如果最后只记住：

\[
\boxed{
\mathbf a^\top\mathbf b
=
\|\mathbf a\|
\|\mathbf b\|
\cos\theta
}
\]

那么看到：

\[
QK^\top
\]

时就应该想到：

> Transformer 正在 learned matching space 中，对每个 Query–Key pair 计算一个同时受方向 alignment 和 vector magnitude 影响的 compatibility score。

然后：

\[
1/\sqrt{d_k}
\]

控制 score 的 dimension-dependent scale。

Softmax：

> 将相对 compatibility 转成读取权重。

最后：

\[
V
\]

提供真正被读取的内容。

---

# 75. 一句话真正理解 Dot Product

> **Dot Product 是标准欧氏空间中的内积：坐标上它等于逐维乘积之和，几何上它等于两个向量长度乘以夹角余弦，因此同时反映 magnitude 与 directional alignment；在 Transformer 中，\(W_Q\) 和 \(W_K\) 会把 hidden states 学习性地投影到一个 task-specific matching space，使 \(q^\top k\) 成为高效可训练的 compatibility score，而不是天然存在的“语义相似度”。**

---

# 76. 下一步：Softmax 为什么能把 Score 变成 Attention Weight？

现在我们已经理解：

\[
QK^\top
\]

产生的是什么：

> raw compatibility scores。

但仍然有下一个问题：

> 为什么不能直接拿这些 scores 乘 Value？

为什么一定要：

\[
\operatorname{softmax}
\left(
\frac{
QK^\top
}{
\sqrt{d_k}
}
\right)
\]

？

下一篇：

> **[Softmax：为什么指数归一化能把分数变成概率式权重？](./softmax.md)**

会真正解释：

- 为什么先取 exponential；
- 为什么负 logits 也能得到正权重；
- 为什么所有权重和为 1；
- 为什么 Softmax 只关心相对差值；
- 为什么统一加一个常数结果不变；
- temperature 怎样改变 distribution sharpness；
- 为什么 logits 很大时 gradient 会变小；
- Scaled Dot-Product Attention 为什么必须在 Softmax 前控制 scale；
- Softmax 与 Sigmoid 到底有什么区别。

---

## Mathematics Sources

### MIT OpenCourseWare

MIT OpenCourseWare 的 Linear Algebra / Dot Products 材料将 Dot Product 作为欧氏向量几何中的基础结构，并用于定义：

- length；
- orthogonality；
- projection；
- angle-related relationships。

参考：

- MIT OpenCourseWare, **Linear Algebra**, Prof. Gilbert Strang
- MIT OCW, **Dot Products**
- MIT OCW, **Orthogonal Vectors and Subspaces**

课程主页：

https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/

MIT 18.06 的推荐教材：

Gilbert Strang,  
**Introduction to Linear Algebra.**

本文数学部分采用标准欧氏空间定义：

\[
a^\top b
=
\sum_i a_ib_i
\]

并通过余弦定理推导：

\[
a^\top b
=
\|a\|\|b\|\cos\theta
\]

---

## AI Primary Source

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,  
Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin.

**Attention Is All You Need.**  
NIPS 2017.

- arXiv: https://arxiv.org/abs/1706.03762
- PDF: https://arxiv.org/pdf/1706.03762

Section 3.2.1 定义 Scaled Dot-Product Attention：

\[
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
\]

原论文明确说明：

- queries 和 keys 的维度为 \(d_k\)；
- values 维度为 \(d_v\)；
- Query 与所有 Keys 做 dot products；
- 结果除以 \(\sqrt{d_k}\)；
- 再经 Softmax 得到 Values 的权重；
- scaling 用于缓解大 \(d_k\) 时 dot-product magnitude 过大导致 Softmax 进入小梯度区域的问题。

---

## 本文知识连接

### 数学前置

- Vector
- Vector Norm
- Pythagorean Theorem
- Law of Cosines
- Angle

### 数学延伸

- Cosine Similarity
- Orthogonality
- Vector Projection
- Inner Product
- Cauchy–Schwarz Inequality
- Matrix Multiplication

### Deep Learning

- [Attention](./attention.md)
- [Query / Key / Value](./qkv.md)
- [Transformer](./transformer.md)
- [Softmax](./softmax.md)
- [Scaled Dot-Product Attention](./attention.md)
- [Multi-Head Attention](./multi-head-attention.md)

### Robot Learning

- [ACT Architecture](../robot-learning/act/architecture.md)

### 下一步

- [Softmax](./softmax.md)
