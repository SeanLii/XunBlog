---
title: "Self-Attention：一个 Token 到底怎样读取整条 Sequence？"
description: "把 QKV、Dot Product、Scaling、Softmax 与 Weighted Sum 重新组装起来，用一个小型矩阵例子完整计算 Self-Attention，并解释 permutation equivariance、mask、residual 与 ACT 中的 1202-token 融合。"
status: reviewed
pageType: concept
canonical: /deep-learning/self-attention
updated: "2026-09-15"
---

# Self-Attention：一个 Token 到底怎样读取整条 Sequence？

前面我们已经把 Attention 拆成了很多零件：

- [Attention](./attention.md)：为什么要动态读取信息；
- [Query / Key / Value](./qkv.md)：Q、K、V 为什么能分别承担查询、匹配、传递内容的角色；
- [Dot Product](./dot-product.md)：为什么 $q^\top k$ 可以作为 compatibility score；
- [Softmax](./softmax.md)：为什么 raw scores 能变成 normalized weights。

现在终于可以把它们重新装起来。

Self-Attention 的核心公式只有一行：

$$
\boxed{
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
}
$$

但真正理解它，应该能够回答：

1. 输入 $X$ 到底是什么？
2. 为什么同一个 $X$ 会生成三份 $Q,K,V$？
3. $QK^\top$ 每一个元素到底表示什么？
4. Softmax 是沿哪个维度做？
5. 为什么每一个 token 最后得到不同的 context？
6. 为什么输出 token 数量不变？
7. 为什么没有 positional information 时 Self-Attention 不知道顺序？
8. 为什么 Residual Connection 后 token 不会“把自己弄丢”？
9. Encoder Self-Attention 与 Decoder Masked Self-Attention 到底差在哪里？
10. ACT 为什么能靠 Self-Attention 融合 1200 个视觉 token、joint token 和 latent token？

这一篇用一个完整的小例子从头算一遍。

---

## 1. Self-Attention 的“Self”到底是什么意思？

先从定义开始。

Transformer 原论文写得非常清楚：

> 在 self-attention layer 中，所有 keys、values 和 queries 都来自同一个地方。

例如 Encoder 第 $l$ 层的输入：

$$
X
=
[x_1,x_2,\ldots,x_n]
$$

其中每个：

$$
x_i\in\mathbb R^{d_{\text{model}}}
$$

则：

$$
\boxed{
Q=XW_Q
}
$$

$$
\boxed{
K=XW_K
}
$$

$$
\boxed{
V=XW_V
}
$$

所以 Self-Attention 不是：

> “一个 token 只关注自己。”

恰恰相反。

它表示：

> **同一个 sequence 内部的位置彼此查询、彼此读取。**

---

## 2. Cross-Attention 与 Self-Attention 的区别先看一眼

Self-Attention：

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

Q/K/V source 全是：

$$
X
$$

---

Cross-Attention：

$$
Q=HW_Q
$$

$$
K=MW_K
$$

$$
V=MW_V
$$

其中：

- $H$：Decoder side；
- $M$：Encoder memory。

所以：

$$
\boxed{
\text{Self-Attention}
=
\text{same source for Q/K/V}
}
$$

$$
\boxed{
\text{Cross-Attention}
=
\text{query source differs from memory source}
}
$$

详细的 Cross-Attention 会另开页面。

---

## 3. 一个 Token 在进入 Self-Attention 前是什么？

假设一个 sequence 有 3 个 token：

```text
Token A
Token B
Token C
```

每个 token 已经被表示成一个向量：

$$
x_1,x_2,x_3
$$

例如：

$$
x_i\in\mathbb R^2
$$

为了便于手算，我们用非常小的：

$$
d_{\text{model}}=2
$$

真实 Transformer 通常大得多。

例如原始 Transformer base：

$$
d_{\text{model}}=512
$$

但计算逻辑完全一样。

---

## 4. 把 3 个 Token 堆成矩阵 X

假设：

$$
x_1=
[1,0]
$$

$$
x_2=
[0,1]
$$

$$
x_3=
[1,1]
$$

堆起来：

$$
\boxed{
X=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix}
}
$$

shape：

$$
[3,2]
$$

其中：

- 行 = token position；
- 列 = hidden features。

---

## 5. 先用一个教学版例子：W_Q、W_K、W_V 取简单矩阵

为了把 Attention 计算本身看清，我们先选：

$$
W_Q=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
$$

即 identity matrix。

所以：

$$
Q=X
$$

再令：

$$
W_K=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
$$

所以：

$$
K=X
$$

最后为了让 Value 和 Key 不完全一样，我们设：

$$
W_V=
\begin{bmatrix}
1&1\\
0&1
\end{bmatrix}
$$

真实模型中这些矩阵不是我们手工指定的。

它们会由任务 loss 学习。

这里这样设置只是为了方便手算。

---

## 6. 计算 Q

因为：

$$
W_Q=I
$$

所以：

$$
Q=XW_Q=X
$$

因此：

$$
\boxed{
Q=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix}
}
$$

即：

$$
q_1=[1,0]
$$

$$
q_2=[0,1]
$$

$$
q_3=[1,1]
$$

---

## 7. 计算 K

同样：

$$
K=XW_K=X
$$

所以：

$$
\boxed{
K=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix}
}
$$

即：

$$
k_1=[1,0]
$$

$$
k_2=[0,1]
$$

$$
k_3=[1,1]
$$

---

## 8. 计算 V

现在：

$$
V=XW_V
$$

其中：

$$
W_V=
\begin{bmatrix}
1&1\\
0&1
\end{bmatrix}
$$

---

对于 token 1：

$$
v_1
=
[1,0]
\begin{bmatrix}
1&1\\
0&1
\end{bmatrix}
=
[1,1]
$$

token 2：

$$
v_2
=
[0,1]
\begin{bmatrix}
1&1\\
0&1
\end{bmatrix}
=
[0,1]
$$

token 3：

$$
v_3
=
[1,1]
\begin{bmatrix}
1&1\\
0&1
\end{bmatrix}
=
[1,2]
$$

所以：

$$
\boxed{
V=
\begin{bmatrix}
1&1\\
0&1\\
1&2
\end{bmatrix}
}
$$

现在我们有：

```text
Token 1:
q1 = [1,0]
k1 = [1,0]
v1 = [1,1]

Token 2:
q2 = [0,1]
k2 = [0,1]
v2 = [0,1]

Token 3:
q3 = [1,1]
k3 = [1,1]
v3 = [1,2]
```

---

## 9. 第一步真正的 Attention：QKᵀ

我们计算：

$$
S=QK^\top
$$

因为：

$$
Q=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix}
$$

$$
K^\top=
\begin{bmatrix}
1&0&1\\
0&1&1
\end{bmatrix}
$$

所以：

$$
S=
\begin{bmatrix}
1&0&1\\
0&1&1\\
1&1&2
\end{bmatrix}
$$

即：

$$
\boxed{
QK^\top
=
\begin{bmatrix}
1&0&1\\
0&1&1\\
1&1&2
\end{bmatrix}
}
$$

---

## 10. 这个 Matrix 每一个元素到底是什么？

$$
S_{ij}
=
q_i^\top k_j
$$

例如：

$$
S_{13}
=
q_1^\top k_3
$$

$$
=
[1,0]
\cdot
[1,1]
=
1
$$

含义：

> token 1 作为 Query 时，对 token 3 的 Key compatibility score 是 1。

---

第 3 行：

$$
[1,1,2]
$$

表示：

> token 3 作为 Query 时，对 token 1、2、3 的 raw compatibility 分别为 1、1、2。

所以：

$$
\boxed{
\text{每一行}
=
\text{一个 Query 对所有 Keys 的 Scores}
}
$$

---

## 11. 为什么矩阵是 3 × 3？

因为：

- 3 个 queries；
- 3 个 keys。

每一个 query 都和每一个 key 比较一次：

$$
3\times3
=
9
$$

个 pairwise scores。

一般 sequence length：

$$
n
$$

则 full Self-Attention score matrix：

$$
\boxed{
n\times n
}
$$

这正是标准 Self-Attention quadratic sequence cost 的来源。

---

## 12. 第二步：除以 √d_k

这里：

$$
d_k=2
$$

所以：

$$
\sqrt{d_k}
=
\sqrt2
\approx1.414
$$

Scaled scores：

$$
\tilde S
=
\frac{S}{\sqrt2}
$$

得到近似：

$$
\boxed{
\tilde S
\approx
\begin{bmatrix}
0.707&0&0.707\\
0&0.707&0.707\\
0.707&0.707&1.414
\end{bmatrix}
}
$$

---

## 13. 为什么 Scale 不改变每一行的排序？

因为所有 score 同除一个正数：

$$
\sqrt{d_k}
$$

所以：

$$
s_i>s_j
$$

仍然意味着：

$$
s_i/\sqrt{d_k}
>
s_j/\sqrt{d_k}
$$

它改变的是：

> score differences 的尺度。

不是：

> 谁比谁大。

所以 scale 主要控制后续 Softmax 的 sharpness / gradient behavior。

---

## 14. 第三步：每一行做 Softmax

现在对：

$$
\tilde S
$$

每一行：

> 沿 Key dimension

单独做 Softmax。

---

### Query 1

scores：

$$
[0.707,0,0.707]
$$

指数：

$$
[e^{0.707},e^0,e^{0.707}]
$$

近似：

$$
[2.028,1,2.028]
$$

总和：

$$
5.056
$$

所以：

$$
A_{1,:}
\approx
[0.401,0.198,0.401]
$$

---

## 15. Query 2

scores：

$$
[0,0.707,0.707]
$$

所以：

$$
A_{2,:}
\approx
[0.198,0.401,0.401]
$$

---

## 16. Query 3

scores：

$$
[0.707,0.707,1.414]
$$

指数近似：

$$
[2.028,2.028,4.113]
$$

总和：

$$
8.169
$$

因此：

$$
A_{3,:}
\approx
[0.248,0.248,0.503]
$$

---

## 17. Attention Matrix

最后：

$$
\boxed{
A
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt2}
\right)
\approx
\begin{bmatrix}
0.401&0.198&0.401\\
0.198&0.401&0.401\\
0.248&0.248&0.503
\end{bmatrix}
}
$$

每一行和为：

$$
1
$$

---

## 18. 怎样读 Attention Matrix？

第一行：

$$
[0.401,0.198,0.401]
$$

表示：

> token 1 读取：

- token 1 的 Value：40.1%
- token 2 的 Value：19.8%
- token 3 的 Value：40.1%

---

第二行：

$$
[0.198,0.401,0.401]
$$

表示 token 2：

> 更少读取 token 1，更多读取 token 2 和 token 3。

---

第三行：

$$
[0.248,0.248,0.503]
$$

表示 token 3：

> 对自己的 Value 权重最大，但仍然读取 token 1 和 token 2。

---

## 19. 注意：这不是“Token 之间唯一真正关系”

这个矩阵只对应：

- 当前这一层；
- 当前这个 Attention head；
- 当前输入；
- 当前 Q/K projections。

如果：

- 输入变；
- layer 变；
- head 变；

Attention Matrix 都会变。

所以：

$$
A
$$

是：

> 当前计算中的动态信息路由矩阵。

不是整个模型永远固定的关系图。

---

## 20. 第四步：A V

现在：

$$
O=AV
$$

其中：

$$
V=
\begin{bmatrix}
1&1\\
0&1\\
1&2
\end{bmatrix}
$$

---

## 21. Token 1 的输出

$$
o_1
=
0.401v_1
+
0.198v_2
+
0.401v_3
$$

代入：

$$
v_1=[1,1]
$$

$$
v_2=[0,1]
$$

$$
v_3=[1,2]
$$

得到：

$$
o_1
\approx
0.401[1,1]
+
0.198[0,1]
+
0.401[1,2]
$$

第一维：

$$
0.401+0+0.401
=
0.802
$$

第二维：

$$
0.401+0.198+0.802
=
1.401
$$

所以：

$$
\boxed{
o_1
\approx
[0.802,\;1.401]
}
$$

---

## 22. Token 2 的输出

$$
o_2
=
0.198v_1
+
0.401v_2
+
0.401v_3
$$

得到：

$$
\boxed{
o_2
\approx
[0.599,\;1.401]
}
$$

---

## 23. Token 3 的输出

$$
o_3
=
0.248v_1
+
0.248v_2
+
0.503v_3
$$

得到近似：

$$
\boxed{
o_3
\approx
[0.751,\;1.502]
}
$$

---

## 24. 最终 Self-Attention Output

所以：

$$
\boxed{
O
\approx
\begin{bmatrix}
0.802&1.401\\
0.599&1.401\\
0.751&1.502
\end{bmatrix}
}
$$

shape：

$$
[3,2]
$$

注意：

> 输入是 3 个 tokens。

输出仍然是：

> 3 个 tokens。

只是每个 token 的 representation 已经根据整条 sequence 更新。

---

## 25. 为什么 Token 数量不变？

因为：

$$
Q
$$

有：

$$
n_q
$$

个 queries。

Attention 最终：

$$
O
=
A V
$$

其中：

$$
A
\in
\mathbb R^{n_q\times n_k}
$$

$$
V
\in
\mathbb R^{n_k\times d_v}
$$

所以：

$$
O
\in
\mathbb R^{n_q\times d_v}
$$

Self-Attention 中：

$$
n_q=n_k=n
$$

因此：

$$
\boxed{
n\text{ input positions}
\rightarrow
n\text{ output positions}
}
$$

每一个 Query position 都得到一个新的 context vector。

---

## 26. Self-Attention 不是把整个 Sequence 压成一个 Vector

这一点非常重要。

它不是：

$$
[x_1,x_2,x_3]
\rightarrow
c
$$

而是：

$$
[x_1,x_2,x_3]
\rightarrow
[o_1,o_2,o_3]
$$

每个：

$$
o_i
$$

都是：

> 以 position $i$ 自己的 Query 为中心，从整条 sequence 动态读取后的 context-aware representation。

---

## 27. 为什么每个 Token 得到不同 Context？

因为每个 token 有不同：

$$
q_i
$$

所以：

$$
q_i^\top K
$$

得到不同 score row。

再经过 Softmax：

$$
A_{i,:}
$$

也不同。

因此：

$$
o_i
=
A_{i,:}V
$$

自然不同。

所以即使：

> 所有 token 都访问同一套 K/V memory，

每个 Query 也能得到不同读取结果。

---

## 28. 这就是 Self-Attention 的核心

可以把整个过程写成：

```text
Token i
↓
产生 Query q_i
↓
和所有 Keys 比较
↓
得到一行 Attention Weights
↓
按这组权重聚合所有 Values
↓
得到新的 Context o_i
```

然后：

> 所有 tokens 同时做这件事。

所以：

$$
\boxed{
\text{Self-Attention}
=
\text{all positions query the same sequence in parallel}
}
$$

---

## 29. 为什么 Self-Attention 可以并行？

RNN：

$$
h_t=f(h_{t-1},x_t)
$$

需要先算：

$$
h_{t-1}
$$

才能算：

$$
h_t
$$

---

Self-Attention：

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

可以对整块：

$$
X
$$

同时矩阵计算。

然后：

$$
QK^\top
$$

也是一次 matrix multiplication。

所以在一层内部：

> 不要求先得到 token 1 的 output，才能计算 token 2。

原 Transformer 论文因此把 Self-Attention layer 的最低 sequential operations 写为：

$$
\boxed{O(1)}
$$

相比 recurrent layer：

$$
\boxed{O(n)}
$$

---

## 30. O(1) Sequential Operations 不等于 O(1) 计算复杂度

必须非常严格。

Self-Attention 的主要计算：

$$
QK^\top
$$

会形成：

$$
n\times n
$$

matrix。

所以 per-layer time complexity：

$$
\boxed{
O(n^2d)
}
$$

而不是：

$$
O(1)
$$

所谓：

$$
O(1)
$$

指的是：

> sequence dependency 导致的串行计算步数。

不是总 FLOPs。

---

## 31. 为什么两个远距离 Token 可以一层直接交互？

假设 sequence length：

$$
1000
$$

token 1 与 token 1000。

Self-Attention 里：

$$
S_{1000,1}
=
q_{1000}^\top k_1
$$

直接存在。

所以 token 1000 可以在一层中直接读取：

$$
v_1
$$

不需要：

```text
1
→ 2
→ 3
→ ...
→ 1000
```

这种 recurrent path。

这就是 Transformer 原论文强调的：

$$
\boxed{
\text{maximum path length}=O(1)
}
$$

---

## 32. 为什么 Self-Attention 能建立“上下文表示”？

因为原始：

$$
x_i
$$

主要是 position $i$ 的当前 hidden representation。

经过：

$$
o_i
=
\sum_j
A_{ij}v_j
$$

后，

它显式依赖：

$$
x_1,\ldots,x_n
$$

通过 K/V projections 进入计算。

所以：

$$
o_i
$$

不再只是：

> “token i 是什么”。

而是：

> “token i 在当前整条 sequence context 中，应该得到什么更新信息”。

---

## 33. 一个语言例子

句子：

```text
The animal didn't cross the street because it was tired.
```

假设处理：

```text
it
```

token。

它的 Query：

$$
q_{\text{it}}
$$

可以对所有 Keys：

```text
The
animal
didn't
cross
street
because
it
was
tired
```

打分。

如果：

$$
k_{\text{animal}}
$$

获得较高 compatibility，

则：

$$
v_{\text{animal}}
$$

会较多进入：

$$
o_{\text{it}}
$$

于是 `it` 的新 hidden representation 可以带上与 `animal` 相关的 context。

---

## 34. 但不能说模型一定靠某个 Head 学会指代

Self-Attention 提供这种能力。

但某个具体模型：

- 哪一层；
- 哪个 head；
- 是否有清晰人类可解释 pattern；

都需要实证分析。

所以正确表述是：

> Self-Attention 允许位置直接基于内容进行全局信息交换。

而不是：

> “Attention 公式天然知道代词应该看名词。”

---

## 35. 现在进入一个最重要的问题：如果没有 Position Encoding 会怎样？

Self-Attention 只看到：

$$
X
$$

并对每个 row 使用相同：

$$
W_Q,W_K,W_V
$$

如果我们只是重新排列 token 顺序：

$$
X'
=
PX
$$

其中：

$$
P
$$

是 permutation matrix，

会发生什么？

---

## 36. 先计算重新排序后的 Q/K/V

$$
Q'
=
X'W_Q
$$

$$
=
PXW_Q
$$

所以：

$$
\boxed{
Q'=PQ
}
$$

同理：

$$
\boxed{
K'=PK
}
$$

$$
\boxed{
V'=PV
}
$$

---

## 37. Score Matrix 怎样变化？

$$
S'
=
Q'K'^\top
$$

代入：

$$
=
(PQ)(PK)^\top
$$

注意：

$$
(PK)^\top
=
K^\top P^\top
$$

所以：

$$
\boxed{
S'
=
P QK^\top P^\top
}
$$

也就是说：

> score matrix 的 rows 和 columns 只是跟着 token permutation 一起重排。

---

## 38. Row-Wise Softmax 也跟着重排

由于 permutation 只是重排元素，

对每一行 Softmax 后：

$$
A'
=
PAP^\top
$$

然后：

$$
O'
=
A'V'
$$

代入：

$$
=
(PAP^\top)(PV)
$$

因为：

$$
P^\top P=I
$$

得到：

$$
\boxed{
O'=PO
}
$$

---

## 39. 这就叫 Permutation Equivariance

如果输入 token 顺序按：

$$
P
$$

重新排列，

输出也只按同一个：

$$
P
$$

重新排列。

即：

$$
\boxed{
SA(PX)
=
P\,SA(X)
}
$$

对于：

> 不带 positional information、full unmasked Self-Attention

成立。

这叫：

> **permutation equivariance**

---

## 40. Equivariance 和 Invariance 不一样

Permutation invariant：

$$
f(PX)=f(X)
$$

输入重排，

输出完全不变。

---

Permutation equivariant：

$$
f(PX)=Pf(X)
$$

输入重排，

输出跟着同样重排。

Self-Attention 更准确是后者。

所以不能简单说：

> “Self-Attention 对顺序完全没反应。”

更准确：

> **它能处理每个 token 的内容关系，但不知道某个 row 原本是第 1 个、第 2 个还是第 10 个；你怎么排列 tokens，它就按同样排列产生对应 outputs。**

---

## 41. 为什么这会造成 Sequence Order 问题？

考虑：

```text
dog bites man
```

和：

```text
man bites dog
```

如果 token representations 只有词内容，

没有任何 position information，

Self-Attention 看到的只是：

> 同一组三个 token 的不同排列。

它会输出对应 representation 的同样排列变换。

模型没有额外机制知道：

> 谁是前面的词、谁是后面的词。

而语言语义显然依赖顺序。

所以 Transformer 需要：

> **Positional Encoding / Position Information**

---

## 42. Positional Encoding 怎样打破这个对称性？

输入不再只是：

$$
x_i=e_i
$$

而是：

$$
\boxed{
x_i=e_i+p_i
}
$$

其中：

$$
p_i
$$

取决于 position。

那么同一个 token：

```text
dog
```

放在 position 1：

$$
e_{\text{dog}}+p_1
$$

放在 position 3：

$$
e_{\text{dog}}+p_3
$$

已经不是同一个输入 vector。

因此：

$$
Q,K,V
$$

也会随 position 改变。

Self-Attention 就有机会学习：

> 内容 + 位置

共同决定的 interaction。

---

## 43. 原始 Transformer 为什么用 Sinusoidal Position？

原论文使用：

$$
PE_{(pos,2i)}
=
\sin
\left(
\frac{pos}
{
10000^{2i/d_{\text{model}}}
}
\right)
$$

$$
PE_{(pos,2i+1)}
=
\cos
\left(
\frac{pos}
{
10000^{2i/d_{\text{model}}}
}
\right)
$$

然后：

$$
X=Embedding+PE
$$

所以 Attention 本身不负责创造顺序。

Position mechanism 负责提供：

> where。

Self-Attention 负责：

> who should exchange information with whom。

---

## 44. ACT 的视觉 Token 为什么也需要 Position？

ACT 每张 camera image 经过 ResNet 得到：

$$
15\times20
$$

spatial grid。

flatten 后：

$$
300
$$

个 visual tokens。

如果完全不带空间位置信息，

Transformer 只知道：

> 有 300 个视觉 feature vectors。

但不知道：

- 哪个来自左上；
- 哪个来自右下；
- 哪些原本相邻。

所以 ACT 给 visual tokens：

> 2D positional encoding。

这和文本中的 sequence position 是同一类问题：

> Self-Attention 本身不知道结构位置。

---

## 45. 为什么 Self-Attention 输出之后还要 Residual？

刚才单独计算的：

$$
O=AV
$$

主要表示：

> 当前 token 从 context 中聚合到的信息。

但原输入：

$$
X
$$

本身也很重要。

原始 Transformer 使用：

$$
\boxed{
X+O
}
$$

再进入 LayerNorm：

$$
\operatorname{LayerNorm}(X+O)
$$

所以 token 的新 representation 并不是：

> 完全替换成别人信息的平均。

而是：

> 保留自己原来的 representation，同时加入从 context 读到的信息。

---

## 46. 这也回答一个常见疑问：Weighted Sum 会不会把 Token 自己“洗掉”？

不会简单如此。

第一：

> Self-Attention 可以关注自己：

$$
A_{ii}
$$

可以非零。

第二：

> Residual 直接保留：

$$
x_i
$$

因此一个 Transformer block 中 token 有至少两条 self-information path：

```text
自己的 Value
→ Self-Attention

以及

自己的原始 hidden state
→ Residual
```

所以 Self-Attention 并不意味着“每层都把所有 token 混成平均”。

---

## 47. 为什么 FFN 还需要存在？

Self-Attention 主要做：

> positions 之间的信息交换。

得到 context 后，

每个 position 还需要 nonlinear processing。

原始 Transformer Encoder layer：

```text
Self-Attention
↓
Residual + LayerNorm
↓
Position-wise FFN
↓
Residual + LayerNorm
```

FFN 对每个 position 独立应用：

$$
FFN(x)
=
\max(0,xW_1+b_1)W_2+b_2
$$

所以可以粗略分工：

$$
\boxed{
Attention
=
\text{mix information across positions}
}
$$

$$
\boxed{
FFN
=
\text{transform information within each position}
}
$$

---

## 48. 为什么叫 Position-Wise FFN？

因为：

$$
x_1,x_2,\ldots,x_n
$$

都经过同一个 FFN 参数。

但 FFN 本身不会计算：

$$
x_i
$$

和：

$$
x_j
$$

之间的 interaction。

所以 token-to-token 交流主要发生在 Attention。

FFN 负责：

> 每个 contextualized token 自己的 nonlinear feature transformation。

---

## 49. Self-Attention 和 Multi-Head Attention 是什么关系？

前面手算的是：

> 单个 Attention head。

真实 Transformer 通常使用：

> Multi-Head Self-Attention。

第 $h$ 个 head：

$$
Q_h=XW_Q^{(h)}
$$

$$
K_h=XW_K^{(h)}
$$

$$
V_h=XW_V^{(h)}
$$

然后：

$$
head_h
=
\operatorname{softmax}
\left(
\frac{
Q_hK_h^\top
}{
\sqrt{d_k}
}
\right)V_h
$$

最后：

$$
\boxed{
MultiHead
=
Concat(head_1,\ldots,head_H)W_O
}
$$

所以：

> “Self”说的是 Q/K/V source 相同。

> “Multi-Head”说的是有多套不同 projection / attention spaces 并行存在。

两者描述不同维度。

---

## 50. 为什么单头不够？

如果只有一个 head，

所有关系都要通过：

> 一套 Q/K matching geometry

来表达。

Multi-Head 给模型：

> 多套不同 matching + message spaces。

原论文明确说：

> Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

详细会在：

- [Multi-Head Attention](./multi-head-attention.md)

单独展开。

---

## 51. Encoder Self-Attention 为什么通常不需要 Causal Mask？

Encoder 已经知道完整 input sequence。

例如：

```text
I love learning AI
```

position 2 读取 position 4 并不算：

> 偷看答案。

所以原始 Transformer Encoder：

> 每个 position 可以 attend 所有 input positions。

这是：

> **full / bidirectional self-attention**

---

## 52. Decoder Self-Attention 为什么要 Mask？

原始 Transformer Decoder 是 autoregressive。

训练预测：

$$
y_t
$$

时不能读取：

$$
y_{t+1},y_{t+2},\ldots
$$

所以把非法 future scores 设：

$$
-\infty
$$

Softmax 后 weight：

$$
0
$$

因此 Decoder Self-Attention 变成：

> **Masked / Causal Self-Attention**

---

## 53. Causal Self-Attention 还算 Self-Attention 吗？

当然。

因为：

$$
Q,K,V
$$

仍然来自同一个 sequence。

区别只是：

> allowed attention edges 被 mask 限制。

所以：

```text
Encoder:
full self-attention

Autoregressive Decoder:
causal self-attention
```

都属于 Self-Attention。

---

## 54. Mask 会破坏任意 Permutation Equivariance

前面证明：

$$
SA(PX)=PSA(X)
$$

依赖的是：

> full unmasked attention，没有额外 position-dependent constraint。

Causal mask 本身包含：

> “谁在谁之前”

的结构。

所以 arbitrary permutation 后，如果 mask 不跟着做相应一致变换，

这种 permutation equivariance 不再成立。

因此说：

> “Self-Attention 永远 permutation-equivariant”

不够严谨。

更准确：

$$
\boxed{
\text{full self-attention without positional information}
}
$$

具有这一性质。

---

## 55. Padding Mask 又是什么？

batch 中不同 sequence 长度：

```text
sample A:
token token token token

sample B:
token token PAD PAD
```

PAD 位置不是实际内容。

所以 score matrix 对 PAD keys 加：

$$
-\infty
$$

防止其他 queries 读取它。

这叫：

> **padding mask**

与 causal mask 不同：

```text
padding mask:
禁止看不存在的位置

causal mask:
禁止看未来
```

---

## 56. 为什么 Self-Attention 是“内容驱动”的？

Attention Matrix：

$$
A(X)
=
\operatorname{softmax}
\left(
\frac{
XW_QW_K^\top X^\top
}{
\sqrt{d_k}
}
\right)
$$

依赖：

$$
X
$$

本身。

所以输入换了：

> token-to-token routing pattern 也会换。

这不同于 CNN 固定 kernel：

> 同一个 filter pattern 在每个位置应用。

Self-Attention 的连接强度：

> 是当前 sample 动态生成的。

---

## 57. 这是不是意味着 Self-Attention 没有 Inductive Bias？

不是。

它仍然有很多结构约束：

- pairwise Q/K compatibility；
- shared projections；
- Softmax normalization；
- Value aggregation；
- head dimensions；
- mask structure；
- positional mechanism。

只是相比：

> locality-heavy convolution

它对“谁可以和谁直接交互”的限制更少。

Full Self-Attention 允许：

> 任意位置直接 interaction。

---

## 58. 为什么 Self-Attention 对 Long-Range Dependency 很适合？

如果 token 1 和 token 1000 相关：

$$
q_{1000}^\top k_1
$$

可以直接产生高 score。

无需通过：

$$
2,3,\ldots,999
$$

逐级传递。

这让 dependency path 非常短。

但代价就是：

> 所有 pairs 都要考虑，

导致：

$$
O(n^2)
$$

sequence interaction cost。

---

## 59. Self-Attention 的时间复杂度怎样来的？

主要看：

$$
QK^\top
$$

如果：

$$
Q,K
\in
\mathbb R^{n\times d}
$$

matrix multiplication 大致：

$$
O(n^2d)
$$

再：

$$
AV
$$

也是：

$$
O(n^2d)
$$

所以标准 full attention per layer：

$$
\boxed{
O(n^2d)
}
$$

其中：

- $n$：sequence length；
- $d$：representation dimension。

---

## 60. Memory 为什么也是 Quadratic？

Attention Matrix：

$$
A
\in
\mathbb R^{n\times n}
$$

显式存储需要：

$$
O(n^2)
$$

空间。

训练时还要保存更多中间量用于 backward。

所以 sequence 很长时：

> Attention Matrix 往往成为显存瓶颈之一。

这也是 FlashAttention 等工作的背景之一。

---

## 61. Self-Attention 和 RNN 的 Trade-Off

RNN：

- sequential operations：$O(n)$
- pairwise direct path：长
- 每步不需要显式 $n\times n$ matrix

Self-Attention：

- layer 内 positions 高度并行
- 任意两位置一层可直接交互
- 但 full attention 有 $O(n^2)$ pairwise cost

所以 Transformer 不是：

> “所有方面都比 RNN 更省。”

而是：

> 换了一种 computation / representation trade-off。

---

## 62. Self-Attention 会自动把所有 Token 变得一样吗？

单层不必然。

每个 query row：

$$
A_{i,:}
$$

通常不同，

所以：

$$
o_i
$$

也不同。

同时 residual：

$$
x_i+o_i
$$

保留 position-specific information。

但如果某些 attention matrices 变得极度均匀，

确实可能出现 representation mixing / oversmoothing-like effects。

这也是为什么 Transformer 不只有纯 Attention：

- residual；
- FFN；
- normalization；

都很重要。

---

## 63. 为什么 `[CLS]` Token 能汇总整个 Sequence？

假设 sequence 前面加入：

$$
x_{\text{CLS}}
$$

它和其他 token 一样产生：

$$
q_{\text{CLS}}
$$

然后：

$$
q_{\text{CLS}}
$$

可以 attend 所有 keys。

所以：

$$
o_{\text{CLS}}
=
\sum_j
A_{\text{CLS},j}v_j
$$

自然可以聚合整个 sequence 信息。

经过多层后：

$$
h_{\text{CLS}}
$$

可以成为 global summary representation。

这不是因为 `[CLS]` 三个字母有魔法。

而是：

- 它有自己的 learned embedding；
- 它有自己的 Query；
- 它可以从全 sequence 读取；
- task loss 又直接使用它；

所以训练会塑造它成为有用的 summary token。

---

## 64. 这正是 ACT CVAE Encoder 里的机制

ACT Training CVAE encoder 输入：

```text
[CLS]
joint token
action 0
action 1
...
action k-1
```

Self-Attention 允许：

$$
[CLS]
$$

读取：

- joint state；
- 整个 demonstration action chunk。

最终：

$$
h_{\text{CLS}}
\rightarrow
\mu,\log\sigma^2
$$

所以：

> `[CLS]` 能总结 action sequence，是 Self-Attention + training objective 共同学出来的结果。

---

## 65. ACT Policy Encoder 中 Self-Attention 又做什么？

ACT policy encoder 输入：

$$
1202
$$

个 tokens：

```text
1 latent token
1 joint token
1200 visual tokens
```

记：

$$
X
\in
\mathbb R^{1202\times512}
$$

Multi-Head Self-Attention 会计算：

$$
Q,K,V
$$

然后：

$$
QK^\top
$$

shape：

$$
\boxed{
1202\times1202
}
$$

每一个 token 都可以与其他所有 tokens 直接 interaction。

---

## 66. 1202 × 1202 代表什么？

总共有：

$$
1202^2
=
1,444,804
$$

个 query-key pair positions。

注意：

> 这不意味着模型真的存储 1,444,804 个独立 learned parameters。

Attention score matrix 是：

> 当前 forward 动态计算出来的。

参数仍然主要是：

$$
W_Q,W_K,W_V
$$

等共享 projection matrices。

---

## 67. Visual Token 可以读取 Joint Token 吗？

从 architecture capability 看：

> 可以。

因为 policy encoder 是 full self-attention。

某个 visual token 的 Query 可以对：

$$
k_{\text{joint}}
$$

产生 attention score。

同样：

joint token 也可以读取：

- top camera；
- wrist camera；
- latent token。

所以 Self-Attention 给 ACT 一个统一的：

> multimodal token interaction space。

---

## 68. 但“可以读取”不代表一定高权重

是否实际出现高 attention：

> 由训练学出。

不能只根据 architecture 说：

> “每个视觉 token 都一定强烈使用 joint state。”

架构只保证：

> 这条 interaction path 是允许存在的。

---

## 69. 为什么 ACT 先用 ResNet 再 Self-Attention？

原图：

$$
4\times480\times640
$$

如果每个 pixel 都作为 token，

sequence 太长。

所以 ResNet 先压成：

$$
4\times15\times20
=
1200
$$

visual positions。

Self-Attention 再在这些较紧凑的 high-level features 上做全局 interaction。

所以：

```text
CNN
→ local / spatial feature extraction + downsampling

Self-Attention
→ global token interaction
```

两者分工互补。

---

## 70. 为什么 Self-Attention 不直接产生 Robot Action？

在 ACT Policy Encoder 中：

Self-Attention 只负责产生：

$$
\boxed{
\text{observation memory}
}
$$

即 contextualized input tokens。

之后还要：

> Transformer Decoder 的 action queries

通过 Cross-Attention 读取这份 memory。

然后：

$$
512\rightarrow14
$$

Action Head 才输出 joint targets。

所以：

```text
Self-Attention
≠
Action Prediction Head
```

---

## 71. 为什么 ACT Decoder 也有 Self-Attention？

ACT 有：

$$
k
$$

个 future action slots。

Decoder self-attention 允许：

$$
slot_i
\leftrightarrow
slot_j
$$

之间 exchange information。

因此 future action positions 不是：

> k 个完全独立的 regressor。

它们可以共同形成：

> coherent action sequence representation。

---

## 72. ACT Decoder 为什么没有 Causal Mask？

因为 ACT 不是 autoregressive 地：

```text
先 aₜ
再 aₜ₊₁
再 aₜ₊₂
```

它一次并行预测整个：

$$
k
$$

action chunk。

因此 future slot $i$ 可以和：

$$
j>i
$$

的 slots 直接 Self-Attend。

所以 ACT Decoder Self-Attention 是：

> non-causal / bidirectional over action slots。

---

## 73. 这和语言 Decoder 是根本差异

Original Transformer Decoder：

$$
y_t
$$

训练时不能看未来：

$$
y_{t+1}
$$

所以需要 causal mask。

ACT Decoder：

> 所有 action slots 本来就一起作为输出 structure 预测。

没有 next-token autoregressive factorization。

所以不需要 causal mask。

因此：

$$
\boxed{
\text{Self-Attention}
\neq
\text{Causal Attention}
}
$$

Causality 是额外 mask constraint。

---

## 74. Self-Attention 为什么“看起来像加权平均”却很强？

单头确实：

$$
o_i=\sum_jA_{ij}v_j
$$

看起来只是 weighted average。

但真正模型有：

1. $V=XW_V$ 是 learned representation；
2. weights $A$ 是 input-dependent；
3. Multi-Head 有多组不同 spaces；
4. 后面还有 $W_O$；
5. Residual；
6. nonlinear FFN；
7. 多层堆叠。

所以完整 Transformer 并不是：

> “反复简单平均 token。”

而是：

> 动态路由 + feature transform + residual composition 的深层网络。

---

## 75. 为什么单层 Attention 本身对 V 是线性的？

固定：

$$
A
$$

时：

$$
O=AV
$$

对：

$$
V
$$

是线性组合。

但：

$$
A
$$

本身又依赖：

$$
Q,K
$$

而 Q/K 依赖：

$$
X
$$

并经过 Softmax 非线性。

所以从：

$$
X
\rightarrow O
$$

整体来看：

> Self-Attention 不是简单线性映射。

---

## 76. Attention Matrix 是对称的吗？

一般：

$$
\boxed{
A\neq A^\top
}
$$

即使在我们教学例子里：

$$
Q=K
$$

导致 raw：

$$
QK^\top
$$

对称，

经过 row-wise Softmax 后也不一定保持对称。

为什么？

因为每一行的 normalization denominator 不同。

更一般地：

$$
W_Q\neq W_K
$$

则 raw score matrix 本身就通常不对称。

所以：

> token i 高度关注 token j

不意味着：

> token j 也同样高度关注 token i。

Attention relation 可以是有方向的。

---

## 77. 为什么这是合理的？

“读取关系”本来就可以不对称。

例如：

> 代词 `it` 需要读取 `animal`。

但：

> `animal` 不一定同样需要读取 `it`。

所以 Query–Key 机制天然允许：

$$
A_{ij}\neq A_{ji}
$$

非常适合表示 directed information flow。

---

## 78. Token 会不会永远最关注自己？

不会。

虽然：

$$
q_i^\top k_i
$$

有时可能高，

但没有任何规则要求：

$$
A_{ii}
$$

最大。

另一个 token：

$$
j
$$

完全可能得到更高 score。

是否 self-focus：

> 由当前 layer/head/input 决定。

---

## 79. 为什么 Attention 可以学 Local，也可以学 Global？

Full Self-Attention 没有限制：

$$
j
$$

必须靠近：

$$
i
$$

所以它可以学：

#### Local pattern

$$
i\rightarrow i\pm1
$$

也可以学：

#### Long-range pattern

$$
i\rightarrow i-100
$$

是否发生取决于：

$$
Q/K
$$

learned compatibility 和 position mechanism。

因此它不像 fixed small convolution kernel 那样天然只看局部。

---

## 80. Positional Encoding 加了以后，还算 Content-Based Attention 吗？

仍然算。

只是 hidden representation 现在包含：

$$
\text{content}
+
\text{position}
$$

所以 Q/K matching 可以同时使用：

- token 内容；
- position 信息。

因此更准确地说：

> Attention score 是 representation-dependent。

而 representation 可以编码多种因素。

---

## 81. Self-Attention 会修改原顺序吗？

通常不会重新排序 tokens。

输入：

$$
[x_1,x_2,\ldots,x_n]
$$

输出：

$$
[o_1,o_2,\ldots,o_n]
$$

position $i$ 的 output 仍然占 position $i$。

它只是：

> 让 position $i$ 的 representation 读取其他 positions 的信息。

所以 Attention：

> 不是 sorting algorithm。

---

## 82. 为什么输出位置仍然对应原 Query 位置？

因为每一个：

$$
o_i
$$

是由：

$$
q_i
$$

发起的一次读取得到。

因此 output row $i$ 的 identity 来自：

> Query row $i$。

所以即使它大量读取别的 tokens，

它仍然是：

> position $i$ 更新后的 representation。

---

## 83. 为什么一个 Head 的 d_v 可以比 d_model 小？

因为每个 head 只工作在一个 subspace。

例如原始 Transformer：

$$
d_{\text{model}}=512
$$

$$
h=8
$$

每头：

$$
d_v=64
$$

8 个 heads：

$$
8\times64=512
$$

concat 后恢复总宽度。

所以每个 head 不需要独立产生完整：

$$
512
$$

维输出。

---

## 84. Self-Attention 的完整 Shape Flow

一般：

$$
X
\in
\mathbb R^{B\times n\times d_{\text{model}}}
$$

单头简化：

$$
W_Q
\in
\mathbb R^{d_{\text{model}}\times d_k}
$$

$$
W_K
\in
\mathbb R^{d_{\text{model}}\times d_k}
$$

$$
W_V
\in
\mathbb R^{d_{\text{model}}\times d_v}
$$

所以：

$$
Q
\in
\mathbb R^{B\times n\times d_k}
$$

$$
K
\in
\mathbb R^{B\times n\times d_k}
$$

$$
V
\in
\mathbb R^{B\times n\times d_v}
$$

---

score：

$$
QK^\top
$$

对最后两维：

$$
\boxed{
[B,n,n]
}
$$

Softmax：

$$
A
\in
[B,n,n]
$$

最后：

$$
AV
$$

得到：

$$
\boxed{
[B,n,d_v]
}
$$

Multi-Head 再 concat 回：

$$
[B,n,d_{\text{model}}]
$$

---

## 85. PyTorch 中常见 Shape 为什么有时是 [n, B, d]？

不同实现 convention 不同。

早期：

```python
nn.MultiheadAttention
```

常使用：

$$
[sequence,batch,embedding]
$$

即：

$$
[n,B,d]
$$

现代很多代码使用：

$$
[B,n,d]
$$

尤其：

```python
batch_first=True
```

数学逻辑完全相同。

不要因为 tensor dimension 顺序不同，误以为 architecture 改了。

---

## 86. Self-Attention 和 Batch 之间会互相 Attention 吗？

不会。

batch 中不同 samples 是独立计算的。

Attention Matrix 实际有：

$$
B
$$

份：

$$
A^{(1)},A^{(2)},\ldots,A^{(B)}
$$

每个 sample 自己：

$$
n\times n
$$

不会让：

> sample 1 的 token

去读取：

> sample 2 的 token。

---

## 87. 为什么 Padding Mask 通常按 Batch 不同？

因为 batch 中每条 sequence 实际长度可能不同。

所以每个 sample 的 valid key positions 不一样。

mask shape 常能 broadcast 到：

$$
[B,h,n_q,n_k]
$$

对应每个 sample / head / query-key pair。

---

## 88. Attention Dropout 又是什么？

原始 Transformer 对 attention weights 也使用 dropout。

高层理解：

> training 时随机丢弃部分 attention contribution，作为 regularization。

这不是 Self-Attention 定义的核心。

但实现 Transformer block 时通常会看到：

```text
attention probabilities
→ dropout
→ weighted value aggregation
```

具体 placement 随实现可能不同。

---

## 89. Self-Attention 和 Attention 的关系

最一般 Attention：

$$
\operatorname{Attention}
(
Q,
K,
V
)
$$

Self-Attention 只是规定：

$$
Q,K,V
$$

来自同一个 source sequence。

所以：

$$
\boxed{
\text{Self-Attention}
\subset
\text{Attention mechanisms}
}
$$

而 Cross-Attention 则是：

> Q source 与 K/V source 不同。

---

## 90. 为什么 Transformer 的突破不是“发明加权平均”？

Weighted sum 很早就存在。

Attention 的关键是：

> **权重由当前 representations 动态计算。**

Self-Attention 更进一步：

> **sequence 内所有 positions 都可以作为 Query，同时动态读取整个 sequence。**

加上高效矩阵计算，

才形成 Transformer 强大的 sequence interaction mechanism。

---

## 91. 常见误解一：Self-Attention = Token 关注自己

**错误。**

“Self”表示 Q/K/V 来自同一 sequence。

---

## 92. 常见误解二：一个 Token 最后只会选择一个其他 Token

**错误。**

Softmax 通常是 soft weighted combination。

---

## 93. 常见误解三：QKᵀ 已经是 Attention Weights

**错误。**

它只是 raw compatibility scores。

还需要：

$$
1/\sqrt{d_k}
$$

以及：

$$
Softmax
$$

---

## 94. 常见误解四：Softmax 对整个 n × n Matrix 一起做

**错误。**

每个 Query row：

> 独立沿 keys 做 Softmax。

---

## 95. 常见误解五：Self-Attention 把 n 个 Tokens 压成一个 Vector

**错误。**

输出仍有：

$$
n
$$

个 positions。

---

## 96. 常见误解六：输出 o_i 就是最相关 Token 的 Value

**通常错误。**

它是：

$$
\sum_jA_{ij}v_j
$$

的 weighted combination。

---

## 97. 常见误解七：没有 Positional Encoding 时，Self-Attention 完全无法处理内容关系

**错误。**

它仍然可以根据 token content 做 Q/K matching。

缺失的是：

> 原始 position/order identity。

---

## 98. 常见误解八：没有 Position 时 Self-Attention 是 Permutation Invariant

更准确是：

> full Self-Attention 对 token permutation 是 **equivariant**。

输入怎么重排，

对应输出怎么重排。

---

## 99. 常见误解九：Self-Attention 自带时间方向

**错误。**

时间方向来自：

- causal mask；
- positional information；
- architecture / objective。

full self-attention 本身没有“过去/未来”的天然概念。

---

## 100. 常见误解十：Decoder Self-Attention 一定 Causal

**错误。**

原始语言 Transformer Decoder：

> causal。

ACT Decoder：

> non-causal。

是否 causal 是 mask choice，不是 Self-Attention 定义。

---

## 101. 常见误解十一：Self-Attention 输出会把自己的信息完全覆盖

**错误。**

除了可以 self-attend，

Transformer 还有 residual connection：

$$
X+Attention(X)
$$

---

## 102. 常见误解十二：Attention Matrix 是 Learned Parameter

**错误。**

它是：

$$
A(X)
$$

每个 input forward 动态计算。

---

## 103. 常见误解十三：A_ij = A_ji

**一般错误。**

Attention 可以是 directional。

---

## 104. 常见误解十四：Full Self-Attention 的计算复杂度是 O(1)

**错误。**

O(1) 指最少 sequential operations。

总 per-layer compute 约：

$$
O(n^2d)
$$

---

## 105. 常见误解十五：Self-Attention 能直接知道 Image Token 的空间坐标

**错误。**

需要额外：

> positional / spatial information。

ACT 使用 2D positional encoding。

---

## 106. 用六步记住 Self-Attention

### Step 1：Input

$$
X
$$

---

### Step 2：Project

$$
\boxed{
Q=XW_Q,\quad
K=XW_K,\quad
V=XW_V
}
$$

---

### Step 3：Pairwise Compatibility

$$
\boxed{
S=QK^\top
}
$$

---

### Step 4：Scale + Mask

$$
\boxed{
\tilde S
=
\frac{S}{\sqrt{d_k}}
+
M
}
$$

没有 mask 时：

$$
M=0
$$

---

### Step 5：Row-Wise Softmax

$$
\boxed{
A
=
softmax(\tilde S)
}
$$

---

### Step 6：Read Values

$$
\boxed{
O=AV
}
$$

对于第 $i$ 个 token：

$$
\boxed{
o_i
=
\sum_j A_{ij}v_j
}
$$

这就是一整个 Self-Attention head。

---

## 107. 再加上 Transformer Block

真实 Transformer 不停在：

$$
O
$$

而是：

```text
X
↓
Multi-Head Self-Attention
↓
Residual + LayerNorm
↓
FFN
↓
Residual + LayerNorm
```

原始 2017 Transformer 使用：

> Post-LN。

现代模型可能使用：

> Pre-LN

或其他 normalization layout。

所以要区分：

> Self-Attention 核心机制

和：

> 完整 Transformer block architecture。

---

## 108. 一句话真正理解 Self-Attention

> **Self-Attention 让同一条 sequence 中的每个位置都先从自己的 hidden representation 生成一个 Query，同时所有位置也生成 Keys 和 Values；每个 Query 与所有 Keys 计算 compatibility，经 scaling 与 row-wise Softmax 得到一组读取权重，再对所有 Values 做加权汇总，因此每个位置都能在一层中根据自己的需求直接构造一个不同的全局 context representation。**

如果只记计算链：

$$
\boxed{
X
\rightarrow
Q,K,V
\rightarrow
QK^\top
\rightarrow
/\sqrt{d_k}
\rightarrow
Softmax
\rightarrow
AV
}
$$

如果只记一句直觉：

> **每个 Token 都拿着自己的 Query，去同一条 Sequence 的 Key–Value Memory 里读一次。**

---

## 109. 下一步：Multi-Head Attention

到这里我们只讨论了：

> 一个 Attention Head。

下一步最自然的问题是：

> **既然一个 Query 已经可以读取整条 sequence，为什么 Transformer 还一定要同时做很多个 heads？**

下一篇：

> **[Multi-Head Attention：为什么一个 Attention Head 不够？](./multi-head-attention.md)**

会详细解释：

- 为什么多个 heads 不是重复计算；
- 每个 head 怎样拥有自己的 $W_Q,W_K,W_V$；
- $d_{\text{model}}=512,h=8,d_k=64$ 的 shape 怎样变化；
- 为什么多个 64-D heads 的总计算量仍接近一个 512-D single-head；
- `Concat(head_1,...,head_h)W_O` 到底在做什么；
- 为什么 “head 1 学语法、head 2 学位置” 只是可能的实证现象，不是 architecture 保证；
- ACT 的 8-head Transformer 中多头机制怎样作用于 visual / joint / latent tokens。

---

### Primary Source

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,  
Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin.

**Attention Is All You Need.**  
Advances in Neural Information Processing Systems 30, 2017.

- arXiv: https://arxiv.org/abs/1706.03762
- PDF: https://arxiv.org/pdf/1706.03762
- Google Research: https://research.google/pubs/attention-is-all-you-need/

本文主要依据：

#### Section 3.2.1 — Scaled Dot-Product Attention

$$
\boxed{
Attention(Q,K,V)
=
softmax
\left(
\frac{
QK^\top
}{
\sqrt{d_k}
}
\right)V
}
$$

---

#### Section 3.2.3 — Applications of Attention in the Model

原论文明确说明：

> 在 Encoder Self-Attention 中，所有 queries、keys、values 都来自前一 Encoder layer 的同一个 output sequence。

因此：

> 每个 Encoder position 可以 attend 前一层的所有 positions。

Decoder Self-Attention 同样来自同一 decoder sequence，

但为了保持 autoregressive property，

会 mask 非法 future connections。

---

#### Section 4 — Why Self-Attention

原论文 Table 1 比较：

| Layer | Complexity per Layer | Sequential Operations | Max Path Length |
|---|---:|---:|---:|
| Self-Attention | $O(n^2d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(nd^2)$ | $O(n)$ | $O(n)$ |

因此本文严格区分：

- $O(1)$ sequential dependency；
- $O(n^2d)$ computational complexity。

---

### 与 ACT 的关系

Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.  
**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.**  
RSS 2023.

- https://arxiv.org/abs/2304.13705

ACT 中 Self-Attention 至少有三处重要作用：

#### CVAE Encoder

```text
[CLS]
joint token
action tokens
```

之间做 Self-Attention，使 `[CLS]` 可以汇总 demonstration action sequence。

#### Policy Encoder

```text
latent token
joint token
1200 visual tokens
```

共：

$$
1202
$$

个 tokens 做 Self-Attention，融合 observation context。

#### Policy Decoder

$k$ 个 future action slots 之间做 non-causal Self-Attention，

从而允许 action chunk 内不同 future positions 相互交流。

---

### 本文知识连接

#### 前置

- [Attention](./attention.md)
- [Query / Key / Value](./qkv.md)
- [Dot Product](./dot-product.md)
- [Softmax](./softmax.md)
- Matrix Multiplication

#### Transformer

- [Transformer](./transformer.md)
- [Multi-Head Attention](./multi-head-attention.md)
- [Cross-Attention](./cross-attention.md)
- [Scaled Dot-Product Attention](./attention.md)
- [Positional Encoding](./positional-encoding.md)
- [Causal Mask](./causal-mask.md)
- [Residual Connection](./residual-connection.md)
- [Layer Normalization](./layer-normalization.md)
- [Feed-Forward Network](./feed-forward-network.md)

#### 数学

- Permutation
- Permutation Matrix
- Equivariance

#### Robot Learning

- [ACT Architecture](../robot-learning/act/architecture.md)
- [CVAE in ACT](../robot-learning/act/cvae-in-act.md)
- [ACT Complete Data Flow](../robot-learning/act/complete-data-flow.md)

#### 下一步

- [Multi-Head Attention](./multi-head-attention.md)
