---
title: "Multi-Head Attention：为什么一个 Attention Head 不够？"
description: "从 single-head 的表示瓶颈出发，严格理解 Multi-Head Attention 如何用多套 Q/K/V 投影并行建立不同 matching subspaces，为什么 8×64 并不会简单带来 8 倍计算，以及 Concat(... )W_O 如何重新融合各个 head 的信息。"
status: reviewed
pageType: concept
canonical: /deep-learning/multi-head-attention
updated: "2026-09-15"
---

# Multi-Head Attention：为什么一个 Attention Head 不够？

在 [Self-Attention](./self-attention.md) 中，我们已经完整走过一次：

\[
X
\rightarrow
Q,K,V
\rightarrow
QK^\top
\rightarrow
\frac{QK^\top}{\sqrt{d_k}}
\rightarrow
\operatorname{Softmax}
\rightarrow
AV
\]

如果只有一个 Attention Head，

那么整个 sequence 中所有位置之间的关系，都必须通过同一套：

\[
W_Q,\;W_K,\;W_V
\]

来表达。

这看起来似乎已经足够强。

毕竟一个 token 已经可以：

> 一次读取整条 sequence。

那为什么 Transformer 原论文还要把它拆成多个 heads？

为什么不直接做：

\[
d_{\text{model}}=512
\]

的一次大 Attention？

为什么原始 Transformer Base 反而使用：

\[
h=8
\]

个 heads，

每个 head 只有：

\[
d_k=d_v=64
\]

维？

更奇怪的是：

> 8 个 heads 难道不会让计算量直接变成 8 倍吗？

还有：

\[
\operatorname{Concat}
(
head_1,\ldots,head_8
)
W^O
\]

为什么先拼起来以后，还要再乘一个：

\[
W^O
\]

？

这一篇只研究 Multi-Head Attention。

---

# 1. 先看原论文真正的动机

《Attention Is All You Need》在 Section 3.2.2 中明确写道：

> 与其使用一组 \(d_{\text{model}}\) 维 queries、keys、values 做一次 attention，作者发现，把 Q/K/V 用多组不同的 learned linear projections 投影到多个较低维空间，再并行做 Attention，会更有效。

原论文给出的核心理由是：

> Multi-Head Attention allows the model to jointly attend to information from different representation subspaces at different positions.

也就是：

\[
\boxed{
\text{不同 Heads}
\rightarrow
\text{不同 learned representation subspaces}
}
\]

并且论文还指出：

> 如果只有一个 head，attention 的 averaging 会抑制这种能力。

这句话是理解 Multi-Head Attention 的关键。

---

# 2. Single-Head Attention 有什么潜在限制？

先假设只有一组：

\[
W_Q,\;W_K,\;W_V
\]

那么所有 Query–Key pair 都在同一个 learned matching space 中计算：

\[
q_i^\top k_j
\]

最终一个 query 得到一组：

\[
\alpha_{ij}
\]

再计算：

\[
o_i
=
\sum_j
\alpha_{ij}v_j
\]

问题在于：

> 一个 token 和其他 token 之间可能同时存在多种不同类型的关系。

例如一句话里，一个 token 可能同时和其他位置存在：

- 局部邻近关系；
- 长距离依赖；
- 主谓关系；
- 指代关系；
- 语义关系；
- 位置关系。

如果只有一个 head，

所有这些关系都要挤在：

> 同一套 Q/K matching geometry

和：

> 同一个 Value aggregation

里。

---

# 3. 一个直觉例子

考虑：

```text
The animal didn't cross the street because it was too tired.
```

处理：

```text
it
```

时，

模型可能同时希望知道：

### 关系 A

`it` 指代谁？

可能需要读取：

```text
animal
```

---

### 关系 B

当前局部短语是什么？

可能需要读取：

```text
was
tired
```

---

### 关系 C

它处在句子的哪个结构位置？

可能需要读取其他位置模式。

如果只有一套 Attention，

最终只产生：

\[
\alpha_{it,1},
\ldots,
\alpha_{it,n}
\]

这一组权重，

所有关系最后都混进：

\[
\sum_j\alpha_{ij}v_j
\]

同一个 weighted sum。

---

# 4. Multi-Head 的基本想法

不要只做一次：

\[
Attention(Q,K,V)
\]

而是做：

\[
head_1
\]

\[
head_2
\]

\[
\cdots
\]

\[
head_h
\]

每个 head 都有自己独立的：

\[
W_i^Q,\;W_i^K,\;W_i^V
\]

因此：

\[
\boxed{
head_i
=
Attention(
QW_i^Q,
KW_i^K,
VW_i^V
)
}
\]

最后：

\[
\boxed{
MultiHead(Q,K,V)
=
Concat(
head_1,\ldots,head_h
)W^O
}
\]

这就是原论文的正式定义。

---

# 5. 每个 Head 到底不同在哪里？

假设输入 hidden representation：

\[
X
\]

对 head 1：

\[
Q_1=XW_1^Q
\]

\[
K_1=XW_1^K
\]

\[
V_1=XW_1^V
\]

对 head 2：

\[
Q_2=XW_2^Q
\]

\[
K_2=XW_2^K
\]

\[
V_2=XW_2^V
\]

因为：

\[
W_1^Q\neq W_2^Q
\]

\[
W_1^K\neq W_2^K
\]

\[
W_1^V\neq W_2^V
\]

所以即使输入：

\[
X
\]

完全一样，

两个 head 看到的 matching / message representations 也可以完全不同。

---

# 6. 因此每个 Head 都有自己的 Matching Space

回忆单头：

\[
score_{ij}
=
q_i^\top k_j
\]

第 \(r\) 个 head：

\[
score_{ij}^{(r)}
=
q_i^{(r)\top}k_j^{(r)}
\]

又因为：

\[
q_i^{(r)}
=
x_iW_r^Q
\]

\[
k_j^{(r)}
=
x_jW_r^K
\]

所以：

\[
\boxed{
score_{ij}^{(r)}
=
x_i
W_r^Q
W_r^{K\top}
x_j^\top
}
\]

不同 head 有不同：

\[
W_r^QW_r^{K\top}
\]

于是它们实际上学习：

> **不同的 pairwise compatibility geometry。**

---

# 7. 这就是“Different Representation Subspaces”的数学含义

原论文说：

> different representation subspaces。

不能只理解成：

> “每个 head 随机拿 64 个原始维度。”

实际上不是简单切片：

```text
Head 1 = 原始维度 1~64
Head 2 = 原始维度 65~128
...
```

而是每个 head 都有 learned projection：

\[
XW_r^Q
\]

因此 head 1 的第一个维度可能是：

> 原 512 个 features 的一种线性组合。

head 2 的第一个维度又可以是：

> 完全不同的线性组合。

所以：

\[
\boxed{
\text{subspace}
\neq
\text{fixed feature slice}
}
\]

它是：

> learned projection space。

---

# 8. 每个 Head 不只 Matching 不同，Value Space 也不同

第 \(r\) 个 head：

\[
V_r=XW_r^V
\]

所以即使两个 heads 最终注意到了同一个 token，

它们从这个 token 中读取的：

\[
v_j^{(1)}
\]

和：

\[
v_j^{(2)}
\]

也可以完全不同。

因此不同 heads 的差异有两层：

### 第一层：从哪里读

由：

\[
Q_rK_r^\top
\]

决定。

### 第二层：读到什么

由：

\[
V_r
\]

决定。

所以 Multi-Head 不是：

> “同一份 Attention Map 重复多次。”

---

# 9. 一个简单双头例子

假设 sequence：

```text
A B C
```

对于 token B，

Head 1 的 attention weights 可能是：

\[
[0.8,\;0.1,\;0.1]
\]

而 Head 2：

\[
[0.1,\;0.2,\;0.7]
\]

所以：

\[
head_1(B)
=
0.8v_A^{(1)}
+
0.1v_B^{(1)}
+
0.1v_C^{(1)}
\]

而：

\[
head_2(B)
=
0.1v_A^{(2)}
+
0.2v_B^{(2)}
+
0.7v_C^{(2)}
\]

同一个 token B：

> 可以同时从不同位置、不同 representation space 中读取不同信息。

---

# 10. 为什么不能让一个 Head 同时做到这些？

理论上一个足够强的 single head：

> 也可能表示复杂关系。

Multi-Head 并不是说：

> 单头数学上绝对不可能完成任务。

它是一种 architecture bias：

> 把多种并行关系显式分配到多个独立 projection / attention channels。

这样模型不需要强迫：

> 一张 attention distribution

同时承担所有信息路由需求。

原论文的实验发现这种设计有益。

---

# 11. “Single-Head Averaging Inhibits This”是什么意思？

单个 head 输出：

\[
o_i
=
\sum_j\alpha_{ij}v_j
\]

如果 query 同时需要从两个完全不同的位置读取不同类型的信息，

它们最终会先被混合进：

\[
o_i
\]

这个同一 vector。

Multi-Head 则可以先分别产生：

\[
head_1(i)
\]

\[
head_2(i)
\]

……

再：

\[
Concat
\]

也就是说：

> 多种读取结果先保持分离，

而不是过早地平均进同一个 representation。

这是理解论文那句话的一种重要直觉。

---

# 12. 原始 Transformer Base 的尺寸

原论文 Base model：

\[
\boxed{
d_{\text{model}}=512
}
\]

heads：

\[
\boxed{
h=8
}
\]

每个 head：

\[
\boxed{
d_k=d_v=
\frac{d_{\text{model}}}{h}
=
64
}
\]

所以：

```text
512-D hidden representation
        │
        ├── Head 1 → 64-D Q/K/V
        ├── Head 2 → 64-D Q/K/V
        ├── Head 3 → 64-D Q/K/V
        ├── ...
        └── Head 8 → 64-D Q/K/V
```

每个 head 输出：

\[
64
\]

维。

8 个拼起来：

\[
8\times64=512
\]

又回到：

\[
d_{\text{model}}
\]

---

# 13. 一个 Sequence 的完整 Shape

假设：

\[
B=32
\]

batch size，

\[
n=100
\]

sequence length，

\[
d_{\text{model}}=512
\]

输入：

\[
\boxed{
X\in
\mathbb R^{32\times100\times512}
}
\]

8 heads，

每头：

\[
64
\]

维。

---

# 14. Q Projection 后怎样分 Head？

实际实现常常不会写 8 个独立 `Linear(512,64)`。

更高效的方式是一次：

\[
XW_Q
\]

得到：

\[
[32,100,512]
\]

然后 reshape 成：

\[
\boxed{
[32,100,8,64]
}
\]

再 transpose：

\[
\boxed{
[32,8,100,64]
}
\]

于是维度含义：

```text
batch
head
sequence
head_dimension
```

K/V 同理。

---

# 15. 每个 Head 的 Attention Matrix

Q：

\[
[32,8,100,64]
\]

K：

\[
[32,8,100,64]
\]

对最后两个维度做：

\[
QK^\top
\]

得到：

\[
\boxed{
[32,8,100,100]
}
\]

也就是说：

- 每个 batch sample；
- 每个 head；

都有自己的一张：

\[
100\times100
\]

Attention Matrix。

---

# 16. 所以 8 Heads 真的有 8 张 Attention Map

对同一个 sample：

\[
A^{(1)},A^{(2)},\ldots,A^{(8)}
\]

每张：

\[
[n,n]
\]

因为它们 Q/K projections 不同，

所以：

\[
A^{(1)}
\neq
A^{(2)}
\]

通常成立。

这就是“多种并行读取模式”的直接表现。

---

# 17. 每个 Head 输出什么 Shape？

每个 head：

\[
A_r
\in
\mathbb R^{n\times n}
\]

对应：

\[
V_r
\in
\mathbb R^{n\times64}
\]

所以：

\[
head_r
=
A_rV_r
\]

得到：

\[
\boxed{
[n,64]
}
\]

8 个 heads：

\[
8\times[n,64]
\]

---

# 18. Concat 到底做什么？

对于同一个 token position \(i\)，

8 个 heads 分别输出：

\[
h_i^{(1)}
\in
\mathbb R^{64}
\]

\[
h_i^{(2)}
\in
\mathbb R^{64}
\]

……

\[
h_i^{(8)}
\in
\mathbb R^{64}
\]

Concat：

\[
\boxed{
[
h_i^{(1)};
h_i^{(2)};
\ldots;
h_i^{(8)}
]
}
\]

得到：

\[
512
\]

维。

注意：

> Concat 不是平均。

它把每个 head 的输出：

> 并排保留下来。

---

# 19. 为什么不把 Heads 直接相加？

如果直接：

\[
head_1+\cdots+head_8
\]

多个 representation subspaces 会立即混在一起。

Concat：

> 先保留每个 head 的独立结果。

然后交给：

\[
W^O
\]

学习怎样组合。

所以：

\[
\boxed{
Concat
=
\text{preserve head-specific outputs}
}
\]

---

# 20. 那 W^O 到底在做什么？

Concat 后：

\[
H
=
Concat(head_1,\ldots,head_h)
\]

shape：

\[
[n,h d_v]
\]

原始 Transformer：

\[
hd_v=512
\]

然后：

\[
\boxed{
O=HW^O
}
\]

其中：

\[
W^O
\in
\mathbb R^{hd_v\times d_{\text{model}}}
\]

原始 Base：

\[
W^O\in\mathbb R^{512\times512}
\]

它的作用是：

> **学习怎样重新混合不同 heads 提取出的信息，并映射回统一 \(d_{\text{model}}\) representation space。**

---

# 21. W^O 不是简单“恢复 Shape”

虽然：

\[
512\rightarrow512
\]

看起来维度没变，

但：

\[
W^O
\]

不是 identity。

它是 learned matrix。

所以它可以：

- 混合 Head 1 和 Head 3 的 feature；
- 抑制某些 head 的某些 dimensions；
- 重新组合不同 subspaces；
- 形成下一层真正需要的 representation。

因此：

\[
\boxed{
W^O
=
\text{learned head integration}
}
\]

而不仅仅是 reshape。

---

# 22. 一个两头小例子

假设每个 head 输出 2 维。

Token \(i\)：

\[
head_1(i)
=
[1,2]
\]

\[
head_2(i)
=
[10,20]
\]

Concat：

\[
[1,2,10,20]
\]

然后假设：

\[
W^O
=
\begin{bmatrix}
1&0\\
0&1\\
0.1&0\\
0&0.1
\end{bmatrix}
\]

那么：

\[
[1,2,10,20]W^O
=
[2,4]
\]

也就是：

> output projection 可以把不同 heads 的 features 学习性地重新融合。

真实模型当然维度更高、矩阵更复杂。

---

# 23. 为什么不是“每个 Head 最终投票”？

Multi-Head Attention 不是 ensemble voting。

各个 heads 不会各自输出一个最终答案：

```text
Head 1 说 cat
Head 2 说 dog
最后投票
```

而是各自输出：

> hidden representations。

然后：

\[
Concat + W^O
\]

融合成一个新的 hidden representation。

最终 task prediction 还在后面。

所以 Multi-Head 是：

> **representation-level parallel computation**

不是模型 ensemble。

---

# 24. 最关键的问题：8 个 Heads 会不会贵 8 倍？

直觉上：

> 做 8 次 Attention，好像应该是 8 倍。

但原论文专门说明：

> 因为每个 head 的维度减小，总计算成本与一个 full-dimensional single-head attention 相近。

我们来算。

---

# 25. 先看一个 Full 512-D Single Head

假设：

\[
d=d_{\text{model}}=512
\]

sequence length：

\[
n
\]

single head 直接使用：

\[
d_k=d_v=d
\]

QK score：

\[
QK^\top
\]

大约需要：

\[
\boxed{
O(n^2d)
}
\]

AV：

\[
A V
\]

又大约：

\[
\boxed{
O(n^2d)
}
\]

所以 Attention 核心大约：

\[
\boxed{
2n^2d
}
\]

量级。

---

# 26. 再看 h 个 Heads

每个 head：

\[
d_k=d_v=\frac dh
\]

每一个 head 的 QK：

\[
O
\left(
n^2\frac dh
\right)
\]

h 个 heads：

\[
h
\times
O
\left(
n^2\frac dh
\right)
\]

h 抵消：

\[
\boxed{
O(n^2d)
}
\]

AV 同理：

\[
\boxed{
O(n^2d)
}
\]

所以多头的 Attention 核心仍然是：

\[
\boxed{
O(n^2d)
}
\]

同数量级。

---

# 27. 为什么 h 正好抵消？

因为：

```text
Head 数量：
× h

每个 Head 宽度：
÷ h
```

于是总 feature width：

\[
h\times\frac dh=d
\]

没有变。

所以不能写成：

> “8 heads = 8 倍 full-width Attention。”

正确是：

> **8 个较窄的 Attention 并行替代 1 个较宽的 Attention。**

---

# 28. Q/K/V Projection 的成本呢？

Single full-width：

\[
W_Q,W_K,W_V
\in
\mathbb R^{d\times d}
\]

总 projection cost：

\[
\sim3nd^2
\]

Multi-Head 实际上也相当于：

\[
W_Q,W_K,W_V
:
d\rightarrow d
\]

再拆成 heads。

所以整体 projection cost 仍约：

\[
\sim3nd^2
\]

再加：

\[
W^O
\]

约：

\[
nd^2
\]

这也是标准 MHA 的主要 linear-projection cost。

---

# 29. 因此“计算成本相近”不是说完全一模一样

具体 wall-clock 还受：

- kernel implementation；
- memory layout；
- hardware；
- batch size；
- sequence length；
- fused operators；

影响。

原论文说的是：

> **总 computational cost similar to single-head attention with full dimensionality。**

所以更严谨的理解是：

> 理论主要数量级和总投影宽度相近，而不是任何硬件上 latency 必然完全相等。

---

# 30. 参数量会不会乘 h？

也不会简单乘 h。

如果每个 head 独立：

\[
W_i^Q
\in
\mathbb R^{d\times(d/h)}
\]

h 个：

\[
h\times d\times\frac dh
=
d^2
\]

所以所有 Query projections 加起来：

\[
d^2
\]

K：

\[
d^2
\]

V：

\[
d^2
\]

再加：

\[
W^O:
d^2
\]

总计大约：

\[
\boxed{
4d^2
}
\]

忽略 bias。

并不会因为：

\[
h=8
\]

变成：

\[
32d^2
\]

---

# 31. 原始 Base Transformer 的 MHA 参数量

\[
d=512
\]

单个：

\[
512\times512
=
262144
\]

Q/K/V 三组：

\[
3\times262144
=
786432
\]

再加：

\[
W^O
\]

：

\[
262144
\]

总计：

\[
\boxed{
1,048,576
}
\]

约：

> 1.05 million weights

忽略 bias。

8 heads 并没有让这个数字再乘 8。

---

# 32. 为什么代码里常只有一个 W_Q，而不是 8 个？

数学公式写：

\[
W_1^Q,\ldots,W_h^Q
\]

方便理解每个 head 有不同 projection。

实现中可以把它们拼成一个大矩阵：

\[
W^Q
\in
\mathbb R^{d\times d}
\]

一次计算：

\[
Q=XW^Q
\]

得到 512 维，

再 reshape：

\[
512
\rightarrow
8\times64
\]

这和分别计算 8 个：

\[
512\rightarrow64
\]

Linear 在数学上等价。

这样更适合 GPU。

---

# 33. 所以 “Head” 不一定对应代码里的一个独立 nn.Linear

这是代码阅读时很重要的点。

你可能看到：

```python
q_proj = nn.Linear(512, 512)
k_proj = nn.Linear(512, 512)
v_proj = nn.Linear(512, 512)
```

然后疑惑：

> “不是 8 heads 吗？为什么只有一个 q_proj？”

因为这个：

\[
512
\]

输出里已经包含：

\[
8\times64
\]

个 head channels。

随后 reshape 即可。

---

# 34. PyTorch 常见 Shape Flow

输入：

\[
X:
[B,N,D]
\]

例如：

\[
[32,100,512]
\]

Q projection：

\[
[B,N,D]
\]

reshape：

\[
[B,N,H,D_h]
\]

得到：

\[
[32,100,8,64]
\]

transpose：

\[
\boxed{
[32,8,100,64]
}
\]

然后：

\[
QK^\top
\]

得到：

\[
[32,8,100,100]
\]

Softmax 后乘 V：

\[
[32,8,100,64]
\]

再 transpose / concat：

\[
[32,100,512]
\]

最后：

\[
W^O
\]

仍是：

\[
[32,100,512]
\]

---

# 35. 为什么 Head 这个维度通常放在 Batch 后面？

因为计算时希望把：

\[
B\times H
\]

看成很多并行 attention problems。

GPU 可以并行计算：

> 每个 sample 的每个 head。

所以常见 layout：

\[
[B,H,N,D_h]
\]

非常自然。

但具体库也可能内部使用不同 memory layout。

---

# 36. 每个 Head 的 Softmax 是独立的吗？

是。

对于：

\[
A^{(r)}
=
softmax
\left(
\frac{
Q_rK_r^\top
}{
\sqrt{d_k}
}
\right)
\]

每一个 head \(r\) 都有自己的 score matrix 和自己的 Softmax。

所以：

> Head 1 的 attention weights 不需要和 Head 2 加起来等于 1。

每个 head 内：

\[
\sum_j
A_{ij}^{(r)}
=
1
\]

对固定 query \(i\) 成立。

---

# 37. Heads 之间在 Attention 阶段会互相交流吗？

在每个 head 独立计算：

\[
head_r
\]

的过程中：

> 不直接。

它们使用不同投影独立完成：

- score；
- Softmax；
- Value aggregation。

真正第一次显式融合发生在：

\[
\boxed{
Concat(\cdots)W^O
}
\]

之后。

下一层又会基于融合后的 hidden representation 重新产生新的 heads。

---

# 38. 所以 Multi-Head 是“先分，再合”

可以把整个过程压成：

```text
统一 hidden representation
        │
        ▼
   多套 projections
        │
 ┌──────┼──────┐
 ▼      ▼      ▼
Head1  Head2  ...
 │      │
不同 attention
不同 values
 │      │
 └──────┼──────┘
        ▼
      Concat
        ▼
       W_O
        ▼
统一 hidden representation
```

然后下一层可以再次：

> 分头处理。

---

# 39. 为什么这种结构有点像“多种视角”？

这是一个不错的直觉。

同一个 input：

\[
X
\]

经过不同 heads 的 projection，

可以被重新表示成：

> 多个不同的 view。

每个 view 都定义：

- 什么关系算匹配；
- 什么内容值得传递。

然后模型把这些不同 view 合并。

但必须注意：

> head 不保证自动对应人类命名好的“语法视角”“语义视角”。

它只是具有这种表示自由度。

---

# 40. 原论文有没有观察到不同 Heads 学不同东西？

有。

Transformer 原论文 Section 4 提到：

> 不同 attention heads 明显学到了不同的行为，一些似乎与句子的 syntactic / semantic structure 相关。

附录也展示了 attention visualizations。

所以：

> “heads 可以出现不同 specialization”

有原论文实验观察支持。

但这仍不等于：

> “第 3 个 head 永远就是 subject head。”

---

# 41. 为什么不能给每个 Head 强行贴固定标签？

因为 head behavior：

- 会随 layer 变化；
- 会随 training run 变化；
- 会随 input 变化；
- 可能同时承担多种作用；
- 某些 head 甚至可能冗余。

所以更严谨：

> Multi-Head 提供多个可独立学习的 representation/attention subspaces。

至于它们最终学成什么：

> 是 empirical outcome。

---

# 42. Head 数是不是越多越好？

不是。

如果：

\[
d_{\text{model}}
\]

固定，

增加：

\[
h
\]

意味着每头：

\[
d_h=\frac dh
\]

变小。

例如：

\[
d=512
\]

---

### 8 Heads

\[
d_h=64
\]

---

### 16 Heads

\[
d_h=32
\]

---

### 64 Heads

\[
d_h=8
\]

head 太窄时，

每个 matching/message space 的容量可能受限。

所以 head count 是：

> architecture hyperparameter。

不是越多越强。

---

# 43. Head 数和 Model Width 是两个不同概念

\[
d_{\text{model}}
\]

表示总 hidden width。

\[
h
\]

表示：

> 把 attention representation 分成多少个 parallel subspaces。

例如：

```text
Model A:
d_model = 512
h = 8
d_head = 64

Model B:
d_model = 512
h = 16
d_head = 32
```

总 width 相同，

但 attention factorization 不同。

---

# 44. 为什么 d_model 通常要能被 h 整除？

标准实现常令：

\[
d_h=
\frac{
d_{\text{model}}
}{
h
}
\]

因此为了 integer dimension：

\[
d_{\text{model}}
\bmod h
=
0
\]

最方便。

这不是 Multi-Head Attention 最一般数学定义的绝对要求，

而是标准实现设计。

原始论文就是这种配置。

---

# 45. Multi-Head 是否一定要求 d_k=d_v？

不一定。

公式允许：

\[
d_k\neq d_v
\]

只要：

Q/K 维度兼容 Dot Product，

而：

V 的输出维度符合后续 concat。

原始 Transformer 为了简单选择：

\[
d_k=d_v=64
\]

---

# 46. 为什么 Scaling 每个 Head 用 √d_k，而不是 √d_model？

因为每个 head 的 Dot Product 实际发生在：

\[
d_k
\]

维：

\[
q_r^\top k_r
=
\sum_{j=1}^{d_k}
q_jk_j
\]

其 variance 与：

\[
d_k
\]

有关。

所以 scale：

\[
\boxed{
1/\sqrt{d_k}
}
\]

而不是：

\[
1/\sqrt{d_{\text{model}}}
\]

---

# 47. 原始 Base 的 Scale 是多少？

每头：

\[
d_k=64
\]

所以：

\[
\sqrt{64}=8
\]

因此每个 head 实际计算：

\[
\boxed{
softmax
\left(
\frac{
Q_rK_r^\top
}{
8
}
\right)
}
\]

---

# 48. Multi-Head Attention 会产生 h 倍的 n×n Matrix

这一点仍然是真的。

单头 full attention：

\[
[n,n]
\]

多头：

\[
[h,n,n]
\]

所以 attention score / probability tensors 在 head dimension 上确实增加。

为什么总体 arithmetic 仍相近？

因为：

> 每个 head 的 feature dimension缩小。

但 attention matrix memory 本身：

\[
h n^2
\]

这一项确实带有 head factor。

实际优化实现会进一步考虑 memory behavior。

所以：

> “计算量相近”

不等于所有中间 tensor 的存储需求和实现常数完全相同。

---

# 49. 这一点为什么值得严谨区分？

原论文的“total computational cost similar”主要来自：

\[
h\times d_h=d
\]

的算术关系。

但现代 Transformer 性能分析还会受到：

- attention score tensor memory；
- kernel launch；
- bandwidth；
- fused attention；

等影响。

因此不能机械地说：

> “head 数完全不影响任何成本。”

更准确：

> 在经典参数化下，主要 FLOP 数量级保持相近，但实际内存与硬件效率仍会随实现改变。

---

# 50. Multi-Head Self-Attention

如果所有 heads 的：

\[
Q,K,V
\]

都来自：

\[
X
\]

那么就是：

\[
\boxed{
Multi\text{-}Head\ Self\text{-}Attention
}
\]

即：

\[
head_r
=
Attention(
XW_r^Q,
XW_r^K,
XW_r^V
)
\]

这就是 Transformer Encoder 的核心 attention sublayer。

---

# 51. Multi-Head Cross-Attention

如果 Decoder hidden states：

\[
H
\]

提供 Query，

Encoder Memory：

\[
M
\]

提供 K/V，

那么：

\[
\boxed{
head_r
=
Attention(
HW_r^Q,
MW_r^K,
MW_r^V
)
}
\]

就是：

> Multi-Head Cross-Attention。

每个 head 可以学习：

> Decoder 对 Encoder Memory 的不同读取方式。

---

# 52. 所以 Multi-Head 和 Self/Cross 是两个独立维度

不要混淆。

你可以有：

- Single-Head Self-Attention；
- Multi-Head Self-Attention；
- Single-Head Cross-Attention；
- Multi-Head Cross-Attention。

“Multi-Head”回答：

> 有几套并行 attention subspaces？

“Self/Cross”回答：

> Q 和 K/V 是否来自同一个 source？

---

# 53. 原始 Transformer 哪里用了 Multi-Head？

原论文明确列了三处：

### Encoder Self-Attention

Q/K/V 都来自 Encoder 前一层。

---

### Decoder Masked Self-Attention

Q/K/V 都来自 Decoder 当前 sequence，

但未来位置被 mask。

---

### Encoder–Decoder Attention

Q 来自 Decoder，

K/V 来自 Encoder output。

三处都使用：

> Multi-Head。

---

# 54. 为什么 Decoder Cross-Attention 特别适合 Multi-Head？

Decoder 当前 position 读取 source sentence 时，

可能同时需要：

- lexical correspondence；
- local phrase；
- long-distance context；
- syntax；
- positional alignment。

多头允许不同 subspaces：

> 并行读取同一 Encoder Memory。

所以 Multi-Head 不只用于 Self-Attention。

---

# 55. ACT 中 Multi-Head Attention 用在哪里？

ACT 原始配置使用：

\[
\boxed{
8\text{ attention heads}
}
\]

所以我们前面讲的：

- CVAE Transformer Encoder；
- Policy Transformer Encoder；
- Policy Transformer Decoder；

不是单头 Attention。

它们内部使用：

> Multi-Head Attention。

---

# 56. ACT Policy Encoder：1202 Tokens × 8 Heads

ACT Policy Encoder 的典型输入：

\[
1200
\]

visual tokens，

加：

\[
1
\]

joint token，

加：

\[
1
\]

latent token，

共：

\[
1202
\]

tokens。

hidden width：

\[
512
\]

8 heads，

每头：

\[
64
\]

维。

---

# 57. 每个 ACT Head 都有自己的 1202×1202 Attention Matrix

对于单个 sample：

\[
A_r
\in
\mathbb R^{1202\times1202}
\]

共有：

\[
8
\]

份：

\[
A_1,\ldots,A_8
\]

架构上，

一个 head 可以形成某种 visual–visual interaction，

另一个可以形成 visual–joint interaction，

另一个可能更多使用 latent token。

但：

> 这是可能性，不是未经分析就能断言的实际 specialization。

---

# 58. ACT 中为什么多头可能特别有用？

ACT observation memory 同时包含不同类型的信息：

```text
top camera
left wrist camera
right wrist camera
other camera
joint state
latent condition
```

这些 modality / spatial sources 之间可能存在多种不同关系。

单一 Attention distribution 需要把这些读取模式混在一起。

多头给模型：

> 多套独立的 matching/message subspaces。

这与原论文 Multi-Head 的通用动机一致。

---

# 59. ACT Decoder 中多个 Heads 又意味着什么？

第 \(i\) 个 future action slot 要 cross-attend observation memory。

对于同一个 action slot：

Head 1：

\[
A_i^{(1)}
\]

可能形成一种读取模式。

Head 2：

\[
A_i^{(2)}
\]

形成另一种。

它们分别得到：

\[
head_i^{(1)}
\]

\[
head_i^{(2)}
\]

……

最后 concat + \(W^O\)。

因此一个 future action representation 可以同时整合：

> 多种 observation-reading patterns。

---

# 60. Action Slot 自己也有 Multi-Head Self-Attention

ACT Decoder 不只有 Cross-Attention。

未来 action slots 之间还有 Self-Attention。

因此：

\[
slot_i
\]

可以通过不同 heads：

> 以多种 representation subspaces 和其他 future slots 交换信息。

这有助于模型形成：

> coordinated action chunk representation。

---

# 61. Multi-Head 是否让每个 Head 看不同 Token 子集？

不一定。

标准 Multi-Head 中每个 head 都可以访问：

> 同样的 allowed token positions。

区别是：

> 它们使用不同 Q/K/V projections，

因此得到不同 attention distributions。

所以不是预先规定：

```text
Head 1 只能看前 10 个 token
Head 2 只能看后 10 个 token
```

除非 architecture 另外加入这种限制。

---

# 62. Multi-Head 是否等于把 Sequence 切成 h 段？

完全不是。

切的是：

> representation dimension / projection subspace。

不是：

> sequence positions。

每个 head 通常仍然面对完整：

\[
n
\]

个 token。

所以：

\[
\boxed{
\text{split hidden channels}
\neq
\text{split sequence}
}
\]

---

# 63. 也不是把原始 512 Features 生硬分成 8 段

再次强调：

projection：

\[
XW_r^Q
\]

意味着每个 head 的 64 维 Q 都可以使用原始：

\[
512
\]

维 hidden state 的任意 learned linear combination。

所以 Head 1 并不是只拥有原 features 1–64。

它可以利用：

> 所有 512 维。

只是最后投影到自己的 64-D space。

---

# 64. 为什么这比简单 Feature Slicing 强？

如果只是：

```text
Head 1 → x[0:64]
Head 2 → x[64:128]
```

每个 head 无法主动选择：

> 哪些原始 features 最适合自己。

Learned projection：

\[
W_r
\]

让每个 head 自己学习：

> 怎样从完整 hidden representation 重新组合出一个适合自己的 subspace。

---

# 65. Heads 会不会学成完全一样？

可能部分冗余。

Architecture 并没有强制：

\[
A_1\neq A_2
\]

也没有一个显式 loss 说：

> 每个 head 必须完全不同。

它们只是：

> 参数独立，并且有机会 specialization。

实践研究后来确实发现某些 heads 可以被剪掉而模型性能变化有限。

所以更准确：

> Multi-Head 提供多样化容量，不保证所有 heads 都同等必要或完全独特。

---

# 66. 原论文有没有“Head Diversity Loss”？

没有。

原始 Transformer 并没有额外添加：

\[
L_{\text{diversity}}
\]

强迫 heads 不同。

各 heads 的 differentiation 是：

> 最终任务 loss 下自然学习的结果。

---

# 67. 如果所有 Heads 一样会怎样？

如果：

\[
W_1^Q=W_2^Q
\]

\[
W_1^K=W_2^K
\]

\[
W_1^V=W_2^V
\]

那么两个 heads 输出就会一样。

Concat 只会重复同一信息。

模型理论上仍能工作，

但浪费了多头带来的额外 representation capacity。

训练一般不会被硬约束成完全相同参数，

但可能出现某种程度冗余。

---

# 68. W^O 能不能完全忽略某个 Head？

可以。

如果最终训练发现某个 head 没用，

\[
W^O
\]

对应这部分输入的权重可以变小。

所以：

> Head 的贡献不仅由自己的 Attention 决定，

还由后续：

\[
W^O
\]

如何使用它决定。

---

# 69. 所以看 Attention Map 时不能忽略 W^O

看到某个 head 有很漂亮的 attention pattern，

不代表它对最终模型输出一定非常重要。

因为：

> 它产生的 representation 后面还会经过 \(W^O\)、Residual、FFN、多层网络。

所以 Attention Map 是：

> 一种中间行为观察。

不是完整 causal attribution。

---

# 70. Multi-Head Attention 的 Gradient 怎么训练各 Head？

最终输出：

\[
O
=
Concat(head_1,\ldots,head_h)W^O
\]

loss：

\[
L
\]

会通过：

\[
W^O
\]

回传到每个：

\[
head_r
\]

再分别回到：

\[
W_r^Q,
W_r^K,
W_r^V
\]

所以所有 heads：

> 由同一个最终任务 objective 联合训练。

它们不是各自有独立 target。

---

# 71. 为什么不同 Head 可能自然分工？

因为参数随机初始化不同，

而且：

\[
W^O
\]

可以联合利用不同 outputs。

优化过程中，如果不同 heads 捕获互补信息能降低 loss，

这种 configuration 就可能被强化。

但：

> 分工是优化结果，不是硬编码规则。

---

# 72. 一个比较深的视角：MHA 是多个低秩交互通道

单个 head 的 score：

\[
XW_QW_K^\top X^\top
\]

其中：

\[
W_QW_K^\top
\]

的 rank 最大受：

\[
d_k
\]

限制。

当：

\[
d_k=d/h
\]

时，每个 head 在一个较低维 interaction space 工作。

多个 heads：

> 提供多个不同的低维 bilinear interaction structures。

然后通过：

\[
W^O
\]

整合。

这是比“多个关注点”更数学化的理解。

---

# 73. 为什么说“低秩”要谨慎？

矩阵：

\[
W_QW_K^\top
\]

如果：

\[
W_Q,W_K
\in
\mathbb R^{d\times d_k}
\]

则：

\[
rank(W_QW_K^\top)\le d_k
\]

所以从 bilinear form 角度确实有 rank upper bound。

但整个 Attention 还有：

- Softmax 非线性；
- input-dependent interaction；
- 多 heads；
- 多 layers；

所以不能简单说：

> “整个 Multi-Head Attention 就是低秩模型。”

这里只是在解释每个 matching projection 的结构约束。

---

# 74. 一个 Head 64 维，会不会信息太少？

单个 head 只有：

\[
64
\]

维，

但它不是单独承担全部任务。

共有：

\[
8
\]

个 heads。

Concat 后仍回：

\[
512
\]

维。

所以设计思路是：

> 每个 head 处理一个较窄表示空间，多 head 合起来保持总模型宽度。

---

# 75. 为什么不做 8 个 512-D Heads 再 Concat 成 4096？

当然可以设计更大的架构。

但成本和参数会巨大增加。

原始 Transformer 的目的不是：

> 简单复制 8 倍 capacity。

而是在大致保持总计算规模的情况下：

> 给 Attention 多个 parallel subspaces。

所以选择：

\[
d_h=d/h
\]

非常关键。

---

# 76. 为什么 MHA 最后维度仍保持 d_model？

Transformer Block 有 Residual：

\[
X+\operatorname{MHA}(X)
\]

两者必须 shape 一致。

所以：

\[
\operatorname{MHA}(X)
\]

最终需要回到：

\[
d_{\text{model}}
\]

维。

这也是：

\[
W^O
\]

输出 dimension 设为：

\[
d_{\text{model}}
\]

的重要原因。

---

# 77. Residual 如何连接 MHA？

原始 Transformer：

\[
\boxed{
LayerNorm(
X+
MHA(X)
)
}
\]

因此 Multi-Head Attention 的最终 output：

\[
[B,N,d_{\text{model}}]
\]

与输入：

\[
X
\]

完全同 shape。

这使每层可以：

> 更新 representation，而不是改变 token count 或 hidden width。

---

# 78. MHA 会改变 Sequence Length 吗？

Self-Attention 中：

> 不会。

输入：

\[
N
\]

个 Queries，

输出仍：

\[
N
\]

个 representations。

Cross-Attention 则输出长度由：

> Query 数量

决定。

例如：

\[
N_q=k
\]

个 ACT action queries，

无论 encoder memory 有：

\[
1202
\]

个 tokens，

Cross-Attention 输出仍是：

\[
k
\]

个 decoder positions。

---

# 79. Multi-Head Cross-Attention 的 Matrix Shape

假设 ACT：

\[
N_q=100
\]

action slots，

memory：

\[
N_k=1202
\]

8 heads。

每头 attention score：

\[
[100,1202]
\]

全部 heads：

\[
\boxed{
[8,100,1202]
}
\]

忽略 batch。

每个 future action slot、每个 head：

> 都可以对 1202 个 observation memory positions 分配一组读取权重。

---

# 80. 这比“Action Query 看图片”更准确

实际上 Action Query 并不是只看 images。

它读取的是：

\[
\boxed{
\text{encoder memory}
}
\]

而 encoder memory 已经融合了：

- visual information；
- joint state；
- latent token；
- positional structure。

所以 Cross-Attention 的 K/V 是：

> contextualized observation representations。

不是 raw pixels。

---

# 81. Multi-Head Attention 和 Mixture of Experts 一样吗？

不一样。

MoE 通常：

> routing 到不同 expert networks。

MHA：

> 所有 heads 通常都并行计算。

没有标准 MoE 那种：

- top-k expert routing；
- expert-specific FFN blocks；
- load-balancing objective。

所以 Multi-Head 更像：

> parallel representation channels。

不是 expert routing system。

---

# 82. Multi-Head Attention 和 Ensemble 一样吗？

也不一样。

Ensemble：

> 多个相对独立模型产生 predictions，再融合。

MHA heads：

> 是一个单一网络 layer 内部的协同 representation components。

它们共享：

- input；
- downstream network；
- loss；

并通过：

\[
W^O
\]

直接联合。

所以：

\[
\boxed{
MHA\neq Model\ Ensemble
}
\]

---

# 83. Multi-Head Attention 和 CNN 多个 Channels 有点像吗？

作为直觉：

> 有一点。

CNN 的不同 output channels 可以学习不同 filters。

MHA 的不同 heads 也提供不同 learned channels。

但机制不同：

CNN filter：

> 通常固定 locality pattern。

Attention head：

> 动态生成 token-to-token weights。

所以只能作为：

> “并行 feature channels”

层面的类比。

---

# 84. 为什么 Multi-Head Attention 特别适合多模态？

不同 modality 的关系可能具有不同性质。

例如：

```text
text token ↔ image patch
joint token ↔ visual patch
action query ↔ wrist camera
```

多头允许：

> 不同 projection spaces 同时处理不同 interaction patterns。

这也是现代 multimodal Transformer 广泛使用 Multi-Head Attention 的原因之一。

---

# 85. 但 Head 并不知道“这是图像 Head”

除非 architecture 特别设计，

普通 MHA 没有显式标签：

```text
Head 1 = vision
Head 2 = proprioception
```

所有 heads 只是：

> 通过最终 loss 学习。

所以不能未经分析就把 head 赋予固定 modality role。

---

# 86. Head Count 是不是一种“解释性参数”？

主要不是。

它首先是：

> model architecture / capacity parameter。

虽然可以可视化不同 heads，

但选择：

\[
h=8
\]

不是为了让人类得到 8 个容易解释的规则。

原始论文选择多头主要为了：

> different representation subspaces + performance。

---

# 87. 为什么有些现代模型使用更多 Heads？

因为：

- model width 更大；
- task 更复杂；
- architecture scaling；
- hardware constraints；

不同模型会选择不同：

\[
h,d_h
\]

例如更大的 \(d_{\text{model}}\) 往往也配更多 heads。

但：

> “现代模型通常多少 heads”

是具体 model design 问题，

不是 Multi-Head Attention 的定义。

---

# 88. Multi-Query Attention 和 Multi-Head Attention 一样吗？

不是。

标准 MHA：

> 每个 query head 通常有自己对应的 K/V head projections。

Multi-Query Attention（MQA）则：

> 多个 query heads 共享一组 K/V。

它主要用于减少：

> autoregressive decoding 时 KV-cache memory / bandwidth。

这是后来的优化方向，

不属于 2017 原始 Transformer MHA 定义。

以后可以单独开页面。

---

# 89. Grouped-Query Attention 又是什么？

GQA 位于两者之间：

```text
MHA:
很多 Q heads
很多 K/V heads

MQA:
很多 Q heads
1 组 K/V

GQA:
很多 Q heads
较少的 K/V groups
```

这也是现代 LLM 推理优化中的重要结构。

但学习 Multi-Head Attention 时，

先掌握原始 MHA：

> 每头独立 Q/K/V projections

最重要。

---

# 90. 为什么这些现代变体反而证明 Q/K/V 的角色要分清？

因为 MQA / GQA 的核心就是：

> Query heads 与 Key/Value heads 的数量可以不同。

如果把 Q/K/V 都笼统理解成：

> “三个差不多的向量”

就很难理解为什么：

> K/V 可以共享，而 Q 保持多头。

所以 Q/K/V 的功能区分不是教学噱头。

它直接影响现代架构设计。

---

# 91. 常见误解一：8 Heads = 做 8 次完整 512-D Attention

**错误。**

原始 Base：

\[
8
\times
64\text{-D heads}
\]

而不是：

\[
8
\times
512\text{-D heads}
\]

---

# 92. 常见误解二：Head 1 使用原始 Features 1–64

**错误。**

每个 head 通过 learned projection：

\[
W_i
\]

从完整：

\[
512
\]

维 representation 生成自己的：

\[
64
\]

维空间。

---

# 93. 常见误解三：多个 Heads 只是重复算同一 Attention

**错误。**

每个 head 有不同：

\[
W_i^Q,W_i^K,W_i^V
\]

所以 matching、weights、Values 都可不同。

---

# 94. 常见误解四：每个 Head 一定学一种固定人类语义

**错误。**

Architecture 只提供多个 subspaces。

具体 specialization 是训练结果。

---

# 95. 常见误解五：Heads 最后直接平均

**错误。**

标准原始 Transformer：

\[
Concat(head_1,\ldots,head_h)W^O
\]

不是直接平均。

---

# 96. 常见误解六：W^O 只是为了改 Shape

**错误。**

它是 learned projection，

负责整合不同 heads 的 representations。

---

# 97. 常见误解七：Heads 越多一定越好

**错误。**

固定：

\[
d_{\text{model}}
\]

时，

head 越多，

每头：

\[
d_h
\]

越小。

存在 capacity / optimization trade-off。

---

# 98. 常见误解八：多头让 FLOPs 必然变 h 倍

**错误。**

标准配置：

\[
d_h=d/h
\]

所以 Attention arithmetic 总量保持同一数量级。

---

# 99. 常见误解九：Head 数完全不影响任何 Memory / Runtime

**也错误。**

FLOP 数量级相近不代表：

- 中间 attention tensor；
- memory traffic；
- kernel efficiency；

完全相同。

---

# 100. 常见误解十：MHA 把 Sequence 分成 h 段

**错误。**

通常每个 head 都看完整 allowed sequence。

拆的是：

> learned representation subspace。

---

# 101. 常见误解十一：Multi-Head = Ensemble

**错误。**

heads 是同一个 layer 内联合训练的 representation components。

---

# 102. 常见误解十二：Multi-Head = Multi-Query Attention

**错误。**

MQA 是后来的 KV-sharing 变体。

---

# 103. 常见误解十三：Attention Map 漂亮的 Head 一定最重要

**不一定。**

最终影响还取决于：

- V representation；
- \(W^O\)；
- Residual；
- 后续 layers。

---

# 104. 用四个步骤记住 Multi-Head Attention

## Step 1：多套 Projection

对于：

\[
r=1,\ldots,h
\]

\[
Q_r=QW_r^Q
\]

\[
K_r=KW_r^K
\]

\[
V_r=VW_r^V
\]

---

## Step 2：每个 Head 独立 Attention

\[
\boxed{
head_r
=
softmax
\left(
\frac{
Q_rK_r^\top
}{
\sqrt{d_k}
}
\right)V_r
}
\]

---

## Step 3：Concat

\[
\boxed{
H
=
Concat(
head_1,\ldots,head_h
)
}
\]

---

## Step 4：Output Projection

\[
\boxed{
O=HW^O
}
\]

这就是：

\[
\boxed{
MultiHead(Q,K,V)
}
\]

---

# 105. 原始 Transformer Base 的完整 Shape

\[
Q,K,V:
[n,512]
\]

经过每个 head：

\[
Q_r,K_r,V_r:
[n,64]
\]

共有：

\[
8
\]

个 heads。

每个：

\[
A_r:
[n,n]
\]

每个：

\[
head_r:
[n,64]
\]

Concat：

\[
[n,512]
\]

最后：

\[
W^O:
[512,512]
\]

输出：

\[
\boxed{
[n,512]
}
\]

所以整个 MHA 可以无缝接 residual：

\[
X+MHA(X)
\]

---

# 106. 一句话理解 Concat + W^O

> **每个 head 先在自己的 learned subspace 中独立回答“我从哪里读、读到什么”，Concat 保留所有 head 的回答，\(W^O\) 再学习怎样把这些不同视角重新组合成下一层所需要的统一 hidden representation。**

---

# 107. 一句话真正理解 Multi-Head Attention

> **Multi-Head Attention 不是把同一个 Attention 重复很多遍，而是让同一组输入通过多套独立的 Q/K/V learned projections，进入多个较低维的 matching 与 message subspaces，在这些空间中并行产生不同的 attention patterns 和 context representations，再通过 concatenation 与 learned output projection \(W^O\) 将这些互补信息整合回统一的 \(d_{\text{model}}\) 空间。**

而为什么不会简单贵：

\[
h
\]

倍？

因为标准 Transformer 令：

\[
\boxed{
d_k=d_v=\frac{d_{\text{model}}}{h}
}
\]

所以：

\[
h\times d_k=d_{\text{model}}
\]

多了 head 数，

同时缩小了每个 head 的宽度。

---

# 108. 下一步：Cross-Attention

现在我们已经理解了：

```text
Attention
↓
Q/K/V
↓
Dot Product
↓
Softmax
↓
Self-Attention
↓
Multi-Head Attention
```

接下来最自然的是：

> **[Cross-Attention：Decoder 到底怎样读取 Encoder Memory？](./cross-attention.md)**

这篇会把：

\[
Q
\]

和：

\[
K,V
\]

来自不同 source 的情况彻底讲透。

重点包括：

- 为什么 output 长度由 Query 数量决定；
- 为什么 Encoder Memory 同时产生 K/V；
- 为什么 Decoder Query 不等于最终公式里的 Q；
- 一个 query 对 1202 个 ACT memory tokens 的 shape 怎样计算；
- Cross-Attention 与 Self-Attention 的矩阵 shape 差异；
- 为什么 ACT action queries 可以读取 visual / joint / latent memory；
- 原始 Transformer 翻译 decoder 和 ACT action decoder 的 Cross-Attention 有什么共同点与区别。

---

## Primary Source

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,  
Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin.

**Attention Is All You Need.**  
Advances in Neural Information Processing Systems 30, 2017.

- arXiv: https://arxiv.org/abs/1706.03762
- HTML: https://arxiv.org/html/1706.03762
- Google Research: https://research.google/pubs/attention-is-all-you-need/

本文主要依据：

### Section 3.2.2 — Multi-Head Attention

原论文定义：

\[
\boxed{
MultiHead(Q,K,V)
=
Concat(head_1,\ldots,head_h)W^O
}
\]

其中：

\[
\boxed{
head_i
=
Attention(
QW_i^Q,
KW_i^K,
VW_i^V
)
}
\]

投影矩阵：

\[
W_i^Q
\in
\mathbb R^{
d_{\text{model}}\times d_k
}
\]

\[
W_i^K
\in
\mathbb R^{
d_{\text{model}}\times d_k
}
\]

\[
W_i^V
\in
\mathbb R^{
d_{\text{model}}\times d_v
}
\]

\[
W^O
\in
\mathbb R^{
hd_v\times d_{\text{model}}
}
\]

原始 Base Transformer：

\[
\boxed{
h=8
}
\]

\[
\boxed{
d_k=d_v=d_{\text{model}}/h=64
}
\]

\[
\boxed{
d_{\text{model}}=512
}
\]

论文明确指出：

> Multi-head attention 允许模型同时关注不同位置、不同 representation subspaces 中的信息。

并指出因为每个 head 维度降低，

> 总 computational cost 与一个 full-dimensional single-head attention 相近。

---

## 与 ACT 的关系

Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.  
**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.**  
RSS 2023.

- https://arxiv.org/abs/2304.13705

ACT 原始 Transformer 配置使用：

\[
\boxed{
8\text{ attention heads}
}
\]

hidden dimension：

\[
512
\]

因此和原始 Transformer Base 一样，

每个标准 head 可理解为在：

\[
64
\]

维 attention subspace 中工作。

ACT 中 Multi-Head Attention 用于：

- CVAE Encoder 的 `[CLS] / joint / action-token` interactions；
- Policy Encoder 的 visual / joint / latent token interactions；
- Policy Decoder 的 action-slot Self-Attention；
- Policy Decoder 对 observation memory 的 Cross-Attention。

---

## 本文知识连接

### 前置

- [Attention](./attention.md)
- [Query / Key / Value](./qkv.md)
- [Self-Attention](./self-attention.md)
- [Dot Product](./dot-product.md)
- [Softmax](./softmax.md)

### Transformer

- [Transformer](./transformer.md)
- [Cross-Attention](./cross-attention.md)
- [Scaled Dot-Product Attention](./attention.md)
- [Residual Connection](./residual-connection.md)
- [Layer Normalization](./layer-normalization.md)
- [Feed-Forward Network](./feed-forward-network.md)

### 后续扩展

- Multi-Query Attention
- Grouped-Query Attention
- KV Cache
- FlashAttention

### Robot Learning

- [ACT Architecture](../robot-learning/act/architecture.md)
- [ACT Complete Data Flow](../robot-learning/act/complete-data-flow.md)

### 下一步

- [Cross-Attention](./cross-attention.md)
