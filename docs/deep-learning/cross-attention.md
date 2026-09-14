---
title: "Cross-Attention：Decoder 到底怎样读取 Encoder Memory？"
description: "从 Q 与 K/V 来自不同 source 这一核心出发，严格理解 Cross-Attention 的矩阵形状、输出长度、Decoder Query、Encoder Memory、位置编码与 ACT 中 k 个 action queries 读取 1202 个 observation tokens 的完整流程。"
status: reviewed
pageType: concept
canonical: /deep-learning/cross-attention
updated: "2026-09-15"
---

# Cross-Attention：Decoder 到底怎样读取 Encoder Memory？

在前面的文章中，我们已经理解：

- [Attention](./attention.md)：Attention 是动态信息读取；
- [Query / Key / Value](./qkv.md)：Q/K 决定“从哪里读”，V 决定“读到什么”；
- [Self-Attention](./self-attention.md)：同一条 sequence 内部的位置彼此读取；
- [Multi-Head Attention](./multi-head-attention.md)：多套不同 Q/K/V subspaces 并行读取。

现在只剩下 Transformer Encoder–Decoder 结构里最重要的一块：

> **Cross-Attention。**

Self-Attention 的输入来源是同一个 sequence：

\[
Q=XW_Q,\qquad
K=XW_K,\qquad
V=XW_V
\]

Cross-Attention 则不一样：

\[
\boxed{
Q=HW_Q
}
\]

\[
\boxed{
K=MW_K
}
\]

\[
\boxed{
V=MW_V
}
\]

其中：

- \(H\)：正在“发起读取请求”的 decoder-side representations；
- \(M\)：已经由 Encoder 建好的 memory。

所以 Cross-Attention 最核心的一句话是：

> **Decoder 拿自己的 Query，去读取 Encoder 建好的 Key–Value Memory。**

原始 Transformer 论文 Section 3.2.3 直接说明：

> encoder-decoder attention 中，queries 来自前一个 decoder layer，而 memory keys 和 values 来自 encoder output。

ACT 论文也直接写道：

> transformer decoder conditions on encoder output through cross-attention，keys 和 values 来自 encoder。

但“Q 来自 Decoder，K/V 来自 Encoder”只是第一层理解。

真正掌握 Cross-Attention，还要回答：

1. 为什么 K 和 V 都来自 Encoder Memory？
2. Encoder Memory 本身是不是就等于 K 和 V？
3. Decoder Query Embedding 是不是就是公式里的 \(Q\)？
4. 为什么 Cross-Attention 输出长度由 Query 数量决定？
5. 如果 Query 有 100 个、Memory 有 1202 个，Attention Matrix 为什么是 \(100\times1202\)？
6. Decoder 到底“读回”什么？
7. Self-Attention 和 Cross-Attention 的计算公式明明一样，为什么功能不同？
8. ACT 里的 \(k\) 个 action slots 到底如何读取视觉、joint 和 latent 信息？
9. 为什么 action query 本身一开始甚至可以没有真实 observation content？
10. Cross-Attention 为什么特别适合“固定数量的输出槽位读取可变长度 Memory”？

这篇把这些问题完整拆开。

---

# 1. 从 Encoder–Decoder 的分工开始

Transformer 原论文沿用经典 Encoder–Decoder 架构。

Encoder 先处理输入 sequence：

\[
x_1,\ldots,x_n
\]

得到：

\[
\boxed{
M=
[m_1,\ldots,m_n]
}
\]

我们把它叫：

> **Encoder Memory**

然后 Decoder 负责产生：

\[
y_1,\ldots,y_m
\]

关键问题是：

> Decoder 怎么利用 Encoder 已经理解好的输入信息？

如果 Encoder 和 Decoder 完全断开，

Decoder 只能根据自己内部状态生成输出。

那显然无法完成：

```text
英文句子
→
中文翻译
```

所以必须有一条：

\[
\boxed{
Decoder
\rightarrow
Encoder Memory
}
\]

的信息读取路径。

这就是 Cross-Attention。

---

# 2. 为什么叫 Cross-Attention？

因为 Query 和 Key/Value：

> **跨越两个不同 representation sources。**

例如：

```text
Decoder sequence
      │
      └── Query source

Encoder sequence
      │
      ├── Key source
      └── Value source
```

所以：

\[
\boxed{
\text{Cross}
=
\text{cross between two representation sets}
}
\]

不是说：

> “用了两个 Attention。”

也不是：

> “Q 和 K 做叉乘。”

它仍然使用普通 Scaled Dot-Product Attention。

区别只是：

> Q 和 K/V 的来源不同。

---

# 3. Cross-Attention 的公式其实一点没变

仍然是：

\[
\boxed{
Attention(Q,K,V)
=
softmax
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
}
\]

唯一变化是：

\[
Q
\]

来自：

\[
H
\]

而：

\[
K,V
\]

来自：

\[
M
\]

所以可以展开：

\[
\boxed{
Q=HW_Q
}
\]

\[
\boxed{
K=MW_K
}
\]

\[
\boxed{
V=MW_V
}
\]

最终：

\[
\boxed{
CrossAttention(H,M)
=
softmax
\left(
\frac{
(HW_Q)(MW_K)^\top
}{
\sqrt{d_k}
}
\right)
(MW_V)
}
\]

这就是 Cross-Attention 的核心数学形式。

---

# 4. Self-Attention 和 Cross-Attention 的真正区别

## Self-Attention

\[
Q=XW_Q
\]

\[
K=XW_K
\]

\[
V=XW_V
\]

同一个：

\[
X
\]

同时扮演：

- query source；
- memory source。

---

## Cross-Attention

\[
Q=HW_Q
\]

\[
K=MW_K
\]

\[
V=MW_V
\]

其中：

\[
H\neq M
\]

一般代表不同 representation sets。

所以可以压缩成：

\[
\boxed{
SelfAttention:
\text{read from myself}
}
\]

\[
\boxed{
CrossAttention:
\text{read from another memory}
}
\]

---

# 5. 一个最简单的翻译例子

输入英文：

```text
I love cats
```

Encoder 输出：

\[
M=
[
m_{\text{I}},
m_{\text{love}},
m_{\text{cats}}
]
\]

Decoder 正在生成中文。

假设当前 Decoder representation：

\[
h_i
\]

对应：

> “现在需要决定下一个目标词。”

这个：

\[
h_i
\]

生成 Query：

\[
q_i=h_iW_Q
\]

而 Encoder Memory 中：

\[
m_{\text{I}},
m_{\text{love}},
m_{\text{cats}}
\]

分别产生：

\[
k_j=m_jW_K
\]

\[
v_j=m_jW_V
\]

于是：

\[
q_i
\]

与所有：

\[
k_j
\]

比较。

如果当前目标应该对应：

```text
cats
```

那么模型可能学出：

\[
q_i^\top k_{\text{cats}}
\]

比较高。

最后从：

\[
v_{\text{cats}}
\]

读取更多信息。

---

# 6. 为什么 Encoder Memory 同时产生 Key 和 Value？

回忆 QKV 的分工：

### Key

回答：

> “我是否和当前 Query 匹配？”

### Value

回答：

> “如果你决定读取我，我真正应该传递什么？”

Encoder Memory 中每个：

\[
m_j
\]

就是一个 candidate information source。

所以它自然同时产生：

\[
k_j=m_jW_K
\]

和：

\[
v_j=m_jW_V
\]

这两个 representation。

---

# 7. Encoder Memory 本身不等于 K

这是一个非常常见的概念偷懒。

论文会说：

> keys and values come from encoder output。

这指的是：

> **K/V 的 source representation 是 encoder output。**

严格计算上：

\[
M
\]

还会经过 Multi-Head Attention 内部的：

\[
W_K
\]

和：

\[
W_V
\]

才得到真正公式里的：

\[
K,V
\]

所以：

\[
\boxed{
M\neq K
}
\]

一般也：

\[
\boxed{
M\neq V
}
\]

更准确是：

\[
M
\rightarrow
\begin{cases}
MW_K=K\\
MW_V=V
\end{cases}
\]

---

# 8. 为什么这个区别重要？

因为它解释了：

> 同一份 Encoder Memory，为什么既能用来“被匹配”，又能用来“传内容”。

如果直接把：

\[
M
\]

同时当 K 和 V，

这两个功能被迫使用同一 representation。

独立：

\[
W_K,W_V
\]

让模型可以学习：

- 怎样建立地址 / 匹配空间；
- 怎样建立消息 / 内容空间。

所以：

> “K/V 来自 Encoder”不等于“K=V=Encoder output”。

---

# 9. Decoder Hidden State 也不等于 Q

同样：

\[
H
\]

只是：

> Query source representation。

真正进入 Attention Dot Product 的：

\[
Q
\]

还要经过：

\[
W_Q
\]

即：

\[
\boxed{
Q=HW_Q
}
\]

所以：

```text
Decoder representation
```

不等于：

```text
Attention Q matrix
```

它只是 Q 的来源。

---

# 10. 为什么这在 ACT 特别容易混淆？

ACT 里还有一个名字：

> `query_embed`

或者：

> action query / query position。

这又是一个“Query”词。

但它依然不等于最终公式里的：

\[
Q
\]

更准确的流程是：

```text
action query embedding
        │
        ▼
decoder-side representation
        │
     + query position
        │
        ▼
MultiHeadAttention 内部 W_Q
        │
        ▼
真正用于 QKᵀ 的 Q
```

所以要区分：

### Action Query Embedding

是 Decoder 的：

> query-side input / positional identity。

### Attention Q

是：

> 进入 scaled dot-product 的 learned projection。

---

# 11. Cross-Attention 最大的 Shape 规律

假设：

\[
H
\in
\mathbb R^{N_q\times d_{\text{model}}}
\]

有：

\[
N_q
\]

个 Queries。

Encoder Memory：

\[
M
\in
\mathbb R^{N_m\times d_{\text{model}}}
\]

有：

\[
N_m
\]

个 memory tokens。

经过 projection：

\[
Q
\in
\mathbb R^{N_q\times d_k}
\]

\[
K
\in
\mathbb R^{N_m\times d_k}
\]

\[
V
\in
\mathbb R^{N_m\times d_v}
\]

---

# 12. Score Matrix Shape

\[
QK^\top
\]

shape：

\[
[N_q,d_k]
\times
[d_k,N_m]
\]

所以：

\[
\boxed{
QK^\top
\in
\mathbb R^{N_q\times N_m}
}
\]

这张矩阵非常重要。

它表示：

> 每一个 Query，对每一个 Memory position 的 compatibility。

---

# 13. 为什么不是 N_m × N_m？

因为不是：

> Memory 自己查询 Memory。

那是 Self-Attention。

Cross-Attention 中有：

\[
N_q
\]

个发起请求的人，

有：

\[
N_m
\]

个可以被读取的 memory slots。

所以 pair 数量：

\[
\boxed{
N_q\times N_m
}
\]

而不是：

\[
N_m^2
\]

---

# 14. Attention Weight Matrix 每一行是什么意思？

固定一个 query：

\[
i
\]

第 \(i\) 行：

\[
A_{i,:}
\]

长度：

\[
N_m
\]

表示：

> 这个 Query 怎样在所有 Encoder Memory positions 之间分配读取权重。

所以：

\[
\sum_{j=1}^{N_m}
A_{ij}=1
\]

---

# 15. 每一列是什么意思？

固定 memory position：

\[
j
\]

第 \(j\) 列：

\[
A_{:,j}
\]

表示：

> 所有不同 Queries 分别有多大程度读取这个 memory token。

但是不同 Query rows：

> 独立 Softmax。

所以列不需要：

\[
\sum_iA_{ij}=1
\]

---

# 16. Cross-Attention 输出 Shape

接下来：

\[
A
\in
\mathbb R^{N_q\times N_m}
\]

乘：

\[
V
\in
\mathbb R^{N_m\times d_v}
\]

得到：

\[
\boxed{
O
=
AV
\in
\mathbb R^{N_q\times d_v}
}
\]

最关键的结论出现了：

\[
\boxed{
\text{Cross-Attention output length}
=
\text{number of Queries}
}
\]

不是 Memory 长度。

---

# 17. 为什么输出长度由 Query 决定？

因为每一个 Query：

\[
q_i
\]

发起一次读取。

它得到：

\[
o_i
=
\sum_{j=1}^{N_m}
A_{ij}v_j
\]

所以：

> 一个 Query → 一个 output representation。

有：

\[
N_q
\]

个 Queries，

就得到：

\[
N_q
\]

个 outputs。

Memory 有多少 token，只决定：

> 每个 Query 可以从多少候选项读取。

---

# 18. 一个数据库类比

假设数据库有：

\[
10000
\]

条 records。

你同时发：

\[
5
\]

个查询。

最终得到：

> 5 个查询结果 representations。

不会因为数据库有 10000 条记录，

就得到 10000 个 query outputs。

所以：

\[
\boxed{
\text{Memory size}
=
\text{number of things available to read}
}
\]

\[
\boxed{
\text{Query count}
=
\text{number of outputs requested}
}
\]

这是 Cross-Attention 最值得记住的 shape intuition。

---

# 19. 一个具体 Shape Example

假设：

\[
N_q=2
\]

Decoder 有两个 Queries。

Encoder Memory：

\[
N_m=3
\]

有三个 tokens。

每个 head：

\[
d_k=2
\]

所以：

\[
Q:
[2,2]
\]

\[
K:
[3,2]
\]

\[
V:
[3,d_v]
\]

那么：

\[
QK^\top:
[2,3]
\]

Softmax：

\[
A:
[2,3]
\]

最终：

\[
AV:
[2,d_v]
\]

所以：

> 2 个 Query 输出 2 个 representations。

---

# 20. 手算一个极小 Cross-Attention

设：

\[
q_1=[1,0]
\]

\[
q_2=[0,1]
\]

三个 Keys：

\[
k_1=[1,0]
\]

\[
k_2=[0,1]
\]

\[
k_3=[1,1]
\]

于是：

\[
Q=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
\]

\[
K=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix}
\]

---

# 21. QKᵀ

\[
QK^\top
=
\begin{bmatrix}
1&0&1\\
0&1&1
\end{bmatrix}
\]

第一行：

> Query 1 对三个 Memory Keys 的 scores。

第二行：

> Query 2 对三个 Memory Keys 的 scores。

---

# 22. Scale + Softmax

如果：

\[
d_k=2
\]

则除：

\[
\sqrt2
\]

第一行：

\[
[0.707,0,0.707]
\]

Softmax 约：

\[
[0.401,0.198,0.401]
\]

第二行：

\[
[0,0.707,0.707]
\]

Softmax 约：

\[
[0.198,0.401,0.401]
\]

所以：

\[
A
\approx
\begin{bmatrix}
0.401&0.198&0.401\\
0.198&0.401&0.401
\end{bmatrix}
\]

---

# 23. 再定义 Values

例如：

\[
v_1=[10,0]
\]

\[
v_2=[0,10]
\]

\[
v_3=[5,5]
\]

那么：

\[
o_1
=
0.401v_1
+
0.198v_2
+
0.401v_3
\]

而：

\[
o_2
=
0.198v_1
+
0.401v_2
+
0.401v_3
\]

虽然：

> 两个 Queries 读取同一套 Memory，

但因为 Query 不同，

它们获得不同 Context。

这就是 Cross-Attention。

---

# 24. Cross-Attention 本质是 Query-Conditioned Memory Read

Self-Attention：

> 每个 token 用自己的 Query 读取同一条 sequence。

Cross-Attention：

> Decoder 中每个位置，用自己的 Query 读取 Encoder Memory。

所以它最本质的抽象是：

\[
\boxed{
\text{Query-conditioned read from external memory}
}
\]

或者：

\[
\boxed{
\text{Learned differentiable retrieval}
}
\]

---

# 25. 原始 Transformer Decoder 一层里发生什么？

原论文 Decoder 每一层主要有：

1. Masked Self-Attention；
2. Encoder–Decoder Cross-Attention；
3. FFN。

所以当前 decoder representation 不是一上来就读取 Encoder。

它先通过 masked Self-Attention：

> 整合已经生成的 target-side context。

然后再通过 Cross-Attention：

> 去读取 source-side Encoder Memory。

---

# 26. 翻译时为什么先 Self，再 Cross 很自然？

假设已经生成：

```text
我 喜欢
```

Decoder 先通过 Self-Attention 理解：

> “目标句目前生成到哪里、当前语法状态是什么？”

形成：

\[
h_i
\]

然后：

\[
h_i
\]

作为 Query source 去读取：

```text
I love cats
```

的 Encoder Memory。

所以可以理解：

```text
我现在已经生成了什么？
↓
Decoder Self-Attention

根据当前生成状态，
我现在需要从源句读取什么？
↓
Cross-Attention
```

---

# 27. Cross-Attention 不会修改 Encoder Memory 本身

对于一个 Decoder layer：

\[
M
\]

作为 K/V source。

Cross-Attention 计算：

\[
O
=
A(MW_V)
\]

输出更新 Decoder representation。

Encoder Memory：

\[
M
\]

本身不会因为某个 Query 读取它就被“消耗”或改变。

不同 Decoder positions 可以反复读取同一 Memory。

这和普通数据库 read 很像。

---

# 28. 所有 Decoder Queries 可以读取同一个 Memory

例如：

\[
q_1,q_2,\ldots,q_m
\]

全部面对：

\[
k_1,\ldots,k_n
\]

和：

\[
v_1,\ldots,v_n
\]

但是每个 Query 有自己的一行：

\[
A_{i,:}
\]

所以同一份 Encoder Memory 可以被不同 Decoder positions：

> 以不同方式解释和读取。

---

# 29. 为什么这比把 Encoder 压成一个 Context Vector 强？

旧式固定 context：

\[
c
\]

整个 Decoder 都只能用：

> 同一个输入摘要。

Cross-Attention 保留：

\[
M=
[m_1,\ldots,m_n]
\]

每个 Query 都重新生成：

\[
c_i
=
\sum_jA_{ij}v_j
\]

所以：

\[
\boxed{
c_i
\text{ depends on query }i
}
\]

这继承了 Bahdanau Attention 的核心思想：

> 每个 Decoder step 动态选择 source information。

---

# 30. Cross-Attention 和 Bahdanau Attention 的关系

Bahdanau Attention：

\[
e_{ij}
=
a(s_{i-1},h_j)
\]

然后：

\[
c_i
=
\sum_j\alpha_{ij}h_j
\]

本质也是：

> Decoder state 去读取 Encoder annotations。

Transformer Cross-Attention 把 matching function换成：

\[
\frac{q_i^\top k_j}{\sqrt{d_k}}
\]

并且：

- 矩阵化；
- Multi-Head；
- 与完整 Transformer layer 结合。

所以 Cross-Attention 不是突然出现的新思想。

它是经典 Encoder–Decoder Attention 的现代矩阵化形式。

---

# 31. Multi-Head Cross-Attention

真实 Transformer 不是只有一个 Cross-Attention head。

第 \(r\) 个 head：

\[
Q_r=HW_r^Q
\]

\[
K_r=MW_r^K
\]

\[
V_r=MW_r^V
\]

然后：

\[
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
\]

最终：

\[
Concat(head_1,\ldots,head_h)W^O
\]

所以一个 Decoder Query 可以：

> 同时用多种 learned matching subspaces 读取同一 Encoder Memory。

---

# 32. 为什么多个 Head 对 Cross-Attention 很自然？

例如翻译一个词时，

一个 head 可能更多捕获：

- lexical alignment；

另一个：

- long-range context；

另一个：

- syntax；

另一个：

- positional pattern。

但再次强调：

> 这是可能的 empirical specialization，不是架构硬编码。

架构真正保证的只是：

> 多套不同 learned Q/K/V spaces 可以并行存在。

---

# 33. Position 信息在 Cross-Attention 里怎么用？

Cross-Attention不仅要知道：

> Memory token 内容是什么，

通常也要知道：

> 它来自哪个位置。

原始 Transformer 中 Encoder representation 本身已经经过 positional encoding。

所以：

\[
M
\]

带有位置上下文。

---

在 DETR 风格实现中，

还经常显式把：

\[
pos
\]

加到 Memory side 的 Key representation：

```python
key = memory + pos
value = memory
```

Query side 也可能：

```python
query = tgt + query_pos
```

这是一种非常常见的实现形式。

---

# 34. 为什么 Position 常加在 Q/K 上，而 Value 可以不加？

这是一个很好的设计直觉。

Q/K 决定：

> **怎么匹配位置。**

所以把 positional identity 加到 Q/K：

> 直接影响 routing。

V 则主要承担：

> 内容传输。

因此某些实现让：

```text
Q/K:
content + position

V:
content
```

但这不是 Cross-Attention 的唯一理论定义。

不同 architecture 可以有不同 positional design。

---

# 35. 现在回到 ACT

ACT Policy 先把 observation 编成：

\[
\boxed{
M\in\mathbb R^{1202\times512}
}
\]

概念上这 1202 个 positions 来自：

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

latent token。

ACT 论文明确写：

\[
1202\times512
\]

作为 Transformer Encoder 输入。

Encoder 输出随后成为 Decoder Cross-Attention 的 Memory。

---

# 36. ACT Decoder 有多少 Queries？

如果 Action Chunk 长度：

\[
k
\]

那么 Decoder 输出：

\[
k
\]

个 action representations。

所以 Query positions 数量：

\[
\boxed{
N_q=k
}
\]

论文正文描述 Decoder 输入 sequence 为：

\[
\boxed{
k\times512
}
\]

的 position embeddings。

因此：

> 一个 future action slot 对应一个 Decoder Query position。

---

# 37. ACT Cross-Attention 的 Shape

忽略 batch 和 heads：

Decoder query-side：

\[
H
\in
\mathbb R^{k\times512}
\]

Encoder Memory：

\[
M
\in
\mathbb R^{1202\times512}
\]

每个 head 若：

\[
d_k=64
\]

则：

\[
Q_r
\in
\mathbb R^{k\times64}
\]

\[
K_r
\in
\mathbb R^{1202\times64}
\]

\[
V_r
\in
\mathbb R^{1202\times64}
\]

---

# 38. ACT 每个 Head 的 Cross-Attention Matrix

\[
Q_rK_r^\top
\]

shape：

\[
\boxed{
k\times1202
}
\]

所以：

> 每个 future action slot，都对 1202 个 observation-memory positions 有一整行 scores。

如果：

\[
k=100
\]

则每个 head：

\[
\boxed{
100\times1202
}
\]

---

# 39. 8 Heads 时是什么 Shape？

忽略 batch：

\[
\boxed{
[8,\;100,\;1202]
}
\]

如果 batch：

\[
B
\]

则通常概念 shape：

\[
\boxed{
[B,\;8,\;100,\;1202]
}
\]

这意味着：

- B 个 samples；
- 8 种 attention subspaces；
- 100 个 future action slots；
- 每个 slot 在 1202 个 observation memory positions 上分配 weights。

---

# 40. 为什么最终输出还是 k × 512？

每一个 Query：

> 聚合 1202 个 Values，

最终得到一个：

\[
64
\]

维 head output。

8 heads：

\[
8\times64=512
\]

所以：

\[
k
\]

个 Queries 最终：

\[
\boxed{
k\times512
}
\]

再经过 action head：

\[
512\rightarrow14
\]

所以：

\[
\boxed{
k\times14
}
\]

这就是 ACT action chunk。

---

# 41. 这解释了一个非常重要的问题

为什么 ACT Encoder Memory 有：

\[
1202
\]

个 tokens，

但 Decoder 最终只输出：

\[
k
\]

个 actions？

因为：

> Cross-Attention 不是把 Memory 每个 token 都变成一个输出。

输出 count 由：

\[
\boxed{
Query count
}
\]

决定。

所以：

\[
1202\text{ memory positions}
\]

只是：

> 每个 action query 可以读取的候选信息库。

而：

\[
k\text{ action queries}
\]

才决定：

> 输出有 \(k\) 个 positions。

---

# 42. 一个 Action Query 可以读哪些信息？

架构允许它读取 Encoder Memory 中：

- top camera features；
- front camera features；
- left wrist camera features；
- right wrist camera features；
- joint-state information；
- latent-style information。

但这里必须说得严谨：

> Encoder Memory 已经经过 Self-Attention contextualization。

所以某一个 memory token 不一定还只是：

> “一个纯 camera patch”。

它已经可能包含从其他 tokens 聚合来的 context。

因此 Decoder 读取的是：

\[
\boxed{
\text{contextualized observation memory}
}
\]

---

# 43. 为什么 Action Query 不直接读 Raw Images？

因为图片先经过：

\[
ResNet18
\]

得到视觉 feature maps。

再加入 position information，

再进入 Transformer Encoder。

Encoder 又通过 Self-Attention 融合多视角、joint、latent context。

所以 Cross-Attention 面对的是：

> 高层 representation memory。

不是：

> raw RGB pixels。

---

# 44. ACT 的 Cross-Attention 可以怎样直觉理解？

第：

\[
i
\]

个 action slot 可以理解为：

> “为了决定 chunk 中第 \(i\) 个未来动作，我现在需要从 observation memory 读取什么？”

于是：

\[
q_i
\]

和：

\[
k_1,\ldots,k_{1202}
\]

比较。

得到：

\[
\alpha_{i1},\ldots,\alpha_{i,1202}
\]

然后：

\[
o_i
=
\sum_{j=1}^{1202}
\alpha_{ij}v_j
\]

形成第 \(i\) 个 future action slot 的 observation-conditioned representation。

---

# 45. 但不要说“第 i 个 Query 就是第 i 个 Action”

还要更精确。

Decoder Query position：

> 对应 action chunk 的一个 output slot identity。

经过多层：

- Decoder Self-Attention；
- Cross-Attention；
- FFN；

得到：

\[
h_i^{decoder}
\]

再经过：

\[
action\_head
\]

才真正得到：

\[
\hat a_i\in\mathbb R^{14}
\]

所以 Query slot 是：

> action output position 的 latent representation slot。

不是一开始就已经是 14-D action。

---

# 46. 为什么不同 Future Action Slots 会读不同 Observation？

因为：

\[
q_0\neq q_1\neq\cdots
\]

一般成立。

所以即使 K/V memory 相同：

\[
A_{0,:}
\]

和：

\[
A_{20,:}
\]

也可以完全不同。

例如架构上允许：

- 较近 future slot 更依赖当前 joint / wrist visual；
- 更远 future slot 使用更全局 observation structure。

但是否真的这样学出来：

> 需要实证分析。

不能从 architecture 直接断言。

---

# 47. Decoder Query 之间还会先 Self-Attend

ACT Transformer Decoder layer 不只是：

> Cross-Attention。

它还有 action slots 之间的：

> Self-Attention。

所以第 \(i\) 个 action slot 的 decoder representation 在读取 Encoder Memory 前/层间，

还可以和其他：

\[
k-1
\]

个 future action slots 交换信息。

这让 action chunk 可以形成：

> sequence-level coordination。

---

# 48. 因此 ACT Decoder 有两种信息流

## Action-to-Action

通过：

> Decoder Self-Attention。

表示：

\[
slot_i
\leftrightarrow
slot_j
\]

---

## Observation-to-Action

通过：

> Cross-Attention。

表示：

\[
memory_j
\rightarrow
slot_i
\]

所以一个 action slot 同时受：

- 其他 future action slots；
- 当前 observation memory；

影响。

---

# 49. 为什么这比直接 MLP(images)→actions 更有结构？

因为 Decoder 明确拥有：

\[
k
\]

个 output slots。

它们：

- 可以彼此交互；
- 可以分别查询同一 observation memory；
- 最终各自变成一个 action vector。

所以 architecture 直接把：

> action sequence structure

编码进了 computation graph。

---

# 50. ACT 论文里的一个重要 Paper Fact

ACT Section IV-C 明确写：

> Transformer decoder 通过 cross-attention 条件化在 Encoder output 上。

并且：

- input sequence 是 \(k\times512\) 的 fixed position embedding；
- keys 和 values 来自 Encoder；
- Decoder output 是 \(k\times512\)；
- 再 down-project 到 \(k\times14\)。

这是论文层面的描述。

---

# 51. ACT Appendix C 进一步说明

附录写：

> Encoder output 被用作 Transformer Decoder cross-attention 的 keys 和 values。

并说：

> 第一层的 queries 是 fixed sinusoidal embeddings。

这一点非常具体。

---

# 52. 但 Released Code 有一个值得记录的差异

当前官方仓库：

```python
self.query_embed =
    nn.Embedding(num_queries, hidden_dim)
```

也就是说：

> released code 中 query embeddings 是 learned embeddings。

随后：

```python
self.query_embed.weight
```

被传进 Transformer。

所以：

\[
\boxed{
\text{Paper description}
\neq
\text{current released-code detail}
}
\]

在这一点上需要明确区分。

---

# 53. 这会改变 Cross-Attention 原理吗？

不会。

无论 query-side positional representation 是：

- fixed sinusoidal；
- learned embedding；

最终 Decoder layer 中仍然会形成：

\[
\text{decoder-side Query representation}
\]

并通过 Multi-Head Attention 内部的：

\[
W_Q
\]

得到真正的：

\[
Q
\]

Encoder Memory 仍然提供：

\[
K,V
\]

所以 Cross-Attention 的数学核心不变。

---

# 54. 为什么必须把 Paper 与 Code 区分，而不是选一个说法硬写？

因为 XunBlog 的原则应该是：

> **Paper Fact 和 Implementation Fact 分开。**

这类差异很常见。

论文描述的是：

> 方法层面 / 某个实验版本。

开源代码可能：

- 有后续修改；
- 继承其他 codebase；
- 更换 position embedding 实现；
- 存在实现细节偏差。

所以最严谨的写法是：

```text
Original ACT paper:
fixed position embeddings / appendix says fixed sinusoidal first-layer queries

Released repository:
nn.Embedding(num_queries, hidden_dim)
```

而不是把其中一个悄悄改写成另一个。

---

# 55. ACT Released Code 的 Decoder Cross-Attention 更具体是什么样？

DETR-style Transformer Decoder 中典型代码逻辑是：

```python
query =
    tgt + query_pos

key =
    memory + pos

value =
    memory
```

然后：

```python
multihead_attn(
    query=query,
    key=key,
    value=value
)
```

这体现了前面提到的设计：

> position information 主要参与 Q/K matching；

> content memory 作为 V 被传递。

---

# 56. 这里的 tgt 又是什么？

DETR-style Decoder 一开始通常：

\[
tgt=0
\]

也就是每个 query slot 的 content state 最初是 zero-like representation。

Query identity 主要来自：

\[
query\_pos
\]

随着 Decoder layers 进行，

`tgt` 会被：

- Self-Attention；
- Cross-Attention；
- FFN；

不断更新。

所以第一层前：

> query position 提供 slot identity；

后续层：

> slot 已经包含逐步累积的 learned content。

---

# 57. 为什么“固定 Query Slot”仍然能输出不同样本的不同动作？

因为 Query slot embedding 本身虽然对所有 samples 可以相同，

但 Cross-Attention 读取的：

\[
M
\]

每个样本都不同。

所以：

\[
o_i
=
Attention(q_i,M)
\]

会随 observation 改变。

即：

\[
q_i
\]

提供：

> “我是第 i 个 output slot。”

而：

\[
M
\]

提供：

> “当前这个具体样本发生了什么。”

最终 action 自然随 observation 变化。

---

# 58. 这和 DETR Object Queries 很像

ACT 的 Transformer 部分直接受到 DETR-style query decoder 设计影响。

DETR 中：

> 一组 object queries 去读取 image encoder memory，形成 object predictions。

ACT 中：

> 一组 action queries 去读取 robot observation memory，形成 action predictions。

所以可以做一个高层类比：

```text
DETR:
object queries
→ image memory
→ object slots

ACT:
action queries
→ observation memory
→ action slots
```

但二者任务、loss、输出语义不同。

---

# 59. Query Slot 的数量为什么固定？

ACT chunk size：

\[
k
\]

固定。

所以 Decoder 需要：

\[
k
\]

个 output positions。

这正好可以由：

\[
k
\]

个 query slots 表示。

因此：

\[
\boxed{
\text{num queries}
=
\text{action chunk length}
}
\]

在 ACT 中非常自然。

---

# 60. Cross-Attention 会不会决定动作顺序？

Cross-Attention 自己主要负责：

> 每个 Query slot 从 Memory 读取什么。

不同 action slot 的 identity / order 还依赖：

- query embeddings / positional identity；
- Decoder Self-Attention；
- architecture；
- training target alignment。

所以不能说：

> “Cross-Attention 本身创造了 action timestep 顺序。”

它只是让已经区分开的 output slots：

> 读取 observation memory。

---

# 61. 为什么 Query Positions 必须彼此可区分？

如果所有：

\[
k
\]

个 Query slots 初始完全相同，

并且 network 对它们完全对称，

它们就很容易产生相同输出。

所以需要某种：

> slot identity / positional distinction。

例如：

- sinusoidal position embeddings；
- learned query embeddings。

这样模型知道：

```text
slot 0
slot 1
slot 2
...
```

不是同一个 position。

---

# 62. 为什么 action slots 不需要把 Ground-Truth Previous Action 喂进去？

因为 ACT 不是原始语言 Transformer 的 autoregressive decoding。

语言翻译：

> Decoder target positions 会输入 shifted previous tokens。

ACT：

> 使用固定 output query slots，并行预测整个 action chunk。

所以：

\[
a_{t+1}
\]

的预测并不要求先把：

\[
\hat a_t
\]

作为 token 再输入下一步。

---

# 63. Cross-Attention 和 Autoregressive 没有必然关系

Cross-Attention 只是：

> 两个 representation sets 之间的 Attention。

它可以出现在：

- autoregressive decoder；
- non-autoregressive decoder；
- object detector；
- vision-language model；
- robot policy。

所以：

\[
\boxed{
CrossAttention
\neq
Autoregressive
}
\]

原始 Transformer translation decoder 是 autoregressive，

ACT decoder 不是。

---

# 64. 为什么 ACT 能一次输出整个 Action Chunk？

因为它拥有：

\[
k
\]

个 action query positions。

这些 query slots：

- 并行存在；
- 可以彼此 Self-Attend；
- 都可以 Cross-Attend observation memory。

所以一次 Decoder forward 就得到：

\[
k
\]

个 output representations。

最后同时投影：

\[
k\times512
\rightarrow
k\times14
\]

不需要 token-by-token rollout。

---

# 65. Cross-Attention 和 Self-Attention 的输出长度对比

## Self-Attention

Query source：

\[
X
\]

Memory source：

\[
X
\]

所以：

\[
N_q=N_m=n
\]

输出：

\[
n
\]

个 positions。

---

## Cross-Attention

Query：

\[
N_q
\]

Memory：

\[
N_m
\]

可以完全不同。

输出：

\[
\boxed{
N_q
}
\]

个 positions。

这是 Cross-Attention 在架构设计里非常强的一点。

---

# 66. Memory 可以很长，Query 可以很短

例如：

\[
N_m=10000
\]

但只用：

\[
N_q=32
\]

个 learned queries。

Cross-Attention 就可以把：

> 大量 Memory

读取成：

> 少量 output slots。

这类结构在很多模型里都会出现。

它是一种：

> Query-based information bottleneck / extraction mechanism。

---

# 67. 反过来也可以 Query 比 Memory 多

理论上：

\[
N_q>N_m
\]

完全合法。

多个 Queries 可以：

> 以不同方式反复读取同一 Memory。

例如同一张图像 Memory，

可能产生：

> 很多 output slots。

Cross-Attention 不要求：

\[
N_q=N_m
\]

---

# 68. 这就是为什么它适合“输入长度 ≠ 输出长度”的问题

Self-Attention 通常保持：

\[
n\rightarrow n
\]

Cross-Attention 可以自然：

\[
N_m
\rightarrow
N_q
\]

例如：

```text
source sentence 20 tokens
→
target decoder positions

1202 observation tokens
→
100 action slots
```

这种 length mismatch 不需要额外 hack。

---

# 69. Cross-Attention 的复杂度

单头主要 score computation：

\[
QK^\top
\]

成本约：

\[
\boxed{
O(N_qN_md)
}
\]

而不是 Self-Attention 的：

\[
O(n^2d)
\]

因为两侧长度可以不同。

---

# 70. ACT 的 Cross-Attention Complexity 直觉

如果：

\[
k=100
\]

Memory：

\[
1202
\]

则每个 head 需要考虑：

\[
100\times1202
=
120200
\]

个 query-key pairs。

相比 Encoder Self-Attention：

\[
1202^2
=
1,444,804
\]

pairs。

所以在这个具体 shape 下：

> policy encoder 的 full Self-Attention pair grid 更大。

但实际总成本还受：

- layers；
- heads；
- feature dimension；
- implementation；

影响。

---

# 71. Cross-Attention Weight 是不是“哪个 Camera 最重要”？

不能直接这样解释。

某个 action query 对某个 memory token 权重高，

只能说：

> 当前 head/layer/query 在这次 attention operation 中较多读取该 memory value。

但 Memory token 已经 contextualized。

而且最终还有：

- 多个 heads；
- 多层；
- residual；
- FFN；
- output head。

所以不能把一张 Cross-Attention Map直接当成：

> “模型最终决策的重要性地图”。

---

# 72. 但 Cross-Attention Map 仍然有分析价值

它可以帮助观察：

- 不同 action slots 是否关注不同 spatial regions；
- 不同 heads 是否有不同模式；
- wrist camera 是否在某些动作阶段更常被读取；
- joint/latent token 是否获得高权重。

但这种分析应该叫：

> intermediate attention behavior。

如果要做严格 attribution，

需要更多：

- ablation；
- intervention；
- gradient-based methods；

来验证。

---

# 73. 为什么 Cross-Attention 能连接不同 Modality？

Q/K/V 只要求：

> 最终进入 compatible hidden spaces。

Query 可以来自：

- text；
- action slots；
- latent queries。

Memory 可以来自：

- image；
- audio；
- robot state；
- another text sequence。

只要 projections：

\[
W_Q,W_K,W_V
\]

把它们映射到 compatible dimensions，

就可以做：

\[
QK^\top
\]

所以 Cross-Attention 天然适合：

> multimodal fusion。

---

# 74. Vision-Language 模型里经常怎样用？

例如：

```text
Text queries
↓
Cross-Attention
↓
Image memory
```

或者：

```text
Learned visual queries
↓
Cross-Attention
↓
Image encoder features
```

ACT 只是同一抽象在 robotics 中的一个实例：

```text
Action queries
↓
Cross-Attention
↓
Observation memory
```

---

# 75. 为什么 Cross-Attention 比简单 Concatenation 更动态？

如果只是：

\[
[H;M]
\]

简单拼起来再 MLP，

模型需要自己隐式学：

> 每个 output slot 应该取哪些 memory positions。

Cross-Attention 显式提供：

\[
A_{ij}
\]

这套：

> Query-dependent content routing。

所以每个 Query：

> 都能拥有独立的 memory reading distribution。

---

# 76. Cross-Attention 是不是“查表”？

直觉上接近：

> differentiable soft lookup。

但它不是硬查：

\[
j^\star=\arg\max_j score
\]

而是：

\[
o_i
=
\sum_jA_{ij}v_j
\]

所以通常一次读取多个 memory values。

这让它：

- 可微；
- 平滑；
- 可以融合多个信息源。

---

# 77. Memory Key 是地址，Value 是内容——这个类比在哪最准确？

假设每个 memory entry：

\[
m_j
\]

生成：

\[
(k_j,v_j)
\]

可以直觉化：

```text
k_j:
我怎样被找到

v_j:
找到我之后我给你什么
```

Query：

```text
q_i:
我想找什么
```

Cross-Attention：

```text
q_i
↓
和所有 k_j 匹配
↓
得到权重
↓
按权重读取 v_j
```

这就是它最经典的 memory-retrieval解释。

---

# 78. 但 Key 不是离散地址

真正数据库地址可能是：

```text
record #12345
```

Key 则是：

> 高维 learned vector。

匹配不是：

```text
== 
```

而是：

\[
q^\top k
\]

所以这是：

> **content-addressable soft memory**

不是传统随机访问内存。

---

# 79. Cross-Attention 的 Memory 会随着 Decoder Layer 变化吗？

在原始 Transformer 中，

同一 Decoder stack 通常读取：

> 同一个最终 Encoder output \(M\)。

但每一 Decoder layer 有自己不同的：

- Multi-Head parameters；
- Query representations。

所以虽然 source Memory 相同，

各层实际得到的：

\[
K,V
\]

projections 和 Attention weights 可以不同。

---

# 80. 同一 Memory 为什么可以被一层层“重新解释”？

因为第 \(l\) 层：

\[
Q^{(l)}
\]

来自当前 decoder hidden states。

而这些 hidden states 已经经过前面层。

所以：

\[
Q^{(1)}
\neq
Q^{(2)}
\]

通常成立。

同时每层：

\[
W_K^{(l)},W_V^{(l)}
\]

也通常不同。

所以每一层都可以：

> 从不同 representation perspective 重新读取同一 Encoder Memory。

---

# 81. 为什么这比一次 Cross-Attention 更强？

一次读取可能只得到初步 context。

后续：

```text
Cross-Attention
↓
Residual
↓
FFN
↓
下一层新的 Query
↓
再次 Cross-Attention
```

模型可以逐层 refinement。

就像：

> 第一次检索后形成新的理解，再带着新的理解重新查 Memory。

这是深层 Decoder 的一个很有用的直觉。

---

# 82. ACT Decoder 也可以这样理解

第一个 Decoder layer 的 action slot：

> 先根据 query identity 读取 observation。

得到第一版：

\[
h_i^{(1)}
\]

之后：

- action-slot self-attention；
- cross-attention；
- FFN；

继续更新。

下一层：

\[
h_i^{(2)}
\]

已经包含更多：

> observation-conditioned action context。

然后再次读 Memory。

所以动作 representation 是逐层 refined 的。

---

# 83. 但关于 Released ACT Decoder Layer Output 要谨慎

官方当前仓库的 Transformer/DETR adaptation 曾有社区 issue 讨论：

> `hs = self.transformer(...)[0]`

是否意外取了 decoder stack 的第一层输出，而不是最后一层。

这是 released-code specific 的潜在实现问题讨论，

不是 ACT Cross-Attention 理论的一部分。

因此知识主页面不应该把社区 issue 当成算法定义。

如果以后写：

> “ACT Official Code Walkthrough”

可以专门分析这一实现细节。

---

# 84. Cross-Attention 需要 Mask 吗？

可以需要。

一般接口允许：

- `memory_mask`
- `memory_key_padding_mask`

等。

例如 Memory 有 padding positions：

> Query 不应该读取 PAD。

那么对应 logits 加：

\[
-\infty
\]

Softmax 后：

\[
0
\]

---

# 85. ACT Observation Memory 通常需要 Causal Mask 吗？

不需要。

因为 Encoder Memory 是：

> 当前 observation representation。

Action query 读取当前完整 observation，

不存在：

> “看未来 target token 泄漏答案”

这种语言 autoregressive问题。

所以 Cross-Attention 本身不需要语言 decoder 那种 causal mask。

---

# 86. Cross-Attention 的 Causality 和 Decoder Self-Attention 要分开

原始 Transformer：

### Decoder Self-Attention

需要 causal mask。

### Encoder–Decoder Cross-Attention

通常可以读取完整 source sequence。

所以：

\[
\boxed{
\text{Causal Mask}
\neq
\text{Cross-Attention 的必需组成}
}
\]

它取决于任务的信息可见性约束。

---

# 87. ACT Decoder 更进一步

ACT Action Slots 自身的 Self-Attention也是：

> non-causal。

Cross-Attention 也读取完整 current observation memory。

因此整个 ACT action decoder：

> 不采用语言生成那种 target-side autoregressive information restriction。

这和一次并行输出整个 action chunk一致。

---

# 88. Cross-Attention 和 Temporal Ensemble 完全是不同层级

ACT 内部：

\[
\boxed{
CrossAttention
}
\]

发生在：

> 一次 policy forward 内。

它解决：

> action slots 怎样读取当前 observation memory。

---

Temporal Ensemble：

\[
\boxed{
\text{Temporal Ensemble}
}
\]

发生在：

> 多个 timestep 的 policy forwards 之后。

它解决：

> 多个 overlapping chunks 对同一个 execution timestep 的预测怎样融合。

所以：

```text
Cross-Attention
= intra-forward representation routing

Temporal Ensemble
= inter-forward action aggregation
```

不要混淆。

---

# 89. Cross-Attention 和 Action Chunking 也不是同一件事

Action Chunking 定义：

\[
o_t
\rightarrow
[a_t,\ldots,a_{t+k-1}]
\]

即：

> 输出任务是一个 action sequence。

Cross-Attention 是实现这个 policy architecture 时的一种：

> observation-to-action-slot 信息路由机制。

所以：

\[
\boxed{
ActionChunking
=
\text{prediction target design}
}
\]

\[
\boxed{
CrossAttention
=
\text{neural architecture mechanism}
}
\]

---

# 90. 如果不用 Cross-Attention 能不能做 Action Chunking？

当然可以。

例如 MLP、CNN、RNN 也可以：

> 一次输出 \(k\) 个动作。

ACT 选择 Transformer Encoder–Decoder，

因为它适合：

- sequence representations；
- multimodal context；
- query-based structured outputs。

所以 Cross-Attention 是 ACT 的实现设计，

不是 Action Chunking 的数学必需条件。

---

# 91. 为什么 Query-based Decoder 很适合结构化输出？

因为可以预先定义：

\[
N_q
\]

个 output slots。

每个 slot：

- 有自己的 identity；
- 可以彼此 self-attend；
- 可以从同一 Memory 中按需读取；
- 最终通过 shared head 变成目标输出。

因此这种 Decoder 不只适合：

> language。

也适合：

- objects；
- actions；
- keypoints；
- trajectories；
- multimodal latent slots。

---

# 92. 一个最重要的 Shape 公式

只要记住：

\[
Q:
[N_q,d_k]
\]

\[
K:
[N_m,d_k]
\]

\[
V:
[N_m,d_v]
\]

那么：

\[
\boxed{
QK^\top:
[N_q,N_m]
}
\]

然后：

\[
\boxed{
softmax(QK^\top)V:
[N_q,d_v]
}
\]

整个 Cross-Attention shape 就不会再乱。

---

# 93. 为什么 K 和 V 必须有相同 Memory Length？

因为第：

\[
j
\]

个 Key：

\[
k_j
\]

对应第：

\[
j
\]

个 Value：

\[
v_j
\]

Attention score：

\[
A_{ij}
\]

决定：

> 对第 \(j\) 个 memory entry 的 Value 读取多少。

所以 Key–Value pairs 数量必须对齐：

\[
\boxed{
N_K=N_V=N_m
}
\]

但：

\[
d_k
\]

和：

\[
d_v
\]

不要求相同。

---

# 94. 为什么 Query Length 可以完全不同？

因为 Query 不需要和 Memory 一一对应。

每个 Query只是提出：

> 一个读取请求。

所以：

\[
N_q
\]

可以是：

- 1；
- 10；
- 100；
- 1000；

完全取决于：

> 你希望得到多少 output representations。

---

# 95. 一个 Query 可以读取所有 Memory Positions 吗？

如果没有 mask：

> 可以。

Cross-Attention score row：

\[
A_{i,:}
\]

覆盖：

\[
1,\ldots,N_m
\]

所有 Memory positions。

所以它具有：

> global receptive field over memory。

---

# 96. 为什么这对 ACT 多摄像头很重要？

某个 future action slot 不需要预先绑定：

```text
只看 wrist camera
```

它可以在一次 Cross-Attention 中读取：

- top view；
- front view；
- wrist views；
- joint / latent contextual information。

到底怎样组合：

> 由 learned weights 动态决定。

这很适合 fine manipulation 中：

> 不同动作阶段需要不同视觉来源

的情形。

---

# 97. 但为什么 Encoder 已经 Self-Attend 了，Decoder 还要 Cross-Attend？

Encoder Self-Attention 负责：

> 把 observation tokens 相互融合，形成 contextualized memory。

但它不知道：

> 具体第 17 个 future action slot 现在最需要什么。

Decoder Cross-Attention 提供：

> output-slot-specific retrieval。

所以两者分工：

\[
\boxed{
EncoderSelfAttention:
\text{build contextual memory}
}
\]

\[
\boxed{
DecoderCrossAttention:
\text{read memory for each output slot}
}
\]

---

# 98. 为什么不直接把 Encoder Memory 平均一下再给所有 Action Slots？

如果平均：

\[
c=
\frac1{N_m}
\sum_jm_j
\]

所有 action slots 都只能拿到：

> 同一个固定 global summary。

Cross-Attention 则让：

\[
c_i
=
\sum_jA_{ij}v_j
\]

随：

\[
q_i
\]

变化。

所以不同 future actions 可以得到：

> 不同 observation summaries。

这正是 Query-based decoding 的核心优势。

---

# 99. 为什么不让每个 Action Slot 只对应一个固定 Image Patch？

因为操作过程中 relevant region 会变化。

例如：

- 物体移动；
- gripper 移动；
- wrist camera 视角变化；
- task phase 变化。

固定 hard mapping：

```text
action slot 20 → patch 37
```

太僵硬。

Cross-Attention：

> 根据当前 sample 动态产生读取分布。

所以：

\[
\boxed{
\text{dynamic content-based routing}
}
\]

比固定映射灵活得多。

---

# 100. 常见误解一：Cross-Attention 的 Q/K/V 都来自 Decoder

**错误。**

Cross-Attention：

\[
Q\leftarrow Decoder
\]

\[
K,V\leftarrow Encoder Memory
\]

---

# 101. 常见误解二：Encoder Output 本身就等于 K 和 V

**不严格。**

它是 K/V source。

通常还经过各 head 的：

\[
W_K,W_V
\]

---

# 102. 常见误解三：Decoder Query Embedding 就是最终 Q

**不严格。**

Query embedding / query position 是 decoder-side representation 的一部分。

真正 Attention Q 还经过：

\[
W_Q
\]

---

# 103. 常见误解四：Memory 有 1202 个 Tokens，所以 Decoder 输出 1202 个 Actions

**错误。**

输出数量由：

\[
N_q
\]

决定。

ACT 中：

\[
N_q=k
\]

所以输出：

\[
k
\]

个 action representations。

---

# 104. 常见误解五：Cross-Attention 把 Encoder Memory 压成一个 Vector

**错误。**

每一个 Query 都得到自己的：

\[
o_i
\]

所以输出有：

\[
N_q
\]

个 context representations。

---

# 105. 常见误解六：Self-Attention 和 Cross-Attention 用不同数学公式

**核心公式相同。**

区别主要是：

> Q/K/V source。

---

# 106. 常见误解七：Cross-Attention 一定是 Autoregressive

**错误。**

ACT 就是明确反例。

---

# 107. 常见误解八：Cross-Attention 一定需要 Causal Mask

**错误。**

Mask 取决于信息可见性约束。

Encoder Memory 通常可以完整读取。

---

# 108. 常见误解九：第 i 个 Action Query 只会读第 i 个 Memory Token

**错误。**

每个 Query 可以读取全部：

\[
N_m
\]

个 allowed memory positions。

---

# 109. 常见误解十：不同 Action Slots 共享同一 Attention Weight Row

**错误。**

每个 Query：

\[
q_i
\]

都有自己的：

\[
A_{i,:}
\]

---

# 110. 常见误解十一：K 决定被读取的具体内容

主要不是。

K 决定：

> 匹配权重。

真正进入 weighted sum 的是：

\[
V
\]

---

# 111. 常见误解十二：V 会直接影响 qᵀk Score

标准 Scaled Dot-Product Attention 中：

**不会直接影响。**

Scores 来自：

\[
QK^\top
\]

---

# 112. 常见误解十三：Cross-Attention Map 就是最终模型解释

**错误。**

它只是某一层、某一 head 的中间 routing weights。

---

# 113. 常见误解十四：Paper 说 fixed Query，所以 Official Code 一定也是 fixed

不能这样推断。

ACT 论文描述：

> fixed position embeddings / appendix fixed sinusoidal first-layer queries。

当前官方仓库：

```python
nn.Embedding(num_queries, hidden_dim)
```

是 learned。

需要把：

> Paper Fact

和：

> Released-Code Fact

分开。

---

# 114. 常见误解十五：Action Query 一开始已经知道应该输出什么动作

**错误。**

它只是：

> output slot identity / decoder-side representation。

真正 action content 来自：

- learned model parameters；
- action-slot interactions；
- Cross-Attention 对 observation memory 的读取；
- FFN；

逐层形成。

---

# 115. 用三句话记住 Cross-Attention

第一句：

\[
\boxed{
Q\text{ comes from the reader}
}
\]

第二句：

\[
\boxed{
K,V\text{ come from the memory}
}
\]

第三句：

\[
\boxed{
\text{number of outputs}
=
\text{number of queries}
}
\]

只要这三句牢牢记住，

绝大多数 Cross-Attention shape 问题都会变简单。

---

# 116. 用一个公式记住 Shape

\[
\boxed{
[N_q,d_k]
\;
[d_k,N_m]
=
[N_q,N_m]
}
\]

然后：

\[
\boxed{
[N_q,N_m]
\;
[N_m,d_v]
=
[N_q,d_v]
}
\]

所以：

```text
Query count
一直保留到输出

Memory count
在 weighted aggregation 中被消掉
```

这就是 Cross-Attention 的矩阵本质。

---

# 117. 一张完整 Cross-Attention 图

```text
              DECODER SIDE
                   │
                   ▼
             representations H
                   │
                  W_Q
                   │
                   ▼
                   Q
                   │
                   │
                   │
                   ▼
                Q Kᵀ
                   │
              / sqrt(d_k)
                   │
                Softmax
                   │
                   ▼
             Attention A
                   │
                   │
                   ▼
                   A V
                   │
                   ▼
            Query-specific
             readout O


              ENCODER SIDE
                   │
                   ▼
               Memory M
               /      \
              /        \
            W_K        W_V
             │          │
             ▼          ▼
             K          V
```

这张图就是 Cross-Attention 的全部骨架。

---

# 118. 一张 ACT 版 Cross-Attention 图

```text
4 camera images
      │
   ResNet18
      │
1200 visual tokens
      │
joint token
latent token
      │
      ▼
Transformer Encoder
      │
      ▼
Observation Memory
[1202 × 512]
      │
      ├────────────── K
      │
      └────────────── V
                       ▲
                       │
k action query slots ──┤
        │              │
        └──── Q ───────┘
                       │
                       ▼
              Cross-Attention
                       │
                       ▼
              [k × 512]
                       │
                       ▼
                Action Head
                       │
                       ▼
                [k × 14]
```

当：

\[
k=100
\]

每个 head 的 score matrix：

\[
\boxed{
100\times1202
}
\]

8 heads：

\[
\boxed{
8\times100\times1202
}
\]

忽略 batch。

这就是 ACT Decoder “读取 observation”的核心。

---

# 119. 一句话真正理解 Cross-Attention

> **Cross-Attention 是一种 Query-conditioned external-memory retrieval：一组 decoder-side representations 产生 Queries，另一组 encoder-side memory representations 分别产生 Keys 和 Values；每个 Query 与所有 Keys 计算 compatibility，经 Softmax 得到对 Memory 的读取分布，再对 Values 加权汇总，因此每个 Query 都能从同一份 Memory 中提取一个不同的 context，而最终输出数量始终由 Query 数量决定。**

回到 ACT：

> **每个 future action slot 都是一个读取请求；它通过 Cross-Attention 从 1202-token observation memory 中动态提取与这个 action position 有关的信息，再经过 Decoder 和 action head 变成对应的 14-D target joint position。**

---

# 120. 下一步：Positional Encoding

现在 Transformer 的信息路由主线已经基本打通：

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
↓
Cross-Attention
```

下一步最自然的问题变成：

> **如果 Attention 本身只看向量内容，它到底怎么知道 token 的位置？**

下一篇：

> **[Positional Encoding：没有 RNN 后，Transformer 怎么知道顺序？](./positional-encoding.md)**

会重点解释：

- 为什么没有 position 时 full Self-Attention permutation-equivariant；
- 为什么“把位置直接加到 embedding”居然能工作；
- 原论文 sinusoidal encoding 的公式到底在做什么；
- 为什么不同维度使用不同频率；
- 为什么 \(\sin/\cos\) 能表达 relative offset；
- learned position embedding 和 sinusoidal encoding 的区别；
- 2D position encoding 怎样扩展到图片；
- ACT 的 \(15\times20\) visual grid 为什么必须保留空间位置；
- ACT 中 visual position、joint/latent position 和 action query position分别是什么。

---

## Primary Source 1：Transformer

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,  
Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin.

**Attention Is All You Need.**  
NeurIPS 2017.

- arXiv: https://arxiv.org/abs/1706.03762
- HTML: https://arxiv.org/html/1706.03762

本文主要依据：

### Section 3.2

Attention 被定义为：

\[
\boxed{
Attention(Q,K,V)
=
softmax
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
}
\]

### Section 3.2.3 — Applications of Attention in our Model

原论文明确指出：

> encoder-decoder attention 中，queries 来自 previous decoder layer，而 memory keys 和 values 来自 encoder output。

这使：

> 每个 decoder position 都可以 attend 整个 input sequence。

---

## Primary Source 2：ACT

Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.  
**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.**  
RSS 2023.

- arXiv: https://arxiv.org/abs/2304.13705
- HTML: https://arxiv.org/html/2304.13705v1

Section IV-C 明确给出：

- 4 张 \(480\times640\) RGB image；
- ResNet18 后每张为 \(15\times20\times512\)；
- flatten 后每张 \(300\times512\)；
- 四张共 \(1200\times512\)；
- 再加 joint 与 \(z\) 两个 512-D features；
- Transformer Encoder 输入为：
  \[
  1202\times512
  \]
- Transformer Decoder 通过 Cross-Attention 条件化在 Encoder output；
- input/query sequence 长度为 \(k\)；
- Decoder output：
  \[
  k\times512
  \]
- Action output：
  \[
  k\times14
  \]

Appendix C 进一步写明：

> Encoder outputs are used as both “keys” and “values” in cross-attention layers of the Transformer Decoder。

并描述：

> first-layer queries 为 fixed sinusoidal embeddings。

---

## Official ACT Implementation Note

Official repository:

https://github.com/tonyzhaozh/act

当前 `detr/models/detr_vae.py` 中：

```python
self.query_embed =
    nn.Embedding(
        num_queries,
        hidden_dim
    )
```

并把：

```python
self.query_embed.weight
```

传给 Transformer。

因此：

> **论文对 action-query positional representation 的描述与当前 released code 存在实现细节差异。**

这不改变 Cross-Attention 的核心角色划分：

\[
Q
\leftarrow
\text{decoder side}
\]

\[
K,V
\leftarrow
\text{encoder memory side}
\]

另外，DETR-style decoder implementation 中典型 Cross-Attention 调用为：

```python
query = tgt + query_pos
key   = memory + pos
value = memory
```

随后进入：

```python
nn.MultiheadAttention
```

内部再执行各 head 的 learned Q/K/V projections。

---

## 本文知识连接

### 前置

- [Attention](./attention.md)
- [Query / Key / Value](./qkv.md)
- [Self-Attention](./self-attention.md)
- [Multi-Head Attention](./multi-head-attention.md)
- [Dot Product](./dot-product.md)
- [Softmax](./softmax.md)

### Transformer

- [Transformer](./transformer.md)
- [Positional Encoding](./positional-encoding.md)
- [Causal Mask](./causal-mask.md)
- [Transformer Encoder](./transformer-encoder.md)
- [Transformer Decoder](./transformer-decoder.md)
- [Feed-Forward Network](./feed-forward-network.md)
- [Residual Connection](./residual-connection.md)
- [Layer Normalization](./layer-normalization.md)

### Robot Learning

- [ACT Architecture](../robot-learning/act/architecture.md)
- [ACT Complete Data Flow](../robot-learning/act/complete-data-flow.md)
- [Action Chunking](../robot-learning/act/action-chunking.md)

### 下一步

- [Positional Encoding](./positional-encoding.md)
