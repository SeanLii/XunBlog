---
title: "Transformer：为什么 Attention 可以替代循环？"
description: "从 RNN 的顺序计算瓶颈出发，基于 Attention Is All You Need 原论文逐步理解 Self-Attention、QKV、Scaled Dot-Product Attention、Multi-Head Attention、Positional Encoding、Encoder/Decoder 与 Transformer 的完整信息流。"
status: reviewed
pageType: concept
canonical: /deep-learning/transformer
updated: "2026-09-15"
---

# Transformer：为什么 Attention 可以替代循环？

今天再看 Transformer，我们很容易把它理解成一套已经理所当然的公式：

\[
Q=XW_Q,\qquad
K=XW_K,\qquad
V=XW_V
\]

然后：

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
\]

接着 Multi-Head Attention、Feed-Forward Network、Residual、LayerNorm……

最后堆很多层。

如果只按这个顺序学习，很容易出现一种情况：

> 每个公式都认识，但不知道为什么模型非要长成这样。

Transformer 原论文《Attention Is All You Need》真正的问题不是：

> “怎样发明 Q、K、V？”

而是：

> **序列模型为什么一定要按照时间顺序，一个位置接一个位置地计算？**

2017 年以前，RNN、LSTM、GRU 已经是 sequence modeling 的主流方法。

它们很自然：

```text
x₁
↓
h₁
↓
x₂
↓
h₂
↓
x₃
↓
h₃
...
```

但这种“顺序”同时也是瓶颈。

Transformer 的核心思想可以先压缩成一句话：

> **不要再让信息必须沿着时间链一个位置一个位置传递，而是让序列中的位置直接根据内容彼此读取信息。**

这就是 Self-Attention。

然后问题才变成：

- “想读取什么”怎样表示？
- “谁和我相关”怎样计算？
- “真正读取什么内容”怎样表示？
- 多种关系怎样同时学习？
- 没有 RNN 后，顺序信息从哪里来？
- Encoder 和 Decoder 怎样协作？

这一篇从这个逻辑开始。

---

# 1. Transformer 最初解决的是 Sequence Transduction

原论文研究的主要任务是机器翻译。

输入：

```text
I love learning AI
```

输出：

```text
我 喜欢 学习 人工智能
```

这类任务叫：

> **Sequence Transduction**

即：

\[
(x_1,\ldots,x_n)
\longrightarrow
(y_1,\ldots,y_m)
\]

输入是一个 sequence，

输出也是一个 sequence。

Transformer 原论文沿用了经典：

> **Encoder–Decoder**

框架。

高层结构：

```text
Input Sequence
      │
      ▼
   Encoder
      │
      ▼
    Memory
      │
      ▼
   Decoder
      │
      ▼
Output Sequence
```

真正革命性的地方在于：

> Encoder 和 Decoder 内部不再依赖 sequence-aligned recurrence。

而是主要依靠：

> **Attention**

建立位置之间的依赖关系。

---

# 2. 为什么 RNN 看起来这么自然？

假设输入：

\[
x_1,x_2,x_3,\ldots,x_n
\]

RNN 计算：

\[
h_t
=
f(h_{t-1},x_t)
\]

因此：

\[
h_1=f(h_0,x_1)
\]

\[
h_2=f(h_1,x_2)
\]

\[
h_3=f(h_2,x_3)
\]

依次继续。

这样做非常符合语言的直觉：

> 读完前面的词，再处理下一个词。

而且：

\[
h_t
\]

理论上可以携带之前 sequence 的信息。

例如：

```text
The animal didn't cross the street because it was too tired.
```

当模型读到：

```text
it
```

时，

前面的 hidden state 已经积累了之前的 context。

所以 RNN 的设计没有“错”。

Transformer 要解决的是：

> **这种信息传递方式是否必须依赖一条严格的时间链？**

---

# 3. RNN 的第一个问题：训练中的顺序依赖

考虑：

\[
h_3
=
f(h_2,x_3)
\]

要算：

\[
h_3
\]

必须先有：

\[
h_2
\]

而：

\[
h_2
\]

又必须等：

\[
h_1
\]

所以在单个 sequence 内：

```text
h₁
↓
h₂
↓
h₃
↓
h₄
...
```

存在：

> **inherently sequential computation**

原论文明确指出：

> recurrent models 把 symbol positions 与 computation steps 对齐，因此 sequence 内部无法完全并行。

即使你已经知道整个训练句子：

\[
x_1,\ldots,x_n
\]

仍不能同时计算全部：

\[
h_1,\ldots,h_n
\]

因为后面的 hidden state 依赖前面的结果。

---

# 4. 为什么这个问题在 GPU 上尤其重要？

GPU 擅长：

> 同时做大量矩阵运算。

例如如果我们可以把所有 token representation 堆成矩阵：

\[
X
\in
\mathbb R^{n\times d}
\]

然后一次计算：

\[
XW
\]

GPU 可以很好地并行处理所有位置。

但 RNN 的 sequence dependency 强制：

```text
position 1
先算

position 2
才能算

position 3
才能算
```

所以序列越长，

单个 sample 中需要的 sequential operations 越多。

原论文把 recurrent layer 的最少 sequential operations 写成：

\[
\boxed{O(n)}
\]

而 self-attention：

\[
\boxed{O(1)}
\]

这里的：

\[
O(1)
\]

不是说 attention 计算量和 sequence length 无关。

而是说：

> **在一层 self-attention 内，所有 sequence positions 可以并行计算，不需要按照 \(1\rightarrow2\rightarrow3\) 的顺序逐步执行。**

这个区别非常重要。

---

# 5. RNN 的第二个问题：远距离信息需要走很长的路径

假设：

\[
x_1
\]

的信息要影响：

\[
x_{100}
\]

在 recurrent chain 中，信息大致要经过：

\[
h_1
\rightarrow
h_2
\rightarrow
\cdots
\rightarrow
h_{100}
\]

也就是：

\[
O(n)
\]

长度的路径。

即使 LSTM / GRU 改善了长期依赖和梯度问题，

结构上：

> 两个遥远 position 之间的信息仍要经过很多 recurrent transitions。

Transformer 想做一件更直接的事情：

```text
position 100
想知道 position 1 的信息？

那就直接看 position 1。
```

于是任意两个位置在一层 full self-attention 中：

\[
\boxed{
\text{maximum path length}=O(1)
}
\]

这就是 Self-Attention 对 long-range dependency 的一个核心结构优势。

---

# 6. 如果所有位置都能直接“看”所有位置，会发生什么？

假设句子：

```text
The animal didn't cross the street because it was too tired.
```

当我们处理：

```text
it
```

时，

我们希望模型能够根据上下文读取：

```text
animal
```

而不是：

```text
street
```

于是可以想象：

```text
当前 token: it

去整个 sequence 里问：

The      和我相关吗？
animal   和我相关吗？
didn't   和我相关吗？
cross    和我相关吗？
street   和我相关吗？
because  和我相关吗？
...
```

然后根据相关程度，

从不同 token 中读取不同数量的信息。

这就是 Attention 最基本的直觉。

---

# 7. Attention 不是“选一个 Token”

这是一个容易形成的误解。

Attention 通常不是：

```text
找到最相关 token
↓
只复制它
```

而是：

> 对多个 value 做 weighted sum。

原论文定义：

> Attention 把一个 query 和一组 key-value pairs 映射为输出；每个 value 的权重由 query 与对应 key 的 compatibility 决定。

因此：

\[
\boxed{
\text{Output}
=
\sum_j
\alpha_jv_j
}
\]

其中：

\[
\alpha_j
\]

表示第 \(j\) 个位置应该贡献多少。

所以 attention 更像：

> **根据当前需求，从整个信息库中软性地读取并混合信息。**

---

# 8. 为什么需要 Query、Key、Value 三种东西？

假设你进入图书馆。

你现在有一个需求：

> “我想找关于 Transformer 的资料。”

这个需求可以类比：

\[
\boxed{Query}
\]

每本书有一个用于检索的描述：

```text
书名
标签
主题
索引
```

可以类比：

\[
\boxed{Key}
\]

当你判断某本书和需求匹配后，

真正从书中读取的内容：

可以类比：

\[
\boxed{Value}
\]

所以：

```text
Query
= 我现在想找什么？

Key
= 我可以怎样被匹配？

Value
= 如果匹配了，我真正提供什么信息？
```

这是一个教学直觉。

更严格的数学定义会在：

- [Query / Key / Value](./qkv.md)

单独展开。

---

# 9. Q、K、V 不是天然存在的语义变量

这一点非常重要。

模型输入一个 token representation：

\[
x_i\in\mathbb R^{d_{\text{model}}}
\]

然后通过三个 learned linear projections：

\[
\boxed{
q_i=x_iW_Q
}
\]

\[
\boxed{
k_i=x_iW_K
}
\]

\[
\boxed{
v_i=x_iW_V
}
\]

也就是说：

> Q、K、V 都只是同一个 representation 在不同 learned projection space 中的向量。

没有人手工告诉模型：

```text
这一维叫“问题”
这一维叫“关键词”
这一维叫“内容”
```

它们之所以逐渐承担 query/key/value 的功能，

是因为整个 attention computation 和 training objective 迫使这些 projection 学出有用表示。

---

# 10. 为什么 Query 要和 Key 做 Dot Product？

假设：

\[
q_i,k_j
\in
\mathbb R^{d_k}
\]

Transformer 使用：

\[
q_i\cdot k_j
\]

作为 compatibility score 的基础。

如果两个向量在 learned representation space 中方向相似，

dot product 往往较大。

所以可以直觉理解成：

> 当前 query \(i\) 与 position \(j\) 的 key 有多匹配？

对所有位置：

\[
j=1,\ldots,n
\]

计算：

\[
s_{ij}
=
q_i^\top k_j
\]

得到一整排 scores。

---

# 11. 单个 Token 的 Attention 先写出来

假设当前 token：

\[
i
\]

有 query：

\[
q_i
\]

整个 sequence 有 keys：

\[
k_1,\ldots,k_n
\]

先计算：

\[
s_{ij}
=
\frac{
q_i^\top k_j
}{
\sqrt{d_k}
}
\]

然后 Softmax：

\[
\alpha_{ij}
=
\frac{
e^{s_{ij}}
}{
\sum_{r=1}^{n}e^{s_{ir}}
}
\]

因此：

\[
\sum_j\alpha_{ij}=1
\]

最后：

\[
\boxed{
o_i
=
\sum_{j=1}^{n}
\alpha_{ij}v_j
}
\]

这个：

\[
o_i
\]

就是 token \(i\) 从整个 sequence 读取后的新 representation。

---

# 12. 为什么乘的是 Value，而不是 Key？

Key 的工作是：

> 用来判断匹配程度。

真正被聚合的信息来自：

\[
V
\]

所以：

\[
QK^\top
\]

负责：

> **决定去哪读。**

而：

\[
\operatorname{softmax}(\cdot)V
\]

负责：

> **真正读取并融合内容。**

可以把它压缩成：

\[
\boxed{
QK^\top
=
\text{where to read}
}
\]

\[
\boxed{
V
=
\text{what to read}
}
\]

这是非常有用的 mental model。

---

# 13. 为什么要除以 \(\sqrt{d_k}\)？

Scaled Dot-Product Attention 的公式：

\[
\boxed{
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
}
\]

为什么不能直接：

\[
\operatorname{softmax}(QK^\top)V
\]

？

原论文给出一个统计解释。

假设：

\[
q_i
\]

和：

\[
k_i
\]

各维独立，

均值：

\[
0
\]

方差：

\[
1
\]

dot product：

\[
q\cdot k
=
\sum_{r=1}^{d_k}q_rk_r
\]

那么：

\[
\operatorname{Var}(q\cdot k)
=
d_k
\]

所以：

\[
d_k
\]

越大，

dot-product magnitude 往往越大。

---

# 14. Dot Product 太大为什么不好？

假设 softmax 输入：

\[
[1,2,3]
\]

输出还有一定平滑性。

但如果整体 magnitude 变成：

\[
[10,20,30]
\]

softmax 会非常接近：

```text
[0, 0, 1]
```

也就是：

> saturation。

这时很多位置的 gradient 会非常小。

所以除：

\[
\sqrt{d_k}
\]

之后：

\[
\operatorname{Var}
\left(
\frac{q\cdot k}{\sqrt{d_k}}
\right)
\]

大致重新回到常数尺度。

原论文正是用这个理由解释 scaling factor。

---

# 15. 为什么所有 Token 可以一起算？

如果 sequence：

\[
X
\in
\mathbb R^{n\times d_{\text{model}}}
\]

那么：

\[
Q=XW_Q
\]

\[
K=XW_K
\]

\[
V=XW_V
\]

全部 token 一次矩阵乘法即可计算。

然后：

\[
QK^\top
\]

shape：

\[
[n,d_k]
\times
[d_k,n]
=
[n,n]
\]

这张矩阵中的：

\[
(i,j)
\]

位置：

\[
q_i^\top k_j
\]

正好表示：

> token \(i\) 对 token \(j\) 的 compatibility。

于是一次矩阵运算同时得到：

> 所有 token 对所有 token 的 attention scores。

这就是 Self-Attention 能够高度并行的重要原因。

---

# 16. Self-Attention 中为什么叫“Self”？

因为：

\[
Q,K,V
\]

都来自：

> 同一个 sequence。

例如 encoder 当前 layer 输入：

\[
X
\]

则：

\[
Q=XW_Q
\]

\[
K=XW_K
\]

\[
V=XW_V
\]

所以 sequence：

> 在读取自己内部不同位置的信息。

这就是：

\[
\boxed{
\text{Self-Attention}
}
\]

或者原论文提到的：

> intra-attention。

---

# 17. Self-Attention 不等于“Token 和自己算 Attention”

另一个常见误解。

“self”不是说：

> 每个 token 只看自己。

恰恰相反。

Full self-attention 中 token \(i\) 可以 attend：

\[
1,2,\ldots,n
\]

所有位置，包括自己。

“self”只是表示：

> queries、keys、values 来自同一个 sequence source。

---

# 18. 为什么需要 Multi-Head Attention？

如果只有一个 attention：

\[
\operatorname{Attention}(Q,K,V)
\]

所有关系都要压在同一个 attention space 里。

原论文发现：

> 使用多个不同 learned projections 并行做 attention 更有效。

对于 head \(h\)：

\[
Q_h=QW_h^Q
\]

\[
K_h=KW_h^K
\]

\[
V_h=VW_h^V
\]

然后：

\[
head_h
=
\operatorname{Attention}(Q_h,K_h,V_h)
\]

最后：

\[
\boxed{
\operatorname{MultiHead}(Q,K,V)
=
\operatorname{Concat}
(
head_1,\ldots,head_H
)
W^O
}
\]

---

# 19. 为什么多个 Head 有意义？

可以用一个直觉例子。

同一句话里，一个 token 可能同时需要关注：

- 语法主语；
- 指代对象；
- 邻近短语；
- 长距离依赖；
- 位置模式。

如果只有一个 attention distribution，

这些关系可能相互干扰。

不同 heads 使用不同：

\[
W_i^Q,W_i^K,W_i^V
\]

所以它们可以在不同 learned representation subspaces 中工作。

原论文的说法是：

> Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

注意：

> 并不保证某一个 head 永远等于“语法 head”，另一个永远等于“指代 head”。

这种语义需要实验分析，而不是架构本身保证。

---

# 20. 原始 Transformer Base 有多少个 Head？

原论文 base model：

\[
d_{\text{model}}=512
\]

\[
h=8
\]

所以每个 head：

\[
d_k=d_v=\frac{512}{8}=64
\]

也就是：

```text
512-D representation
↓
8 个不同 projections
↓
每个 head 在 64-D 空间做 attention
↓
8 × 64 = 512
↓
concat
↓
W^O
↓
512-D output
```

这样 Multi-Head Attention 输入输出 dimension 保持：

\[
512
\]

方便 residual connection。

---

# 21. 到这里 Attention 做的事情是什么？

对于每一个 token：

> 根据当前 representation 生成 query。

再用 query 与所有 token 的 keys 比较，

得到 weights。

最后用这些 weights 聚合 values，

得到新的 contextual representation。

所以：

```text
原 token representation
↓
“我现在需要哪些 context？”
↓
读取整个 sequence
↓
新的 context-aware representation
```

Attention 本质上是一种：

> **content-dependent information routing mechanism。**

---

# 22. Transformer Encoder 不只有 Attention

一个常见误解是：

> Transformer = Attention。

原论文 Encoder 每层实际上包含两个主要 sublayers：

1. Multi-Head Self-Attention；
2. Position-wise Feed-Forward Network。

此外还有：

- Residual Connection；
- LayerNorm。

原论文 base model 堆：

\[
\boxed{N=6}
\]

层 encoder。

---

# 23. 一个 Encoder Layer

可以简化画成：

```text
X
│
├─────────────┐
│             ▼
│     Multi-Head
│     Self-Attention
│             │
└──── + ──────┘
      │
      ▼
  LayerNorm
      │
      ├─────────────┐
      │             ▼
      │            FFN
      │             │
      └──── + ──────┘
            │
            ▼
        LayerNorm
```

原论文写：

\[
\boxed{
\operatorname{LayerNorm}
(
x+\operatorname{Sublayer}(x)
)
}
\]

也就是今天通常称为：

> **Post-LN**

结构。

注意：

> 很多后来 Transformer 使用 Pre-LN。

所以不要把“Transformer 必须是 Post-LN”当成现代统一定义。

这里讲的是 2017 原论文。

---

# 24. Residual Connection 为什么重要？

Attention layer 输出某种 transformation：

\[
F(x)
\]

Residual：

\[
x+F(x)
\]

意味着 layer 不必从头重新构造所有 representation。

它可以学：

> 在原 representation 上增加什么信息。

同时 residual path 为深层网络提供更直接的 gradient propagation route。

Residual connection 并不是 Transformer 首创，

它来自 ResNet 思想。

---

# 25. LayerNorm 在做什么？

原论文每个 sublayer 后使用：

\[
\operatorname{LayerNorm}
\]

帮助稳定 hidden representation 的尺度和训练。

注意：

> LayerNorm 与我们在 ACT 数据 preprocessing 中讲的 dataset normalization 不一样。

Dataset standardization：

\[
(x-\mu)/\sigma
\]

是数据预处理。

LayerNorm：

> 是网络内部对单个 representation feature dimensions 做 normalization 的 neural network operation。

详细内容以后可以单独阅读：

- [Layer Normalization](./layer-normalization.md)

---

# 26. Feed-Forward Network 为什么还需要？

Attention 做的是：

> 不同 positions 之间的信息交换。

但只交换信息还不够。

每个 position 还需要对已经融合的信息做 nonlinear transformation。

所以每层还有：

\[
\boxed{
FFN(x)
=
\max(0,xW_1+b_1)W_2+b_2
}
\]

原论文：

\[
d_{\text{model}}=512
\]

先扩大：

\[
512\rightarrow2048
\]

经过 ReLU，

再：

\[
2048\rightarrow512
\]

---

# 27. 为什么叫 Position-Wise FFN？

因为同一个 FFN：

> 独立应用在每个 sequence position。

假设：

\[
X\in\mathbb R^{n\times512}
\]

对每一个：

\[
x_i
\]

使用相同：

\[
W_1,W_2
\]

但：

> 不在这个 FFN 里混合不同 sequence positions。

位置之间的交流已经由 attention 完成。

所以可以理解为：

```text
Attention:
token ↔ token 交流

FFN:
每个 token 自己做非线性加工
```

两者分工非常清晰。

---

# 28. 没有 RNN 后，Transformer 怎么知道顺序？

这是最关键的问题之一。

考虑：

```text
dog bites man
```

和：

```text
man bites dog
```

token 集合一样。

顺序不同，

意思完全不同。

Self-attention 如果只输入：

\[
X
\]

而完全没有 position information，

它本身并不知道：

> token 在第几个位置。

所以 Transformer 必须额外加入：

> **Positional Encoding**

---

# 29. 原论文使用 Sinusoidal Positional Encoding

输入 token embedding：

\[
e_{pos}
\]

加：

\[
PE_{pos}
\]

得到：

\[
\boxed{
x_{pos}
=
e_{pos}
+
PE_{pos}
}
\]

两者 dimension 相同：

\[
d_{\text{model}}
\]

原论文：

\[
PE_{(pos,2i)}
=
\sin
\left(
\frac{pos}{
10000^{2i/d_{\text{model}}}
}
\right)
\]

\[
PE_{(pos,2i+1)}
=
\cos
\left(
\frac{pos}{
10000^{2i/d_{\text{model}}}
}
\right)
\]

不同 dimensions 使用不同频率。

---

# 30. 为什么把 Position “加”到 Embedding 上？

因为：

\[
embedding
\in
\mathbb R^{d_{\text{model}}}
\]

\[
PE
\in
\mathbb R^{d_{\text{model}}}
\]

维度一致。

相加后：

\[
x
=
embedding+position
\]

每个 token representation 同时包含：

- token identity / content；
- position information。

之后 QKV projections 可以学习利用这两部分信息。

原论文也试过 learned positional embeddings，

结果和 sinusoidal 版本接近。

最终选择 sinusoidal 主要是作者认为它可能更容易 extrapolate 到训练时没见过的更长 sequence。

---

# 31. Positional Encoding 不是 Attention Weight

不要混淆：

```text
Positional Encoding
```

告诉模型：

> token 在哪里。

而：

```text
Attention Weight
```

告诉模型：

> 当前 query 应该从哪些 token 读取多少信息。

Position information 会影响 token representation，

进而间接影响：

\[
Q,K
\]

和 attention scores。

但二者不是同一个东西。

---

# 32. 原始 Transformer Encoder 的完整 Flow

输入 tokens：

\[
x_1,\ldots,x_n
\]

先 embedding：

\[
e_i
\]

再加 position：

\[
h_i^{(0)}
=
e_i+PE_i
\]

然后 6 层：

\[
H^{(1)}
=
EncoderLayer_1(H^{(0)})
\]

\[
H^{(2)}
=
EncoderLayer_2(H^{(1)})
\]

……

\[
H^{(6)}
=
EncoderLayer_6(H^{(5)})
\]

最终：

\[
\boxed{
M=H^{(6)}
}
\]

形成 encoder memory。

每个位置都有自己的：

\[
512
\]

维 contextual representation。

---

# 33. Decoder 为什么比 Encoder 多一个 Attention？

原论文 decoder 每层有三个主要 sublayers：

1. Masked Multi-Head Self-Attention；
2. Encoder–Decoder Cross-Attention；
3. Feed-Forward Network。

为什么多一个？

因为 decoder 不仅需要理解：

> 已经生成的 output sequence。

还必须读取：

> input sequence 的 encoder memory。

所以需要：

> **Cross-Attention**

---

# 34. Self-Attention 和 Cross-Attention 的根本区别

## Self-Attention

\[
Q,K,V
\]

来自：

> 同一个 sequence。

例如 encoder：

\[
Q=HW_Q,\quad
K=HW_K,\quad
V=HW_V
\]

---

## Cross-Attention

Query 来自一个 sequence：

> Decoder。

Key / Value 来自另一个 sequence：

> Encoder Memory。

即：

\[
Q
=
H_{\text{decoder}}W_Q
\]

\[
K
=
M_{\text{encoder}}W_K
\]

\[
V
=
M_{\text{encoder}}W_V
\]

所以可以理解为：

> Decoder 根据自己的当前需求，去 Encoder Memory 中检索输入信息。

---

# 35. 一个翻译例子

输入：

```text
I love cats
```

Encoder memory：

```text
M_I
M_love
M_cats
```

Decoder 正在生成：

```text
我 喜欢 ...
```

当它需要预测：

```text
猫
```

对应 hidden state 的 query 可以通过 cross-attention：

> 强烈读取 encoder 中 `cats` 对应 memory。

因此：

```text
Decoder Query
↓
Encoder Keys
↓
决定关注位置
↓
读取 Encoder Values
```

这就是 encoder-decoder attention。

---

# 36. 为什么 Decoder Self-Attention 要 Mask？

原始 Transformer 是 autoregressive translation model。

预测：

\[
y_i
\]

时不能偷看：

\[
y_{i+1},y_{i+2},\ldots
\]

否则训练时：

> 答案已经泄漏。

所以 decoder self-attention 对未来位置加 mask。

例如 sequence length 4：

\[
\begin{bmatrix}
0&-\infty&-\infty&-\infty\\
0&0&-\infty&-\infty\\
0&0&0&-\infty\\
0&0&0&0
\end{bmatrix}
\]

加到 attention logits。

softmax 后：

> 被 mask 的 future positions 权重为 0。

因此 position \(i\) 只能看：

\[
\le i
\]

的位置。

---

# 37. 为什么 Encoder 不需要 Causal Mask？

Encoder 已经一次拿到完整输入：

```text
I love cats
```

没有：

> “未来输入 token 是答案泄漏”

这个问题。

所以 encoder self-attention 通常允许：

> 每个 position 看所有 input positions。

这叫：

> **Bidirectional Self-Attention**

在原始 Transformer encoder 中：

\[
x_i
\]

可以直接读取：

\[
x_j
\]

无论：

\[
j<i
\]

还是：

\[
j>i
\]

---

# 38. 一个完整原始 Transformer

可以画成：

```text
SOURCE TOKENS
     │
     ▼
 Embedding
     +
Positional Encoding
     │
     ▼
┌──────────────────┐
│ Encoder Layer ×6 │
│                  │
│ Self-Attention   │
│ FFN              │
└──────────────────┘
     │
     ▼
Encoder Memory
     │
     │ K,V
     ▼
┌──────────────────┐
│ Decoder Layer ×6 │
│                  │
│ Masked Self-Attn │
│ Cross-Attention  │
│ FFN              │
└──────────────────┘
     ▲
     │
Target tokens shifted right
     │
 Embedding
     +
Positional Encoding
     │
     ▼
 Decoder States
     │
     ▼
 Linear + Softmax
     │
     ▼
Next-token probabilities
```

这才是 2017 原论文完整 Transformer。

---

# 39. 为什么论文标题叫 Attention Is All You Need？

这句话经常被理解成：

> “整个 Transformer 只有 Attention。”

其实不是。

原始架构仍然有：

- embeddings；
- positional encoding；
- feed-forward networks；
- residual connections；
- layer normalization；
- output linear layer；
- softmax。

标题真正强调的是：

> **不需要 RNN recurrence 或 CNN convolution 来完成 sequence-to-sequence representation learning；attention mechanism 可以承担 sequence positions 间主要的信息交互。**

所以：

\[
\boxed{
\text{Attention Is All You Need}
\neq
\text{Only Attention Exists}
}
\]

---

# 40. Self-Attention 为什么更容易并行？

RNN：

\[
h_t=f(h_{t-1},x_t)
\]

存在数据依赖链。

Self-attention：

\[
Q=XW_Q
\]

\[
K=XW_K
\]

\[
V=XW_V
\]

\[
A
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)
\]

\[
O=AV
\]

都是大规模矩阵运算。

所有 sequence positions：

> 可以在同一层一起处理。

所以原论文 Table 1 把最少 sequential operations 写成：

\[
\boxed{O(1)}
\]

对比 recurrent layer：

\[
\boxed{O(n)}
\]

---

# 41. 但 Self-Attention 不是“计算复杂度永远更低”

必须非常严谨。

原论文给出的 per-layer complexity：

### Self-Attention

\[
\boxed{
O(n^2d)
}
\]

### Recurrent Layer

\[
\boxed{
O(nd^2)
}
\]

所以当：

\[
n<d
\]

时，

self-attention 往往更有计算优势。

原论文指出机器翻译中常见 representation 满足这个条件。

但当 sequence：

\[
n
\]

非常长，

\[
n^2
\]

会成为严重问题。

所以后来才有大量：

- sparse attention；
- linear attention；
- FlashAttention；
- local attention；
- state-space models；

等工作尝试解决长 context 成本。

因此不要记成：

> “Transformer attention 在任何情况下都比 RNN 计算复杂度低。”

原论文从来没有这么说。

---

# 42. Transformer 的另一个优势：Path Length

原论文除了 parallelization，

还特别比较：

> 任意两个 positions 之间的信息路径长度。

Self-attention：

\[
\boxed{O(1)}
\]

Recurrent：

\[
\boxed{O(n)}
\]

Convolution：

取决于 kernel 和层数，

大致可能：

\[
O(\log_k n)
\]

或者更长。

为什么 path length 重要？

因为：

> forward information 和 backward gradient 都需要沿 network path 传播。

路径越短，

理论上越容易让 distant positions 直接发生 interaction。

---

# 43. 一个简单的 Long-Range Dependency 例子

sequence：

```text
The robot that was placed next to the blue box ...
...
...
moves.
```

如果：

```text
robot
```

和很后面的：

```text
moves
```

有依赖，

RNN 中信息要经过很多 intermediate states。

Self-attention 中：

> 后面的 token 可以直接给 `robot` 对应 key 较高 attention weight。

于是：

\[
\text{robot}
\rightarrow
\text{moves}
\]

在一层 attention 中就可以建立直接 connection。

---

# 44. 但是 Attention Weight 不等于因果解释

一个重要边界。

看到：

\[
\alpha_{ij}
\]

很大，

可以说：

> 在这个 attention computation 中，position \(i\) 从 position \(j\) 的 value 读取了较大权重。

但不能自动推出：

> “这就是模型最终预测的唯一原因。”

整个模型还有：

- 多个 heads；
- 多层；
- residual；
- FFN；
- output transformation。

所以：

> attention visualization 可以提供信息，但 attention weight 不自动等于完整 causal explanation。

---

# 45. Transformer 是怎样训练出 QKV 的？

没有：

```text
Q 标签
K 标签
V 标签
```

训练监督仍然来自最终任务。

机器翻译中：

> target token prediction loss。

反向传播：

```text
prediction loss
↓
decoder
↓
attention
↓
WQ, WK, WV
```

所以 projection matrices：

\[
W_Q,W_K,W_V
\]

被优化成：

> 使最终任务 loss 更小的参数。

因此 QKV 的功能是：

> **emergent learned representation roles。**

---

# 46. Q、K、V 不是三个不同的“输入文件”

在 Self-Attention 中：

\[
X
\]

是同一个 input representation。

然后：

\[
Q=XW_Q
\]

\[
K=XW_K
\]

\[
V=XW_V
\]

所以不是：

```text
模型收到 Query 文件
模型收到 Key 文件
模型收到 Value 文件
```

而是：

> 同一个 token representation 被投影成三个不同用途的 view。

这也是理解 QKV 最重要的一步。

---

# 47. 为什么不用 X 本身同时做 Q、K、V？

理论上可以设计：

\[
Q=K=V=X
\]

但 learned projections 提供了更灵活的 representation。

例如：

> 用某些 feature 维度判断“我在找什么”；

用另一些 feature：

> 判断“我适不适合被找到”；

再用另一个空间：

> 决定“真正传递什么内容”。

这种功能分离由：

\[
W_Q,W_K,W_V
\]

学习出来。

Multi-head 又进一步提供多组这样的 projection。

---

# 48. 为什么 Attention 输出还是一个向量？

对 token \(i\)：

\[
o_i
=
\sum_j\alpha_{ij}v_j
\]

每个：

\[
v_j\in\mathbb R^{d_v}
\]

weighted sum 后：

\[
o_i\in\mathbb R^{d_v}
\]

所以 attention 输出仍是一个 vector representation。

对于所有：

\[
n
\]

个 queries，

就得到：

\[
O\in\mathbb R^{n\times d_v}
\]

Multi-head concatenate 后回到：

\[
n\times d_{\text{model}}
\]

因此 attention 可以作为普通 neural-network layer 继续堆叠。

---

# 49. Softmax 是不是让模型“选择最相关的一个”？

不一定。

如果 scores 很接近：

\[
[1.0,1.1,0.9]
\]

softmax 可能得到：

```text
[0.33, 0.37, 0.30]
```

三个位置都参与。

只有 score difference 很大时，

distribution 才可能非常尖锐。

所以 attention 是：

> **soft selection / weighted routing**

而不是天然的 hard selection。

---

# 50. 为什么 Attention Matrix 是 n × n？

Self-attention 中：

\[
Q\in\mathbb R^{n\times d_k}
\]

\[
K^\top
\in\mathbb R^{d_k\times n}
\]

所以：

\[
QK^\top
\in
\mathbb R^{n\times n}
\]

第：

\[
i
\]

行：

> query \(i\) 对所有 keys 的 scores。

第：

\[
j
\]

列：

> 所有 queries 对 key \(j\) 的 compatibility。

这张：

\[
n\times n
\]

matrix 也是 standard self-attention：

\[
O(n^2)
\]

sequence-length cost 的来源。

---

# 51. 一个极小 Shape Example

假设：

\[
n=3
\]

\[
d_{\text{model}}=4
\]

输入：

\[
X
\in
\mathbb R^{3\times4}
\]

假设单头：

\[
d_k=d_v=2
\]

则：

\[
W_Q,W_K,W_V
\in
\mathbb R^{4\times2}
\]

于是：

\[
Q,K,V
\in
\mathbb R^{3\times2}
\]

scores：

\[
QK^\top
\in
\mathbb R^{3\times3}
\]

attention weights：

\[
A
\in
\mathbb R^{3\times3}
\]

最终：

\[
AV
\in
\mathbb R^{3\times2}
\]

这就是一整个 attention head。

---

# 52. Multi-Head Shape Example

原始 Transformer Base：

\[
n
\]

个 tokens，

\[
d_{\text{model}}=512
\]

\[
h=8
\]

每头：

\[
64
\]

维。

因此：

\[
X:
[n,512]
\]

每个 head：

\[
Q_i,K_i,V_i:
[n,64]
\]

attention output：

\[
head_i:
[n,64]
\]

8 heads concatenate：

\[
[n,512]
\]

然后：

\[
W^O
\]

得到：

\[
[n,512]
\]

shape 回到原维度。

---

# 53. 为什么 Transformer Encoder 可以一次看到整个 Sequence？

因为 encoder 用的是：

> unmasked self-attention。

所以每一个位置：

\[
i
\]

可以 attend：

\[
1,\ldots,n
\]

所有 positions。

因此：

\[
h_i
\]

不是只包含当前 token 信息。

而是：

> 当前 token 在整个 input context 下的 contextual representation。

例如同一个词：

```text
bank
```

在：

```text
river bank
```

和：

```text
bank account
```

中最终 hidden representation 可以不同。

因为它读取到的 surrounding context 不同。

---

# 54. Embedding 和 Contextual Representation 的区别

初始 embedding：

\[
e_{\text{bank}}
\]

主要由 token identity 决定。

经过多层 self-attention：

\[
h_{\text{bank}}
\]

已经包含 context。

所以：

```text
Embedding:
“bank 这个 token 大致是什么？”

Contextual representation:
“在当前句子里，这个 bank 应该怎样理解？”
```

这是 Transformer representation learning 的一个核心能力。

---

# 55. Encoder 最终不是把整个句子压成一个 Vector

经典 seq2seq RNN 有时会把 input 压成一个 fixed context vector。

Transformer Encoder 不需要这样。

它输出：

\[
\boxed{
M
=
[m_1,\ldots,m_n]
}
\]

整个 memory sequence。

Decoder 每个 timestep / query 可以：

> 直接 cross-attend 所有 encoder positions。

因此 input information 不需要全部塞进一个单一 bottleneck vector。

---

# 56. 原始 Decoder 为什么还是顺序生成？

这有时会让人困惑：

> Transformer 不是解决 sequential computation 吗？为什么 decoder 还 autoregressive？

答案是：

> Transformer 主要移除了 **layer 内 sequence-aligned recurrence**。

但原始 machine translation 的生成过程仍然 factorize：

\[
p(y_1,\ldots,y_m\mid x)
=
\prod_{t=1}^{m}
p(y_t\mid y_{<t},x)
\]

所以 inference 时：

```text
先生成 y₁
↓
再生成 y₂
↓
再生成 y₃
```

仍然是 sequential。

训练时因为完整 target 已知，

masked self-attention 可以并行处理所有 target positions。

所以：

> Transformer 的“parallelization”不能简单理解成所有 inference 场景都完全并行。

---

# 57. 这和 ACT 有一个重要区别

ACT 虽然使用 Transformer Encoder–Decoder，

但 ACT 的 action decoder：

> **不是原始语言 Transformer 那种 autoregressive decoder。**

ACT 一次给出：

\[
k
\]

个 action query slots。

没有 causal mask。

所有 future action slots 可以双向 self-attend。

然后同时输出：

\[
k\times14
\]

actions。

因此：

```text
Original Transformer Decoder:
autoregressive target modeling

ACT Transformer Decoder:
parallel action-chunk prediction
```

这是一个非常重要的区别。

---

# 58. 所以“Transformer Decoder = 自回归”是错误的

Transformer decoder layer 提供的是：

- self-attention；
- cross-attention；
- FFN。

它是不是 autoregressive，

取决于：

- causal mask；
- target representation；
- training objective；
- inference procedure。

原始论文用于 machine translation：

> autoregressive。

ACT：

> non-autoregressive chunk output。

因此要区分：

> **architecture component**

和：

> **generation strategy**

---

# 59. Transformer 在 ACT 中具体出现在哪里？

ACT 至少使用 Transformer 的三个位置。

## 1. CVAE Encoder

输入：

```text
[CLS]
joint state
ground-truth action sequence
```

使用 self-attention。

最终 `[CLS]` hidden representation：

\[
\rightarrow
\mu,\log\sigma^2
\]

---

## 2. Policy Transformer Encoder

输入：

```text
visual tokens
joint token
latent z token
```

通过 self-attention 融合：

- 多视角视觉；
- proprioception；
- latent style。

---

## 3. Policy Transformer Decoder

输入：

\[
k
\]

个 action-query positions。

通过：

- decoder self-attention；
- cross-attention to observation memory；

产生：

\[
k
\]

个 future action representations。

---

# 60. ACT 为什么没有“Word Embedding”？

因为 Transformer 本身并不要求 input 一定是文字。

它真正要求的是：

> 一串固定维度 vector representations。

在 NLP：

```text
word/token
↓
Embedding
↓
d_model vector
```

在 ACT：

```text
image feature
↓
projection
↓
512-D token
```

或者：

```text
joint state
↓
Linear
↓
512-D token
```

或者：

```text
latent z
↓
Linear
↓
512-D token
```

所以 Transformer 是：

> **sequence representation architecture**

而不是“文字专用模型”。

---

# 61. Transformer 怎么理解图片？

严格来说：

> Transformer 不直接“知道这是图片”。

ACT 先让 ResNet 把图片转成：

\[
15\times20
\]

spatial feature grid。

flatten 成：

\[
300
\]

个 visual tokens。

每个 token：

\[
512
\]

维。

再加 2D positional information。

对 Transformer 来说，

它看到的就是：

> 一串带位置的 vectors。

至于这些 vectors 来自：

- text；
- image；
- robot state；

取决于前面的 tokenizer / backbone / projection。

---

# 62. Transformer 怎么理解 Robot Joint State？

同样：

\[
q\in\mathbb R^{14}
\]

先 Linear：

\[
14\rightarrow512
\]

成为：

\[
1
\]

个 token。

Transformer 并不知道：

> “这 14 个数字是关节角。”

但 training objective 会让这组 representation 对 action prediction 有用。

所以深度学习里常见的一条规律是：

> **先把不同 modality 映射到共同 hidden dimension，再让通用 sequence model 学习它们之间的关系。**

ACT 就是这样做的。

---

# 63. Transformer 的真正核心不是 QKV 三个字母

学到这里，可以重新看 Transformer。

它真正解决的是：

> **如何让每个 position 根据当前内容动态读取其他 positions，并且让所有 positions 在一层中并行完成这种信息交换。**

QKV：

> 是实现这种动态读取的一种参数化方式。

Multi-head：

> 让多组不同读取关系并行存在。

Positional Encoding：

> 补回 attention 本身缺失的 sequence order。

FFN：

> 对每个 position 的 contextual information 做非线性加工。

Residual + LayerNorm：

> 让深层堆叠更稳定。

Encoder–Decoder Cross-Attention：

> 让 output representation 动态读取 input memory。

这些组件合在一起，

才是 Transformer。

---

# 64. 为什么 Attention 能“替代循环”？

现在终于可以严格回答标题的问题。

RNN 使用：

\[
h_t=f(h_{t-1},x_t)
\]

让信息沿 hidden-state chain 传播。

Transformer 不再依赖这种：

\[
t-1\rightarrow t
\]

的强制 sequential path。

它让每一个位置：

\[
i
\]

直接通过 attention：

\[
\alpha_{ij}
\]

从所有位置：

\[
j
\]

读取信息。

因此 sequence context 的传播方式从：

```text
链式传播
```

变成：

```text
内容驱动的全连接信息路由
```

这就是：

\[
\boxed{
\text{Recurrence}
\rightarrow
\text{Self-Attention}
}
\]

最核心的结构变化。

---

# 65. 但 Transformer 并没有“消灭时间”

没有 recurrence 后，

sequence order 不会自然存在。

所以必须显式加入：

> position information。

这说明：

> Transformer 并不是认为 sequence order 不重要。

恰恰相反。

它只是把：

> “顺序信息”

和：

> “信息交互机制”

解耦了。

RNN 中：

> 顺序由 computation path 本身提供。

Transformer 中：

> position 由 encoding 提供，interaction 由 attention 提供。

这是一个非常深刻的设计变化。

---

# 66. RNN 与 Transformer 的核心差别可以怎样总结？

## RNN

```text
结构本身：
顺序传播

信息交互：
通过 recurrent hidden state

position：
天然来自 timestep 顺序

parallelism：
受 recurrent chain 限制
```

---

## Transformer

```text
结构本身：
所有位置直接 attention

信息交互：
Q-K matching + V aggregation

position：
额外 positional encoding

parallelism：
单层所有 positions 可并行
```

---

# 67. Transformer 的代价是什么？

最明显的代价是 full self-attention：

\[
QK^\top
\]

需要生成：

\[
n\times n
\]

attention matrix。

所以 memory / compute 随：

\[
n^2
\]

增长。

当 sequence 很长：

\[
n=100000
\]

标准 attention 的成本会非常大。

所以 Transformer 并不是：

> “没有缺点的 RNN 替代品”。

它只是改变了 trade-off：

> 用更强并行性和短路径，交换 quadratic sequence interaction cost。

---

# 68. 为什么 2017 年这个 Trade-off 很划算？

原论文机器翻译场景里：

\[
n
\]

通常远小于 hidden representation dimension：

\[
d
\]

论文指出在：

\[
n<d
\]

时，

self-attention 每层：

\[
O(n^2d)
\]

可能比 recurrent：

\[
O(nd^2)
\]

更有优势。

同时 GPU 对大矩阵计算非常高效。

所以当时：

> 这个 trade-off 非常适合 machine translation。

后来 sequence lengths 不断扩展，

quadratic attention 才越来越成为核心研究问题。

---

# 69. Transformer 的成功不只是“速度快”

如果只理解成：

> “Transformer 能并行，所以成功。”

太片面。

原论文同时强调：

1. parallelization；
2. shorter dependency path；
3. content-dependent global interaction；
4. multi-head representation subspaces；
5. strong translation quality。

所以它的价值既有：

> optimization / hardware efficiency，

也有：

> representation architecture。

---

# 70. 常见误解一：Transformer 就是 Attention

**不完整。**

Transformer layer 还有：

- FFN；
- residual；
- LayerNorm；
- positional information。

---

# 71. 常见误解二：Attention 就是找最相关的一个 Token

**错误。**

它通常是：

\[
\sum_j\alpha_jv_j
\]

weighted combination。

---

# 72. 常见误解三：Q、K、V 是人工定义的语义

**错误。**

它们由：

\[
W_Q,W_K,W_V
\]

从 representation 学习得到。

---

# 73. 常见误解四：QKV 是三个不同 Input Sequences

在 self-attention 中：

**错误。**

它们都来自同一个：

\[
X
\]

的不同 projections。

Cross-attention 时 source 才不同。

---

# 74. 常见误解五：Self-Attention 意味着 Token 只看自己

**错误。**

“Self”表示 Q/K/V 来自同一个 sequence。

full self-attention 中每个 token 可以看所有 token。

---

# 75. 常见误解六：Transformer 不需要顺序信息

**错误。**

没有 recurrence/convolution 后，

原论文专门加入 positional encoding 来提供 order。

---

# 76. 常见误解七：Transformer 在任何情况下都比 RNN 计算复杂度低

**错误。**

原论文：

\[
SelfAttention:
O(n^2d)
\]

\[
RNN:
O(nd^2)
\]

具体取决于：

\[
n,d
\]

以及硬件与实现。

Transformer 的突出优势之一是：

> sequence positions 的并行化。

---

# 77. 常见误解八：Transformer Decoder 一定 Autoregressive

**错误。**

原始机器翻译 Transformer：

> 是。

ACT decoder：

> 不是。

是否 autoregressive 取决于 mask 与生成方式。

---

# 78. 常见误解九：Encoder 最终只输出一个 Context Vector

**错误。**

它输出整个 memory sequence：

\[
[m_1,\ldots,m_n]
\]

Decoder 可以 cross-attend 所有 positions。

---

# 79. 常见误解十：Attention Score 就是最终模型解释

**不准确。**

Attention 只是复杂网络的一部分。

高 attention weight 表示某次 attention operation 中较大的 value contribution，

不等于完整 causal explanation。

---

# 80. 常见误解十一：Multi-Head 的每个 Head 都一定有固定人类语义

**没有保证。**

不同 heads 有不同 learned subspaces，

但具体语义需要实证分析。

---

# 81. 常见误解十二：Position-Wise FFN 会让不同 Token 互相交流

**错误。**

不同 token 的交流主要发生在 attention。

FFN 独立地作用于每个 position，

只是共享参数。

---

# 82. 常见误解十三：Scaled Attention 除 \(\sqrt{d_k}\) 是为了让概率和为 1

**错误。**

概率和为 1 是：

\[
softmax
\]

的作用。

除：

\[
\sqrt{d_k}
\]

是为了控制 dot-product logits 的尺度，避免大 \(d_k\) 时 softmax 过度饱和。

---

# 83. 用五条公式记住 Transformer 核心

如果最后只记住五条公式：

## 1. QKV Projection

\[
\boxed{
Q=XW_Q,\quad
K=XW_K,\quad
V=XW_V
}
\]

---

## 2. Scaled Dot-Product Attention

\[
\boxed{
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
}
\]

---

## 3. Multi-Head

\[
\boxed{
\operatorname{MultiHead}(Q,K,V)
=
\operatorname{Concat}
(
head_1,\ldots,head_h
)
W^O
}
\]

---

## 4. Feed-Forward Network

\[
\boxed{
FFN(x)
=
\max(0,xW_1+b_1)W_2+b_2
}
\]

---

## 5. Original Transformer Sublayer

\[
\boxed{
\operatorname{LayerNorm}
(
x+\operatorname{Sublayer}(x)
)
}
\]

再加上：

> Positional Encoding

补充 sequence order。

这几乎就是原始 Transformer layer 的骨架。

---

# 84. 用一句话理解 Self-Attention

> **对于 sequence 中的每个位置，模型根据这个位置当前“想找什么”（Query），与所有位置“适合怎样被检索”（Key）进行匹配，把匹配分数变成权重后，对各位置真正提供的信息（Value）做加权汇总，从而一次并行地得到所有位置的 context-aware representations。**

---

# 85. 用一句话理解 Transformer

> **Transformer 把传统序列模型中依赖时间链传播的信息交换，替换成基于 attention 的直接全局信息路由：Self-Attention 负责不同位置之间的内容依赖，Multi-Head 提供多个 representation subspaces，Positional Encoding 提供顺序，FFN 对每个位置做非线性变换，而 Encoder–Decoder Cross-Attention 允许输出序列根据自身需求动态读取输入 memory。**

---

# 86. Transformer 和 ACT 最后怎样连起来？

现在回到我们最初学 Transformer 的目的。

ACT 中：

## CVAE Encoder

```text
[CLS]
joint
action sequence
↓
Self-Attention
↓
h_CLS
```

这里 Transformer 用来：

> 汇总一整段 demonstration action sequence。

---

## Policy Encoder

```text
1200 visual tokens
joint token
z token
↓
Self-Attention
↓
observation memory
```

这里 Transformer 用来：

> 融合多视角视觉、机器人状态与 latent condition。

---

## Policy Decoder

```text
k action queries
↓
Self-Attention
↓
Cross-Attention to observation memory
↓
k action representations
```

这里 Transformer 用来：

> 让未来 action positions 相互协调，并从当前 observation 中读取需要的信息。

所以 ACT 不是：

> “因为 Transformer 很强，所以用了 Transformer。”

而是 Transformer 提供的三种能力正好适合：

1. sequence summary；
2. multimodal token interaction；
3. query-based sequence output。

---

# 87. ACT 和原始 Transformer 的关键差异

| | Original Transformer | ACT Policy |
|---|---|---|
| 主要任务 | Machine Translation | Robot Action Prediction |
| Encoder 输入 | Text Tokens | Visual + Joint + Latent Tokens |
| Decoder 输入 | Shifted Target Tokens | Action Query Slots |
| Decoder causal mask | 有 | 无 |
| Decoder 是否 autoregressive | 是 | 否 |
| Cross-attention | Decoder → Source Memory | Action Queries → Observation Memory |
| 输出 | Vocabulary Probabilities | 14-D Joint Targets |
| 输出长度 | 逐步生成 | 固定 \(k\) 个 Action Slots |

所以：

> ACT 借用了 Transformer architecture，但并没有照搬机器翻译的 generation procedure。

---

# 88. 下一步应该拆什么？

现在 Transformer 主页面已经建立。

接下来最自然的是把其中最核心、也是最容易“懂公式但不懂本质”的部分独立出来：

> [Attention：模型到底是怎样“关注”信息的？](./attention.md)

然后继续：

> [Query / Key / Value：三个向量为什么能实现检索？](./qkv.md)

推荐顺序：

```text
Transformer
↓
Attention
↓
QKV
↓
Dot Product
↓
Softmax
↓
Multi-Head Attention
↓
Positional Encoding
↓
Cross-Attention
```

这样 ACT 页面中所有 Transformer 链接就会逐渐成为真正可递归阅读的知识图谱。

---

## Primary Source

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,  
Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin.

**Attention Is All You Need.**  
Advances in Neural Information Processing Systems 30, NIPS 2017.

- arXiv: https://arxiv.org/abs/1706.03762
- HTML: https://arxiv.org/html/1706.03762
- Google Research: https://research.google/pubs/attention-is-all-you-need/
- NeurIPS Proceedings: https://papers.nips.cc/paper/7181-attention-is-all-you-need

本文主要依据原论文：

- Section 1 — Introduction
- Section 2 — Background
- Section 3 — Model Architecture
- Section 3.1 — Encoder and Decoder Stacks
- Section 3.2 — Attention
- Section 3.2.1 — Scaled Dot-Product Attention
- Section 3.2.2 — Multi-Head Attention
- Section 3.2.3 — Applications of Attention in the Model
- Section 3.3 — Position-wise Feed-Forward Networks
- Section 3.5 — Positional Encoding
- Section 4 — Why Self-Attention

---

## 原论文 Base Transformer 关键参数

| Hyperparameter | Value |
|---|---:|
| Encoder layers | 6 |
| Decoder layers | 6 |
| \(d_{\text{model}}\) | 512 |
| \(d_{ff}\) | 2048 |
| Attention heads | 8 |
| \(d_k\) | 64 |
| \(d_v\) | 64 |
| Dropout | 0.1 |

原论文 Section 4 对比：

| Layer | Per-layer Complexity | Sequential Operations | Max Path Length |
|---|---:|---:|---:|
| Self-Attention | \(O(n^2d)\) | \(O(1)\) | \(O(1)\) |
| Recurrent | \(O(nd^2)\) | \(O(n)\) | \(O(n)\) |
| Convolutional | \(O(knd^2)\) | \(O(1)\) | \(O(\log_k n)\) |

这里：

- \(n\)：sequence length；
- \(d\)：representation dimension；
- \(k\)：convolution kernel size。

---

## 与 ACT 的关系

Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.  
**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.**  
RSS 2023.

- https://arxiv.org/abs/2304.13705

ACT 使用：

- BERT-like Transformer encoder 作为 training CVAE encoder；
- Transformer encoder 融合 visual / joint / latent tokens；
- Transformer decoder 根据 action queries cross-attend observation memory；
- non-autoregressive decoder 一次产生完整 action chunk。

因此 ACT 使用的是：

> **Transformer 的 attention-based sequence representation 与 encoder–decoder memory-reading机制，**

而不是原始机器翻译中的完整 autoregressive generation procedure。

---

## 本文知识连接

### 前置知识

- Vector
- Matrix
- [Dot Product](./dot-product.md)
- [Softmax](./softmax.md)
- Neural Network

### Transformer 核心

- [Attention](./attention.md)
- [Query / Key / Value](./qkv.md)
- [Self-Attention](./self-attention.md)
- [Cross-Attention](./cross-attention.md)
- [Multi-Head Attention](./multi-head-attention.md)
- [Positional Encoding](./positional-encoding.md)
- [Transformer Encoder](./transformer-encoder.md)
- [Transformer Decoder](./transformer-decoder.md)
- [Feed-Forward Network](./feed-forward-network.md)
- [Residual Connection](./residual-connection.md)
- [Layer Normalization](./layer-normalization.md)
- [Causal Mask](./causal-mask.md)
- [Dropout](./dropout.md)

### Transformer 之前

- RNN
- LSTM
- GRU
- Sequence-to-Sequence

### Robot Learning

- [ACT Architecture](../robot-learning/act/architecture.md)
- [CVAE in ACT](../robot-learning/act/cvae-in-act.md)
- [ACT Complete Data Flow](../robot-learning/act/complete-data-flow.md)

### 下一步

- [Attention](./attention.md)
- [Query / Key / Value](./qkv.md)
