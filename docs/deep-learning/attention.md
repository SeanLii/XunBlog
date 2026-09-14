---
title: "Attention：模型到底是怎样“关注”信息的？"
description: "从 Bahdanau Attention 的固定上下文瓶颈出发，逐步理解 query-dependent scoring、Softmax 权重、weighted sum、soft alignment，以及 additive attention、dot-product attention、self-attention 和 cross-attention 的统一视角。"
status: reviewed
pageType: concept
canonical: /deep-learning/attention
updated: "2026-09-15"
---

# Attention：模型到底是怎样“关注”信息的？

“Attention” 是深度学习里最容易被一句话讲懂、又最容易被一句话讲错的概念之一。

很多入门解释会说：

> Attention 就是让模型“关注重要的信息”。

这句话方向没错。

但它太模糊了。

真正的问题是：

> **模型怎么知道什么叫“重要”？**

如果当前任务不同，

同一份输入里“重要”的部分也会不同。

例如一句话：

```text
The cat sat on the mat because it was tired.
```

如果当前要理解：

```text
it
```

可能应该更多读取：

```text
cat
```

如果当前要判断：

```text
where did the cat sit?
```

又可能更需要：

```text
mat
```

所以“重要”不是一个固定属性。

更准确地说：

> **Attention 是一种根据当前需求，动态计算不同信息源应该贡献多少，再把它们加权组合起来的机制。**

这篇文章只研究 Attention 本身。

暂时不把 Transformer 的所有其他模块混进来。

我们会从 Attention 最早在神经机器翻译中的经典问题开始，然后一路推到：

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
\]

最后你会看到：

> Bahdanau Attention、Transformer Attention、Self-Attention、Cross-Attention，其实都可以放在同一个“动态检索”框架里理解。

---

# 1. Attention 最早想解决什么问题？

在早期 Encoder–Decoder 神经机器翻译中，

输入：

```text
I really enjoy learning artificial intelligence
```

先经过 Encoder，

被压缩成一个固定长度 context vector：

\[
c
\]

然后 Decoder 只依赖这个：

\[
c
\]

来生成整句翻译。

可以画成：

```text
x₁ x₂ x₃ ... xₙ
      │
      ▼
   Encoder
      │
      ▼
      c
      │
      ▼
   Decoder
      │
      ▼
y₁ y₂ y₃ ... yₘ
```

问题在于：

> 不管输入句子有 5 个词还是 50 个词，所有信息都必须压进同一个固定长度向量 \(c\)。

Bahdanau、Cho、Bengio 2014/2015 的经典工作指出：

> 这个 fixed-length vector 可能成为瓶颈。

他们提出：

> 不要强迫 Encoder 把整个句子压成一个单一向量；保留一整串 source representations，让 Decoder 每次生成一个词时，动态选择当前最相关的部分。

这就是经典 Attention 的出发点。

---

# 2. 固定 Context Vector 的真正问题

假设 Encoder 对 source sequence：

\[
x_1,x_2,\ldots,x_n
\]

产生 hidden representations：

\[
h_1,h_2,\ldots,h_n
\]

旧式 Encoder–Decoder 可能最终只保留：

\[
c=h_n
\]

或者某个固定汇总：

\[
c=q(h_1,\ldots,h_n)
\]

然后所有 Decoder steps 都使用同一个：

\[
c
\]

也就是说：

```text
生成第 1 个目标词：
用 c

生成第 2 个目标词：
还是用 c

生成第 10 个目标词：
还是同一个 c
```

但不同输出位置真正需要的信息显然不同。

所以 Attention 的第一个核心改变就是：

\[
\boxed{
c
\rightarrow
c_i
}
\]

也就是：

> **每一个输出位置都有自己的 context vector。**

---

# 3. Context 不再固定，而是动态计算

Bahdanau Attention 中，第 \(i\) 个目标词使用：

\[
\boxed{
c_i
=
\sum_{j=1}^{T_x}
\alpha_{ij}h_j
}
\]

这里：

- \(h_j\)：source 第 \(j\) 个位置的 representation；
- \(\alpha_{ij}\)：当生成第 \(i\) 个 target 时，source position \(j\) 应该贡献多少；
- \(c_i\)：当前 decoder step 真正读取到的 context。

这一个公式其实已经包含 Attention 最核心的思想：

> **先计算权重，再做 weighted sum。**

---

# 4. Attention 最核心的三步

无论后面是 Bahdanau Attention 还是 Transformer Attention，都可以先抽象成三步。

## Step 1：打分

对于当前需求和每个候选信息：

\[
\boxed{
e_j
=
\operatorname{score}(
\text{current need},
\text{candidate}_j
)
}
\]

---

## Step 2：把分数变成权重

通常：

\[
\boxed{
\alpha_j
=
\operatorname{softmax}(e)_j
}
\]

于是：

\[
\alpha_j\ge0
\]

并且：

\[
\sum_j\alpha_j=1
\]

---

## Step 3：加权读取

\[
\boxed{
c
=
\sum_j
\alpha_jv_j
}
\]

所以 Attention 可以压缩成：

```text
当前需求
↓
和每个候选计算相关性
↓
Softmax 变成权重
↓
对候选内容做加权平均
↓
得到当前 context
```

---

# 5. Attention 不是“重要性检测器”

这是一个必须先纠正的误解。

很多人会说：

> “Attention 找出输入里最重要的部分。”

更准确的是：

> **Attention 找出“对于当前 query 而言”更相关的部分。**

同一个 information source：

\[
h_j
\]

对于不同 query：

\[
q_1,q_2
\]

可以有完全不同的 attention weight。

所以重要性是：

\[
\boxed{
\text{query-dependent}
}
\]

不是某个 token 自带的永久属性。

---

# 6. 一个简单例子

假设我们有三个候选信息：

\[
v_1,v_2,v_3
\]

当前 query 对它们得到 scores：

\[
e=[2,1,0]
\]

经过 Softmax：

\[
\alpha
\approx
[0.665,0.245,0.090]
\]

所以输出：

\[
c
=
0.665v_1
+
0.245v_2
+
0.090v_3
\]

注意：

> Attention 没有只保留 \(v_1\)。

它是：

> 让 \(v_1\) 贡献最多，\(v_2\) 次之，\(v_3\) 仍然贡献一点。

这就是：

> **soft attention**

的含义。

---

# 7. 为什么 Softmax 很适合做 Attention Weight？

我们先有任意实数 scores：

\[
e_1,e_2,\ldots,e_n
\]

它们可能是：

\[
-3.2,\quad0.7,\quad4.1,\quad\ldots
\]

这些值不能直接当成 normalized weights。

Softmax：

\[
\boxed{
\alpha_j
=
\frac{
e^{e_j}
}{
\sum_k e^{e_k}
}
}
\]

会得到：

\[
\alpha_j>0
\]

以及：

\[
\sum_j\alpha_j=1
\]

所以这些 weights 可以很自然地解释为：

> 每个 candidate 对当前 context 的相对贡献。

详细数学可以阅读：

- [Softmax](./softmax.md)

---

# 8. Attention 的输出为什么是 Weighted Sum？

假设：

\[
v_j\in\mathbb R^d
\]

每个 candidate 都是同维度 vector。

则：

\[
c
=
\sum_j\alpha_jv_j
\]

仍然：

\[
c\in\mathbb R^d
\]

所以 Attention 可以把：

> 任意数量的 candidate vectors

动态汇总成：

> 一个固定维度的 context vector。

而且这个 context 会随着：

\[
\alpha
\]

变化。

也就是说：

> 同一组 memory，在不同 query 下可以被读取成不同 context。

---

# 9. Bahdanau Attention 中“当前需求”是什么？

在经典神经机器翻译里，

Decoder 正要生成：

\[
y_i
\]

这时它已经有上一步 hidden state：

\[
s_{i-1}
\]

可以把：

\[
s_{i-1}
\]

看成当前 Decoder 的状态：

> “根据我已经翻译到这里，我现在需要 source sentence 中的什么信息？”

Source 每个位置有 annotation：

\[
h_j
\]

于是 Bahdanau Attention 计算：

\[
\boxed{
e_{ij}
=
a(
s_{i-1},
h_j
)
}
\]

其中：

\[
a
\]

是一个 learned alignment model。

---

# 10. “Alignment”是什么意思？

机器翻译里，经常存在词和词之间的对应关系。

例如：

```text
English:
The   cat   eats   fish

Chinese:
这只  猫    吃     鱼
```

大致有：

```text
cat ↔ 猫
eats ↔ 吃
fish ↔ 鱼
```

所以当 Decoder 正准备生成：

```text
鱼
```

时，

理想情况下应该更多读取 source 中：

```text
fish
```

Bahdanau 论文把：

\[
e_{ij}
\]

称为 alignment score。

它表示：

> source position \(j\) 和当前 target position \(i\) 有多匹配。

然后：

\[
\alpha_{ij}
=
\operatorname{softmax}_j(e_{ij})
\]

形成：

> soft alignment。

---

# 11. 为什么叫“Soft Alignment”？

传统 alignment 可以想象成：

```text
target word i
只对应 source word j
```

也就是 hard choice。

Attention 不这么做。

它允许：

\[
\alpha_{i1}=0.05
\]

\[
\alpha_{i2}=0.15
\]

\[
\alpha_{i3}=0.70
\]

\[
\alpha_{i4}=0.10
\]

所以 target position \(i\) 可以同时参考多个 source positions。

这就是：

> **soft alignment**

而且整个过程：

- score；
- Softmax；
- weighted sum；

都是可微的。

所以模型可以端到端训练。

---

# 12. Bahdanau 论文中最核心的两个公式

第一：

\[
\boxed{
c_i
=
\sum_{j=1}^{T_x}
\alpha_{ij}h_j
}
\]

第二：

\[
\boxed{
\alpha_{ij}
=
\frac{
\exp(e_{ij})
}{
\sum_{k=1}^{T_x}\exp(e_{ik})
}
}
\]

其中：

\[
\boxed{
e_{ij}
=
a(s_{i-1},h_j)
}
\]

这三个量的角色分别是：

```text
e_ij
= 未归一化匹配分数

α_ij
= 归一化 attention 权重

c_i
= 根据当前 query 动态得到的 context
```

这就是 Attention 最初非常经典的数学骨架。

---

# 13. Attention 真正解决的是“动态访问 Memory”

可以换一个更通用的视角。

假设模型拥有一个 memory：

\[
M=
\{m_1,m_2,\ldots,m_n\}
\]

现在有一个 query：

\[
q
\]

模型想知道：

> memory 中哪些内容和 \(q\) 相关？

于是：

\[
q
\]

与每个：

\[
m_j
\]

计算 score。

再把 score 转成 weights。

最终得到：

\[
c
=
\sum_j\alpha_jm_j
\]

从这个角度看，

Attention 是：

\[
\boxed{
\text{Differentiable Content-Based Memory Retrieval}
}
\]

即：

> **可微的、基于内容的 memory 检索。**

这是比“关注重点”更本质的理解。

---

# 14. Query 是“需求”，Candidate 是“信息源”

为了进入更现代的 Attention 记号，我们可以先抽象：

当前需求：

\[
q
\]

候选信息源有：

\[
(k_1,v_1),
(k_2,v_2),
\ldots,
(k_n,v_n)
\]

每个信息源拆成：

- \(k_j\)：用来匹配；
- \(v_j\)：真正被读取的信息。

于是：

\[
e_j
=
\operatorname{score}(q,k_j)
\]

然后：

\[
\alpha_j
=
\operatorname{softmax}(e)_j
\]

最后：

\[
\boxed{
c
=
\sum_j\alpha_jv_j
}
\]

这就自然进入了：

> **Query–Key–Value**

形式。

---

# 15. 为什么 Key 和 Value 要分开？

这是 Attention 中非常关键的一步。

假设你在数据库里搜索：

```text
“Transformer”
```

每条记录可能有：

### Key

```text
标题
标签
索引字段
```

用于判断：

> 这条记录与 query 匹不匹配。

### Value

```text
完整内容
```

用于：

> 匹配成功后真正读取。

所以：

\[
Key
\]

回答：

> “你为什么应该读我？”

\[
Value
\]

回答：

> “如果你决定读我，我真正提供什么？”

这就是为什么现代 Attention 把：

> matching representation

和：

> transmitted information

分开。

---

# 16. Key 不等于 Value

很多人刚学 QKV 会想：

> 既然 Key 和 Value 都来自同一个 token，为什么不直接用同一个向量？

可以。

某些简单 Attention 形式确实没有严格分开。

但 learned projections：

\[
k_j=x_jW_K
\]

\[
v_j=x_jW_V
\]

允许模型学习：

> 用一套 feature 判断相关性，

再用另一套 feature 传递内容。

例如一个 token 中：

- 某些 feature 更适合“被找到”；
- 另一些 feature 更适合“被读取”。

这让 Attention 更灵活。

详细会在：

- [Query / Key / Value](./qkv.md)

中继续拆。

---

# 17. Transformer 对 Attention 做了什么改变？

Bahdanau Attention 已经有：

```text
当前 Decoder 状态
↓
与所有 Encoder states 打分
↓
Softmax
↓
Weighted Sum
```

Transformer 并不是“第一次发明 Attention”。

它做的关键改变之一是：

> 把 Attention 本身提升为主干计算机制。

原始 Transformer 直接定义：

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

这里最核心的骨架仍然完全一样：

```text
score
↓
normalize
↓
weighted sum
```

只是 compatibility function 变成了：

> scaled dot product。

---

# 18. Bahdanau Attention 和 Transformer Attention 有什么共同点？

共同结构：

\[
\boxed{
\text{Query}
+
\text{Candidates}
\rightarrow
\text{Scores}
\rightarrow
\text{Weights}
\rightarrow
\text{Weighted Sum}
}
\]

不同点主要是：

> **score function 怎么定义。**

Bahdanau：

\[
e_{ij}
=
a(s_{i-1},h_j)
\]

其中 \(a\) 是 learned feed-forward network。

Transformer：

\[
e_{ij}
=
\frac{
q_i^\top k_j
}{
\sqrt{d_k}
}
\]

使用 scaled dot product。

所以 Attention 的本质并不等于：

> dot product。

Dot product 只是：

> 一种 compatibility function。

---

# 19. Additive Attention 是什么？

Bahdanau Attention 通常被归入：

> **Additive Attention**

一种常见写法是：

\[
\boxed{
e_{ij}
=
v_a^\top
\tanh(
W_q q_i
+
W_k k_j
)
}
\]

这里：

- \(W_q q_i\)：query 投影；
- \(W_k k_j\)：key 投影；
- 相加；
- 经过 \(\tanh\)；
- 再由 \(v_a\) 压成一个 scalar score。

所以 score 不是：

\[
q^\top k
\]

而是通过一个小 feed-forward network 学出来。

---

# 20. Dot-Product Attention 是什么？

更简单：

\[
\boxed{
e_{ij}
=
q_i^\top k_j
}
\]

Transformer 又加入 scale：

\[
\boxed{
e_{ij}
=
\frac{
q_i^\top k_j
}{
\sqrt{d_k}
}
}
\]

再：

\[
\alpha_{ij}
=
\operatorname{softmax}_j(e_{ij})
\]

最后：

\[
o_i
=
\sum_j
\alpha_{ij}v_j
\]

所以 Transformer Attention 的核心计算可以展开成：

\[
\boxed{
o_i
=
\sum_j
\operatorname{softmax}_j
\left(
\frac{
q_i^\top k_j
}{
\sqrt{d_k}
}
\right)
v_j
}
\]

---

# 21. 为什么 Dot Product 很适合现代硬件？

如果有一批 queries：

\[
Q
\in
\mathbb R^{n_q\times d_k}
\]

keys：

\[
K
\in
\mathbb R^{n_k\times d_k}
\]

那么所有 pairwise scores 可以一次计算：

\[
\boxed{
QK^\top
}
\]

得到：

\[
[n_q,n_k]
\]

attention score matrix。

这是一整块矩阵乘法。

GPU / TPU 对这种操作非常高效。

Transformer 原论文也明确指出：

> dot-product attention 可以利用高度优化的 matrix multiplication code，因此实践上速度和空间效率很好。

---

# 22. Scaled Dot-Product 为什么需要 Scale？

如果：

\[
q,k
\]

每一维均值约为 0、方差约为 1，

那么：

\[
q^\top k
=
\sum_{r=1}^{d_k}q_rk_r
\]

其方差会随：

\[
d_k
\]

增长。

原论文给出的近似关系：

\[
\operatorname{Var}(q^\top k)
=
d_k
\]

所以：

\[
d_k
\]

大时 dot product magnitude 会变大。

Softmax 输入过大容易非常尖锐，

进入 gradient 很小的区域。

因此除以：

\[
\boxed{
\sqrt{d_k}
}
\]

让 logits 的典型尺度更加稳定。

---

# 23. Attention Weight 到底表示什么？

对于 query：

\[
q_i
\]

和 key：

\[
k_j
\]

最终：

\[
\alpha_{ij}
\]

表示：

> **在当前这一次 Attention operation 中，输出位置 \(i\) 聚合 value \(v_j\) 时使用的权重。**

这是一个非常精确的说法。

不要立刻升级成：

> “模型认为 token \(j\) 在整个任务中有 \(\alpha_{ij}\) 的重要性。”

因为一个完整模型里还有：

- 多个 attention heads；
- 多个 layers；
- residual connections；
- FFN；
- downstream computation。

所以：

\[
\alpha_{ij}
\]

只描述：

> 当前 head、当前 layer、当前 query 下的一次信息聚合权重。

---

# 24. Attention Weight 是概率吗？

Softmax 后：

\[
\alpha_j\ge0
\]

且：

\[
\sum_j\alpha_j=1
\]

所以数学上它形成一个：

> categorical probability distribution 形式。

Bahdanau 论文甚至把：

\[
\alpha_{ij}
\]

解释成：

> target word \(i\) 与 source word \(j\) alignment 的概率式权重。

但在现代 Transformer 中，

通常更安全的说法是：

> normalized attention weights。

因为它不一定是一个经过概率模型语义定义的真实事件概率。

---

# 25. Attention 是 Hard Selection 吗？

标准 soft attention 不是。

Hard selection 可能是：

\[
j^\star
=
\arg\max_j e_j
\]

然后：

\[
c=v_{j^\star}
\]

这会只选一个 candidate。

而 soft attention：

\[
c=\sum_j\alpha_jv_j
\]

让所有 positions 都可以参与。

这有一个很重要的优化优势：

> weighted sum 对 scores 是连续可微的。

所以 gradient 可以通过：

\[
c
\rightarrow
\alpha
\rightarrow
e
\]

反向传播。

---

# 26. Bahdanau Attention 为什么可以 End-to-End 训练？

因为：

\[
e_{ij}
\]

由 neural network 计算；

\[
\alpha_{ij}
\]

通过 Softmax；

\[
c_i
\]

通过 weighted sum。

这些都是可微操作。

所以 translation loss：

\[
L
\]

可以一路反传：

\[
L
\rightarrow
c_i
\rightarrow
\alpha_{ij}
\rightarrow
e_{ij}
\rightarrow
\text{alignment model parameters}
\]

因此不需要额外人工 alignment labels。

Bahdanau 论文强调：

> alignment model 与整个 translation model 一起 joint training。

---

# 27. “模型自己学会看哪里”具体是什么意思？

不是模型突然出现了人类意义上的视觉注意力。

真正发生的是：

1. 初始 score function 参数随机；
2. attention weights 可能很乱；
3. 模型生成 translation；
4. prediction loss 告诉模型结果好不好；
5. gradient 通过 attention 路径更新 scoring function；
6. 某些更有利于降低 loss 的匹配模式被强化。

所以最终：

> 某些 query 对某些 keys 得到更高 scores。

这就是“学会关注”的数学含义。

---

# 28. Attention 本质上是动态权重，而不是固定权重

普通 Linear layer：

\[
y=Wx
\]

其中：

\[
W
\]

训练完成后，对所有输入都固定。

Attention 不一样。

虽然：

\[
W_Q,W_K,W_V
\]

这些 projection parameters 固定，

但真正的：

\[
A
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)
\]

取决于当前：

\[
Q,K
\]

也就是取决于当前 input。

因此：

\[
\boxed{
\text{Attention weights are input-dependent}
}
\]

这是 Attention 非常强的一点。

---

# 29. 可以把 Attention 看成“动态生成一个临时连接矩阵”

普通神经网络某一层的连接权重：

\[
W
\]

训练后固定。

Self-Attention 每一次 forward 都会产生：

\[
A
\in
\mathbb R^{n\times n}
\]

例如：

\[
A_{ij}
\]

表示：

> 当前输入下，position \(i\) 从 position \(j\) 读取多少。

所以可以直觉理解为：

> **模型根据当前样本，临时生成一张信息路由图。**

输入变了：

\[
A
\]

也会变。

---

# 30. Attention Output 不是简单“复制 Memory”

因为：

\[
o_i
=
\sum_j\alpha_{ij}v_j
\]

通常会把多个 values 混合。

例如：

\[
\alpha=
[0.5,0.3,0.2]
\]

则：

\[
o
=
0.5v_1+0.3v_2+0.2v_3
\]

所以输出可能是：

> memory 中没有任何单一位置完全等于的新 vector。

Attention 是一种：

> **context construction mechanism**

而不只是检索原始项。

---

# 31. 一个更像数据库的例子

假设 memory 中有：

```text
Record 1:
Key = "cat"
Value = [animal-related features]

Record 2:
Key = "street"
Value = [location-related features]

Record 3:
Key = "tired"
Value = [state-related features]
```

query：

```text
“Who was tired?”
```

可能产生：

\[
[0.7,0.05,0.25]
\]

于是 context：

\[
0.7v_{\text{cat}}
+
0.05v_{\text{street}}
+
0.25v_{\text{tired}}
\]

模型并不是硬查一个 SQL row。

而是：

> soft retrieval + feature mixing。

---

# 32. Self-Attention 是什么？

现在把 query source 和 memory source 设成同一个 sequence。

输入：

\[
X=
[x_1,\ldots,x_n]
\]

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

于是每个 position 都在问：

> “在同一条 sequence 里，我现在应该从哪些 positions 读取信息？”

这就是：

\[
\boxed{
\text{Self-Attention}
}
\]

---

# 33. Cross-Attention 是什么？

如果 query 来自：

\[
X
\]

而 key/value 来自另一组：

\[
M
\]

则：

\[
Q=XW_Q
\]

\[
K=MW_K
\]

\[
V=MW_V
\]

这就是：

\[
\boxed{
\text{Cross-Attention}
}
\]

它表达：

> 一组 representations 根据自己的需求，去另一组 memory 中读取信息。

原始 Bahdanau Attention 本质上就很接近这种：

> Decoder state → Encoder memory

的 cross-attention。

---

# 34. Bahdanau Attention 可以看成 Cross-Attention 的祖先形式吗？

从现代统一视角：

> 可以这样理解。

Bahdanau 模型中：

Query-like information：

\[
s_{i-1}
\]

来自 Decoder。

Memory：

\[
h_1,\ldots,h_{T_x}
\]

来自 Encoder。

然后：

\[
e_{ij}=a(s_{i-1},h_j)
\]

生成 weights，

再读取：

\[
h_j
\]

形成：

\[
c_i
\]

所以结构上就是：

```text
Decoder state
↓ query-like
Encoder states
↓ keys / values-like
动态读取
```

只是当时还没有 Transformer 中标准化的 Q/K/V 矩阵写法。

---

# 35. Self-Attention 和 Bahdanau Attention 最大的结构差异

Bahdanau：

```text
Decoder 一个位置
↓
读取 Encoder 所有位置
```

主要是：

> 两个 sequence 之间的信息交互。

Transformer Encoder Self-Attention：

```text
Input position 1
↔
Input position 2
↔
...
↔
Input position n
```

是：

> 同一个 sequence 内部所有位置互相动态读取。

这一步非常关键。

因为 Attention 从：

> Encoder–Decoder 之间的辅助机制

变成：

> sequence representation 的主干机制。

---

# 36. Attention 为什么可以处理 Variable-Length Memory？

因为 weighted sum：

\[
c=\sum_{j=1}^n\alpha_jv_j
\]

最终 output dimension 由：

\[
v_j
\]

的维度决定，

而不是由：

\[
n
\]

决定。

所以 memory 可以有：

\[
n=5
\]

也可以：

\[
n=50
\]

Attention 仍输出：

\[
d_v
\]

维 context。

这也是它非常适合处理 variable-length sequence 的原因之一。

---

# 37. Attention 是不是压缩？

某种意义上，

单个 query 对：

\[
n
\]

个 values 做 weighted sum，

确实得到一个：

\[
d_v
\]

维向量。

所以对这个 query 来说是：

> 动态汇总。

但与固定 context bottleneck 不同，

Attention 的汇总方式：

> 每个 query 都可以重新计算。

所以：

```text
旧方案：
整个输入永远压成同一个 c

Attention：
query 1 得到 c₁
query 2 得到 c₂
query 3 得到 c₃
```

这就是巨大区别。

---

# 38. 一个 Input 可以同时产生很多不同 Context

如果有：

\[
m
\]

个 queries，

则每个 query 都有自己的 attention distribution：

\[
\alpha_1,\ldots,\alpha_m
\]

最终：

\[
c_1,\ldots,c_m
\]

所以 memory：

\[
V
\]

没有改变，

但不同 query 可以从中得到完全不同的读取结果。

这就是：

> **query-conditioned representation**

---

# 39. Matrix Form 为什么这么重要？

单个 query：

\[
o_i
=
\sum_j
\alpha_{ij}v_j
\]

如果有很多 queries，

把它们堆成：

\[
Q
\]

keys 堆成：

\[
K
\]

values 堆成：

\[
V
\]

就得到：

\[
S=QK^\top
\]

然后：

\[
A=
\operatorname{softmax}(S)
\]

最后：

\[
\boxed{
O=AV
}
\]

Transformer 再加 scale：

\[
\boxed{
O=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
}
\]

这就是把“一个 query 查 memory”的逻辑，

并行扩展到：

> 很多 queries 同时查 memory。

---

# 40. Attention Matrix 每一行是什么意思？

如果：

\[
A
\in
\mathbb R^{n_q\times n_k}
\]

则第：

\[
i
\]

行：

\[
A_{i,:}
\]

表示：

> query \(i\) 对所有 keys 的 attention distribution。

例如：

\[
[0.1,0.7,0.2]
\]

表示：

> query \(i\) 主要读取第二个 value。

---

# 41. Attention Matrix 每一列是什么意思？

第：

\[
j
\]

列：

\[
A_{:,j}
\]

表示：

> 不同 queries 对 key/value position \(j\) 分配了多少权重。

但注意：

> Softmax 通常按 row，也就是每个 query 自己的 key dimension 做归一化。

所以不同 rows 各自和为 1。

columns 一般不需要和为 1。

---

# 42. 为什么 Softmax 通常沿 Key Dimension 做？

对每个 query：

\[
q_i
\]

我们想回答：

> “所有 candidate keys 中，我应该怎样分配读取权重？”

所以固定 query \(i\)，

对：

\[
j=1,\ldots,n_k
\]

做 Softmax。

因此：

\[
\sum_j\alpha_{ij}=1
\]

而不是：

\[
\sum_i\alpha_{ij}=1
\]

---

# 43. Mask 在 Attention 中是什么？

有时某些 key positions：

> 不允许被当前 query 读取。

例如 autoregressive language model 中，

当前位置不能看未来 token。

那么在 Softmax 前给这些 score 加：

\[
-\infty
\]

例如：

\[
S=
[2,1,-\infty]
\]

Softmax 后：

\[
\alpha_3=0
\]

所以 mask 本质是：

> **在 attention candidate set 中禁用某些位置。**

---

# 44. Padding Mask 又是什么？

假设 batch 中 sequence 长度不同。

为了对齐 tensor：

```text
真实 token
真实 token
PAD
PAD
```

PAD 不应该被模型当作有意义 memory。

所以 attention score 对 PAD positions 也会被 mask。

这叫：

> **Padding Mask**

和：

> **Causal Mask**

作用不同。

Causal mask：

> 禁止看未来。

Padding mask：

> 禁止读取不存在的 padded positions。

---

# 45. Attention 可以有很多种 Score Function

Attention 的统一形式：

\[
e_{ij}
=
\operatorname{score}(q_i,k_j)
\]

score 可以有很多选择。

例如：

### Additive

\[
v^\top\tanh(W_qq+W_kk)
\]

### Dot Product

\[
q^\top k
\]

### Scaled Dot Product

\[
\frac{q^\top k}{\sqrt{d_k}}
\]

### Bilinear

\[
q^\top Wk
\]

所以：

> **Attention 的本质是“动态打分 + 加权读取”，不是某一个固定 score 公式。**

---

# 46. 为什么 Transformer 选 Scaled Dot Product？

Transformer 原论文给出两个主要工程原因：

1. dot-product attention 和 additive attention 理论复杂度相近；
2. dot product 可以高度利用优化后的 matrix multiplication，因此实践上更快、更省空间。

同时为了避免大：

\[
d_k
\]

导致 logits 过大，

加入：

\[
1/\sqrt{d_k}
\]

scale。

所以 Transformer Attention 是：

> 一个特别适合并行硬件的 Attention 实现。

---

# 47. Attention 和 Weighted Average 有什么关系？

如果：

\[
\alpha_j\ge0
\]

且：

\[
\sum_j\alpha_j=1
\]

那么：

\[
c=\sum_j\alpha_jv_j
\]

确实是：

> weighted average / convex combination。

所以 Attention output 落在 values 的 convex hull 中——如果只看这一层 weighted sum 且 values 固定。

但在 Transformer 中：

- values 本身是 learned projections；
- 多 head；
- 后面还有 output projection；
- residual；
- FFN。

所以整层 Transformer 的输出当然不受限于简单“原 token 的平均”。

---

# 48. Attention 会不会把信息“平均糊掉”？

单个 attention head 的确是在做 weighted sum。

但它不是固定平均：

\[
\frac1n\sum_jv_j
\]

而是：

\[
\sum_j\alpha_j(q)v_j
\]

权重依赖 query。

如果某个位置特别相关：

\[
\alpha_j
\]

可以很大。

同时 Multi-Head Attention 又允许：

> 不同 heads 读取不同 patterns。

所以它比简单平均灵活得多。

---

# 49. Attention Weight 是 Learned Parameter 吗？

这是一个非常关键的问题。

\[
W_Q,W_K,W_V
\]

是 learned parameters。

但：

\[
\alpha_{ij}
\]

通常不是直接存储的 learned parameter。

它是每次 forward 动态算出来的：

\[
\alpha_{ij}
=
f(
x_i,x_j;
W_Q,W_K
)
\]

所以：

```text
模型学的是：
怎样计算 attention

不是：
把每一对 token 的权重直接背下来
```

---

# 50. 为什么 Attention 可以泛化到没见过的句子？

因为模型不是记：

```text
token 3 永远关注 token 8
```

它学的是：

\[
W_Q,W_K,W_V
\]

以及 scoring rule。

新输入进来后，

根据新 token representations 动态生成：

\[
Q,K,V
\]

再计算新的 attention matrix。

所以 relationship 是：

> content-dependent。

---

# 51. Attention 和 Fully Connected Layer 有什么不同？

Linear / fully connected：

\[
Y=XW
\]

参数：

\[
W
\]

对所有 samples 固定。

Attention：

\[
O=A(X)V(X)
\]

其中：

\[
A(X)
=
\operatorname{softmax}
(
Q(X)K(X)^\top
)
\]

所以：

> token-to-token mixing matrix \(A\) 会随着输入变化。

这是一个很大的本质差异。

---

# 52. 可以把 Self-Attention 看成动态 Graph 吗？

作为高级直觉：

> 可以。

每个 token 是 node。

Attention weight：

\[
\alpha_{ij}
\]

可以理解为：

> 当前 forward 中 node \(i\) 从 node \(j\) 接收多少 message。

所以：

```text
tokens
↓
动态计算 edge weights
↓
message aggregation
```

这和 message passing 有一定结构相似性。

但不要直接把 Transformer 等同于传统 Graph Neural Network。

这里只是帮助理解：

> Attention 动态构建信息连接强度。

---

# 53. Attention 和 Memory 有什么关系？

从最通用视角：

\[
K,V
\]

可以看成 memory。

Query：

\[
Q
\]

是读取请求。

Attention：

> 根据 query 在 memory 中做软寻址。

所以很多模型中可以看到：

```text
Query
→ Memory
→ Retrieved Context
```

Cross-Attention 特别符合这个视角。

---

# 54. Attention 为什么特别适合多模态？

假设 memory 中同时有：

- image tokens；
- text tokens；
- robot-state tokens。

只要都映射到 compatible representation space，

query 就可以对这些 keys 计算 score。

所以 Attention 并不要求：

> 所有候选信息来自同一种 modality。

这也是为什么后来 Transformer 很自然进入：

- vision；
- speech；
- robotics；
- multimodal models。

---

# 55. 回到 ACT：Policy Encoder 的 Self-Attention

ACT policy encoder 输入：

```text
z token
joint token
1200 visual tokens
```

所以每一个 token 都可以通过 self-attention 读取其他 token。

例如一个 wrist-camera visual token 理论上可以：

- 读取 top camera tokens；
- 读取 joint state；
- 读取 latent \(z\)。

所以 ACT 用 Self-Attention 做：

> 多视角视觉 + proprioception + latent condition 的动态融合。

---

# 56. 回到 ACT：Policy Decoder 的 Cross-Attention

ACT decoder 有：

\[
k
\]

个 action query slots。

每个 slot 的 query 会对 policy encoder memory：

\[
M
\]

计算 attention。

所以第 \(i\) 个 future action slot 可以理解为：

> “为了预测 chunk 中第 \(i\) 个动作，我应该从当前 observation memory 的哪些位置读取信息？”

这里：

\[
Q
\]

来自 action decoder。

\[
K,V
\]

来自 observation encoder memory。

所以是：

\[
\boxed{
Cross\text{-}Attention
}
\]

---

# 57. ACT 中的 Temporal Ensemble 不是 Attention

这个区别非常重要。

Temporal Ensemble：

\[
a_t
=
\sum_i
\alpha_i
\hat a_t^{(i)}
\]

看起来也是 weighted sum。

但它的 weights：

\[
\alpha_i
\propto
e^{-mi}
\]

是：

> 人工设计的时间衰减公式。

Transformer Attention：

\[
\alpha_{ij}
=
\operatorname{softmax}
(
q_i^\top k_j
)
\]

是：

> 根据当前 representations 动态计算。

所以：

```text
Temporal Ensemble
= hand-designed weighted average

Attention
= learned, content-dependent weighted retrieval
```

两者不能因为都有“加权平均”就混为一谈。

---

# 58. Attention 和 Average Pooling 有什么本质区别？

Average pooling：

\[
c
=
\frac1n
\sum_jv_j
\]

所有位置权重：

\[
\alpha_j=\frac1n
\]

固定。

Attention：

\[
c(q)
=
\sum_j
\alpha_j(q)v_j
\]

权重：

> 随 query 和输入动态变化。

所以可以说：

\[
\boxed{
\text{Attention}
=
\text{Adaptive Weighted Pooling}
}
\]

但它比普通 pooling 更强，因为 score 本身是 learned content matching。

---

# 59. 为什么 Attention 不一定等于“视觉注意力”？

“Attention”这个词来自人类直觉，

但神经网络里的 Attention 是一个数学机制。

它可以用于：

- word → word；
- action query → image token；
- image patch → image patch；
- latent token → joint token；
- decoder state → encoder memory。

所以 Attention 不要求：

> 模型真的像人类一样“把眼睛看向某处”。

它的正式含义是：

> 动态加权信息聚合。

---

# 60. Attention 有没有可能完全均匀？

当然。

如果所有 scores 相同：

\[
e_j=c
\]

那么 Softmax：

\[
\alpha_j=\frac1n
\]

于是 Attention 退化成：

> average pooling。

所以 Attention 并不强制模型一定产生尖锐 focus。

它可以：

- 非常集中；
- 比较分散；
- 几乎均匀。

完全取决于当前 learned scores。

---

# 61. Attention 有没有可能只关注一个位置？

Softmax 不会在有限 logits 下产生严格：

\[
1,0,0,\ldots
\]

但如果一个 score 远大于其他：

\[
e_1\gg e_j
\]

就可以得到：

\[
\alpha_1\approx1
\]

所以 soft attention 可以近似 hard selection。

---

# 62. Temperature 和 Attention Sharpness 有什么关系？

一般 Softmax 可以写：

\[
\operatorname{softmax}
\left(
\frac{e}{\tau}
\right)
\]

如果：

\[
\tau
\]

小，

distribution 更尖锐。

如果：

\[
\tau
\]

大，

distribution 更平滑。

Transformer 的：

\[
\sqrt{d_k}
\]

不是一个训练 temperature parameter，

但从形式上它确实在调节 logits 尺度，

从而影响 Softmax saturation。

---

# 63. 为什么 Attention 可以看成期望？

如果：

\[
\alpha_j
\]

看成离散 distribution：

\[
P(J=j)=\alpha_j
\]

那么：

\[
c
=
\sum_j\alpha_jv_j
\]

就是：

\[
\boxed{
c
=
\mathbb E_{J\sim\alpha}[v_J]
}
\]

Bahdanau 论文也明确给出了类似解释：

> context vector 可以理解为在 soft alignment distribution 下 annotation 的期望。

这个视角很漂亮。

---

# 64. 但 Attention Output 不是随机 Sample

虽然我们可以把：

\[
\alpha
\]

解释成 distribution，

标准 soft attention 并不会：

\[
J\sim\alpha
\]

然后只取：

\[
v_J
\]

而是直接计算：

\[
\sum_j\alpha_jv_j
\]

所以 output 是 deterministic weighted expectation。

这也是它可以稳定反向传播的原因之一。

---

# 65. Attention 和 Mixture 有什么相似之处？

形式上：

\[
c
=
\sum_j\alpha_jv_j
\]

看起来像 mixture。

不同 candidates 像不同 components，

\[
\alpha
\]

像 mixture weights。

但不要直接把 Attention 当成概率 mixture model。

因为：

- values 不一定定义概率分布；
- weights 的语义通常是 representation routing；
- 没有要求满足 generative mixture model 的全部概率结构。

只是数学形式相似。

---

# 66. Attention 为什么能处理 Long-Range Dependency？

因为一个 query 可以直接对任何 key 计算 score。

例如 position：

\[
i=1000
\]

可以直接与：

\[
j=1
\]

形成：

\[
q_{1000}^\top k_1
\]

不需要信息逐位置经过：

\[
2,3,4,\ldots,999
\]

这让远距离 interaction path 非常短。

这正是 Transformer 之后大量 sequence modeling 成功的重要结构原因之一。

---

# 67. Attention 的代价是什么？

Full self-attention 需要计算：

\[
QK^\top
\]

如果 sequence length：

\[
n
\]

则 score matrix：

\[
n\times n
\]

所以时间 / memory 对 sequence length 通常是：

\[
O(n^2)
\]

级别。

因此 Attention 的优势不是：

> “完全免费地看全局。”

而是：

> 用 quadratic pairwise interaction cost，换取全局直接信息访问和高度并行。

---

# 68. 为什么“Attention 会自动理解关系”这句话不够严谨？

Attention architecture 只提供：

> pairwise scoring + weighted aggregation 的能力。

它是否真的学到：

- grammar；
- object correspondence；
- robot geometry；

取决于：

- data；
- loss；
- model capacity；
- optimization；
- positional information；
- architecture。

所以更准确地说：

> Attention 提供一种允许模型学习内容依赖关系的机制。

不是：

> Attention 公式本身已经理解语义。

---

# 69. Attention 会不会丢失顺序？

Attention score 如果只由 content：

\[
q_i^\top k_j
\]

决定，

而没有任何 position information，

确实不天然知道：

> token 的绝对/相对顺序。

所以 Self-Attention-based sequence model 通常还需要：

- positional encoding；
- relative position bias；
- rotary embedding；
- 其他 position mechanisms。

因此：

> Attention 负责“谁和谁交互”。

Position mechanism 负责：

> “这些 token 在序列/空间中的位置关系是什么”。

---

# 70. Attention 中 Position 信息通常怎么进入？

一种经典方式：

先：

\[
x_i
=
embedding_i
+
PE_i
\]

再：

\[
q_i=x_iW_Q
\]

\[
k_i=x_iW_K
\]

于是 query / key 已经包含 position information。

现代模型也可能直接在 score 中加入：

- relative position bias；
- RoPE 形成的相对旋转关系。

这些是后续扩展。

原始 Transformer 使用：

> sinusoidal positional encoding。

---

# 71. Multi-Head Attention 为什么是 Attention 的自然扩展？

单头 Attention：

\[
q_i
\]

只能在一个 learned matching space 中决定：

> 哪些 keys 相关。

Multi-head 则：

\[
head_1
\]

用一组：

\[
W_Q^{(1)},W_K^{(1)},W_V^{(1)}
\]

而：

\[
head_2
\]

使用另一组。

所以同一个 query 可以同时：

> 用多种 learned criteria 读取 memory。

然后：

\[
\operatorname{Concat}(head_1,\ldots,head_h)
\]

综合起来。

详细会在：

- [Multi-Head Attention](./multi-head-attention.md)

单独展开。

---

# 72. Attention 的真正“可学习部分”有哪些？

以 Transformer 为例：

\[
W_Q,W_K,W_V
\]

是 learned。

Multi-head output：

\[
W_O
\]

也是 learned。

但是公式：

\[
QK^\top
\]

本身不是参数。

Softmax 也没有 learned parameters。

所以模型学习的是：

> 怎样把原 representation 投影成适合匹配和传递的空间。

然后 Attention 公式利用这些 learned representations 动态生成路由。

---

# 73. Attention Score 和 Attention Weight 必须区分

### Score / Logit

\[
e_{ij}
\]

可以是任意实数：

\[
-5,\;0.3,\;8.2
\]

表示未归一化 compatibility。

---

### Weight

\[
\alpha_{ij}
\]

经过 Softmax：

\[
0\le\alpha_{ij}\le1
\]

并且：

\[
\sum_j\alpha_{ij}=1
\]

所以不要把：

\[
q^\top k
\]

直接称为最终 attention weight。

它只是：

> attention score / logit。

---

# 74. 为什么 Score 可以是负数，Weight 却非负？

因为 Softmax 使用：

\[
e^{e_j}
\]

无论：

\[
e_j
\]

正负，

指数：

\[
e^{e_j}>0
\]

所以最终：

\[
\alpha_j>0
\]

负 score 只是意味着：

> 相对于其他 candidates，匹配程度较低。

不是最终给一个负贡献系数。

---

# 75. Attention Output 是否一定是 Candidate Values 的“平均”？

如果 weights 是 Softmax：

> 是 convex weighted sum。

但很多现代机制会对：

- values；
- outputs；

再做线性 projection。

例如：

\[
head
=
AV
\]

后面：

\[
headW_O
\]

因此最终 attention block 输出并不只是简单的 candidate 原向量平均。

---

# 76. 为什么 Value 要经过 W_V？

如果直接：

\[
V=X
\]

那么 Attention 只能原样混合 input representation。

使用：

\[
V=XW_V
\]

允许模型先把：

> “适合传递的信息”

变换到一个新的 representation space。

所以：

\[
W_V
\]

可以理解为：

> learn what information should be sent when this token is attended to。

再次强调：

> 这只是功能直觉，不是某个维度有固定语义。

---

# 77. 为什么 Query 要经过 W_Q？

原 token representation 可能同时包含很多信息。

但当前匹配需求未必需要全部 feature。

\[
q=xW_Q
\]

允许模型学习：

> 哪些 representation components 对“我在找什么”最有用。

---

# 78. 为什么 Key 要经过 W_K？

同理：

\[
k=xW_K
\]

允许模型学习：

> 哪些 feature 最适合判断“我和别人的 query 是否匹配”。

所以 Q/K 分开的价值是：

> matching 可以在一个专门学习出来的 interaction space 里发生。

---

# 79. 一个非常核心的统一公式

Attention 可以最一般地写成：

\[
\boxed{
\operatorname{Attention}
(q,\{k_j,v_j\})
=
\sum_j
\operatorname{Normalize}
(
\operatorname{score}(q,k_j)
)
v_j
}
\]

这里：

- `score` 决定匹配机制；
- `Normalize` 通常是 Softmax；
- \(v_j\) 是真正被读取的信息。

这个公式比背：

\[
QK^\top
\]

更本质。

因为它同时覆盖：

- Bahdanau additive attention；
- dot-product attention；
- scaled dot-product attention；
- self-attention；
- cross-attention。

---

# 80. Bahdanau 到 Transformer 的真正演化

可以用一条线看：

```text
Fixed context vector
↓
问题：所有 Decoder steps 被迫共用一个摘要

Bahdanau Attention
↓
每个 Decoder step 动态读取 Encoder sequence

Transformer
↓
进一步让同一 sequence 内部的每个位置
也通过 Attention 动态读取其他位置

Multi-Head Attention
↓
在多个 learned subspaces 中并行读取
```

所以 Transformer 不是凭空出现。

它把 Attention 从：

> Encoder–Decoder 间的辅助检索模块

推到了：

> 整个 sequence representation 的核心。

---

# 81. 常见误解一：Attention 就是找最重要的 Token

**不准确。**

Attention 是：

> query-dependent weighted retrieval。

同一个 token 对不同 query 可以完全不同。

---

# 82. 常见误解二：Attention 只选择一个位置

**错误。**

标准 soft attention：

\[
c=\sum_j\alpha_jv_j
\]

通常组合多个 positions。

---

# 83. 常见误解三：Attention Weight 是模型直接学出来的一张固定表

**错误。**

模型学：

\[
W_Q,W_K,W_V
\]

而 attention weights 是：

> 每次 forward 根据当前输入动态计算。

---

# 84. 常见误解四：Dot Product 就是 Attention 的定义

**错误。**

Dot product 是：

> compatibility score 的一种选择。

Bahdanau 使用 learned feed-forward alignment function。

---

# 85. 常见误解五：Softmax 前的 Score 就是 Attention Weight

**错误。**

Score：

\[
e_{ij}
\]

经过 Softmax 后才得到：

\[
\alpha_{ij}
\]

---

# 86. 常见误解六：Attention Weight 就是模型完整解释

**错误。**

它只说明某一 layer/head/query 下的一次信息聚合。

不是完整因果解释。

---

# 87. 常见误解七：Key 和 Value 必须来自不同数据

**错误。**

在 Self-Attention 中，它们通常都来自同一个 input：

\[
X
\]

只是经过不同 learned projections。

---

# 88. 常见误解八：Self-Attention 就是每个 Token 关注自己

**错误。**

“Self”表示：

> Query、Key、Value 来自同一个 sequence。

每个 token 通常可以读取所有允许的位置。

---

# 89. 常见误解九：Attention 自带顺序信息

**错误。**

标准 content-based attention 本身不自动知道 sequence order。

Transformer 需要 position mechanism。

---

# 90. 常见误解十：Attention 没有信息压缩

单个 query 确实把：

\[
n
\]

个 values 汇总为一个 vector。

区别是：

> 每个 query 都能得到不同的动态汇总，而不是整个 sequence 永远共用一个固定 summary。

---

# 91. 常见误解十一：Scaled Dot Product 除 \(\sqrt{d_k}\) 是为了让权重和为 1

**错误。**

权重和为 1：

> 是 Softmax 的作用。

Scaling：

> 是为了控制 logits magnitude，缓解大 \(d_k\) 下 Softmax saturation。

---

# 92. 常见误解十二：Attention = Temporal Ensemble

**错误。**

即使两者最终都包含 weighted sum，

它们的权重来源完全不同。

Attention：

> learned representation-dependent scores。

Temporal Ensemble：

> hand-designed exponential time weights。

---

# 93. 用一个完整 Attention Example 收尾

假设一个 query：

\[
q
\]

面对三个 memory entries：

\[
(k_1,v_1),
(k_2,v_2),
(k_3,v_3)
\]

scaled dot-product scores：

\[
s_1=2
\]

\[
s_2=1
\]

\[
s_3=0
\]

Softmax：

\[
\alpha_1\approx0.665
\]

\[
\alpha_2\approx0.245
\]

\[
\alpha_3\approx0.090
\]

最终：

\[
\boxed{
o
=
0.665v_1
+
0.245v_2
+
0.090v_3
}
\]

整个 Attention 机制其实就是：

```text
Query
↓
和 Keys 比较
↓
[2, 1, 0]
↓
Softmax
↓
[0.665, 0.245, 0.090]
↓
对 Values 加权
↓
Context / Output
```

所有复杂 Transformer Attention，

本质上都建立在这个结构上。

---

# 94. 一句话真正理解 Attention

> **Attention 是一种可学习的动态信息读取机制：模型先根据当前 query 与每个 key 的匹配程度计算 scores，再把这些 scores 归一化成 attention weights，最后用这些 weights 对对应 values 做加权汇总，因此同一份 memory 可以根据不同 query 被读取成不同的 context。**

如果只记一句：

\[
\boxed{
\text{Attention}
=
\text{Match}
+
\text{Normalize}
+
\text{Retrieve}
}
\]

也就是：

```text
匹配
↓
归一化
↓
读取
```

---

# 95. 下一步：为什么 Q、K、V 真的能工作？

现在我们已经知道：

\[
Q
\]

是“当前需求”，

\[
K
\]

用于匹配，

\[
V
\]

用于传递信息。

但还有一个更深的问题：

> **它们本质上都只是一堆向量，为什么 Linear Projection 以后就能自动承担这些角色？**

下一篇：

> [Query / Key / Value：三个向量为什么能实现检索？](./qkv.md)

会重点解释：

- \(W_Q,W_K,W_V\) 到底在学什么；
- 为什么 dot product 可以作为相似度 / compatibility；
- 一个 token 如何同时拥有 query/key/value 三种 representation；
- 为什么 Q 和 K 的维度要兼容；
- \(QK^\top\) 矩阵每个元素到底是什么意思；
- gradient 怎样让 Q/K 学会“匹配”；
- gradient 怎样让 V 学会“传递有用内容”；
- ACT cross-attention 中 Q/K/V 分别来自哪里。

---

## Primary Source 1：Attention / Soft Alignment

Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.  
**Neural Machine Translation by Jointly Learning to Align and Translate.**  
ICLR 2015.  
arXiv:1409.0473.

- Paper: https://arxiv.org/abs/1409.0473
- PDF: https://arxiv.org/pdf/1409.0473

本文关于 Attention 的历史出发点主要依据：

- Section 1 — fixed-length context bottleneck
- Section 3 — Learning to Align and Translate
- Section 3.1 — Decoder: General Description
- Equation (5)：
  \[
  c_i=\sum_j\alpha_{ij}h_j
  \]
- Equation (6)：
  \[
  \alpha_{ij}
  =
  \frac{\exp(e_{ij})}{\sum_k\exp(e_{ik})}
  \]
- alignment score：
  \[
  e_{ij}=a(s_{i-1},h_j)
  \]

原论文明确把这种 weighted sum 解释为：

> 对 possible alignments 下 source annotations 的 expected annotation。

---

## Primary Source 2：Scaled Dot-Product Attention

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,  
Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin.

**Attention Is All You Need.**  
NIPS 2017.  
arXiv:1706.03762.

- Paper: https://arxiv.org/abs/1706.03762
- PDF: https://arxiv.org/pdf/1706.03762

本文关于 Transformer Attention 主要依据：

- Section 3.2 — Attention
- Section 3.2.1 — Scaled Dot-Product Attention
- Section 3.2.2 — Multi-Head Attention
- Section 3.2.3 — Applications of Attention

原论文定义：

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

并明确指出：

- additive attention 使用一个单 hidden-layer feed-forward network 计算 compatibility；
- dot-product attention 可以利用高度优化的 matrix multiplication；
- scale \(1/\sqrt{d_k}\) 用于避免大 \(d_k\) 时 dot products magnitude 过大导致 Softmax 梯度过小。

---

## 与 ACT 的关系

Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.  
**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.**  
RSS 2023.

- https://arxiv.org/abs/2304.13705

ACT 中 Attention 主要出现在：

### CVAE Encoder Self-Attention

```text
[CLS]
joint
action tokens
```

之间的信息融合。

### Policy Encoder Self-Attention

```text
visual tokens
joint token
latent token
```

之间的信息融合。

### Policy Decoder Cross-Attention

\[
Q
\]

来自 action-query / decoder representations，

而：

\[
K,V
\]

来自 observation encoder memory。

---

## 本文知识连接

### 前置知识

- Vector
- Matrix
- [Dot Product](./dot-product.md)
- Weighted Average
- [Softmax](./softmax.md)

### Transformer 主线

- [Transformer](./transformer.md)
- [Query / Key / Value](./qkv.md)
- [Self-Attention](./self-attention.md)
- [Cross-Attention](./cross-attention.md)
- [Multi-Head Attention](./multi-head-attention.md)
- [Causal Mask](./causal-mask.md)
- [Positional Encoding](./positional-encoding.md)

### 历史背景

- Sequence-to-Sequence
- RNN
- LSTM

### Robot Learning

- [ACT Architecture](../robot-learning/act/architecture.md)
- [ACT Complete Data Flow](../robot-learning/act/complete-data-flow.md)
- [Temporal Ensemble](../robot-learning/act/temporal-ensemble.md)

### 下一步

- [Query / Key / Value](./qkv.md)
