---
title: "Query / Key / Value：三个向量为什么能实现检索？"
description: "从线性投影、点积匹配与梯度路径出发，严格理解 Q、K、V 为什么能够分别学习“查询条件、匹配索引与被传递内容”，并连接 Self-Attention、Cross-Attention 与 ACT。"
status: reviewed
pageType: concept
canonical: /deep-learning/qkv
updated: "2026-09-15"
---

# Query / Key / Value：三个向量为什么能实现检索？

在前面的 [Attention](./attention.md) 中，我们把 Attention 压缩成了三步：

$$
\boxed{
\text{Match}
\rightarrow
\text{Normalize}
\rightarrow
\text{Retrieve}
}
$$

在 Transformer 中，这三步写成：

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

最常见的解释是：

```text
Query:
我要找什么？

Key:
我适不适合被找到？

Value:
如果你找到我，我真正给你什么？
```

这个类比很好用。

但真正让人困惑的是下一层：

> **Q、K、V 本质上不就是几个普通向量吗？**

Transformer 并没有给 $Q$ 一个真的“问题字符串”，也没有给 $K$ 一个数据库索引，更没有人为规定 $V$ 里面哪几维是“答案”。

在 Self-Attention 中，它们甚至都来自同一个输入：

$$
X
$$

只是经过三个 Linear：

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

那为什么训练之后：

$$
Q
$$

真的能承担“查询”的作用，

$$
K
$$

真的能承担“匹配”的作用，

而：

$$
V
$$

真的能承担“传递内容”的作用？

答案不是：

> “因为我们把它们分别命名成 Query、Key、Value。”

真正的答案是：

> **它们在计算图中的位置不同。**

$$
W_Q
$$

和：

$$
W_K
$$

影响的是：

> attention 权重如何产生。

而：

$$
W_V
$$

影响的是：

> attention 权重确定以后，到底传递什么表示。

最终任务 loss 通过反向传播，会给这三组参数施加不同的训练压力。

这一篇我们把这件事彻底拆开。

---

## 1. 先忘记 Query、Key、Value 这三个名字

假设我们只有三个 Linear layers：

$$
A=XW_A
$$

$$
B=XW_B
$$

$$
C=XW_C
$$

然后网络规定：

$$
S=AB^\top
$$

$$
P=\operatorname{softmax}(S)
$$

$$
O=PC
$$

如果训练目标要求：

$$
O
$$

最终对任务有用，

那么随着训练进行：

- $W_A$ 会学成适合产生“查询侧匹配表示”的 projection；
- $W_B$ 会学成适合产生“被查询侧匹配表示”的 projection；
- $W_C$ 会学成适合产生“真正被聚合的信息”的 projection。

所以即使我们从来不叫它们 Q、K、V，

它们仍会因为：

> **在公式中承担的位置不同**

而逐渐形成不同功能。

后来我们只是把：

$$
A
$$

命名为：

$$
Q
$$

把：

$$
B
$$

命名为：

$$
K
$$

把：

$$
C
$$

命名为：

$$
V
$$

因为这种命名非常符合它们的计算角色。

---

## 2. Transformer 原论文到底怎么定义 Q、K、V？

《Attention Is All You Need》把 Attention 描述为：

> 将一个 query 与一组 key-value pairs 映射为一个 output。

其中：

- query 和 keys 维度是 $d_k$；
- values 维度是 $d_v$。

Transformer 的 Scaled Dot-Product Attention：

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

原论文进一步说明 Multi-Head Attention 会：

> 对 queries、keys、values 分别使用不同的 learned linear projections。

所以：

$$
W_Q,W_K,W_V
$$

不是事后解释出来的方便符号。

它们本来就是：

> 被任务 loss 共同训练的可学习参数。

---

## 3. 一个 Token 为什么需要三种 Representation？

假设某个 token 当前 representation 是：

$$
x_i\in\mathbb R^{d_{\text{model}}}
$$

这个 representation 里可能混合了很多信息：

- token identity；
- context；
- position；
- semantic features；
- syntactic features；
- modality features。

但 Attention 中，同一个 token 有三种不同“职责”。

---

### 当它作为 Query

它需要表达：

> **我现在需要从别人那里找什么？**

---

### 当它作为 Key

它需要表达：

> **别人用什么条件，可以判断我是否值得被读取？**

---

### 当它作为 Value

它需要表达：

> **如果别人决定读取我，我应该真正传递什么信息？**

---

这三种职责并不要求使用同一套 feature。

所以 Transformer 给它三个 projection：

$$
q_i=x_iW_Q
$$

$$
k_i=x_iW_K
$$

$$
v_i=x_iW_V
$$

让模型自己学：

> 哪些 feature 对匹配有用，哪些 feature 对传输内容有用。

---

## 4. 一个现实类比：论文数据库

假设数据库里有一篇论文。

这篇论文作为一个完整对象包含：

- 标题；
- 作者；
- abstract；
- keywords；
- 正文；
- references；
- figures。

如果有人搜索：

> “robot imitation learning”

用于搜索匹配的可能主要是：

```text
title
keywords
abstract embedding
```

这更像：

$$
Key
$$

真正返回给用户的可能是：

```text
论文正文内容
```

这更像：

$$
Value
$$

而用户搜索框：

```text
robot imitation learning
```

更像：

$$
Query
$$

所以：

> 一个信息对象完全可以用一种 representation 来“被搜索”，用另一种 representation 来“被返回”。

Q/K/V 的分离，本质上就是这种自由度。

---

## 5. 但 Transformer 里的 Query 不是自然语言问题

这一点必须严格区分。

Attention 中的：

$$
q_i
$$

通常只是：

> 一个 learned vector representation。

它不是：

```text
"What does this token need?"
```

这样的文字。

“我要找什么”只是功能解释。

真正数学上：

$$
q_i=x_iW_Q
$$

然后它参与：

$$
q_i^\top k_j
$$

只要这种计算最终帮助降低 loss，

训练就会把：

$$
q_i
$$

塑造成适合匹配的 representation。

---

## 6. 为什么 Linear Projection 就足够？

你可能会觉得：

> 一个 Linear layer 太简单了吧？

例如：

$$
q=xW_Q
$$

只是：

$$
q_r
=
\sum_s x_sW_{Q,sr}
$$

怎么可能产生复杂“查询语义”？

关键在于：

> $x$ 本身往往已经是前面网络层产生的复杂 contextual representation。

尤其在深层 Transformer 中，第 $l$ 层输入：

$$
x_i^{(l)}
$$

已经经历了前面多层：

- Attention；
- FFN；
- Residual；
- nonlinear transformation。

所以 $W_Q$ 并不是从 raw word ID 直接创造所有语义。

它更像：

> 从当前 hidden representation 中，挑出并重新组合最适合当前 attention matching 的 feature。

---

## 7. Linear Layer 到底做了什么？

假设：

$$
x=
[x_1,x_2,x_3,x_4]
$$

一个 projection：

$$
W_Q
$$

可以产生：

$$
q_1
=
0.7x_1
-
0.2x_2
+
1.1x_4
$$

另一维：

$$
q_2
=
-0.3x_1
+
0.9x_3
+
0.4x_4
$$

也就是说：

> 每个 query dimension 都是原 representation features 的 learned linear combination。

$W_K$ 会学另一套组合。

所以同一个输入：

$$
x
$$

可以被重新组织成：

> query space

和：

> key space。

---

## 8. 为什么 Q 和 K 要进入同一个 Dot-Product Space？

Attention score：

$$
q_i^\top k_j
$$

要求：

$$
q_i,k_j\in\mathbb R^{d_k}
$$

因为只有维度一致，

才能做：

$$
\sum_{r=1}^{d_k}q_{ir}k_{jr}
$$

所以 Q/K projection 的一个基本要求是：

> 输出到 compatible matching space。

它们不需要：

$$
W_Q=W_K
$$

也不需要：

$$
q=k
$$

但 dimension 要允许 compatibility operation。

---

## 9. 为什么 Q 和 K 不直接用同一个 Projection？

理论上完全可以设计：

$$
W_Q=W_K
$$

那么：

$$
q_i=x_iW
$$

$$
k_i=x_iW
$$

这会让 score 更像标准 embedding similarity。

但 Transformer 使用不同：

$$
W_Q,W_K
$$

提供更高自由度。

因为“查询侧应该表达什么”和“被查询侧应该表达什么”不一定完全对称。

例如：

> 一个 token 在询问 subject 时需要的 feature，

和：

> 一个 token 表示“我是一个可能的 subject”时需要的 feature，

未必完全相同。

所以 asymmetric learned projections 更灵活。

---

## 10. Dot Product 到底在计算什么？

对于：

$$
q_i=
[q_{i1},\ldots,q_{id_k}]
$$

和：

$$
k_j=
[k_{j1},\ldots,k_{jd_k}]
$$

dot product：

$$
\boxed{
q_i^\top k_j
=
\sum_{r=1}^{d_k}
q_{ir}k_{jr}
}
$$

如果很多 dimension：

- 同号；
- magnitude 都比较大；

它们对 score 有正贡献。

如果相反方向：

> 会降低 score。

所以在 learned matching space 里，

训练可以让：

> 应该匹配的 query/key pair 在很多相关 dimensions 上具有较高 compatibility。

---

## 11. Dot Product 不等于“语义相似度”本身

必须注意。

如果使用预训练 semantic embeddings，

dot product 可能经常被解释成 semantic similarity。

但 Attention 里的：

$$
q_i^\top k_j
$$

更准确是：

> **learned compatibility score。**

它可以学习：

- semantic relation；
- syntactic dependency；
- positional interaction；
- modality correspondence；
- task-specific relation。

不一定是：

> “两个 token 意义越相似，dot product 越大。”

有时一个 verb 会高度关注它的 subject。

它们语义并不相同，

但任务关系非常强。

所以：

$$
\boxed{
\text{compatibility}
\neq
\text{semantic identity}
}
$$

---

## 12. 一个非常重要的例子

句子：

```text
The dog chases the ball.
```

假设当前 query 来自：

```text
chases
```

模型可能希望读取：

```text
dog
```

因为它是 subject。

那么训练可以让：

$$
q_{\text{chases}}^\top
k_{\text{dog}}
$$

比较大。

这不意味着：

> `chases` 和 `dog` 的词义很相似。

而是：

> 在当前 attention head 学到的 matching space 中，这两个 representation 对当前任务具有强关系。

---

## 13. $QK^\top$ 矩阵到底是什么？

假设 sequence 有：

$$
n
$$

个 tokens。

$$
Q\in\mathbb R^{n\times d_k}
$$

$$
K\in\mathbb R^{n\times d_k}
$$

则：

$$
K^\top
\in
\mathbb R^{d_k\times n}
$$

所以：

$$
\boxed{
S=QK^\top
\in
\mathbb R^{n\times n}
}
$$

其中：

$$
\boxed{
S_{ij}
=
q_i^\top k_j
}
$$

这一个元素就表示：

> query position $i$ 对 key position $j$ 的未归一化 compatibility。

---

## 14. 每一行是什么？

第：

$$
i
$$

行：

$$
S_{i,:}
$$

是：

> query $i$ 对 sequence 中所有 keys 的 scores。

例如：

$$
S_{i,:}
=
[1.2,-0.4,3.1,0.8]
$$

说明当前 query 对第三个 key 的 compatibility 最大。

之后沿这一行 Softmax：

$$
A_{i,:}
=
\operatorname{softmax}(S_{i,:})
$$

得到当前 query 的 attention distribution。

---

## 15. 每一列是什么？

第：

$$
j
$$

列：

$$
S_{:,j}
$$

表示：

> 所有不同 queries 对 key $j$ 的 compatibility。

但是 attention normalization 通常按：

> 每一行。

所以：

$$
\sum_jA_{ij}=1
$$

而不是要求：

$$
\sum_iA_{ij}=1
$$

---

## 16. 为什么要 Softmax？

Raw score：

$$
q_i^\top k_j
$$

可以是任意实数。

例如：

$$
[3.0,-1.0,0.4]
$$

不能直接当作稳定的 mixture weights。

Softmax：

$$
\alpha_{ij}
=
\frac{
e^{s_{ij}}
}{
\sum_r e^{s_{ir}}
}
$$

得到：

$$
\alpha_{ij}>0
$$

以及：

$$
\sum_j\alpha_{ij}=1
$$

所以：

> 对于当前 query，可以得到一组标准化读取权重。

---

## 17. 然后为什么乘 V？

得到：

$$
A
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)
$$

之后：

$$
O=AV
$$

对于单个 query $i$：

$$
\boxed{
o_i
=
\sum_j
A_{ij}v_j
}
$$

所以：

> $Q$ 和 $K$ 决定“谁贡献多少”。

而：

> $V$ 决定“贡献的内容是什么”。

这就是三者功能分离最准确的数学描述。

---

## 18. 如果没有 V Projection 会怎样？

可以直接：

$$
V=X
$$

那么输出：

$$
O=AX
$$

仍然是一种完全合法的 attention。

但加入：

$$
V=XW_V
$$

让模型能够学习：

> token 被别人读取时，应该传递哪一种 transformed representation。

也就是说：

$$
W_V
$$

不参与：

> 谁应该被选择。

它主要参与：

> 被选择以后传什么。

这是非常重要的角色区别。

---

## 19. 为什么不能用 K 同时当 V？

也可以设计：

$$
V=K
$$

但这会强迫：

> “用于匹配的 representation”

和：

> “用于传递的 representation”

必须相同。

Transformer 不施加这种约束。

独立：

$$
W_K,W_V
$$

使模型能够：

> 用一组 features 做索引，另一组 features 做内容。

这通常更灵活。

---

## 20. 现在进入最关键的问题：为什么这些角色真的能被学出来？

我们假设最终 loss：

$$
L
$$

依赖 Attention output：

$$
O
$$

而：

$$
O=AV
$$

其中：

$$
A=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)
$$

所以整个 computation graph：

```text
X
├── W_Q → Q ──┐
│             │
├── W_K → K ──┤→ scores → Softmax → A ──┐
│                                      │
└── W_V → V ────────────────────────────┤
                                       ▼
                                       O
                                       │
                                       ▼
                                       L
```

loss gradient 会沿不同路径回传。

这正是 Q/K/V 角色出现的根本原因。

---

## 21. W_V 收到什么样的训练压力？

先暂时把 attention weights：

$$
A
$$

当作固定。

那么：

$$
O=AV
$$

如果最终 output 不好，

gradient 会告诉：

$$
V
$$

应该如何改变，

让被 attention 读取后的内容更有助于任务。

而：

$$
V=XW_V
$$

所以 gradient 继续更新：

$$
W_V
$$

因此长期训练后：

$$
W_V
$$

会倾向学出：

> **“当某个 token 被读取时，什么信息最值得传递给其他 token？”**

这就是为什么 Value 成为：

> content representation。

---

## 22. W_Q 收到什么样的训练压力？

$$
W_Q
$$

不会只影响被传递的内容。

它首先改变：

$$
Q
$$

然后改变：

$$
QK^\top
$$

再改变：

$$
A
$$

最终决定：

> 当前 query 应该从哪里读取。

所以如果模型原本：

> 看错了位置，

loss gradient 可以通过：

$$
L
\rightarrow
A
\rightarrow
Q
\rightarrow
W_Q
$$

修改 query projection。

长期来看：

$$
W_Q
$$

会学出：

> **怎样从当前 token representation 中提取适合表达“我的检索需求”的 feature。**

---

## 23. W_K 收到什么样的训练压力？

类似地：

$$
W_K
$$

改变：

$$
K
$$

再改变：

$$
QK^\top
$$

因此它影响：

> 一个 token 在面对不同 queries 时有多容易匹配。

如果某个 token 应该被某类 queries 读取，

gradient 可以推动：

$$
k_j
$$

在 matching space 中移动到更合适的位置。

长期训练后：

$$
W_K
$$

会学出：

> **怎样从当前 token representation 中提取适合被匹配的索引特征。**

---

## 24. 所以角色不是“名字教会”的，而是“梯度路径塑造”的

这句话非常重要。

$$
W_Q
$$

之所以成为 Query projection，

不是因为代码变量叫：

```python
query
```

而是因为它在公式里：

> 位于 attention score 的查询侧。

$$
W_K
$$

之所以成为 Key projection，

是因为它：

> 位于 attention score 的候选侧。

$$
W_V
$$

之所以成为 Value projection，

是因为它：

> 位于 weighted retrieval 的内容侧。

所以：

$$
\boxed{
\text{Role}
\leftarrow
\text{position in computation graph}
+
\text{task gradient}
}
$$

---

## 25. 一个很小的训练直觉例子

假设 query token：

```text
it
```

应该读取：

```text
animal
```

而不是：

```text
street
```

但模型当前 scores：

$$
s_{\text{it,animal}}=0.4
$$

$$
s_{\text{it,street}}=2.1
$$

于是 attention 错误地更多读取：

```text
street
```

最后 prediction loss 变大。

反向传播会尝试调整：

$$
q_{\text{it}}
$$

和：

$$
k_{\text{animal}},
k_{\text{street}}
$$

使未来：

$$
q_{\text{it}}^\top k_{\text{animal}}
$$

提高，

而：

$$
q_{\text{it}}^\top k_{\text{street}}
$$

相对降低。

这就是：

> Q/K matching structure 被任务监督间接训练。

---

## 26. Value 在同一个例子里怎样学？

即使模型已经正确关注：

```text
animal
```

如果：

$$
v_{\text{animal}}
$$

传出去的信息对后续预测没有帮助，

模型仍然可能表现不好。

这时 gradient 会调整：

$$
W_V
$$

让 `animal` 的 value representation 更适合向当前 query 传递有用信息。

所以：

> “看对地方”

和：

> “从那里读到有用内容”

是两个不同问题。

Q/K 主要负责前者，

V 主要负责后者。

---

## 27. 为什么 Q 和 K 都需要训练，不能只训练 Query？

假设 keys 固定。

Query 当然可以学习：

> 去匹配现有 key geometry。

但如果 keys 也可训练，

则双方可以一起形成：

> 更适合任务的 matching coordinate system。

可以类比两种语言的人：

```text
Query 侧学会怎样提问
Key 侧也学会怎样贴标签
```

二者共同适配，

比只有一侧适配更灵活。

---

## 28. Q/K Space 可以怎样理解？

可以把：

$$
\mathbb R^{d_k}
$$

想象成一个模型自己学习出来的：

> **matching space**

在这个空间里，

不是要求：

> “意思相同的 token 靠近。”

而是要求：

> “在这个 head 的当前任务关系中，应该发生 attention 的 query-key pair 具有高 compatibility。”

这比“相似度空间”更准确。

---

## 29. V Space 又是什么？

$$
\mathbb R^{d_v}
$$

可以看成：

> **message space / transmitted information space**

一个 token 被 attention 后，

它通过：

$$
v_j
$$

向其他位置传递信息。

所以从 message-passing 视角：

$$
K
$$

更像：

> route / address representation。

$$
V
$$

更像：

> message payload。

---

## 30. 一个非常好用的类比：网络请求

可以类比：

```text
Query:
请求条件

Key:
路由匹配规则 / 地址索引

Value:
真正返回的数据 payload
```

请求：

$$
q
$$

与不同 route keys：

$$
k_j
$$

比较。

匹配得越高，

对应 payload：

$$
v_j
$$

贡献越大。

但仍然要记住：

> 这只是功能类比。

数学本体仍然是 vectors + dot products + weighted sums。

---

## 31. 一个 Token 同时有 Q、K、V，会不会矛盾？

不会。

一个 token 可以同时：

> 向别人读取信息，

也可以：

> 被别人读取。

所以它自然同时拥有：

#### Query role

“我现在想从别人那里得到什么？”

#### Key role

“别人根据什么判断是否应该读我？”

#### Value role

“别人如果读我，我真正提供什么？”

在 Self-Attention 中，每个 token 都同时承担这三种角色。

---

## 32. Self-Attention 中的一次完整交互

假设 sequence：

$$
x_1,x_2,x_3
$$

每个产生：

$$
q_i,k_i,v_i
$$

对于 token 2：

$$
q_2
$$

与：

$$
k_1,k_2,k_3
$$

比较：

$$
s_{21}=q_2^\top k_1
$$

$$
s_{22}=q_2^\top k_2
$$

$$
s_{23}=q_2^\top k_3
$$

Softmax 得：

$$
[\alpha_{21},\alpha_{22},\alpha_{23}]
$$

然后：

$$
o_2
=
\alpha_{21}v_1
+
\alpha_{22}v_2
+
\alpha_{23}v_3
$$

这就是 token 2 的一次 context update。

---

## 33. 为什么 Value 不参与 Attention Score？

标准 Transformer 中：

$$
score_{ij}
$$

只由：

$$
q_i,k_j
$$

决定。

这意味着：

> “应不应该读取某个位置”

和：

> “那个位置具体传递什么”

被结构上分离。

这样可以让模型专门学习匹配空间和内容空间。

当然，其他 attention 变体理论上可以设计不同 scoring mechanism。

这里讲的是原始 Transformer。

---

## 34. 为什么 Query 不参与最终 Weighted Sum？

最终：

$$
o_i=\sum_j\alpha_{ij}v_j
$$

query 自己没有直接被加进去。

但 Transformer block 后面有：

> residual connection。

也就是说通常：

$$
x_i
$$

会通过 residual 保留下来。

所以 attention sublayer 可以理解为：

> 计算“我应该从别人那里新增什么信息”。

然后：

$$
x_i+\text{AttentionOutput}_i
$$

形成更新 representation。

因此不需要在 value weighted sum 中再次显式加入 query。

---

## 35. 为什么 Transformer 要用 Residual？

这与 QKV 关系很密切。

Attention output：

$$
o_i
$$

更像：

> 从 context 中读取到的信息。

原 representation：

$$
x_i
$$

则保留：

> 当前 token 原有信息。

Residual：

$$
x_i+o_i
$$

让模型同时保留：

- 自己原本是什么；
- 从别人那里读到了什么。

这是一种很自然的信息更新形式。

---

## 36. Q/K 的 Dot Product 为什么不使用 Euclidean Distance？

可以设计：

$$
-\|q-k\|^2
$$

作为 compatibility。

Attention 并不理论上要求必须 dot product。

但 dot product：

1. 可以一次矩阵乘法算所有 pair；
2. 与 learned projections 配合灵活；
3. GPU 高效；
4. Transformer 原论文实验证明有效。

所以 Scaled Dot-Product Attention 采用：

$$
q^\top k
$$

主要是一个：

> 很有效且高效的参数化选择。

不是“Attention 唯一可能的数学形式”。

---

## 37. Dot Product 为什么和向量方向有关？

回忆：

$$
q^\top k
=
\|q\|
\|k\|
\cos\theta
$$

所以 score 同时受：

- 向量 magnitude；
- 夹角；

影响。

如果方向接近：

$$
\cos\theta
$$

较大。

如果相反：

$$
\cos\theta<0
$$

score 可以为负。

因此 learned Q/K space 可以通过：

- 方向；
- magnitude；

共同编码 compatibility。

---

## 38. Attention 为什么不用 Cosine Similarity？

Cosine similarity 会归一化：

$$
\frac{q^\top k}{\|q\|\|k\|}
$$

从而移除 magnitude 信息。

Transformer 保留 dot-product magnitude，

让网络可以利用 vector norm 作为 matching 的一部分。

然后用：

$$
1/\sqrt{d_k}
$$

控制 dimension 导致的整体尺度。

这是一种设计选择。

---

## 39. 为什么除以 $\sqrt{d_k}$？

如果：

$$
q_r,k_r
$$

独立，均值 0，方差 1，

则：

$$
q^\top k
=
\sum_{r=1}^{d_k}q_rk_r
$$

有：

$$
\operatorname{Var}(q^\top k)=d_k
$$

所以：

$$
d_k
$$

越大，

score magnitude 往往越大。

Softmax 会更容易饱和。

因此：

$$
\boxed{
\frac{q^\top k}{\sqrt{d_k}}
}
$$

把标准差大致从：

$$
\sqrt{d_k}
$$

重新压到：

$$
O(1)
$$

尺度。

---

## 40. 一个数字例子

假设：

$$
d_k=64
$$

那么：

$$
\sqrt{d_k}=8
$$

raw dot products：

$$
[16,8,0]
$$

不 scale：

$$
\operatorname{softmax}([16,8,0])
$$

会非常尖锐。

scale 后：

$$
[2,1,0]
$$

Softmax 大约：

$$
[0.665,0.245,0.090]
$$

仍然偏向第一个，

但不会过度饱和。

这就是 scale 的直觉。

---

## 41. Q/K/V Projection 有没有 Bias？

从抽象公式上常写：

$$
Q=XW_Q
$$

省略 bias。

实际 framework 实现中是否带 bias：

> 是具体 implementation choice。

Transformer 的理论核心不依赖：

$$
b_Q,b_K,b_V
$$

是否存在。

所以学习原理时重点应该放在：

> 独立 learned affine/linear projections 的角色，

而不是把某个库的 bias 配置当成定义。

---

## 42. Multi-Head 为什么要有不同的 W_Q/W_K/W_V？

单头只有一个 matching space：

$$
\mathbb R^{d_k}
$$

Multi-Head 给第 $h$ 个 head 自己的：

$$
W_Q^{(h)}
$$

$$
W_K^{(h)}
$$

$$
W_V^{(h)}
$$

于是：

> 不同 heads 可以学习不同的匹配规则和不同 message representations。

一个 head 可能更偏：

- local relation；

另一个可能更偏：

- long-range relation；

另一个：

- positional pattern。

但这些具体语义不是架构保证的。

---

## 43. 为什么每个 Head 维度通常更小？

原始 Transformer：

$$
d_{\text{model}}=512
$$

$$
h=8
$$

每个 head：

$$
d_k=d_v=64
$$

所以 8 个 head concat：

$$
8\times64=512
$$

这样总 representation width 不膨胀。

而且每个 head 在不同 64-D subspace 中做匹配。

---

## 44. QKV 的参数量大概是多少？

单个 attention layer，如果：

$$
d_{\text{model}}=512
$$

且 Q/K/V 输出也总宽度 512，

那么三组 projection 大致各有：

$$
512\times512
$$

参数。

总计：

$$
3\times512^2
=
786432
$$

还没算：

$$
W_O
$$

这说明：

> QKV 不是三个小规则。

它们本身包含大量可学习参数，

可以学习复杂 projection。

---

## 45. 为什么 Attention Weights 不是直接训练参数？

你可能想：

> 为什么不直接存一个 $n\times n$ 的 matrix：

$$
A
$$

然后训练它？

因为 sequence 内容会变化。

如果：

$$
A
$$

固定，

模型只能学：

> position 3 永远看 position 7。

而 Q/K 机制学的是：

$$
A(X)
$$

即：

> 根据当前 input content 动态产生连接权重。

所以 Q/K 是一种：

> **生成 attention matrix 的参数化机制。**

---

## 46. 可以把 Q/K 看成一个动态邻接矩阵生成器

Self-Attention：

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

虽然通常不会这样合并写，

但它揭示：

> 输入 $X$ 一变，attention connectivity 就会变。

所以 Q/K projections 让网络每个 sample 都动态构造：

> token-to-token routing pattern。

---

## 47. Value 则像 Message Function

动态连接确定后：

$$
A_{ij}
$$

表示：

> 从 node/token $j$ 向 query $i$ 传多少信息。

真正 message：

$$
v_j=x_jW_V
$$

于是：

$$
o_i
=
\sum_j
A_{ij}v_j
$$

这和 message passing 的结构非常相似：

```text
edge weight
×
message
↓
aggregate
```

所以：

$$
Q/K
$$

更像学习 routing，

$$
V
$$

更像学习 message content。

---

## 48. 为什么一个 Token 的 Key 对所有 Query 都一样？

在一次单头 attention forward 中，

token $j$ 的：

$$
k_j
$$

固定。

不同 queries：

$$
q_1,q_2,\ldots
$$

分别和它匹配。

这就像：

> 一个数据库条目有一个当前索引 representation，

不同搜索请求会得到不同 matching score。

所以同一个 key：

> 对一个 query 很匹配，

对另一个 query 可能完全不匹配。

---

## 49. 为什么一个 Token 的 Value 对不同 Query 也一样？

在标准 attention 中，

对于同一个 head 和同一次 forward：

$$
v_j
$$

不会针对每个 query 改变。

不同 query 主要通过：

$$
\alpha_{ij}
$$

决定读取多少。

所以：

> Value 是“这个位置能提供什么”。

> Query 决定“我要读取多少”。

当然多层网络里，下一层的 token representation 已经会因为上一层 query-specific aggregation 改变。

---

## 50. Attention Output 为什么是 Contextual Representation？

假设：

$$
x_i
$$

本来主要描述 token $i$。

经过：

$$
o_i
=
\sum_j
\alpha_{ij}v_j
$$

它包含了来自其他 positions 的信息。

所以新的 representation 不再只由：

$$
x_i
$$

决定，

还依赖整个 context：

$$
X
$$

因此叫：

> **contextual representation**

这就是为什么同一个词在不同句子里 hidden representation 可以不同。

---

## 51. Self-Attention 和 Cross-Attention 的 QKV 来源

这是必须牢牢记住的。

---

### Self-Attention

同一个 source：

$$
X
$$

产生：

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

所以：

> sequence 内部自己查询自己。

---

### Cross-Attention

假设 decoder states：

$$
H
$$

encoder memory：

$$
M
$$

那么：

$$
Q=HW_Q
$$

$$
K=MW_K
$$

$$
V=MW_V
$$

所以：

> decoder 用自己的 query 去读取 encoder memory。

---

## 52. 为什么 Cross-Attention 里 K 和 V 通常来自同一个 Memory？

因为 encoder memory 中的每个 position 是一个 candidate information source。

对于这个 candidate：

- Key 用来判断是否相关；
- Value 用来真正提供内容。

所以自然成对：

$$
(k_j,v_j)
$$

来自同一个 memory position 的不同 projections。

---

## 53. ACT 中 Policy Decoder 的 QKV 到底是什么？

回到 ACT。

Policy encoder 已经把：

```text
visual tokens
joint token
z token
```

融合成 memory：

$$
M
$$

Transformer decoder 有：

$$
k
$$

个 action-query positions。

Cross-attention 时：

$$
\boxed{
Q
=
\text{decoder action representations}
}
$$

而：

$$
\boxed{
K,V
=
\text{observation encoder memory}
}
$$

所以第 $i$ 个 future action slot 在问：

> **为了预测 action chunk 中第 $i$ 个动作，我应该从当前 observation memory 的哪些位置读取信息？**

---

## 54. ACT 中一个 Action Query 可能关注什么？

假设未来第 10 个 action 对应：

> gripper 即将接近目标物。

这个 query 可能通过 learned Q/K matching：

- 对 wrist camera 某些 visual tokens 权重较高；
- 对 joint token 有一定权重；
- 对 top-view 某些 region 也有权重。

然后通过对应 Values：

> 聚合这些 observation features。

但必须强调：

> 这是架构允许的行为，不是未经可视化验证就能断言的实际 learned pattern。

---

## 55. ACT 的 Action Query Embedding 本身就是 Attention Q 吗？

不完全是。

ACT decoder 有：

> action query embedding / query position。

但在 MultiheadAttention 内部，

还会经过：

$$
W_Q
$$

projection，

才得到真正 attention 公式里的：

$$
Q
$$

所以：

```text
action query embedding
```

是：

> query-side input representation。

而：

```text
attention Q matrix
```

是：

> 经过 learned query projection 后的 matching representation。

二者不能完全画等号。

---

## 56. 同理，Encoder Memory 本身也不是 K/V

Memory：

$$
M
$$

进入 cross-attention 后，

会分别经过：

$$
W_K
$$

和：

$$
W_V
$$

形成：

$$
K,V
$$

所以：

```text
memory token
```

不等于：

```text
key
```

也不等于：

```text
value
```

它是产生 K/V 的 source representation。

---

## 57. 为什么这个区别很重要？

如果你把：

$$
memory=key=value
$$

直接当成严格事实，

就会错过 Transformer 的关键自由度：

> 同一个 memory representation 可以被重新投影成一种用于匹配的形式，以及另一种用于传递的形式。

正是：

$$
W_K,W_V
$$

让这种角色分离发生。

---

## 58. QKV 是不是一定要由 Linear Layer 产生？

在标准 Transformer：

> 是 learned linear projections。

但更广义的 Attention 机制不一定。

例如某些模型可以：

- 直接使用 input vectors；
- 使用 convolution；
- 使用 nonlinear network；
- 使用不同 modality encoder。

所以 Q/K/V 是：

> 功能角色，

而不是“必须由 `nn.Linear` 生成”的数学定义。

这里讨论的是 Transformer 经典实现。

---

## 59. 一个非常小的数值例子

假设有两个 candidate tokens。

Query：

$$
q=
[1,0]
$$

Keys：

$$
k_1=
[1,0]
$$

$$
k_2=
[0,1]
$$

则：

$$
q^\top k_1=1
$$

$$
q^\top k_2=0
$$

Softmax：

$$
\alpha
=
\operatorname{softmax}([1,0])
$$

大约：

$$
[0.731,0.269]
$$

如果 Values：

$$
v_1=
[10,0]
$$

$$
v_2=
[0,20]
$$

则：

$$
o
=
0.731[10,0]
+
0.269[0,20]
$$

得到：

$$
\boxed{
o
\approx
[7.31,5.38]
}
$$

你可以清楚看到：

> Q/K 决定比例。

> V 决定混进来的具体内容。

---

## 60. 如果只改变 Value，会发生什么？

保持：

$$
q,k_1,k_2
$$

不变，

所以：

$$
\alpha=[0.731,0.269]
$$

也不变。

但把：

$$
v_1
$$

改成：

$$
[100,100]
$$

那么输出会完全改变。

这说明：

> Value 不负责“被选多少”，但负责“被选之后造成什么结果”。

---

## 61. 如果只改变 Key，会发生什么？

保持 Values 不变。

把：

$$
k_2
$$

改成：

$$
[2,0]
$$

那么：

$$
q^\top k_2=2
$$

现在第二个 candidate 反而匹配更高。

attention weights 改变，

即使：

$$
v_1,v_2
$$

完全没变。

所以：

> Key 控制匹配结构。

---

## 62. 如果只改变 Query，会发生什么？

Query 从：

$$
[1,0]
$$

改成：

$$
[0,1]
$$

原来：

$$
k_1=[1,0]
$$

$$
k_2=[0,1]
$$

于是注意力会更偏：

$$
k_2
$$

同一组 memory：

> 因为 query 不同，读取结果就不同。

这正是 Attention 最核心的 query-conditioned retrieval。

---

## 63. 为什么同一个 Token 可以对不同 Head 产生不同 Q/K/V？

因为每个 head 都有自己的：

$$
W_Q^{(h)},W_K^{(h)},W_V^{(h)}
$$

所以同一个：

$$
x_i
$$

在 head 1：

$$
q_i^{(1)},k_i^{(1)},v_i^{(1)}
$$

和 head 2：

$$
q_i^{(2)},k_i^{(2)},v_i^{(2)}
$$

完全可以不同。

因此 Multi-Head Attention 不是：

> 同一套 attention 做 8 次。

而是：

> 8 套不同 learned projection spaces 并行工作。

---

## 64. Gradient 怎么穿过 Softmax？

如果某个 attention weight：

$$
\alpha_j
$$

对 loss 的影响不好，

gradient 会穿过 Softmax 回到 score：

$$
s_j
$$

Softmax 的 Jacobian 具有：

$$
\frac{\partial \alpha_i}{\partial s_j}
=
\alpha_i(\delta_{ij}-\alpha_j)
$$

所以一个 score 改变时：

> 不只影响自己对应 weight，也影响同一 row 的其他 weights。

因为这些 weights 必须共同归一化到：

$$
1
$$

这意味着 Q/K 学到的是：

> 相对竞争关系。

---

## 65. Attention Weight 是相对的，不是绝对的

假设当前 scores：

$$
[3,2,1]
$$

如果全部加：

$$
100
$$

变成：

$$
[103,102,101]
$$

Softmax 完全相同。

所以：

$$
\alpha
$$

主要由：

> score differences

决定。

这说明 key 的意义不是：

> “我的 score 绝对越大越重要。”

而是：

> 相对于当前 query 的其他候选，我有多匹配。

---

## 66. 为什么 Attention 是竞争式读取？

因为 Softmax normalization：

$$
\sum_j\alpha_j=1
$$

当某个 key 权重增加，

通常意味着其他 key 的相对份额会降低。

所以一个 query 的候选 values 在：

> 有限总 attention mass

下进行软竞争。

这和 sigmoid 独立 gating 不同。

---

## 67. Q/K 的 Scale 为什么会影响 Gradient？

如果：

$$
q^\top k
$$

magnitude 太大，

Softmax 很容易：

$$
[0.99999,0.00001,\ldots]
$$

进入非常饱和状态。

这时：

$$
\alpha(1-\alpha)
$$

会很小，

score gradient 也会很弱。

所以：

$$
1/\sqrt{d_k}
$$

并不只是为了“数字好看”。

它直接关系到：

> Q/K matching mechanism 是否容易训练。

---

## 68. Q/K/V 有“标准答案”吗？

没有。

神经网络可能存在很多不同参数组合：

$$
W_Q,W_K,W_V
$$

都能产生相似最终行为。

而且 representation 通常不可唯一解释。

所以不能问：

> “第 17 维 Query 到底代表什么？”

然后期待一定存在明确答案。

Q/K/V 是：

> distributed learned representations。

重要的是整体几何和计算效果。

---

## 69. 可以手动查看 Q、K、V 吗？

当然可以。

运行模型时可以 hook：

- Q tensor；
- K tensor；
- V tensor；
- attention weights。

但看到数值：

```text
0.13
-0.42
1.08
...
```

通常无法直接用人类语义解释。

更常见分析方法是：

- attention maps；
- similarity matrices；
- probing；
- ablations；
- representation visualization。

所以“能看到 tensor”不等于“能直接读懂每一维语义”。

---

## 70. Q/K/V 和 Embedding 有什么区别？

Embedding / hidden representation：

$$
x_i
$$

是：

> 当前 layer 输入时 token 的通用 representation。

Q/K/V：

$$
q_i,k_i,v_i
$$

是：

> 为当前 attention operation 临时投影出的任务角色 representation。

attention 完成后，

这些 Q/K/V 通常不会直接作为下一层 token hidden state。

下一层接收到的是：

> attention output + residual + FFN 后的新 hidden representation。

然后下一层会重新计算新的 Q/K/V。

---

## 71. 所以每一层的 QKV 都不同

第 1 层：

$$
Q^{(1)},K^{(1)},V^{(1)}
$$

来自：

$$
X^{(1)}
$$

第 2 层输入已经变成：

$$
X^{(2)}
$$

所以又计算：

$$
Q^{(2)},K^{(2)},V^{(2)}
$$

即使参数结构相似，

每层参数通常也独立。

所以模型可以逐层形成不同类型的信息交互。

---

## 72. 为什么越深层可能学不同关系？

早期 layer representation 可能更接近：

- local pattern；
- surface features。

经过多层 context mixing 后，

高层 representation 可以包含更多：

- semantic；
- task-level；
- global structure。

因此不同层 Q/K matching 可能建立在不同 representation level 上。

但具体“第几层一定做什么”并没有固定理论保证。

---

## 73. QKV 和 Position 信息是什么关系？

如果 input representation：

$$
x_i
$$

已经加了：

$$
PE_i
$$

那么：

$$
q_i=(e_i+PE_i)W_Q
$$

$$
k_i=(e_i+PE_i)W_K
$$

所以 Q/K 可以利用：

> token content + position

共同决定 compatibility。

这就是原始 Transformer positional encoding 能影响 attention pattern 的途径之一。

---

## 74. 没有 Position 时 Q/K 会怎样？

如果两个 token 内容完全相同，

而输入 representation 又完全相同，

则它们产生：

$$
q,k,v
$$

也相同。

Self-Attention 本身无法知道：

> 哪个在前、哪个在后。

所以 sequence model 必须提供：

> positional information。

QKV 负责 matching，

不是天然的顺序编码机制。

---

## 75. Mask 会怎样影响 QK？

Causal mask 并不会改变：

$$
Q,K
$$

本身。

它通常作用在 score matrix：

$$
S=
\frac{QK^\top}{\sqrt{d_k}}
$$

上。

禁止位置加入：

$$
-\infty
$$

然后 Softmax：

$$
A=\operatorname{softmax}(S+M)
$$

所以被 mask 的 key：

$$
A_{ij}=0
$$

即使：

$$
q_i^\top k_j
$$

本来很高。

---

## 76. Padding Mask 同理

PAD token 也许仍然经过网络得到某个：

$$
k_j,v_j
$$

但在 attention score 阶段被 mask，

所以其他 queries 不会真正读取它。

因此 mask 改变的是：

> allowed routing graph。

---

## 77. Q/K/V 为什么不能简单说成“问题、关键词、答案”？

这个类比容易让人误解：

> Value 是最终答案。

其实：

$$
v_j
$$

只是：

> 某个 memory position 提供的一段 representation。

真正最终 prediction 还要经过：

- attention aggregation；
- residual；
- FFN；
- many layers；
- output head。

所以更准确的词：

```text
Q:
retrieval request representation

K:
matching / addressing representation

V:
message / content representation
```

比：

> 问题 / 关键词 / 答案

更严谨。

---

## 78. 一个关键问题：Q 和 K 为什么不需要人类标注？

因为最终任务本身提供 supervision。

例如翻译：

$$
L=-\log p(y_{\text{correct}})
$$

如果某次 attention routing 导致翻译更好，

gradient 会强化产生该 routing 的 Q/K projections。

反之会修正。

所以 alignment / retrieval structure 可以：

> 从最终 task objective 中间接学出来。

这也是 Attention powerful 的原因之一。

---

## 79. 但 Attention 并不保证学出“人类觉得合理”的对应关系

模型只需要：

> 降低任务 loss。

如果存在一种内部 attention pattern：

- 人类不容易解释；
- 但能产生正确输出；

优化完全可能选择它。

所以：

> learned Q/K geometry 是 task-effective representation，

不一定等于 human-interpretable ontology。

---

## 80. QKV 和数据库检索最大的不同

数据库检索常常：

> Key 是人工设计好的索引。

Attention 中：

$$
K=XW_K
$$

Key 本身是学习出来的。

而 Query：

$$
Q=XW_Q
$$

也学习。

甚至 Value：

$$
V=XW_V
$$

也学习。

所以 Attention 更像：

> **连“怎样提问、怎样建立索引、怎样返回内容”都由 end-to-end learning 共同优化的软检索系统。**

---

## 81. 为什么 QKV 特别适合机器人多模态输入？

假设 ACT memory 同时有：

- image tokens；
- joint token；
- latent token。

它们都被映射到：

$$
512
$$

维 hidden space。

Cross-attention 可以进一步把这些 memory representations 投影成：

$$
K,V
$$

Action query projection 成：

$$
Q
$$

于是一个未来 action slot 可以根据当前需求动态判断：

> 哪些 image region、joint state、latent context 更相关。

这种 routing 并不要求所有 token 原始模态相同。

---

## 82. 一个 ACT 小例子

假设：

$$
q_{action}^{(20)}
$$

表示 chunk 第 20 个 future action slot。

observation memory 有：

```text
m₁ = top-camera token
m₂ = wrist-camera token
m₃ = joint token
m₄ = latent token
...
```

Cross-attention：

$$
Q=q_{action}^{(20)}W_Q
$$

每个 memory：

$$
k_j=m_jW_K
$$

$$
v_j=m_jW_V
$$

然后：

$$
\alpha_j
=
\operatorname{softmax}
\left(
\frac{Qk_j}{\sqrt{d_k}}
\right)
$$

最后：

$$
o_{20}
=
\sum_j\alpha_jv_j
$$

这个：

$$
o_{20}
$$

再经过 decoder 后续网络，

最终投影成：

$$
14
$$

维 target joint position。

---

## 83. 为什么未来不同 Action Slot 会有不同 Attention？

因为它们的 decoder representations / query positions 不同。

所以：

$$
q_{action}^{(0)}
\neq
q_{action}^{(20)}
$$

一般情况下，

它们经过：

$$
W_Q
$$

产生不同 Q。

因此即使 K/V memory 完全相同，

attention distribution 也可以不同。

这就是：

> 同一 observation memory 被不同 future action positions 以不同方式读取。

---

## 84. Self-Attention 中为什么 Token 会“互相影响”？

因为：

$$
o_i
=
\sum_j\alpha_{ij}v_j
$$

所以：

$$
x_j
$$

不仅决定自己的：

$$
v_j
$$

还可能通过被：

$$
i
$$

attention 到，

影响：

$$
o_i
$$

因此一层 self-attention 就建立了：

> all-to-all possible information flow。

---

## 85. 一个 Token 会不会影响自己？

会。

在没有 mask self-position 的普通 self-attention 中：

$$
j=i
$$

也是合法 key/value。

所以：

$$
\alpha_{ii}
$$

可能非零。

这意味着 token 可以：

> 读取自己的 value。

再加 residual，

自己的信息通常有多条保留路径。

---

## 86. 为什么 Attention 不是简单“复制最相关 Token”？

因为即使：

$$
\alpha_{ij}
$$

最大，

其他：

$$
\alpha
$$

仍可非零。

而且：

$$
v_j
$$

是 learned projection，

不是原 token representation 本身。

最后还有：

$$
W_O
$$

等 transformation。

所以 Attention 是：

> learned soft information composition。

---

## 87. Q/K/V 和 Temporal Ensemble 再区分一次

ACT Temporal Ensemble：

$$
a_t
=
\sum_i
\alpha_i
\hat a_t^{(i)}
$$

也有 weights。

但：

$$
\alpha_i
$$

来自：

$$
e^{-mi}
$$

只由 prediction age 决定。

Attention：

$$
\alpha_{ij}
$$

来自：

$$
q_i^\top k_j
$$

由 learned representation content 决定。

所以：

$$
\boxed{
\text{Attention}
=
\text{learned content-dependent routing}
}
$$

$$
\boxed{
\text{Temporal Ensemble}
=
\text{hand-designed time-dependent averaging}
}
$$

---

## 88. 常见误解一：Q 就是输入 Token 本身

**不准确。**

标准 Transformer：

$$
Q=XW_Q
$$

Q 是：

> input hidden representation 的 learned projection。

---

## 89. 常见误解二：K 是关键词

只是类比。

严格来说：

$$
K=XW_K
$$

是：

> matching / addressing representation。

它可以编码非常抽象、task-specific 的关系。

---

## 90. 常见误解三：V 是最终答案

**错误。**

Value 是：

> 被 attention 聚合的 message representation。

最终预测还要经过后续网络。

---

## 91. 常见误解四：Q、K、V 三个 Linear Layer 各自有专门标签监督

**没有。**

它们都由最终 task loss 端到端训练。

---

## 92. 常见误解五：Q/K 只是在学语义相似度

**错误。**

它们学的是：

> task-specific compatibility。

强关系不一定意味着语义相似。

---

## 93. 常见误解六：Q 和 K 必须相同

**错误。**

它们通常用不同 learned projections。

只需要进入可比较的 matching dimension。

---

## 94. 常见误解七：Key 决定传递什么内容

主要不是。

Key 主要影响：

> attention score。

真正被 weighted sum 的是：

$$
Value
$$

---

## 95. 常见误解八：Value 影响 Attention Weight

在标准 scaled dot-product attention 中：

> 不直接影响。

权重由：

$$
QK^\top
$$

决定。

---

## 96. 常见误解九：Attention Matrix 是固定 Parameter

**错误。**

它根据当前：

$$
Q,K
$$

每次 forward 动态生成。

---

## 97. 常见误解十：一个 Token 在所有层里 QKV 一样

**错误。**

每层有新的 hidden representation 和自己的 projection parameters。

---

## 98. 常见误解十一：Action Query Embedding 就等于最终 Q

**不完全是。**

它还会经过 attention module 的：

$$
W_Q
$$

projection。

---

## 99. 常见误解十二：Encoder Memory 就等于 Key / Value

**不完全是。**

memory 是 source representation。

它会分别经过：

$$
W_K,W_V
$$

形成 K 和 V。

---

## 100. 常见误解十三：W_Q、W_K、W_V 的名字决定了它们功能

**错误。**

真正决定角色的是：

> 它们在计算图中连接到哪里，以及 loss 如何通过它们反向传播。

---

## 101. 用一个统一计算图记住 QKV

```text
                  INPUT REPRESENTATIONS X
                 /          |           \
                /           |            \
               ▼            ▼             ▼
             W_Q          W_K           W_V
               │            │             │
               ▼            ▼             ▼
               Q            K             V
                \          /
                 \        /
                  ▼      ▼
                   QKᵀ
                    │
                    ▼
                 scaling
                    │
                    ▼
                 Softmax
                    │
                    ▼
                    A ─────────────┐
                                   │
                                   ▼
                                  A V
                                   │
                                   ▼
                              Attention Output
                                   │
                                   ▼
                              downstream model
                                   │
                                   ▼
                                  Loss
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
                 gradient       gradient       gradient
                    │              │              │
                    ▼              ▼              ▼
                  W_Q            W_K            W_V
```

从这张图就能看出：

- Q/K 在 weight-generation path；
- V 在 content path；
- 三者最终都被同一个 task loss 训练。

---

## 102. 用一句话解释 Q

> **Query 是当前 token / decoder state 为了决定“我应该从候选信息中读取什么”而学习出来的匹配侧 representation；它本身没有人类写好的问题语义，而是因为它位于 attention score 的查询侧，被任务 gradient 训练成能够提出有用检索条件的向量。**

---

## 103. 用一句话解释 K

> **Key 是每个候选信息源为了参与匹配而学习出来的 addressing representation；它决定某个候选面对不同 Query 时会获得怎样的 compatibility score。**

---

## 104. 用一句话解释 V

> **Value 是候选信息源真正用于传递内容的 message representation；Attention 权重决定它被读取多少，而 $W_V$ 被训练成让被读取的信息对最终任务有用。**

---

## 105. 最核心的一句话

> **Q、K、V 之所以能够分别承担“查询、匹配、传递内容”的角色，不是因为向量本身天然具有这些语义，而是因为三个 learned projections 被放在了 Attention 计算图中的不同位置：Q/K 决定 attention weights，V 决定被这些 weights 聚合的内容；最终任务 loss 通过不同梯度路径长期塑造 $W_Q,W_K,W_V$，于是这些 representation roles 在训练中形成。**

如果只记一个公式和一个解释：

$$
\boxed{
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)
V
}
$$

可以读成：

```text
Q 和 K：
决定“从哪里读、读多少”

V：
决定“真正读到什么”
```

---

## 106. 下一步

现在 QKV 最容易卡住的核心问题已经解决。

下一篇最自然的是继续拆 Attention 中另一个经常“会算但不理解”的基础：

> **[Dot Product：为什么两个向量点乘就能衡量匹配？](./dot-product.md)**

这篇会从：

$$
a^\top b
$$

最基础的几何意义开始，严格解释：

- 点积为什么和夹角有关；
- $\cos\theta$ 从哪里来；
- 为什么同方向给大正值；
- 为什么正交是 0；
- 为什么反方向是负数；
- magnitude 为什么也会影响 score；
- dot product 和 cosine similarity 有什么区别；
- 为什么 QK Attention 选 dot product 而不是“向量距离”。

之后再接：

> [Softmax](./softmax.md)

把：

$$
QK^\top
$$

如何变成真正 attention weights 彻底讲透。

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

- Section 3.2 — Attention
- Section 3.2.1 — Scaled Dot-Product Attention
- Section 3.2.2 — Multi-Head Attention
- Section 3.2.3 — Applications of Attention in the Model

原论文明确说明：

> Attention 的 input 包括 queries、keys 和 values。

Scaled Dot-Product Attention：

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

原论文还明确说明 Multi-Head Attention 中：

> queries、keys 和 values 会分别经过不同的 learned linear projections，再并行执行 attention。

因此本文关于：

$$
W_Q,W_K,W_V
$$

的功能解释建立在原始计算图之上。

---

### Historical Background

Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.  
**Neural Machine Translation by Jointly Learning to Align and Translate.**  
ICLR 2015.  
arXiv:1409.0473.

- https://arxiv.org/abs/1409.0473

Bahdanau Attention 使用 decoder state 与 encoder annotations 之间的 learned alignment model：

$$
e_{ij}=a(s_{i-1},h_j)
$$

再经 Softmax 得到：

$$
\alpha_{ij}
$$

最后：

$$
c_i=\sum_j\alpha_{ij}h_j
$$

它说明了 QKV 形式背后更一般的 Attention 原理：

> **根据当前需要，对候选 memory 做动态匹配并加权读取。**

---

### 与 ACT 的关系

Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.  
**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.**  
RSS 2023.

- https://arxiv.org/abs/2304.13705

在 ACT Policy Transformer Decoder 的 cross-attention 中，可以概念化为：

$$
Q
\leftarrow
\text{action decoder representations}
$$

$$
K,V
\leftarrow
\text{observation encoder memory}
$$

所以每一个 future action slot 可以根据自己的 Query：

> 对 visual / joint / latent observation memory 进行动态检索。

---

### 本文知识连接

#### 前置知识

- [Attention](./attention.md)
- [Transformer](./transformer.md)
- Vector
- Matrix
- Linear Transformation

#### 数学核心

- [Dot Product](./dot-product.md)
- Cosine Similarity
- Matrix Multiplication
- [Softmax](./softmax.md)
- Gradient & Chain Rule

#### Attention 主线

- [Self-Attention](./self-attention.md)
- [Cross-Attention](./cross-attention.md)
- [Multi-Head Attention](./multi-head-attention.md)
- [Scaled Dot-Product Attention](./attention.md)
- [Causal Mask](./causal-mask.md)

#### ACT

- [ACT Architecture](../robot-learning/act/architecture.md)
- [ACT Complete Data Flow](../robot-learning/act/complete-data-flow.md)

#### 下一步

- [Dot Product](./dot-product.md)
- [Softmax](./softmax.md)
