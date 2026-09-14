---
title: "从 DETR 到 ACT：为什么 Object Query 最后变成了 Action Query？"
description: "从 DETR 的 set prediction、learned object queries 和 non-autoregressive parallel decoder 出发，解释 ACT 如何继承 DETR-style query-slot decoder，并把无序目标检测槽位改造成有序未来动作槽位；重点区分 query embedding、decoder state、真实 Q 向量、tgt=0、self-attention、cross-attention，以及论文与官方代码中的 query embedding差异。"
status: reviewed
pageType: application
canonical: /robot-learning/act/detr-to-act
updated: "2026-09-15"
---

# 从 DETR 到 ACT：为什么 Object Query 最后变成了 Action Query？

如果你已经把 ACT 的 Transformer Decoder 看过几遍，很可能会产生一种奇怪的感觉：

```python
self.query_embed = nn.Embedding(num_queries, hidden_dim)
```

然后：

```python
tgt = torch.zeros_like(query_embed)
```

再：

```python
hs = self.decoder(
    tgt,
    memory,
    pos=pos_embed,
    query_pos=query_embed
)
```

最后：

```python
a_hat = self.action_head(hs)
```

看起来就像：

> 凭空创建 \(k\) 个 query，

然后 Transformer Decoder 居然就输出了：

\[
k\times14
\]

个未来动作。

这很容易让人产生几个问题：

1. `query_embed` 到底是什么？
2. 为什么一个“query”最后会变成动作？
3. 它里面一开始有动作信息吗？
4. 为什么 `tgt=0` 还能工作？
5. 第 0 个 query 为什么最后预测 \(a_t\)？
6. 第 1 个 query 为什么预测 \(a_{t+1}\)？
7. 这些 query 是不是就等于 Attention 中真正的 \(Q\)？
8. 为什么 decoder 一次能同时产生 \(k\) 个 outputs？
9. 为什么不需要：
   \[
   a_t\rightarrow a_{t+1}\rightarrow a_{t+2}
   \]
   逐步生成？
10. 为什么 ACT Decoder 没有 causal mask？
11. 为什么 ACT 的代码里还残留大量：
    ```text
    DETR
    object queries
    detection slot
    ```
    这样的注释？

真正的答案是：

\[
\boxed{
\text{ACT 的 Transformer Decoder 在架构血统上明显继承了 DETR-style decoder。}
}
\]

而 DETR 最核心的创新之一，正是：

\[
\boxed{
\text{用一组 query slots 并行地产生多个结构化输出。}
}
\]

在 DETR 中：

\[
\text{Query Slot}
\rightarrow
\text{Object Prediction}
\]

到了 ACT：

\[
\text{Query Slot}
\rightarrow
\text{Future Action Prediction}
\]

但是这句话如果只理解成：

> “Object Query 改名成 Action Query”

仍然太浅。

因为 DETR 和 ACT 的输出结构其实有一个根本区别：

\[
\boxed{
\text{DETR 输出的是 set}
}
\]

而：

\[
\boxed{
\text{ACT 输出的是 ordered temporal sequence}
}
\]

所以 ACT 真正继承的是：

> **DETR 的 parallel query-slot decoder mechanism，**

而不是：

> DETR 的完整 set-prediction semantics。

这一篇就把这条演化链完整拆开。

---

# 1. 先忘掉 ACT：传统 Object Detection 为什么麻烦？

给一张图片：

\[
I
\]

我们想输出图片中的所有 objects：

\[
\{
(o_1,b_1),
(o_2,b_2),
\ldots
\}
\]

其中：

- \(o_i\)：类别；
- \(b_i\)：bounding box。

问题是：

> 一张图到底有几个物体？

可能：

\[
1
\]

个，

也可能：

\[
17
\]

个。

---

# 2. 传统 Detector 常使用大量 Candidate Mechanisms

DETR 论文指出，当时主流 detector 通常依赖诸如：

- anchors；
- proposals；
- window centers；
- target assignment heuristics；
- NMS（Non-Maximum Suppression）；

等机制。

模型可能先产生：

> 很多候选框，

然后再通过各种规则：

> 去重、筛选、匹配。

DETR 提出了一个非常不同的想法：

\[
\boxed{
\text{直接把 Object Detection 当成 Set Prediction。}
}
\]

---

# 3. 什么叫 Set Prediction？

假设图里真正有：

```text
dog
person
ball
```

作为一个集合：

\[
\{
dog,\ person,\ ball
\}
\]

顺序：

```text
dog, person, ball
```

和：

```text
ball, dog, person
```

从集合意义上：

> 是同一个答案。

所以 detection 的目标天然没有一个固定 sequence ordering。

---

# 4. DETR 直接输出固定数量的 Prediction Slots

DETR 不先产生几万个 anchors。

而是设定固定：

\[
N
\]

个 decoder output slots。

论文 baseline 使用：

\[
\boxed{
N=100
}
\]

因此每张图都产生：

\[
100
\]

个 predictions。

---

# 5. 图里只有 3 个 Object 怎么办？

100 个 slots 中：

- 3 个可以预测真实 objects；
- 其余预测：
  \[
  \varnothing
  \]
  即：
  > no object。

所以：

\[
\boxed{
\text{fixed number of output slots}
}
\]

也可以处理：

> variable number of actual objects。

---

# 6. 这一步非常关键

我们第一次看到一个思想：

> **输出数量可以由一组 decoder slots 预先固定。**

每个 slot：

> 都是一个“等待被填充”的 output position。

DETR 把这些 slot 的身份编码称为：

\[
\boxed{
\text{Object Queries}
}
\]

---

# 7. DETR 的整体数据流

论文 Figure 2：

```text
image
  │
  ▼
CNN backbone
  │
  ▼
2D feature map
  │
  ▼
flatten + spatial positional encoding
  │
  ▼
Transformer Encoder
  │
  ▼
image memory
  │
  ├──────────────────────┐
  │                      │
  ▼                      │
Transformer Decoder ◄──── Object Queries
  │
  ▼
N decoder outputs
  │
  ├─→ class
  ├─→ box
  ├─→ class
  ├─→ box
  ...
```

每个 decoder slot最终产生：

\[
\boxed{
\text{class}+\text{bounding box}
}
\]

---

# 8. Object Query 第一层直觉：它是一个 Output Slot Identity

先不要把“Query”理解成自然语言问题。

可以先把：

\[
q_1,\ldots,q_N
\]

想成：

> N 个不同的 output-slot identifiers。

例如：

```text
slot 1
slot 2
slot 3
...
slot 100
```

如果所有 slot一开始完全相同，

对称的 Transformer Decoder就容易：

> 产生相同输出。

所以必须有某种方式：

> 区分这些 slots。

---

# 9. DETR 用 Learned Positional Embeddings 区分 Output Slots

DETR 论文明确说明：

> Decoder需要 \(N\) 个不同 input embeddings 来产生不同结果。

这些 learned output positional encodings：

\[
\boxed{
q_1,\ldots,q_N
}
\]

被称为：

> Object Queries。

所以一个非常重要的纠错：

\[
\boxed{
\text{Object Query 首先是一种 learned output-slot positional identity。}
}
\]

---

# 10. 它不是“这个 Query 天生代表 Dog”

DETR 并没有人工规定：

```text
query 1 = dog
query 2 = person
query 3 = car
```

也没有：

```text
query 17 = 左上角物体
```

这样的固定语义。

Object Query 的语义：

> 通过训练涌现。

---

# 11. 为什么不能固定 Query 1 = Dog？

因为一张图里可能有：

- 0 条狗；
- 1 条狗；
- 5 条狗。

Detection slots需要：

> 灵活承担不同实例。

所以 Object Queries更像：

\[
\boxed{
\text{generic prediction slots}
}
\]

而不是 class-specific slots。

---

# 12. Query 也不是 Ground-Truth Object Feature

在 Decoder 开始前：

> Object Query 并不知道当前图片中哪个物体存在。

图片内容在：

\[
\boxed{
Encoder\ Memory
}
\]

里面。

Query 的作用是：

> 提供不同 decoder slots 的身份，让这些 slots去读取 image memory。

---

# 13. 为什么叫 Query？

因为在 Cross-Attention 中：

\[
\text{decoder slot}
\]

站在：

\[
Q
\]

这一侧，

然后去读取：

\[
\text{encoder memory}
\]

中的：

\[
K,V
\]

所以它承担：

> “发起读取”的角色。

---

# 14. 但这里必须做一个更严格的区分

很多人会说：

\[
\boxed{
\text{Object Query} = Q
}
\]

这并不严格。

在 DETR-style implementation里，

我们有：

\[
\boxed{
query\_pos
}
\]

即：

> Object Query Embedding。

真正进入 Multi-Head Attention 的 query vector还要经过：

\[
tgt+query\_pos
\]

再经过：

\[
W_Q
\]

---

# 15. 所以三个东西不要混淆

## 1. Query Embedding / Query Positional Embedding

\[
q_i^{pos}
\]

这是：

> slot identity。

---

## 2. Decoder Hidden State

\[
h_i
\]

这是：

> slot当前已经积累的 content representation。

---

## 3. 真正 Attention Q

某一 head：

\[
\boxed{
Q_i^{(h)}
=
(h_i+q_i^{pos})W_Q^{(h)}
}
\]

它才是我们 [Q/K/V](../../deep-learning/qkv.md) 文章里的真正 Query vector。

---

# 16. 这个区别以后读任何 DETR-style 模型都非常重要

代码变量：

```python
query_embed
```

不等于：

> Attention公式中的最终 \(Q\)。

它更接近：

\[
\boxed{
\text{decoder query positional embedding}
}
\]

---

# 17. DETR Decoder 一开始的 Content State 是什么？

这是一个很反直觉的地方。

DETR supplement明确写：

> decoder receives queries initially set to zero, output positional encoding (object queries), and encoder memory.

也就是说：

\[
\boxed{
tgt_0=0
}
\]

同时有：

\[
\boxed{
query\_pos=q^{object}
}
\]

---

# 18. 为什么一开始 Content 可以是 0？

因为：

> 当前图像里的 object content不需要预先放进 decoder slots。

Decoder真正需要的 information：

> 在 Encoder Memory 中。

Query slot只需要：

1. 有独立 identity；
2. 能通过 Cross-Attention从 memory读取内容。

---

# 19. 第一个 Cross-Attention 才是关键内容注入

概念上：

\[
Q=
(tgt+query\_pos)W_Q
\]

\[
K=
(memory+pos)W_K
\]

\[
V=
memoryW_V
\]

当：

\[
tgt=0
\]

时：

\[
Q=
query\_pos W_Q
\]

所以：

> query positional embedding已经足够产生不同的 query vectors。

然后这些 queries 去读取：

\[
memory
\]

---

# 20. Cross-Attention Output 把 Image Content 写进 Decoder Slot

\[
Attention(Q,K,V)
\]

得到：

> 当前 slot 从 image memory检索出的内容。

Residual以后：

\[
tgt
\]

不再是0。

它逐渐变成：

> 一个 image-conditioned object representation。

---

# 21. 所以 Object Query 本身不是 Object Representation

非常重要：

\[
\boxed{
\text{Object Query}
\neq
\text{detected object feature}
}
\]

更准确：

```text
Object Query
= learned slot identity / search prior

Decoder Hidden State
= query slot after repeatedly reading image memory
```

---

# 22. 为什么 DETR 第一层 Self-Attention 甚至可以跳过？

DETR supplement明确指出：

> 第一 decoder layer 的 first self-attention 可以被跳过。

原因从结构上很好理解。

一开始：

\[
tgt=0
\]

主要差异来自：

\[
query\_pos
\]

真正 sample-specific内容尚未通过 Cross-Attention进入。

因此第一层最关键的工作：

> 是让每个 slot读取 image memory。

---

# 23. 到第二层以后 Self-Attention 就变得更有意义

经过第一层 Cross-Attention：

\[
h_1,\ldots,h_N
\]

已经包含：

> 不同 image-conditioned information。

这时 Self-Attention可以让 output slots：

> 相互交流。

---

# 24. DETR 为什么需要 Output-Slot Self-Attention？

例如：

> 两个 query slots都盯上了同一个 object。

通过 decoder self-attention，

slots可以：

> 彼此感知、协调。

DETR论文强调 Transformer可以：

> globally reason about all objects together using pairwise relations。

这对减少：

> duplicate detections

很重要。

---

# 25. 但 Self-Attention 并不是 DETR 去重的唯一机制

真正让 set prediction成立的另一个核心：

\[
\boxed{
\text{Bipartite Matching Loss}
}
\]

也就是 Hungarian matching。

---

# 26. 为什么 DETR 必须 Matching？

假设 GT：

```text
dog
person
```

Model slots输出：

```text
slot 7 → person
slot 23 → dog
```

完全正确。

不能要求：

```text
dog 必须 slot 1
person 必须 slot 2
```

因为 object set：

> 没有固定 ordering。

---

# 27. 所以 Training 必须先决定谁对谁

DETR寻找 prediction slots与ground truth objects之间的：

> 最优 one-to-one assignment。

概念：

\[
\boxed{
\sigma^*
=
\arg\min_{\sigma}
\text{matching cost}
}
\]

然后再对匹配后的 pairs计算：

- class loss；
- box L1；
- GIoU；

等。

---

# 28. 这让 DETR 的 Query Slots 不需要固定 Object Identity

今天：

```text
query 3
```

可能负责：

> person。

另一张图：

> 可能负责 car。

重要的是：

> 整体 set预测正确。

---

# 29. DETR 最重要的 Parallelism

传统 autoregressive sequence model：

\[
y_1
\rightarrow
y_2
\rightarrow
y_3
\rightarrow\cdots
\]

DETR则：

\[
\boxed{
N\text{ outputs in parallel}
}
\]

论文明确把这一点称为：

> non-autoregressive parallel decoding。

---

# 30. 什么叫 Autoregressive？

一般：

\[
p(y_{1:N}|x)
=
\prod_{i=1}^{N}
p(
y_i
|
y_{<i},
x
)
\]

生成：

```text
先 y1
再 y2
再 y3
...
```

后一个输出依赖：

> 已经生成的前面输出。

---

# 31. DETR 不这样

DETR不是：

```text
先找到一个 object
↓
把这个 object 喂回来
↓
再找下一个
```

而是：

```text
100 query slots
↓
共同读取 image memory
↓
一次产生100个 predictions
```

---

# 32. 为什么 DETR 可以 Parallel？

因为输出目标是：

> set。

没有天然要求：

\[
object_1
\]

必须先于：

\[
object_2
\]

产生。

再配合：

> permutation-invariant matching loss，

所以所有 slots可以同时预测。

---

# 33. 到这里，我们已经看到了 ACT 的影子

ACT 想做的事情：

不是输出：

\[
1
\]

个 action。

而是：

\[
\boxed{
k\text{ 个未来 actions}
}
\]

例如：

\[
k=100
\]

目标：

\[
[a_t,a_{t+1},\ldots,a_{t+99}]
\]

---

# 34. 一个最自然的设计问题

怎么一次输出100个 14-D actions？

可以让：

> 一个巨大 MLP直接输出 \(100\times14\)。

例如：

\[
512
\rightarrow
1400
\]

当然理论上可以。

---

# 35. ACT 选择了另一种更结构化的方式

使用：

\[
\boxed{
100\text{ decoder slots}
}
\]

每一个 slot最终：

\[
512
\rightarrow14
\]

预测一个 action。

于是：

\[
100\times512
\]

Decoder hidden states经过共享：

\[
action\_head
\]

得到：

\[
\boxed{
100\times14
}
\]

---

# 36. 这就是 DETR Skeleton 被迁移到 Action Sequence

DETR：

```text
N Object Query Slots
↓
Decoder
↓
N Object Representations
↓
shared prediction head
↓
N Object Predictions
```

ACT：

```text
k Action Query Slots
↓
Decoder
↓
k Action-Slot Representations
↓
shared action head
↓
k Action Predictions
```

---

# 37. 从结构上看，它们极其相似

DETR：

\[
\boxed{
\text{one decoder output embedding per object-prediction slot}
}
\]

ACT：

\[
\boxed{
\text{one decoder output embedding per future-action slot}
}
\]

---

# 38. ACT 官方代码甚至直接保留了 DETR 血统

官方 `detr/models/transformer.py` 文件自己写着：

```text
DETR Transformer class.
Copy-paste from torch.nn.Transformer with modifications
```

而 `detr_vae.py` 中类名就是：

```python
class DETRVAE(nn.Module):
```

甚至 docstring仍写：

```text
This is the DETR module that performs object detection
```

---

# 39. 代码注释里的 num_queries 也残留 DETR 原文

官方：

```python
num_queries:
    number of object queries,
    ie detection slot.
    This is the maximal number
    of objects DETR can detect...
```

但在 ACT：

```python
'num_queries':
    args['chunk_size']
```

也就是说：

\[
\boxed{
num\_queries
=
chunk\_size
}
\]

---

# 40. 这是最直接的代码证据

DETR 中：

\[
num\_queries
=
\text{maximum prediction slots}
\]

ACT 中：

\[
\boxed{
num\_queries
=
k
=
\text{number of future action slots}
}
\]

所以同一个 architecture interface：

> 被重新解释了。

---

# 41. 但这里有一个极其重要的区别

DETR：

\[
\boxed{
\text{output slots form an unordered set}
}
\]

ACT：

\[
\boxed{
\text{output slots form an ordered temporal sequence}
}
\]

这意味着训练 semantics完全不同。

---

# 42. DETR Query Slot 没有固定 Target Index

假设 3 个 objects：

```text
A B C
```

预测：

```text
slot 17 → C
slot 2  → A
slot 91 → B
```

只要 Hungarian matching可以匹配：

> 就没问题。

---

# 43. ACT Action Slot 有固定 Temporal Index

假设：

\[
k=4
\]

target chunk：

\[
[
a_t,
a_{t+1},
a_{t+2},
a_{t+3}
]
\]

那么 decoder output：

\[
[
\hat a^{(0)},
\hat a^{(1)},
\hat a^{(2)},
\hat a^{(3)}
]
\]

直接逐位置监督：

\[
\hat a^{(0)}
\leftrightarrow
a_t
\]

\[
\hat a^{(1)}
\leftrightarrow
a_{t+1}
\]

\[
\hat a^{(2)}
\leftrightarrow
a_{t+2}
\]

\[
\hat a^{(3)}
\leftrightarrow
a_{t+3}
\]

---

# 44. 所以 ACT 不需要 Hungarian Matching

因为 target sequence已经有：

\[
\boxed{
\text{canonical temporal order}
}
\]

第0位置就是：

> chunk中的第一步。

第1位置就是：

> 第二步。

---

# 45. 这就是从 DETR 到 ACT 最重要的“语义改造”

DETR：

\[
\boxed{
\text{slot identity breaks symmetry, matching determines object assignment}
}
\]

ACT：

\[
\boxed{
\text{slot identity represents temporal position, index directly determines target assignment}
}
\]

---

# 46. 所以“Action Query”比“Object Query”更接近 Position Slot

对于 ACT，

一个很有用的 mental model是：

```text
query slot 0:
“我要生成未来第0步动作”

query slot 1:
“我要生成未来第1步动作”

...

query slot k-1:
“我要生成未来第k-1步动作”
```

注意：

> 这是一种功能直觉，不是 query embedding里真的存着中文问题。

---

# 47. Action Query 不是 Action 本身

这是非常重要的纠错。

\[
\boxed{
query_i
\neq
a_{t+i}
}
\]

Query只是：

> 一个 output position identity / decoder positional signal。

真正 action：

\[
\hat a_{t+i}
\]

要经过：

1. query-conditioned decoder computation；
2. cross-attention读取 observation memory；
3. multiple decoder layers；
4. action head；

才能产生。

---

# 48. Action Query 也不是 Ground-Truth Action Embedding

训练时 target：

\[
a_{t:t+k}
\]

不会作为 Policy Decoder 的 input tokens喂给 decoder。

它们用于：

- CVAE Encoder，帮助推断 \(z\)；
- reconstruction target。

Policy Decoder本身并不是 teacher-forced autoregressive action decoder。

---

# 49. 这点和语言 Transformer Decoder 非常不同

传统 autoregressive translation decoder训练时：

> 常把 shifted target tokens作为 decoder input。

ACT：

> 不把前面 ground-truth action作为下一 action slot的 decoder input。

---

# 50. ACT Decoder 的 Content Input 一开始是什么？

官方代码：

```python
query_embed =
    query_embed.unsqueeze(1)
    .repeat(1, bs, 1)

tgt =
    torch.zeros_like(
        query_embed
    )
```

所以：

\[
\boxed{
tgt_0=0
}
\]

与 DETR 同样的 design pattern。

---

# 51. 所以最开始 Decoder 没有 Previous Actions

不是：

\[
tgt=
[a_t,a_{t+1},...]
\]

也不是：

\[
tgt=
[\text{previous predicted actions}]
\]

而是：

\[
\boxed{
k\times512\text{ zero content slots}
}
\]

外加：

\[
query\_pos
\]

来区分 positions。

---

# 52. ACT Code 的 Query Embedding

官方 release：

```python
self.query_embed =
    nn.Embedding(
        num_queries,
        hidden_dim
    )
```

因此每一个：

\[
i\in[0,k-1]
\]

都有一个 learned：

\[
512\text{-D}
\]

embedding。

---

# 53. 所以 Released Code 中 Query Slots 是 Learned

例如：

\[
k=100
\]

则：

\[
query\_embed.weight
\in
\mathbb R^{100\times512}
\]

这些都是：

> trainable parameters。

---

# 54. 但 ACT 论文 Appendix 写了什么？

这里存在一个值得明确记录的：

\[
\boxed{
\text{paper/code discrepancy}
}
\]

论文详细架构描述写：

> Transformer decoder的“queries”在 first layer 是 fixed sinusoidal embeddings。

Figure 11 也标注：

> position embeddings (fixed)。

---

# 55. 但 Released Official Code 明确是

```python
nn.Embedding(
    num_queries,
    hidden_dim
)
```

也就是：

\[
\boxed{
\text{learned query embeddings}
}
\]

而不是 fixed sinusoidal table。

---

# 56. 所以我们必须区分两个事实

## Paper Fact

ACT论文描述：

\[
\boxed{
\text{fixed sinusoidal decoder query embeddings}
}
\]

至少在其 detailed architecture文字/图中如此。

---

## Released Code Fact

官方 GitHub：

\[
\boxed{
\text{learned }nn.Embedding(k,512)
}
\]

并以：

```python
query_pos=query_embed
```

传入 DETR-style decoder。

---

# 57. 不应该偷偷把两者混成一个

如果文章只写：

> “ACT使用fixed sinusoidal action queries”

会与官方代码冲突。

如果只写：

> “ACT论文设计learned action queries”

又与论文 Appendix不符。

因此正确写法是：

\[
\boxed{
\text{论文与released implementation在这一细节上存在差异。}
}
\]

---

# 58. 但两者的核心功能相同

无论 query identity是：

- fixed sinusoidal；
- learned embedding；

它们都承担：

\[
\boxed{
\text{distinguish future action slots}
}
\]

这才是 architecture层面的核心。

---

# 59. 为什么必须区分 100 个 Action Slots？

假设全部 query positional signals都一样：

\[
q_0=q_1=\cdots=q_{99}
\]

且：

\[
tgt=0
\]

Decoder对称结构就缺乏：

> “这是future step 0还是future step 57”

的信息。

---

# 60. Query Embedding 打破 Output-Slot Symmetry

\[
q_0\neq q_1\neq\cdots
\]

使每个 slot具有：

> distinct identity。

训练进一步把这些 identities与：

\[
a_t,
a_{t+1},
...
\]

对应起来。

---

# 61. 所以 Query Positional Signal 是 Decoder 的“时间地址”

对 ACT 可以把它类比成：

> future action address。

例如：

```text
slot 0 address
slot 1 address
...
slot 99 address
```

Decoder在同一个 observation memory上回答：

> 每一个 future address对应什么动作。

---

# 62. 这是类比，不是严格说它一定编码物理时间间隔

如果控制频率固定：

\[
50Hz
\]

那么 slot index确实对应固定相对时间步。

例如：

\[
i
\]

对应：

\[
t+i
\]

但 query embedding内部的 geometry：

> 不保证线性表示秒数。

---

# 63. ACT Decoder 中 Query Pos 不是加到 Value 上

官方：

```python
q = k =
    self.with_pos_embed(
        tgt,
        query_pos
    )

tgt2 =
    self.self_attn(
        q,
        k,
        value=tgt,
        ...
    )
```

也就是说：

Self-Attention：

\[
Q/K
\]

使用：

\[
tgt+query\_pos
\]

而 Value：

\[
V
\]

来自：

\[
tgt
\]

---

# 64. Cross-Attention

官方：

```python
self.multihead_attn(
    query =
        tgt + query_pos,

    key =
        memory + pos,

    value =
        memory
)
```

概念：

\[
\boxed{
Qsource=tgt+query\_pos
}
\]

\[
\boxed{
Ksource=memory+pos
}
\]

\[
\boxed{
Vsource=memory
}
\]

---

# 65. 第一个 Decoder Layer：tgt = 0

所以第一层 Cross-Attention的 query-side source：

\[
\boxed{
query\_pos
}
\]

也就是说：

> slot identity直接决定第一次如何去读取 observation memory。

---

# 66. 这回答“tgt=0 为什么还能工作”

因为：

\[
\boxed{
0+query\_pos
=
query\_pos
}
\]

Query projection：

\[
Q_h
=
query\_pos W_Q^{(h)}
\]

仍然是非零、slot-specific的。

---

# 67. Encoder Memory 中已经有什么？

ACT Policy Encoder已经把：

- four-camera visual tokens；
- current qpos token；
- latent \(z\) token；

融合为：

\[
memory
\]

所以 Decoder不需要自己从0“创造动作知识”。

它可以：

> 用 query slots读取已经 contextualized的 observation memory。

---

# 68. ACT Cross-Attention 的真正含义

对于 action slot \(i\)：

\[
q_i
\]

去读取：

\[
memory_1,\ldots,memory_{1202}
\]

得到：

\[
c_i
\]

可以理解为：

> **为了预测 future action position \(i\)，当前 observation memory中哪些信息重要？**

---

# 69. 不同 Future Slots 可以读不同 Memory

例如只作为直觉：

较近 future slot可能更多依赖：

> 当前 gripper附近状态。

较远 slot可能需要：

> 更全局任务context。

Architecture允许：

\[
A_i\neq A_j
\]

但不能未经实验就断言：

> ACT一定学成这种模式。

---

# 70. 为什么 Action Slots 之间还需要 Self-Attention？

如果每个 future action完全独立预测：

\[
\hat a_i=f_i(memory)
\]

可能缺少：

> 整段动作的内部一致性。

Decoder Self-Attention让：

\[
slot_i
\]

可以读取：

\[
slot_j
\]

的 hidden state。

---

# 71. 于是整个 Chunk 可以联合建模

例如机械臂轨迹需要：

- 平滑；
- 连贯；
- 前后动作协调。

Action-slot Self-Attention提供：

\[
\boxed{
\text{within-chunk interaction}
}
\]

---

# 72. 这与 DETR 的 Slot Coordination 很像

DETR：

> Object slots交流，避免/处理重复对象、建模对象间关系。

ACT：

> Action slots交流，形成 coherent action sequence。

论文也明确说：

> transformer decoder generates a coherent action sequence。

---

# 73. 但二者协调目标不同

DETR：

\[
\text{set-level object consistency}
\]

ACT：

\[
\text{temporal action-sequence coherence}
\]

所以：

> 同一个 decoder mechanism，被不同 loss塑造成不同功能。

---

# 74. 这再次印证我们之前 Linear/QKV文章里的原则

\[
\boxed{
\text{Architecture role}
+
\text{training objective}
\rightarrow
\text{learned semantics}
}
\]

同样的 DETR-style query decoder：

> 不会天然只适合 objects。

---

# 75. 为什么 ACT 不需要 Autoregressive Action Generation？

一种可能方案是：

\[
p(
a_t,\ldots,a_{t+k-1}
|
o_t
)
\]

分解：

\[
\prod_{i=0}^{k-1}
p(
a_{t+i}
|
a_{t:t+i-1},
o_t
)
\]

然后一步一步生成。

---

# 76. 但 ACT 没这么做

它直接用一个 policy forward：

\[
\boxed{
\pi_\theta(
a_{t:t+k}
|
o_t,z
)
}
\]

输出：

\[
k\times14
\]

tensor。

Decoder所有 slots：

> 一次并行计算。

---

# 77. 所以 ACT 是 Non-Autoregressive Chunk Prediction

不是：

```text
predict a0
feed a0 back
predict a1
feed a1 back
...
```

而是：

```text
k query slots
      │
      ▼
parallel decoder
      │
      ▼
a0, a1, ..., a{k-1}
```

---

# 78. Parallel 不等于 Slots Completely Independent

这是非常重要的。

虽然 outputs：

> 并行产生，

但 Decoder Self-Attention让：

\[
slot_i
\]

和：

\[
slot_j
\]

相互交互。

所以：

\[
\boxed{
\text{parallel}
\neq
\text{independent}
}
\]

---

# 79. 这是 ACT Decoder 最漂亮的地方之一

它同时获得：

### 并行性

不用逐 action生成。

### 联合建模

通过 Self-Attention：

> action slots彼此交流。

### Observation Conditioning

通过 Cross-Attention：

> 所有 slots读取同一 observation memory。

---

# 80. 为什么没有 Causal Mask？

Autoregressive language decoder需要：

\[
token_i
\]

不能看到：

\[
token_{i+1}
\]

否则训练时会泄露future targets。

所以使用：

> causal mask。

---

# 81. ACT Action Decoder 不喂 Ground-Truth Future Actions

它的 decoder slots不是：

\[
[a_t,a_{t+1},...]
\]

的 shifted target embeddings。

所以不存在：

> “slot 0 偷看到 future ground-truth action”

这种 teacher-forcing leakage。

---

# 82. ACT 想联合预测完整 Chunk

它本来就希望：

\[
slot_0,\ldots,slot_{k-1}
\]

彼此协调。

因此：

> 没有必要禁止 slot \(i\) 与 slot \(j>i\) 的 hidden state交互。

---

# 83. 官方 Code 确实没有提供 Causal Target Mask

Transformer forward：

```python
self.decoder(
    tgt,
    memory,
    memory_key_padding_mask=mask,
    pos=pos_embed,
    query_pos=query_embed
)
```

没有：

```python
tgt_mask=causal_mask
```

所以默认：

\[
\boxed{
\text{bidirectional self-attention among action slots}
}
\]

---

# 84. 这里“看未来”不会违反因果吗？

不会。

因为它不是在真实时间：

> 执行一步后获得未来真实 observation。

而是在当前时刻：

\[
t
\]

内部共同计算：

> 一整段 planned/predicted actions。

所有 slot hidden states都是：

> 当前 forward内部的 model representations。

不是未来环境真值。

---

# 85. 这是 Planning-like Parallel Representation，而不是 Future Information Leakage

模型只知道当前输入：

\[
o_t,z
\]

没有：

\[
o_{t+1},
o_{t+2}
\]

真实 future observations。

所以 slot之间交流：

> 只是让预测动作彼此协调。

---

# 86. 这和 Autoregressive 模型的 Factorization 不同

Autoregressive：

\[
\boxed{
\text{causal ordering is built into architecture}
}
\]

ACT：

\[
\boxed{
\text{temporal ordering is encoded in output-slot identity and supervised target index}
}
\]

但 computation：

> 可以并行、双向协调。

---

# 87. 这是一个非常高级且重要的区别

“输出是 sequence”：

\[
\neq
\]

“必须 autoregressive生成”。

Sequence可以：

- autoregressive；
- non-autoregressive；
- diffusion；
- masked prediction；
- parallel regression；

等多种方式生成。

ACT选择：

\[
\boxed{
\text{parallel sequence regression/generation}
}
\]

---

# 88. Action Slot 为什么最终会绑定到特定时间位置？

Released code中的 query embeddings：

> 一开始只是随机 learned vectors。

它们并不知道：

```text
query 0 = now
query 1 = +20ms
...
```

---

# 89. Temporal Semantics 来自 Training Alignment

训练时：

\[
a_{hat}[:,0]
\]

总是和：

\[
actions[:,0]
\]

比较。

\[
a_{hat}[:,1]
\]

总是和：

\[
actions[:,1]
\]

比较。

---

# 90. 于是 Gradient 把 Slot Identity 绑定到 Time Index

如果 query slot \(i\) 最终总负责：

\[
a_{t+i}
\]

那么它收到的 loss：

> 永远来自该相对 future position。

久而久之：

\[
\boxed{
q_i
\text{ acquires temporal-slot semantics through supervision}
}
\]

---

# 91. 这与 DETR 的 Hungarian Matching 完全不同

DETR query \(i\)：

> 每张图匹配哪个GT object可以变化。

ACT query \(i\)：

> target位置固定就是第 \(i\) 个 future action。

---

# 92. 所以 ACT Queries 更“有序”

从 loss语义：

\[
q_0,q_1,\ldots,q_{k-1}
\]

形成：

\[
\boxed{
\text{ordered slots}
}
\]

DETR queries则主要是：

> set slots。

---

# 93. 为什么 ACT 论文可能倾向 Fixed Sinusoidal Position Embeddings？

从 conceptual standpoint，

因为 future slots天然有：

\[
0,1,\ldots,k-1
\]

序列位置。

Sinusoidal position encoding正适合：

> 提供不同 ordered positional identities。

---

# 94. Released Code 改成 Learned Query Embeddings 也合理

因为模型可以直接学习：

\[
q_i
\]

作为：

> 每个 horizon position最适合的 slot identity。

不必强制 sinusoidal geometry。

---

# 95. Fixed vs Learned 的真正差别

## Fixed Sinusoidal

位置结构：

> 人工定义、无参数。

可能带有：

> smooth frequency-based positional structure。

---

## Learned Embedding

每个位置：

\[
q_i
\]

直接训练。

更灵活，

但不自带：

> 外推到未训练 slot index的 sinusoidal structure。

---

# 96. 对 ACT 原始固定 Chunk Size，这种外推通常不是主要需求

训练：

\[
k=100
\]

推理也：

\[
k=100
\]

所以 learned query table：

> 完全可以工作。

---

# 97. Action Query 是不是“时间 Positional Encoding”？

可以说：

> 它承担 decoder output temporal slot identity。

但最好不要完全等同普通 sequence input positional encoding。

因为：

- 它用于产生 output slots；
- 参与 decoder queries；
- 与 encoder image positional encoding角色不同。

---

# 98. Encoder Positional Encoding 回答

对于 visual memory：

> “这个 feature来自图像哪个位置？”

ACT有2D sinusoidal position encoding。

---

# 99. Decoder Query Position 回答

对于 output slot：

> “这是哪一个未来动作槽位？”

两个都叫 position information，

但位置空间不同：

\[
\boxed{
\text{image spatial position}
}
\]

vs：

\[
\boxed{
\text{action horizon position}
}
\]

---

# 100. Cross-Attention 就是在两个坐标系统之间建立联系

Query side：

\[
\text{future action positions}
\]

Key side：

\[
\text{observation memory positions}
\]

Attention学习：

\[
\boxed{
\text{which observation information each future action slot should read}
}
\]

---

# 101. 这就是 ACT Decoder 的高级统一视角

可以把 Cross-Attention看成：

\[
\boxed{
\text{future-action slots}
\rightarrow
\text{observation memory retrieval}
}
\]

而不是仅仅：

> “Decoder里面又一个Attention。”

---

# 102. DETR 也是同样形式

\[
\boxed{
\text{object prediction slots}
\rightarrow
\text{image memory retrieval}
}
\]

所以 architectural analogy非常清晰。

---

# 103. 从 DETR 到 ACT，可以写成一张映射表

| DETR | ACT |
|---|---|
| image | multi-camera observation + joints + z |
| CNN backbone | ResNet18 backbone |
| encoder image memory | multimodal observation memory |
| object query slots | future action query slots |
| N queries | k action slots |
| decoder self-attention | action-slot self-attention |
| decoder cross-attention | action-slot → observation-memory cross-attention |
| box/class head | 14-D action head |
| set prediction | ordered chunk prediction |
| Hungarian matching | direct temporal index supervision |
| no-object slots | padded action positions when needed |
| non-autoregressive parallel objects | non-autoregressive parallel future actions |

---

# 104. 但“no-object”和Padding也不能完全类比

DETR：

\[
\varnothing
\]

是一个真正参与分类的：

> no-object class。

ACT：

> padded action positions通过 `is_pad` mask在 L1 中被忽略。

所以两者只是：

> 都需要处理固定 slot数与有效目标数不完全一致。

Loss机制并不相同。

---

# 105. ACT 有 `is_pad_head` 是否就等于 DETR 的 no-object head？

不应直接等同。

虽然代码保留：

```python
is_pad_head
```

但当前官方 `policy.py` 的 ACT total loss：

> 并没有使用 `is_pad_hat`。

因此 released training objective主要通过：

> L1 mask

处理 padded target positions。

---

# 106. 这是 DETR 代码演化留下的另一个痕迹

DETR本来天然需要：

> slot是否有object。

ACT代码保留了：

> padding-related head。

但当前官方 loss逻辑：

> 并未对它施加专门监督。

这也再次说明：

\[
\boxed{
\text{ACT 不是 DETR 原封不动复制}
}
\]

---

# 107. ACT 的 Encoder Memory 与 DETR 也不同

DETR memory主要来自：

\[
\boxed{
\text{image feature map}
}
\]

ACT memory包含：

- 4-camera visual tokens；
- current qpos；
- latent \(z\)。

所以：

\[
\boxed{
\text{ACT memory is multimodal state-conditioned memory}
}
\]

---

# 108. ACT 在 Encoder 输入前添加两个特殊 Tokens

官方：

```python
addition_input =
    torch.stack(
        [latent_input, proprio_input],
        axis=0
    )

src =
    torch.cat(
        [addition_input, src],
        axis=0
    )
```

所以 Policy Encoder sequence：

```text
z token
joint token
visual tokens...
```

---

# 109. Decoder 因此可以通过 Cross-Attention读取

不仅是：

> pixels / visual information。

还可以读取：

- current robot configuration；
- latent style condition。

所以 Action Query的内容形成过程：

> 比 DETR object query更具控制条件。

---

# 110. DETR Query Slot 最终输出什么？

Decoder output embedding：

\[
h_i
\]

进入：

- class head；
- box FFN。

所以：

\[
\boxed{
h_i
\rightarrow
(class_i,box_i)
}
\]

---

# 111. ACT Query Slot 最终输出什么？

Decoder output embedding：

\[
h_i\in\mathbb R^{512}
\]

进入：

```python
action_head =
    nn.Linear(
        512,
        14
    )
```

得到：

\[
\boxed{
\hat a_i\in\mathbb R^{14}
}
\]

---

# 112. 同一个 Action Head 被所有 Slots 共享

不是：

```text
slot 0 有一个 head
slot 1 有另一个 head
...
```

而是同一个：

\[
W_{action}
\]

对每个：

\[
h_i
\]

独立应用。

---

# 113. 那不同时间位置怎么产生不同动作？

因为：

\[
h_0\neq h_1\neq\cdots
\]

它们受：

- 不同 query positions；
- self-attention；
- cross-attention；

影响。

所以共享 readout：

\[
W_{action}
\]

仍能产生不同：

\[
a_i
\]

---

# 114. 这和 DETR Shared Prediction Head 完全同一种设计哲学

DETR也对不同 decoder output embeddings：

> 使用 shared prediction heads。

slot-specific差异主要存在于：

> hidden representation，

而不是每个 slot拥有完全独立 output network。

---

# 115. 为什么共享 Head 是合理的？

因为所有 slot输出都属于：

> 同一种数据类型。

DETR：

> 都是 object prediction。

ACT：

> 都是 14-D robot joint targets。

区别只是：

> slot context/position不同。

---

# 116. 如果每个 ACT Slot 使用独立 Action Head 呢？

理论上可以：

\[
W_0,\ldots,W_{k-1}
\]

但参数会大幅增加，

并把 temporal-position identity：

> 硬编码到 output heads。

共享 Action Head + query identities：

> 更结构化、更参数共享。

---

# 117. Query Embedding 是不是每个 Sample 都不同？

不是。

Released ACT：

\[
query\_embed.weight
\]

是 model parameters。

同一个：

\[
q_i
\]

用于所有 samples。

---

# 118. 什么随 Sample 改变？

Encoder memory：

\[
M(o_t,z)
\]

不同。

因此同一个 action query \(q_i\)：

> 面对不同 observation memory，

通过 Cross-Attention得到不同 hidden output。

---

# 119. 所以 Action Query 更像“问题模板”

例如：

> “给定当前 observation，第 17 个 future action slot应该是什么？”

这个“第17个slot”：

> 是固定 query identity。

但答案：

> 随 observation变化。

---

# 120. 再强调：这是类比

网络没有中文语义：

> “第17步是什么？”

它只有：

- learned/fixed vector；
- attention projections；
- loss。

这种语言只是帮助理解 computation role。

---

# 121. 为什么 Query Embeddings 能学出 Temporal Role？

Backprop。

对于第 \(i\) 个 slot：

\[
\hat a_i
\]

总和：

\[
a_{t+i}
\]

比较。

所以：

\[
\frac{\partial L}{\partial q_i}
\]

持续告诉 query embedding：

> 怎样变化能让第 \(i\) 个 future action预测更准。

---

# 122. 因此 Query Embeddings 本身也是 Trainable Policy Parameters

Released code：

```python
nn.Embedding
```

默认：

\[
requires\_grad=True
\]

所以：

> 它们和 Transformer weights一起由 L1 gradient训练。

---

# 123. Fixed Sinusoidal Paper Variant 则没有 Query Parameter Gradient

如果 query positions是 fixed sinusoidal：

> query values不是 learnable parameters。

Loss只能调整：

- W_Q；
- Decoder layers；
- Action head；

等，

去适应这些固定 positional codes。

---

# 124. 两种方案都能打破 Slot Symmetry

所以核心要求不是：

> “Query必须 learnable。”

而是：

\[
\boxed{
\text{different slots need distinguishable positional identities}
}
\]

---

# 125. 为什么 ACT 不直接用整数 i 当 Query？

Transformer内部需要：

\[
512\text{-D}
\]

representations。

整数：

\[
i
\]

必须先映射成：

> vector representation。

Sinusoidal / learned embedding：

> 都是这种映射方式。

---

# 126. 为什么 Position Embedding 不直接就是 Action？

因为 position只是：

> “我要预测哪个 horizon index”。

动作值还依赖：

- observation；
- joint state；
- task state；
- z；
- entire decoder computation。

所以：

\[
\boxed{
\text{position identity}
+
\text{context}
\rightarrow
\text{action}
}
\]

---

# 127. 为什么 Query Slot 数量 = Chunk Size？

因为最终希望一一对应：

\[
\boxed{
1\ decoder\ slot
\leftrightarrow
1\ future\ action
}
\]

所以：

\[
N_q=k
\]

是自然设计。

---

# 128. 官方 Training Config 直接证明

`imitate_episodes.py`：

```python
policy_config = {
    ...
    'num_queries':
        args['chunk_size'],
    ...
}
```

因此：

\[
\boxed{
\text{DETR 的 num\_queries interface}
\rightarrow
\text{ACT 的 chunk size}
}
\]

---

# 129. 这其实是非常漂亮的 Architecture Reuse

把 DETR 原本的：

\[
N\text{ object slots}
\]

重新解释为：

\[
k\text{ temporal action slots}
\]

就自然得到：

> parallel action chunk decoder。

---

# 130. DETR 为什么是 ACT 很自然的母体？

因为 ACT 正好需要：

1. 固定数量输出 slots；
2. 输出之间能相互协调；
3. 每个 output都能读取同一个 rich encoder memory；
4. 所有 outputs可以并行生成。

DETR-style decoder恰好都提供。

---

# 131. 原始 Language Transformer Decoder 反而有一个不必要限制

标准 autoregressive translation：

> causal mask + previous target tokens。

但 ACT不想：

> 一个动作一个动作 sequential decode。

它想一次：

\[
k
\]

个。

DETR已经证明：

> Transformer Decoder完全可以用 query slots进行 non-autoregressive parallel decoding。

---

# 132. 所以 ACT 借的不是 Object Detection 本身

它借的是：

\[
\boxed{
\text{DETR 对 Transformer Decoder 的重新用途}
}
\]

即：

> “Decoder Query Slots 可以代表待预测的 structured outputs，而不一定是语言 token。”

---

# 133. 这是 DETR 对后续很多模型影响很大的原因之一

DETR之后，大量视觉/多模态架构都采用：

> query-based decoding / latent query slots

思想。

Query不一定代表：

- object；
- word；

它可以代表：

- mask；
- track；
- region；
- action；
- latent slot；

等。

---

# 134. Query-Based Decoder 是一种非常通用的接口

可以抽象成：

\[
\boxed{
\text{Context Memory}
+
\text{Output Queries}
\rightarrow
\text{Structured Outputs}
}
\]

这其实比“DETR是检测模型”更重要。

---

# 135. ACT 正是在这个抽象上工作

Context：

\[
M=
Encoder(
images,
qpos,
z
)
\]

Queries：

\[
Q_{slots}
=
[
q_0,\ldots,q_{k-1}
]
\]

Decoder：

\[
H=
Decoder(
Q_{slots},
M
)
\]

Readout：

\[
\hat A=
ActionHead(H)
\]

---

# 136. 形式化写法

设：

\[
M
\in
\mathbb R^{N_m\times d}
\]

Action slot embeddings：

\[
P
=
[
p_0,\ldots,p_{k-1}
]
\in
\mathbb R^{k\times d}
\]

初始 content：

\[
H^{(0)}=0
\]

---

# 137. 第 l 层 Self-Attention

概念上：

\[
\tilde H^{(l)}
=
SelfAttn(
H^{(l)}+P
)
\]

实际还包括：

- residual；
- norm；
- Q/K/V projections。

---

# 138. Cross-Attention

\[
\hat H^{(l)}
=
CrossAttn(
\tilde H^{(l)}+P,
M+P_M,
M
)
\]

其中：

\[
P_M
\]

是 memory positional information。

---

# 139. FFN

\[
H^{(l+1)}
=
FFN(
\hat H^{(l)}
)
\]

加 residual/norm。

经过多层：

\[
H^{(L)}
\in
\mathbb R^{k\times d}
\]

---

# 140. Action Readout

\[
\boxed{
\hat A
=
H^{(L)}
W_A^\top+b_A
}
\]

shape：

\[
[k,d]
[d,14]
\rightarrow
[k,14]
\]

---

# 141. 每一行对应一个 Future Action Slot

\[
\hat A_i
=
\hat a_{t+i}
\]

因此：

\[
\boxed{
\text{query-slot axis}
=
\text{future-time axis}
}
\]

这是 ACT 对 DETR slot axis 的重新语义化。

---

# 142. 为什么 Query Slot Self-Attention 不破坏 Time Identity？

因为每层进行 self-attention时：

> query positional embedding会重新加进 Q/K side。

所以即使 hidden contents互相混合，

每个 slot仍带有：

> 自己的位置 identity。

---

# 143. Official Decoder Code 每层都使用 query_pos

```python
q = k =
    self.with_pos_embed(
        tgt,
        query_pos
    )
```

Cross-Attention也：

```python
query =
    tgt + query_pos
```

所以 query identity：

> 不是只在 decoder最开始加一次后就完全忘掉。

---

# 144. 这来自 DETR 的 Positional-Encoding Design

DETR论文也特别研究：

> output positional encodings应该只在 decoder input加一次，还是在每层 attention中使用。

其 baseline把这些 positional encodings：

> 直接送入每层 attention。

ACT code沿用了这一 DETR-style实现。

---

# 145. 为什么 `tgt` 和 `query_pos` 要分开？

这是一种很好的 architecture decomposition：

### `tgt`

\[
\boxed{
\text{content state}
}
\]

随 decoder layers不断更新。

### `query_pos`

\[
\boxed{
\text{slot identity}
}
\]

跨 layers保持。

---

# 146. 类似于“内容”和“地址”分离

可以类比：

```text
query_pos = 这个邮箱格子的编号
tgt       = 这个格子当前装了什么内容
```

一开始：

```text
格子内容为空
```

但：

```text
格子编号不同
```

然后每一层从 memory读取内容填进去。

---

# 147. 这个类比很适合解释 tgt=0

\[
tgt=0
\]

不是：

> “Decoder什么都不知道，所以不能工作。”

而是：

> “所有 output slots一开始没有 sample-specific content。”

但是：

- slot identity存在；
- observation memory存在。

Cross-Attention负责：

> 把 context写进 slots。

---

# 148. 为什么不直接把 Query Embedding 当 tgt？

有些 Transformer实现确实可以采用不同 parameterization。

DETR-style code选择：

\[
\boxed{
\text{content}
+
\text{position}
}
\]

分离，

使 positional information可以在每层 Q/K computation中重新注入。

---

# 149. 这和普通 Transformer Input Token + Position Embedding 很像

普通 encoder：

\[
token\_content
+
position
\]

DETR decoder：

\[
slot\_content
+
query\_position
\]

只是 decoder初始：

\[
slot\_content=0
\]

---

# 150. ACT Action Query 其实因此更像“Output Positional Encoding”

这是比“Action Query”更精确的某一层理解。

DETR论文自己就把 object queries称为：

> output positional encodings。

ACT paper也把 action decoder输入画成：

> position embeddings。

---

# 151. 所以不要把 Query 神秘化

它不是：

- 一个问题句子；
- 一个动作候选；
- 一个 latent action；
- 一个 observation；
- 一个 future action GT。

它首先是：

\[
\boxed{
\text{output-slot positional identity}
}
\]

---

# 152. 什么时候它开始有 Sample-Specific 内容？

第一次 Cross-Attention之后。

从：

\[
memory
\]

读取 Value，

得到：

\[
h_i^{(1)}
\]

之后：

> slot hidden state才带有当前 sample的信息。

---

# 153. 后续层不断 Refine

第2层：

> slot已经不是空的。

Self-Attention：

> slots之间协调。

Cross-Attention：

> 再次读取 observation memory。

FFN：

> 重新加工 feature。

因此：

\[
\boxed{
\text{decoder is iterative slot refinement}
}
\]

---

# 154. DETR 中也是这种 Iterative Refinement Intuition

每层 decoder：

> 逐渐把 generic prediction slots变成 object-specific output embeddings。

ACT：

> 逐渐把 generic temporal slots变成 observation-conditioned action representations。

---

# 155. 这是一条非常漂亮的对应关系

\[
\boxed{
\text{generic slot}
\rightarrow
\text{contextualized slot}
\rightarrow
\text{task-specific output}
}
\]

DETR和ACT共享这个宏观范式。

---

# 156. 为什么 ACT 用 7 Decoder Layers？

论文 Table III给出：

\[
\boxed{
7
}
\]

个 decoder layers。

这意味着 action slot不是：

> 一次 cross-attention就直接输出。

而是：

> 多轮 self/cross-attention + FFN refinement。

---

# 157. Query Embedding 自己是不是在每层改变？

Released code中：

\[
query\_embed.weight
\]

作为：

> 固定参数值

在一次 forward的不同 decoder layers中重复使用。

它不会：

> layer 1输出后变成新的 query embedding parameter。

改变的是：

\[
tgt / hidden\ state
\]

---

# 158. 训练 across steps 时 Query Parameter 会改变

Optimizer更新：

\[
query\_embed.weight
\]

所以不同 training steps：

> query embedding参数逐渐学习。

但一次 forward中：

> 它作为 positional identity跨 decoder layers重复使用。

---

# 159. ACT 中 Action Query 是否取决于当前 Observation？

query embedding本身：

> 不取决于 observation。

它是全局 parameter / fixed code。

但是最终 Attention Q：

\[
(tgt+query\_pos)W_Q
\]

从第二层起的：

\[
tgt
\]

已经 observation-conditioned。

所以真正 Q：

> 会随 sample变化。

---

# 160. 这又说明为什么不能把 Query Embedding = Attention Q

第一层：

> 很大程度由 query_pos决定。

后续层：

\[
Q
\]

同时包含：

- slot positional identity；
- accumulated contextual content。

---

# 161. 为什么 Self-Attention 中 Action Slots 可以 Asymmetric？

即使 slot \(i,j\)都来自同一 sequence，

\[
Q_i
\]

和：

\[
K_j
\]

经过不同 projection。

所以：

\[
score(i\rightarrow j)
\]

一般不等于：

\[
score(j\rightarrow i)
\]

因此 temporal slot interactions可以是：

> directional learned relations。

---

# 162. 没有 Causal Mask 不等于 Attention Matrix 必须对称

它只是说：

> 所有 slot pairs都允许参与。

真正 attention weights：

> 仍然由 learned Q/K决定，

通常不对称。

---

# 163. ACT 能不能学习“前一步影响后一步”？

Architecture允许。

例如后续 slot：

\[
i+1
\]

可高度读取：

\[
i
\]

的 hidden state。

---

# 164. ACT 也能学习“后一步反过来约束前一步”

因为没有 causal mask，

slot \(i\)也可以读取：

\[
i+1
\]

这可以理解为：

> 整个 chunk共同协调。

这不是执行时未来信息泄露，

因为两个 slot都只是：

> 当前 observation条件下同时生成的 plan variables。

---

# 165. 这和 Trajectory Optimization 有一点直觉相似

在 trajectory optimization 中：

> 整段 trajectory可以作为一个整体优化。

ACT并不是显式 trajectory optimizer，

但 non-causal action-slot self-attention也体现：

> “整段输出共同协调”

的结构直觉。

这只能作为类比。

---

# 166. ACT Action Chunk 是 Open-Loop 的吗？

如果一次预测：

\[
k
\]

步然后全执行完才重新观察，

那是 naive chunk execution意义上的：

> chunk-level open-loop segment。

---

# 167. 但 Final ACT + Temporal Ensemble 每个 Timestep 都重新 Query

所以最终 policy execution：

> 仍不断接收新 observation，

并融合 overlapping chunks。

因此不能说：

> “ACT完全100步开环。”

我们在 [ACT Inference](./inference.md) 已详细讲过。

---

# 168. Decoder Parallelism 和 Temporal Ensemble 是两个不同维度

### Decoder Parallelism

一次 policy call：

> 同时产生 \(k\) 个 future action slots。

### Temporal Ensemble

不同 policy calls：

> 对同一 execution timestep产生多个 overlapping predictions，再融合。

不要混淆。

---

# 169. DETR 的 Parallel Output 解决什么？

主要：

> structured set prediction效率与设计简化。

---

# 170. ACT 的 Parallel Output 又解决什么？

与 Action Chunking目标结合：

> 一次产生未来一段动作，

从而缩短 effective horizon并捕捉 temporally coherent behavior。

---

# 171. 所以 DETR Decoder 只是 ACT Action Chunking 的实现工具之一

ACT真正算法思想包括：

- action chunking；
- CVAE；
- temporal ensemble。

DETR-style Transformer Decoder负责：

\[
\boxed{
\text{how to represent and predict the whole chunk in parallel}
}
\]

---

# 172. 不要说“ACT = DETR for actions”

作为一句类比：

> 有启发。

但严格来说不够。

因为 ACT还加入：

- CVAE latent；
- multiple camera + proprioception；
- temporal target semantics；
- imitation-learning objective；
- temporal ensemble。

---

# 173. 更准确的说法

\[
\boxed{
\text{ACT reuses a DETR-style query-based Transformer decoder to parameterize action-chunk prediction.}
}
\]

这比：

> “ACT就是动作版DETR”

严谨得多。

---

# 174. DETR 与 ACT 的 Loss 根本不同

DETR：

- Hungarian matching；
- classification；
- box L1；
- GIoU。

ACT：

- direct aligned action L1；
- CVAE KL。

所以即使 decoder skeleton相似，

训练 pressure完全不同。

---

# 175. Loss 最终决定 Query 学什么

DETR query parameter gradient来自：

> object detection losses。

ACT query parameter gradient来自：

> action reconstruction loss。

所以它们最终学习的 representations：

> 必然服务不同任务。

---

# 176. 这也是“Architecture 迁移”的真正含义

一个 architecture pattern：

> 可以迁移到新 domain。

但其 learned semantics：

> 由新数据 + 新 objective重新产生。

---

# 177. ACT Decoder 是不是 Set Prediction？

严格说：

\[
\boxed{
\text{不是。}
}
\]

它输出：

> ordered action sequence。

如果把两个 output slots交换：

\[
\hat a_0\leftrightarrow\hat a_5
\]

一般会产生完全不同的 L1 loss。

所以 objective：

> 不 permutation-invariant。

---

# 178. DETR Loss 则必须对 Prediction Order 不敏感

通过 Hungarian matching：

> 只要 prediction set正确，

slot排列本身不重要。

这是两者数学任务定义的根本不同。

---

# 179. 一个非常有用的对比公式

## DETR

预测：

\[
\hat Y=
\{
\hat y_1,\ldots,\hat y_N
\}
\]

Loss：

\[
\boxed{
L_{set}
=
\min_{\sigma}
\sum_i
\ell(
y_i,
\hat y_{\sigma(i)}
)
}
\]

忽略具体检测loss细节。

---

# 180. ACT

预测：

\[
\hat A=
[
\hat a_0,\ldots,\hat a_{k-1}
]
\]

Loss：

\[
\boxed{
L_{act}
=
\sum_i
\ell(
a_i,
\hat a_i
)
}
\]

带 padding mask/reduction。

这里没有：

\[
\min_\sigma
\]

---

# 181. 这决定 Query Slot Identity 的性质不同

DETR：

> slot identity用于打破对称和并行分工，但目标 assignment可变化。

ACT：

> slot identity本身就与 temporal target index绑定。

---

# 182. 从这个角度看，ACT Query 更像 Learned Positional Token

尤其 released implementation：

\[
q_i
\]

训练成：

> 第 \(i\) 个 output horizon位置的专属 decoder position code。

---

# 183. 为什么仍然叫 Query 而不是 Position Embedding？

主要因为它进入：

> Transformer Decoder query-side attention path。

而代码直接继承 DETR命名。

从概念上：

> “action query” 与 “output positional embedding” 两种称呼都有帮助。

---

# 184. Query 数量变化会怎样？

如果训练：

\[
k=100
\]

`query_embed`：

\[
[100,512]
\]

如果改：

\[
k=50
\]

则：

\[
[50,512]
\]

整个 action horizon也变为50 slots。

---

# 185. 不能随便在训练后把 Learned Query Table 延长到 200

因为：

> slots 100–199没有学过 embeddings。

Fixed sinusoidal query的理论优势之一：

> 可以为新 index计算 positional code。

但整个 ACT模型是否能可靠外推到更长 horizon：

> 仍远远不是仅靠 positional code就能保证。

---

# 186. 所以不要过度解读 Sinusoidal Extrapolation

即使位置编码能生成：

\[
i>99
\]

的 vector，

模型：

- decoder；
- loss；
- data；

都没有训练过那些 horizons。

因此：

\[
\boxed{
\text{positional encoding extrapolates}
\not\Rightarrow
\text{policy behavior extrapolates}
}
\]

---

# 187. Query-Based Parallel Decoder 的优点

### 1. 并行

所有 action positions一起计算。

### 2. Slot interaction

Self-Attention允许整段协调。

### 3. Shared context

所有 slots读取统一 observation memory。

### 4. Shared readout

同一个 action head用于全部 slots。

### 5. Fixed tensor interface

输出稳定：

\[
[B,k,14]
\]

便于训练与部署。

---

# 188. 它的潜在限制呢？

固定：

\[
k
\]

意味着：

> horizon architecture通常固定。

另外一次性预测整个 chunk：

> 不会在 chunk内部获得新的真实 observations。

Final ACT通过：

> 每 timestep重新query + Temporal Ensemble

缓解这一点。

---

# 189. 另一个限制：长期 Horizon 预测难度

越远的 slot：

\[
a_{t+i}
\]

可能越不确定，

因为环境未来会受到：

- contact；
- object motion；
- accumulated action errors；

影响。

所以 fixed parallel action chunk也存在：

> horizon uncertainty。

---

# 190. 这也是后续 Robot Policy 方法继续演化的原因

后续方法会探索：

- diffusion over action chunks；
- receding horizon；
- VLA；
- richer multimodal conditioning；
- hierarchical planning。

ACT不是终点。

---

# 191. 但 Query-Based Parallel Decoding 是理解 ACT 的一个关键 Architecture Insight

如果这一点没懂，

很多代码看起来会很诡异：

```python
nn.Embedding(num_queries, hidden_dim)
```

```python
tgt = zeros_like(query_embed)
```

```python
action_head(hs)
```

一旦理解 DETR，

它们就非常自然。

---

# 192. 重新看 Official ACT Code

```python
self.query_embed =
    nn.Embedding(
        num_queries,
        hidden_dim
    )
```

现在我们知道：

> 创建 \(k\) 个 output slot identities。

---

# 193. 然后

```python
query_embed =
    query_embed
    .unsqueeze(1)
    .repeat(1, bs, 1)
```

从：

\[
[k,512]
\]

变：

\[
[k,B,512]
\]

也就是：

> 每个 batch sample使用同一套 query-slot parameters。

---

# 194. 然后

```python
tgt =
    torch.zeros_like(
        query_embed
    )
```

创建：

\[
[k,B,512]
\]

zero content states。

---

# 195. 然后 Policy Encoder

```python
memory =
    self.encoder(
        src,
        ...
    )
```

产生：

\[
[N_m,B,512]
\]

observation memory。

---

# 196. 然后 Decoder

```python
hs =
    self.decoder(
        tgt,
        memory,
        pos=pos_embed,
        query_pos=query_embed
    )
```

把：

> empty slot contents + slot identities

通过：

> self/cross-attention

变成：

\[
[k,B,512]
\]

contextualized action-slot representations。

---

# 197. 最后

```python
a_hat =
    self.action_head(
        hs
    )
```

把每个：

\[
512
\]

维 slot表示变成：

\[
14
\]

维 robot action。

---

# 198. 一条完整 Shape 链

假设：

\[
B=8
\]

\[
k=100
\]

\[
d=512
\]

Query table：

\[
\boxed{
[100,512]
}
\]

Batch repeat：

\[
\boxed{
[100,8,512]
}
\]

---

# 199. tgt

\[
\boxed{
[100,8,512]
}
\]

全0。

---

# 200. Memory

例如：

\[
N_m=1202
\]

则：

\[
\boxed{
[1202,8,512]
}
\]

---

# 201. Cross-Attention

每 head：

\[
Q:
[8,8,100,64]
\]

\[
K,V:
[8,8,1202,64]
\]

attention matrix：

\[
\boxed{
[8,8,100,1202]
}
\]

其中第一个8：

> batch；

第二个8：

> heads。

---

# 202. Decoder Output

最终：

\[
\boxed{
[100,8,512]
}
\]

根据代码后续 transpose等接口，

最终 policy层观察到：

\[
[B,k,512]
\]

---

# 203. Action Head

\[
[8,100,512]
\rightarrow
[8,100,14]
\]

所以：

\[
\boxed{
a_{hat}:
[B,k,14]
}
\]

---

# 204. 这整个 Pipeline 没有一步需要 Previous Predicted Action

所以它确实是：

\[
\boxed{
\text{non-autoregressive action chunk decoder}
}
\]

---

# 205. 它是不是“一次 Matrix 直接算出所有动作”？

不是这么简单。

虽然并行，

但 decoder中：

- multi-head self-attention；
- cross-attention；
- FFN；

会多层迭代。

所以是：

> parallel multi-layer structured decoding。

---

# 206. 为什么 Parallel Decoder 还能表达 Action Correlation？

因为 action slot hidden states：

> 通过 Self-Attention彼此耦合。

输出 joint distribution虽然在 implementation最终通过 deterministic decoder/readout产生，

但内部 representation不是：

> 每个 slot完全独立 MLP。

---

# 207. CVAE 又给整个 Chunk 一个 Shared Latent z

所以训练时：

\[
z
\]

对整个 action chunk提供：

> shared style condition。

这进一步让 chunk prediction：

> 在全局 latent层面保持一致性。

---

# 208. 所以 ACT 有两种跨时间协调机制

### Decoder Self-Attention

\[
\boxed{
\text{slot-to-slot interaction}
}
\]

### Shared CVAE z

\[
\boxed{
\text{chunk-level latent conditioning}
}
\]

二者作用不同。

---

# 209. 再加 Temporal Ensemble

Inference时还有：

\[
\boxed{
\text{cross-query-time prediction aggregation}
}
\]

因此 ACT的 temporal coherence来自：

> 多个层次。

---

# 210. 不要把三者混在一起

### Action Query Slot

表示：

> chunk内部的输出位置。

### Latent z

表示：

> demonstration/action style variation。

### Temporal Ensemble

融合：

> 不同 policy calls对同一 execution timestep的预测。

---

# 211. 为什么 Object Query 这个思想适合迁移到 Actions？

因为它真正抽象出来的不是：

> “Object”。

而是：

\[
\boxed{
\text{a finite bank of output slots that retrieve from shared context}
}
\]

只要任务能定义：

> 一组固定数量的结构化 outputs，

就可能采用类似设计。

---

# 212. 这就是读 Architecture 最值得学习的方法

不要只记：

> “DETR用Object Query。”

而问：

\[
\boxed{
\text{这个结构解决的抽象问题是什么？}
}
\]

答案：

> **如何让 Transformer Decoder 用多个彼此可区分、可交互的 output slots，并行地从共享 memory中生成多个结构化结果。**

---

# 213. ACT 正好有同一个抽象问题

它需要：

> 从一个 observation memory中，

并行生成：

\[
k
\]

个彼此关联的 future actions。

所以 DETR skeleton非常自然。

---

# 214. Common Misconception 1：ACT 自己发明了 Query Decoder

**不准确。**

其 released Transformer实现明显继承 DETR-style code与设计。

---

# 215. Common Misconception 2：Object Query 就是 Attention 的最终 Q Vector

**错误。**

它是：

> query positional embedding / slot identity。

真正 Q还经过：

\[
(tgt+query\_pos)W_Q
\]

---

# 216. Common Misconception 3：Object Query 里面已经存着检测到的 Object

**错误。**

sample-specific object content主要通过：

> Cross-Attention从 image memory进入 decoder state。

---

# 217. Common Misconception 4：Action Query 里面已经存着未来动作

**错误。**

它主要标识：

> future output slot。

---

# 218. Common Misconception 5：`tgt=0` 意味着 Decoder 没输入，所以不可能工作

**错误。**

Query positional embeddings与encoder memory都存在。

---

# 219. Common Misconception 6：第一层 tgt=0，所以所有 Slots 一样

**错误。**

\[
query\_pos_i
\]

不同。

因此 Q/K positional sources不同。

---

# 220. Common Misconception 7：ACT 的 100 Queries 对应 100 个 Objects

**错误。**

代码注释残留 DETR vocabulary。

在 ACT：

\[
\boxed{
num\_queries=chunk\_size
}
\]

---

# 221. Common Misconception 8：ACT 就是 DETR 把 Box Head 换成 Action Head

**过度简化。**

ACT还有：

- CVAE；
- multimodal encoder memory；
- ordered targets；
- temporal ensemble；
- imitation learning。

---

# 222. Common Misconception 9：DETR 和 ACT Query Slots 都是 Ordered

**错误。**

DETR目标是 set prediction，

ACT目标是 temporal sequence。

---

# 223. Common Misconception 10：ACT 也需要 Hungarian Matching

**错误。**

Action targets有固定 temporal alignment。

---

# 224. Common Misconception 11：第 i 个 ACT Query 天生知道自己是第 i 步

如果是 learned embedding：

> initialization时不知道任务语义。

它通过：

> index-specific supervision

逐渐学到该 temporal role。

---

# 225. Common Misconception 12：Action Query 等于 Positional Encoding，没有任何学习

Paper variant偏向 fixed sinusoidal描述，

但 released official code：

\[
\boxed{
nn.Embedding
}
\]

是 learned。

必须区分 paper/code。

---

# 226. Common Misconception 13：ACT Official Paper 和 Code 对 Query Embedding 完全一致

**并不一致。**

论文 Appendix描述 fixed sinusoidal queries；

released code使用 learned query embeddings。

---

# 227. Common Misconception 14：Action Decoder 是 Autoregressive

**错误。**

所有 \(k\) slots并行产生。

---

# 228. Common Misconception 15：Parallel 表示 Actions 之间完全没有 Interaction

**错误。**

Decoder Self-Attention使 slots彼此交流。

---

# 229. Common Misconception 16：没有 Causal Mask 是 Bug

**不是。**

ACT目标就是：

> parallel coherent chunk prediction，

不是 causal next-token generation。

---

# 230. Common Misconception 17：没有 Causal Mask 意味着用了未来真实 Observation

**错误。**

slots只交流 model hidden states，

没有未来环境 observation输入。

---

# 231. Common Misconception 18：Action Slot i 会输入 Ground-Truth Action i

**错误。**

Policy Decoder的 tgt初始为0。

Ground-truth actions用于：

- CVAE Encoder；
- loss target。

---

# 232. Common Misconception 19：Query Embedding 每个 Observation 都重新生成

**错误。**

它是 shared parameter / fixed positional code。

---

# 233. Common Misconception 20：同一个 Query 在不同 Sample 产生同一个 Action

**错误。**

它读取不同：

\[
memory(o_t,z)
\]

所以输出不同。

---

# 234. Common Misconception 21：每个 Slot 有独立 Action Head

**错误。**

所有 slots共享：

\[
Linear(512,14)
\]

---

# 235. Common Misconception 22：Shared Action Head 意味着所有 Slots 输出一样

**错误。**

Hidden states不同。

---

# 236. Common Misconception 23：Action Query Self-Attention 是为了读取 Images

**错误。**

读取 observation memory主要是：

> Cross-Attention。

Self-Attention负责：

> action slots之间 interaction。

---

# 237. Common Misconception 24：Cross-Attention 只读取 Visual Tokens

**错误。**

ACT policy memory还包含：

- qpos；
- latent z；

context。

---

# 238. Common Misconception 25：Decoder Query Position 与 Image Position Encoding 是同一种位置

它们都是 positional signals，

但分别表示：

- output horizon slot；
- observation spatial position。

---

# 239. Common Misconception 26：Fixed Query Count 代表执行时一定一次执行完 k 步

**错误。**

Final ACT使用 Temporal Ensemble时：

> 每 timestep重新query policy。

---

# 240. Common Misconception 27：Query-Based Decoder = Temporal Ensemble

**完全不同。**

一个是：

> 单次 forward内部的output architecture。

一个是：

> 多个 forward predictions的execution-time aggregation。

---

# 241. Common Misconception 28：ACT Query 和 CVAE z 是同一种 latent

**错误。**

Query：

> output slot identity。

z：

> latent style variable。

---

# 242. Common Misconception 29：DETR 的 Object Query 有固定 Category Meaning

**不保证。**

其 assignment由 set loss/matching动态决定。

---

# 243. Common Misconception 30：只要有 Query Slots，就自动能预测 Sequence

**错误。**

还需要：

- positional distinction；
- correct target alignment；
- decoder architecture；
- loss；
- data。

Query slots只是 structural mechanism。

---

# 244. 用一张图记住 DETR

```text
Image
  │
  ▼
CNN
  │
  ▼
Encoder Memory
  │
  │          Object Query Slots
  │           q₁ q₂ ... qN
  │             │
  └─────────────┼───────┐
                ▼       │
         Transformer Decoder
                │
                ▼
      h₁ h₂ ... hN
       │  │      │
       ▼  ▼      ▼
    class+box predictions

Training:
Hungarian matching
↓
unordered set supervision
```

---

# 245. 用一张图记住 ACT

```text
Images + qpos + z
        │
        ▼
 Policy Encoder
        │
        ▼
Observation Memory
        │
        │       Future Action Slots
        │        q₀ q₁ ... q{k-1}
        │              │
        └──────────────┼─────────┐
                       ▼         │
              Transformer Decoder
                       │
                       ▼
               h₀ h₁ ... h{k-1}
                │  │       │
                ▼  ▼       ▼
              shared Action Head
                       │
                       ▼
             âₜ âₜ₊₁ ... âₜ₊ₖ₋₁

Training:
direct temporal alignment
↓
L1 per corresponding time slot
```

---

# 246. 一张图看出真正的继承关系

```text
DETR abstraction:

context memory
+
N output queries
↓
parallel decoder
↓
N structured outputs


             │
             │ reuse abstraction
             ▼


ACT:

observation memory
+
k future-action queries
↓
parallel decoder
↓
k ordered actions
```

---

# 247. 最关键的变化不是 Object → Action 这两个名词

真正变化的是：

\[
\boxed{
\text{set-slot semantics}
\rightarrow
\text{temporal-slot semantics}
}
\]

也就是说：

DETR：

> “这些 slots共同覆盖当前图像中的object set。”

ACT：

> “这些 slots依次对应未来 action horizon中的各个位置。”

---

# 248. 如果只记一句话理解 Object Query

> **DETR 的 Object Query 并不是“某个物体的特征”，而是一组彼此不同的 learned output positional embeddings，用来给 decoder 创建固定数量、彼此可区分的 prediction slots；这些 slots再通过 cross-attention从 encoder image memory中获取当前图片的 object-specific content。**

---

# 249. 如果只记一句话理解 Action Query

> **ACT 的 Action Query 不是未来动作本身，而是 future-action output slot 的位置身份：第 \(i\) 个 slot通过 query positional signal与其他 slot区分，再从当前 observation memory中 cross-attend出与该 horizon position相关的信息，并通过多层 self/cross-attention形成512维 action-slot representation，最后由共享 `Linear(512,14)`读出第 \(i\) 个未来动作。**

---

# 250. 如果只记一句话理解 tgt=0

> **`tgt=0` 只表示 decoder slots在最初没有 sample-specific content，并不意味着 decoder没有信息：不同 query positional embeddings提供 slot identity，encoder memory提供当前 observation content；第一轮 cross-attention正是把 memory中的信息写入这些空的 output slots，此后 decoder hidden states便不再是0。**

---

# 251. 如果只记一句话理解 ACT 为什么没有 Causal Mask

> **ACT 并不是把未来动作按 \(a_t\rightarrow a_{t+1}\rightarrow\cdots\) 的 autoregressive factorization逐步生成，而是把整个 action chunk当作一组有序输出变量并行预测；所有 action slots只依赖当前 observation/context而不接收未来真实动作或未来真实 observation，因此它们可以通过双向 self-attention共同协调整个 chunk，而不需要语言模型式 causal mask。**

---

# 252. 如果只记一句话理解 DETR → ACT

> **ACT 最值得从 DETR 借来的不是“物体检测技术”，而是 query-based parallel structured decoding：DETR证明了 Transformer Decoder可以用一组output queries作为空的结构化预测槽位，让这些槽位彼此self-attend并共同cross-attend共享context memory，再并行输出多个结果；ACT把这套机制从无序object slots改造成有序future-action slots，并用直接时间索引监督代替DETR的Hungarian set matching。**

---

# 253. 现在再回头看 ACT Decoder，就不再神秘

你看到：

```python
self.query_embed =
    nn.Embedding(
        num_queries,
        hidden_dim
    )
```

应该想到：

\[
\boxed{
k\text{ 个 future output slot identities}
}
\]

---

看到：

```python
tgt =
    torch.zeros_like(
        query_embed
    )
```

应该想到：

\[
\boxed{
\text{empty initial slot contents}
}
\]

---

看到：

```python
query =
    tgt + query_pos
```

应该想到：

\[
\boxed{
\text{current slot content + persistent slot identity}
}
\]

---

看到：

```python
key =
    memory + pos
```

应该想到：

\[
\boxed{
\text{context content + context position}
}
\]

---

看到：

```python
value =
    memory
```

应该想到：

\[
\boxed{
\text{actual context information being retrieved}
}
\]

---

看到：

```python
action_head(hs)
```

应该想到：

\[
\boxed{
\text{one shared readout applied to every future-action slot}
}
\]

---

# 254. ACT Architecture 到这里发生了一次重要升级

之前我们已经知道：

> “Decoder有100个Action Queries。”

现在应该升级成：

\[
\boxed{
\text{ACT uses a DETR-style non-autoregressive query-slot decoder,}
}
\]

其中：

- slot positional identities指定未来输出位置；
- `tgt`保存逐层更新的slot content；
- self-attention协调整段动作；
- cross-attention让每个future slot读取observation memory；
- shared action head把slot representation读出成14-D joint target；
- ordered L1 supervision把slot \(i\)绑定到future timestep \(t+i\)。

这才是对 Action Query 完整的理解。

---

# 255. 下一篇建议：Imitation Learning 的 Distribution Shift

现在 Query Decoder已经真正讲通，

下一块更值得进入的是：

> **为什么 ACT 即使 architecture很好，仍然必须面对 imitation learning本身的根本问题？**

下一篇建议：

> **[Behavior Cloning：为什么训练集上动作预测很准，机器人 Rollout 还是会崩？](../imitation-learning/behavior-cloning-distribution-shift.md)**

它会正式从：

\[
p_{\mathcal D}(s)
\]

和：

\[
p_{\pi}(s)
\]

开始，深入解释：

- Behavior Cloning；
- supervised imitation；
- train-state distribution；
- policy-induced state distribution；
- covariate shift；
- compounding error；
- \(O(T^2\epsilon)\) 一类经典误差累积直觉；
- DAgger为什么被提出；
- closed-loop control为什么仍会偏离demonstrations；
- ACT Action Chunking究竟缓解了什么；
- Action Chunking又没有解决什么；
- Temporal Ensemble与distribution shift的关系；
- 为什么 VLA / Diffusion Policy等后续方法仍必须面对policy rollout distribution。

---

## Primary Source：DETR

Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko.

**End-to-End Object Detection with Transformers.**  
ECCV 2020.

- arXiv: https://arxiv.org/abs/2005.12872
- PDF: https://arxiv.org/pdf/2005.12872
- Official repository: https://github.com/facebookresearch/detr

论文核心事实：

### Direct Set Prediction

DETR把 Object Detection定义为：

\[
\boxed{
\text{direct set prediction}
}
\]

并使用：

> bipartite matching loss

把prediction slots与GT objects一一匹配。

---

### Learned Object Queries

论文 Figure 2 / Section 3说明：

> Transformer decoder接收固定数量的 learned positional embeddings，称为 object queries。

这些 query slots经过 decoder后：

> 被转换为 output embeddings，

再分别预测：

- class；
- bounding box。

---

### Parallel Decoding

DETR明确区别于原始 Transformer的autoregressive sequence generation：

\[
\boxed{
\text{N objects are decoded in parallel}
}
\]

因此 DETR 是理解 ACT parallel action decoder的重要直接前身。

---

### `tgt=0` / Query Position Separation

DETR Appendix A.3说明：

> Decoder receives queries initially set to zero, output positional encoding (object queries), and encoder memory.

所以 DETR-style architecture明确区分：

\[
\boxed{
\text{decoder content state}
}
\]

与：

\[
\boxed{
\text{output positional queries}
}
\]

ACT official code沿用了同样的结构。

---

## ACT Primary Source

Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.

**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.**  
RSS 2023.

- arXiv: https://arxiv.org/abs/2304.13705
- PDF: https://arxiv.org/pdf/2304.13705
- Official repository: https://github.com/tonyzhaozh/act

论文 Section IV-C / Appendix C说明：

- Policy由 ResNet image encoders、Transformer Encoder、Transformer Decoder组成；
- Encoder综合多视角image features、joint positions和style variable \(z\)；
- Encoder outputs作为 Decoder Cross-Attention的 Keys/Values；
- Decoder预测 coherent action sequence；
- Policy输出：
  \[
  k\times14
  \]
  action tensor。

论文 Appendix描述：

> decoder “queries” are fixed sinusoidal embeddings for the first layer。

这一点与released repository存在实现差异，见下文。

---

## ACT Official Code：DETR Lineage

`detr/models/transformer.py`:

https://github.com/tonyzhaozh/act/blob/main/detr/models/transformer.py

文件自己注明：

```text
DETR Transformer class.
Copy-paste from torch.nn.Transformer with modifications
```

其 forward：

```python
query_embed =
    query_embed.unsqueeze(1)
    .repeat(1, bs, 1)

tgt =
    torch.zeros_like(
        query_embed
    )

memory =
    self.encoder(
        src,
        ...
    )

hs =
    self.decoder(
        tgt,
        memory,
        ...,
        pos=pos_embed,
        query_pos=query_embed
    )
```

这与 DETR 的：

\[
\boxed{
\text{zero initial decoder content}
+
\text{query positional embeddings}
+
\text{encoder memory}
}
\]

结构直接对应。

---

## ACT Official Code：Query Embeddings

`detr/models/detr_vae.py`:

https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py

released code：

```python
self.query_embed =
    nn.Embedding(
        num_queries,
        hidden_dim
    )
```

并在 forward：

```python
hs =
    self.transformer(
        src,
        None,
        self.query_embed.weight,
        pos,
        latent_input,
        proprio_input,
        self.additional_pos_embed.weight
    )[0]
```

因此 released official implementation：

\[
\boxed{
\text{uses learned decoder query embeddings}
}
\]

---

## Paper / Code Query Discrepancy

论文 Appendix C：

> queries are fixed sinusoidal embeddings for the first layer.

Official released code：

```python
nn.Embedding(
    num_queries,
    hidden_dim
)
```

因此：

\[
\boxed{
\text{paper description}
\neq
\text{released code in this detail}
}
\]

本文保留这一差异，而不强行把二者写成同一实现。

无论固定还是learned，

共同 architecture role都是：

> **为不同 future action output slots提供可区分的位置身份。**

---

## ACT Official Code：Chunk Size = Number of Queries

`imitate_episodes.py`:

https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py

官方：

```python
policy_config = {
    ...
    'num_queries':
        args['chunk_size'],
    ...
}
```

所以：

\[
\boxed{
N_{queries}
=
k_{chunk}
}
\]

这是理解“DETR prediction slots → ACT future-action slots”的最直接代码连接。

---

## ACT Official Decoder

`detr/models/transformer.py`:

Self-Attention：

```python
q = k =
    self.with_pos_embed(
        tgt,
        query_pos
    )

tgt2 =
    self.self_attn(
        q,
        k,
        value=tgt,
        ...
    )[0]
```

Cross-Attention：

```python
tgt2 =
    self.multihead_attn(
        query =
            tgt + query_pos,

        key =
            memory + pos,

        value =
            memory,
        ...
    )[0]
```

因此：

\[
\boxed{
Qsource=tgt+query\_pos
}
\]

\[
\boxed{
Ksource=memory+pos
}
\]

\[
\boxed{
Vsource=memory
}
\]

并且 standard forward没有传入 causal `tgt_mask`，

所以 action slots：

> 可以彼此双向Self-Attention。

---

## 本文知识连接

### Transformer

- [Transformer](../../deep-learning/transformer.md)
- [Attention](../../deep-learning/attention.md)
- [Q / K / V](../../deep-learning/qkv.md)
- [Self-Attention](../../deep-learning/self-attention.md)
- [Cross-Attention](../../deep-learning/cross-attention.md)
- [Multi-Head Attention](../../deep-learning/multi-head-attention.md)
- [Transformer Decoder](../../deep-learning/transformer-decoder.md)
- [Causal Mask](../../deep-learning/causal-mask.md)
- [Positional Encoding](../../deep-learning/positional-encoding.md)

### Deep Learning

- Embedding
- [Linear Layer](../../deep-learning/linear-layer.md)
- [MLP](../../deep-learning/mlp.md)
- [Backpropagation](../../deep-learning/backpropagation.md)

### Computer Vision

- DETR
- ResNet
- Object Detection
- Hungarian Matching
- Set Prediction

### Robot Learning

- [ACT Architecture](./architecture.md)
- [Action Chunking](./action-chunking.md)
- [CVAE in ACT](./cvae-in-act.md)
- [ACT Training](./training.md)
- [ACT Inference](./inference.md)
- [ACT Complete Data Flow](./complete-data-flow.md)
- [Temporal Ensemble](./temporal-ensemble.md)

### 下一步

- [Behavior Cloning：为什么训练集上动作预测很准，机器人 Rollout 还是会崩？](../imitation-learning/behavior-cloning-distribution-shift.md)
