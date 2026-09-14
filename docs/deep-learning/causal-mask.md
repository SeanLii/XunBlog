---
title: "Causal Mask：为什么 Decoder 训练时不能偷看未来？"
description: "从 autoregressive factorization 与 shifted target sequence 出发，严格理解 causal mask 为什么必要、为什么用 upper-triangular -∞ 屏蔽未来、训练为何仍可并行，以及为什么 ACT 的 action decoder 不需要 causal mask。"
status: reviewed
pageType: concept
canonical: /deep-learning/causal-mask
updated: "2026-09-15"
---

# Causal Mask：为什么 Decoder 训练时不能偷看未来？

在前面的文章里，我们已经知道：

- [Self-Attention](./self-attention.md) 允许一个位置直接读取整条 sequence；
- [Softmax](./softmax.md) 会把 Attention logits 变成 normalized weights；
- [Positional Encoding](./positional-encoding.md) 告诉模型“你在哪里”。

但原始 Transformer Decoder 还必须解决另一个问题：

> **你虽然知道自己在哪里，但你到底允许看谁？**

这就是：

$$
\boxed{
\text{Causal Mask}
}
$$

它解决的不是：

> 位置识别问题。

而是：

> **信息可见性约束。**

如果模型正在预测第：

$$
t
$$

个 token，

它只能依赖：

$$
y_1,\ldots,y_{t-1}
$$

而不能偷看：

$$
y_t,y_{t+1},\ldots
$$

的真实答案。

原始 Transformer 论文 Section 3.1 明确写道：

> Decoder Self-Attention 被修改，使每个位置不能 attend subsequent positions。

Section 3.2.3 又进一步说明：

> 对非法连接，在 Softmax 输入处把对应值设为 $-\infty$。

最终目的就是保留：

$$
\boxed{
\text{autoregressive property}
}
$$

---

## 1. 先从 Autoregressive 到底是什么意思开始

假设要生成一句：

```text
I love robots
```

一个 autoregressive model 不直接一次定义：

$$
P(
I,\ love,\ robots
)
$$

而是根据概率链式法则写成：

$$
\boxed{
P(y_1,\ldots,y_T)
=
\prod_{t=1}^{T}
P(
y_t
\mid
y_1,\ldots,y_{t-1}
)
}
$$

也就是：

$$
P(y_1)
$$

乘：

$$
P(y_2\mid y_1)
$$

乘：

$$
P(y_3\mid y_1,y_2)
$$

……

所以：

> 第 $t$ 个 token 的预测只能使用过去。

这不是 Transformer 独有。

这是 autoregressive sequence modeling 的核心定义。

---

## 2. 为什么不能让模型看未来？

假设训练句子：

```text
I love robots
```

我们要训练模型预测：

```text
love
```

如果模型在这个位置能够直接读取真实：

```text
love
```

甚至后面的：

```text
robots
```

那它可以走捷径。

它根本不需要学习：

> 根据前缀 `I` 推断下一词。

而可以直接从答案位置读取信息。

于是训练 loss 可能非常低，

但推理时真实未来 token 根本不存在。

这就叫：

$$
\boxed{
\text{information leakage}
}
$$

---

## 3. 一个更极端的例子

训练输入如果直接是：

```text
I love robots
```

然后要求每个 position 输出自己：

```text
I love robots
```

又不做 mask，

Self-Attention 可以让：

$$
position\ 2
$$

直接 attend：

$$
position\ 2
$$

甚至直接复制当前目标 token 的 representation。

这根本不是 next-token prediction。

所以目标 sequence 必须经过：

1. shift；
2. causal visibility constraint。

---

## 4. Transformer Decoder 的 Target 为什么要右移一位？

原始 Transformer 论文写道：

> output embeddings are offset by one position。

假设目标：

```text
I love robots <EOS>
```

Decoder 输入通常概念化为：

```text
<BOS> I love robots
```

而 prediction targets：

```text
I love robots <EOS>
```

对齐：

```text
Decoder input:   <BOS>   I      love    robots
Target:          I       love   robots  <EOS>
```

于是 position $t$ 的输入只提供：

> 前一个已知 target token。

---

## 5. 为什么 Shift 之后仍然需要 Causal Mask？

这是最容易误解的一点。

很多人会想：

> “既然输入已经右移一位了，不就没有答案了吗？”

没有这么简单。

例如 Decoder 输入：

```text
<BOS> I love robots
```

我们预测：

```text
I love robots <EOS>
```

考虑预测：

```text
love
```

的那个位置。

它自己的 Decoder input 是：

```text
I
```

很好。

但如果 Self-Attention 没有 mask，

它仍然可以向右看：

```text
love
robots
```

这些 positions。

而：

```text
love
```

正好就是当前要预测的真实答案。

所以：

$$
\boxed{
\text{Shift alone is not enough}
}
$$

必须再加：

$$
\boxed{
\text{Causal Mask}
}
$$

---

## 6. Shift 和 Mask 分别负责什么？

### Shift

让 position $t$ 的本地 input：

> 是上一个 token，而不是当前目标 token。

---

### Causal Mask

保证 position $t$ 的 Self-Attention：

> 只能读取 position $\le t$ 的 decoder inputs。

二者配合，

原论文才能保证：

> 对 position $i$ 的预测只依赖小于 $i$ 的已知 outputs。

---

## 7. 一个 4-Token 的可见性矩阵

假设 Decoder input 有 4 个位置：

$$
0,1,2,3
$$

我们要求：

#### Query 0

只能看：

$$
0
$$

#### Query 1

可以看：

$$
0,1
$$

#### Query 2

可以看：

$$
0,1,2
$$

#### Query 3

可以看：

$$
0,1,2,3
$$

所以 allowed matrix：

$$
\boxed{
\begin{bmatrix}
1&0&0&0\\
1&1&0&0\\
1&1&1&0\\
1&1&1&1
\end{bmatrix}
}
$$

这是一个：

> lower-triangular visibility pattern。

---

## 8. 为什么常说是 Upper-Triangular Mask？

因为“允许矩阵”是下三角，

而真正要 mask 掉的是：

> 主对角线上方。

所以 mask matrix 可以写成：

$$
M=
\begin{bmatrix}
0&-\infty&-\infty&-\infty\\
0&0&-\infty&-\infty\\
0&0&0&-\infty\\
0&0&0&0
\end{bmatrix}
$$

也就是：

> upper triangular part 被设成 $-\infty$。

所以两种说法其实是在讲同一个结构：

```text
allowed region:
lower triangular

masked region:
strict upper triangular
```

---

## 9. Causal Mask 到底加在哪里？

Attention 原始 logits：

$$
S
=
\frac{
QK^\top
}{
\sqrt{d_k}
}
$$

加入 mask：

$$
\boxed{
\tilde S=S+M
}
$$

其中：

$$
M_{ij}
=
\begin{cases}
0,&j\le i\\
-\infty,&j>i
\end{cases}
$$

然后：

$$
\boxed{
A=
softmax(
\tilde S
)
}
$$

---

## 10. 为什么使用 -∞？

因为 Softmax：

$$
softmax(z_j)
=
\frac{
e^{z_j}
}{
\sum_k e^{z_k}
}
$$

如果：

$$
z_j=-\infty
$$

那么：

$$
e^{-\infty}=0
$$

所以：

$$
\boxed{
A_{ij}=0
}
$$

该位置对最终：

$$
AV
$$

完全没有贡献。

这正是我们想要的：

> 非法 future connection 被彻底删除。

---

## 11. 为什么不是 Mask 后再把 Weight 改成 0？

理论上你可以：

1. 先 Softmax；
2. 再把非法位置变 0；
3. 再重新归一化。

但这非常绕。

在 logits 阶段直接加：

$$
-\infty
$$

一次 Softmax 就自动同时完成：

- 禁用；
- 重新归一化。

所以这是更自然的实现。

---

## 12. 一个具体数值例子

假设 Query 1 的 raw logits：

$$
[2,5,10,7]
$$

但它只允许看：

$$
position\ 0,1
$$

mask：

$$
[0,0,-\infty,-\infty]
$$

得到：

$$
[2,5,-\infty,-\infty]
$$

Softmax 后：

$$
\approx
[0.0474,0.9526,0,0]
$$

注意：

虽然未来 position 2 原始 score：

$$
10
$$

甚至最大，

mask 后仍然：

$$
0
$$

所以 Causal Mask 是：

> architecture-level hard information constraint。

不是：

> 希望模型自己学会“不看未来”。

---

## 13. 为什么不能让模型自己学“不要作弊”？

如果训练数据里未来答案就在输入中，

最容易降低 loss 的方式通常就是：

> 利用未来答案。

没有理由期待优化器主动拒绝一个有用捷径。

所以 causality 必须通过 architecture / mask：

> 硬编码。

这和数据泄漏问题一样：

> 不能把答案放进特征，再希望模型自觉不看。

---

## 14. Causal Mask 不是一种 Regularization

它不是：

> “防止 overfitting 的随机技巧。”

它定义了模型的条件依赖结构：

$$
P(y_t\mid y_{<t})
$$

而不是：

$$
P(y_t\mid y_1,\ldots,y_T)
$$

所以：

$$
\boxed{
\text{Causal Mask}
=
\text{model factorization constraint}
}
$$

而不是普通 dropout-like regularization。

---

## 15. 为什么叫 Causal？

这里的 “causal” 主要表示：

> 信息只能沿时间/sequence 顺序从过去流向未来。

不是在说：

> 模型学到了现实世界哲学意义上的因果关系。

更准确地说它是：

> causal / autoregressive attention mask。

即：

$$
j>i
$$

的 future key 对 query $i$ 不可见。

---

## 16. Causal Mask 会允许看自己吗？

标准 next-token Transformer training：

> 通常允许当前 Decoder input position attend 自己。

即：

$$
j=i
$$

合法。

为什么？

因为由于 shifted input：

> 当前 input token 本身对应的是前一个真实 output。

它不是当前要预测的 target。

所以 lower-triangular matrix 包含：

> 主对角线。

---

## 17. 这里 Shift 再次非常关键

假设预测 target：

$$
y_i
$$

Decoder position $i$ 上实际输入的是：

$$
y_{i-1}
$$

所以允许 attention 到自己：

> 只是允许使用已知前一个 token。

并没有泄漏：

$$
y_i
$$

---

## 18. 如果不 Shift，却只做包含对角线的 Causal Mask 会怎样？

如果 Decoder position $i$ 输入正好就是：

$$
y_i
$$

那么它可以看到自己的 Value。

因此当前 target 会泄漏。

这说明：

$$
\boxed{
\text{Shifted Input}
+
\text{Causal Mask}
}
$$

是配套设计。

不能只理解其中一个。

---

## 19. 为什么训练时整个 Target Sentence 已经在 Tensor 里？

这是很多初学者最反直觉的地方。

例如 target：

```text
I love robots
```

训练时我们当然已经知道整句 ground truth。

为了 GPU 并行，我们把整个 shifted target：

```text
<BOS> I love robots
```

一次送进 Decoder。

这看起来好像：

> 模型同时拿到了整个未来。

Tensor 确实在显存里。

但：

> **存在于 Tensor 中不等于当前 Query 有权限读取。**

Causal Mask 规定了每个位置的信息通路。

---

## 20. 这就是训练可以并行的关键

对于所有 positions：

$$
1,\ldots,T
$$

我们可以一次算：

$$
Q,K,V
$$

一次算：

$$
QK^\top
$$

得到整个：

$$
T\times T
$$

score matrix。

然后一次加入 triangular mask。

于是所有 positions 的 loss：

$$
L_1,\ldots,L_T
$$

可以同时计算。

所以：

$$
\boxed{
\text{Autoregressive objective}
\neq
\text{training must be sequential}
}
$$

---

## 21. 为什么这和 RNN 不同？

RNN training 即使有完整 ground truth，

hidden state：

$$
h_t
$$

仍然依赖：

$$
h_{t-1}
$$

所以 sequence 内部 computation 存在真正的 recurrent dependency。

Transformer training 中：

> 所有 positions 的 Q/K/V 可以同时计算，

mask 只限制信息流。

因此 training 能高度并行。

这正是 Transformer 的重要优势之一。

---

## 22. 那为什么 Inference 还是 Sequential？

训练时我们知道真实：

$$
y_1,\ldots,y_T
$$

所以可以构造完整 shifted input。

推理时：

> 未来 token 根本还不存在。

假设当前只有：

```text
<BOS> I love
```

模型先预测：

```text
robots
```

然后把生成的：

```text
robots
```

加入 prefix：

```text
<BOS> I love robots
```

再预测下一 token。

所以：

$$
\boxed{
\text{training can parallelize positions}
}
$$

但：

$$
\boxed{
\text{standard autoregressive inference generates sequentially}
}
$$

---

## 23. 这是不是矛盾？

不矛盾。

训练时：

> Ground-truth prefix 对每个 position 都已知。

推理时：

> Model-generated prefix 只能一步一步产生。

同一个 probability factorization：

$$
P(y)
=
\prod_tP(y_t\mid y_{<t})
$$

在训练和推理阶段有不同计算条件。

---

## 24. Teacher Forcing 到底是什么？

在经典序列模型语境里，

Teacher Forcing 通常指训练时：

> 下一步模型输入使用 ground-truth previous token，

而不是模型自己上一步生成的 token。

Transformer 训练中 shifted target sequence：

```text
<BOS>, y₁, y₂, ..., yₜ₋₁
```

正体现这种 ground-truth-prefix training 思想。

不过要注意：

> 《Attention Is All You Need》正文这里主要用“output embeddings offset by one position”描述，并没有把 “teacher forcing” 作为这段架构说明的核心术语。

所以知识库中可以说：

> 这与通常所说的 teacher-forced autoregressive training 一致。

不要说：

> “Transformer 论文在这里正式定义了 Teacher Forcing。”

---

## 25. Training 和 Inference 的 Prefix 不同

### Training

模型看到：

$$
y_{<t}^{groundtruth}
$$

---

### Inference

模型看到：

$$
\hat y_{<t}^{model}
$$

所以训练和推理存在：

> prefix distribution mismatch。

这类现象常被讨论为：

> exposure bias。

但它不是理解 Causal Mask 必需的主线。

Causal Mask 只负责：

> 确保 position $t$ 不读取未来。

---

## 26. Causal Mask 和 Teacher Forcing 不是一回事

#### Teacher Forcing

回答：

> 已知过去 token 是 ground truth 还是 model prediction？

#### Causal Mask

回答：

> 当前 position 可以访问哪些 sequence positions？

所以：

$$
\boxed{
\text{Teacher Forcing}
\neq
\text{Causal Mask}
}
$$

训练时通常二者同时出现。

---

## 27. Causal Mask 和 Positional Encoding 也不是一回事

#### Positional Encoding

告诉模型：

> token 在哪里。

#### Causal Mask

规定：

> token 能看哪里。

所以：

$$
\boxed{
Position
=
\text{where am I?}
}
$$

$$
\boxed{
Mask
=
\text{what am I allowed to see?}
}
$$

这是一个非常好用的区分。

---

## 28. 为什么有 Mask 之后仍然需要 Position？

考虑 Query 3。

它被允许读取：

$$
0,1,2,3
$$

但仅靠 mask，

它知道的是：

> 这四个 positions 都可见。

它仍需要 position representation 来理解：

- 谁最靠近；
- 谁更早；
- 当前是第几个位置；
- relative distance。

所以 mask 不能替代 PE。

---

## 29. 为什么有 Position 之后仍然需要 Mask？

因为即使 position 2 知道：

> position 3 在未来，

如果 full Attention 仍允许：

$$
2\rightarrow3
$$

模型仍然可以读取真实未来 representation。

所以 PE 不会自动创建硬 causality。

---

## 30. Causal Mask 的 Matrix 形式

设 sequence length：

$$
T
$$

mask：

$$
M_{ij}
=
\begin{cases}
0,&j\le i\\
-\infty,&j>i
\end{cases}
$$

例如：

$$
T=5
$$

：

$$
\boxed{
M=
\begin{bmatrix}
0&-\infty&-\infty&-\infty&-\infty\\
0&0&-\infty&-\infty&-\infty\\
0&0&0&-\infty&-\infty\\
0&0&0&0&-\infty\\
0&0&0&0&0
\end{bmatrix}
}
$$

---

## 31. 最终 Attention 公式

标准 full Attention：

$$
A
=
softmax
\left(
\frac{
QK^\top
}{
\sqrt{d_k}
}
\right)
$$

Causal Attention：

$$
\boxed{
A
=
softmax
\left(
\frac{
QK^\top
}{
\sqrt{d_k}
}
+
M
\right)
}
$$

其中：

$$
M
$$

就是 causal mask。

---

## 32. 一个完整 4×4 数值结构

raw logits：

$$
S=
\begin{bmatrix}
s_{00}&s_{01}&s_{02}&s_{03}\\
s_{10}&s_{11}&s_{12}&s_{13}\\
s_{20}&s_{21}&s_{22}&s_{23}\\
s_{30}&s_{31}&s_{32}&s_{33}
\end{bmatrix}
$$

加入 mask：

$$
S+M=
\begin{bmatrix}
s_{00}&-\infty&-\infty&-\infty\\
s_{10}&s_{11}&-\infty&-\infty\\
s_{20}&s_{21}&s_{22}&-\infty\\
s_{30}&s_{31}&s_{32}&s_{33}
\end{bmatrix}
$$

row-wise Softmax 后：

$$
A=
\begin{bmatrix}
1&0&0&0\\
*&*&0&0\\
*&*&*&0\\
*&*&*&*
\end{bmatrix}
$$

其中每一行的：

$$
*
$$

加起来等于：

$$
1
$$

---

## 33. 为什么第一行一定是 [1,0,0,0]？

因为 Query 0 只有：

$$
Key\ 0
$$

合法。

Softmax 候选只有一个。

所以无论：

$$
s_{00}
$$

是多少，

归一化后：

$$
A_{00}=1
$$

这也说明：

> Causal structure 是由 mask 决定的，不由 learned scores 决定。

---

## 34. 为什么最后一行没有任何 Mask？

因为最后 position：

$$
T-1
$$

已经位于 sequence 尾部。

它允许读取：

$$
0,\ldots,T-1
$$

所有当前与过去 positions。

所以它的 attention row 是普通 full Softmax。

---

## 35. Causal Attention 是一种 Directed Graph

可以把 sequence positions 当 nodes。

普通 full self-attention：

```text
0 ↔ 1 ↔ 2 ↔ 3
所有位置互相可连接
```

Causal：

```text
0 → 1 → 2 → 3
```

更准确地按“信息流”：

> 过去可以影响未来；

> 未来不能影响过去。

所以 allowed connectivity 形成一个：

> directed acyclic information graph。

---

## 36. 多层之后 Future 真的完全不能影响 Past 吗？

如果每一层都严格 causal，

答案是：

> 对同一次 forward 中的 target sequence，未来位置没有路径影响较早位置。

因为每层都只允许：

$$
j\le i
$$

信息流。

层层堆叠仍保持这个因果方向。

---

## 37. 为什么 Residual Connection 不会绕过 Mask？

Residual：

$$
x_i+Attention_i
$$

只把：

> 当前 position 自己已有 representation

加入。

而当前 position 的 representation 在前面层也已经遵守 causal constraint。

所以 residual 不会突然引入 future token。

---

## 38. FFN 会不会泄漏 Future？

Transformer position-wise FFN：

> 每个 position 独立处理自己的 hidden vector。

不跨 positions 混合。

所以如果输入 hidden states 已满足 causality，

FFN 也不会创建 future-to-past information flow。

---

## 39. Cross-Attention 呢？

原始 translation Decoder 的 Cross-Attention：

> Query 可以读取完整 Encoder source sequence。

这通常没问题。

因为 source sentence：

> 在生成开始前就已完整已知。

例如翻译：

```text
English source
```

本来就全部可见。

所以：

$$
\boxed{
\text{Decoder Self-Attention}
\rightarrow
\text{causal}
}
$$

而：

$$
\boxed{
\text{Encoder–Decoder Cross-Attention}
\rightarrow
\text{usually full source visibility}
}
$$

---

## 40. Encoder 为什么通常也不需要 Causal Mask？

原始 Transformer Encoder 任务是：

> 对完整 source input 建 representation。

整个 source sequence 在输入时已经知道。

所以 token 2 读取 token 10：

> 不算作弊。

因此 Encoder Self-Attention通常：

> full bidirectional。

---

## 41. Decoder 为什么特殊？

因为 Decoder target sequence：

> 是模型要生成的答案。

在生成第 $t$ 个 target token 时，

未来 target：

$$
y_{>t}
$$

在真实推理场景不存在。

所以必须模拟这个条件。

这正是 causal mask 的任务。

---

## 42. Encoder-Only 模型一定没有 Causal Mask 吗？

不一定。

“Encoder/Decoder”命名和 mask 结构并不是永远绑定。

例如某些 architecture 可以：

> 使用 Transformer encoder-style stack，但配 causal self-attention。

现代模型命名更加多样。

更本质的判断方法是：

> **这个 position 是否应该访问未来信息？**

而不是只看 class 名字叫 Encoder 还是 Decoder。

---

## 43. GPT 为什么也需要 Causal Mask？

GPT 是 Decoder-style autoregressive language model。

它要建模：

$$
P(x_t\mid x_{<t})
$$

所以必须阻止：

$$
x_t
$$

读取：

$$
x_{>t}
$$

真实训练 token。

因此 causal attention 是 GPT 的核心结构之一。

---

## 44. BERT 为什么不使用标准 Causal Mask？

BERT 的 pretraining 目标不是标准 left-to-right autoregressive factorization。

它希望 token representation 能同时利用：

- 左 context；
- 右 context。

因此使用：

> bidirectional Self-Attention。

这也是 BERT 和 GPT 的核心结构差异之一。

---

## 45. 所以 Mask 其实定义了信息流范式

#### Full / Bidirectional

$$
i
\leftrightarrow
j
$$

适合：

> 编码完整已知输入。

---

#### Causal

$$
j\le i
$$

适合：

> left-to-right autoregressive generation。

---

#### Arbitrary Mask

也可以定义：

- local windows；
- block sparse；
- prefix language modeling；
- multimodal visibility constraints。

所以 Attention mask 是一个很通用的：

> connectivity specification。

---

## 46. Causal Mask 和 Padding Mask 必须区分

### Causal Mask

根据：

> relative sequence position

屏蔽未来。

---

### Padding Mask

根据：

> token 是否是真实数据

屏蔽 PAD。

例如：

```text
A:
token token token token

B:
token token PAD PAD
```

第二个 sample 的 PAD：

> 无论它在“过去还是未来”，都不应该被读取。

---

## 47. 两个 Mask 可以同时存在

实际 attention logits 可以同时加入：

$$
M_{\text{causal}}
$$

和：

$$
M_{\text{padding}}
$$

最终：

$$
\boxed{
softmax(
S
+
M_{\text{causal}}
+
M_{\text{padding}}
)
}
$$

于是某个 connection 只要：

- 位于未来；
- 或是 PAD；

就会被屏蔽。

---

## 48. 为什么 Padding Mask 不等于 Causal Mask？

Padding mask 的 pattern：

> 每个 sample 不同。

Causal mask：

> 通常只取决于 sequence length 和位置关系。

一个真实过去 token：

> 不该被 padding mask。

一个未来真实 token：

> 不该被 padding mask，但应该被 causal mask。

---

## 49. PyTorch 里 Mask 有时是 Bool，有时是 Float，为什么？

不同 API 支持不同 representation。

#### Boolean Mask

例如：

```text
True = forbidden
False = allowed
```

framework 内部转换。

---

#### Additive Float Mask

直接：

```text
0
-inf
```

加入 logits。

数学本质都是：

> 禁止某些 attention connections。

读代码时一定看具体 API semantics，

不要只凭 mask 变量名猜。

---

## 50. 为什么实际代码有时用一个很大的负数，而不是 -∞？

某些 dtype / fused kernels 可能使用：

$$
-10^9
$$

或 dtype 最小值近似：

$$
-\infty
$$

因为：

$$
e^{-10^9}\approx0
$$

数值效果等价于：

> attention weight 约为 0。

但数学定义最清晰仍然是：

$$
-\infty
$$

---

## 51. Causal Mask 在 Multi-Head Attention 中怎样广播？

如果：

$$
A
$$

shape：

$$
[B,H,T,T]
$$

标准 causal mask 可以先是：

$$
[T,T]
$$

然后对：

- batch；
- head；

broadcast。

因为所有 heads 通常共享：

> 同样的时间可见性约束。

但每个 head 的 actual attention weights 仍不同。

---

## 52. 每个 Head 都不能偷看 Future

即使：

Head 3 的 raw score：

$$
q_i^{(3)\top}k_j^{(3)}
$$

对未来 position 极高，

只要：

$$
j>i
$$

mask 都会把它变：

$$
-\infty
$$

所以所有 heads 都服从同一个 causal graph。

---

## 53. 为什么 Training 一次可以预测所有 Target Positions？

假设 target 长度：

$$
T
$$

Decoder 一次输出：

$$
h_1,\ldots,h_T
$$

然后 output projection：

$$
z_t=W_oh_t+b
$$

每个 position 同时得到 next-token logits。

loss：

$$
L
=
\sum_{t=1}^T
-\log
P(
y_t
\mid
y_{<t}
)
$$

所有 terms 可以并行计算。

Causal mask 保证：

> 虽然计算并行，但信息依赖仍然满足 autoregressive factorization。

---

## 54. 这是一个很重要的区分

$$
\boxed{
\text{parallel computation}
\neq
\text{non-causal dependency}
}
$$

Transformer training 做到了：

> 计算图在硬件上并行，

同时：

> 概率模型在信息依赖上保持 causal。

这是 Causal Mask 最漂亮的价值之一。

---

## 55. 为什么 Inference 不能也一次算完整 T？

因为要算：

$$
P(y_3\mid y_1,y_2)
$$

你必须先知道：

$$
y_2
$$

而推理时：

$$
y_2
$$

是模型生成的未知变量。

所以标准 exact autoregressive generation：

> 必须逐步产生新 token。

Causal Mask 不能凭空提供尚未生成的 token。

---

## 56. Training 的 Future Ground Truth 虽存在，但被隔离

可以这样理解：

```text
GPU memory:
整句 target 都在

Model information graph:
每个 position 只能访问合法 prefix
```

所以：

> 数据物理存在

和：

> 模型逻辑可见

是两件不同的事。

---

## 57. 为什么这和考试很像？

所有答案可能都已经印在同一张系统后台数据库里。

但每个学生考试界面：

> 只开放当前可看的题目信息。

数据存在不代表：

> 访问权限存在。

Causal Mask 就是：

> information-access control。

---

## 58. Causal Mask 会不会让训练速度变回 RNN？

不会。

虽然 dependency 是 causal，

但 mask 是一个静态矩阵。

所有 Q/K/V 和 score matrix 仍可以一次并行计算。

RNN 的问题是：

$$
h_t
$$

数值本身必须等：

$$
h_{t-1}
$$

算完。

Causal Attention 不存在这种计算依赖：

> 只是某些 score entries 被禁用。

---

## 59. Causal Mask 和 KV Cache 是什么关系？

推理时 autoregressive generation 每步都重复处理越来越长 prefix。

KV Cache 通过缓存过去 positions 的：

$$
K,V
$$

避免反复重算。

但：

> KV Cache 不改变 causal factorization。

它只是：

> 推理优化。

后续可以单独写：

> **KV Cache：为什么 Transformer 推理不用每步重算所有历史？**

---

## 60. 为什么 ACT Decoder 不需要 Causal Mask？

现在进入最关键的 ACT 对比。

ACT 不是在建模：

$$
P(a_{t+i}\mid a_{t:t+i-1},o_t)
$$

这种逐 action autoregressive factorization。

它的核心目标是：

$$
\boxed{
\pi(
a_{t:t+k}
\mid
o_t
)
}
$$

一次输出整个 action chunk。

所以：

> future action slot $i$ 没有“必须先生成 slot $i-1$ 才能生成”的 autoregressive constraint。

---

## 61. ACT 的 k 个 Action Slots 是并行输出结构

Decoder 一开始就存在：

$$
q_0,q_1,\ldots,q_{k-1}
$$

这些 output slots：

- 同时存在；
- 可以彼此 Self-Attend；
- 都可以读取 Encoder Memory；
- 最后一起投影成 $k$ 个动作。

所以：

$$
\boxed{
\text{ACT Decoder}
=
\text{non-autoregressive structured decoder}
}
$$

---

## 62. 为什么 ACT Action Slot 可以看“未来 Slot”？

因为这里的：

> “future”

只是：

> action chunk 内时间更后的 output slot。

它不是训练时不该泄漏的 ground-truth future token。

Decoder slots 一开始不是 ground-truth future actions。

它们只是：

> learned/fixed query representations + evolving hidden states。

所以 slot 2 和 slot 20 彼此 Self-Attend：

> 不会直接把真实 $a_{t+20}$ 答案泄漏给 $a_{t+2}$。

---

## 63. 这和语言 Decoder 本质不同

语言 training 中 Decoder 输入 sequence 包含：

> shifted ground-truth target tokens。

所以更右边的位置携带：

> 真实 future target content。

不 mask 就会泄漏。

---

ACT Policy Decoder 中：

> target actions 并没有作为 decoder input tokens 喂进去。

它的 decoder input 是：

> action query slots。

所以不存在同样的 target-token leakage。

---

## 64. ACT Ground-Truth Future Actions 去哪里了？

Training 时 ground-truth chunk：

$$
a_{t:t+k}
$$

当然存在。

但它主要用于：

1. CVAE encoder 产生 posterior $z$；
2. 计算 policy output reconstruction loss。

它不是：

> 作为 Policy Transformer Decoder 的 shifted target-action sequence 输入。

所以 Policy Decoder 不需要用 causal mask 防止：

> “slot i 看到真实 slot i+1 action”。

因为真实未来 action 不在那里。

---

## 65. CVAE Encoder 能看完整 Future Action Chunk，不算作弊吗？

这又是另一个容易混淆的问题。

Training-only CVAE encoder：

$$
q_\phi(
z
\mid
a_{t:t+k},
q_t
)
$$

确实看完整 target action chunk。

但它是：

> approximate posterior / recognition network。

它在测试时会被丢弃。

其任务是：

> 在训练时根据 demonstration action chunk 推断 latent style $z$。

这不是 Policy Decoder 的 autoregressive target leakage。

详见：

- [CVAE in ACT](../robot-learning/act/cvae-in-act.md)

---

## 66. ACT Policy Decoder 的 Self-Attention 在官方代码里有 Mask 参数吗？

有接口。

当前官方 `TransformerDecoderLayer` 的 self-attention 调用：

```python
self.self_attn(
    q,
    k,
    value=tgt,
    attn_mask=tgt_mask,
    ...
)
```

所以代码结构：

> 支持传 `tgt_mask`。

但是关键问题是：

> ACT 实际调用时传了吗？

---

## 67. 当前 Official ACT Forward 没有传 tgt_mask

官方 `Transformer.forward(...)` 中：

```python
tgt = torch.zeros_like(query_embed)

hs = self.decoder(
    tgt,
    memory,
    memory_key_padding_mask=mask,
    pos=pos_embed,
    query_pos=query_embed
)
```

没有传：

```python
tgt_mask
```

所以：

$$
\boxed{
tgt\_mask=None
}
$$

沿默认参数传入 Decoder layers。

因此 ACT action-slot Self-Attention：

> **没有 causal target mask。**

这是 released-code 直接可验证的事实。

---

## 68. 为什么官方代码保留 tgt_mask 参数？

因为 ACT 的 Transformer 文件是：

> 从 DETR / `torch.nn.Transformer` 风格实现修改来的通用 Decoder。

通用 Decoder API 可以支持 mask。

但：

> 一个参数存在 ≠ 当前模型实际使用它。

读代码时必须看：

> call site。

ACT 当前调用没有传 causal `tgt_mask`。

---

## 69. ACT Cross-Attention 也不需要 Causal Mask

Action Queries 读取的是：

> 当前 observation Encoder Memory。

这些 observation tokens：

- camera features；
- joint state；
- latent representation；

在当前 policy forward 已全部可用。

所以 action slot 可以读取完整 Memory。

这里没有：

> 未来 target action answer

藏在 Memory 中。

---

## 70. 那 ACT 的 Future Action Slots 彼此互看，会不会破坏时间顺序？

不会自动破坏。

时间 slot identity 由：

> action query positions / embeddings

提供。

所以模型仍知道：

```text
slot 0
slot 1
...
slot 99
```

只是它允许这些 slots：

> 共同协商整个 chunk structure。

这更像：

> 一次预测一条完整轨迹。

而不是逐点生成。

---

## 71. 一个类比：一次画完整曲线

语言 autoregressive：

```text
先写第 1 个字
→ 再写第 2 个字
→ 再写第 3 个字
```

后面的真实字在训练时必须遮住。

---

ACT chunk decoder：

```text
给 100 个未来时间槽位
→
一次共同预测一整条 100-step action curve
```

各时间槽位可以彼此交互，

因为它们不是：

> 已知答案 token。

它们是：

> 待求解的 latent output slots。

---

## 72. ACT 为什么不做 Autoregressive Action Decoder？

论文的核心就是：

> Action Chunking。

一次预测一段动作减少 effective horizon，

并对 temporally correlated behavior 更友好。

如果 Decoder 再内部逐 action autoregressively rollout，

会重新引入：

> chunk 内 sequential generation dependency。

ACT 选择并行 action queries，

更符合：

> 一次预测 action sequence

的设计。

---

## 73. Non-Autoregressive 不代表 Action Positions 完全独立

这是非常关键的。

ACT 不 autoregressive，

但：

> action slots 之间仍有 Decoder Self-Attention。

所以：

$$
a_i
$$

和：

$$
a_j
$$

的 hidden representations 可以相互影响。

区别只是：

> 这种依赖不是“必须先生成 $a_i$，再把生成值作为输入生成 $a_j$”的 autoregressive chain。

而是：

> parallel joint representation interaction。

---

## 74. Joint Prediction 和 Autoregressive Prediction 的区别

### Autoregressive

概率结构：

$$
P(a_1,\ldots,a_k)
=
\prod_i
P(
a_i
\mid
a_{<i}
)
$$

生成：

```text
a1
↓
a2
↓
a3
```

---

### ACT-Style Parallel Chunk Prediction

policy 直接输出：

$$
\boxed{
\hat A
=
f(o,z)
\in
\mathbb R^{k\times d_a}
}
$$

所有 action slots 在同一个 forward 中得到。

Decoder 内部可以相互 Attention，

但不需要逐 action sampling。

---

## 75. CVAE 又如何表示 Multimodality？

ACT 不通过：

> autoregressive action sampling

表达 demonstration variability。

它通过：

$$
z
$$

latent variable，

用 CVAE 建模 action sequence 的 latent style / variation。

所以：

> non-autoregressive decoder

并不意味着模型完全无法建模复杂 sequence distribution。

它的 stochasticity / multimodality来自另一条设计：

> latent variable $z$。

测试时 ACT 又固定：

$$
z=0
$$

实现 deterministic decoding。

---

## 76. 为什么 ACT Training 仍然可以一次算整个 Chunk Loss？

Policy 输出：

$$
\hat A
\in
[B,k,14]
$$

ground truth：

$$
A
\in
[B,k,14]
$$

直接计算：

$$
L_{L1}
$$

以及 KL。

不存在：

> 逐 action teacher forcing。

所以训练与推理的 decoder structural pattern：

> 都是一次并行产生 $k$ 个 action slots。

---

## 77. 语言 Transformer Training/Inference Gap 和 ACT 不一样

语言：

#### Training

ground-truth prefix。

#### Inference

model-generated prefix。

---

ACT：

#### Training

current observation + posterior-sampled $z$ → whole chunk。

#### Inference

current observation + $z=0$ → whole chunk。

主要 train/test latent difference是：

> posterior sample vs prior mean，

而不是：

> teacher-forced action prefix vs model-generated action prefix。

---

## 78. 为什么这对理解 ACT 很重要？

如果误以为：

> “只要用了 Transformer Decoder，就一定要 causal mask。”

你就会把 ACT 理解错成：

> 一个 GPT 式动作生成器。

但 ACT 的 Decoder 更接近：

> DETR-style query decoder。

一组 output queries：

> 并行读取 Memory，并行产生 structured outputs。

所以：

$$
\boxed{
\text{Transformer Decoder}
\neq
\text{Autoregressive Decoder by definition}
}
$$

---

## 79. Decoder 这个名字本身不决定 Causality

一个 Transformer Decoder layer 通常有：

- self-attention；
- cross-attention；
- FFN。

但它是否 causal：

> 取决于 Self-Attention mask 和输入结构。

原始 translation Transformer：

> causal。

DETR：

> non-causal object queries。

ACT：

> non-causal action queries。

所以一定要看：

> **mask + target representation + objective。**

---

## 80. 为什么 ACT Query Slots 可以 Full Self-Attend？

因为它们本质上是一组：

> output latent variables / slots。

Full Self-Attention 让 slot $i$ 可以利用：

> 其他 slots 当前 hidden representations。

这样模型能联合协调：

- action timing；
- smoothness patterns；
- multi-joint consistency；
- sequence-level structure。

注意：

> 这些具体能力是 architecture intuition。

论文并没有逐条证明每个 head 学到这些功能。

---

## 81. Causal Mask 和 Temporal Ensemble 也完全无关

Causal Mask：

> 单个 Decoder forward 内的 token visibility。

Temporal Ensemble：

> 不同 policy query timesteps 产生的 overlapping chunks，在执行时融合同一绝对 timestep 的动作预测。

所以：

$$
\boxed{
\text{Causal Mask}
=
\text{intra-network visibility}
}
$$

$$
\boxed{
\text{Temporal Ensemble}
=
\text{inter-query action aggregation}
}
$$

---

## 82. Causal Mask 和 Action Chunking 也不是反义词

理论上你完全可以设计：

> 一个 action chunk model，

内部仍使用 autoregressive masked decoder。

所以：

$$
\text{chunking}
$$

和：

$$
\text{causal decoding}
$$

是两个不同设计轴。

ACT 具体选择的是：

> action chunking + non-causal query decoder。

---

## 83. 为什么未来动作在物理上是“未来”，却不一定要 Mask？

因为 Attention mask 的判断标准不是：

> “这个 slot 代表未来时间吗？”

而是：

> “这个 slot 的 representation 里是否含有推理时本不该可见的真实信息？”

ACT future action slot：

> 只是待预测位置 identity。

它没有装着 ground-truth future action。

所以彼此可见没有 label leakage。

这是非常重要的判断原则。

---

## 84. 一个泛化判断法

遇到任何模型，问：

> “要不要 causal mask？”

不要先看：

> token 名字叫不叫 future。

而应该问：

1. 当前 Query 要预测什么？
2. 其他 positions 中是否含有推理时未知的目标真值？
3. 目标分布是否被定义成 left-to-right factorization？
4. 这些 positions 是否允许联合推理？

只有这样才能正确判断。

---

## 85. Prefix-LM 为什么 Mask 又不一样？

有些模型允许：

> 一段 prefix 内 full attention，

而生成区域 causal。

那么 mask 可以长成：

```text
Prefix:
全部互看

Generated region:
可看 prefix + 过去 generated tokens
不可看未来 generated tokens
```

这说明：

> causal mask 不一定永远是最简单纯三角形。

它可以根据 task information structure 定义。

---

## 86. Vision Transformer 为什么通常不 Causal？

图像所有 patches：

> 输入时同时已知。

没有 left-to-right generation constraint。

所以 image encoder通常 full self-attention。

但如果做：

> autoregressive image generation，

则可以使用 spatial causal ordering / mask。

所以是否 causal：

> 由 objective 决定，不由 modality 决定。

---

## 87. Robot Policy 也可能需要 Causal Attention

例如输入一段 observation history：

$$
o_{t-L:t}
$$

如果训练一个 online temporal model，

可能要确保：

> past hidden representation 不读取未来 observation。

这时 temporal causal mask就有意义。

ACT 当前 policy architecture主要输入：

> current observation，

所以不是这种 history-sequence causal modeling。

---

## 88. Causal Mask 和 Online Control 不是同义词

机器人是 online control，

不代表内部所有 Transformer 都必须 causal。

ACT 当前 observation：

$$
o_t
$$

已经可见。

它一次预测未来 action chunk：

$$
a_{t:t+k}
$$

这些 output slots 联合推理。

Online causality体现在：

> policy 不能使用未来真实 observation $o_{t+1}$。

而不是：

> future action slots 必须互相遮住。

---

## 89. 这一区分非常重要

现实时间因果性：

$$
o_{t+1}
$$

尚未发生。

所以 ACT policy 在时刻 $t$：

> 不输入未来 observation。

---

模型内部 output-slot interaction：

$$
slot_i
\leftrightarrow
slot_j
$$

只是：

> 一次计算内部的 latent coordination。

不等于访问未来世界信息。

所以：

$$
\boxed{
\text{world-time causality}
\neq
\text{decoder-slot causal mask}
}
$$

---

## 90. 为什么 ACT 仍然是 Closed-Loop？

虽然一次预测：

$$
k
$$

个未来动作，

最终 ACT + Temporal Ensemble 会：

> 每个 environment timestep重新观察当前状态并再次 query policy。

所以系统层面仍不断利用新 observation。

这和：

> Decoder 内部是否 causal

又是第三个独立问题。

详见：

- [ACT Inference](../robot-learning/act/inference.md)
- [Temporal Ensemble](../robot-learning/act/temporal-ensemble.md)

---

## 91. 三种“因果”不要混淆

### 1. Autoregressive Token Causality

$$
y_t
\text{ cannot see }
y_{>t}
$$

由：

> Causal Mask

实现。

---

### 2. Real-World Temporal Causality

时刻 $t$ 不能获得：

$$
o_{t+1}
$$

真实未来 observation。

由：

> data availability / control loop

保证。

---

### 3. Statistical Causality

例如：

> $X$ 是否真正导致 $Y$。

这是 causal inference 的概念。

和 Transformer causal mask：

> 完全不是一回事。

---

## 92. 常见误解一：Causal Mask 是为了告诉模型 Token 顺序

**错误。**

顺序主要由 positional information 表示。

Mask 规定可见性。

---

## 93. 常见误解二：有 Positional Encoding 就不需要 Causal Mask

**错误。**

知道未来在哪里，

不等于禁止读取未来。

---

## 94. 常见误解三：有 Causal Mask 就不需要 Positional Encoding

**错误。**

Mask 给出可见集合，

但不充分表达精确位置关系。

---

## 95. 常见误解四：Training 时只把 Prefix 一个个送进去

原始 Transformer training 不需要这样。

可以把完整 shifted target 一次送入，

通过 causal mask 并行计算所有 positions。

---

## 96. 常见误解五：既然整个 Target 在 GPU 里，模型一定看到了未来

**错误。**

物理存在和计算图可访问是两回事。

Mask 阻断非法 attention edges。

---

## 97. 常见误解六：右移一位后就不需要 Mask

**错误。**

更右侧 Decoder inputs 仍然含有当前/未来 target ground truth。

---

## 98. 常见误解七：只要有 Mask 就不需要右移

标准 Transformer 不是这样。

如果对角线允许访问，

当前 ground-truth token 会直接泄漏。

---

## 99. 常见误解八：Mask 是在 Softmax 后简单乘 0

原始 Transformer 论文明确描述：

> 在 Softmax 输入处，把非法连接设为 $-\infty$。

---

## 100. 常见误解九：-∞ 只是一个经验 trick

它有直接数学理由：

$$
e^{-\infty}=0
$$

从而 Softmax weight 精确为 0。

---

## 101. 常见误解十：Causal Mask 会让 Transformer 训练也必须逐 token

**错误。**

矩阵运算仍然并行。

Mask 只是限制 connection pattern。

---

## 102. 常见误解十一：Autoregressive Training 和 Inference 都同样并行

**错误。**

training 有完整 ground-truth prefix，

standard inference 的 future token 尚未生成。

---

## 103. 常见误解十二：Teacher Forcing = Causal Mask

**错误。**

一个决定：

> 用谁作为过去输入。

另一个决定：

> 哪些 positions 可见。

---

## 104. 常见误解十三：Encoder 永远不需要 Causal Mask

不绝对。

是否 mask 取决于 task objective / information availability。

---

## 105. 常见误解十四：所有 Transformer Decoder 都必须 Causal

**错误。**

ACT 和 DETR 都是重要反例。

---

## 106. 常见误解十五：ACT Future Action Slot 之间互看就是偷看未来答案

**错误。**

这些 slots 不包含 ground-truth future actions。

它们是待预测的 latent/output representations。

---

## 107. 常见误解十六：ACT Training Action Chunk 已知，所以 Decoder 一定会泄漏

**错误。**

Ground-truth chunk 用于：

- CVAE posterior encoder；
- reconstruction loss。

不是作为 policy decoder 的 shifted target-action input。

---

## 108. 常见误解十七：CVAE Encoder 看 Future Action 就违反推理规则

**错误。**

它是 training-only recognition network，

测试时被丢弃。

这属于 VAE/CVAE posterior inference 机制。

---

## 109. 常见误解十八：ACT 没 Causal Mask，所以不是时间序列模型

**错误。**

它仍预测：

$$
k
$$

步有序动作序列。

只是采用：

> parallel structured prediction

而不是：

> autoregressive factorization。

---

## 110. 常见误解十九：Non-Causal Decoder 意味着不知道时间顺序

**错误。**

action query slots 有 position / slot identity。

Causality 和 position 是两个概念。

---

## 111. 常见误解二十：机器人系统是在线的，所以 Decoder 一定要 causal

**错误。**

Online world causality 和 internal output-slot masking 是不同问题。

---

## 112. 用一张表彻底区分

| 机制 | 回答的问题 | 原始语言 Transformer Decoder | ACT Decoder |
|---|---|---:|---:|
| Positional / Query Encoding | 我在哪里？ | ✓ | ✓ |
| Causal Self-Attention Mask | 我能看未来 target slots 吗？ | 不允许 | 允许 action slots 互看 |
| Cross-Attention | 我怎样读取 Encoder Memory？ | ✓ | ✓ |
| Teacher-forced target prefix | 过去 target 输入来自哪里？ | Ground truth during training | 不使用 action-prefix teacher forcing |
| Parallel training | 所有 output positions 能否同时算 loss？ | ✓ | ✓ |
| Autoregressive inference | 是否逐 output token 生成？ | ✓ | ✗ |
| Parallel structured output | 是否一次输出全部 slots？ | ✗ | ✓ |

---

## 113. 原始 Transformer 的核心链条

可以把语言 Decoder 写成：

```text
Ground-truth target:
I love robots <EOS>

Shift right:
<BOS> I love robots

        │
        ▼
Positional Encoding
        │
        ▼
Masked Self-Attention
        │
只允许每个位置访问自己的合法 prefix
        │
        ▼
Cross-Attention to Encoder
        │
        ▼
FFN
        │
        ▼
Next-token logits for every position
        │
        ▼
Cross-Entropy
```

Training 中这些位置：

> 同时计算。

Inference 中：

> 一个个生成。

---

## 114. ACT 的对应链条

```text
Current observation
        │
        ▼
Policy Encoder
        │
        ▼
Observation Memory

k action query slots
        │
        ▼
Full / non-causal
Decoder Self-Attention
        │
        ▼
Cross-Attention
to Observation Memory
        │
        ▼
FFN
        │
        ▼
k action representations
        │
        ▼
Linear 512 → 14
        │
        ▼
[k × 14] action chunk
```

没有：

> shifted action targets。

也没有：

> causal action-slot mask。

---

## 115. 官方 ACT 代码证据

当前官方 `Transformer.forward(...)`：

```python
tgt = torch.zeros_like(query_embed)

memory = self.encoder(
    src,
    src_key_padding_mask=mask,
    pos=pos_embed
)

hs = self.decoder(
    tgt,
    memory,
    memory_key_padding_mask=mask,
    pos=pos_embed,
    query_pos=query_embed
)
```

注意：

> 没有传 `tgt_mask`。

而 `TransformerDecoder.forward(...)` 中：

```python
tgt_mask: Optional[Tensor] = None
```

所以默认：

$$
\boxed{
tgt\_mask=None
}
$$

Decoder layer 再把它传：

```python
self.self_attn(
    q,
    k,
    value=tgt,
    attn_mask=tgt_mask,
    ...
)
```

因此：

> ACT released code 的 action-slot Self-Attention 没有 causal mask。

---

## 116. 为什么这个 Official-Code Fact 很有价值？

因为只看 class：

```python
TransformerDecoderLayer
```

很容易凭语言模型经验自动脑补：

> “Decoder = causal。”

但 code 清楚告诉我们：

> 是否 causal 由 `tgt_mask` 决定。

ACT 不传。

所以知识库以后读任何 Transformer 代码时，都应该养成习惯：

> **不要只看模块名字，要追 mask 的 call path。**

---

## 117. 一句话理解 Causal Mask

> **Causal Mask 是一种硬信息流约束：在 autoregressive Self-Attention 中，它把当前 position 对所有 future keys 的 logits 设为 $-\infty$，使这些 connections 经 Softmax 后权重严格为 0，从而保证第 $t$ 个预测只能依赖合法 prefix，而不能利用训练时已经存在于 tensor 中的未来 ground-truth targets。**

---

## 118. 一句话理解“为什么训练并行但推理串行”

> **训练时完整 ground-truth target 已知，因此可以一次构造所有 shifted prefixes，并用 triangular mask 在一张 Attention Matrix 中同时隔离每个 position 的合法历史；推理时未来 token 尚不存在，只能先生成当前 token，再把它加入 prefix 后继续生成下一个，所以 autoregressive inference 仍然顺序进行。**

---

## 119. 一句话理解 ACT 为什么不需要 Causal Mask

> **ACT 的 Policy Decoder 不是把 ground-truth action sequence 右移后逐动作预测，而是使用 $k$ 个 action-query slots 在同一次 forward 中联合预测整个 action chunk；这些 slots 本身不含未来真实动作，因此互相 Self-Attend 不构成 target leakage，官方实现也没有向 Decoder 传 `tgt_mask`。**

---

## 120. 最核心的判断标准

以后看到任何 Transformer，

不要问：

> “这是 Decoder，所以要不要 causal mask？”

而应该问：

$$
\boxed{
\text{当前 position 是否可能访问推理时本不该知道的真实未来信息？}
}
$$

如果答案是：

> 是，

就需要相应 visibility constraint。

如果答案是：

> 否，

那 causal mask 不一定有必要。

这比死记：

```text
Encoder = no mask
Decoder = mask
```

更接近真正的 Transformer 原理。

---

## 121. 下一步：Transformer Encoder

现在 Attention 主线的关键组件已经完整：

```text
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
↓
Positional Encoding
↓
Causal Mask
```

下一篇最自然的是把这些模块组装成完整的：

> **Transformer Encoder Layer。**

下一篇：

> **[Transformer Encoder：一层到底对 Token 做了什么？](./transformer-encoder.md)**

会讲：

- Multi-Head Self-Attention；
- Residual；
- LayerNorm；
- FFN；
- 第二次 Residual；
- 原始 Transformer 为什么是 Post-LN；
- $d_{\text{model}}=512\rightarrow d_{ff}=2048\rightarrow512$；
- FFN 为什么每个 token 独立但参数共享；
- Attention 和 FFN 为什么是互补分工；
- 多层 Encoder 怎样逐渐建立 contextualized memory；
- ACT Policy Encoder 与 CVAE Encoder 分别怎样使用 Encoder stack。

---

### Primary Source：Transformer

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,  
Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin.

**Attention Is All You Need.**  
NeurIPS 2017.

- arXiv: https://arxiv.org/abs/1706.03762
- HTML: https://arxiv.org/html/1706.03762

#### Section 3.1 — Decoder

原论文明确说明：

> Decoder Self-Attention 被修改，以防止 positions attend subsequent positions。

并指出：

> masking 与 output embeddings offset by one position 配合，使 position $i$ 的 prediction 只能依赖小于 $i$ 的已知 outputs。

---

#### Section 3.2.3 — Applications of Attention

原论文进一步明确写：

> Decoder Self-Attention 只允许每个 position attend 到自己以及之前的 positions。

为了保持 autoregressive property：

> 所有非法连接在 Softmax 输入处被设为 $-\infty$。

数学上即：

$$
\boxed{
A
=
softmax
\left(
\frac{QK^\top}{\sqrt{d_k}}
+
M_{\text{causal}}
\right)
}
$$

---

### ACT Primary Source

Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.  
**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.**  
RSS 2023.

- arXiv: https://arxiv.org/abs/2304.13705
- HTML: https://arxiv.org/html/2304.13705v1

ACT 的 policy 不是语言式 autoregressive decoder。

论文描述 Transformer Decoder：

- 输入为 $k$ 个 action-query position representations；
- 通过 Cross-Attention 条件化在 Encoder Memory；
- 输出：
  $$
  k\times512
  $$
- 再投影为：
  $$
  k\times14
  $$

整个 action chunk 在一次 policy forward 中产生。

---

### Official ACT Implementation

Repository:

https://github.com/tonyzhaozh/act

当前：

`detr/models/transformer.py`

中 `TransformerDecoder.forward(...)` 接受：

```python
tgt_mask: Optional[Tensor] = None
```

Decoder Self-Attention 使用：

```python
self.self_attn(
    q,
    k,
    value=tgt,
    attn_mask=tgt_mask,
    ...
)
```

但 ACT 的 `Transformer.forward(...)` 调用：

```python
hs = self.decoder(
    tgt,
    memory,
    memory_key_padding_mask=mask,
    pos=pos_embed,
    query_pos=query_embed
)
```

并没有传：

```python
tgt_mask
```

因此当前 released implementation 中：

$$
\boxed{
tgt\_mask=None
}
$$

即：

> action-query Self-Attention 不使用 causal target mask。

---

### 本文知识连接

#### 前置

- [Self-Attention](./self-attention.md)
- [Softmax](./softmax.md)
- [Positional Encoding](./positional-encoding.md)
- [Transformer](./transformer.md)

#### Transformer

- [Transformer Encoder](./transformer-encoder.md)
- [Transformer Decoder](./transformer-decoder.md)
- [Cross-Attention](./cross-attention.md)
- [Multi-Head Attention](./multi-head-attention.md)
- Padding Mask
- Autoregressive Modeling
- Teacher Forcing
- KV Cache

#### Probability

- Chain Rule of Probability
- Conditional Probability

#### Robot Learning

- [ACT Architecture](../robot-learning/act/architecture.md)
- [Action Chunking](../robot-learning/act/action-chunking.md)
- [CVAE in ACT](../robot-learning/act/cvae-in-act.md)
- [ACT Training](../robot-learning/act/training.md)
- [ACT Inference](../robot-learning/act/inference.md)
- [Temporal Ensemble](../robot-learning/act/temporal-ensemble.md)

#### 下一步

- [Transformer Encoder](./transformer-encoder.md)
