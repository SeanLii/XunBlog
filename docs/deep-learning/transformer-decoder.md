---
title: "Transformer Decoder：为什么比 Encoder 多一个 Cross-Attention？"
description: "从原始 Transformer 的 masked self-attention、encoder-decoder cross-attention 与 FFN 出发，完整拆解 Decoder Layer；再对照 ACT 的 non-causal action-query decoder，解释 zero tgt、query embeddings、7-layer decoder、shape flow，以及论文与 released code 的差异。"
status: reviewed
pageType: concept
canonical: /deep-learning/transformer-decoder
updated: "2026-09-15"
---

# Transformer Decoder：为什么比 Encoder 多一个 Cross-Attention？

在上一篇 [Transformer Encoder](./transformer-encoder.md) 中，我们已经把 Encoder Layer 完整组装起来：

\[
\boxed{
\text{Self-Attention}
\rightarrow
\text{Residual + LayerNorm}
\rightarrow
\text{FFN}
\rightarrow
\text{Residual + LayerNorm}
}
\]

Decoder 看起来非常像 Encoder。

但它多了一件极其重要的事：

> **它不仅要理解“自己已经生成到哪里”，还要去读取 Encoder 已经构建好的输入 Memory。**

所以原始 Transformer Decoder 比 Encoder 多一个 sub-layer：

\[
\boxed{
\text{Encoder–Decoder Cross-Attention}
}
\]

完整结构变成：

```text
Masked Self-Attention
↓
Residual + LayerNorm
↓
Cross-Attention
↓
Residual + LayerNorm
↓
Feed-Forward Network
↓
Residual + LayerNorm
```

原始 Transformer 论文 Section 3.1 明确写道：

> Decoder 也是 \(N=6\) 层；除了 Encoder Layer 原本的两个 sub-layers 外，Decoder 额外插入第三个 sub-layer，用 Multi-Head Attention 读取 Encoder Stack 的输出。

这句话已经把 Decoder 与 Encoder 的根本差异点出来了。

但 ACT 又会让事情变得更有意思。

ACT 同样使用 Transformer Decoder，

却不是：

> 一个 token 一个 token autoregressive 地生成动作。

它一次放入：

\[
k
\]

个 action-query slots，

然后并行产生：

\[
k
\]

个 future action representations。

所以这一篇我们要同时理解两件事：

1. **原始语言 Transformer Decoder 是怎么工作的；**
2. **ACT 怎样保留 Decoder 的 Self-Attention + Cross-Attention + FFN 骨架，但把它改造成 non-autoregressive action decoder。**

---

# 1. Encoder 和 Decoder 的职责为什么不同？

先看 Encoder。

Encoder 输入：

\[
x_1,\ldots,x_n
\]

得到：

\[
M=
[m_1,\ldots,m_n]
\]

这份：

\[
M
\]

就是：

> Encoder Memory。

它的任务主要是：

> **把输入 sequence 变成 contextualized representations。**

---

Decoder 则要产生：

\[
y_1,\ldots,y_m
\]

因此它同时面临两个信息源：

### Source 1：自己的输出侧状态

例如已经生成：

```text
我 喜欢
```

当前 Decoder 必须知道：

> 目标 sequence 已经发展到哪里。

---

### Source 2：Encoder Memory

它还必须知道：

```text
I love robots
```

源输入表达了什么。

所以 Decoder 天然需要两次不同性质的信息读取：

\[
\boxed{
\text{Self-Attention}
}
\]

处理 output-side context，

和：

\[
\boxed{
\text{Cross-Attention}
}
\]

读取 input-side memory。

---

# 2. 为什么 Encoder 只需要两个 Sub-Layers，而 Decoder 要三个？

Encoder：

```text
输入 token 之间互相交流
↓
Self-Attention

每个 token 内部加工
↓
FFN
```

---

Decoder：

```text
输出 positions 之间互相交流
↓
Self-Attention

输出 positions 读取输入 Memory
↓
Cross-Attention

每个输出 position 内部加工
↓
FFN
```

所以 Decoder 多出来的不是：

> “又一个重复 Attention。”

它解决的是一个 Encoder 根本没有的需求：

\[
\boxed{
\text{output representation must condition on encoded input}
}
\]

---

# 3. 原始 Transformer Decoder 一层的正式结构

输入当前 decoder hidden sequence：

\[
H
\]

Encoder Memory：

\[
M
\]

第一个 sub-layer：

\[
\boxed{
S=
MaskedMHA(H)
}
\]

Post-LN 原论文形式：

\[
\boxed{
H_1=
LN_1(
H+
Dropout(S)
)
}
\]

---

第二个 sub-layer：

\[
\boxed{
C=
CrossMHA(
Q=H_1,
K=M,
V=M
)
}
\]

然后：

\[
\boxed{
H_2=
LN_2(
H_1+
Dropout(C)
)
}
\]

---

第三个 sub-layer：

\[
F=
FFN(H_2)
\]

最后：

\[
\boxed{
H_3=
LN_3(
H_2+
Dropout(F)
)
}
\]

这就是一整个原始 Transformer Decoder Layer。

---

# 4. 为什么 Decoder 有三个 LayerNorm？

因为有三个 residualized sub-layers：

1. Self-Attention；
2. Cross-Attention；
3. FFN。

原始 Transformer 对每一个 sub-layer 都使用：

\[
\boxed{
LayerNorm(
x+Sublayer(x)
)
}
\]

所以 Decoder 自然有：

\[
3
\]

次 Add & Norm。

Encoder 只有两个 sub-layers，

所以有：

\[
2
\]

次。

---

# 5. Decoder 第一个 Self-Attention 在做什么？

在语言 Transformer 中，

假设 Decoder 当前 target-side input：

```text
<BOS> 我 喜欢 机器人
```

每个位置先通过：

> Masked Multi-Head Self-Attention

读取合法的 target-side prefix。

例如：

```text
位置 2
```

只能读取：

```text
<BOS>
我
喜欢
```

不能读取更右边的真实未来 target。

所以这个 sub-layer 解决：

> **当前输出位置应该怎样理解已经可见的输出侧上下文？**

---

# 6. 为什么必须先 Self-Attention，再 Cross-Attention？

这是一个非常自然的设计。

当前 Decoder Query 在去读 Encoder 前，

最好先形成：

> “我现在已经生成了什么、当前 output-side state 是什么？”

例如：

```text
我 喜欢 ...
```

和：

```text
它 是 ...
```

即使都在读取同一个 Encoder Memory，

它们当前需要的信息很可能不同。

所以先：

\[
H
\rightarrow
SelfAttention
\rightarrow
H_1
\]

然后用：

\[
H_1
\]

产生 Cross-Attention Query。

可以直觉化成：

> **先弄清自己现在需要什么，再去输入 Memory 里查。**

---

# 7. Cross-Attention 中 Q/K/V 来自哪里？

原始论文 Section 3.2.3 明确说明：

\[
\boxed{
Q
\leftarrow
\text{previous decoder layer / decoder-side representation}
}
\]

\[
\boxed{
K,V
\leftarrow
\text{encoder output}
}
\]

更精确地：

\[
Q=H_1W_Q
\]

\[
K=MW_K
\]

\[
V=MW_V
\]

然后：

\[
\boxed{
CrossAttention(H_1,M)
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
\]

---

# 8. 为什么 Cross-Attention 输出长度跟 Decoder 走？

如果 Decoder 有：

\[
N_q
\]

个 positions，

Encoder Memory 有：

\[
N_m
\]

个 tokens。

则：

\[
Q:
[N_q,d_k]
\]

\[
K:
[N_m,d_k]
\]

所以：

\[
QK^\top:
[N_q,N_m]
\]

再乘：

\[
V:
[N_m,d_v]
\]

得到：

\[
\boxed{
[N_q,d_v]
}
\]

因此 Decoder 仍保持：

\[
N_q
\]

个输出 positions。

Memory 长度只决定：

> 每个 Query 可以从多少个 source positions 中读取。

---

# 9. Decoder Cross-Attention 不是把 Encoder Memory 替换掉

Encoder Memory：

\[
M
\]

保持存在。

Decoder 每个 position 只是从中读取：

\[
c_i
=
\sum_j
\alpha_{ij}v_j
\]

多个 Decoder positions：

> 可以反复读取同一 Memory。

Memory 不会因为被读过一次就被消费掉。

所以它很像：

> differentiable read-only memory。

---

# 10. Cross-Attention 后为什么还需要 Residual？

当前 decoder state：

\[
H_1
\]

已经包含 output-side context。

Cross-Attention 返回：

\[
C
\]

主要带入：

> source-side information。

Residual：

\[
H_1+C
\]

可以理解成：

\[
\boxed{
\text{我当前的输出侧状态}
+
\text{我刚从 Encoder Memory 读到的信息}
}
\]

这非常符合 Decoder 的功能。

---

# 11. 最后 FFN 为什么仍然不可少？

经过 Cross-Attention，

每个 Decoder position 已拥有：

- target-side context；
- source-side context。

但这些 feature 还需要进一步做：

> nonlinear per-position transformation。

所以：

\[
H_2
\rightarrow
FFN
\]

继续进行：

\[
d_{\text{model}}
\rightarrow
d_{ff}
\rightarrow
d_{\text{model}}
\]

的特征加工。

因此三段职责可以非常清楚地写成：

\[
\boxed{
\text{Self-Attention}
=
\text{输出 positions 之间交流}
}
\]

\[
\boxed{
\text{Cross-Attention}
=
\text{输出 positions 读取 Encoder Memory}
}
\]

\[
\boxed{
\text{FFN}
=
\text{每个输出 position 做 nonlinear feature transform}
}
\]

---

# 12. 一层 Decoder 的信息流

可以画成：

```text
Decoder hidden H
      │
      ▼
Masked Self-Attention
      │
      ▼
Residual + LayerNorm
      │
      ▼
      H₁
      │
      │
      ├─────────────── Query
      │
      ▼
Cross-Attention ◀──────── Encoder Memory M
      │                      │
      │                      ├── Key
      │                      └── Value
      ▼
Residual + LayerNorm
      │
      ▼
      H₂
      │
      ▼
Position-wise FFN
      │
      ▼
Residual + LayerNorm
      │
      ▼
Decoder Layer Output H₃
```

---

# 13. 原始 Transformer Decoder 有多少层？

原始 2017 Transformer：

\[
\boxed{
N=6
}
\]

Decoder stack 由：

> 6 个相同结构的 Decoder Layers

堆叠。

再次强调：

> 相同结构 ≠ 参数共享。

每层拥有自己的：

- Self-Attention parameters；
- Cross-Attention parameters；
- FFN；
- LayerNorms。

---

# 14. 为什么要堆 6 层？

一层 Cross-Attention 已经能读取整个 Encoder Memory。

所以深度不是为了：

> “第六层才能终于看到远处 input。”

而是为了：

> 多轮 representation refinement。

第一层：

> 根据初始 decoder state 做第一轮 target/self + source read。

第二层：

> 基于已经融入 source context 的 representation 再次产生新的 Queries，重新读取 Memory。

如此反复。

---

# 15. 第二层 Cross-Attention 和第一层有什么不同？

第一层：

\[
Q^{(1)}
=
H_1^{(1)}W_Q^{(1)}
\]

第二层输入已经是：

\[
H^{(2)}
\]

其中包含第一层：

- Self-Attention；
- Cross-Attention；
- FFN；

的结果。

所以：

\[
Q^{(2)}
\]

已经代表一个更成熟的 decoder state。

同时第二层还有自己的：

\[
W_Q^{(2)},W_K^{(2)},W_V^{(2)}
\]

因此它可以：

> 重新解释 Encoder Memory。

---

# 16. Decoder 是一种 Iterative Retrieval Process

一个有用的 mental model：

```text
第 1 层：
初步知道自己需要什么
→ 第一次读取 Memory

第 2 层：
基于读到的信息重新形成需求
→ 再次读取 Memory

第 3 层：
继续 refinement
...
```

这不是论文规定每层有固定语义阶段，

但很适合理解深层 Cross-Attention 为什么有价值。

---

# 17. 原始语言 Decoder 为什么第一个 Self-Attention 必须 Causal？

因为它的 Decoder input 中包含：

> shifted ground-truth target tokens。

如果 full Self-Attention，

较早 position 就会看到更右边的真实答案。

因此原论文明确把非法 future connections 在 Softmax 前设成：

\[
-\infty
\]

保证：

\[
\boxed{
P(y_i)
\text{ only depends on known outputs before }i
}
\]

详见：

- [Causal Mask](./causal-mask.md)

---

# 18. Cross-Attention 为什么不需要同样的 Causal Mask？

因为 Encoder source：

> 在翻译开始前已经完整已知。

例如输入：

```text
I love robots
```

Decoder 正在生成：

```text
我 ...
```

读取 source 中：

```text
robots
```

并不算偷看 target future。

所以 Decoder Cross-Attention 通常可以读取：

> 全部 Encoder positions。

---

# 19. Decoder Output 最后怎样变成 Token Probability？

原始 Transformer：

Decoder 最后一层输出：

\[
H^{(N)}
\]

每个 target position：

\[
h_i
\in
\mathbb R^{512}
\]

经过 learned linear projection：

\[
z_i=W_oh_i+b
\]

映射到：

> vocabulary logits。

再：

\[
softmax(z_i)
\]

得到：

> next-token distribution。

所以 Transformer Decoder Layer 本身：

> 不直接输出文字。

它输出 hidden representations。

真正 token prediction 还需要：

> vocabulary output head。

---

# 20. 这点和 ACT 非常相似

ACT Decoder 最后一层也不直接输出：

\[
14
\]

维机器人动作。

它先输出：

\[
h_i^{decoder}
\in
\mathbb R^{512}
\]

然后 action head：

\[
\boxed{
Linear:
512\rightarrow14
}
\]

得到：

\[
\hat a_i
\]

所以：

```text
语言 Transformer:
Decoder hidden
→ vocabulary head
→ token logits

ACT:
Decoder hidden
→ action head
→ 14-D target joint position
```

Decoder 提供的是：

> output-slot representation。

最终 task head 决定：

> representation 被解释成什么。

---

# 21. 现在正式进入 ACT Decoder

ACT 论文 Section IV-C：

Policy Encoder 产生：

\[
\boxed{
1202\times512
}
\]

observation memory。

Decoder 输入：

\[
\boxed{
k\times512
}
\]

的 output-position representations。

Decoder 通过 Cross-Attention：

> 读取 Encoder output。

最后得到：

\[
\boxed{
k\times512
}
\]

再投影：

\[
\boxed{
k\times14
}
\]

对应未来：

\[
k
\]

步 target joint positions。

---

# 22. ACT Decoder 的任务和语言 Decoder 根本不同

语言 Decoder：

> 生成一个离散 token sequence。

ACT Decoder：

> 生成一个连续 action-vector sequence。

语言输出：

\[
y_i\in\text{Vocabulary}
\]

ACT 输出：

\[
a_i\in\mathbb R^{14}
\]

但二者共享：

> Decoder 用 Query-side representations 读取 Encoder Memory。

---

# 23. ACT 的 Action Query Slot 是什么？

如果：

\[
k=100
\]

ACT 建立：

\[
100
\]

个 action query positions。

可以直觉化：

```text
query 0
→ chunk 中第 0 个 future action slot

query 1
→ chunk 中第 1 个 future action slot

...

query 99
→ chunk 中第 99 个 future action slot
```

它们不是动作本身。

它们是：

> **用于生成对应 future action 的 latent output slots。**

---

# 24. Paper 如何描述这些 Queries？

ACT 论文正文写：

> Transformer Decoder 的 input sequence 是 fixed position embedding，维度 \(k\times512\)。

Appendix C 进一步写：

> 第一层的 queries 是 fixed sinusoidal embeddings。

所以论文层面的设计是：

\[
\boxed{
\text{fixed positional action-query representations}
}
\]

---

# 25. Released Code 又是什么？

当前官方仓库：

```python
self.query_embed =
    nn.Embedding(
        num_queries,
        hidden_dim
    )
```

所以 released code：

\[
\boxed{
\text{learned query embeddings}
}
\]

而不是论文附录描述的 fixed sinusoidal first-layer queries。

这是一个明确的：

\[
\boxed{
\text{Paper vs Released-Code difference}
}
\]

---

# 26. 不管 Fixed 还是 Learned，核心作用是什么？

让：

\[
k
\]

个 output slots：

> 彼此有不同 identity。

否则如果：

\[
q_0=q_1=\cdots=q_{k-1}
\]

并且其他初始状态完全对称，

Decoder 很难知道：

> 哪一个应该代表较早动作，哪一个代表较晚动作。

所以 action query 的重要功能是：

\[
\boxed{
\text{output-slot identity}
}
\]

---

# 27. ACT Official Code 为什么还有一个 tgt？

当前 Transformer forward：

```python
query_embed =
    query_embed.unsqueeze(1).repeat(1, bs, 1)

tgt =
    torch.zeros_like(query_embed)
```

因此最开始有两件东西：

### `tgt`

\[
\boxed{
0
}
\]

初始化的 decoder content state。

### `query_embed`

\[
\boxed{
\text{learned slot identity}
}
\]

作为：

> `query_pos`

传给 Decoder。

---

# 28. 为什么 `tgt=0` 还能工作？

这是一个很重要的问题。

因为 Decoder 第一层并不是只拿：

\[
tgt
\]

做所有事情。

Self-Attention 里的 Q/K：

```python
q = k = tgt + query_pos
```

第一层时：

\[
tgt=0
\]

所以：

\[
q=k=query\_pos
\]

因此每个 action slot 已经能通过：

> query position / embedding

彼此区分。

---

# 29. 第一层 Self-Attention 的 Value 是什么？

官方代码：

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

第一层：

\[
tgt=0
\]

所以 Value：

\[
V
\]

来自 zero content states。

这意味着：

> 第一层 Self-Attention 本身无法从 zero Values 中产生丰富 content。

它主要建立 query-side structural computation，

真正重要的 sample-specific information随后由：

> Cross-Attention

从 observation memory 注入。

---

# 30. 那第一层 Self-Attention 输出是不是严格 0？

忽略 projection bias / implementation细节时，

如果 Value input：

\[
tgt=0
\]

则 projected Values 可以受 linear biases 影响。

PyTorch `MultiheadAttention` 默认 projection 通常带 bias，

因此不能简单断言整个 self-attention output mathematically 必为严格零。

但在高层结构上，

第一层真正的 sample-specific observation content：

> 并不来自 `tgt`，

而是后面的 Cross-Attention Memory。

所以不要把 zero `tgt` 误解成：

> “Decoder 完全没有输入。”

它还有：

- query slot identity；
- Encoder Memory。

---

# 31. ACT 第一层 Cross-Attention 才真正读取 Observation

官方 Post-LN：

```python
tgt2 =
    self.multihead_attn(
        query =
            tgt + query_pos,

        key =
            memory + pos,

        value =
            memory
    )[0]
```

所以：

\[
\boxed{
Q
\leftarrow
tgt+query\_pos
}
\]

\[
\boxed{
K
\leftarrow
memory+pos
}
\]

\[
\boxed{
V
\leftarrow
memory
}
\]

这一步把：

> 当前 observation

真正注入 action slots。

---

# 32. ACT Cross-Attention Matrix 是多大？

论文常用 chunk size：

\[
k=100
\]

Policy Encoder Memory：

\[
1202
\]

positions。

每个 head：

\[
Q:
[100,64]
\]

\[
K:
[1202,64]
\]

所以：

\[
\boxed{
QK^\top:
[100,1202]
}
\]

8 heads：

\[
\boxed{
[8,100,1202]
}
\]

忽略 batch。

---

# 33. 每一行代表什么？

例如：

\[
A_{17,:}
\]

表示：

> 第 17 个 future action slot 在当前 head 中如何分配对 1202 个 observation-memory positions 的读取权重。

因此第 17 个 action slot 可以：

- 看 wrist-camera features；
- 看 top-view features；
- 看 joint context；
- 看 latent context；

具体如何组合由当前 learned attention决定。

---

# 34. 为什么 Memory Token 已经不是 Raw Patch？

因为它先经过了 Policy Transformer Encoder。

所以：

\[
m_j
\]

已经是：

> contextualized observation representation。

即使它起源于某个 visual spatial token，

也可能已经通过 Encoder Self-Attention 融入：

- 其他摄像头；
- qpos；
- latent \(z\)；

的信息。

因此 Cross-Attention 读的是：

\[
\boxed{
\text{contextual observation memory}
}
\]

---

# 35. ACT Decoder Self-Attention 为什么不是 Causal？

当前官方 Transformer 调用：

```python
hs =
    self.decoder(
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

而 Decoder 的默认：

```python
tgt_mask=None
```

所以：

\[
\boxed{
\text{ACT action-slot self-attention is non-causal}
}
\]

---

# 36. 为什么这不会偷看 Future Action？

因为 Decoder 中没有：

> shifted ground-truth action sequence。

它的 positions 是：

> query slots。

slot 20 里面并没有真实：

\[
a_{t+20}
\]

给 slot 5 去抄。

它们只是：

> 待生成的 hidden representations。

所以：

\[
slot_5
\leftrightarrow
slot_{20}
\]

不会造成 target leakage。

---

# 37. 这和语言 Transformer 的区别

语言 Decoder training：

```text
<BOS> y₁ y₂ y₃ ...
```

右侧 positions 真正包含 ground-truth target tokens。

所以必须：

> causal mask。

---

ACT：

```text
query₀ query₁ query₂ ...
```

这些 positions 不包含真实 action target。

所以可以：

> full self-attention。

---

# 38. ACT Decoder 更接近 DETR-style Query Decoder

ACT 官方模型代码直接由 DETR-style architecture 修改而来。

DETR：

```text
object queries
↓
Decoder
↓
read image memory
↓
object slots
```

ACT：

```text
action queries
↓
Decoder
↓
read observation memory
↓
action slots
```

所以 ACT Decoder 更适合理解成：

\[
\boxed{
\text{query-based structured output decoder}
}
\]

而不是：

\[
\boxed{
\text{GPT-like autoregressive action generator}
}
\]

---

# 39. 为什么 Query Decoder 很适合 ACT？

因为 ACT 本来就想一次输出：

\[
k
\]

个未来动作。

那么自然可以准备：

\[
k
\]

个 output slots。

每个 slot：

- 有自己的 position identity；
- 可以和其他 action slots 交流；
- 可以读取 observation memory；
- 最后投影成 14-D action。

这与 Action Chunking 目标非常契合。

---

# 40. ACT Decoder 第一层的完整 Post-LN 流程

定义：

\[
T^{(0)}=0
\]

query positions：

\[
QPos
\]

Memory：

\[
M
\]

Memory positional representation：

\[
P
\]

---

## Sub-layer 1：Self-Attention

\[
q=k=
T^{(0)}+QPos
\]

\[
S^{(1)}
=
MHA_{self}
(
q,k,V=T^{(0)}
)
\]

然后：

\[
\boxed{
U^{(1)}
=
LN_1(
T^{(0)}
+
Dropout(S^{(1)})
)
}
\]

---

# 41. Sub-layer 2：Cross-Attention

Query side：

\[
U^{(1)}+QPos
\]

Memory Key side：

\[
M+P
\]

Value：

\[
M
\]

所以：

\[
C^{(1)}
=
MHA_{cross}
(
Q=U^{(1)}+QPos,
K=M+P,
V=M
)
\]

然后：

\[
\boxed{
V^{(1)}
=
LN_2(
U^{(1)}
+
Dropout(C^{(1)})
)
}
\]

---

# 42. Sub-layer 3：FFN

ACT：

\[
512
\rightarrow
3200
\rightarrow
512
\]

所以：

\[
F^{(1)}
=
W_2
ReLU(
W_1V^{(1)}+b_1
)
+b_2
\]

带 dropout。

最后：

\[
\boxed{
T^{(1)}
=
LN_3(
V^{(1)}
+
Dropout(F^{(1)})
)
}
\]

这就是第一层 output。

---

# 43. 第二层为什么不再是 zero tgt？

第二层 input：

\[
T^{(1)}
\]

已经包含：

- 第一层 Self-Attention update；
- 第一层 Cross-Attention observation content；
- FFN transformation。

所以第二层的 action slots：

> 已经是 observation-conditioned hidden representations。

它们不再只是空 slot identity。

---

# 44. 第二层 Self-Attention 做的事更有意义

现在：

\[
T^{(1)}_i
\]

已经包含第 \(i\) 个 action slot 第一轮 observation理解。

第二层 Self-Attention可以让：

\[
slot_i
\]

读取：

\[
slot_j
\]

已经形成的内容。

所以从第二层开始，

action-to-action information exchange 就非常直观：

> 不同 future time slots 互相协调隐藏表示。

---

# 45. 然后第二层再次 Cross-Attend Memory

第二层根据新的：

\[
T^{(1)}
\]

构建 Query。

所以：

\[
A^{(2)}
\]

可以和：

\[
A^{(1)}
\]

完全不同。

这意味着第 17 个 future slot：

> 第一次读取 observation 后，可以在第二层基于新的理解重新决定该看哪里。

---

# 46. 这就是 Deep Decoder 的 Refinement

可以把 7 层 ACT Decoder 直觉化为：

```text
Layer 1:
slot identity
→ 初步读取 observation

Layer 2:
已有初步 action context
→ 再协调 action slots
→ 再读 observation

Layer 3:
继续 refinement
...
```

注意：

> 这只是结构直觉。

不是说每一层被人为分配固定功能。

---

# 47. ACT 论文 Decoder 有多少层？

ACT Table III：

\[
\boxed{
\#decoder\ layers=7
}
\]

同时：

\[
\boxed{
hidden\ dimension=512
}
\]

\[
\boxed{
feedforward\ dimension=3200
}
\]

\[
\boxed{
heads=8
}
\]

\[
\boxed{
dropout=0.1
}
\]

chunk size 论文典型：

\[
\boxed{
100
}
\]

---

# 48. 为什么 ACT Decoder 比 Encoder 更深？

论文配置：

\[
4
\]

Encoder layers，

\[
7
\]

Decoder layers。

原论文没有在 Table III 旁给出一个严格理论证明：

> 为什么必须 4/7。

所以不能编造：

> “因为动作生成一定比视觉编码复杂，所以数学上需要 7 层。”

更准确：

> 这是作者选择并使用的实验 architecture hyperparameter。

---

# 49. Released Code 的 Layer Stack

官方：

```python
decoder_layer =
    TransformerDecoderLayer(...)

self.decoder =
    TransformerDecoder(
        decoder_layer,
        num_decoder_layers,
        decoder_norm,
        return_intermediate=...
    )
```

内部：

```python
self.layers =
    _get_clones(
        decoder_layer,
        num_layers
    )
```

其中 `_get_clones`：

```python
copy.deepcopy(module)
```

所以 7 层：

> 结构相同，但参数独立。

---

# 50. Decoder Norm 有一个和 Encoder 不同的代码细节

当前 ACT/DETR-style `Transformer`：

```python
decoder_norm =
    nn.LayerNorm(d_model)
```

无论：

```python
normalize_before
```

是否开启，

Decoder stack 都配置：

> final decoder norm。

而 Encoder：

```python
encoder_norm =
    LayerNorm(d_model)
    if normalize_before
    else None
```

所以 released code：

> Decoder stack 有一个额外 final norm object。

---

# 51. 为什么 Return Intermediate？

官方：

```python
return_intermediate_dec=True
```

所以 Decoder 每一层 output 都会被收集。

概念上：

\[
T^{(1)},
T^{(2)},\ldots,T^{(7)}
\]

都可以返回。

这继承自 DETR 的设计：

> DETR 经常对不同 decoder stages 做 auxiliary prediction / analysis。

---

# 52. ACT Official Decoder 返回什么 Shape？

Decoder：

```python
return torch.stack(intermediate)
```

如果：

\[
L=7
\]

内部 convention：

\[
[k,B,512]
\]

那么 stack：

\[
\boxed{
[7,k,B,512]
}
\]

随后 Transformer：

```python
hs = hs.transpose(1, 2)
```

得到：

\[
\boxed{
[7,B,k,512]
}
\]

---

# 53. 接下来 detr_vae.py 做了什么？

当前官方：

```python
hs =
    self.transformer(
        ...
    )[0]
```

因此当 Transformer 返回：

\[
[7,B,k,512]
\]

时：

\[
[0]
\]

会选择：

\[
\boxed{
[B,k,512]
}
\]

对应 stack 的第一个 intermediate entry。

然后：

```python
a_hat =
    self.action_head(hs)
```

得到：

\[
[B,k,14]
\]

---

# 54. 这里存在一个值得单独记录的 Released-Code Concern

官方 GitHub 长期存在 issue 指出：

> 当前 `return_intermediate_dec=True` 时，`self.transformer(...)[0]` 会选中第一个 Decoder Layer 的 intermediate output，而不是最后一个。

Issue #25、#52 等都讨论过这一行为。

这与论文 Table III 写：

\[
7
\]

个 Decoder Layers，

以及“Transformer Decoder predicts action sequence”这种标准 stack 理解之间，

存在值得审查的实现疑点。

---

# 55. 应该怎样严谨地写这个问题？

最安全的表达不是：

> “论文就是错的。”

也不是：

> “GitHub issue 一定证明代码 bug。”

而是分成三层：

### Paper Fact

论文配置：

\[
7
\]

decoder layers。

---

### Released-Code Fact

当前 `transformer.py`：

> 返回所有 intermediate decoder outputs。

当前 `detr_vae.py`：

> 对返回 tensor 使用 `[0]`。

---

### Community-Reported Concern

GitHub issue 提出：

> 这可能导致 action head 实际使用第一层 output，而非最终 decoder-layer output。

官方仓库当前代码中该行为仍可观察到。

这才是严谨的写法。

---

# 56. 为什么不能直接把社区 Issue 当成论文事实？

GitHub issue 是：

> 用户/社区分析。

它不是：

- 原论文；
- 作者正式勘误；
- peer-reviewed result。

所以知识库应该记录：

> “当前代码行为 + 社区 concern”

而不是替作者作最终裁决。

---

# 57. 为什么这个实现细节对理解理论仍然重要？

因为理论 Decoder stack：

\[
T^{(1)}
\rightarrow
T^{(2)}
\rightarrow
\cdots
\rightarrow
T^{(7)}
\]

的意义就是：

> 多轮 refinement。

如果 task head 真的只使用：

\[
T^{(1)}
\]

那后面层对这个主 action prediction path 的作用就值得重新审查。

所以这是：

> code audit 中非常重要的问题。

但它应该放在：

> “Official Implementation Note”

而不是用来改写 Decoder 的 canonical theory。

---

# 58. 本文主线应该以什么为准？

Canonical Transformer Decoder：

> 按原始 Transformer paper 理解完整 stack。

ACT Architecture：

> 按 ACT paper 的 7-layer decoder设计理解方法。

Released code：

> 单独标记当前实现行为。

这样可以同时满足：

- 原理准确；
- 论文忠实；
- 代码真实；
- 不掩盖 discrepancy。

---

# 59. ACT Decoder 最后如何变成 Action？

理想结构主线：

\[
T^{(7)}
\in
[B,k,512]
\]

经过：

\[
\boxed{
action\_head:
512\rightarrow14
}
\]

逐 slot：

\[
\hat a_i
=
W_aT_i+b_a
\]

得到：

\[
\boxed{
\hat A
\in
[B,k,14]
}
\]

这就是 predicted action chunk。

---

# 60. 为什么 Action Head 在所有 Slots 共享参数？

`nn.Linear(512,14)` 会对：

\[
k
\]

个 slots 使用同一套：

\[
W_a,b_a
\]

不同 future positions 的输出之所以不同，

不是因为：

> 每个 slot 有不同 action head。

而是因为它们的 hidden representations：

\[
T_i
\]

不同。

---

# 61. Slot Difference 从哪里来？

主要来自：

- query positional identity；
- Self-Attention；
- Cross-Attention；
- 多层 hidden refinement。

最终 shared action head：

> 把每个 slot 的不同 representation 用同一规则映射到 14-D action space。

这和 Transformer language model：

> 所有 positions 共享同一个 vocabulary projection

的思想类似。

---

# 62. Action Head 为什么不需要自己知道 t+i？

因为：

\[
T_i
\]

已经通过 query position / decoder computation 编码了：

> 当前是哪个 output slot。

所以 shared head只需：

> 把当前 hidden representation 翻译成动作向量。

---

# 63. ACT 还有一个 is_pad_head

官方：

```python
self.is_pad_head =
    nn.Linear(hidden_dim, 1)
```

Decoder hidden 还可以投影为：

> padding prediction。

不过当前经典 ACT policy loss 主线主要使用：

- action reconstruction；
- KL；

而 `is_pad_head` 在 released policy path 中并不是我们理解 Decoder 原理的核心。

因此 canonical Decoder page 不应把它放在主线中心。

---

# 64. Transformer Decoder Output 和 CVAE Decoder 是不是一回事？

ACT 论文会说：

> CVAE decoder，即 policy。

这里的：

\[
\text{CVAE decoder}
\]

是大概念：

> 从 \(z+\) observation 生成 action sequence 的整个 policy network。

它包含：

- ResNet；
- Policy Transformer Encoder；
- Transformer Decoder；
- action head。

---

而：

\[
\text{Transformer Decoder}
\]

只是这个大 policy 中的一个具体模块。

所以：

\[
\boxed{
\text{CVAE Decoder}
\neq
\text{Transformer Decoder}
}
\]

这两个 “decoder” 不要混。

---

# 65. 为什么这个名字非常容易混乱？

ACT 同时存在：

### CVAE Encoder

推 posterior \(z\)。

### CVAE Decoder / Policy

预测 action sequence。

而 CVAE Decoder / Policy 内部又包含：

### Transformer Encoder

构建 observation memory。

### Transformer Decoder

生成 action-slot hidden representations。

所以层级是：

```text
ACT CVAE

├── CVAE Encoder
│   └── Transformer Encoder
│
└── CVAE Decoder / Policy
    ├── ResNet
    ├── Transformer Encoder
    ├── Transformer Decoder
    └── Action Head
```

这张关系必须牢牢记住。

---

# 66. Transformer Decoder 与 CVAE 的概率角色不同

CVAE层面：

\[
p_\theta(
A
\mid
o,z
)
\]

表示：

> 条件生成 policy。

Transformer Decoder只是实现：

> 这个条件生成函数内部的一段神经网络。

所以 Transformer Decoder 本身不自动意味着：

> probabilistic decoder。

概率语义来自整个 CVAE建模。

---

# 67. ACT 推理时 z=0 会怎样进入 Decoder？

先：

\[
z=0
\]

经过：

\[
latent\_out\_proj
\]

得到：

\[
512
\]

维 latent token。

这个 token 和：

- joint；
- visual tokens；

一起进入 Policy Encoder。

所以 \(z\) 不是：

> 直接塞给每个 action query。

它先参与：

> observation memory 构建。

Decoder 随后通过 Cross-Attention间接读取：

> 已经被 z condition 的 Encoder Memory。

---

# 68. 所以 Decoder Query 不直接等于 z

另一个容易混淆的点。

Action Query：

> 表示 output slot identity。

Latent \(z\)：

> 表示 CVAE latent style condition。

二者职责完全不同。

流程：

```text
z
↓
latent token
↓
Policy Encoder
↓
Memory

action query
↓
Transformer Decoder
↓
Cross-Attend Memory
```

所以：

\[
\boxed{
z\neq action\ query
}
\]

---

# 69. Joint State 也不是 Action Query

joint：

\[
q_t
\]

被投影成：

> observation token。

它是 Memory side information。

Action query是：

> Decoder side output slot representation。

所以：

```text
joint token:
我当前机器人状态是什么

action query:
我要生成未来第 i 个动作
```

这两类 token 位于不同角色。

---

# 70. 为什么 Decoder 能同时利用 Image、Joint 和 z？

因为 Policy Encoder 已经把三者放进统一 Memory：

\[
M
\]

Decoder Cross-Attention只需面对：

> 一个统一的 contextual memory sequence。

它不需要自己知道：

> “这部分先走 ResNet，那部分来自 qpos。”

这些 modality structure已经通过 Encoder representations 和 positions 进入 Memory。

---

# 71. Decoder Self-Attention 有什么作用？

ACT 论文只高层说：

> Transformer Decoder generates a coherent action sequence。

从 architecture 来看，

non-causal action-slot Self-Attention允许：

\[
slot_i
\]

与：

\[
slot_j
\]

直接交换信息。

这为 sequence-wide coordination 提供机制。

但不要进一步编造：

> “Self-Attention mathematically guarantees smooth trajectories。”

论文没有这种定理。

更准确：

> 它提供 action-slot interaction capacity，coherence 是训练任务希望形成的结果。

---

# 72. 为什么 Action Chunk 的 Temporal Order 不会因为 Full Self-Attention 消失？

因为每个 slot 有自己的：

> query positional identity。

所以虽然：

\[
slot_5
\]

可以读取：

\[
slot_{20}
\]

模型仍能区分：

> 谁代表较早 future time，谁代表较晚 future time。

Full interaction：

> 不等于无顺序。

Position 与 Causality 是两件事。

---

# 73. 为什么 ACT 不需要语言式 Shifted Action Input？

语言 Transformer：

> previous generated token 本身就是生成下一 token 的重要显式输入。

ACT：

> 当前 observation + output slot queries 已经用于一次预测整个 chunk。

Ground-truth previous actions 不被作为：

> shifted target decoder input。

所以不存在：

```text
<BOS>, a₀, a₁, ...
```

这种 teacher-forced policy-decoder input。

---

# 74. 这避免了 Chunk 内 Autoregressive Rollout

如果 ACT 这样做：

```text
先预测 a₀
↓
把 a₀ 喂回去
↓
预测 a₁
...
```

chunk 内又产生：

\[
O(k)
\]

sequential generation dependency。

而 ACT 的 query decoder：

> 一次 forward 并行产生所有 action slots。

这更符合 Action Chunking：

> 一次预测一段动作

的设计目标。

---

# 75. Non-Autoregressive 不意味着独立预测

这是必须再次强调的点。

如果只是：

\[
k
\]

个完全独立 MLP heads，

那每个动作 slot 没有 interaction。

ACT Decoder：

> 有 Self-Attention。

因此动作 slots 是：

\[
\boxed{
\text{parallel but interacting}
}
\]

而不是：

\[
\boxed{
\text{parallel and independent}
}
\]

---

# 76. Joint Prediction 的高层形式

可以写成：

\[
\boxed{
\hat A
=
f_\theta(
o_t,z
)
}
\]

其中：

\[
\hat A
=
[
\hat a_t,
\hat a_{t+1},
\ldots,
\hat a_{t+k-1}
]
\]

整个 matrix 同时输出。

这和 autoregressive：

\[
\prod_i
p(
a_i
\mid
a_{<i},o
)
\]

是不同的结构选择。

---

# 77. ACT 训练时 Decoder 输入会不会使用 Ground-Truth Action？

Policy Transformer Decoder：

> 不直接使用。

Ground-truth action chunk：

> 用于 CVAE encoder 推 \(z\)，

并用于：

> L1 reconstruction target。

所以 target action确实会影响 training forward，

但路径是：

```text
ground-truth action chunk
↓
CVAE Encoder
↓
z
↓
Policy Encoder
↓
Memory
↓
Decoder
```

而不是：

```text
ground-truth action chunk
↓
shift
↓
Policy Decoder input
```

---

# 78. 为什么这仍然不是 Deployment Leakage？

因为 CVAE Encoder：

> training-only。

推理时：

\[
z=0
\]

不需要未来 action。

所以整个 deployed policy input仍是：

- current images；
- current qpos；
- fixed latent zero condition。

未来真实 action：

> 不存在于 inference input。

---

# 79. ACT Decoder Training 和 Inference 结构几乎一致吗？

Policy Decoder这部分：

> 基本是一致的。

训练：

\[
Memory(o_t,z_{\text{posterior sample}})
\]

推理：

\[
Memory(o_t,z=0)
\]

然后同一个：

- action queries；
- Decoder；
- action head；

生成 chunk。

主要差别在上游 latent：

> training sample posterior vs inference zero。

而不是 Decoder 由 teacher forcing 切换成 autoregressive generation。

---

# 80. 为什么这使 ACT 推理很快？

一次 Policy forward：

> 同时得到 \(k\) 个动作预测。

不需要在 neural network 内：

> 为 100 个动作运行 100 次 sequential decoder forward。

当然实际闭环系统仍可能：

> 每个环境 timestep 重新 query policy，

尤其 Temporal Ensemble 开启时。

但那是 control-loop level 的重复，

不是 chunk 内 autoregressive decoding。

---

# 81. Decoder 一次输出 k 个动作，为什么最后只执行一个？

如果使用 Temporal Ensemble：

> 每 timestep 都重新预测一个 chunk。

然后当前绝对 timestep 收集所有 overlapping predictions，

加权平均后：

> 只执行当前 action。

所以：

\[
\boxed{
\text{Decoder output}
=
\text{future action chunk}
}
\]

而：

\[
\boxed{
\text{robot execution}
=
\text{current action only}
}
\]

二者层级不同。

---

# 82. Decoder 与 Temporal Ensemble 的关系

Decoder：

> 单次 forward 内生成未来 sequence。

Temporal Ensemble：

> 多次 forward 之间融合相同绝对时间的 prediction。

所以：

```text
Decoder:
intra-forward structured prediction

Temporal Ensemble:
inter-forward execution-time aggregation
```

---

# 83. Decoder 与 Action Chunking 的关系

Action Chunking规定：

\[
o_t
\rightarrow
a_{t:t+k-1}
\]

这是：

> policy target form。

Transformer Decoder是：

> 实现这个 structured output 的 architecture。

所以 Action Chunking 并不理论上要求 Transformer Decoder。

ACT 只是选择：

> Query-based Transformer Decoder

来实现它。

---

# 84. Decoder 的输出是不是完整 Action Plan？

可以把它直觉化为：

> 一个 local future action plan。

但要加限定。

ACT 并没有显式 dynamics model、cost function、search algorithm。

所以它不是经典 model-based planning。

更准确：

> **它是由 imitation policy 一次预测出的 future action chunk。**

“local plan”只是一种帮助理解的类比。

---

# 85. Transformer Decoder 和 Planner 不要混淆

Planner通常可能显式：

- rollout dynamics；
- evaluate cost；
- search trajectories。

ACT Decoder：

> 直接从 learned policy mapping 产生 action sequence。

所以：

\[
\boxed{
\text{sequence prediction}
\neq
\text{explicit planning}
}
\]

---

# 86. ACT Decoder 的 Shape Flow：假设 k=100

Memory：

\[
\boxed{
M:
[B,1202,512]
}
\]

Query Embedding：

\[
\boxed{
QPos:
[100,512]
}
\]

repeat batch 后：

\[
[100,B,512]
\]

`tgt`：

\[
\boxed{
[100,B,512]
}
\]

初始化为 zero。

---

# 87. Decoder Self-Attention Shape

8 heads：

每 head：

\[
Q,K,V:
[B,8,100,64]
\]

Self-Attention score：

\[
\boxed{
[B,8,100,100]
}
\]

因为 ACT 不 causal：

> 100 × 100 全部 slot pairs 都可以交互。

---

# 88. Cross-Attention Shape

Query：

\[
[B,8,100,64]
\]

Memory K/V：

\[
[B,8,1202,64]
\]

score：

\[
\boxed{
[B,8,100,1202]
}
\]

输出每 head：

\[
[B,8,100,64]
\]

concat：

\[
[B,100,512]
\]

---

# 89. FFN Shape

\[
[B,100,512]
\]

第一 Linear：

\[
\boxed{
[B,100,3200]
}
\]

ReLU + dropout，

第二 Linear：

\[
\boxed{
[B,100,512]
}
\]

---

# 90. 一层结束 Shape

仍：

\[
\boxed{
[B,100,512]
}
\]

7 层理论 stack：

\[
[B,100,512]
\rightarrow
\cdots
\rightarrow
[B,100,512]
\]

最终 action head：

\[
\boxed{
[B,100,14]
}
\]

---

# 91. 为什么 Decoder 不改变 Query Count？

和 Cross-Attention 原理一样：

> 每个 query slot 始终对应一个 output position。

Self-Attention：

\[
100\rightarrow100
\]

Cross-Attention：

\[
100\text{ queries}
\times1202\text{ memory}
\rightarrow100
\]

FFN：

> per slot。

所以整个 Decoder stack一直保留：

\[
k
\]

个 output slots。

---

# 92. 为什么 Encoder Memory Length 也不变？

Decoder只是读取 Memory。

它不会把：

\[
M
\]

修改回 Encoder。

所以：

\[
1202
\]

仍然只是 Cross-Attention K/V bank。

Decoder output count：

\[
100
\]

和 Memory count：

\[
1202
\]

彼此独立。

---

# 93. 为什么 Decoder Layer 需要自己的 Self-Attention 和 Cross-Attention 两套 MHA 参数？

官方代码：

```python
self.self_attn =
    nn.MultiheadAttention(...)

self.multihead_attn =
    nn.MultiheadAttention(...)
```

它们是两个不同 modules。

原因很自然：

### Self-Attention

建立：

> action-slot ↔ action-slot

matching。

### Cross-Attention

建立：

> action-slot ↔ observation-memory

matching。

这两个任务不应该被迫共用一套：

\[
W_Q,W_K,W_V
\]

---

# 94. Cross-Attention 里的 Q/K/V Width 一样吗？

每个 head：

\[
d_k=d_v=64
\]

在 ACT 标准配置下。

但 sequence length 不一样：

Self：

\[
100
\leftrightarrow100
\]

Cross：

\[
100
\leftrightarrow1202
\]

所以区别主要在：

> source set。

---

# 95. Why Self-Attention Before Cross-Attention in ACT?

对于 ACT 第一层，

因为 `tgt` 初始接近空 content，

第一层 Self-Attention更多提供：

> output-slot structural interaction。

Cross-Attention才注入 observation content。

之后层中，

Self-Attention可以先协调：

> 已经 observation-conditioned 的 action slots，

再根据新的 slot states重新读取 Memory。

这是很自然的 iterative structure。

---

# 96. 如果把 Cross-Attention 放 Self-Attention 前面会怎样？

那是另一种 architecture。

数学上不是不可能。

但原始 Transformer / DETR-style Decoder选择：

\[
\boxed{
Self
\rightarrow
Cross
\rightarrow
FFN
}
\]

网络会围绕这一 computation order 训练。

所以 canonical Decoder 应按照实际 architecture 理解，

而不是认为三个 sub-layers 可以任意交换。

---

# 97. 为什么三次 Residual 都有意义？

### Self Residual

保留当前 slot state：

\[
T+Self(T)
\]

---

### Cross Residual

保留已有 output-side state：

\[
U+Cross(U,M)
\]

---

### FFN Residual

保留 contextual state：

\[
V+FFN(V)
\]

每个 sub-layer都更像：

> 在当前 representation 上写入一个 update。

---

# 98. Decoder LayerNorm 和 Encoder 一样是对 Feature Dimension

每个 slot：

\[
x_i
\in
\mathbb R^{512}
\]

LayerNorm 对：

\[
512
\]

features 求 mean/variance。

不会在：

- 100 action slots；
- 1202 memory tokens；
- batch samples；

之间求平均。

所以 Decoder 的 LayerNorm同样不会：

> 混合不同 action positions。

---

# 99. Decoder Cross-Attention 才是 Observation→Action 的主桥梁

ACT 中：

Encoder：

> 把 observation 建成 Memory。

Decoder：

> 把 action slots 条件化在 Memory 上。

所以从信息流来看：

\[
\boxed{
Observation
\rightarrow
Encoder Memory
\rightarrow
Cross-Attention
\rightarrow
Action Slots
}
\]

这是 ACT 从 perception/state representation 到 action sequence最核心的神经网络桥梁。

---

# 100. 但不要把 Cross-Attention 当成唯一 Observation 信息路径

Action slots 接收到 observation后，

后续 layers 的 Self-Attention会让：

> 一个 slot 已经读取到的 observation context

传播给其他 slots。

所以多层 Decoder 中 observation information可以通过：

- direct Cross-Attention；
- indirect slot-to-slot Self-Attention；

共同传播。

---

# 101. 一个 Slot 的最终表示依赖哪些东西？

第 \(i\) 个最终 Decoder slot：

\[
T_i^{(L)}
\]

理论上可以依赖：

1. 自己的 query identity；
2. 其他 action slots；
3. Encoder Memory 的所有 allowed positions；
4. 所有 preceding Decoder layers 的 transformations；
5. FFN nonlinear computation。

所以最终：

\[
T_i^{(L)}
\]

是一个复杂的：

> future-action-position-specific, observation-conditioned representation。

---

# 102. 为什么 Shared Action Head 足够？

因为复杂性已经在：

\[
T_i^{(L)}
\]

里。

Action head只需把：

\[
512
\]

维 representation 映射到：

\[
14
\]

维 robot joint target。

所以最终 Linear 可以很简单。

这也是深度学习常见设计：

> 大模型负责 representation，最后 task head 可以很浅。

---

# 103. 为什么输出是 Absolute Joint Positions？

ACT 论文明确：

> action space 是两个机器人 arms 的 absolute joint positions，共 14-D。

作者也报告：

> 使用 delta joint positions 会降低性能。

所以 action head：

\[
14
\]

维不是：

- torque；
- velocity；
- delta pose；

而是：

> target joint positions。

低层 PID 再执行这些 targets。

---

# 104. Decoder 不负责 PID

Transformer Decoder输出：

\[
\hat a_t
\]

是高层 target joint configuration。

真实 actuator控制由：

> low-level controller

完成。

所以：

\[
\boxed{
Transformer Decoder
\neq
motor torque controller
}
\]

这对理解 robot policy stack 很重要。

---

# 105. Decoder 预测 k 个动作，不代表机器人一次 Open-Loop 执行 k 步

ACT 原始 naïve chunking可以：

> 每 k 步 query 一次并执行完整 chunk。

但最终 ACT 使用 Temporal Ensemble：

> 每 timestep 都可以重新 query policy。

因此 Decoder 每次仍输出 k-step future chunk，

系统却保持高频重新观察。

所以：

\[
\boxed{
\text{chunk prediction}
\neq
\text{necessarily chunk execution}
}
\]

---

# 106. Decoder 里有没有 Temporal Ensemble？

**没有。**

Temporal Ensemble发生在 Decoder 输出之后：

```text
policy forward
↓
predicted action chunk
↓
store overlapping predictions
↓
temporal weighted average
↓
execute current action
```

Decoder内部只负责：

> 产生单次 chunk prediction。

---

# 107. Decoder Loss 在哪里计算？

Decoder output经 action head：

\[
\hat A
\]

和 ground truth：

\[
A
\]

计算 reconstruction loss。

官方 ACT policy 使用：

\[
L1
\]

再加 CVAE：

\[
\beta KL
\]

所以梯度：

\[
L
\rightarrow
action\_head
\rightarrow
Decoder
\rightarrow
Encoder
\]

可以端到端回传。

---

# 108. Cross-Attention 的 Gradient 也会训练 Encoder

因为 Decoder output依赖：

\[
K=MW_K
\]

\[
V=MW_V
\]

而：

\[
M
\]

来自 Policy Encoder。

所以 reconstruction loss 会通过 Cross-Attention：

> 回传到 Encoder Memory，

进而训练：

- Policy Transformer Encoder；
- qpos projection；
- latent projection；
- image feature projections；
- ResNet backbone。

这说明 Encoder 与 Decoder不是分开独立训练。

---

# 109. Decoder Query Embeddings 也会被训练

released code：

```python
nn.Embedding(num_queries, hidden_dim)
```

所以 action query slots：

> 是 trainable parameters。

最终 loss 可以更新：

\[
QPos_i
\]

让每个 output slot学到：

> 对自己的 future position有用的 representation identity。

---

# 110. Query Embedding 会不会直接记住一个固定动作？

理论上参数中可以编码 task prior，

但最终 output强烈依赖：

> Cross-Attention observation memory。

所以不能简单说：

> query 0 存的就是“手向左移动”。

更准确：

> query embedding提供 slot-specific learned prior / identity；sample-specific action content通过整个 Decoder尤其 Cross-Attention形成。

---

# 111. 同一个 Query Slot 在所有样本都一样吗？

初始 learned embedding：

> 是共享的。

例如 slot 20：

\[
qpos_{20}
\]

对所有 batch samples相同。

但 Cross-Attention面对的：

\[
M_b
\]

每个样本不同。

所以最终：

\[
T_{b,20}
\]

会不同。

这就是：

> fixed/learned slot identity + sample-dependent memory

的组合。

---

# 112. 为什么 Query Slot 数等于 Chunk Size？

因为：

> 一个 Query Slot 对应一个 Decoder output position。

ACT 要输出：

\[
k
\]

个 future actions，

所以：

\[
\boxed{
num\_queries=k
}
\]

代码：

```python
self.query_embed =
    nn.Embedding(
        num_queries,
        hidden_dim
    )
```

中的 `num_queries` 就来自：

> chunk size。

---

# 113. 如果 k 从 100 改成 50，会发生什么？

Query count：

\[
100\rightarrow50
\]

所以：

- decoder self-attention：\(100\times100\rightarrow50\times50\)；
- cross-attention：\(100\times1202\rightarrow50\times1202\)；
- output：\(100\times14\rightarrow50\times14\)。

这说明 Action Chunking hyperparameter：

> 会直接改变 Decoder sequence length。

---

# 114. k 也会影响 CVAE Encoder

Training 时 target action sequence length也是：

\[
k
\]

所以 CVAE encoder input：

\[
k+2
\]

也会改变。

因此 chunk size 同时影响：

- training latent encoder sequence；
- policy decoder query count；
- output length；
- temporal overlap structure。

它是 ACT 的核心 architecture/task hyperparameter。

---

# 115. 原始 Transformer 与 ACT Decoder 对比

| 特性 | Original Transformer Decoder | ACT Decoder |
|---|---|---|
| 输出 | language tokens | 14-D robot actions |
| Query-side input | shifted target embeddings | action query slots |
| Self-Attention | causal | non-causal |
| Cross-Attention | reads encoder source | reads observation memory |
| Output count | target sequence positions | fixed chunk size \(k\) |
| Prediction mode | autoregressive | parallel chunk |
| Final head | vocab linear + softmax | linear \(512\to14\) |
| Decoder layers | 6 | 7 |
| hidden dim | 512 | 512 |
| FFN dim | 2048 | 3200 |
| heads | 8 | 8 |

---

# 116. 它们真正共享的核心是什么？

虽然 task 完全不同，

两者都保留：

\[
\boxed{
\text{Self-Attention}
\rightarrow
\text{Cross-Attention}
\rightarrow
\text{FFN}
}
\]

这意味着 Decoder abstraction并不属于语言。

它本质是：

> **一组 output-side representations，通过内部交互并读取外部 Encoder Memory，逐层形成条件输出 representations。**

语言只是其中一种应用。

---

# 117. 常见误解一：Transformer Decoder 就是 Autoregressive Decoder

**错误。**

是否 autoregressive取决于：

- decoder input；
- mask；
- objective。

ACT 就是 non-autoregressive decoder。

---

# 118. 常见误解二：Decoder 比 Encoder 多一个 Self-Attention

**错误。**

Encoder 和 Decoder都有 Self-Attention。

Decoder额外的是：

\[
\boxed{
Cross-Attention
}
\]

---

# 119. 常见误解三：Cross-Attention 只是第二次 Self-Attention

**错误。**

Self：

\[
Q,K,V
\]

来自同一 Decoder sequence。

Cross：

\[
Q
\]

来自 Decoder，

\[
K,V
\]

来自 Encoder Memory。

---

# 120. 常见误解四：Decoder 的 Query Embedding 就是最终 Attention Q

**不严格。**

它先参与 decoder representation / query position，

MultiheadAttention内部还有：

\[
W_Q
\]

projection。

---

# 121. 常见误解五：ACT 的 Action Query 一开始就是 14-D Action

**错误。**

它是：

\[
512
\]

维 hidden slot identity。

真正 action 直到：

\[
action\_head:
512\to14
\]

才产生。

---

# 122. 常见误解六：tgt=0 表示 Decoder 没有输入

**错误。**

它还有：

- query embeddings；
- Encoder Memory。

Cross-Attention会注入 sample-specific observation content。

---

# 123. 常见误解七：ACT Decoder 的 100 Slots 独立预测

**错误。**

它们通过 Self-Attention彼此交互。

---

# 124. 常见误解八：因为 Slots 代表未来，所以必须 Causal

**错误。**

关键不是“代表未来”，而是：

> 是否包含推理时不可见的 ground-truth future information。

ACT Query Slots 不包含。

---

# 125. 常见误解九：ACT Ground-Truth Action Chunk 被喂进 Policy Decoder

**错误。**

它用于：

- CVAE encoder；
- reconstruction target。

不是 shifted policy-decoder input。

---

# 126. 常见误解十：Decoder 输出 k×512 就已经是动作

**错误。**

还要：

\[
512\rightarrow14
\]

task head。

---

# 127. 常见误解十一：Encoder Memory 有 1202 个位置，所以 Decoder 也输出 1202 个位置

**错误。**

Decoder output count由：

\[
k
\]

个 Queries 决定。

---

# 128. 常见误解十二：Decoder 每层共享参数

**错误。**

`copy.deepcopy` 创建独立 module layers。

---

# 129. 常见误解十三：ACT Decoder FFN 是 2048

**论文/典型配置不是。**

ACT Table III：

\[
\boxed{
3200
}
\]

---

# 130. 常见误解十四：ACT Decoder 只有 1 层

**论文配置明确是 7 层。**

但 released code目前的 intermediate-output indexing行为需要单独审查，不能把“代码 action head 当前取 `[0]`”反推成论文方法只有一层。

---

# 131. 常见误解十五：GitHub Issue 说是 bug，所以我们可以直接改写官方算法

**不严谨。**

应该区分：

- paper definition；
- released code；
- community concern。

---

# 132. 常见误解十六：Cross-Attention 会改变 Encoder Memory

**不会。**

它读取 Memory，

更新的是 Decoder hidden representations。

---

# 133. 常见误解十七：Cross-Attention 权重就是“最终模型解释”

**错误。**

只是某层、某 head 的中间 routing weights。

---

# 134. 常见误解十八：Decoder 负责控制 Motor Torque

**错误。**

ACT输出的是：

> target joint positions。

低层控制器执行。

---

# 135. 常见误解十九：Decoder 一次输出 100 步，所以机器人一定 Open-Loop 跑 100 步

**错误。**

Temporal Ensemble 模式下 policy 可每 timestep 重新 query。

---

# 136. 常见误解二十：CVAE Decoder = Transformer Decoder

**错误。**

ACT 的 CVAE Decoder是整个 policy。

Transformer Decoder只是 policy 内部模块。

---

# 137. 用三块记住 Decoder

## Block 1：Self-Attention

\[
\boxed{
\text{output positions talk to each other}
}
\]

---

## Block 2：Cross-Attention

\[
\boxed{
\text{output positions read Encoder Memory}
}
\]

---

## Block 3：FFN

\[
\boxed{
\text{each output position transforms its own features}
}
\]

每块外面：

\[
\boxed{
Residual + LayerNorm
}
\]

原始 Transformer就是这样构成一层 Decoder。

---

# 138. 用一个公式记住原始 Post-LN Decoder

\[
\boxed{
H_1
=
LN_1(
H+
SelfAttn(H)
)
}
\]

\[
\boxed{
H_2
=
LN_2(
H_1+
CrossAttn(H_1,M)
)
}
\]

\[
\boxed{
H_3
=
LN_3(
H_2+
FFN(H_2)
)
}
\]

其中 language Decoder 的：

\[
SelfAttn
\]

带 causal mask。

---

# 139. 用一个公式记住 ACT Decoder

设：

\[
T^{(0)}=0
\]

Action Query Positions：

\[
P_q
\]

Observation Memory：

\[
M
\]

Memory Position：

\[
P_m
\]

一层概念上：

\[
\boxed{
U=
LN_1(
T+
SelfMHA(
Q=T+P_q,
K=T+P_q,
V=T
)
)
}
\]

\[
\boxed{
V=
LN_2(
U+
CrossMHA(
Q=U+P_q,
K=M+P_m,
V=M
)
)
}
\]

\[
\boxed{
T'=
LN_3(
V+
FFN(V)
)
}
\]

然后理论上层层迭代。

最终：

\[
\boxed{
\hat A
=
Linear_{512\rightarrow14}(T^{(L)})
}
\]

---

# 140. 一句话真正理解 Transformer Decoder

> **Transformer Decoder 是一个 output-side representation refinement stack：每一层先让各输出位置通过 Self-Attention整合自身 sequence/slot context，再通过 Cross-Attention用当前输出状态作为 Query 去读取 Encoder Memory，最后用 position-wise FFN 做非线性特征加工；三个 sub-layers 都通过 residual connection 与 LayerNorm稳定地写回当前 decoder representation。**

---

# 141. 一句话真正理解 ACT Decoder

> **ACT 把语言 Transformer 的 autoregressive target decoder 改造成了一个 non-causal query-based action decoder：\(k\) 个 action slots 以 learned/fixed positional query identity 开始，彼此可以 Full Self-Attend，然后每层都通过 Cross-Attention读取 \(1202\)-token observation memory，并经过 FFN逐层形成 observation-conditioned future-action representations，最后由共享的 \(512\rightarrow14\) action head并行输出整个 action chunk。**

---

# 142. 到这里 Transformer 主骨架已经闭环

现在你应该已经可以从头读懂：

```text
Input Tokens
↓
Positional Information
↓
Transformer Encoder
    ├── Self-Attention
    ├── Residual + Norm
    ├── FFN
    └── Residual + Norm
↓
Encoder Memory
↓
Transformer Decoder
    ├── Self-Attention
    ├── Residual + Norm
    ├── Cross-Attention
    ├── Residual + Norm
    ├── FFN
    └── Residual + Norm
↓
Task Head
```

而 ACT 将其具体化为：

```text
Images + qpos + z
↓
Policy Transformer Encoder
↓
Observation Memory
↓
k Action Queries
↓
Transformer Decoder
↓
k Hidden Action Slots
↓
Linear
↓
k × 14 Action Chunk
```

---

# 143. 下一步：Feed-Forward Network

Transformer 大骨架已经讲完。

但其中还有一个经常被严重低估的模块：

> **FFN。**

下一篇：

> **[Feed-Forward Network：Attention 已经交换信息了，为什么还需要 MLP？](./feed-forward-network.md)**

会专门深入：

- 为什么 Position-wise FFN 不做 token mixing；
- 为什么它仍然占 Transformer 大量参数和 FLOPs；
- \(512\rightarrow2048\rightarrow512\) / ACT 的 \(512\rightarrow3200\rightarrow512\) 到底增加了什么表示能力；
- 为什么没有 activation 时两层 Linear 可以合并；
- ReLU/GELU/SwiGLU 分别改变什么；
- 为什么很多现代 LLM 的参数反而主要在 FFN；
- FFN 可以怎样理解成 per-token key-value feature memory；
- ACT 中 FFN 如何作用于 visual memory 和 action slots。

---

## Primary Source：Transformer

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,  
Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin.

**Attention Is All You Need.**  
NeurIPS 2017.

- arXiv: https://arxiv.org/abs/1706.03762
- HTML: https://arxiv.org/html/1706.03762

### Section 3.1 — Decoder

原论文明确写：

- Decoder 也是：
  \[
  N=6
  \]
  层；
- 相比 Encoder，每层额外插入：
  > Multi-Head Attention over Encoder Stack output；
- 每个 sub-layer 外：
  > residual connection followed by LayerNorm；
- Decoder Self-Attention被 mask；
- 配合 shifted output embeddings，保证 position \(i\) 只依赖此前已知 outputs。

---

### Section 3.2.3 — Encoder–Decoder Attention

原论文明确：

\[
\boxed{
Q\leftarrow Decoder
}
\]

\[
\boxed{
K,V\leftarrow Encoder
}
\]

并指出每一个 Decoder position 可以 attend：

> 所有 input-sequence positions。

---

### Section 3.3 — FFN

Decoder 和 Encoder 都使用：

\[
\boxed{
FFN(x)
=
\max(
0,
xW_1+b_1
)W_2+b_2
}
\]

Base Transformer：

\[
d_{\text{model}}=512
\]

\[
d_{ff}=2048
\]

---

## ACT Primary Source

Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.

**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.**  
RSS 2023.

- arXiv: https://arxiv.org/abs/2304.13705
- HTML: https://arxiv.org/html/2304.13705v1

ACT Section IV-C：

- Policy使用：
  - ResNet image encoders；
  - Transformer Encoder；
  - Transformer Decoder；
- Encoder input：
  \[
  1202\times512
  \]
- Decoder通过 Cross-Attention condition on Encoder output；
- Decoder input/query sequence：
  \[
  k\times512
  \]
- K/V来自 Encoder；
- Decoder output：
  \[
  k\times512
  \]
- 最终投影：
  \[
  k\times14
  \]

Appendix C 进一步写：

> Encoder outputs are used as both keys and values in Transformer Decoder cross-attention layers；

并描述：

> first-layer queries 是 fixed sinusoidal embeddings。

---

## ACT Hyperparameters

ACT Table III：

\[
\boxed{
\#encoder\ layers=4
}
\]

\[
\boxed{
\#decoder\ layers=7
}
\]

\[
\boxed{
feedforward\ dimension=3200
}
\]

\[
\boxed{
hidden\ dimension=512
}
\]

\[
\boxed{
heads=8
}
\]

\[
\boxed{
chunk\ size=100
}
\]

\[
\boxed{
dropout=0.1
}
\]

---

## Official ACT Implementation

Repository:

https://github.com/tonyzhaozh/act

### `detr/models/detr_vae.py`

Current released code：

```python
self.query_embed =
    nn.Embedding(
        num_queries,
        hidden_dim
    )
```

即：

> learned action-query embeddings。

forward：

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

然后：

```python
a_hat =
    self.action_head(hs)
```

---

### `detr/models/transformer.py`

Decoder 初始化：

```python
decoder_layer =
    TransformerDecoderLayer(
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        activation,
        normalize_before
    )
```

ACT build：

```python
return_intermediate_dec=True
```

---

### Decoder Initialization

```python
query_embed =
    query_embed.unsqueeze(1).repeat(1, bs, 1)

tgt =
    torch.zeros_like(query_embed)

memory =
    self.encoder(...)

hs =
    self.decoder(
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

因此：

\[
\boxed{
tgt\_mask=None
}
\]

ACT action-query Self-Attention是：

> non-causal。

---

### Post-LN Decoder Layer

当前默认 `normalize_before=False` 时：

```python
q = k =
    tgt + query_pos

tgt2 =
    self.self_attn(
        q,
        k,
        value=tgt,
        ...
    )[0]

tgt =
    norm1(
        tgt + dropout1(tgt2)
    )

tgt2 =
    self.multihead_attn(
        query=tgt + query_pos,
        key=memory + pos,
        value=memory,
        ...
    )[0]

tgt =
    norm2(
        tgt + dropout2(tgt2)
    )

tgt2 =
    linear2(
        dropout(
            activation(
                linear1(tgt)
            )
        )
    )

tgt =
    norm3(
        tgt + dropout3(tgt2)
    )
```

---

## Released-Code Intermediate Output Note

Current official `TransformerDecoder` with：

```python
return_intermediate=True
```

collects each decoder-layer output and returns:

```python
torch.stack(intermediate)
```

ACT `build_transformer(...)` explicitly sets:

```python
return_intermediate_dec=True
```

After transpose, this produces a tensor whose leading dimension corresponds to decoder-layer intermediates.

Current `detr_vae.py` then calls:

```python
self.transformer(...)[0]
```

which selects the first leading entry.

Official GitHub issues including:

- Issue #25
- Issue #52

have raised the concern that this means current action prediction uses the first returned decoder-layer representation rather than the final one.

This should be recorded as:

\[
\boxed{
\text{Released-Code Behavior / Community-Reported Concern}
}
\]

not silently merged into the paper definition.

The ACT paper itself specifies a:

\[
7\text{-layer Transformer Decoder}
\]

as its architecture.

---

## 本文知识连接

### 前置

- [Transformer](./transformer.md)
- [Transformer Encoder](./transformer-encoder.md)
- [Self-Attention](./self-attention.md)
- [Cross-Attention](./cross-attention.md)
- [Multi-Head Attention](./multi-head-attention.md)
- [Causal Mask](./causal-mask.md)
- [Positional Encoding](./positional-encoding.md)

### Decoder Components

- [Feed-Forward Network](./feed-forward-network.md)
- [Residual Connection](./residual-connection.md)
- [Layer Normalization](./layer-normalization.md)
- [Dropout](./dropout.md)

### Sequence Modeling

- Autoregressive Modeling
- Teacher Forcing
- Non-Autoregressive Decoding

### Robot Learning

- [ACT Architecture](../robot-learning/act/architecture.md)
- [Action Chunking](../robot-learning/act/action-chunking.md)
- [CVAE in ACT](../robot-learning/act/cvae-in-act.md)
- [ACT Training](../robot-learning/act/training.md)
- [ACT Inference](../robot-learning/act/inference.md)
- [Temporal Ensemble](../robot-learning/act/temporal-ensemble.md)
- [ACT Complete Data Flow](../robot-learning/act/complete-data-flow.md)

### 下一步

- [Feed-Forward Network](./feed-forward-network.md)
