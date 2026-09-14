---
title: "ACT Architecture：从 4 张图像到 k × 14 个动作"
description: "基于 ACT 原论文与官方实现，逐层拆解 ResNet18、visual tokens、joint/z tokens、Transformer encoder、action queries、cross-attention、Transformer decoder 与最终 k×14 action chunk 的完整 tensor flow。"
status: reviewed
pageType: application
canonical: /robot-learning/act/architecture
updated: "2026-09-15"
---

# ACT Architecture：从 4 张图像到 $k\times14$ 个动作

前面我们已经理解了 ACT 的几个核心思想：

- [Action Chunking](./action-chunking.md)：为什么一次预测一段动作；
- [Temporal Ensemble](./temporal-ensemble.md)：为什么同一个 timestep 会有多个预测；
- [CVAE in ACT](./cvae-in-act.md)：训练时 latent $z$ 怎样得到；
- [为什么 ACT 推理时令 $z=0$](./why-z-zero-at-inference.md)。

现在终于可以回答一个更工程化的问题：

> **ACT 真正拿到 4 张相机图像、当前机器人关节状态和 latent $z$ 后，到底怎样一步一步输出未来 $k$ 个动作？**

ACT 原论文给出的 policy 可以高层写成：

$$
\boxed{
\pi_\theta
\left(
\hat a_{t:t+k}
\mid
o_t,z
\right)
}
$$

其中 observation：

$$
o_t
=
(
I_t^{(1)},
I_t^{(2)},
I_t^{(3)},
I_t^{(4)},
q_t
)
$$

包含：

- 4 路 RGB camera images；
- 当前双臂 joint positions。

最终输出：

$$
\hat A_t
\in
\mathbb R^{k\times14}
$$

也就是未来 $k$ 个 timestep 的双臂 target joint positions。

这一篇会严格沿着 tensor 流动顺序来拆：

```text
4 RGB Images
↓
ResNet18
↓
Visual Feature Maps
↓
Visual Tokens
         ┐
Joint ───┤
z ───────┘
↓
Transformer Encoder
↓
Memory
↑
k Action Queries
↓
Transformer Decoder
↓
k Hidden Features
↓
Action Head
↓
k × 14 Actions
```

---

## 1. 先把整个 ACT Policy 看成一个函数

暂时忽略 CVAE training encoder。

真正执行动作的 policy / CVAE decoder 可以写成：

$$
f_\theta(
I_1,I_2,I_3,I_4,q,z
)
=
\hat A
$$

输入：

#### 四张图片

$$
I_i
\in
\mathbb R^{3\times480\times640}
$$

#### 当前 Joint State

ALOHA 两条机械臂各 7 个 joint dimensions：

$$
q
\in
\mathbb R^{14}
$$

#### Latent Style Variable

官方实现：

$$
z
\in
\mathbb R^{32}
$$

输出：

$$
\hat A
\in
\mathbb R^{k\times14}
$$

如果原论文默认：

$$
k=100
$$

那么一次 policy forward 的 action output 就是：

$$
\boxed{
100\times14
}
$$

也就是 100 个未来双臂 target joint vectors。

---

## 2. 为什么需要 Transformer？

ACT 输入并不是一种单一数据。

它同时拥有：

```text
Top Camera
Front Camera
Left Wrist Camera
Right Wrist Camera
Joint State
Latent z
```

而输出又不是一个 action：

```text
a_t
```

而是一整段：

$$
a_t,
a_{t+1},
\ldots,
a_{t+k-1}
$$

所以 ACT 同时面临两个问题：

### 输入侧

如何融合：

- 多视角视觉；
- proprioception；
- latent style；

这些异构信息？

### 输出侧

如何一次生成：

> 具有时间结构的 $k$ 个 actions？

ACT 的设计是：

```text
Transformer Encoder
→ 融合 observation context

Transformer Decoder
→ 生成 k 个 action positions
```

这和原始 Transformer 的：

```text
Encoder
→ 编码 source sequence

Decoder
→ 根据 source memory 生成 target sequence
```

有结构上的相似性。

---

## 3. 第一部分：4 张 RGB 图像

ACT 原论文使用四个 camera viewpoints：

1. top camera；
2. front camera；
3. left wrist camera；
4. right wrist camera。

每一张原始图像分辨率：

$$
480\times640
$$

RGB channel：

$$
3
$$

因此单张图片可以表示为：

$$
\boxed{
I
\in
\mathbb R^{3\times480\times640}
}
$$

如果加 batch dimension：

$$
I
\in
\mathbb R^{B\times3\times480\times640}
$$

四路 camera：

$$
\boxed{
I_{\mathrm{all}}
\in
\mathbb R^{B\times4\times3\times480\times640}
}
$$

---

## 4. 为什么不能直接把原始 Pixels 扔给 Transformer？

一张：

$$
480\times640
$$

图片包含：

$$
480\times640
=
307200
$$

个 pixel locations。

四张就是：

$$
1,228,800
$$

个 pixel locations。

如果把每个 pixel 都变成 Transformer token：

> sequence 会极其长。

Self-attention 的标准计算量近似：

$$
O(N^2)
$$

其中：

$$
N
$$

是 sequence length。

因此 ACT 先使用 CNN：

> **ResNet18**

把高分辨率 pixels 压缩成较小的 spatial feature map。

---

## 5. ResNet18 在这里做什么？

对于每张：

$$
480\times640\times3
$$

RGB image，

论文写明经过 ResNet18 后得到：

$$
\boxed{
15\times20\times512
}
$$

feature map。

用 channel-first tensor 表示：

$$
\boxed{
[512,15,20]
}
$$

加 batch：

$$
[B,512,15,20]
$$

这意味着：

> 原来 $480\times640$ 的细密 pixel grid，被 ResNet 压缩成 $15\times20$ 个 spatial locations。

每一个 spatial location 不再只有 RGB 三个数。

而是一个：

$$
512
$$

维 feature vector。

---

## 6. 一个 Spatial Location 可以怎样理解？

假设 feature map：

$$
F
\in
\mathbb R^{15\times20\times512}
$$

其中：

$$
F_{i,j}
\in
\mathbb R^{512}
$$

可以把：

$$
F_{i,j}
$$

理解为：

> ResNet 对原图某个局部 receptive field 提取出的视觉 representation。

它可能包含与：

- edges；
- texture；
- object parts；
- gripper appearance；
- geometry；

有关的信息。

注意：

> 这不意味着某个 512 维 channel 有明确的人类语义。

它只是 learned visual representation。

---

## 7. 为什么是 15 × 20？

因为 ResNet18 会逐层下采样。

原始：

$$
480\times640
$$

空间尺寸最终缩小约 32 倍：

$$
480/32=15
$$

$$
640/32=20
$$

所以得到：

$$
15\times20
$$

spatial grid。

这也是标准 ResNet backbone 很常见的输出尺度。

---

## 8. 从 Feature Map 变成 Token Sequence

Transformer 习惯处理 sequence：

$$
[\text{token}_1,\text{token}_2,\ldots]
$$

而 ResNet 输出是二维 spatial grid：

$$
15\times20
$$

所以 ACT 把空间维度 flatten：

$$
15\times20
=
300
$$

于是单张图片：

$$
15\times20\times512
$$

变成：

$$
\boxed{
300\times512
}
$$

也就是说：

> 一张 camera image 变成 300 个 visual tokens。

每个 token：

$$
\in\mathbb R^{512}
$$

---

## 9. 四路 Camera 一共多少 Visual Tokens？

每个 camera：

$$
300
$$

个 token。

四个 camera：

$$
4\times300
=
1200
$$

因此所有视觉信息合起来：

$$
\boxed{
1200\times512
}
$$

如果加 batch：

$$
\boxed{
[B,1200,512]
}
$$

这就是 ACT Transformer policy 中最大的一部分 input sequence。

---

## 10. 官方代码实际上怎样拼 4 个 Camera？

论文可以理解成：

```text
Camera 1
300 tokens
+
Camera 2
300 tokens
+
Camera 3
300 tokens
+
Camera 4
300 tokens
↓
1200 visual tokens
```

官方代码稍微更具体：

```python
all_cam_features = []

for cam_id in range(num_cameras):
    features, pos = backbone(image[:, cam_id])
    features = input_proj(features)
    all_cam_features.append(features)

src = torch.cat(
    all_cam_features,
    axis=3
)
```

这里 camera feature maps 沿：

> width dimension

拼接。

原来单个 camera：

$$
[B,512,15,20]
$$

四个拼起来：

$$
[B,512,15,80]
$$

因为：

$$
20\times4=80
$$

然后 Transformer 内部 flatten：

$$
15\times80
=
1200
$$

得到同样的：

$$
1200
$$

visual sequence positions。

所以论文的“concatenate feature sequences”和代码的“沿 width 拼 feature maps 再 flatten”在 token 数量上是等价的。

---

## 11. 只有 Visual Feature 还不够：位置在哪里？

如果只给 Transformer 一组：

$$
1200
$$

个 feature vectors，

Transformer 自身并不知道：

> 哪个 feature 来自图像左上角？

> 哪个来自右下角？

> 哪些 token 在空间上相邻？

因为 vanilla self-attention 对 sequence order 本身没有天然空间概念。

所以 ACT 给 visual features 加：

> **2D sinusoidal positional embedding**

详细参见：

- [Positional Encoding](../../deep-learning/positional-encoding.md)

---

## 12. 为什么是 2D Position Encoding？

文本 Transformer 的 token 通常排列在一条一维序列：

$$
1,2,3,\ldots
$$

图像 feature map 却有两个坐标：

$$
(row,column)
$$

所以 visual position 更自然写成：

$$
(i,j)
$$

2D positional encoding 就是给 Transformer 提供：

> “这个视觉 feature 原本位于哪里”

的信息。

因此：

$$
\text{visual token}
+
\text{2D position embedding}
$$

共同进入 attention。

---

## 13. ResNet Feature Channel 和 Transformer Hidden Dimension 为什么恰好都是 512？

论文中：

- ResNet feature channel：512；
- Transformer hidden dimension：512。

但官方代码仍保留：

```python
self.input_proj = nn.Conv2d(
    backbone_channels,
    hidden_dim,
    kernel_size=1
)
```

它是一个：

> $1\times1$ convolution projection。

作用是把 backbone feature channels 映射到 Transformer 的：

$$
d_{\mathrm{model}}
$$

维度。

即使当前：

$$
512\rightarrow512
$$

这个 projection 仍然提供一层可学习映射，并让代码对其他 backbone channel dimensions 更通用。

---

## 14. 第二类输入：Current Joint Positions

图像只告诉模型：

> 外部世界长什么样。

机器人还必须知道：

> 自己现在的手臂处于什么姿态。

ALOHA 双臂 joint positions：

$$
q_t
\in
\mathbb R^{14}
$$

分别来自：

$$
7+7
$$

个 joint dimensions。

ACT 使用一个 Linear layer：

$$
\mathbb R^{14}
\rightarrow
\mathbb R^{512}
$$

官方代码：

```python
self.input_proj_robot_state =
    nn.Linear(14, hidden_dim)
```

所以得到：

$$
\boxed{
e_q
\in
\mathbb R^{512}
}
$$

可以把它视为：

> 一个 proprioception token。

---

## 15. 为什么 Joint State 只变成一个 Token？

一种可能设计是：

> 每个 joint 一个 token。

但 ACT 没有这么做。

它直接把整个：

$$
14
$$

维 joint vector 输入一个 Linear layer：

$$
q_t
\rightarrow
e_q
$$

因此整个 robot state 被表示成：

$$
1\times512
$$

一个 token。

这个设计意味着：

> Transformer 接收到的是“整个当前双臂姿态”的 learned embedding，而不是 14 个独立 joint tokens。

---

## 16. 第三类输入：Latent z

ACT 官方 latent dimension：

$$
z\in\mathbb R^{32}
$$

训练时：

$$
z
=
\mu+\sigma\odot\epsilon
$$

推理时：

$$
z=0
$$

不管来源怎样，

进入 policy 前都要经过：

$$
\mathbb R^{32}
\rightarrow
\mathbb R^{512}
$$

Linear projection。

官方代码：

```python
self.latent_out_proj =
    nn.Linear(
        latent_dim,
        hidden_dim
    )
```

得到：

$$
\boxed{
e_z
\in
\mathbb R^{512}
}
$$

可以把它看成：

> 一个 latent-style token。

---

## 17. 所以 Transformer Encoder 最终到底收到多少 Token？

现在有：

#### Visual Tokens

$$
1200\times512
$$

#### Latent Token

$$
1\times512
$$

#### Proprioception Token

$$
1\times512
$$

总共：

$$
1200+1+1
=
1202
$$

所以论文明确写：

$$
\boxed{
1202\times512
}
$$

如果加 batch：

$$
\boxed{
[1202,B,512]
}
$$

在 PyTorch `MultiheadAttention` 的 sequence-first 表示中。

---

## 18. 官方代码里的顺序是什么？

Transformer 代码中：

```python
addition_input = torch.stack(
    [latent_input, proprio_input],
    axis=0
)

src = torch.cat(
    [addition_input, src],
    axis=0
)
```

所以 sequence 顺序大致是：

```text
token 0:
latent z

token 1:
proprioception

token 2...:
visual features
```

即：

$$
[
e_z,
e_q,
v_1,
v_2,
\ldots,
v_{1200}
]
$$

论文主要强调内容，而不是这个顺序本身的语义。

---

## 19. Joint Token 和 z Token 也有 Position Embedding 吗？

有。

官方实现定义：

```python
self.additional_pos_embed =
    nn.Embedding(2, hidden_dim)
```

对应：

- latent token；
- proprioception token。

Transformer forward 中：

```python
additional_pos_embed =
    additional_pos_embed.unsqueeze(1)

pos_embed = torch.cat(
    [additional_pos_embed, visual_pos],
    axis=0
)
```

所以：

> visual tokens 使用 2D positional embeddings；

而：

> latent 和 joint 两个特殊 token 有各自 learned positional/type-like embeddings。

这有助于 Transformer 区分：

```text
“这是 z”
“这是 qpos”
“这是 image feature”
```

---

## 20. Transformer Encoder 在 ACT 中真正做什么？

现在 sequence：

$$
X
\in
\mathbb R^{1202\times512}
$$

进入 Transformer Encoder。

每一层主要包含：

1. Multi-Head Self-Attention；
2. Feed-Forward Network；
3. Residual connections；
4. LayerNorm。

在 self-attention 中，每个 token 都可以与其他 token 交换信息。

例如：

#### 一个 Left Wrist Visual Token

可以 attend：

- 其他 left wrist regions；
- front camera；
- top camera；
- right wrist camera；
- joint token；
- latent token。

#### Joint Token

也可以与所有 image tokens 交互。

因此论文说：

> Transformer encoder synthesizes information from different camera viewpoints, the joint positions, and the style variable.

这就是非常准确的总结。

---

## 21. 为什么多视角图像适合 Self-Attention？

考虑一个精细操作：

> 右手正在将电池插槽。

top camera 可能更容易看到：

> 电池整体位置。

wrist camera 可能更容易看到：

> gripper 和插槽的局部几何关系。

front camera 又可能提供：

> 深度方向上的相对位置线索。

如果分别处理完以后永不交互，

模型很难建立：

> “top camera 里的这个电池，就是 wrist camera 里现在接近 gripper 的那个东西。”

Transformer self-attention 提供一个全局的信息融合机制：

$$
\text{view}_1
\leftrightarrow
\text{view}_2
\leftrightarrow
\text{state}
\leftrightarrow
z
$$

注意：

> 这只是 architecture capability。

不代表 attention 一定学到某种特定几何匹配。

---

## 22. Encoder 输出什么？

输入：

$$
1202\times512
$$

经过 Transformer encoder 后，

shape 不变：

$$
\boxed{
M
\in
\mathbb R^{1202\times512}
}
$$

这个：

$$
M
$$

通常叫：

> **memory**

每个 memory token 已经经过多层 self-attention，

所以它不再只是原来的局部 feature。

而是 contextual representation。

---

## 23. 为什么叫 Memory？

因为 Transformer Decoder 后面不会直接再看原始：

- images；
- qpos；
- z。

它只通过 cross-attention 访问：

$$
M
$$

也就是说：

```text
raw observation
↓
Transformer Encoder
↓
Memory
↓
Transformer Decoder
```

Decoder 把 encoder memory 当成：

> 已经融合好的上下文信息库。

---

## 24. 现在进入最容易困惑的部分：Transformer Decoder 输入是什么？

ACT 要输出：

$$
k
$$

个 actions。

于是它需要：

$$
k
$$

个 output slots。

每个 slot 对应一个未来 action position：

```text
slot 0
→ 当前 timestep 的 action

slot 1
→ 下一 timestep

slot 2
→ 再下一 timestep

...

slot k-1
→ chunk 最后一个 action
```

ACT 使用：

> **action query embeddings**

来表示这 $k$ 个 output positions。

---

## 25. Query 在这里到底是什么意思？

先不要把它和自然语言问题混在一起。

这里的 query 本质是：

$$
q_i
\in
\mathbb R^{512}
$$

一个 embedding vector。

第 $i$ 个 query 可以直觉理解为：

> **“我要生成 action chunk 中第 $i$ 个位置的动作，请从 encoder memory 中取出对这个输出位置有用的信息。”**

例如：

$$
q_0
$$

对应：

> 第一个 action slot。

$$
q_{20}
$$

对应：

> chunk 中第 21 个 action slot。

注意：

> query embedding 本身一开始并不知道真正的 action 数值。

它只是区分：

> 不同 output positions。

---

## 26. 为什么不能只用一个 Query，然后一次输出 k × 14？

理论上可以设计别的网络。

但 ACT 的 Transformer decoder 采用：

> 一个 query 对应一个 output position

的序列生成方式。

于是：

$$
k
$$

个 query：

$$
Q
\in
\mathbb R^{k\times512}
$$

经过 decoder 后得到：

$$
H
\in
\mathbb R^{k\times512}
$$

再逐位置投影为：

$$
k\times14
$$

这样每个未来 timestep 都拥有自己的 hidden representation。

---

## 27. Decoder 一开始的 Content 是什么？

官方 Transformer 代码：

```python
tgt = torch.zeros_like(query_embed)
```

也就是说 decoder 初始 content：

$$
T_0
=
0
$$

shape：

$$
k\times512
$$

真正区分这些 output slots 的，是：

```python
query_pos = query_embed
```

所以 decoder 可以理解为：

```text
初始内容：
全 0

+

每个位置自己的 query embedding
↓
通过 self-attention + cross-attention
逐渐形成 action representations
```

这和 DETR-style object queries 的设计非常接近。

事实上 ACT 官方代码正是基于 DETR 实现改造的。

---

## 28. Decoder 中先有 Self-Attention

Transformer decoder layer 里首先：

$$
Q=K=T+\text{query\_pos}
$$

Value：

$$
V=T
$$

进行 self-attention。

这使不同 action slots 可以相互交流。

因此：

```text
future action 1
↔
future action 2
↔
future action 3
↔
...
```

它不是 $k$ 个完全独立的 regressor。

这正是论文所说：

> Transformer decoder generates a coherent action sequence

的架构基础之一。

---

## 29. 这里为什么没有 Causal Mask？

这是理解 ACT 和语言 Transformer 差异的关键。

语言生成通常：

> token $t$ 不允许看到未来 token。

因为 inference 时要 autoregressive：

$$
y_1
\rightarrow
y_2
\rightarrow
y_3
$$

所以需要 causal mask。

ACT 不是这样。

它一次性预测完整：

$$
k
$$

个 action positions。

官方代码没有给 decoder self-attention 传 causal `tgt_mask`。

因此所有 action slots 可以：

> **双向相互 attention。**

也就是说：

$$
h_i
$$

可以与：

$$
h_j
$$

交互，无论：

$$
i<j
$$

还是：

$$
i>j
$$

所以 ACT decoder：

> **不是 autoregressive decoder。**

---

## 30. 这点非常重要：ACT 一次并行输出整个 Chunk

ACT 不做：

```text
预测 aₜ
↓
把 aₜ 喂回 decoder
↓
预测 aₜ₊₁
↓
再喂回
...
```

而是：

```text
k queries
↓
Transformer decoder
↓
同时产生 k 个 hidden states
↓
同时得到 k 个 actions
```

因此：

$$
\boxed{
\text{ACT action generation is parallel, not autoregressive}
}
$$

这也使 action chunk inference 更快。

---

## 31. Cross-Attention：ACT 架构最核心的一步

Decoder self-attention 让 action positions 彼此交流。

但它还不知道：

> 当前机器人到底看到了什么。

所以接下来是：

> **Cross-Attention**

论文明确写：

> Transformer decoder conditions on encoder output through cross-attention.

这里：

#### Query

来自 decoder action slots。

#### Key

来自 encoder memory。

#### Value

也来自 encoder memory。

也就是：

$$
Q
=
\text{decoder action representation}
$$

$$
K
=
M+\text{encoder position}
$$

$$
V
=
M
$$

官方代码直接写：

```python
tgt2 = self.multihead_attn(
    query=self.with_pos_embed(
        tgt,
        query_pos
    ),
    key=self.with_pos_embed(
        memory,
        pos
    ),
    value=memory,
)[0]
```

这正好就是标准 cross-attention。

---

## 32. 用一句人话理解 Cross-Attention

第 $i$ 个 action query 在问：

> **“为了预测 future slot $i$ 的动作，我应该从当前 observation memory 的哪些部分读取信息？”**

它可以 attend：

- top camera 某区域；
- wrist camera 某区域；
- robot joint token；
- latent style token；
- 其他所有 encoder memory positions。

然后把这些信息组合成：

$$
h_i
$$

最终用于预测第 $i$ 个 action。

---

## 33. 一个例子

假设 chunk 中前半段是：

> 右手接近电池。

后半段是：

> 对准插槽并插入。

那么不同 action query 可能需要不同 observation clues。

较早 query 可能更依赖：

> 当前 gripper 和 battery 的相对位置。

较后 query 可能更需要：

> battery slot 的目标位置、当前全局姿态等。

Transformer decoder 允许每个 output slot：

> 用自己的 query 去读取同一份 encoder memory。

注意：

> 这是架构提供的能力。

我们不能未经 attention visualization 就断言模型一定形成上述精确分工。

---

## 34. Q、K、V 在 ACT Cross-Attention 里到底分别是什么？

这是非常适合直接记住的一张表。

| Cross-Attention 部分 | ACT 中来自哪里 |
|---|---|
| Query $Q$ | Transformer decoder 的 action query / current decoder state |
| Key $K$ | Transformer encoder memory + position embedding |
| Value $V$ | Transformer encoder memory |

所以：

$$
\boxed{
\text{Action Queries}
\;\xrightarrow{\text{search}}\;
\text{Observation Memory}
}
$$

如果不熟悉为什么 attention 要分 Q/K/V：

- [Query / Key / Value](../../deep-learning/qkv.md)

---

## 35. Encoder Self-Attention 的 QKV 又来自哪里？

在 policy encoder 中：

$$
Q=K=\text{src}+\text{position}
$$

$$
V=\text{src}
$$

其中 src 包含：

```text
z token
joint token
1200 visual tokens
```

所以这里是：

> **Self-Attention**

所有 token 都来自同一个 observation sequence。

而 decoder 中读取 memory 的第二个 attention 是：

> **Cross-Attention**

这是两者最大的区别。

---

## 36. Decoder 最后得到什么？

经过 Transformer decoder 后：

$$
H_{\mathrm{dec}}
\in
\mathbb R^{k\times512}
$$

每一个：

$$
h_i
\in
\mathbb R^{512}
$$

对应一个未来 action position。

然后使用 action head：

$$
\boxed{
\mathbb R^{512}
\rightarrow
\mathbb R^{14}
}
$$

官方代码：

```python
self.action_head =
    nn.Linear(
        hidden_dim,
        state_dim
    )
```

其中：

$$
hidden\_dim=512
$$

$$
state\_dim=14
$$

所以：

$$
a_i
=
W_ah_i+b_a
$$

最终：

$$
\boxed{
\hat A
\in
\mathbb R^{k\times14}
}
$$

---

## 37. 为什么 Action 是 14 维？

ALOHA 使用两条机械臂。

每条：

$$
7
$$

个 joint dimensions。

因此：

$$
7+7
=
14
$$

论文的 action 定义是：

> **两个机械臂下一 timestep 的 absolute target joint positions。**

所以：

$$
a_t
=
[
q_{L,1},\ldots,q_{L,7},
q_{R,1},\ldots,q_{R,7}
]
$$

shape：

$$
\mathbb R^{14}
$$

---

## 38. ACT 输出的不是 Joint Delta

这是论文明确强调的一个设计。

ACT 使用：

$$
\boxed{
\text{absolute target joint positions}
}
$$

而不是：

$$
\Delta q_t
=
q_{t+1}-q_t
$$

论文说他们实验观察到：

> 使用 delta joint positions 会降低性能。

所以 output head 的 14 维不是：

> “应该移动多少”。

而是：

> “下一时刻应该到达哪个 target joint position”。

---

## 39. ACT 输出的也不是 Motor Torque

ACT policy 不是低层 torque controller。

输出：

$$
14\text{-D target joint positions}
$$

之后由 Dynamixel motors 内部的：

> low-level high-frequency PID controller

去追踪这些目标。

所以控制链：

```text
ACT
↓
target joint position
↓
PID Controller
↓
motor command / physical motion
```

理解这一点很重要。

否则容易误以为 ACT 自己直接负责：

> contact force / current / torque 的高频闭环控制。

---

## 40. 把完整 Tensor Shape 串起来

现在用一个 batch size：

$$
B
$$

完整走一遍。

---

### RGB Images

$$
[B,4,3,480,640]
$$

拆每个 camera：

$$
[B,3,480,640]
$$

---

### ResNet18

每路：

$$
[B,512,15,20]
$$

---

### Four Cameras Concatenated

官方实现沿 width：

$$
[B,512,15,80]
$$

---

### Flatten for Transformer

$$
15\times80=1200
$$

得到：

$$
[1200,B,512]
$$

---

### Joint State

原始：

$$
[B,14]
$$

Linear：

$$
[B,512]
$$

变成 1 token：

$$
[1,B,512]
$$

---

### Latent z

原始：

$$
[B,32]
$$

Linear：

$$
[B,512]
$$

变成：

$$
[1,B,512]
$$

---

### Transformer Encoder Input

$$
\boxed{
[1202,B,512]
}
$$

---

### Transformer Encoder Memory

$$
\boxed{
[1202,B,512]
}
$$

---

### Action Queries

$$
\boxed{
[k,B,512]
}
$$

---

### Transformer Decoder Output

$$
\boxed{
[k,B,512]
}
$$

---

### Action Head

$$
512\rightarrow14
$$

得到：

$$
\boxed{
[B,k,14]
}
$$

这就是一次 ACT policy forward。

---

## 41. 如果 k = 100，会是什么规模？

论文默认：

$$
k=100
$$

那么 decoder 有：

$$
100
$$

个 action slots。

输出：

$$
[B,100,14]
$$

对单个 sample：

$$
100\times14
=
1400
$$

个连续 action values。

但它们不是 1400 个独立数字。

Transformer decoder 让不同 future positions 之间能够交互，

因此输出是一段：

> **jointly modeled action sequence**

---

## 42. 为什么固定 Query 能输出不同动作？

很容易疑惑：

> 每个 query 都只是一个 embedding，为什么它知道自己应该生成第几步？

因为：

$$
q_0,q_1,\ldots,q_{k-1}
$$

不是同一个 vector。

每个 position 都有自己的 query / position representation。

训练时：

$$
q_i
$$

对应的 decoder output 始终被监督为：

$$
a_{t+i}
$$

所以 optimization 会逐渐让：

$$
q_i
$$

承担：

> chunk 第 $i$ 个 action slot

的角色。

它不是人手写进去“你是第 37 步”。

而是通过：

- distinct position embeddings；
- fixed tensor position；
- supervised action target；

共同形成这种功能。

---

## 43. Query 本身包含当前图像信息吗？

一开始：

> **不包含。**

query embedding 本身只是 output-slot representation。

当前 observation information 来自：

$$
\text{encoder memory}
$$

通过 cross-attention 注入。

所以可以粗略分工：

```text
Query:
“我要预测哪个 output slot？”

Memory:
“当前世界是什么样？”

Cross-Attention:
“为了这个 slot，从当前世界读取什么？”
```

这是非常好用的理解方式。

---

## 44. 为什么 Query 不直接是 Future Timestep 数字？

理论上也可以编码成显式 timestep scalar。

ACT / DETR-style architecture 使用：

$$
512
$$

维 positional/query representation，

给模型更大的 learned representation space。

它可以让不同 output positions 在高维空间拥有不同身份。

关键不是：

> query vector 的每一维具体是什么意义。

关键是：

> 每个 action slot 有一个可区分的位置表示，并在训练中被赋予对应 timestep 的监督。

---

## 45. Action Queries 会互相交流吗？

会。

因为 Transformer decoder 在 cross-attention 之前还有 self-attention。

所以：

$$
q_i
$$

对应的 decoder state 可以读取：

$$
q_j
$$

对应的 decoder state。

这提供了：

> action sequence 内部建模

的能力。

所以 ACT 并不是：

```text
100 个 query
→ 100 个彼此独立的动作预测器
```

而是：

```text
100 个 output positions
↕ self-attention
共同形成 action sequence
```

---

## 46. 这和 Action Chunking 为什么天然匹配？

Action Chunking 的核心目标就是：

> 把未来一段动作作为整体建模。

Transformer decoder 正好允许：

$$
a_t,\ldots,a_{t+k-1}
$$

对应的 hidden states 相互交互。

因此 action chunk 不只是：

> 一个 Linear layer 一次输出 1400 个数字。

它拥有显式的：

> **sequence positions + attention-based interaction**

这就是 Transformer 在 ACT 中的重要价值之一。

---

## 47. ACT Decoder 是不是像 GPT 一样一个 Action 一个 Action 地生成？

**不是。**

GPT：

```text
token 1
↓
token 2
↓
token 3
...
```

需要 causal autoregressive generation。

ACT：

```text
k action queries
↓
同时进入 decoder
↓
并行输出 k actions
```

没有通过：

$$
a_t
$$

作为 decoder input 再预测：

$$
a_{t+1}
$$

所以不要把所有 Transformer decoder 都理解成：

> “必须一个 token 一个 token 地生成”。

Transformer decoder 是一种架构。

是否 autoregressive 取决于：

- mask；
- target inputs；
- inference procedure。

---

## 48. ACT 为什么还需要 Decoder Self-Attention？

如果每个 query 只独立 cross-attend observation memory，

那么每个 future action slot 可以独立预测。

但 action trajectory 本身具有强时间关联：

$$
a_{t+1}
$$

通常不会和：

$$
a_t
$$

完全无关。

Decoder self-attention 允许 output positions 共享信息，

从架构上有能力学习：

- temporal consistency；
- coordinated bimanual motion；
- local trajectory structure。

论文把这总结为：

> decoder generates a coherent action sequence。

---

## 49. 双臂协调在哪里体现？

每个 action vector 已经是：

$$
14
$$

维。

也就是说：

> 左臂和右臂的 joint targets 在同一个 output vector 中联合预测。

不是：

```text
左臂一个 policy
右臂另一个 policy
```

所以：

$$
a_i\in\mathbb R^{14}
$$

本身就允许模型表示：

> 同一个 timestep 内的双臂协同。

再加上：

> action queries 之间的 decoder self-attention，

模型还可以学习：

> 跨 timestep 的双臂协同序列。

---

## 50. Position Encoding 在整个 ACT 里一共有几类？

ACT 中至少可以区分三类 positional / role information。

### 1. CVAE Encoder Sequence Position

训练时：

```text
[CLS]
joint
action 0
action 1
...
```

使用 sinusoidal positional table。

---

### 2. Visual 2D Position

告诉 policy encoder：

> visual token 来自哪一个 spatial location。

---

### 3. Policy Decoder Action Query Position

区分：

$$
0,\ldots,k-1
$$

不同 future action slots。

所以 position information 在 ACT 中不是一个单一模块。

不同 sequence 各自需要：

> “这些 token 谁是谁、在哪里”。

---

## 51. Camera Identity 怎么表示？

这个问题很有意思。

论文把四路 camera feature sequences concatenated。

官方实现把它们沿 width 拼成：

$$
15\times80
$$

再用对应 positional embeddings 一起拼接。

因此 camera identity 不是通过显式：

```text
camera_type_embedding
```

单独标注。

它主要通过：

- 在拼接后的空间位置；
- 不同 camera feature content；
- positional encoding；

被模型区分。

这一点属于具体实现设计。

现代多模态架构也可能选择显式 camera embeddings，

但原始 ACT 没有额外引入这样的 token type embedding。

---

## 52. ACT 的 ResNet18 是四个独立 Backbone 吗？

从论文表述：

> process each image observation with ResNet18 backbones

很容易理解成四套独立网络。

但官方代码值得注意：

```python
features, pos =
    self.backbones[0](
        image[:, cam_id]
    )
```

对不同 camera 都调用：

$$
\text{backbones}[0]
$$

也就是说：

> **官方当前代码对多路 camera 共享同一套 visual backbone weights。**

不同相机图像逐个通过相同 ResNet backbone。

这是一个重要的实现细节。

---

## 53. 为什么共享 Backbone 合理？

因为四路输入都是：

> RGB camera images。

共享 backbone 可以：

- 降低参数量；
- 学习共用视觉 feature extractor；
- 让不同 views 进入统一 representation space。

但后面的 Transformer 仍然可以根据：

- visual content；
- spatial position；

学习不同视角的作用。

---

## 54. Encoder Memory 的 Key 和 Value 为什么来自同一个东西？

标准 cross-attention：

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
$$

在 ACT decoder：

$$
K
=
M+\text{position}
$$

$$
V=M
$$

也就是说：

- Key 用于判断“哪个 memory position 与当前 query 匹配”；
- Value 提供真正被读取的信息。

这是标准 Transformer cross-attention 结构。

详细参见：

- [Attention](../../deep-learning/attention.md)
- [Query / Key / Value](../../deep-learning/qkv.md)

---

## 55. 为什么 Key 加 Position，而 Value 不一定加？

官方 Transformer 代码：

```python
key =
    memory + pos

value =
    memory
```

直觉上：

Key 负责：

> “我是谁、在哪里、是否与 query 匹配？”

所以加入 positional information 很自然。

Value 负责：

> “如果选择我，要读取什么内容？”

因此直接使用 contextual memory。

这是 DETR 风格 Transformer 的实现习惯。

---

## 56. Query Embedding 是固定的还是 Learned 的？

这里存在一个值得明确记录的：

> **论文描述与当前官方代码差异。**

论文正文 Section IV-C 描述 decoder input 为：

> fixed position embedding

Appendix C 又写：

> queries are fixed sinusoidal embeddings for the first layer.

也就是说，从论文文字看：

> action query positions 被描述为固定 sinusoidal embeddings。

---

但当前官方 ACT repository 中：

```python
self.query_embed =
    nn.Embedding(
        num_queries,
        hidden_dim
    )
```

`nn.Embedding` 默认参数是：

> **learnable**

并且：

```python
self.query_embed.weight
```

直接传入 Transformer 作为：

```python
query_embed
```

所以当前官方代码实现实际上使用：

> **learned action query embeddings**

而不是固定 sinusoidal query table。

---

## 57. 这个差异该怎么理解？

在讲：

> **ACT 论文 architecture**

时，

应该诚实地写：

> 论文文字描述的是 fixed positional/query embeddings。

在讲：

> **当前官方 code**

时，

应该写：

> repository 使用 learnable `nn.Embedding`.

不要把两者悄悄混成一个版本。

对于理解架构核心而言，

两者共同点仍然是：

> 有 $k$ 个 distinct query positions，对应 $k$ 个 future action slots。

它们究竟是：

- fixed sinusoidal；
- learned embeddings；

属于更具体的 positional representation choice。

---

## 58. 另一个更重要的实现差异：7 Decoder Layers

论文 Table III 给出：

$$
\boxed{
\#\text{ decoder layers}=7
}
$$

官方训练配置同样设置：

```python
dec_layers = 7
```

所以论文意图 architecture 是：

> 7 层 Transformer decoder。

---

## 59. 但当前官方代码存在一个值得警惕的输出索引行为

`TransformerDecoder` 在：

```python
return_intermediate=True
```

时，会返回所有 decoder layers 的 outputs：

$$
[L,B,k,d]
$$

其中：

$$
L=7
$$

但 `detr_vae.py` 当前写：

```python
hs = self.transformer(...)[0]
```

由于 `Transformer.forward()` 当前只返回 `hs`，

这个：

```python
[0]
```

会取：

> decoder intermediate stack 的第一个 element。

也就是看起来会使用：

> **第一层 decoder output**

而不是最后一层。

---

## 60. 这是论文设计的一部分吗？

论文明确写的是：

$$
7
$$

个 decoder layers。

因此：

> “只使用第一个 decoder layer 的 output”

并不是论文文本中描述的 architecture 设计。

这个行为后来在官方 repository 的多个 issue 中被指出。

LeRobot 当前 ACT implementation 甚至明确注释：

> 原始 ACT implementation 配置了 7 decoder layers，但由于原代码行为，实际上使用第一层输出；LeRobot 为了匹配原实现行为直接设置 1 层。

所以这应该理解为：

> **paper-intended architecture 与 released implementation behavior 之间的一个已知差异 / 疑似 bug。**

学习 ACT 理论结构时，

本文仍以论文的：

> multi-layer Transformer decoder

为主线。

复现实验时则必须意识到这一代码细节。

---

## 61. 为什么这类“论文 vs Code”差异值得记录？

因为如果只读论文：

你可能实现：

```text
7 decoder layers
↓
使用最后一层 output
```

而如果只照官方 repo：

可能实际使用：

```text
7 decoder layers 被计算
↓
但 action head 接第一层 output
```

二者不是同一个 computation graph。

所以科研复现时：

> **“官方代码”也不能被当成绝对无误的真理。**

正确流程应该是：

```text
Paper
↕
Official Code
↕
Ablation / Reproduction
```

互相核对。

---

## 62. ACT Architecture 和 DETR 有什么关系？

官方源码文件甚至叫：

```text
detr_vae.py
```

ACT 的 Transformer policy 很明显借用了：

> **DETR-style encoder–decoder architecture**

包括：

- CNN backbone；
- positional encoding；
- Transformer encoder；
- learned / positional queries；
- Transformer decoder cross-attention；
- per-query prediction head。

区别是 DETR 的 query slot 用来预测：

> objects

而 ACT 中 query slot 用来预测：

> future actions。

可以做一个高层类比：

```text
DETR:

object query 1
→ object 1

object query 2
→ object 2


ACT:

action query 1
→ action at future slot 1

action query 2
→ action at future slot 2
```

但不要把它们等同。

ACT 的 target structure、loss、CVAE 和 temporal semantics 都不同。

---

## 63. 为什么 Action Query 数量等于 Chunk Size？

官方：

```python
num_queries = chunk_size
```

如果：

$$
k=100
$$

就创建：

$$
100
$$

个 query embeddings。

因为每一个 query 最终对应：

$$
1
$$

个 action vector。

所以：

$$
\boxed{
\#query
=
\#future action slots
=
k
}
$$

这是 architecture 与 Action Chunking 最直接的连接点。

---

## 64. 如果改变 k，网络哪部分会变化？

假设：

$$
k=100
\rightarrow
k=50
$$

那么至少：

#### Action Query Count

从：

$$
100
$$

变为：

$$
50
$$

#### Decoder Output

从：

$$
100\times512
$$

变为：

$$
50\times512
$$

#### Action Output

从：

$$
100\times14
$$

变为：

$$
50\times14
$$

#### CVAE Encoder Sequence Length

训练时：

$$
[CLS]+q+\text{actions}
$$

也从：

$$
k+2
$$

改变。

因此 chunk size 不只是 inference-time 参数。

它直接改变：

> network sequence dimensions 和 training target。

---

## 65. 为什么 Encoder Sequence Length 1202 与 k 无关？

注意这里有两个不同 encoder。

### CVAE Training Encoder

sequence length：

$$
k+2
$$

取决于 action chunk size。

---

### Policy Transformer Encoder

sequence length：

$$
1202
$$

来自：

$$
1200\text{ visual}
+
1\text{ joint}
+
1\text{ z}
$$

与：

$$
k
$$

无关。

---

而 Policy Transformer Decoder：

sequence length：

$$
k
$$

所以 ACT 有三个不同 sequence lengths：

```text
CVAE encoder:
k + 2

Policy encoder:
1202

Policy decoder:
k
```

这是读 architecture 时非常值得牢记的一张表。

---

## 66. 三个 Sequence 分别在建模什么？

### CVAE Encoder Sequence

$$
[CLS],q,a_t,\ldots,a_{t+k-1}
$$

目标：

> infer latent style $z$。

---

### Policy Encoder Sequence

$$
z,q,\text{visual tokens}
$$

目标：

> fuse current observation context。

---

### Policy Decoder Sequence

$$
k\text{ action query slots}
$$

目标：

> generate future action sequence。

它们都用了 Transformer，

但任务完全不同。

---

## 67. 为什么不能把三个 Transformer 混成“ACT 有个 Transformer”？

因为它们的角色完全不同。

ACT 至少有：

#### CVAE Transformer Encoder

training only。

---

#### Policy Transformer Encoder

training + inference。

---

#### Policy Transformer Decoder

training + inference。

所以一句：

> “ACT 用 Transformer”

信息量太少。

真正应该知道：

```text
Transformer #1
把 demonstration action sequence
压成 latent posterior parameters

Transformer #2
融合 observation + qpos + z

Transformer #3
根据 fused memory
生成 action chunk
```

这样 architecture 才真正清晰。

---

## 68. 一张完整 Architecture 图

```text
                         CURRENT OBSERVATION
              ┌──────────────────────────────────┐
              │                                  │
              │  4 × RGB Images                  │
              │  [4, 3, 480, 640]               │
              │                                  │
              │  Joint State q                   │
              │  [14]                            │
              └───────────────┬──────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
  Each Image                                   Joint q
        │                                           │
        ▼                                           ▼
    ResNet18                                   Linear 14→512
        │                                           │
        ▼                                           ▼
 [512,15,20]                                   1 joint token
        │
        ▼
 flatten spatial
        │
        ▼
 [300,512] per camera
        │
        ▼
 concatenate 4 cameras
        │
        ▼
 [1200,512]
        │
        │
        │           latent z [32]
        │                │
        │                ▼
        │          Linear 32→512
        │                │
        │                ▼
        │           1 latent token
        │                │
        └──────────┬─────┴───────────┐
                   │                 │
                   ▼                 │
          Transformer Encoder ◄──────┘
                   │
                   ▼
          Memory [1202,512]
                   │
                   │ K,V
                   ▼
       ┌─────────────────────┐
       │ Transformer Decoder │
       └─────────────────────┘
                   ▲
                   │ Q
                   │
       k Action Query Slots
            [k,512]
                   │
                   ▼
      Decoder Features [k,512]
                   │
                   ▼
          Linear 512→14
                   │
                   ▼
       Predicted Action Chunk
             [k,14]
```

---

## 69. Training 时图里还多一条 CVAE Encoder Branch

```text
Ground-truth Action Chunk [k,14]
                │
                ▼
          Linear 14→512
                │
Joint q ────────┤
                │
[CLS] ──────────┤
                ▼
     CVAE Transformer Encoder
                │
                ▼
             h_CLS
                │
                ▼
        Linear 512→64
          ↙           ↘
       μ[32]       logσ²[32]
          \           /
           \         /
          ε ~ N(0,I)
               │
               ▼
        z = μ + σε
               │
               └──────→ ACT Policy
```

推理时：

> 整条黄色 / training encoder branch 消失。

直接：

$$
z=0
$$

进入 policy。

---

## 70. ACT Architecture 最核心的三个“压缩 / 展开”

可以从信息结构角度重新看。

### 视觉压缩

$$
4\times480\times640\times3
$$

经过 ResNet：

$$
\rightarrow
1200\times512
$$

---

### Context Fusion

$$
1200\text{ visual}
+
q
+
z
$$

经过 Transformer encoder：

$$
\rightarrow
1202\text{ contextual memory tokens}
$$

---

### Future Expansion

$$
k\text{ action queries}
$$

通过 cross-attention 读取 memory：

$$
\rightarrow
k\text{ action representations}
$$

最后：

$$
\rightarrow
k\times14
$$

future actions。

这比只记组件名字更容易理解整个架构。

---

## 71. 常见误解一：ResNet 直接预测动作

**错误。**

ResNet 只负责：

> visual feature extraction。

完整流程：

$$
Image
\rightarrow
ResNet
\rightarrow
Visual Tokens
\rightarrow
Transformer
\rightarrow
Actions
$$

---

## 72. 常见误解二：四个 Camera 各有一套完全独立 Transformer

**错误。**

四路 visual features 最终被 concatenated，

一起进入：

> 同一个 policy Transformer encoder。

这正是模型跨视角融合信息的地方。

---

## 73. 常见误解三：Joint Positions 直接和 Image Tensor 做 Concatenate

不是在 raw pixel level。

joint state：

$$
14
$$

维，

先：

$$
14\rightarrow512
$$

变成 token。

images 也先经过：

$$
ResNet
$$

变成 512-D visual tokens。

它们统一到同一个 embedding dimension 后，

才进入 Transformer sequence。

---

## 74. 常见误解四：z 直接控制某个 Joint

**错误。**

$z$ 被：

$$
32\rightarrow512
$$

project 成一个 token，

然后和所有 observation features 一起 self-attend。

它不是：

```text
z₁ → joint 1
z₂ → joint 2
```

这种对应关系。

---

## 75. 常见误解五：Transformer Encoder 输出一个向量

ACT policy encoder 不只输出一个 vector。

它保持整段 sequence：

$$
1202\times512
$$

作为 memory。

Decoder 可以对不同 memory positions 做 cross-attention。

---

## 76. 常见误解六：Decoder Query 就是 QKV 中最终的 Q Matrix

不完全一样。

`query_embed` 是：

> action-slot positional/query representation。

在 attention layer 内部，它还会经过：

- addition with current decoder state；
- learned $W_Q$ projection；

才得到真正 attention 公式里的：

$$
Q
$$

matrix。

所以：

```text
query embedding
```

和：

```text
attention Q after projection
```

不要完全等同。

详细参见：

- [Query / Key / Value](../../deep-learning/qkv.md)

---

## 77. 常见误解七：Decoder 是 Autoregressive 的

**错误。**

ACT 没有 causal mask，

也不把前一个 predicted action 逐步喂回去。

它并行预测整个 chunk。

---

## 78. 常见误解八：k 个 Actions 是 k 个独立 Linear Heads

**错误。**

先经过：

> 一个 Transformer decoder sequence。

每个 action slot 拥有自己的 contextual hidden state，

最后共享 action projection：

$$
512\rightarrow14
$$

得到动作。

---

## 79. 常见误解九：Transformer Decoder 直接读取原图

**错误。**

Decoder cross-attend 的是：

$$
\boxed{
\text{Transformer Encoder Memory}
}
$$

原图已经经过：

```text
ResNet
↓
projection
↓
policy encoder
```

---

## 80. 常见误解十：ACT 只有一个 Encoder

**错误。**

至少要区分：

- CVAE training encoder；
- policy Transformer encoder。

两者输入、目的、生命周期完全不同。

---

## 81. 常见误解十一：论文和官方代码细节完全一致

并不总是。

至少目前值得注意：

#### Query Representation

论文文字：

> fixed positional / sinusoidal query embeddings。

当前官方 code：

> learnable `nn.Embedding`.

#### Decoder Layers

论文：

$$
7
$$

layers。

当前 repo：

> 配置 7 层，但 action path 的 `[0]` 索引行为会选择 intermediate decoder stack 的第一项，这已被社区多次指出。

因此复现时应明确自己采用：

> paper-intended architecture

还是：

> released-code behavior。

---

## 82. 常见误解十二：1202 是固定属于所有 ACT 的数字

不是。

它来自原论文 ALOHA 设置：

$$
4
$$

路 cameras，

每路：

$$
15\times20=300
$$

tokens。

所以：

$$
4\times300+2
=
1202
$$

如果你改变：

- image resolution；
- backbone stride；
- camera count；

visual token 数都会改变。

真正一般形式：

$$
\boxed{
N_{\text{encoder}}
=
N_{\text{camera}}
\times
H_f
\times
W_f
+
2
}
$$

其中 +2 是：

- joint token；
- latent token。

---

## 83. 为什么原始 ACT 能在约 0.01 秒推理？

论文报告：

> 在 RTX 2080 Ti 上 inference time 约 0.01 秒。

虽然 sequence length：

$$
1202
$$

并不算小，

但：

- hidden dimension 512；
- encoder 4 layers；
- ResNet18 relatively light；
- action chunk parallel generation；

使单次 forward 仍可较快完成。

不过使用 Temporal Ensemble 时：

> policy 每 timestep 都 query，

所以总运行计算量仍然需要考虑控制频率。

---

## 84. Architecture 和 Temporal Ensemble 不要混

ACT Policy Architecture 负责：

$$
o_t
\rightarrow
\hat a_{t:t+k}
$$

即：

> 一次 forward 产生一个 action chunk。

Temporal Ensemble 则发生在：

> 多个不同 timestep 的 policy forwards 之后。

所以：

```text
ACT Architecture
↓
产生一个 chunk


多次 Architecture forward
↓
得到 overlapping chunks
↓
Temporal Ensemble
↓
决定当前真正执行 action
```

它不是 Transformer 里面的一层。

---

## 85. Architecture 和 CVAE 也不要混

Training 时：

> CVAE 是整个 probabilistic training framework。

ACT policy architecture：

```text
ResNet
+
Transformer Encoder
+
Transformer Decoder
```

实际上就是：

> CVAE decoder / policy。

所以：

$$
\text{CVAE}
$$

不是 policy Transformer 内部的某一个 layer。

关系应该是：

```text
ACT CVAE
│
├── CVAE Encoder
│   └── BERT-like Transformer Encoder
│
└── CVAE Decoder / Policy
    ├── ResNet18
    ├── Transformer Encoder
    ├── Transformer Decoder
    └── Action Head
```

---

## 86. 用一句话理解每个模块

### ResNet18

> 把高分辨率图片变成紧凑 spatial visual features。

### 2D Positional Encoding

> 告诉 Transformer visual features 在哪里。

### Joint Projection

> 把当前机器人姿态变成 Transformer token。

### Latent Projection

> 把 style variable 变成 Transformer token。

### Transformer Encoder

> 融合多视角视觉、robot state 和 latent condition。

### Action Queries

> 定义 $k$ 个未来动作输出位置。

### Transformer Decoder Self-Attention

> 让未来 action positions 相互协调。

### Cross-Attention

> 让每个 action slot 从当前 observation memory 中读取所需信息。

### Action Head

> 把每个 512-D action representation 转成 14-D joint target。

---

## 87. 用三条 Shape 记住整个 Policy

如果最后只记住三个 shape：

### Observation Memory

$$
\boxed{
1202\times512
}
$$

### Action Decoder Features

$$
\boxed{
k\times512
}
$$

### Final Action Chunk

$$
\boxed{
k\times14
}
$$

整个 ACT policy 可以压缩成：

$$
\boxed{
1202\times512
\;\xrightarrow[\text{k queries}]{\text{Transformer Decoder}}\;
k\times512
\;\xrightarrow{\text{Action Head}}\;
k\times14
}
$$

---

## 88. 一句话重新理解 ACT Architecture

> **ACT 先用 ResNet18 把 4 路视觉压缩成 1200 个 512 维 visual tokens，再把当前 14 维 joint state 和 32 维 latent $z$ 各自投影成一个 512 维 token；这 1202 个 tokens 经过 Transformer encoder 融合成 observation memory。随后 $k$ 个 action-query slots 通过 Transformer decoder 的 self-attention 建模未来动作之间的关系，并通过 cross-attention 从 observation memory 中读取信息，最后每个 512 维 decoder output 被投影成一个 14 维双臂 target joint vector，从而一次并行输出 $k\times14$ 的 action chunk。**

这就是 ACT 从：

$$
\text{Pixels}
$$

到：

$$
\text{Robot Actions}
$$

的完整架构主线。

---

## 89. 下一步

现在我们已经知道：

> 模型“长什么样”。

下一步最自然的是把：

> **训练**

和：

> **推理**

完全分开。

### 下一篇之一：ACT Training

会回答：

- training sample 怎样从 episode 切出 action chunk；
- normalization 怎样做；
- padding 为什么存在；
- CVAE encoder 和 policy 怎样联合优化；
- L1 reconstruction 怎样计算；
- KL 怎样计算；
- $\beta=10$ 怎样进入 loss；
- optimizer 实际更新哪些参数。

见：

- [ACT Training](./training.md)

---

### 下一篇之二：ACT Inference

会回答：

- test observation 怎样进入网络；
- 为什么 $z=0$；
- 一次 forward 怎样得到 $k$ actions；
- 为什么每 timestep 都重新 query；
- overlapping chunks 怎样进入 Temporal Ensemble；
- 最终只执行哪个 action；
- 下一 timestep 又怎样重复。

见：

- [ACT Inference](./inference.md)

---

### Primary Source

Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.  
**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.**  
Robotics: Science and Systems (RSS), 2023.

- Paper: https://arxiv.org/abs/2304.13705
- Project: https://tonyzhaozh.github.io/aloha/
- RSS Proceedings: https://roboticsproceedings.org/rss19/p016.html

本文主要依据：

- Section IV-C — Implementing ACT
- Figure 3 — ACT architecture
- Appendix C — Detailed Architecture Diagram
- Figure 10 — Detailed ACT architecture
- Table III — ACT hyperparameters

论文给出的核心 shape：

$$
480\times640\times3
\rightarrow
15\times20\times512
\rightarrow
300\times512
$$

4 cameras：

$$
1200\times512
$$

加入 joint state 和 latent $z$：

$$
1202\times512
$$

decoder：

$$
k\times512
$$

最终：

$$
k\times14
$$

---

### Official Implementation

ACT official repository:

https://github.com/tonyzhaozh/act

主要对应：

```text
detr/models/detr_vae.py
detr/models/transformer.py
detr/models/backbone.py
detr/models/position_encoding.py
imitate_episodes.py
```

当前官方实现可确认：

- `state_dim = 14`
- `latent_dim = 32`
- `action_head: hidden_dim → 14`
- `query_embed: num_queries × hidden_dim`
- `num_queries = chunk_size`
- visual features、latent token、proprio token 一起进入 policy encoder
- decoder cross-attention 的 keys / values 来自 encoder memory
- action queries 作为 decoder query positions

---

### Paper / Code Implementation Notes

#### Action Query Embedding

论文文字描述：

> fixed positional / sinusoidal embeddings。

当前官方 repo：

```python
self.query_embed =
    nn.Embedding(
        num_queries,
        hidden_dim
    )
```

即 learnable query embeddings。

---

#### Decoder Layer Output

论文 Table III：

$$
7
$$

decoder layers。

当前官方 repo 配置同样为：

```text
dec_layers = 7
```

但 released implementation 的 intermediate-output indexing 行为已在 GitHub issues #10、#25、#52 中被指出可能只把第一层 decoder output 送入 action head。

Hugging Face LeRobot 的 ACT 配置目前明确记录了这一点，并使用 1 个 decoder layer 来匹配原始 released implementation 的实际行为。

因此：

> 本文 architecture 主线按照论文描述理解；复现官方代码时，需要单独注意这一实现差异。

---

### 本文知识连接

#### ACT 主线

- [ACT 到底解决了什么问题？](./act-what-problem-does-it-solve.md)
- [Action Chunking](./action-chunking.md)
- [Temporal Ensemble](./temporal-ensemble.md)
- [Vision Pipeline](./vision-pipeline.md)
- [从 DETR 到 ACT](./detr-to-act.md)
- [CVAE in ACT](./cvae-in-act.md)
- [为什么 ACT 推理时令 z = 0？](./why-z-zero-at-inference.md)

#### Transformer

- [Transformer](../../deep-learning/transformer.md)
- [Attention](../../deep-learning/attention.md)
- [Query / Key / Value](../../deep-learning/qkv.md)
- [Self-Attention](../../deep-learning/self-attention.md)
- [Cross-Attention](../../deep-learning/cross-attention.md)
- [Multi-Head Attention](../../deep-learning/multi-head-attention.md)
- [Transformer Encoder](../../deep-learning/transformer-encoder.md)
- [Transformer Decoder](../../deep-learning/transformer-decoder.md)
- [Positional Encoding](../../deep-learning/positional-encoding.md)

#### Generative Models

- [CVAE](../../generative-models/cvae.md)
- [Latent Variable](../../generative-models/latent-variable.md)

#### Vision

- CNN
- ResNet

#### Robot Learning

- Joint Position
- PID Controller

#### 下一步

- [ACT Training](./training.md)
- [ACT Inference](./inference.md)
