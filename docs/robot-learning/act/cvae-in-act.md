---
title: "CVAE in ACT：z 到底是怎么产生和使用的？"
description: "基于 ACT 原论文与官方实现，逐步拆解 CVAE encoder、[CLS] token、μ/logσ²、reparameterization、latent z、Transformer policy，以及训练与推理时 z 的不同来源。"
status: reviewed
pageType: application
canonical: /robot-learning/act/cvae-in-act
updated: "2026-09-15"
---

# CVAE in ACT：z 到底是怎么产生和使用的？

在前面的几篇文章里，我们已经分别理解了：

- [Latent Variable](../../generative-models/latent-variable.md)
- [VAE](../../generative-models/vae.md)
- [Reparameterization Trick](../../generative-models/reparameterization-trick.md)
- [CVAE](../../generative-models/cvae.md)
- [Posterior Collapse](../../generative-models/posterior-collapse.md)

现在终于可以回到 ACT 本身。

ACT 论文里最容易让人卡住的一条数据流是：

```text
action sequence
+
joint positions
↓
CVAE encoder
↓
[CLS]
↓
μ, logσ²
↓
sample z
↓
ACT policy
↓
action chunk
```

如果只看网络图，很容易产生一连串疑问：

> `[CLS]` 为什么能“读懂”整段动作？

> 为什么 `[CLS]` 后面接一个 Linear 就能得到 $\mu$ 和 $\sigma^2$？

> $z$ 是随机采出来的，为什么还能帮助 decoder 预测正确动作？

> 为什么训练时 $z$ 来自 encoder，推理时 encoder 却被直接删掉？

> 为什么推理不是从 $\mathcal N(0,I)$ 随机采样，而是直接令：

$$
z=0
$$

> 既然推理时永远 $z=0$，训练这个 CVAE 到底有什么意义？

这一篇不再泛泛解释 CVAE。

我们只做一件事：

> **严格按照 ACT 原论文和官方代码，把 ACT 中的 CVAE 从输入到输出完整走一遍。**

---

## 1. 先回答：ACT 为什么要有 CVAE？

先回到论文提出的问题。

ACT 要从 human demonstrations 学习机器人动作。

但人类示范不是完全 deterministic 的。

即使面对相似 observation，人类也可能：

- 走略有不同的轨迹；
- 在不需要精确控制的位置更随意；
- 在某次演示中停顿得久一点；
- 选择略微不同的手部路径；
- 在仍然能完成任务的范围内产生变化。

论文明确指出：

> Given the same observation, a human can use different trajectories to solve the task.

所以 ACT 不希望把 human demonstration 简单看成：

$$
o_t
\longrightarrow
\text{唯一的 action chunk}
$$

而是希望学习：

$$
p(
a_{t:t+k}
\mid
o_t
)
$$

即：

> **给定当前 observation 后，未来 action sequence 的条件分布。**

这就是为什么 ACT 把 action-chunking policy 训练成一个：

> **Conditional Variational Autoencoder（CVAE）**

---

## 2. 在 ACT 里，CVAE 的三个变量分别是什么？

把一般 CVAE 的符号映射到 ACT。

一般 CVAE：

$$
p(y\mid c)
$$

其中：

- $c$：condition
- $y$：output
- $z$：latent variable

在 ACT 中：

#### Condition

$$
c
=
o_t
$$

即当前 observation。

包含：

- 当前 joint positions；
- 多个 camera images。

---

#### Output

$$
y
=
a_{t:t+k}
$$

即未来的 action chunk。

---

#### Latent Variable

$$
z
$$

论文称为：

> **style variable**

用于帮助模型表达 human demonstration 中的 latent variation。

因此 ACT 从高层上是在建模：

$$
\boxed{
p_\theta(
a_{t:t+k}
\mid
o_t
)
}
$$

并通过 latent $z$ 写成类似：

$$
p_\theta(
a_{t:t+k}
\mid
o_t,z
)
$$

---

## 3. 但 ACT 并没有完全照搬原始 CVAE

在一般 CVAE 中，我们可以有：

$$
p_\theta(z\mid c)
$$

即 conditional prior。

但 ACT 没有单独学习一个：

$$
p_\theta(z\mid o_t)
$$

它使用固定 prior：

$$
\boxed{
p(z)=\mathcal N(0,I)
}
$$

训练时 approximate posterior：

$$
q_\phi(
z
\mid
a_{t:t+k},
\bar o_t
)
$$

被 KL regularize toward：

$$
\mathcal N(0,I)
$$

这里：

$$
\bar o_t
$$

表示：

> observation 去掉 image observations 后剩余的 proprioceptive observation。

对 ALOHA 来说，核心就是当前 joint positions。

所以 ACT 属于我们在 [CVAE](../../generative-models/cvae.md) 中讲过的：

> **fixed-prior CVAE**

---

## 4. 为什么 Encoder 不是 q(z | observation, actions) 的完整 observation？

论文 Algorithm 1 写：

$$
q_\phi(
z
\mid
a_{t:t+k},
\bar o_t
)
$$

而不是：

$$
q_\phi(
z
\mid
a_{t:t+k},
o_t
)
$$

区别在：

$$
\bar o_t
$$

不包含 image observations。

论文解释：

> 为了加快实际训练，CVAE encoder 不使用图像，只使用 proprioceptive observation 和 action sequence。

也就是说：

```text
CVAE Encoder 看到：

current joint positions
+
ground-truth future action chunk
```

但不看：

```text
camera images
```

这是 ACT 的具体工程设计。

因此不能简单写：

> “ACT encoder 输入完整 current observation 和 actions。”

更准确的是：

> **训练时 ACT 的 latent encoder 使用 joint positions + demonstration action sequence 推断 $z$，而视觉信息主要进入 decoder / policy。**

---

## 5. 为什么 Encoder 要看 Ground-Truth Action Sequence？

这是 CVAE 最关键的问题之一。

训练时我们有一条真实 demonstration：

$$
a_{t:t+k}
$$

假设相似 observation 下可能存在两条合理轨迹：

```text
Trajectory A
↙

Trajectory B
↘
```

仅仅给 encoder 看当前：

$$
o_t
$$

它可能无法知道：

> 训练数据这次到底采取的是 A 还是 B。

但如果把真实 action chunk：

$$
a_{t:t+k}
$$

也给它，

encoder 就能回答：

> “既然这条 demonstration 实际上是这样走的，什么样的 latent $z$ 可以描述这次动作序列中的 variation？”

因此训练时：

$$
q_\phi(
z
\mid
a_{t:t+k},
\bar o_t
)
$$

实际上是在做：

> **posterior inference**

而不是预测 future action。

future action prediction 是 decoder / policy 的工作。

---

## 6. ACT CVAE Encoder 的输入到底长什么样？

论文实现部分给得非常具体。

假设 action chunk 中有：

$$
k
$$

个 actions。

CVAE encoder 的 token sequence 为：

```text
[CLS]
current joint positions
action_t
action_t+1
...
action_t+k-1
```

因此总 token 数：

$$
\boxed{
k+2
}
$$

分别是：

1. 一个 learned `[CLS]` token；
2. 一个 current joint-position token；
3. $k$ 个 action tokens。

---

## 7. 每个输入原本维度并不一样

ALOHA 是双臂机器人。

论文中：

$$
7+7=14
$$

个 joint dimensions。

因此当前 joint positions：

$$
q_t\in\mathbb R^{14}
$$

每一个 action 也是两个机械臂的 absolute target joint positions：

$$
a_t\in\mathbb R^{14}
$$

但 Transformer 不能直接要求：

```text
CLS: hidden_dim
joint: 14
action: 14
```

三种不同宽度的 token 混在一起。

所以 ACT 先通过 Linear projection，把它们投影到统一 embedding dimension。

---

## 8. Joint Position Projection

官方实现中：

```python
self.encoder_joint_proj = nn.Linear(14, hidden_dim)
```

于是：

$$
q_t
\in
\mathbb R^{14}
$$

变成：

$$
e_q
\in
\mathbb R^{d}
$$

其中：

$$
d=\text{hidden dimension}
$$

论文 hyperparameter 中：

$$
d=512
$$

所以可以理解为：

$$
\boxed{
\mathbb R^{14}
\rightarrow
\mathbb R^{512}
}
$$

---

## 9. Action Projection

每一个 action：

$$
a_{t+i}\in\mathbb R^{14}
$$

也使用 Linear：

```python
self.encoder_action_proj = nn.Linear(14, hidden_dim)
```

投影成：

$$
e_{a_{t+i}}
\in
\mathbb R^{512}
$$

所以 action sequence：

$$
[k,14]
$$

变成：

$$
[k,512]
$$

---

## 10. [CLS] Token 是什么？

ACT 还额外定义一个：

```python
self.cls_embed = nn.Embedding(1, hidden_dim)
```

也就是一个可学习向量：

$$
e_{\mathrm{CLS}}
\in
\mathbb R^{512}
$$

它不是：

> “提前装了一个动作摘要。”

初始化时它只是：

> **一组需要训练的参数。**

然后它与 joint token 和 action tokens 一起进入 Transformer encoder。

完整输入：

$$
[
e_{\mathrm{CLS}},
e_q,
e_{a_t},
\ldots,
e_{a_{t+k-1}}
]
$$

shape 为：

$$
\boxed{
(k+2)\times512
}
$$

---

## 11. [CLS] 为什么最后能代表整段序列？

这是一个非常容易被讲成“魔法”的地方。

`[CLS]` 本身并不会自动读取整个 sequence。

真正发挥作用的是：

> **Transformer self-attention**

因为 `[CLS]` 和其他 token 一起进入 Transformer。

在 self-attention 中，`[CLS]` 对应的 hidden state 可以与：

- joint token；
- 每一个 action token；

发生信息交互。

经过多层 Transformer 后：

$$
h_{\mathrm{CLS}}
$$

已经不再只是最初那个 learned embedding。

它是：

> **经过整段 sequence 上下文更新后的 hidden representation。**

所以可以把它直觉理解为：

```text
[CLS]
  │
  ├── attention → joint
  ├── attention → action 1
  ├── attention → action 2
  ├── ...
  └── attention → action k
          ↓
     多层信息融合
          ↓
       h_CLS
```

如果不熟悉 self-attention，可以先读：

- [Attention](../../deep-learning/attention.md)
- [Query / Key / Value](../../deep-learning/qkv.md)
- [Transformer](../../deep-learning/transformer.md)

---

## 12. 但“CLS 读取所有 token”只是直觉说法

更严格地说：

Transformer encoder 对整个 token sequence 产生：

$$
H
=
[
h_{\mathrm{CLS}},
h_q,
h_{a_t},
\ldots
]
$$

ACT 只取：

$$
\boxed{
h_{\mathrm{CLS}}
}
$$

作为后续 latent distribution prediction 的 summary representation。

官方代码正是：

```python
encoder_output = self.encoder(...)

encoder_output = encoder_output[0]
```

这里：

```python
[0]
```

选的是 sequence 第一位置，也就是 `[CLS]`。

因此：

> `[CLS]` 能否形成有用的全局 summary，是由整个 training objective 学出来的。

没有额外 supervision 直接告诉它：

> “你要学会当摘要。”

它之所以会承担这个角色，是因为后续所有：

$$
\mu,\log\sigma^2
$$

都只能从这个 token 的输出产生。

---

## 13. h_CLS 如何变成 μ 和 log σ²？

拿到：

$$
h_{\mathrm{CLS}}
\in\mathbb R^{512}
$$

以后，

官方实现使用：

```python
self.latent_proj = nn.Linear(
    hidden_dim,
    latent_dim * 2
)
```

官方代码中：

$$
latent\_dim=32
$$

所以：

$$
512
\rightarrow
64
$$

输出：

$$
latent\_info
\in
\mathbb R^{64}
$$

然后一分为二：

```python
mu = latent_info[:, :latent_dim]
logvar = latent_info[:, latent_dim:]
```

即：

$$
\boxed{
\mu\in\mathbb R^{32}
}
$$

$$
\boxed{
\log\sigma^2\in\mathbb R^{32}
}
$$

于是 approximate posterior 为：

$$
\boxed{
q_\phi(z\mid a_{t:t+k},\bar o_t)
=
\mathcal N
\left(
\mu,
\operatorname{diag}(\sigma^2)
\right)
}
$$

---

## 14. 这个 Linear Layer 到底在学什么？

这里非常容易产生一种错误理解：

> “这个 Linear layer 的目标，就是把 $h_{\mathrm{CLS}}$ 映射成 $\mu\approx0$、$\sigma^2\approx1$。”

不准确。

这个 Linear layer 的作用是：

> **把 sequence representation $h_{\mathrm{CLS}}$ 转换成 approximate posterior 的参数。**

也就是：

$$
h_{\mathrm{CLS}}
\longrightarrow
(\mu,\log\sigma^2)
$$

这些参数同时受到两股 training pressure。

---

### Reconstruction Pressure

模型需要用从这个 distribution 采样出的：

$$
z
$$

帮助 decoder 重建当前 demonstration action chunk。

因此：

> $q_\phi(z|\cdot)$ 不能完全丢掉与这条 demonstration 有关的信息。

---

### KL Pressure

同时：

$$
q_\phi(z|\cdot)
$$

受到：

$$
\mathcal N(0,I)
$$

的约束。

因此：

> 不能为了重建每个 training sample 而随意跑到 latent space 的任何位置。

所以 $\mu$ 和 $\sigma^2$ 是：

> **reconstruction 与 prior regularization 共同优化出来的结果。**

不是单独追求：

$$
\mu=0,\qquad\sigma^2=1
$$

---

## 15. 如果所有 μ 都真的变成 0、σ² 都变成 1 会怎样？

如果对于所有 training examples：

$$
q_\phi(z\mid a,\bar o)
=
\mathcal N(0,I)
$$

那么：

$$
z
$$

将几乎不再包含关于当前 action sequence 的 information。

这时 encoder 等于：

> 看了 demonstration，但最后什么都没告诉 decoder。

这属于类似：

> **posterior collapse**

的情况。

所以正常训练不是：

```text
KL
→ 把所有 z 信息消灭
```

而是：

```text
Reconstruction
→ 希望 z 有信息

          VS

KL
→ 希望 z 不要离 prior 太远
```

最终达到某种平衡。

---

## 16. ACT 为什么叫 z “Style Variable”？

论文把：

$$
z
$$

称为：

> **style variable**

可以用一个直觉理解：

对于相似 observation：

```text
这次 human demo
动作稍微更靠左

另一次 human demo
动作稍微更靠右
```

两条都合理。

$z$ 可以帮助 decoder 表达：

> 当前 training demonstration 中没有被 observation 唯一确定的 action-sequence variation。

所以“style”是一个很自然的名字。

但是必须强调：

> **论文并没有监督 z 的每一个 dimension 对应具体人类语义。**

不能未经实验说：

```text
z₁ = 左右方向
z₂ = 速度
z₃ = 力量
```

“style variable”是作用层面的解释，不是 disentangled semantic label。

---

## 17. μ 和 log σ² 得到以后发生什么？

现在 encoder 给出：

$$
\mu
$$

和：

$$
\log\sigma^2
$$

先恢复：

$$
\sigma
=
\exp
\left(
\frac12\log\sigma^2
\right)
$$

然后采：

$$
\epsilon
\sim
\mathcal N(0,I)
$$

最后：

$$
\boxed{
z
=
\mu
+
\sigma\odot\epsilon
}
$$

官方代码：

```python
def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(
        std.data.new(std.size()).normal_()
    )

    return mu + std * eps
```

这就是我们在：

- [Reparameterization Trick](../../generative-models/reparameterization-trick.md)

中讲过的 Gaussian reparameterization。

---

## 18. z 是随机的，那它怎么能“正确”？

这也是理解 ACT 最容易卡住的问题。

假设 encoder 输出：

$$
q_\phi(z\mid a,\bar o)
=
\mathcal N(\mu,\sigma^2)
$$

然后随机采一个：

$$
z
$$

看起来像：

> “随机猜一个 latent code，然后期待 decoder 输出正确动作？”

不是。

关键是：

> $z$ 不是从整个空间无约束地随机抽。

它是从：

$$
q_\phi(z\mid a,\bar o)
$$

中采样。

而这个 distribution 的：

$$
\mu,\sigma^2
$$

正是 encoder 根据当前：

- joint positions；
- ground-truth action chunk；

推断出来的。

也就是说：

```text
demonstration
↓
encoder
↓
一个专门围绕“这条 demonstration”
形成的 posterior distribution
↓
sample z
```

随机性只是在这个 learned distribution 内采样。

不是：

```text
完全随机一个 32 维向量
↓
祈祷 decoder 猜对
```

---

## 19. Decoder / Policy 到底输入什么？

训练阶段得到：

$$
z
$$

以后，

ACT 的 decoder，也就是最终 policy：

$$
\pi_\theta(
\hat a_{t:t+k}
\mid
o_t,z
)
$$

输入包括：

#### 1. Camera Images

论文使用 4 个 RGB images：

$$
480\times640
$$

---

#### 2. Current Joint Positions

$$
q_t\in\mathbb R^{14}
$$

---

#### 3. Latent Style z

官方实现：

$$
z\in\mathbb R^{32}
$$

---

输出：

$$
\boxed{
\hat a_{t:t+k}
\in
\mathbb R^{k\times14}
}
$$

即未来 $k$ 步两个机械臂的 target joint positions。

---

## 20. z 进入 Decoder 前也要做 Projection

因为 Transformer hidden dimension 是：

$$
512
$$

而官方：

$$
z\in\mathbb R^{32}
$$

所以不能直接作为同宽 token 使用。

代码：

```python
self.latent_out_proj = nn.Linear(
    self.latent_dim,
    hidden_dim
)
```

于是：

$$
z\in\mathbb R^{32}
$$

被投影为：

$$
e_z\in\mathbb R^{512}
$$

同样，joint positions：

$$
q_t\in\mathbb R^{14}
$$

也被投影成：

$$
e_q\in\mathbb R^{512}
$$

于是它们可以和 image features 一起进入 Transformer encoder。

---

## 21. 图像是怎么变成 Transformer 输入的？

论文实现使用 ResNet18。

每张：

$$
480\times640\times3
$$

图像经过 ResNet18 后得到：

$$
15\times20\times512
$$

feature map。

把空间维度：

$$
15\times20
$$

flatten：

$$
300
$$

得到：

$$
300\times512
$$

的 token sequence。

4 个 camera：

$$
4\times300=1200
$$

所以图像总共形成：

$$
1200\times512
$$

visual tokens。

再加入：

- joint token；
- latent $z$ token；

得到：

$$
\boxed{
1202\times512
}
$$

作为 Transformer encoder 的输入。

---

## 22. 这里有两个完全不同的 Transformer Encoder

ACT 很容易因为“encoder”这个词太多而混乱。

至少存在：

### A. CVAE Encoder

作用：

> training 时推断 $z$。

输入：

```text
[CLS]
joint positions
ground-truth action sequence
```

输出：

$$
\mu,\log\sigma^2
$$

它只在训练时存在。

---

### B. Policy Transformer Encoder

作用：

> 融合当前视觉、joint positions 和 latent $z$。

输入：

```text
image feature tokens
joint token
z token
```

输出 memory representation。

这个 encoder 是：

> policy / CVAE decoder 的一部分。

训练和推理都会使用。

所以：

```text
CVAE encoder
≠
policy transformer encoder
```

千万不要因为名字都有 “encoder” 就混在一起。

---

## 23. Decoder 也有两个含义

这里同样容易混。

论文说：

> CVAE decoder = policy

这个“大 decoder”内部实际上包含：

```text
ResNet image encoders
+
Transformer encoder
+
Transformer decoder
+
action output head
```

所以：

> **CVAE decoder**

不是单指：

> Transformer decoder layer。

更准确关系：

```text
CVAE Decoder / ACT Policy
│
├── ResNet visual backbone
├── Transformer Encoder
├── Transformer Decoder
└── Action Head
```

这是阅读论文架构时非常重要的命名层级。

---

## 24. Policy Transformer Encoder 在做什么？

它接收：

```text
1200 visual tokens
+
1 joint token
+
1 latent-z token
```

论文的直觉解释是：

> Transformer encoder synthesizes information from different camera viewpoints, the joint positions, and the style variable.

也就是把：

- 当前环境视觉；
- 机器人自身姿态；
- latent style；

融合成一个上下文 memory。

如果用 Transformer 语言：

> 这些输出后面作为 Transformer decoder cross-attention 的 keys 和 values。

---

## 25. Transformer Decoder 如何生成 k 个 Actions？

Transformer decoder 需要产生：

$$
k
$$

个输出位置。

论文中使用长度：

$$
k
$$

的位置/query embeddings。

经过 cross-attention：

```text
k 个 action query positions
          │
          ▼
Transformer Decoder
          ▲
          │ cross-attention
          │
Policy Encoder Memory
```

最终得到：

$$
k\times512
$$

decoder features。

再经过 output head：

$$
512
\rightarrow14
$$

得到：

$$
\boxed{
k\times14
}
$$

未来 action chunk。

---

## 26. 所以 z 并不是直接变成 Action

这是一个非常重要的认识。

不是：

```text
z
↓
Linear
↓
动作
```

而是：

```text
z
↓
project
↓
和 images + joint positions 一起融合
↓
Transformer encoder
↓
memory
↓
Transformer decoder
↓
future action sequence
```

所以 $z$ 更像：

> **一个额外 conditioning signal。**

它告诉 policy：

> 在当前 observation 之下，用哪一种 latent variation 来解释 / 生成动作序列。

真正动作仍然高度依赖：

- image observations；
- current joint state。

---

## 27. 训练阶段完整 Data Flow

现在把所有东西一次串起来。

---

### Step 1：从 Demonstration 采样

取：

$$
o_t
$$

以及：

$$
a_{t:t+k}
$$

其中：

$$
o_t
=
(
\text{images}_t,
q_t
)
$$

---

### Step 2：CVAE Encoder 推断 z

Encoder 不使用 images。

输入：

```text
[CLS]
q_t
a_t
a_t+1
...
a_t+k-1
```

经过 BERT-like Transformer encoder：

$$
\longrightarrow
h_{\mathrm{CLS}}
$$

---

### Step 3：预测 Distribution Parameters

$$
h_{\mathrm{CLS}}
\longrightarrow
\mu,\log\sigma^2
$$

定义：

$$
q_\phi(
z\mid
a_{t:t+k},q_t
)
$$

---

### Step 4：Reparameterization

$$
\epsilon\sim\mathcal N(0,I)
$$

$$
z
=
\mu+\sigma\odot\epsilon
$$

---

### Step 5：ACT Policy Predicts Action Chunk

Policy 输入：

```text
4 camera images
+
current q_t
+
z
```

输出：

$$
\hat a_{t:t+k}
$$

---

### Step 6：Reconstruction Loss

比较：

$$
\hat a_{t:t+k}
$$

和：

$$
a_{t:t+k}
$$

---

### Step 7：KL Regularization

$$
D_{KL}
\left(
q_\phi(
z\mid
a_{t:t+k},q_t
)
\parallel
\mathcal N(0,I)
\right)
$$

---

### Step 8：Joint Optimization

更新：

$$
\phi
$$

和：

$$
\theta
$$

即：

- CVAE encoder；
- ACT policy / decoder。

---

## 28. 把训练过程画成一张图

```text
                 DEMONSTRATION
        ┌──────────────────────────┐
        │ images_t                 │
        │ joint positions q_t      │
        │ future actions A_t       │
        └─────────────┬────────────┘
                      │
          ┌───────────┴────────────┐
          │                        │
          │                        │
          ▼                        ▼
  CVAE ENCODER                ACT POLICY
(training only)               / DECODER
          │                        ▲
          │                        │
 [CLS] + q_t + A_t                 │
          │                        │
          ▼                        │
 Transformer Encoder               │
          │                        │
          ▼                        │
       h_CLS                       │
          │                        │
          ▼                        │
      μ , logσ²                    │
          │                        │
 ε ~ N(0,I)                        │
          │                        │
          ▼                        │
   z = μ + σε ─────────────────────┘
                                   │
              images_t + q_t + z   │
                     │             │
                     ▼             │
              Transformer Policy   │
                     │
                     ▼
            predicted action chunk
                     │
                     ▼
              reconstruction loss

同时：

q(z | A_t,q_t)
        │
        └──── KL ──── N(0,I)
```

---

## 29. ACT 的 Loss 到底是什么？

论文 Algorithm 1 概括为：

$$
\mathcal L
=
\mathcal L_{\mathrm{reconst}}
+
\beta
\mathcal L_{\mathrm{reg}}
$$

其中：

$$
\mathcal L_{\mathrm{reg}}
=
D_{KL}
\left(
q_\phi(z\mid a_{t:t+k},\bar o_t)
\parallel
\mathcal N(0,I)
\right)
$$

论文给出的 ACT hyperparameter：

$$
\boxed{
\beta=10
}
$$

所以 KL 并不是和 reconstruction 等权。

---

## 30. 一个论文里非常值得注意的小矛盾：MSE 还是 L1？

如果只看 Algorithm 1，会看到：

$$
\mathcal L_{\mathrm{reconst}}
=
MSE(
\hat a_{t:t+k},
a_{t:t+k}
)
$$

但论文正文 Section IV-C 明确写：

> 实际实现中使用 L1 reconstruction loss，而不是更常见的 L2，因为作者观察到 L1 对 action sequence 的建模更精确。

官方代码也确认这一点：

```python
all_l1 = F.l1_loss(
    actions,
    a_hat,
    reduction="none"
)
```

最终：

```python
loss =
    l1
    +
    kl_weight * kl
```

因此如果问题是：

> **“ACT 官方实际实现用什么 reconstruction loss？”**

答案应该是：

$$
\boxed{
L1
}
$$

而不是照抄 Algorithm 1 的 MSE。

这是阅读论文时一个很好的例子：

> **算法框、正文和官方实现如果出现不一致，要进一步核对，而不是机械复制。**

---

## 31. Padding 为什么还要 Mask？

action chunks 在 episode 尾部可能不足完整长度。

所以 batch 中通常会进行 padding。

官方实现传入：

```python
is_pad
```

训练 reconstruction 时：

```python
all_l1 = F.l1_loss(
    actions,
    a_hat,
    reduction='none'
)

l1 = (
    all_l1
    * ~is_pad.unsqueeze(-1)
).mean()
```

也就是说：

> padded action positions 不应该被当成真实监督信号。

CVAE encoder 的 Transformer 也使用 padding mask，

避免把 padding token 当成真实 action sequence 内容参与 attention。

---

## 32. 为什么 [CLS] 和 Joint Token 不能被 Mask？

官方代码专门创建：

```python
cls_joint_is_pad = torch.full(
    (bs, 2),
    False
)
```

因为前两个 token：

```text
[CLS]
joint positions
```

永远是真实有效输入。

只有后面的：

```text
action sequence
```

可能因为 episode 尾部而存在 padding。

所以 mask sequence 对应：

```text
[CLS]       → valid
joint       → valid
action 1    → valid
...
padding     → masked
```

这是很具体但很重要的实现细节。

---

## 33. 现在进入最关键的问题：推理时 z 从哪里来？

训练时：

$$
z
\sim
q_\phi(
z\mid
a_{t:t+k},q_t
)
$$

但是推理时我们没有：

$$
a_{t:t+k}
$$

因为：

> 这正是我们想预测的答案。

因此：

$$
q_\phi(
z\mid
a_{t:t+k},q_t
)
$$

根本无法使用。

所以 CVAE encoder 在 test time：

> **直接被丢弃。**

论文明确写：

> The CVAE encoder only serves to train the CVAE decoder and is discarded at test time.

---

## 34. 这不是 ACT 特有的奇怪操作

在 VAE / CVAE 中：

> approximate posterior network 本来就主要用于 inference during training。

真正的 generative model 是：

$$
p(z)
$$

加：

$$
p_\theta(y\mid c,z)
$$

训练时 encoder：

$$
q_\phi
$$

帮助我们近似 posterior。

生成时并不一定需要它。

所以 ACT test time 丢掉 encoder：

> 完全符合 VAE / CVAE 的一般结构。

真正特别的地方是：

> **ACT 如何选择 test-time $z$。**

---

## 35. 一般 CVAE 推理时可以随机采 z

因为 ACT prior 为：

$$
p(z)
=
\mathcal N(0,I)
$$

最直接的 CVAE 推理方式其实可以是：

$$
z
\sim
\mathcal N(0,I)
$$

每次 sample 不同的：

$$
z
$$

然后得到不同 action chunks。

例如：

```text
same observation
+
z₁
↓
trajectory 1

same observation
+
z₂
↓
trajectory 2
```

这符合一般 conditional generative model 的思路。

但 ACT 并没有选择这样做。

---

## 36. ACT 推理直接设 z = 0

ACT 原论文明确规定：

$$
\boxed{
z=0
}
$$

原因首先是一个非常直接的概率事实：

prior：

$$
p(z)
=
\mathcal N(0,I)
$$

它的均值是：

$$
\boxed{
\mathbb E[z]=0
}
$$

所以：

$$
z=0
$$

就是 prior mean。

论文说：

> At test time, we set $z$ to be the mean of the prior distribution i.e. zero to deterministically decode.

官方代码完全一致：

```python
latent_sample = torch.zeros(
    [bs, self.latent_dim]
)
```

然后：

```python
latent_input =
    self.latent_out_proj(latent_sample)
```

---

## 37. 为什么 ACT 不随机采样？

论文给出的直接目的非常明确：

> **deterministically decode**

也就是说：

相同：

$$
o_t
$$

输入时，

希望 policy 输出保持 deterministic。

这对机器人 policy evaluation 很重要。

否则：

```text
同一初始条件
↓
每次 random z 不同
↓
动作风格不同
↓
policy evaluation 额外增加随机性
```

Appendix 也明确指出：

> 输入 observation 给定时，policy output 总是 deterministic，有利于 policy evaluation。

---

## 38. 但 z = 0 为什么还能工作？

现在可以给出比“因为 prior 均值是 0”更完整的理解。

训练时 KL 在不断约束：

$$
q_\phi(
z\mid a,q
)
$$

不要离：

$$
\mathcal N(0,I)
$$

太远。

所以 decoder / policy 在训练过程中看到的 $z$ 被鼓励分布在：

> unit Gaussian prior 所覆盖的区域附近。

而：

$$
z=0
$$

正是 prior 的中心。

因此推理时选择：

$$
0
$$

不是：

> 一个完全没训练过的奇怪 latent value。

它位于训练所使用 prior 的中心位置。

不过这里必须保持严谨：

> **这并不意味着 KL 数学上保证“z=0 一定是最优动作 style”。**

论文做的是一个具体 inference design：

- 用 prior mean；
- 获得 deterministic decoding；
- 实验上有效。

不要把它升级成不存在的 theorem。

---

## 39. z=0 也不意味着“把 style 信息全部删除了”

这是一个非常常见的直觉误区。

会有人说：

> 训练时 $z$ 编码 style，推理时 $z=0$，那岂不是 style 被删掉了？

要更仔细一点。

训练的最终对象是：

$$
\pi_\theta(
a_{t:t+k}
\mid
o_t,z
)
$$

这个 policy 的参数：

$$
\theta
$$

在整个训练过程中已经通过大量：

- observations；
- demonstrations；
- sampled $z$；

被优化过。

推理时 $z=0$ 的意思是：

> **固定 latent conditioning 到 prior mean 所对应的一个 canonical condition。**

不是：

> 把整个训练过程中关于 human behavior 学到的知识清空。

那些知识已经存在：

- model weights；
- image features；
- joint-state dependence；
- learned action-generation mapping；

里面。

所以 $z$ 是一个 conditioning variable，

而不是：

> 所有动作知识的唯一存储位置。

---

## 40. 一个更合适的直觉：训练时让模型“解释 variation”，推理时选一个中心 style

可以这样理解：

训练时 human demonstrations 有额外 variation：

```text
observation
+
不同 demonstration-specific variation
↓
action sequence
```

CVAE 用：

$$
z
$$

帮助模型把其中一部分 variation 显式表示出来。

而 inference 时 ACT 不希望随机选择 demonstration style。

于是：

> 固定 $z$ 到 prior center。

这相当于选择一个：

> **canonical latent condition**

再让 observation 本身决定动作。

这是一个有用的直觉。

但仍然要记住：

> “canonical style”是帮助理解的说法，不是论文对 $z=0$ 的严格语义证明。

---

## 41. β 为什么很重要？

ACT loss：

$$
L
=
L_{\mathrm{reconst}}
+
\beta
L_{\mathrm{KL}}
$$

论文使用：

$$
\beta=10
$$

并明确指出：

> 更大的 $\beta$ 会让 $z$ 传输更少的信息。

为什么？

如果：

$$
\beta
$$

很大，

模型会更强烈地惩罚：

$$
q_\phi(z|\cdot)
$$

偏离：

$$
\mathcal N(0,I)
$$

那么 encoder 就更难用：

$$
\mu,\sigma
$$

携带大量 demonstration-specific information。

所以：

```text
β ↑
↓
KL pressure ↑
↓
posterior 更接近 prior
↓
z 可携带的信息倾向减少
```

这与 $\beta$-VAE 的思想有关。

---

## 42. 那为什么不让 β 无限大？

如果：

$$
\beta\rightarrow\infty
$$

最简单的方式就是：

$$
q_\phi(z|\cdot)
\approx
\mathcal N(0,I)
$$

对于所有 demonstrations 都一样。

这样：

$$
z
$$

几乎完全不再表达 demonstration variation。

那么 CVAE encoder 就失去意义。

所以又是一个 trade-off：

```text
β 太小
→ z 可以记太多 demonstration-specific 信息
→ posterior 与 prior 差距可能很大

β 太大
→ z 几乎没信息
→ CVAE 可能退化
```

ACT 通过实验选择：

$$
\beta=10
$$

---

## 43. ACT 的 Ablation 证明 CVAE 有作用吗？

论文专门做了：

> Training with CVAE

的 ablation。

作者比较：

```text
ACT
vs
ACT without CVAE
```

结果发现：

> CVAE objective 对 human demonstration setting 尤其重要。

这与论文动机一致：

human demonstrations 具有：

- variability；
- non-stationarity；
- noisy / imprecise regions。

而 scripted demonstrations 更 deterministic。

所以 CVAE 的价值并不是：

> “Transformer 一定需要一个 latent variable。”

而是针对：

> **human demonstration distribution**

所引入的建模机制。

---

## 44. 为什么 Images 不进入 CVAE Encoder，却仍然能学到合理 z？

这是一个很好的问题。

CVAE encoder 的任务不是：

> 直接决定机器人现在应该怎么动。

它只需要：

> 从当前 proprioception + actual action sequence 中推断 latent style distribution。

真正要根据视觉环境决定动作的是：

> decoder / policy。

所以训练时：

```text
Encoder:
q_t + demonstration actions
→ z
```

而：

```text
Policy:
images + q_t + z
→ predicted actions
```

这种角色分工允许 encoder 省掉昂贵的 image processing，

同时 decoder 仍然拥有完整视觉 condition。

论文明确说这么做是：

> 为了更快训练。

---

## 45. z 会不会偷偷编码整个 Action Chunk？

理论上：

> encoder 有能力尝试这样做。

因为它看到了 ground-truth action sequence。

这正是 KL regularization 的重要原因之一。

如果没有 KL，encoder 可以把：

$$
a_{t:t+k}
$$

几乎完整编码进：

$$
z
$$

然后 decoder 只需把它解码回来。

这样 reconstruction 很好，

但 test time：

$$
z
$$

来自 prior 时就会崩溃。

所以 KL 限制了：

> training posterior 不能无限自由地把 target actions 全部塞进 $z$。

这也是为什么论文说：

> higher $\beta$ means less information transmitted in $z$。

---

## 46. [CLS] 是不是把 Action Chunk “压缩”成 z？

作为直觉：

> 可以这么说。

更严格地分成两步：

#### Step 1

Transformer encoder 将：

```text
[CLS]
+
joint state
+
action sequence
```

融合成 contextual hidden states。

其中：

$$
h_{\mathrm{CLS}}
$$

被选为 sequence-level representation。

---

#### Step 2

Linear projection 将：

$$
h_{\mathrm{CLS}}
$$

映射为：

$$
\mu,\log\sigma^2
$$

参数化：

$$
q_\phi(z|\cdot)
$$

再从中 sample：

$$
z
$$

所以严格来说：

> **不是 `[CLS]` 直接等于 $z$**。

关系是：

$$
\boxed{
[CLS]
\rightarrow
h_{\mathrm{CLS}}
\rightarrow
(\mu,\log\sigma^2)
\rightarrow
q(z|\cdot)
\rightarrow
z
}
$$

这条链非常重要。

---

## 47. h_CLS 是 z 吗？

**不是。**

官方代码中：

$$
h_{\mathrm{CLS}}
\in\mathbb R^{512}
$$

而：

$$
z\in\mathbb R^{32}
$$

中间还经过：

```python
latent_info =
    self.latent_proj(encoder_output)
```

因此：

$$
h_{\mathrm{CLS}}
$$

只是：

> approximate posterior parameter prediction 的 input representation。

它本身不是 latent style variable。

---

## 48. μ 是 z 吗？

也不是。

$$
\mu
$$

只是：

> $q(z|\cdot)$ 的 mean。

训练时：

$$
z
=
\mu+\sigma\epsilon
$$

所以一般：

$$
z\neq\mu
$$

除非：

$$
\epsilon=0
$$

或者：

$$
\sigma=0
$$

等特殊情况。

而 inference 时 ACT 使用：

$$
z=0
$$

甚至不计算：

$$
\mu
$$

因为整个 CVAE encoder 都被删掉了。

---

## 49. z=0 是不是 Encoder 学出来的？

**不是。**

训练阶段：

$$
z
$$

来自：

$$
q_\phi(z|\cdot)
$$

推理阶段：

CVAE encoder 不运行。

官方代码：

```python
mu = logvar = None

latent_sample = torch.zeros(
    [bs, latent_dim]
)
```

因此：

$$
z=0
$$

是：

> **人为规定的 inference-time prior-mean choice。**

不是 encoder 对 test observation 预测出来的。

---

## 50. 这也解释了为什么 inference 不需要 μ 和 σ

很多人会问：

> 推理时 observation 进入模型，为什么不再预测一个 $\mu$ 和 $\sigma$？

因为 ACT 没有一个：

$$
q(z\mid o_t)
$$

这样的 test-time encoder。

训练 encoder 是：

$$
q(z\mid a_{t:t+k},q_t)
$$

它依赖：

> ground-truth future actions。

推理时这些 actions 不存在。

因此这条网络没有办法使用。

ACT 没有另外训练 conditional prior：

$$
p(z\mid o_t)
$$

所以 test-time 直接使用固定 prior mean：

$$
z=0
$$

逻辑就闭合了。

---

## 51. 用概率图重新看 ACT CVAE

训练时可以概念化成：

```text
          q_t
         /   \
        /     \
       ▼       ▼
      z ─────→ future actions
       ▲
       │
ground-truth future actions
  (recognition only)

images ───────────────→ future actions
```

其中 recognition side：

$$
q_\phi(z\mid a,q)
$$

只帮助训练。

generative / policy side：

$$
\pi_\theta(a\mid o,z)
$$

才是真正 test-time 使用的模型。

---

## 52. 训练时 z 和推理时 z 的区别

可以做成一张表。

| | Training | Inference |
|---|---|---|
| Ground-truth action chunk | 有 | 没有 |
| CVAE encoder | 使用 | 丢弃 |
| $\mu,\log\sigma^2$ | 计算 | 不计算 |
| $z$ 来源 | $q_\phi(z\mid a,q)$ sample | fixed $z=0$ |
| Camera images | policy 使用 | policy 使用 |
| Joint positions | encoder + policy 使用 | policy 使用 |
| Policy decoder | 使用 | 使用 |
| 输出 | predicted action chunk | predicted action chunk |
| 随机性 | latent sampling | deterministic latent |

这个表基本就是 ACT CVAE 最重要的 train / test difference。

---

## 53. 官方代码中的 Training Branch

官方 `detr_vae.py` 判断：

```python
is_training = actions is not None
```

如果有 ground-truth：

```python
actions
```

就走 training branch：

```python
action_embed =
    self.encoder_action_proj(actions)

qpos_embed =
    self.encoder_joint_proj(qpos)

encoder_input = torch.cat(
    [cls_embed, qpos_embed, action_embed],
    axis=1
)

encoder_output =
    self.encoder(...)

encoder_output =
    encoder_output[0]

latent_info =
    self.latent_proj(encoder_output)

mu = latent_info[:, :latent_dim]

logvar =
    latent_info[:, latent_dim:]

latent_sample =
    reparametrize(mu, logvar)
```

这段代码几乎逐行对应论文 Appendix architecture。

---

## 54. 官方代码中的 Inference Branch

如果：

```python
actions is None
```

则：

```python
mu = logvar = None

latent_sample = torch.zeros(
    [bs, latent_dim]
)
```

然后训练和推理重新汇合：

```python
latent_input =
    self.latent_out_proj(latent_sample)
```

说明：

> policy 后半部分根本不在意这个 $z$ 是 training posterior sample 还是 zero vector。

它只接收一个 latent conditioning vector。

区别完全发生在：

> **z 是怎样产生的。**

---

## 55. 这也是理解代码最好的方式：网络不是两套，而是 z 的来源分叉

可以画成：

```text
                    TRAINING
actions + qpos
      │
      ▼
 CVAE Encoder
      │
      ▼
 μ, logσ²
      │
      ▼
sample z
      │
      │
      ├─────────────────────┐
                            │
                    shared policy
                            │
                            ▼
                      action chunk


                    INFERENCE

                   z = 0
                     │
                     └──────→ shared policy
                                   │
                                   ▼
                             action chunk
```

也就是说：

> ACT 并不是 train 时一个 policy、test 时另一个 policy。

真正的 action-generating policy：

$$
\pi_\theta
$$

是共享的。

变化的只是 latent source。

---

## 56. ACT 为什么需要 CVAE，而不是直接加 Noise？

你也可以想：

> 既然 human demonstrations 有 variation，那训练 action prediction 时随便加 noise 不行吗？

区别在于：

随机 noise：

$$
\epsilon
$$

本身并不会根据 demonstration 学会：

> 哪些 variation 有助于解释 action sequence。

而 CVAE encoder：

$$
q_\phi(z\mid a,q)
$$

会根据当前实际 action sequence 推断一个 posterior。

所以：

$$
z
$$

不是纯粹无语义 noise。

它是：

> **被训练成有助于解释 human action-chunk variation 的 latent variable。**

reparameterization 里的：

$$
\epsilon
$$

才是纯辅助随机噪声。

再次强调：

$$
\boxed{
z\neq\epsilon
}
$$

---

## 57. 为什么不是直接把 Action Sequence 编成一个 Deterministic Vector？

例如：

$$
z=f(a,q)
$$

然后 decoder reconstruction。

这样就变成类似：

> conditional autoencoder。

问题是：

> test time 没有 action sequence。

如果 training latent codes 可以随意形成一个 deterministic latent cloud，

test time 没有一个明确、匹配 training distribution 的方式生成：

$$
z
$$

而 CVAE 通过：

$$
D_{KL}
(
q_\phi(z|\cdot)
\parallel
\mathcal N(0,I)
)
$$

把 training posterior 约束到一个已知 prior 周围。

于是 test time 才有：

$$
p(z)=\mathcal N(0,I)
$$

这个明确的 latent reference distribution。

ACT 再进一步选择它的 mean：

$$
0
$$

作为 deterministic inference condition。

---

## 58. 一个非常重要的理解：CVAE 在 ACT 里主要是 Training Device + Latent Regularizer

从最终 inference 角度：

```text
CVAE encoder
消失

z
固定为 0
```

这容易让人觉得：

> “那 CVAE 不是白训练了吗？”

不是。

CVAE 在 training 阶段改变了 policy 学习 demonstration data 的方式。

它允许：

> action-sequence variability 在训练时通过 latent variable 被显式建模。

并通过：

$$
KL
$$

限制这种 latent information。

最终得到的 decoder / policy 参数：

$$
\theta
$$

是在这种训练目标下学出来的。

所以即使 test time 不再运行 encoder：

> **CVAE training objective 仍然塑造了最终 policy。**

这和：

> dropout 训练时存在、推理时关闭

虽然数学机制完全不同，但有一个高层共同点：

> 某个 training mechanism 不一定需要在 test time 原样存在，仍然可以影响最终 learned model。

这只是帮助理解的类比，不要把 CVAE 和 dropout 视为同一机制。

---

## 59. 常见误解一：[CLS] 本身就是 z

**错误。**

真正链条：

$$
[CLS]
\rightarrow
h_{\mathrm{CLS}}
\rightarrow
\mu,\log\sigma^2
\rightarrow
z
$$

---

## 60. 常见误解二：Linear Layer 学习把 μ 变成 0、variance 变成 1

**不准确。**

Linear layer 学的是：

$$
h_{\mathrm{CLS}}
\rightarrow
q_\phi(z|\cdot)
\text{ 的参数}
$$

KL 会推动 posterior 向 prior 靠近，

reconstruction 又要求 $z$ 保留有用信息。

---

## 61. 常见误解三：z 是完全随机猜出来的

**错误。**

训练时：

$$
z
$$

来自：

$$
q_\phi(z\mid a,q)
$$

而这个 distribution 是 encoder 根据当前 demonstration 推断出的。

随机只发生在 learned posterior 内部。

---

## 62. 常见误解四：z 就是 ε

**错误。**

$$
\epsilon
\sim
\mathcal N(0,I)
$$

是 auxiliary noise。

$$
z
=
\mu+\sigma\epsilon
$$

是 latent sample。

---

## 63. 常见误解五：CVAE Encoder 训练时看 Camera Images

ACT 实际实现中：

**不看。**

论文为了加快训练，让 encoder 只 condition on：

- proprioceptive observation；
- action sequence。

images 进入的是：

> decoder / policy。

---

## 64. 常见误解六：推理时 z 还是 Encoder 预测的

**错误。**

test time：

> encoder 被 discard。

官方代码直接：

$$
z=0
$$

---

## 65. 常见误解七：推理时 z=0 是因为 z 没用

**错误。**

$z$ 在 training objective 中承担 latent variation modeling。

test time：

$$
z=0
$$

是选取 prior mean 做 deterministic decoding。

不能反推：

> “训练时 z 没发挥作用。”

---

## 66. 常见误解八：一般 CVAE 推理都必须 z=0

**错误。**

一般 CVAE 完全可以：

$$
z\sim p(z)
$$

随机生成多个 outputs。

$$
z=0
$$

是 ACT 的具体 inference design。

---

## 67. 常见误解九：论文 ACT reconstruction 就是 MSE

需要更谨慎。

Algorithm 1 写 MSE。

但正文明确说实际使用：

$$
L1
$$

官方代码也是：

$$
L1
$$

所以如果讨论：

> **实际 ACT implementation**

应以 L1 为准。

---

## 68. 常见误解十：CVAE Decoder 就是 Transformer Decoder

**错误。**

ACT 论文里的：

> CVAE decoder / policy

是完整 action-generating network。

其中包含：

- ResNet；
- Transformer encoder；
- Transformer decoder；
- output projection。

---

## 69. 用五条公式记住 ACT 的 CVAE

如果最后只留下五条公式：

---

### 1. Training Encoder

$$
\boxed{
q_\phi(
z
\mid
a_{t:t+k},
\bar o_t
)
}
$$

其中：

$$
\bar o_t
$$

不包含 images。

---

### 2. Approximate Posterior

$$
\boxed{
q_\phi(z|\cdot)
=
\mathcal N
(
\mu,
\operatorname{diag}(\sigma^2)
)
}
$$

---

### 3. Reparameterization

$$
\boxed{
z
=
\mu+\sigma\odot\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I)
}
$$

---

### 4. Policy

$$
\boxed{
\pi_\theta(
\hat a_{t:t+k}
\mid
o_t,z
)
}
$$

---

### 5. Prior Regularization

$$
\boxed{
D_{KL}
\left(
q_\phi(z|\cdot)
\parallel
\mathcal N(0,I)
\right)
}
$$

推理时：

$$
\boxed{
z=0
}
$$

---

## 70. 一句话重新理解

> **ACT 的 CVAE encoder 在训练时阅读当前 joint state 和真实 future action chunk，用 `[CLS]` 的 Transformer representation 预测一个 Gaussian latent posterior $q_\phi(z|\cdot)$；从中重参数化采样得到的 $z$ 作为额外 condition 输入真正的 ACT policy，帮助模型解释 human demonstrations 中的 action-sequence variation。**

KL：

$$
D_{KL}
(
q_\phi(z|\cdot)
\parallel
\mathcal N(0,I)
)
$$

限制 $z$ 不要成为一个完全自由的 demonstration memory。

而推理时 ground-truth future actions 不存在，

所以 training encoder 被丢弃。

ACT 不随机 sample prior，而是直接选择：

$$
z=0
$$

即 unit Gaussian prior 的均值，

从而让 policy deterministic 地产生 action chunk。

---

## 71. 下一步

现在我们已经完整走通：

```text
Action Chunking
↓
Temporal Ensemble
↓
Latent Variable
↓
VAE
↓
Reparameterization
↓
CVAE
↓
CVAE in ACT
```

下一篇最自然的主题有两个。

如果继续解决最容易困惑的概率问题：

> [为什么 ACT 推理时令 z = 0？](./why-z-zero-at-inference.md)

它会专门回答：

- 为什么训练 posterior 必须向 $\mathcal N(0,I)$ 靠近；
- 为什么 prior mean 是 0；
- $z=0$ 到底删除了什么、又没有删除什么；
- 为什么不是随机 sample；
- 为什么不能把它理解成“把额外因素全部消除掉”；
- $\beta$ 和 $z=0$ 有什么关系。

如果开始进入 ACT 的完整 Transformer 网络：

> [ACT Architecture](./architecture.md)

会继续拆：

```text
4 Camera Images
↓
ResNet18
↓
Visual Tokens
↓
Transformer Encoder
+
Joint Token
+
Latent Token
↓
Transformer Decoder
↓
k Action Queries
↓
k × 14 Actions
```

---

### Primary Source

Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.  
**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.**  
Robotics: Science and Systems (RSS), 2023.

- Paper: https://arxiv.org/abs/2304.13705
- RSS Proceedings: https://roboticsproceedings.org/rss19/p016.html
- Project: https://tonyzhaozh.github.io/aloha/

本文主要依据：

- Section IV-B — Modeling human data
- Section IV-C — Implementing ACT
- Algorithm 1 — ACT Training
- Algorithm 2 — ACT Inference
- Figure 3 — ACT architecture
- Appendix C — Detailed Architecture Diagram
- Figure 10 — Detailed ACT architecture
- Appendix D / Table III — ACT hyperparameters

---

### Official Implementation

Official ACT repository:

https://github.com/tonyzhaozh/act

主要对应：

```text
detr/models/detr_vae.py
policy.py
```

官方实现确认：

- latent dimension = 32；
- `[CLS]` + joint + action sequence 进入 CVAE encoder；
- `[CLS]` output 经过 Linear 得到 $\mu$ 和 `logvar`；
- training 使用 reparameterized $z$；
- inference 使用全零 latent vector；
- actual ACT reconstruction implementation 使用 L1；
- loss 为 L1 + weighted KL。

---

### 本文知识连接

#### 前置知识

- [ACT 到底解决了什么问题？](./act-what-problem-does-it-solve.md)
- [Action Chunking](./action-chunking.md)
- [Latent Variable](../../generative-models/latent-variable.md)
- [VAE](../../generative-models/vae.md)
- [Reparameterization Trick](../../generative-models/reparameterization-trick.md)
- [CVAE](../../generative-models/cvae.md)

#### Transformer 基础

- [Attention](../../deep-learning/attention.md)
- [Query / Key / Value](../../deep-learning/qkv.md)
- [Transformer](../../deep-learning/transformer.md)
- CLS Token
- [Positional Encoding](../../deep-learning/positional-encoding.md)

#### 数学

- [Normal Distribution](../../mathematics/normal-distribution.md)
- Standard Normal Distribution
- Mean
- Variance
- [KL Divergence](../../mathematics/kl-divergence.md)

#### 下一步

- [为什么 ACT 推理时令 z = 0？](./why-z-zero-at-inference.md)
- [ACT Architecture](./architecture.md)
- [ACT Training](./training.md)
- [ACT Inference](./inference.md)
