---
title: "ACT Complete Data Flow：从一条 Demonstration 到机器人真正执行动作"
description: "用一个 k=4 的具体例子，把 ACT 的训练与推理放到同一条时间线上：从 demonstration sample、CVAE posterior、Transformer policy，到 overlapping chunks、Temporal Ensemble 与最终机器人动作。"
status: reviewed
pageType: application
canonical: /robot-learning/act/complete-data-flow
updated: "2026-09-15"
---

# ACT Complete Data Flow：从一条 Demonstration 到机器人真正执行动作

到这里，我们已经分别学过：

- [ACT 到底解决了什么问题？](./act-what-problem-does-it-solve.md)
- [Action Chunking](./action-chunking.md)
- [Temporal Ensemble](./temporal-ensemble.md)
- [CVAE in ACT](./cvae-in-act.md)
- [为什么 ACT 推理时令 \(z=0\)？](./why-z-zero-at-inference.md)
- [ACT Architecture](./architecture.md)
- [ACT Training](./training.md)
- [ACT Inference](./inference.md)

这些文章分别回答了一个局部问题。

但真正理解一个模型，最后必须做到一件事：

> **不用在不同章节之间跳来跳去，也能从原始 demonstration 一直讲到机器人真正执行某一个 action。**

这一篇不增加新的核心理论。

我们做一次完整“通关”。

为了让所有 index 都能直接写出来，本文使用一个很小的教学例子：

\[
\boxed{k=4}
\]

真实 ACT 论文常用：

\[
k=100
\]

但 \(k=4\) 和 \(k=100\) 的算法逻辑完全相同。

---

# 1. 先约定本文的记号

为了避免论文记号：

\[
a_{t:t+k}
\]

可能带来的端点歧义，本文明确规定：

> 一个长度为 \(k\) 的 action chunk 覆盖：
>
> \[
> t,t+1,\ldots,t+k-1
> \]

因此当：

\[
k=4
\]

时：

\[
A_t
=
[
a_t,
a_{t+1},
a_{t+2},
a_{t+3}
]
\]

可以把它写成半开区间：

\[
a_{t:t+4}
\]

本文所有例子都遵守这个 convention。

---

# 2. 我们先构造一条非常短的 Demonstration

假设一条 human demonstration 有 8 个 timestep：

\[
t=0,1,\ldots,7
\]

每个 timestep 都记录：

\[
o_t
=
(
I_t,q_t
)
\]

其中：

- \(I_t\)：4 路 RGB images；
- \(q_t\in\mathbb R^{14}\)：当前双臂 joint positions。

专家还给出 action：

\[
a_t\in\mathbb R^{14}
\]

所以整条 episode：

```text
t=0   o0   a0
t=1   o1   a1
t=2   o2   a2
t=3   o3   a3
t=4   o4   a4
t=5   o5   a5
t=6   o6   a6
t=7   o7   a7
```

ACT 的任务不是学习：

\[
o_t\rightarrow a_t
\]

而是学习：

\[
\boxed{
o_t
\rightarrow
[a_t,a_{t+1},a_{t+2},a_{t+3}]
}
\]

在我们的 \(k=4\) 例子里。

---

# 3. Training Sample：随机抽到 t=2

假设这次 dataset 随机选择：

\[
t=2
\]

那么当前 observation 是：

\[
o_2
\]

即：

\[
o_2=(I_2,q_2)
\]

target action chunk 是：

\[
\boxed{
A_2
=
[a_2,a_3,a_4,a_5]
}
\]

所以这一条监督样本可以先粗略写成：

\[
\boxed{
(o_2,A_2)
}
\]

这已经体现了 Action Chunking：

> 一个当前 observation，对应未来连续 4 步动作。

---

# 4. 先做数据预处理

真实机器人 qpos：

\[
q_2
\]

不会直接送进模型。

训练集已经统计：

\[
\mu_q,\sigma_q
\]

所以：

\[
\tilde q_2
=
\frac{
q_2-\mu_q
}{
\sigma_q
}
\]

同样，每一个 action：

\[
a_i
\]

也变成 normalized：

\[
\tilde a_i
=
\frac{
a_i-\mu_a
}{
\sigma_a
}
\]

所以模型真正看到的 target chunk 是：

\[
\tilde A_2
=
[
\tilde a_2,
\tilde a_3,
\tilde a_4,
\tilde a_5
]
\]

图片也经过与训练一致的 image preprocessing。

---

# 5. 第一次分叉：Training 时 Target Action 有两个用途

这是理解 ACT 非常重要的一点。

ground-truth chunk：

\[
\tilde A_2
\]

一方面是：

> **最终 reconstruction target。**

但它还有另一个用途：

> **送进 CVAE encoder，帮助推断 training latent \(z\)。**

也就是说，同一份 expert action chunk 会走两条路：

```text
                     ┌──→ CVAE Encoder
ground-truth A₂ ─────┤
                     └──→ Reconstruction Target
```

这不是 data leakage。

因为 CVAE encoder：

> 只在 training 存在。

部署时它会完全消失。

---

# 6. CVAE Encoder 的输入序列

Training 时 encoder 输入：

```text
[CLS]
q₂
a₂
a₃
a₄
a₅
```

因为：

\[
k=4
\]

所以 sequence length：

\[
k+2=6
\]

每一个 qpos / action 原本是：

\[
14
\]

维。

经过 Linear projection 后变成：

\[
512
\]

维 token。

所以 CVAE encoder 输入可以写成：

\[
X_{\text{CVAE}}
\in
\mathbb R^{6\times512}
\]

忽略 batch dimension。

---

# 7. Transformer Encoder 让 [CLS] 汇总这条 Demonstration

经过 BERT-like Transformer encoder：

\[
X_{\text{CVAE}}
\longrightarrow
H_{\text{CVAE}}
\]

仍然有 6 个 contextual hidden states：

\[
[
h_{\mathrm{CLS}},
h_q,
h_{a_2},
h_{a_3},
h_{a_4},
h_{a_5}
]
\]

ACT 只取：

\[
\boxed{
h_{\mathrm{CLS}}
}
\]

然后：

\[
h_{\mathrm{CLS}}
\rightarrow
\mu,\log\sigma^2
\]

官方 latent dimension 为：

\[
32
\]

所以：

\[
\mu\in\mathbb R^{32}
\]

\[
\log\sigma^2\in\mathbb R^{32}
\]

---

# 8. 现在模型得到的是一个 Distribution，不是一个固定 z

CVAE encoder 定义：

\[
q_\phi(
z\mid
\tilde A_2,\tilde q_2
)
=
\mathcal N
\left(
\mu,
\operatorname{diag}(\sigma^2)
\right)
\]

这句话非常重要。

Encoder 并不是说：

> “这条 demonstration 的 latent code 就是某个固定向量。”

而是：

> “给定这条 demonstration，我认为合理的 latent \(z\) 分布是这个 Gaussian。”

---

# 9. Reparameterization：从 Posterior 中采样

采：

\[
\epsilon
\sim
\mathcal N(0,I)
\]

然后：

\[
\boxed{
z
=
\mu+\sigma\odot\epsilon
}
\]

得到：

\[
z\in\mathbb R^{32}
\]

这个 \(z\)：

- 不是 `[CLS]`；
- 不是 \(\mu\)；
- 不是 \(\epsilon\)；
- 是从 learned posterior 中得到的 latent sample。

---

# 10. 到这里，CVAE Encoder 的工作结束了

现在 Training Branch 已经得到：

\[
z
\]

接下来进入真正负责动作预测的：

> **ACT Policy / CVAE Decoder**

它的输入是：

\[
(I_2,\tilde q_2,z)
\]

而不是：

\[
A_2
\]

注意：

> Ground-truth future action 不会直接输入真正的 action-generating Transformer policy。

它只通过：

\[
z
\]

间接影响 training policy。

---

# 11. 4 张图片进入 ResNet18

当前时刻：

\[
t=2
\]

有 4 张：

\[
480\times640
\]

RGB images。

每张经过 ResNet18：

\[
480\times640\times3
\rightarrow
15\times20\times512
\]

flatten spatial dimensions：

\[
15\times20=300
\]

所以每张：

\[
300\times512
\]

四张合起来：

\[
\boxed{
1200\times512
}
\]

visual tokens。

---

# 12. qpos 和 z 也变成 Token

normalized qpos：

\[
\tilde q_2\in\mathbb R^{14}
\]

投影：

\[
14\rightarrow512
\]

得到：

\[
e_q\in\mathbb R^{512}
\]

latent：

\[
z\in\mathbb R^{32}
\]

投影：

\[
32\rightarrow512
\]

得到：

\[
e_z\in\mathbb R^{512}
\]

于是 policy encoder 输入一共：

\[
1200+1+1
=
1202
\]

个 token。

---

# 13. Policy Transformer Encoder 融合 Current Context

输入：

\[
X_{\text{policy}}
\in
\mathbb R^{1202\times512}
\]

其中包含：

```text
z token
joint token
1200 visual tokens
```

经过多层 self-attention：

\[
X_{\text{policy}}
\longrightarrow
M_2
\]

得到：

\[
\boxed{
M_2\in\mathbb R^{1202\times512}
}
\]

这个：

\[
M_2
\]

就是当前时刻：

\[
t=2
\]

的 observation memory。

---

# 14. k=4，所以 Decoder 有 4 个 Action Slots

现在我们需要预测：

\[
[
a_2,a_3,a_4,a_5
]
\]

所以 decoder 有 4 个 action query positions：

\[
Q=
[
q^{action}_0,
q^{action}_1,
q^{action}_2,
q^{action}_3
]
\]

可以直觉理解为：

```text
query 0 → chunk 第 0 个 action
query 1 → chunk 第 1 个 action
query 2 → chunk 第 2 个 action
query 3 → chunk 第 3 个 action
```

不是：

> 4 个独立模型。

它们会通过 decoder self-attention 相互交流。

---

# 15. Decoder 读取 Encoder Memory

每个 action slot 通过 cross-attention：

\[
Q
\rightarrow
M_2
\]

读取：

- visual features；
- qpos context；
- latent condition。

得到：

\[
H_{\text{dec}}
\in
\mathbb R^{4\times512}
\]

然后 action head：

\[
512\rightarrow14
\]

所以输出：

\[
\boxed{
\hat A_2
=
[
\hat a_2,
\hat a_3,
\hat a_4,
\hat a_5
]
\in
\mathbb R^{4\times14}
}
\]

---

# 16. 训练时不会真正执行这 4 个预测

这是训练和 inference 最根本的区别之一。

Training 时：

\[
\hat A_2
\]

只是一个 prediction。

不会：

```text
发给机器人
↓
让机器人移动
```

而是直接与 demonstration target：

\[
\tilde A_2
\]

比较。

---

# 17. Reconstruction Loss

实际 ACT implementation 使用 L1：

\[
\boxed{
L_{\text{recon}}
=
\operatorname{L1}
(
\hat A_2,
\tilde A_2
)
}
\]

如果 episode 尾部存在 padding，

则 padded positions 被 mask。

我们的例子：

\[
t=2
\]

离 episode 结尾还够远，

所以：

\[
a_2,a_3,a_4,a_5
\]

全部是真实动作，不需要 padding。

---

# 18. 同时计算 KL

posterior：

\[
q_\phi(
z\mid
\tilde A_2,\tilde q_2
)
\]

被 regularize toward：

\[
p(z)=\mathcal N(0,I)
\]

所以：

\[
\boxed{
L_{KL}
=
D_{KL}
\left(
q_\phi(z|\cdot)
\parallel
\mathcal N(0,I)
\right)
}
\]

最终：

\[
\boxed{
L
=
L_{\text{recon}}
+
\beta L_{KL}
}
\]

原始 ACT：

\[
\beta=10
\]

---

# 19. 一次 Backward 会同时训练哪些东西？

从 L1：

\[
L_{\text{recon}}
\]

梯度会经过：

```text
Action Head
↑
Policy Transformer Decoder
↑
Policy Transformer Encoder
↑
ResNet / qpos projection / z projection
```

同时由于：

\[
z=\mu+\sigma\epsilon
\]

reconstruction gradient 还能继续：

```text
z
↑
μ, σ
↑
CVAE Encoder
```

而 KL 又直接训练：

\[
\mu,\sigma
\]

对应的 posterior。

因此一次 optimizer step 联合更新：

> **CVAE encoder + ACT policy。**

---

# 20. 这一个 Training Sample 做完之后发生什么？

optimizer：

```text
backward
↓
AdamW step
↓
parameters change
```

下一次 dataset 可能：

- 选另一条 episode；
- 或同一条 episode 的另一个 timestep。

例如下一次可能选：

\[
t=3
\]

则 target 变成：

\[
A_3
=
[a_3,a_4,a_5,a_6]
\]

你会发现：

\[
A_2
\]

和：

\[
A_3
\]

高度 overlap。

这正是 action-chunk training 中的正常现象。

---

# 21. Training 到这里结束，下面进入 Inference

现在假设模型已经训练好了。

保存了：

- policy weights；
- dataset qpos/action mean/std。

CVAE training encoder 仍然存在于 checkpoint 参数中，

但 rollout 时：

> **不会使用它。**

---

# 22. Inference t=0：第一次看到环境

机器人刚 reset。

获得：

\[
o_0=(I_0,q_0)
\]

qpos normalize：

\[
\tilde q_0
=
\frac{
q_0-\mu_q
}{
\sigma_q
}
\]

图片做相同 preprocessing。

然后：

\[
\boxed{z=0}
\]

注意：

- 不计算 \(\mu\)；
- 不计算 \(\sigma\)；
- 不运行 CVAE encoder；
- 不需要 future ground-truth actions。

---

# 23. t=0：Policy 输出第一个 Chunk

policy：

\[
\pi_\theta(
o_0,z=0
)
\]

得到：

\[
\hat A_0^{(0)}
=
[
\hat a_0^{(0)},
\hat a_1^{(0)},
\hat a_2^{(0)},
\hat a_3^{(0)}
]
\]

这里我把整个 chunk 上标也写成：

\[
(0)
\]

表示：

> 这是 query time 0 产生的预测。

---

# 24. t=0：Buffer 里有什么？

目前只有一个 chunk。

可以画成：

```text
execution time
          0       1       2       3

query 0   a₀⁰     a₁⁰     a₂⁰     a₃⁰
```

当前 execution time：

\[
t=0
\]

只有：

\[
\hat a_0^{(0)}
\]

所以 Temporal Ensemble 退化成：

\[
a_0^{norm}
=
\hat a_0^{(0)}
\]

没有真正的“ensemble”。

---

# 25. t=0：De-normalize 并执行

policy output 仍在 normalized action space。

所以：

\[
a_0
=
a_0^{norm}
\odot
\sigma_a
+
\mu_a
\]

得到真实：

\[
14
\]

维 target joint positions。

发送给 low-level controller：

```text
ACT target qpos
↓
PID
↓
motors
↓
robot moves
```

环境变化。

---

# 26. t=1：ACT 不会直接执行刚才 Chunk 中的 a₁⁰

这是最终 ACT 与 naive chunk execution 的关键区别。

如果 naive：

> 现在直接执行：

\[
\hat a_1^{(0)}
\]

就行。

但最终 ACT 开启 Temporal Ensemble 时：

> **先重新观察。**

获取：

\[
o_1
\]

然后再运行 policy。

---

# 27. t=1：产生第二个 Chunk

\[
\hat A_1^{(1)}
=
[
\hat a_1^{(1)},
\hat a_2^{(1)},
\hat a_3^{(1)},
\hat a_4^{(1)}
]
\]

现在 buffer：

```text
execution time
          0       1       2       3       4

query 0   a₀⁰     a₁⁰     a₂⁰     a₃⁰

query 1           a₁¹     a₂¹     a₃¹     a₄¹
```

---

# 28. t=1：为什么有两个候选？

因为 execution time：

\[
1
\]

已经被两个 chunks 预测过：

### query 0 的旧预测

\[
\hat a_1^{(0)}
\]

它是在：

\[
o_0
\]

基础上提前一步预测的。

### query 1 的新预测

\[
\hat a_1^{(1)}
\]

它已经看到了最新：

\[
o_1
\]

---

# 29. t=1：Temporal Ensemble

按照从旧到新排列：

\[
A_1=
[
\hat a_1^{(0)},
\hat a_1^{(1)}
]
\]

权重：

\[
w_0=e^{-m\cdot0}=1
\]

\[
w_1=e^{-m}
\]

归一化：

\[
\alpha_i
=
\frac{w_i}{w_0+w_1}
\]

最终：

\[
\boxed{
a_1^{norm}
=
\alpha_0
\hat a_1^{(0)}
+
\alpha_1
\hat a_1^{(1)}
}
\]

然后 de-normalize 并执行。

---

# 30. t=2：第三次 Re-plan

机器人执行完：

\[
a_1
\]

后，环境来到：

\[
o_2
\]

重新 forward：

\[
\hat A_2^{(2)}
=
[
\hat a_2^{(2)},
\hat a_3^{(2)},
\hat a_4^{(2)},
\hat a_5^{(2)}
]
\]

现在：

```text
          0       1       2       3       4       5

q0        a₀⁰     a₁⁰     a₂⁰     a₃⁰

q1                a₁¹     a₂¹     a₃¹     a₄¹

q2                         a₂²     a₃²     a₄²     a₅²
```

当前：

\[
t=2
\]

有三个候选：

\[
\hat a_2^{(0)}
\]

\[
\hat a_2^{(1)}
\]

\[
\hat a_2^{(2)}
\]

---

# 31. t=2：三个不同 Observation 对同一个执行时刻投票

注意它们的目标都是：

\[
\boxed{
\text{execution time}=2
}
\]

但产生它们时看到的信息分别是：

\[
o_0
\]

\[
o_1
\]

\[
o_2
\]

所以可以想成：

```text
旧计划：
“我在 t=0 时觉得 t=2 应该这么做”

中间计划：
“我在 t=1 时重新看环境，觉得 t=2 应该这么做”

最新计划：
“现在就是 t=2，我根据最新 observation 觉得应该这么做”
```

Temporal Ensemble 把三者组合。

---

# 32. t=3：第一次积累满 k=4 个候选

在：

\[
t=3
\]

新 chunk：

\[
[
\hat a_3^{(3)},
\hat a_4^{(3)},
\hat a_5^{(3)},
\hat a_6^{(3)}
]
\]

现在 execution time 3 的候选：

\[
\boxed{
\hat a_3^{(0)},
\hat a_3^{(1)},
\hat a_3^{(2)},
\hat a_3^{(3)}
}
\]

正好：

\[
4=k
\]

从此进入 episode 中间以后，

当前 timestep 通常最多保持：

\[
k
\]

个候选。

---

# 33. 为什么 t=4 时不会变成 5 个？

看覆盖：

query 0 的 chunk：

\[
0,1,2,3
\]

它并没有预测：

\[
t=4
\]

所以到了：

\[
t=4
\]

候选来自：

- query 1；
- query 2；
- query 3；
- query 4。

仍然是：

\[
4
\]

个。

所以 candidate window 像一个不断向前滑动的窗口：

```text
t=3:
q0 q1 q2 q3

t=4:
   q1 q2 q3 q4

t=5:
      q2 q3 q4 q5
```

---

# 34. 这就是 ACT 的 Rolling Action-Chunk Window

从整个 rollout 看：

```text
             future
t=0  [0 1 2 3]
t=1    [1 2 3 4]
t=2      [2 3 4 5]
t=3        [3 4 5 6]
t=4          [4 5 6 7]
```

每个 timestep：

> 新增一个 chunk。

同时最旧的 chunk 在超过自己的 horizon 后：

> 自动退出当前 candidate set。

---

# 35. Training 和 Inference 中“同一个 Chunk”其实角色不同

Training：

\[
A_t
\]

是：

> ground-truth expert chunk。

模型的任务：

\[
\hat A_t
\approx
A_t
\]

---

Inference：

\[
\hat A_t
\]

是：

> policy 自己产生的 future prediction。

它不会拿来计算 loss。

而会：

> 写进 temporal buffer。

这两个阶段都叫 action chunk，

但一个是：

> supervision target，

另一个是：

> rollout prediction。

---

# 36. Training 和 Inference 中 z 的角色也不同

## Training

\[
z
\sim
q_\phi(z\mid A_t,q_t)
\]

它可以随不同 demonstration 改变。

目的是：

> 帮助模型解释 human action-sequence variation。

---

## Inference

\[
\boxed{z=0}
\]

固定。

目的：

> prior-mean deterministic decoding。

因此：

```text
Training:
demonstration-specific stochastic latent

Inference:
fixed canonical latent
```

---

# 37. 一个非常重要的总关系

ACT 真正 deployed 的 policy 不是：

\[
o_t
\rightarrow a_t
\]

而是：

\[
\boxed{
(o_t,z=0)
\rightarrow
\hat A_t
}
\]

其中：

\[
\hat A_t
=
[
\hat a_t,
\ldots,
\hat a_{t+k-1}
]
\]

然后 execution layer 再把：

> 多次 policy forward

组合成当前：

\[
a_t
\]

所以 ACT 有两个明显层次：

## Prediction Layer

\[
o_t
\rightarrow
\text{action chunk}
\]

## Execution Layer

\[
\text{overlapping chunks}
\rightarrow
\text{current action}
\]

Temporal Ensemble 属于第二层。

---

# 38. Transformer 在这条完整链里到底负责什么？

现在可以非常清楚地回答。

Transformer **不是**：

> 负责 Temporal Ensemble。

也不是：

> 负责 KL。

它主要有两个位置。

---

## CVAE Transformer Encoder

Training only：

\[
[CLS]+q+A
\rightarrow
h_{CLS}
\rightarrow
\mu,\log\sigma^2
\]

目的：

> infer latent posterior。

---

## Policy Transformer

Training + Inference：

```text
images + q + z
↓
Transformer Encoder
↓
observation memory
↓
Transformer Decoder
↓
future action sequence
```

目的：

> sequence prediction。

---

# 39. ResNet 在整条链里只负责视觉编码

ResNet 不知道：

- KL；
- Temporal Ensemble；
- action chunk buffer。

它只做：

\[
I_t
\rightarrow
\text{visual feature sequence}
\]

然后这些 features 才进入 Transformer。

所以模块责任可以拆成：

```text
ResNet
→ 看图

Transformer Encoder
→ 融合当前上下文

Transformer Decoder
→ 形成未来 action sequence

CVAE
→ 训练时处理 demonstration variation

Temporal Ensemble
→ 推理时融合 overlapping predictions
```

---

# 40. Action Chunking 到底发生在哪？

其实存在三个层面的体现。

## 数据层

training target：

\[
A_t
=
[a_t,\ldots,a_{t+k-1}]
\]

---

## 网络层

decoder 有：

\[
k
\]

个 action-query positions，

一次输出：

\[
k\times14
\]

---

## 推理层

每次 query 都产生长度：

\[
k
\]

的 future prediction window。

所以 Action Chunking 不是某一行代码。

它贯穿：

> target definition → architecture → inference。

---

# 41. Temporal Ensemble 到底发生在哪？

只发生在：

> inference execution layer。

它不参与：

- dataset target construction；
- CVAE encoder；
- training loss；
- backpropagation。

可以写成：

\[
\boxed{
\text{Temporal Ensemble}
=
\text{post-policy inference aggregation}
}
\]

---

# 42. KL 到底发生在哪？

只发生在：

> training objective。

它约束：

\[
q_\phi(z\mid A,q)
\]

靠近：

\[
N(0,I)
\]

它不直接平滑动作，

不参与 test-time weighted average。

---

# 43. Normalization 到底发生在哪？

它横跨 train 和 test。

Training：

\[
q,a
\rightarrow
\text{normalized coordinates}
\]

Inference input：

\[
q_{\text{physical}}
\rightarrow
q_{\text{norm}}
\]

Inference output：

\[
a_{\text{norm}}
\rightarrow
a_{\text{physical}}
\]

所以 normalization statistics 实际上也是：

> deployed model contract 的一部分。

---

# 44. 一个从训练到部署的完整生命周期

可以把 ACT 整个项目理解成 4 个阶段。

---

## Phase 1：Collect Demonstrations

Human：

```text
teleoperate robot
↓
record images
record follower qpos
record leader joint positions as actions
```

得到：

\[
\mathcal D
\]

---

## Phase 2：Train

不断采：

\[
(o_t,A_t)
\]

然后：

```text
A_t + q_t
→ posterior z

images + q_t + z
→ predicted A_t

L1 + β KL
→ update network
```

---

## Phase 3：Select Checkpoint

validation：

\[
\rightarrow
policy\_best.ckpt
\]

保存 normalization stats。

---

## Phase 4：Rollout

每 timestep：

```text
observation
↓
z = 0
↓
predict chunk
↓
temporal ensemble
↓
execute current action
↓
new observation
```

---

# 45. 为什么 ACT 不是“训练时预测 Chunk，推理时预测单步”？

因为 inference policy forward 仍然输出：

\[
k
\]

个 actions。

只是 execution layer 最终只从 overlapping chunks 中选出：

\[
\text{current timestep}
\]

真正执行的一个 action。

所以：

> **模型层面始终是 chunk predictor。**

execution layer 才把多个 chunk prediction 转成逐 timestep 控制。

---

# 46. 为什么每一步重新 Query 不破坏“预测未来”的意义？

假设：

\[
t=0
\]

模型预测：

\[
a_0,a_1,a_2,a_3
\]

虽然它最终马上只执行：

\[
a_0
\]

但在形成：

\[
a_0
\]

对应 hidden representation 时，

decoder 的 future action slots 可以相互 self-attend。

模型是在：

> 一个 sequence prediction problem

中学习当前 action。

下一步虽然重新规划，

但不改变这一事实。

---

# 47. 为什么 ACT 可以理解成“局部计划 + 高频重规划”？

这是非常好的高层直觉。

每个 forward：

\[
o_t
\rightarrow
\hat A_t
\]

可以看成：

> 当前 observation 下的一个短期 local action plan。

每 timestep 再：

> 重新生成新 local plan。

Temporal Ensemble 不直接选择一个 plan 完全覆盖旧 plan，

而是把重叠部分做融合。

所以从行为上：

```text
local plan
↓
execute a little
↓
observe again
↓
re-plan
```

但必须注意：

> ACT 不是显式 dynamics-based planner。

这些 future actions 是 neural policy 直接预测的。

---

# 48. 一个完整 k=4 Timeline

现在把 training 与 rollout 放进一张图。

```text
══════════════════════════════════════════════
TRAINING
══════════════════════════════════════════════

demonstration:

o0 → a0
o1 → a1
o2 → a2
o3 → a3
o4 → a4
o5 → a5
...

randomly choose t=2

o2
+
A2 = [a2,a3,a4,a5]

         A2 + q2
            │
            ▼
       CVAE Encoder
            │
            ▼
         μ, logσ²
            │
            ▼
       z = μ + σε
            │
            │
images2 + q2 + z
            │
            ▼
        ACT Policy
            │
            ▼
Â2 = [â2,â3,â4,â5]
            │
       ┌────┴─────┐
       ▼          ▼
      L1          KL
       │          │
       └────┬─────┘
            ▼
         backward
            ▼
         AdamW


══════════════════════════════════════════════
INFERENCE
══════════════════════════════════════════════

t=0:
o0 + z=0
→ [â0⁰, â1⁰, â2⁰, â3⁰]
→ execute â0⁰


t=1:
o1 + z=0
→ [â1¹, â2¹, â3¹, â4¹]

candidates for t=1:
[â1⁰, â1¹]

weighted average
→ execute a1


t=2:
o2 + z=0
→ [â2², â3², â4², â5²]

candidates for t=2:
[â2⁰, â2¹, â2²]

weighted average
→ execute a2


t=3:
o3 + z=0
→ [â3³, â4³, â5³, â6³]

candidates for t=3:
[â3⁰, â3¹, â3², â3³]

weighted average
→ execute a3
```

这就是整个 ACT。

---

# 49. 如果把所有复杂名词都拿掉，ACT 实际上在做什么？

训练：

> 看人类在当前画面下，接下来一小段时间怎么动；学习一次预测这一整小段动作。

但 human demonstrations 不完全一致，

所以训练时额外允许一个受约束 latent \(z\) 去解释 variation。

---

部署：

> 每个 timestep 都重新看一眼当前环境，再预测一次接下来一小段动作。

因为不同预测 window 重叠，

所以同一个当前 action 有多个估计。

ACT 把它们做 weighted ensemble，

然后只执行当前一步。

接着再看环境。

---

# 50. 为什么这套设计适合 Fine Manipulation？

论文的问题背景是：

> millimeter-level error 都可能导致失败。

这意味着同时需要：

### 长一点的行为结构

不能只把每一步当作完全独立动作。

所以：

> Action Chunking。

---

### 高频反馈

机器人实际执行总会有偏差。

所以：

> every-step re-query。

---

### 平滑切换

不能每一步完全突然抛弃之前的规划。

所以：

> Temporal Ensemble。

---

### Human Demonstration Variability

人类数据不是完全 deterministic。

所以：

> CVAE training。

---

### 高维视觉

需要直接从多视角 pixels 做闭环控制。

所以：

> ResNet + Transformer。

---

# 51. 一张“问题 → 设计”总表

| 问题 | ACT 的设计 |
|---|---|
| Single-step BC 容易累积误差 | Action Chunking |
| Human behavior 有 temporal structure | Action-sequence prediction |
| Naive chunk execution 缺少高频反馈 | Every-step re-query |
| Overlapping chunks 怎样决定当前动作 | Temporal Ensemble |
| Human demonstrations 有 variability | CVAE |
| Training posterior 与 test-time latent 要接得上 | KL to \(N(0,I)\) |
| Test-time 不想随机选择 style | \(z=0\) prior mean |
| 多路视觉 + proprioception + latent 要融合 | Transformer Encoder |
| 要一次产生未来 \(k\) 个动作 | Transformer Decoder + \(k\) queries |
| 高分辨率图片 token 太多 | ResNet18 visual backbone |

这张表比背 architecture 名字更重要。

---

# 52. 哪些东西属于“训练辅助机制”，哪些属于“部署机制”？

## Training Only

- ground-truth future action chunk；
- CVAE encoder；
- \(\mu,\sigma^2\)；
- stochastic posterior sampling；
- KL；
- L1 loss；
- backprop；
- optimizer。

---

## Training + Inference

- ResNet18；
- qpos projection；
- latent projection；
- policy Transformer encoder；
- policy Transformer decoder；
- action head。

---

## Inference Only

- fixed \(z=0\) choice；
- repeated every-step query；
- action prediction buffer；
- Temporal Ensemble；
- de-normalization；
- robot execution。

这个分类非常值得记住。

---

# 53. 一个常见大误解：ACT 是“CVAE 先生成动作，再 Transformer 修正”

**错误。**

真正关系是：

> **Transformer policy 本身就是 CVAE decoder。**

不是：

```text
CVAE
↓
先生成 action

然后 Transformer
↓
再改 action
```

而是：

```text
CVAE Encoder
→ training latent z

ACT Transformer Policy
= CVAE Decoder
→ action chunk
```

---

# 54. 另一个大误解：Temporal Ensemble 在 Transformer 内部

**错误。**

Transformer 一次 forward 只输出：

\[
\hat A_t
\]

Temporal Ensemble 是：

> 多次不同 timestep forward 之后的外部 aggregation。

所以执行顺序：

\[
\boxed{
Transformer
\rightarrow
Action Chunks
\rightarrow
Temporal Ensemble
}
\]

不是：

\[
Temporal Ensemble
\rightarrow
Transformer
\]

---

# 55. 另一个大误解：z 决定动作，Observation 只是辅助

不是。

policy：

\[
\pi(a\mid o,z)
\]

中：

\[
o
\]

包含非常丰富的：

- 4 路视觉；
- 当前 joint state。

尤其推理时：

\[
z=0
\]

固定，

真正随环境变化、驱动 closed-loop behavior 的核心输入就是：

\[
o_t
\]

所以更正确的理解是：

> \(z\) 是额外 latent conditioning，而 observation 是实时控制的主要信息来源。

---

# 56. 另一个大误解：Training 里有 Temporal Ensemble

没有。

Training sample：

\[
(o_t,A_t)
\]

一次 forward 只产生：

\[
\hat A_t
\]

然后直接和：

\[
A_t
\]

算 loss。

不会收集：

\[
t-1,t-2,\ldots
\]

的其他预测。

---

# 57. 另一个大误解：Inference z=0 意味着模型退化成普通 BC

也不能这样说。

虽然 inference latent 固定，

但最终 policy parameters：

\[
\theta
\]

是在：

> CVAE training objective

下学习出来的。

训练过程和普通 deterministic BC 不同。

所以：

\[
\pi_\theta(o,0)
\]

不是简单等同于：

> “从头训练一个没有 CVAE 的 BC model”。

论文 CVAE ablation 也说明这两者的学习结果不同。

---

# 58. 一个非常值得记住的“信息流”视角

Training 时 information flow：

```text
                   action chunk
                    /        \
                   /          \
                  ▼            ▼
          posterior z        target
                 │             │
                 ▼             │
observation → policy → prediction
                 │             │
                 └──── loss ───┘
```

---

Inference 时：

```text
observation
     │
     ▼
 policy with z=0
     │
     ▼
 future chunk
     │
     ▼
temporal buffer
     │
     ▼
 current action
     │
     ▼
   robot
     │
     ▼
new observation
```

这两张图几乎概括了 ACT 的全部核心。

---

# 59. 从概率模型角度再压缩一次

Training：

\[
q_\phi(z\mid A_t,q_t)
\]

近似 latent posterior。

然后：

\[
\pi_\theta(A_t\mid o_t,z)
\]

重建 expert chunk。

通过：

\[
KL(
q_\phi
\parallel
N(0,I)
)
\]

regularize posterior。

---

Inference：

\[
z^\star=0
\]

然后：

\[
\hat A_t
=
\pi_\theta(o_t,z^\star)
\]

所以最终 deployed model 只需要：

> policy / CVAE decoder。

---

# 60. 从控制角度再压缩一次

每一步：

\[
o_t
\]

产生一个短期 future plan：

\[
\hat A_t
\]

但只执行当前：

\[
a_t
\]

执行后再读取：

\[
o_{t+1}
\]

所以：

\[
\boxed{
\text{observe}
\rightarrow
\text{predict chunk}
\rightarrow
\text{execute one}
\rightarrow
\text{observe again}
}
\]

Temporal Ensemble 负责：

> 当多个过去的 local plans 都对当前 action 有意见时，怎样把它们融合。

---

# 61. 从 Tensor 角度再压缩一次

Training sample：

\[
images:
[4,3,480,640]
\]

\[
q:
[14]
\]

\[
A:
[k,14]
\]

CVAE posterior：

\[
\mu,\logvar:
[32]
\]

\[
z:
[32]
\]

policy visual sequence：

\[
[1200,512]
\]

加 q/z：

\[
[1202,512]
\]

decoder：

\[
[k,512]
\]

output：

\[
\boxed{
[k,14]
}
\]

---

# 62. 从时间轴角度再压缩一次

对于：

\[
k=4
\]

```text
query 0 → predicts 0 1 2 3
query 1 → predicts   1 2 3 4
query 2 → predicts     2 3 4 5
query 3 → predicts       3 4 5 6
```

execution time 3：

\[
\boxed{
4\text{ candidate predictions}
}
\]

然后：

\[
\boxed{
a_3
=
weighted\ average
}
\]

---

# 63. 如果你能回答下面 10 个问题，就已经真正理解 ACT 主流程

1. 为什么 ACT 不只预测 \(a_t\)，而预测 \(a_{t:t+k}\)？
2. 为什么 training encoder 能看 ground-truth action sequence？
3. `[CLS]` 最终为什么能用来预测 \(\mu,\sigma^2\)？
4. \(z\) 和 \(\epsilon\) 有什么区别？
5. KL 为什么要把 posterior 拉向 \(N(0,I)\)？
6. 为什么推理时 CVAE encoder 被丢弃？
7. 为什么 \(z=0\) 不等于“模型没有 latent input”？
8. 为什么每个 timestep 都重新 query，Action Chunking 仍然存在？
9. Temporal Ensemble 为什么平均的是同一 execution timestep 的 predictions？
10. 为什么 policy 一次输出 \(k\) 个动作，但机器人当前只执行一个？

如果这些问题都能自己解释，

ACT 的主干就已经真正打通了。

---

# 64. 最终一张 ACT 全景图

```text
════════════════════════════════════════════════════
                DATA COLLECTION
════════════════════════════════════════════════════

Human Teleoperation
        │
        ▼
Images + Follower qpos + Expert actions
        │
        ▼
Demonstration Dataset


════════════════════════════════════════════════════
                    TRAINING
════════════════════════════════════════════════════

random timestep t
        │
        ├─────────────── observation o_t
        │
        └─────────────── expert chunk A_t
                               │
                               ▼
                       CVAE Encoder
                    [CLS] + q_t + A_t
                               │
                               ▼
                           μ, logσ²
                               │
                               ▼
                         z = μ + σε
                               │
                               │
Images + q_t ──────────────────┤
                               ▼
                         ACT Policy
             ResNet + Transformer Enc/Dec
                               │
                               ▼
                        predicted Â_t
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
                  L1                    KL
                    │                     │
                    └──────────┬──────────┘
                               ▼
                         optimize model


════════════════════════════════════════════════════
                    INFERENCE
════════════════════════════════════════════════════

current observation o_t
        │
        ▼
normalize
        │
        ├──── z = 0
        │
        ▼
ACT Policy
        │
        ▼
future chunk Â_t
        │
        ▼
store into temporal buffers
        │
        ▼
collect all predictions for execution time t
        │
        ▼
Temporal Ensemble
        │
        ▼
normalized current action
        │
        ▼
de-normalize
        │
        ▼
14-D target joint positions
        │
        ▼
low-level PID
        │
        ▼
robot moves
        │
        ▼
new observation o_{t+1}
        │
        └────────────── repeat
```

---

# 65. 一句话真正概括 ACT

> **ACT 是一个从当前多视角视觉和机器人状态出发、一次预测未来一段 target joint actions 的 imitation-learning policy；训练时用 CVAE latent variable 帮助建模 human demonstration variation，并用 KL 将 latent posterior 约束到标准高斯 prior；部署时固定 \(z=0\)，每个 timestep 根据最新 observation 重新预测一个 action chunk，再对多个 overlapping chunks 对当前 timestep 的预测做 Temporal Ensemble，只执行融合后的当前 action，从而同时保留 action-sequence modeling 与高频 closed-loop feedback。**

如果这句话里的每一个部分你都能展开解释，

那么 ACT 的整体理论已经基本掌握。

---

# 66. 下一步应该学什么？

ACT 主流程到这里已经闭环。

接下来不建议继续把 ACT 本身无限拆碎。

更好的方向是补齐它依赖的几个 canonical concepts。

优先顺序可以是：

1. [Transformer](../../deep-learning/transformer.md)
2. [Attention](../../deep-learning/attention.md)
3. [Query / Key / Value](../../deep-learning/qkv.md)
4. [KL Divergence](../../mathematics/kl-divergence.md)
5. [Normal Distribution](../../mathematics/normal-distribution.md)
6. Standard Normal Distribution
7. [Imitation Learning](../imitation-learning.md)
8. [Behavior Cloning](../imitation-learning/behavior-cloning-distribution-shift.md)

然后再进入：

> **Minimal ACT Implementation**

把这一整套理论映射到真正 PyTorch 代码。

---

## Primary Source

Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.  
**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.**  
Robotics: Science and Systems (RSS), 2023.

- Paper: https://arxiv.org/abs/2304.13705
- RSS Proceedings: https://roboticsproceedings.org/rss19/p016.html
- Project: https://tonyzhaozh.github.io/aloha/

本文主要依据：

- Section IV — Action Chunking with Transformers
- Section IV-A — Action Chunking and Temporal Ensemble
- Algorithm 1 — ACT Training
- Algorithm 2 — ACT Inference
- Section IV-B — Modeling Human Data
- Section IV-C — Implementing ACT
- Figure 3 — ACT architecture
- Figure 4 — Action Chunking + Temporal Ensemble
- Appendix C — Detailed Architecture Diagram

论文核心流程包括：

\[
(o_t,a_{t:t+k})
\sim
\mathcal D
\]

\[
z
\sim
q_\phi(
z\mid
a_{t:t+k},\bar o_t
)
\]

\[
\hat a_{t:t+k}
\sim
\pi_\theta(
\cdot
\mid
o_t,z
)
\]

训练目标包含 reconstruction 与：

\[
D_{KL}
(
q_\phi
\parallel
N(0,I)
)
\]

而 inference 时：

\[
z=0
\]

并在每个 timestep 对 overlapping action chunks 使用：

\[
w_i=e^{-mi}
\]

进行 Temporal Ensemble。

---

## Official Implementation

ACT official repository:

https://github.com/tonyzhaozh/act

对应核心文件：

```text
imitate_episodes.py
utils.py
policy.py
detr/models/detr_vae.py
detr/models/transformer.py
```

本文中的 training / inference 总流程与 released implementation 保持一致，包括：

- demonstration timestep sampling；
- action chunk target；
- CVAE posterior；
- reparameterization；
- ACT Transformer policy；
- normalized action prediction；
- inference \(z=0\)；
- every-step policy query under temporal aggregation；
- `all_time_actions` overlapping prediction storage；
- exponential Temporal Ensemble；
- action de-normalization；
- target qpos execution。

---

## 本文知识连接

### ACT 专题

- [ACT 到底解决了什么问题？](./act-what-problem-does-it-solve.md)
- [Action Chunking](./action-chunking.md)
- [Temporal Ensemble](./temporal-ensemble.md)
- [CVAE in ACT](./cvae-in-act.md)
- [为什么 ACT 推理时令 z = 0？](./why-z-zero-at-inference.md)
- [ACT Architecture](./architecture.md)
- [Vision Pipeline](./vision-pipeline.md)
- [ACT Training](./training.md)
- [ACT Inference](./inference.md)

### Generative Models

- [Latent Variable](../../generative-models/latent-variable.md)
- [VAE](../../generative-models/vae.md)
- [Reparameterization Trick](../../generative-models/reparameterization-trick.md)
- [CVAE](../../generative-models/cvae.md)

### Deep Learning

- [Transformer](../../deep-learning/transformer.md)
- [Attention](../../deep-learning/attention.md)
- [Query / Key / Value](../../deep-learning/qkv.md)
- [Cross-Attention](../../deep-learning/cross-attention.md)
- [Transformer Encoder](../../deep-learning/transformer-encoder.md)
- [Transformer Decoder](../../deep-learning/transformer-decoder.md)
- ResNet

### Mathematics

- [Normal Distribution](../../mathematics/normal-distribution.md)
- Standard Normal Distribution
- [KL Divergence](../../mathematics/kl-divergence.md)
- Weighted Average

### Robot Learning

- [Imitation Learning](../imitation-learning.md)
- [Behavior Cloning](../imitation-learning/behavior-cloning-distribution-shift.md)
- Closed-Loop Control

### 后续

- Minimal ACT Implementation
