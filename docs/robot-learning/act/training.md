---
title: "ACT Training：一个 Demonstration Sample 到底怎样变成一次参数更新？"
description: "从 episode 数据、随机 timestep 采样、action chunk、padding、normalization、CVAE posterior、L1 reconstruction、KL regularization 到 AdamW 更新，完整拆解 ACT 的训练流程。"
status: reviewed
pageType: application
canonical: /robot-learning/act/training
updated: "2026-09-15"
---

# ACT Training：一个 Demonstration Sample 到底怎样变成一次参数更新？

前面我们已经知道 ACT 的网络长什么样：

\[
\text{Images} + q_t + z
\longrightarrow
\hat a_{t:t+k}
\]

也知道训练时 latent \(z\) 不是固定为 0，而是由 CVAE encoder 从 demonstration 中推断：

\[
q_\phi(z\mid a_{t:t+k},q_t)
\]

但“知道网络结构”还不等于“知道模型怎么训练”。真正运行训练代码时，发生的是一条更长的 pipeline：

```text
Demonstration Episodes
↓
随机选择 Episode
↓
随机选择 start timestep
↓
读取当前 Images + qpos
↓
截取未来 Action Sequence
↓
Padding + Mask
↓
Normalization
↓
CVAE Encoder 推断 μ, logσ²
↓
Reparameterization 得到 z
↓
ACT Policy 预测 Action Chunk
↓
L1 Reconstruction
+
β × KL
↓
Backward
↓
AdamW
↓
更新 Encoder + Policy
```

这一篇只回答：

> **ACT 官方实现中，一个 demonstration sample 到底怎样一步一步变成一次 gradient update？**

---

# 1. ACT 训练本质上仍然是监督学习

最基本的 Behavior Cloning 数据形式是：

\[
(o_t,a_t)
\]

ACT 只是把 target 从单步：

\[
a_t
\]

改成一段未来动作：

\[
a_{t:t+k}
\]

所以训练 sample 的核心仍然是：

\[
\boxed{(o_t,a_{t:t+k})}
\]

论文 Algorithm 1 直接写的是：从 demonstration dataset \(\mathcal D\) 中采样当前 observation 与对应 action chunk。

因此 ACT 从最外层看仍然是：

> **supervised imitation learning**

只是监督目标变成 sequence，并加入了 CVAE latent structure。

---

# 2. 一条 Demonstration Episode 中有什么？

一条长度为 \(T\) 的 trajectory 可以抽象写成：

\[
\mathcal E
=
\{o_0,a_0,o_1,a_1,\ldots,o_{T-1},a_{T-1}\}
\]

其中：

\[
o_t=(I_t^{(1)},I_t^{(2)},I_t^{(3)},I_t^{(4)},q_t)
\]

包含：

- 多路 RGB camera images；
- 当前 robot joint positions。

而动作：

\[
a_t\in\mathbb R^{14}
\]

表示双臂 target joint positions。

官方 HDF5 数据中主要对应：

```text
/observations/qpos
/observations/qvel
/observations/images/<camera_name>
/action
```

ACT policy 训练实际主要使用 images、qpos 和 action。

---

# 3. 官方实现不是提前把所有 Action Chunks 保存成独立样本

一个很容易产生的想象是：

```text
Episode
↓
预先切成：
(o0, a0:a100)
(o1, a1:a101)
(o2, a2:a102)
...
↓
保存成很多 training samples
```

released code 并不是这样做的。

`EpisodicDataset` 的一个 item 首先对应某个 episode id。每次调用：

```python
__getitem__(index)
```

代码会进入对应 episode，然后随机选择：

```python
start_ts = np.random.choice(episode_len)
```

即：

\[
\boxed{t\sim\text{Uniform discrete episode timesteps}}
\]

因此同一个 episode 在不同 epoch 中可以产生完全不同的 training timestep。

---

# 4. 一个 Dataset Item 实际上是在 Episode 中随机抽一个时刻

假设：

\[
T=400
\]

这次随机抽到：

\[
t=137
\]

dataset 会读取当前 observation：

\[
o_{137}
\]

即 timestep 137 的：

- camera images；
- qpos。

然后读取从这个时刻开始的未来 action sequence。

对于 simulation：

\[
a_{137},a_{138},\ldots,a_{399}
\]

后面再 padding。到了 policy 中才截到 chunk size：

\[
k
\]

如果：

\[
k=100
\]

最终 supervision 大致就是：

\[
a_{137:237}
\]

这里为了清楚使用半开区间记号。

---

# 5. 为什么随机采 start timestep？

因为我们希望 policy 学会：

> 在 episode 的任何阶段看到当前 observation 后，都能够预测接下来一段动作。

如果永远只训练：

\[
t=0
\]

模型就会过度集中于任务开头。

随机 timestep 相当于不断从长 trajectory 中抽不同局部监督：

```text
episode
────────────────────────────

      ↑
      t
      └──────── future chunk ────────

下一次：
                 ↑
                 t'
                 └──── future chunk ──
```

长期来看，同一 demonstration episode 可以贡献许多不同的 action-chunk training examples。

---

# 6. Simulation 和 Real Robot Data 有一个 Action Alignment 差异

官方 `utils.py` 中，对 simulation：

```python
action = root['/action'][start_ts:]
```

即：

\[
a_{t:}
\]

但对真实机器人数据：

```python
action = root['/action'][max(0, start_ts - 1):]
```

旁边的原注释直接写：

```python
# hack, to make timesteps more aligned
```

也就是说 released implementation 对 real data 做了一个：

> **向前偏移一个 action timestep 的 alignment hack**。

这是原实现的数据对齐细节，不是 ACT 数学定义本身。因此理论上仍然写：

\[
(o_t,a_{t:t+k})
\]

即可；但如果严格复现 original code，就必须注意这个行为。

---

# 7. Episode 尾部不够 k 步怎么办？

假设：

\[
T=400,
\qquad
k=100
\]

但随机采到：

\[
t=370
\]

从这里到 episode 结束只剩：

\[
400-370=30
\]

个 actions。

可是 ACT 始终需要一个固定长度 action tensor。

解决方式是：

> **Padding**

---

# 8. Padding 如何做？

官方代码先创建全零 action array：

```python
padded_action = np.zeros(original_action_shape)
```

然后把真实 action 放到前面：

```python
padded_action[:action_len] = action
```

概念上：

```text
真实：
a370
a371
...
a399

Padding 后：
a370
a371
...
a399
0
0
0
...
```

随后 policy 再截取前：

\[
k=100
\]

个位置，因此 shape 始终可以固定成：

\[
[100,14]
\]

---

# 9. Padding 的 0 不能当成真实 Ground Truth

如果直接让模型对 padding 位置计算 reconstruction loss，那么模型会被错误监督成：

> episode 结束以后，joint target 应该变成全零。

所以 dataset 同时生成：

\[
\boxed{is\_pad}
\]

例如：

```text
False False False ... False True True True ...
```

其中：

\[
is\_pad_i=
\begin{cases}
0,&\text{真实 action}\
1,&\text{padding}
\end{cases}
\]

---

# 10. Padding Mask 有两个不同用途

## 第一处：CVAE Encoder

训练 latent encoder 时，padding action tokens 不应该参加 self-attention。

所以 `is_pad` 会作为：

> `src_key_padding_mask`

传入 CVAE Transformer encoder。

## 第二处：Reconstruction Loss

padding positions 不能产生监督。

因此 L1 loss 会乘：

\[
\neg is\_pad
\]

所以两者要区分：

```text
Attention Mask
→ 防止 padding 污染 z inference

Loss Mask
→ 防止 padding 产生假监督
```

---

# 11. 为什么 [CLS] 和 Joint Token 永远不 Mask？

CVAE encoder 的输入是：

```text
[CLS]
joint
action 0
action 1
...
```

只有 action sequence 可能在 episode 尾部不足。

所以官方代码创建：

```python
cls_joint_is_pad = torch.full((bs, 2), False)
```

再与 action mask 拼接。

因此：

```text
[CLS]   → valid
joint   → valid
action  → valid / padding
```

---

# 12. Dataset 最终返回什么？

官方 `EpisodicDataset` 返回：

```python
image_data,
qpos_data,
action_data,
is_pad
```

对于单个 sample，可理解为：

### Images

\[
[N_{cam},3,H,W]
\]

原始 ALOHA 常见：

\[
[4,3,480,640]
\]

### Qpos

\[
[14]
\]

### Action Data

在 dataset 层先 pad，policy 层再截到：

\[
[k,14]
\]

### Padding Mask

\[
[k]
\]

一个 batch 后：

\[
[B,4,3,H,W]
\]

\[
[B,14]
\]

\[
[B,k,14]
\]

\[
[B,k]
\]

---

# 13. 训练前为什么要 Normalize？

不同 joint dimensions 的数值范围和变化尺度可能不同。

如果一维典型变化只有：

\[
0.02
\]

另一维经常变化：

\[
2.0
\]

那么直接用原始数值训练时，大尺度维度可能更强地影响 loss 与 gradient。

所以 original code 对 qpos 和 action 做 dataset-level standardization。

---

# 14. Qpos 和 Action 怎样 Standardize？

官方统计所有数据中每一维的：

\[
\mu_a,\sigma_a
\]

然后：

\[
\boxed{
a_{norm}
=
\frac{a-\mu_a}{\sigma_a}
}
\]

qpos 同样：

\[
\boxed{
q_{norm}
=
\frac{q-\mu_q}{\sigma_q}
}
\]

官方还把 std clip 到至少：

\[
10^{-2}
\]

避免某个几乎不变化的 dimension 出现：

\[
\sigma\approx0
\]

导致数值不稳定。

---

# 15. Qpos 和 Action 分别统计 Mean / Std

虽然 qpos 和 action 都是 14-D joint-related vectors，代码仍分别保存：

```text
qpos_mean
qpos_std
action_mean
action_std
```

推理时：

- 当前 observation qpos 使用 qpos stats normalize；
- policy output 使用 action stats 反归一化。

所以：

> 输入标准化和输出反标准化使用的是两套统计量。

---

# 16. Image 也有两层预处理

dataset 先把：

\[
0\ldots255
\]

除以 255：

\[
[0,1]
\]

然后 `ACTPolicy` 又执行 ImageNet-style normalization：

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

所以视觉输入 pipeline 是：

\[
uint8
\rightarrow
[0,1]
\rightarrow
\text{channel-wise normalized tensor}
\]

再进入 ResNet18。

---

# 17. Train / Validation 怎样划分？

released code 在 episode level 随机打乱所有 episode id，然后：

\[
80\%
\]

作为 train，

\[
20\%
\]

作为 validation。

这是 episode-level split，而不是把同一 episode 的不同 timesteps 随机拆到两边。

这很重要，因为相邻 robot observations 极其相似。如果按 timestep 随机拆分，很容易造成严重的数据相关性泄漏。

---

# 18. 一个值得知道的实现细节：Normalization Stats 使用全部 Episodes

当前 released implementation 在构造 train / val dataset 前就调用：

```python
get_norm_stats(dataset_dir, num_episodes)
```

因此 normalization stats 来自：

> 全部 episodes，而不只是 train split。

严格的实验 pipeline 也可以选择：

> 仅使用 training split 统计 mean/std。

但那不是 original released code 当前的行为。

这是实现细节，不是 ACT 算法核心。

---

# 19. 一个 Batch 怎样送进 Policy？

训练 loop 中：

```python
image_data, qpos_data, action_data, is_pad = data
```

移动到 GPU 后调用：

```python
policy(
    qpos_data,
    image_data,
    action_data,
    is_pad
)
```

因为：

```python
actions is not None
```

`ACTPolicy` 知道这次 forward 拥有 training target，因此会进入：

> **CVAE training branch**

而不是 inference branch。

---

# 20. Policy 首先截到 Chunk Size

Dataset 返回的 padded action sequence 可能比 chunk size 长。

`ACTPolicy` 会：

```python
actions = actions[:, :self.model.num_queries]
is_pad = is_pad[:, :self.model.num_queries]
```

其中：

\[
num\_queries=k
\]

于是最终 target：

\[
\boxed{
A_t\in\mathbb R^{B\times k\times14}
}
\]

mask：

\[
\boxed{
M\in\mathbb R^{B\times k}
}
\]

---

# 21. 进入 CVAE Encoder

训练 branch 中：

```text
qpos
+
ground-truth future actions
↓
CVAE Encoder
↓
μ, logσ²
↓
z
```

图像并不进入这个 latent encoder。

详细结构参见：

- [CVAE in ACT](./cvae-in-act.md)

---

# 22. CVAE Encoder 的 Shape

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

qpos：

\[
[8,14]
\]

投影成：

\[
[8,512]
\]

actions：

\[
[8,100,14]
\]

投影成：

\[
[8,100,512]
\]

再加入 `[CLS]`：

\[
[8,1,512]
\]

总 sequence：

\[
\boxed{[8,102,512]}
\]

因为：

\[
k+2=102
\]

---

# 23. Encoder 得到 μ 和 logσ²

ACT 只取 Transformer encoder 的 `[CLS]` hidden state：

\[
h_{CLS}\in\mathbb R^{B\times512}
\]

再通过 Linear：

\[
512\rightarrow64
\]

官方 latent dimension：

\[
d_z=32
\]

因此拆成：

\[
\mu\in\mathbb R^{B\times32}
\]

和：

\[
\log\sigma^2\in\mathbb R^{B\times32}
\]

定义 approximate posterior：

\[
q_\phi(z\mid A_t,q_t)
\]

---

# 24. Reparameterization 得到 z

先：

\[
\sigma
=
\exp\left(\frac12\log\sigma^2\right)
\]

采：

\[
\epsilon\sim\mathcal N(0,I)
\]

再：

\[
\boxed{
z
=
\mu+\sigma\odot\epsilon
}
\]

shape：

\[
[B,32]
\]

这一步让 reconstruction gradient 能反向传播到 encoder。

详细见：

- [Reparameterization Trick](../../generative-models/reparameterization-trick.md)

---

# 25. z 与 Observation 一起进入 ACT Policy

现在 policy / CVAE decoder 接收：

```text
normalized images
+
normalized qpos
+
sampled z
```

经过：

```text
ResNet18
↓
Transformer Encoder
↓
Transformer Decoder
↓
Action Head
```

得到：

\[
\boxed{
\hat A_t
\in
\mathbb R^{B\times k\times14}
}
\]

目标是重建 demonstration 中真实：

\[
A_t
\]

---

# 26. 第一部分 Loss：Reconstruction

从 CVAE 概率角度，reconstruction term 对应：

\[
-\log p_\theta(A_t\mid o_t,z)
\]

ACT 的实际 released implementation 使用：

> **L1 loss**

官方代码：

```python
all_l1 = F.l1_loss(
    actions,
    a_hat,
    reduction='none'
)
```

得到：

\[
[B,k,14]
\]

逐元素 absolute error。

---

# 27. 为什么不能直接对 all_l1 Mean？

因为 chunk 尾部可能包含 padding。

官方：

```python
all_l1 * ~is_pad.unsqueeze(-1)
```

将 padding positions 的 loss 变为 0。

因此概念上：

\[
\boxed{
L_{recon}
=
\operatorname{mean}
\left(
|\hat A-A|\cdot M
\right)
}
\]

其中 mask：

\[
M=1
\]

代表真实 action，

\[
M=0
\]

代表 padding。

---

# 28. 一个很细但真实的 Reduction Detail

official code 写的是：

```python
l1 = (
    all_l1
    * ~is_pad.unsqueeze(-1)
).mean()
```

也就是：

> 先把 padding error 乘成 0，再对整个固定 shape tensor 求 mean。

所以 denominator 仍包含 padded positions。

这和严格只对 valid elements 求：

\[
\frac{\sum |e|M}{\sum M}
\]

并不完全相同。

因此如果某个 sample 有很多 padding，它对 reconstruction loss 的平均贡献会相对更小。

这是 original released implementation 的具体行为。

---

# 29. Algorithm 1 写 MSE，为什么这里是 L1？

ACT 论文 Algorithm 1 写：

\[
L_{reconst}
=
MSE(\hat A,A)
\]

但正文 Section IV-C 明确说明：

> 实际实现使用 L1，而不是更常见的 L2，因为作者观察到 L1 对 action sequence 建模更精确。

官方 `policy.py` 也明确是：

```python
F.l1_loss(...)
```

所以如果问题是：

> **original ACT actual implementation 用什么 reconstruction loss？**

应回答：

\[
\boxed{L1}
\]

而不是机械照抄 Algorithm 1 的 MSE。

---

# 30. 第二部分 Loss：KL Divergence

posterior：

\[
q_\phi(z\mid A_t,q_t)
=
\mathcal N(
\mu,
\operatorname{diag}(\sigma^2)
)
\]

prior：

\[
p(z)=\mathcal N(0,I)
\]

ACT 计算：

\[
\boxed{
L_{KL}
=
D_{KL}
\left(
q_\phi(z\mid A_t,q_t)
\parallel
\mathcal N(0,I)
\right)
}
\]

对于 diagonal Gaussian：

\[
\boxed{
D_{KL}
=
\frac12
\sum_{j=1}^{d_z}
\left(
\mu_j^2
+
\sigma_j^2
-
\log\sigma_j^2
-
1
\right)
}
\]

---

# 31. 官方 KL 代码怎样对应公式？

```python
klds = -0.5 * (
    1
    + logvar
    - mu.pow(2)
    - logvar.exp()
)
```

因为：

\[
\sigma^2=e^{\log\sigma^2}
\]

所以与上式完全等价。

随后：

```python
total_kld = klds.sum(1).mean(0, True)
```

即：

### 对 latent dimensions 求和

\[
\sum_{j=1}^{32}KLD_{b,j}
\]

### 再对 batch 求平均

\[
\boxed{
L_{KL}
=
\frac1B
\sum_{b=1}^{B}
\sum_{j=1}^{32}
KLD_{b,j}
}
\]

---

# 32. 最终 ACT Loss

官方：

```python
loss = l1 + kl_weight * kl
```

因此：

\[
\boxed{
L
=
L_{L1}
+
\beta L_{KL}
}
\]

原论文 Table III：

\[
\boxed{\beta=10}
\]

所以典型 original ACT：

\[
L=L_{L1}+10L_{KL}
\]

注意：

> 不能因为系数是 10，就说“KL 比 L1 重要 10 倍”。

两项原始数值尺度和 reduction 方式不同。

\(\beta\) 只是控制相对权重的 hyperparameter。

---

# 33. β 控制什么？

如果：

\[
\beta=0
\]

encoder 可以自由使用 \(z\) 去记 action sequence。

这可能带来很好的 reconstruction，

但 training posterior 可能与 test-time prior：

\[
\mathcal N(0,I)
\]

完全脱节。

如果 \(\beta\) 很大：

posterior 被强烈推向 prior，

\(z\) 能携带的 demonstration-specific information 会减少。

所以 \(\beta\) 控制：

```text
Action Reconstruction
        ↕
Latent Prior Regularization
```

之间的 trade-off。

论文明确指出：

> higher \(\beta\) means less information transmitted in \(z\).

---

# 34. Reconstruction Gradient 会更新哪些模块？

L1 loss 从：

\[
\hat A
\]

向后传播。

它首先更新：

- Action Head；
- Policy Transformer Decoder；
- Policy Transformer Encoder；
- ResNet Backbone。

同时因为：

\[
z=\mu+\sigma\epsilon
\]

是可微的，reconstruction gradient 还会继续回到：

- \(\mu\) / `logvar` projection；
- `[CLS]` representation；
- CVAE Transformer Encoder；
- action / joint embedding layers。

所以：

> **reconstruction loss 不只训练 decoder。**

它也参与训练 latent encoder。

---

# 35. KL Gradient 会更新哪些模块？

KL 直接依赖：

\[
\mu,\log\sigma^2
\]

因此主要沿：

```text
KL
↓
μ / logσ²
↓
latent projection
↓
h_CLS
↓
CVAE Transformer Encoder
↓
action / qpos embeddings
```

反向传播。

KL 不需要经过：

- ResNet；
- policy Transformer decoder；
- action head。

所以可以粗略理解：

```text
Reconstruction
→ 训练 Encoder + Policy

KL
→ 主要约束 CVAE Encoder posterior
```

---

# 36. 整个 Loss Graph

```text
                   ground-truth action chunk A
                            │
                  ┌─────────┴───────────┐
                  │                     │
                  ▼                     ▼
           CVAE Encoder           Reconstruction Target
                  │                     ▲
                  ▼                     │
             μ, logσ²                   │
                  │                     │
          ε ~ N(0,I)                    │
                  │                     │
                  ▼                     │
            z = μ + σε                  │
                  │                     │
          images + qpos + z             │
                  │                     │
                  ▼                     │
              ACT Policy                │
                  │                     │
                  ▼                     │
                  Â ───── masked L1 ────┘

q(z|A,q) ───────── KL ───────── N(0,I)
                  │
                  ▼
             L = L1 + β KL
```

---

# 37. `loss.backward()` 到底做什么？

官方 training loop：

```python
loss = forward_dict['loss']
loss.backward()
```

PyTorch 从 scalar loss 开始，对所有需要学习的 parameter 计算：

\[
\frac{\partial L}{\partial w}
\]

例如：

- CNN convolution weights；
- attention 的 \(W_Q,W_K,W_V\)；
- Transformer feed-forward layers；
- action query embeddings；
- `[CLS]` embedding；
- latent projection；
- action head。

但：

> `backward()` 只计算 gradient。

它还没有修改 weights。

---

# 38. 真正修改 Weight 的是 Optimizer

之后：

```python
optimizer.step()
```

才根据当前 parameter 与 gradient 进行更新。

最后：

```python
optimizer.zero_grad()
```

清空 gradient。

所以一次训练 iteration 的核心顺序是：

```text
forward
↓
loss
↓
backward
↓
optimizer.step
↓
zero_grad
```

---

# 39. 论文写 Adam，但官方代码实际使用 AdamW

论文 Algorithm 1 写：

> Update \(\theta,\phi\) with ADAM.

但 released `detr/main.py` 实际是：

```python
optimizer = torch.optim.AdamW(...)
```

因此：

### 讲算法高层

可以说：

> Adam-family gradient optimization。

### 精确复现 released code

应该说：

\[
\boxed{AdamW}
\]

这和前面 MSE / L1 一样，也是论文高层描述与 released implementation 的一个差异。

---

# 40. Optimizer 把 Backbone 单独分组

官方 parameter groups：

## Non-backbone Parameters

使用：

\[
lr
\]

## Backbone Parameters

使用：

\[
lr_{backbone}
\]

代码：

```python
{
  "params": non_backbone_params
},
{
  "params": backbone_params,
  "lr": args.lr_backbone
}
```

所以 ResNet 可以使用不同 learning rate fine-tune。

---

# 41. ResNet 是 Frozen 吗？

不是。

backbone parameters 被明确放进 optimizer：

```python
if "backbone" in n and p.requires_grad
```

所以 original released ACT 会训练 / fine-tune visual backbone。

它不是固定 feature extractor。

---

# 42. Weight Decay

released `detr/main.py` 默认：

\[
\boxed{weight\ decay=10^{-4}}
\]

并由 AdamW 使用。

论文 Table III 没有把 ACT weight decay 单独列出来，因此这是：

> released implementation configuration

而不是论文主表中强调的核心 hyperparameter。

---

# 43. 一个 Epoch 实际上看多少 Samples？

这是 original dataset loader 一个很特别的地方。

`EpisodicDataset.__len__()` 返回：

\[
\#episodes
\]

而不是：

\[
\#timesteps
\]

因此一个 epoch 中，每个 training episode 大致被访问一次。

但每次访问该 episode 时：

\[
start\_ts
\]

随机变化。

例如：

```text
Epoch 1:
episode 7 → t=42

Epoch 2:
episode 7 → t=186

Epoch 3:
episode 7 → t=15
```

所以需要通过很多 epoch 才逐渐覆盖同一 episode 中大量不同 timesteps。

---

# 44. 这也解释了为什么训练 Epoch 数很大

官方 README 的 simulation example 使用：

\[
2000
\]

epochs。

当前官方 tuning note 对 real-world data 甚至建议训练更久。

原因之一就是：

> 一个 epoch 不等于把所有 trajectory timesteps 全部看一遍。

每个 episode 只随机贡献一个 start point。

因此较大的 epoch 数与这种 dataset sampling strategy 是绑定在一起的。

---

# 45. Validation 怎样做？

每个 epoch：

```python
policy.eval()
with torch.inference_mode():
    ...
```

先遍历 validation dataloader。

代码计算 validation loss 的 batch average。

如果：

\[
L_{val}
\]

低于历史最好值，就保存：

```python
best_state_dict
```

训练结束后：

> 最低 validation loss 的 checkpoint 被保存为 `policy_best.ckpt`。

论文同样说明 test time 使用 lowest validation loss 的 policy。

---

# 46. Validation 时还会运行 CVAE Encoder 吗？

会。

因为 validation dataset 仍然有 ground-truth action sequence。

所以 validation forward 仍可以计算：

\[
q_\phi(z\mid A,q)
\]

并得到：

\[
L1+\beta KL
\]

这与真实 robot rollout inference 不一样。

---

# 47. Validation Loss 不等于 Robot Success Rate

validation loss 衡量：

> held-out demonstrations 上的 action reconstruction + latent regularization。

而 rollout success 衡量：

> policy 真正在闭环环境中能不能完成任务。

后者还受到：

- compounding error；
- observation distribution shift；
- perception error；
- control dynamics；
- Temporal Ensemble；
- contact uncertainty；

影响。

所以：

\[
\text{lower validation loss}
\]

不严格等价于：

\[
\text{higher task success rate}
\]

但 original ACT 用 validation loss 作为 checkpoint selection criterion。

---

# 48. Training 时有 Temporal Ensemble 吗？

**没有。**

Temporal Ensemble 是：

\[
\boxed{\text{Inference Only}}
\]

训练 sample 只做：

```text
一个 observation
↓
一个 predicted action chunk
↓
一个 ground-truth action chunk
↓
loss
```

不会把其他 timesteps 的 overlapping predictions 拉进同一次 training loss。

论文明确指出 Temporal Ensemble：

> 没有额外 training cost，只增加 inference-time computation。

---

# 49. Training 时会执行预测 Action 吗？

不会。

ACT training 是 offline behavior cloning：

```text
Dataset Observation
↓
Policy
↓
Prediction
↓
Loss
```

并不会：

```text
执行预测动作
↓
得到新 observation
↓
根据 reward 更新
```

所以 ACT 不是 reinforcement learning。

训练中没有：

- reward；
- critic；
- policy gradient；
- Q function；
- online exploration。

---

# 50. 一次 Training Forward 中有几个 Action Predictions？

一个当前 observation：

\[
o_t
\]

得到一个完整 predicted chunk：

\[
\hat A_t
=
(\hat a_t,\ldots,\hat a_{t+k-1})
\]

也就是说：

> 一次 forward 产生 1 个 chunk，chunk 内有 \(k\) 个 future action positions。

它不是 inference 时那种：

> “同一个 timestep 有多个不同历史 chunk 的预测”。

后者属于 Temporal Ensemble。

---

# 51. Training Target Chunks 会不会彼此重叠？

长期 dataset sampling 中当然会。

例如一次：

\[
t=10
\]

监督：

\[
a_{10:110}
\]

另一次：

\[
t=11
\]

监督：

\[
a_{11:111}
\]

两者大部分相同。

但这是：

> 两个不同 training examples 的 target windows overlap。

并不意味着训练时对它们做 Temporal Ensemble。

---

# 52. 一个完整 Batch Shape 例子

假设：

\[
B=8,
\quad
k=100,
\quad
d_z=32
\]

### Images

\[
[8,4,3,480,640]
\]

### Qpos

\[
[8,14]
\]

### Target Actions

\[
[8,100,14]
\]

### Padding Mask

\[
[8,100]
\]

### CVAE Sequence

\[
[8,102,512]
\]

### μ / logσ²

\[
[8,32]
\]

### z

\[
[8,32]
\]

### Policy Output

\[
[8,100,14]
\]

### L1

\[
[8,100,14]
\rightarrow
scalar
\]

### KL

\[
[8,32]
\rightarrow
scalar
\]

最后得到一个 scalar loss，进行一次 optimizer step。

---

# 53. 为什么原论文 Batch Size 是 8？

Table III：

\[
\boxed{batch\ size=8}
\]

意味着一次 optimizer update 同时处理 8 个 sampled episode-timestep examples。

它不是：

> 一次把 8 条完整 episode 的全部 timesteps 都输入网络。

每个 dataset item 只提供：

> 一个 current observation + 一段 future action target。

---

# 54. 为什么 Image 输入只有当前一帧？

ACT policy 建模：

\[
\pi(a_{t:t+k}\mid o_t)
\]

而不是显式：

\[
\pi(a_t\mid o_{t-h:t})
\]

所以 dataset 只取：

```python
image[start_ts]
```

也就是当前 timestep 的图片。

时间结构主要体现在：

> **输出 future action sequence**

而不是输入 image history。

---

# 55. 为什么 Action Target 是未来一整段？

因为 Action Chunking 的核心就是：

\[
\boxed{
\pi_\theta(a_{t:t+k}\mid o_t)
}
\]

所以监督关系是：

```text
一个当前 observation
↓
一段未来 expert behavior
```

这是一种：

> sequence output problem。

---

# 56. CVAE Encoder 看到 Future Actions 算不算作弊？

不算 deployed policy 的 information leakage。

因为 CVAE encoder 是：

> training-time approximate posterior network。

它的工作是用 ground-truth target 推断 latent variable。

真正 test-time 使用的是：

\[
\pi_\theta(A\mid o,z)
\]

而不是：

\[
q_\phi(z\mid A,q)
\]

推理时 encoder 被丢弃，future target 不会进入 deployed policy。

这是标准 VAE / CVAE training structure。

---

# 57. 如果删掉 CVAE，Training 会变成什么？

大致退化为：

```text
Images + qpos
↓
Transformer Policy
↓
action chunk
↓
L1
```

也就是：

\[
o_t
\rightarrow
A_t
\]

没有：

- posterior \(q(z|\cdot)\)；
- \(\mu\)；
- \(\sigma\)；
- reparameterization；
- KL。

论文 ablation 表明，对 human demonstration tasks，CVAE objective 非常重要。

---

# 58. 官方 Training Loop 可以压缩成什么？

```python
for epoch in range(num_epochs):

    # validation
    policy.eval()
    with torch.inference_mode():
        for batch in val_loader:
            val_loss = forward(batch)

    # remember best checkpoint

    # training
    policy.train()
    optimizer.zero_grad()

    for batch in train_loader:

        losses = forward(batch)
        loss = losses["loss"]

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

而 `forward(batch)` 内部才发生：

```text
normalization
↓
CVAE posterior
↓
reparameterization
↓
ACT policy
↓
L1 + β KL
```

---

# 59. 为什么 `optimizer.zero_grad()` 必须存在？

PyTorch 默认 gradients 会累加。

如果不清零：

\[
\nabla L_1
\]

会和下一 batch：

\[
\nabla L_2
\]

累积成：

\[
\nabla L_1+\nabla L_2
\]

除非你故意进行 gradient accumulation。

original ACT 不是这样做，因此每次 optimizer step 后清空 gradient。

---

# 60. Released Code 实际有没有 Gradient Clipping？

parser 中虽然存在：

```python
clip_max_norm = 0.1
```

但代码直接标注：

```python
# not used
```

training loop 也没有调用：

```python
clip_grad_norm_
```

所以不能因为 parser 里出现这个参数，就说：

> “original ACT training 使用 gradient clipping 0.1。”

当前 released training path 实际没有用。

---

# 61. 有没有 Learning Rate Scheduler？

parser 里也保留：

```python
lr_drop
```

但同样标记：

```python
# not used
```

original released training loop 没有 scheduler step。

因此这条代码路径使用：

> 固定 optimizer learning rate。

---

# 62. ACT 是 Joint End-to-End Training

并不是：

```text
先单独训练 CVAE Encoder
↓
冻结
↓
再训练 ACT Policy
```

而是一个 total loss：

\[
L=L1+\beta KL
\]

一次：

```python
loss.backward()
```

一个 optimizer 联合更新：

- CVAE encoder；
- ResNet；
- policy Transformer encoder；
- policy Transformer decoder；
- action head。

论文 Appendix 也明确指出 reparameterization 让 encoder 与 decoder 可以 jointly optimized。

---

# 63. “联合训练”真正意味着什么？

reconstruction 要求：

\[
\hat A\approx A
\]

为了做到这一点，decoder 希望获得一个有用的：

\[
z
\]

于是 reconstruction gradient 会推动 encoder 学会：

> 怎样形成一个有助于解释当前 demonstration 的 posterior。

但 KL 同时限制：

> 这个 posterior 不能随意偏离 prior。

所以 Encoder 与 Decoder 是一起寻找一个平衡解：

```text
有信息的 latent
+
可生成动作的 policy
+
与 prior 不脱节
```

---

# 64. 为什么 Loss Plateau 后 Policy 仍可能继续变好？

official repo 当前 tuning note 特别提醒：

> real-world ACT 的 success rate 和 smoothness 可能在 loss plateau 很久以后仍继续改善。

这说明：

> supervised average loss 与 closed-loop behavior quality 并不是线性对应。

对于高精度 manipulation：

- critical contact phase 的几毫米误差；
- action smoothness；
- rare but important states；

可能对 task success 影响极大，却只改变一点平均 L1。

因此不要只凭：

> “loss 已经不降了”

就断言 policy 已经训练充分。

---

# 65. 原论文 ACT Hyperparameters

Table III 给出的核心设置：

\[
\boxed{learning\ rate=10^{-5}}
\]

\[
\boxed{batch\ size=8}
\]

\[
\boxed{encoder\ layers=4}
\]

\[
\boxed{decoder\ layers=7}
\]

\[
\boxed{feedforward\ dim=3200}
\]

\[
\boxed{hidden\ dim=512}
\]

\[
\boxed{attention\ heads=8}
\]

\[
\boxed{chunk\ size=100}
\]

\[
\boxed{\beta=10}
\]

\[
\boxed{dropout=0.1}
\]

这些是理解 original ACT training scale 最重要的一组配置。

---

# 66. 一次 Parameter Update 的完整故事

现在把所有东西串起来。

## Step 1：DataLoader 取一批 Episodes

例如：

\[
B=8
\]

## Step 2：每条 Episode 随机一个 start timestep

例如：

```text
episode 4  → t=53
episode 8  → t=201
episode 15 → t=17
...
```

## Step 3：读取当前 Observation

```text
4 camera images
+
qpos
```

## Step 4：读取未来 Actions

从该 timestep 开始向后取。

## Step 5：Padding + Mask

不足长度的后面补 0，并标记 `is_pad`。

## Step 6：Normalization

\[
q\rightarrow(q-\mu_q)/\sigma_q
\]

\[
a\rightarrow(a-\mu_a)/\sigma_a
\]

image：

\[
0\ldots255
\rightarrow
0\ldots1
\rightarrow
ImageNet normalization
\]

## Step 7：截到前 k 个 Actions

\[
A\in[B,k,14]
\]

## Step 8：CVAE Encoder

\[
[CLS]+q+A
\rightarrow
\mu,\log\sigma^2
\]

## Step 9：Reparameterization

\[
z=\mu+\sigma\epsilon
\]

## Step 10：ACT Policy

\[
(images,q,z)
\rightarrow
\hat A
\]

## Step 11：Masked L1

\[
L_{L1}
\]

## Step 12：KL

\[
L_{KL}
=
KL(q_\phi(z|A,q)\|N(0,I))
\]

## Step 13：Combine

\[
L=L_{L1}+10L_{KL}
\]

## Step 14：Backprop

\[
\nabla_{\theta,\phi}L
\]

## Step 15：AdamW Update

修改整个模型参数。

这就是一次 ACT training iteration。

---

# 67. 完整 Training Diagram

```text
                 DEMONSTRATION EPISODE
                         │
                         ▼
               random start timestep t
                         │
          ┌──────────────┴───────────────┐
          │                              │
          ▼                              ▼
 current images + qpos          future actions
          │                              │
          │                              ▼
          │                         padding
          │                              │
          │                         is_pad mask
          │                              │
          ├──────── normalization ────────┤
          │                              │
          │                              ▼
          │                         action chunk A
          │                              │
          │                    ┌─────────┴─────────┐
          │                    │                   │
          │                    ▼                   │
          │              CVAE Encoder              │
          │                    │                   │
          │                    ▼                   │
          │                 μ, logσ²               │
          │                    │                   │
          │          ε ~ N(0,I)│                   │
          │                    ▼                   │
          │                    z                   │
          │                    │                   │
          └──────────────┬─────┘                   │
                         ▼                         │
                     ACT Policy                    │
                         │                         │
                         ▼                         │
                    predicted Â                    │
                         │                         │
                         ├──── masked L1 ──────────┘
                         │
                         └──── KL(q || N(0,I))
                                  │
                                  ▼
                         L = L1 + β KL
                                  │
                                  ▼
                              backward
                                  │
                                  ▼
                               AdamW
                                  │
                                  ▼
                       update θ and φ jointly
```

---

# 68. 常见误解一：训练时所有 Chunk 已经预先切好

**不符合 original released implementation。**

当前 official dataset 每次读取 episode 时随机选择：

\[
start\_ts
\]

action window 是动态产生的。

---

# 69. 常见误解二：一个 Epoch 会看到所有 Timesteps

**错误。**

dataset length 是 episode 数量。

每个 episode 每次只随机贡献一个 start timestep。

需要多 epochs 才逐步覆盖 episode 中不同位置。

---

# 70. 常见误解三：Padding 0 是真实 Action

**错误。**

`is_pad` 明确用于：

- attention mask；
- reconstruction mask。

Padding 只是 tensor 对齐。

---

# 71. 常见误解四：Dataset Normalization 和 LayerNorm 是一回事

不是。

Dataset standardization：

\[
(x-\mu)/\sigma
\]

属于数据预处理。

Transformer LayerNorm 是模型内部 normalization layer。

两者概念不同。

---

# 72. 常见误解五：ACT Training Loss 就是 MSE + KL

如果只读 Algorithm 1，会看到 MSE。

但实际 original implementation 使用：

\[
\boxed{L1+\beta KL}
\]

正文和 code 都支持这一点。

---

# 73. 常见误解六：KL 是 z 和 0 的距离

**错误。**

KL 比较的是：

\[
q_\phi(z|A,q)
\]

与：

\[
N(0,I)
\]

两个 probability distributions。

不是：

\[
\|z-0\|
\]

这样的 vector distance。

---

# 74. 常见误解七：β=10 表示 KL 比 L1 重要 10 倍

**错误。**

两项的 raw scale 与 reduction 不一样。

10 只是 relative weighting hyperparameter。

---

# 75. 常见误解八：Reconstruction Loss 只训练 Decoder

**错误。**

因为：

\[
z=\mu+\sigma\epsilon
\]

可微，L1 gradient 会一路传回 CVAE encoder。

---

# 76. 常见误解九：KL 直接训练 ResNet

通常没有直接路径。

KL 依赖 \(\mu,\log\sigma^2\)，而 ACT latent encoder 不输入 images。

ResNet 主要通过 action reconstruction gradient 被训练。

---

# 77. 常见误解十：Temporal Ensemble 参与 Training Loss

**错误。**

Temporal Ensemble 是 inference-only。

---

# 78. 常见误解十一：ACT 是 Reinforcement Learning

**错误。**

ACT 是 offline supervised imitation learning。

不需要 reward、critic 或 online exploration。

---

# 79. 常见误解十二：训练时 z 也是 0

**错误。**

training：

\[
z=\mu+\sigma\epsilon
\]

inference：

\[
z=0
\]

必须严格区分。

---

# 80. 常见误解十三：Validation 和真正 Robot Inference 完全相同

不是。

validation 有 ground-truth action，可以运行 CVAE encoder 并计算：

\[
L1+\beta KL
\]

真正 rollout：

- 没有 target actions；
- encoder 被丢弃；
- \(z=0\)；
- 可能使用 Temporal Ensemble。

---

# 81. 常见误解十四：Optimizer 只更新 Policy Decoder

**错误。**

original ACT joint optimization 会更新：

- CVAE encoder；
- ResNet；
- policy Transformer；
- action head。

---

# 82. 用四个对象记住 ACT Training

如果不想记代码，只需要记四个核心对象。

## Input

\[
\boxed{
o_t=(images_t,q_t)
}
\]

## Target

\[
\boxed{
A_t=a_{t:t+k}
}
\]

## Training Posterior

\[
\boxed{
q_\phi(z\mid A_t,q_t)
}
\]

## Objective

\[
\boxed{
L
=
L_1(\hat A_t,A_t)
+
\beta
D_{KL}
\left(
q_\phi(z\mid A_t,q_t)
\parallel
N(0,I)
\right)
}
\]

其中：

\[
\hat A_t
=
\pi_\theta(o_t,z)
\]

这就是 ACT training 的数学核心。

---

# 83. 一句话重新理解 ACT Training

> **ACT 训练时从 demonstration episode 中随机选择一个 timestep，把该时刻的多视角图像和 joint state 作为当前 observation，把随后 \(k\) 个 expert actions 作为 target chunk；CVAE encoder 利用 joint state 与 ground-truth action chunk 推断一个 Gaussian posterior 并重参数化采样 \(z\)，ACT policy 再根据 images、joint state 和 \(z\) 重建 action chunk。模型通过 masked L1 reconstruction 与 \(\beta\)-weighted KL regularization 联合训练，梯度同时更新 CVAE encoder、ResNet 和 Transformer policy。**

而所有：

- Temporal Ensemble；
- \(z=0\)；
- overlapping chunks 的加权融合；

都属于：

> **Inference**

不是 training。

---

# 84. 下一步：ACT Inference

现在训练已经完整。

下一篇会真正回答：

> 模型训练完以后，机器人每一个控制 timestep 究竟做什么？

从：

```text
当前 camera images
+
当前 qpos
↓
normalize
↓
z = 0
↓
ACT Policy
↓
k future actions
```

继续到：

```text
每 timestep 重新 query
↓
保存 overlapping chunks
↓
找到所有针对当前 timestep 的 predictions
↓
Temporal Ensemble
↓
反 normalization
↓
执行 target joint positions
↓
获取下一 observation
```

下一篇：

- [ACT Inference](./inference.md)

---

## Primary Source

Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.  
**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.**  
Robotics: Science and Systems (RSS), 2023.

- Paper: https://arxiv.org/abs/2304.13705
- RSS Proceedings: https://roboticsproceedings.org/rss19/p016.html
- Project: https://tonyzhaozh.github.io/aloha/

本文主要依据：

- Algorithm 1 — ACT Training
- Section IV-B — Modeling Human Data
- Section IV-C — Implementing ACT
- Appendix C — Detailed Architecture Diagram
- Table III — Hyperparameters of ACT

论文 Algorithm 1 的核心训练过程是：

\[
(o_t,a_{t:t+k})\sim\mathcal D
\]

\[
z\sim q_\phi(z\mid a_{t:t+k},\bar o_t)
\]

\[
\hat a_{t:t+k}
\sim
\pi_\theta(\cdot\mid o_t,z)
\]

以及：

\[
L=L_{reconst}+\beta L_{reg}
\]

其中：

\[
L_{reg}=D_{KL}(q_\phi\|N(0,I))
\]

---

## Paper / Implementation Notes

### Reconstruction Loss

Algorithm 1 写：

\[
MSE
\]

但正文 Section IV-C 与 official `policy.py` 实际使用：

\[
\boxed{L1}
\]

因此本文描述 actual original implementation 时采用：

\[
L1+\beta KL
\]

### Optimizer

Algorithm 1 写：

> ADAM

released official code 实际使用：

\[
\boxed{AdamW}
\]

### Real-Data Alignment

released `utils.py` 对 real-world data 使用：

```python
max(0, start_ts - 1)
```

作为 action 起始 index，并明确标注为 timestep alignment hack。

### Gradient Clipping / LR Scheduler

参数 parser 中保留相关选项，但 original released training path 标注为 `not used`，实际 training loop 没有调用它们。

---

## Official Implementation

ACT official repository:

https://github.com/tonyzhaozh/act

本文主要核对：

```text
utils.py
policy.py
imitate_episodes.py
detr/main.py
detr/models/detr_vae.py
```

released code 可确认：

- dataset item 每次随机选择 `start_ts`；
- observation 只取当前 timestep；
- future actions 从 start timestep 向后读取；
- episode 尾部 zero padding；
- 使用 `is_pad` mask；
- qpos / action 做 dataset standardization；
- image 先 `/255`，随后 ImageNet normalization；
- action target 在 `ACTPolicy` 中截到 `num_queries = chunk_size`；
- reconstruction 使用 masked L1；
- KL 使用 diagonal-Gaussian closed form；
- final loss 为 `l1 + kl_weight * kl`；
- optimizer 为 AdamW；
- ResNet backbone 会参与优化；
- CVAE encoder 与 policy joint optimization；
- best checkpoint 由最低 validation loss 选择。

---

## 原论文 ACT Hyperparameters

| Hyperparameter | Value |
|---|---:|
| Learning rate | \(1\times10^{-5}\) |
| Batch size | 8 |
| Encoder layers | 4 |
| Decoder layers | 7 |
| Feedforward dimension | 3200 |
| Hidden dimension | 512 |
| Attention heads | 8 |
| Chunk size | 100 |
| \(\beta\) | 10 |
| Dropout | 0.1 |

---

## 本文知识连接

### ACT 主线

- [ACT 到底解决了什么问题？](./act-what-problem-does-it-solve.md)
- [Action Chunking](./action-chunking.md)
- [Temporal Ensemble](./temporal-ensemble.md)
- [ACT Architecture](./architecture.md)
- [CVAE in ACT](./cvae-in-act.md)
- [为什么 ACT 推理时令 z = 0？](./why-z-zero-at-inference.md)

### Generative Models

- [Latent Variable](../../generative-models/latent-variable.md)
- [VAE](../../generative-models/vae.md)
- [Reparameterization Trick](../../generative-models/reparameterization-trick.md)
- [CVAE](../../generative-models/cvae.md)
- [Posterior Collapse](../../generative-models/posterior-collapse.md)

### 数学

- Mean
- Variance
- Standard Deviation
- [Normal Distribution](../../mathematics/normal-distribution.md)
- [KL Divergence](../../mathematics/kl-divergence.md)
- Gradient & Chain Rule

### Deep Learning

- L1 Loss
- [Backpropagation](../../deep-learning/backpropagation.md)
- AdamW
- Normalization
- Padding Mask

### 下一步

- [ACT Inference](./inference.md)
