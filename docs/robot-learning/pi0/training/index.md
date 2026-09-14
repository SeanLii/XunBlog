---
title: "Training"
kind: "pi0-topic"
domain: "Robot Learning / π0"
parent: "π0"
canonical: "/robot-learning/pi0/training/"
prerequisites:
  - "/robot-learning/pi0/flow-matching-in-pi0/"
  - "/robot-learning/pi0/architecture/"
related:
  - "/robot-learning/pi0/pretraining-and-posttraining/"
  - "/robot-learning/pi0/inference/"
  - "/robot-learning/pi0/complete-data-flow/"
---

# Training

π0 的单个训练样本可以先理解成：

```text
当前 observation o_t
+
真实未来 action chunk A_t
```

其中 observation 包含 images、language instruction 与 robot state：

\[
o_t=[I_t^1,\ldots,I_t^n,\ell_t,q_t],
\]

真实 action chunk 为

\[
A_t=[a_t,\ldots,a_{t+H-1}].
\]

训练的目标不是直接让网络输出 $A_t$，而是把真实 action 与随机 noise 混合，构造一个 Flow Matching regression problem。

## Step 1：得到真实 action chunk

从 robot demonstration 的时间序列中，以当前时刻 $t$ 为起点截取：

\[
A_t=[a_t,a_{t+1},\ldots,a_{t+H-1}].
\]

论文使用 $H=50$。

所以一个训练 target 本身就是二维结构：

\[
A_t\in\mathbb R^{H\times d_a}.
\]

## Step 2：采样随机 noise

采样与 action chunk shape 完全相同的 Gaussian noise：

\[
\epsilon\sim\mathcal N(0,I).
\]

直观上，它是“完全不像正确动作”的起点。

## Step 3：采样 flow timestep

再采样

\[
\tau\in[0,1].
\]

π0 论文不是 uniform sampling，而使用 shifted beta distribution，强调更 noisy 的区域。

$\tau$ 决定当前训练样本离真实动作有多远。

## Step 4：构造 noisy action

论文 convention 下：

\[
A_t^\tau
=
\tau A_t+(1-\tau)\epsilon.
\]

例如 $\tau=0.2$ 时，noise 占比仍然很高；$\tau=0.9$ 时，当前 action 已经非常接近 ground truth。

训练不是只看“轻微加噪”或“完全 noise”，而是在不同 flow stages 上学习 vector field。

## Step 5：构造 target velocity

这条线性 path 的速度是：

\[
u_t=A_t-\epsilon.
\]

注意 target 是一个与 $A_t$ 同 shape 的向量场：

\[
u_t\in\mathbb R^{H\times d_a}.
\]

模型每个 action position 都要预测相应 velocity。

## Step 6：编码 observation prefix

Images 经过 vision encoder 得到 visual tokens；language prompt 得到 text embeddings。

它们进入 VLM weights：

```text
images ─→ visual tokens ─┐
                         ├──→ VLM prefix
language ─→ text tokens ─┘
```

Robot state $q_t$ 则通过 robotics-specific projection 进入 Action Expert 一侧。

## Step 7：把 noisy actions 与 τ 送入 Action Expert

每个 action vector 先投影到 expert hidden dimension，再与 timestep embedding 组合：

```text
A_t^τ ─→ action projection ─┐
                            ├─→ MLP ─→ action tokens
τ ─────→ time embedding ────┘
```

然后 image/language/state/action tokens 按 π0 的 blockwise mask 一起进入 Transformer computation。

## Step 8：只读取 action positions 的输出

模型最后只取 action block 的 $H$ 个 hidden states：

\[
h^{action}
\in\mathbb R^{H\times d_h}.
\]

再通过 output projection：

\[
v_\theta
=
W_{out}h^{action}
\in\mathbb R^{H\times d_a}.
\]

这个输出就是预测 vector field。

## Step 9：Flow Matching Loss

论文形式：

\[
\mathcal L(\theta)
=
\mathbb E
\left[
\|v_\theta(A_t^\tau,o_t)-(A_t-\epsilon)\|_2^2
\right].
\]

因此一次训练 forward 的本质是：

```text
(o_t, A_t)
  │
  ├─ sample ε, τ
  │
  ├─ build A_t^τ
  │
  ↓
 π0 predicts vθ
  │
  ↓
MSE with target velocity
```

网络没有在这一次训练中真的运行 10 个 denoising steps。那是 inference 的事情。

## 当前 openpi implementation 的符号方向

当前官方 openpi 采用与论文相反的 time convention：

```text
code t = 1 → noise
code t = 0 → data
```

因此代码构造：

\[
x_t=t\epsilon+(1-t)A,
\]

目标是：

\[
u_t=\epsilon-A.
\]

loss 仍然是 predicted velocity 与 target velocity 的 squared error。

这不是训练目标发生了本质变化，而是同一条 flow path 的参数方向反过来了。

## Training 与 Inference 的关键区别

训练时真实 action chunk 存在，所以能直接构造任意中间状态和 target velocity。

推理时没有真实 $A_t$，只能：

```text
从 noise 开始
→ 预测 velocity
→ 更新 action
→ 再预测 velocity
→ ...
→ 最终 action
```

因此“训练一次 forward，推理多次 forward”是 Flow Matching 架构的自然结果。

## Sources

- Black et al., **π0: A Vision-Language-Action Flow Model for General Robot Control**, Section IV. https://arxiv.org/abs/2410.24164
- Official openpi `Pi0.compute_loss`. https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py
