---
title: "Inference"
kind: "pi0-topic"
domain: "Robot Learning / π0"
parent: "π0"
canonical: "/robot-learning/pi0/inference/"
prerequisites:
  - "/deep-learning/transformer/kv-cache/"
  - "/robot-learning/pi0/flow-matching-in-pi0/"
  - "/mathematics/numerical-methods/euler-method/"
related:
  - "/robot-learning/pi0/blockwise-causal-attention-mask/"
  - "/robot-learning/pi0/complete-data-flow/"
---

# Inference

π0 的 inference 不是一次 forward 直接得到动作。

真正流程是：

```text
当前 observation
      │
      ↓
先编码一次并缓存
      │
random action noise
      │
      ↓
Flow step 1
      ↓
Flow step 2
      ↓
...
      ↓
Flow step 10
      ↓
final action chunk
      ↓
执行其中一部分动作
      ↓
重新观察并再次 inference
```

理解 π0 inference，要把“robot timestep”和“flow timestep”分开。

## 两种时间同时存在

### Robot timestep

$t$ 表示真实机器人世界中的控制时间：

```text
t, t+1, t+2, ...
```

### Flow timestep

$\tau$ 表示一次 action generation 内部，从 noise 走向 data 的生成进度：

```text
τ=0 → 0.1 → ... → 1
```

论文中，一次 robot inference 会内部完成 10 个 flow steps，然后才得到一个未来 action chunk。

这两个“时间”不能混为一谈。

## Step 1：固定当前 observation

当前时刻 observation：

\[
o_t=[I_t^1,\ldots,I_t^n,\ell_t,q_t].
\]

images 与 language 不会在这 10 个 flow steps 中变化，robot state 也固定为当前 $q_t$。

因此 π0 先计算 observation-side representations。

## Step 2：建立 KV Cache

由于 [Blockwise Causal Attention Mask](/robot-learning/pi0/blockwise-causal-attention-mask/) 保证 prefix 不依赖后面的 action block，observation 的 keys / values 可以缓存。这个机制本身见 [KV Cache](/deep-learning/transformer/kv-cache/)。

```text
images + language + state
          │
          ↓
      prefix forward
          │
          ↓
        KV cache
```

后面 10 次 flow step 不需要重复完整计算这些固定 token。

## Step 3：从 Gaussian noise 开始

论文 convention：

\[
A_t^0\sim\mathcal N(0,I).
\]

这个 tensor 已经有最终 action chunk 的 shape：

\[
H\times d_a.
\]

只是数值目前完全是随机 noise。

## Step 4：预测当前 vector field

把当前 noisy actions、flow timestep 和缓存 observation 一起送入模型：

\[
v_\theta(A_t^\tau,o_t).
\]

模型输出一个同 shape 的 velocity field。

## Step 5：Euler update

论文使用 forward Euler：

\[
A_t^{\tau+\delta}
=
A_t^\tau+
\delta v_\theta(A_t^\tau,o_t).
\]

实验中：

\[
\delta=0.1,
\]

因此共执行 10 次。

直观看：

```text
random action
  + small predicted correction
      ↓
less random action
  + next correction
      ↓
...
      ↓
executable action chunk
```

每一步都重新根据**更新后的 action chunk**计算新的 velocity。

## Step 6：得到 H-step action chunk

最终生成：

\[
A_t=[a_t,\ldots,a_{t+H-1}].
\]

论文使用 $H=50$。

但生成 50 步不代表一定连续执行完 50 步才重新观察。

## π0 最终没有使用 Temporal Ensemble

这一点与 ACT 很不一样。

π0 论文 Appendix D 明确说明：他们早期尝试过 [Temporal Ensemble](/robot-learning/act/temporal-ensemble/)，但发现它降低 policy performance，因此最终没有融合不同 inference call 对同一时刻的预测，而是 open-loop 执行 action chunk 的一部分。

论文中的执行设置是：

- 20 Hz 的 UR5e / Franka：每执行 16 个动作，大约 0.8 秒，重新 inference；
- 其他 50 Hz robots：每执行 25 个动作，大约 0.5 秒，重新 inference。

因此 π0 的 closed-loop 节奏更像：

```text
observe
  ↓
generate 50-action chunk
  ↓
execute first N actions open-loop
  ↓
observe again
  ↓
regenerate
```

而不是 ACT 那种每个 timestep 都重新预测并用 Temporal Ensemble 融合。

## 当前 openpi 的 time convention

官方 openpi 当前 implementation 使用相反方向：

```text
t=1 : noise
      ↓
      ↓ dt < 0
      ↓
t=0 : action
```

代码默认：

```text
num_steps = 10
dt = -1 / num_steps
```

然后更新：

\[
x_{t+dt}=x_t+dt\,v_t.
\]

所以阅读 paper 与 code 时必须先确认当前采用哪一种 flow-time convention。

## Action Expert 主导重复推理成本

论文报告的 3-camera、RTX 4090 timing 中：

- image encoders：约 14 ms；
- observation forward：约 32 ms；
- 10 次 action flow forward：合计约 27 ms；
- on-board total：约 73 ms。

这里最重要的结构原因是 prefix KV caching：VLM observation 不必每个 flow step 全部重跑。

## Sources

- Black et al., **π0: A Vision-Language-Action Flow Model for General Robot Control**, Appendix D. https://arxiv.org/abs/2410.24164
- Official openpi `Pi0.sample_actions`. https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py
