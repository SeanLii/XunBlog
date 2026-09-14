---
title: "Complete Data Flow"
kind: "pi0-topic"
domain: "Robot Learning / π0"
parent: "π0"
canonical: "/robot-learning/pi0/complete-data-flow/"
prerequisites:
  - "/robot-learning/pi0/architecture/"
  - "/robot-learning/pi0/training/"
  - "/robot-learning/pi0/inference/"
related:
  - "/robot-learning/pi0/action-expert/"
  - "/robot-learning/pi0/flow-matching-in-pi0/"
---

# Complete Data Flow

这一页不再引入新的 π0 模块，而是把 training 与 inference 从原始输入一路走到最终动作。

先固定符号：

\[
o_t=[I_t^1,\ldots,I_t^n,\ell_t,q_t]
\]

是当前 observation，

\[
A_t=[a_t,\ldots,a_{t+H-1}]
\]

是未来 action chunk。

论文使用 $H=50$。

## Training Data Flow

### 1. 从 demonstration 取 observation 与 future actions

```text
robot trajectory
      │
      ├── current images I_t
      ├── language ℓ_t
      ├── robot state q_t
      └── future actions A_t
```

这里 $A_t$ 是真实监督数据。

### 2. 构造 Flow Matching training state

采样：

\[
\epsilon\sim\mathcal N(0,I),
\qquad
\tau\sim p(\tau).
\]

论文 convention：

\[
A_t^\tau
=
\tau A_t+(1-\tau)\epsilon.
\]

同时 target velocity 为：

\[
u_t=A_t-\epsilon.
\]

此时一条训练样本变成：

```text
condition:       images + language + robot state
current sample:  noisy action chunk A_t^τ
flow position:   τ
target:          velocity u_t
```

### 3. 图像变成 visual tokens

每路 RGB image：

```text
I_t^i
  │
  ↓
Vision Encoder
  │
  ↓
visual tokens
```

这些 tokens 使用 VLM expert weights。

### 4. Language 变成 text embeddings

```text
instruction ℓ_t
      │
      ↓
 tokenizer
      │
      ↓
 token ids
      │
      ↓
 embeddings
```

Visual 与 language tokens 组成 VLM block。

### 5. Robot state 变成 state token

\[
q_t\in\mathbb R^{d_q}
\]

通过 linear projection 进入 Action Expert hidden space：

```text
q_t → Linear → state token
```

### 6. Noisy action chunk 变成 H 个 action tokens

每个

\[
a_{t+i}^\tau
\]

先做 action projection，再和 timestep embedding $\phi(\tau)$ 通过 MLP 混合。

最终得到：

\[
H\text{ action tokens}.
\]

### 7. 拼成三个 blocks

```text
┌──────────────────┬───────────┬────────────────────┐
│ images + language│ state q_t │ noisy actions A^τ  │
│    VLM expert    │           Action Expert        │
└──────────────────┴───────────┴────────────────────┘
```

通过 blockwise causal mask：

- image/language 不读 state/action；
- state 可读 image/language，但不读 actions；
- actions 可读所有 prefix，并在 action block 内双向读取。

### 8. Transformer interaction

两套 expert weights 共同参与一个 Transformer system。

可以把每一层想成：

```text
VLM-side hidden states ──┐
                         ├── self-attention mixing
Action-side states ──────┘
                         │
                         ↓
                 each side's expert MLP
```

具体参数实现不是两个串联模型。

### 9. 只取 H 个 action outputs

从 Transformer 最终输出中取 action positions：

\[
h_{act}\in\mathbb R^{H\times d_h}.
\]

再投影到 action dimension：

\[
v_\theta
\in\mathbb R^{H\times d_a}.
\]

### 10. 计算 Flow Matching loss

\[
\mathcal L
=
\|v_\theta-u_t\|^2.
\]

到这里，一次 training forward 结束。

---

## Inference Data Flow

Inference 没有真实 action chunk，所以最前面的数据流完全不同。

### 1. 获取当前 observation

```text
camera images
language instruction
robot state
```

它们与 training 的 condition 一样。

### 2. 先计算 observation prefix

```text
images → visual tokens ─┐
language → text tokens ─┼→ prefix Transformer forward
state → state token ────┘
                              │
                              ↓
                           KV cache
```

这些输入在本次 action generation 的 10 个 flow steps 中固定不变。

### 3. 初始化 action noise

\[
A_t^0\sim\mathcal N(0,I)
\]

shape 已经是

\[
H\times d_a.
\]

### 4. Flow step 1

```text
current noisy actions A^0
        + τ=0
        + cached observation
              │
              ↓
         Action Expert
              │
              ↓
          velocity vθ
              │
              ↓
      Euler update to A^0.1
```

### 5. 重复更新

同一 observation cache 保持不变，只替换：

- 当前 action chunk；
- 当前 flow timestep。

```text
A^0
 ↓
A^0.1
 ↓
A^0.2
 ↓
...
 ↓
A^1
```

论文使用 10 steps。

### 6. 得到 final action chunk

\[
A_t=[a_t,\ldots,a_{t+H-1}].
\]

这时才真正得到机器人动作。

### 7. 执行 chunk 的前一部分

π0 论文没有采用 Temporal Ensemble，而是 open-loop 执行一部分生成的 actions。

```text
50 actions generated
       │
       ├── execute first N actions
       │
       ↓
new observation
       │
       ↓
run π0 again
```

所以 π0 仍然会周期性重新观察环境，但不是每一个 action timestep 都重新完整生成。

## 一张最终总图

```text
                    ┌──────── TRAIN ONLY ────────┐
                    │ real future action A_t     │
                    │        + noise ε           │
                    │        + flow time τ       │
                    │             │              │
                    │             ↓              │
                    │       noisy A_t^τ           │
                    └─────────────┬───────────────┘
                                  │
images ─→ vision tokens ─┐        │
language ─→ text tokens ─┼─→ Transformer with VLM + Action Expert
state ─→ state token ────┤        │
noisy actions ───────────┘        ↓
                            predicted vector field
                                  │
                TRAIN: compare with target velocity
                INFER: Euler update and repeat
                                  │
                                  ↓
                             action chunk
                                  │
                                  ↓
                              robot executes
```

这张图里最重要的分界是：

- training 知道 ground-truth actions，因此能直接制造任意 noisy intermediate state；
- inference 不知道 ground truth，只能从 random noise 开始沿 learned vector field 积分。

## Sources

- Black et al., **π0: A Vision-Language-Action Flow Model for General Robot Control**, 2024. https://arxiv.org/abs/2410.24164
- Official openpi implementation. https://github.com/Physical-Intelligence/openpi
