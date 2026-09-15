---
title: "Complete Data Flow"
kind: "act-topic"
domain: "Robot Learning / ACT"
parent: "ACT"
canonical: "/robot-learning/act/complete-data-flow/"
prerequisites:
  - "/robot-learning/act/architecture/"
  - "/robot-learning/act/training/"
  - "/robot-learning/act/inference/"
related:
  - "/robot-learning/act/cvae-in-act/"
  - "/robot-learning/act/vision-pipeline/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Complete Data Flow

> **知识边界**：本文的 canonical 对象是 **Complete Data Flow**。依赖机制由 [Architecture](/robot-learning/act/architecture/)、[Training](/robot-learning/act/training/)、[Inference](/robot-learning/act/inference/) 的 canonical page 定义；本文只在当前语境中调用其接口。


这一页把 ACT 从 dataset sample 到 robot execution 完整串起来。前面的页面分别解释了局部机制；这里的目标是让所有变量回到同一条数据流中。

## Training Data Flow

从一条 demonstration trajectory 选时刻 $t$：

\[
(I_t,q_t,a_t,a_{t+1},\ldots).
\]

取长度为 $k$ 的 future action chunk：

\[
A_t=a_{t:t+k-1}.
\]

### 1. Latent branch

```text
q_t ──────────────┐
                  │
A_t ──────────────┤
                  ↓
  project to embeddings
                  ↓
[CLS, q_t, a_t, ..., a_t+k-1]
                  ↓
   Transformer encoder
                  ↓
          h_CLS
          ↙    ↘
         μ    logσ²
           \  /
            z
```

概率上：

\[
q_\phi(z\mid q_t,A_t)
=
\mathcal N(\mu_t,\operatorname{diag}(\sigma_t^2)).
\]

随后

\[
z_t=\mu_t+\sigma_t\odot\epsilon,
\quad
\epsilon\sim\mathcal N(0,I).
\]

### 2. Vision branch

每个 camera：

```text
I_t^c
 │
 ↓
ResNet
 │
 ↓
feature map
 │
 ↓
1×1 projection
 │
 ↓
visual features in hidden_dim
```

多 camera features 与 positional representations 被拼接后送入 policy Transformer。

### 3. Robot-state and latent conditioning

\[
q_t\rightarrow e_q,
\qquad
z_t\rightarrow e_z.
\]

两者都被 linear projection 到 hidden dimension，与 visual features 一起作为 observation-side information。

### 4. Action queries

模型有 $k$ 个 learned query embeddings：

\[
Q\in\mathbb R^{k\times d}.
\]

Transformer decoder 输出 $k$ 个 action representations：

\[
H\in\mathbb R^{B\times k\times d}.
\]

Linear action head 得到

\[
\hat A_t\in\mathbb R^{B\times k\times d_a}.
\]

### 5. Loss

有效 timestep 上计算 reconstruction loss：

\[
\mathcal L_{L1}
=
\operatorname{masked\;L1}(A_t,\hat A_t).
\]

同时：

\[
\mathcal L_{KL}
=
D_{KL}(q_\phi(z\mid q_t,A_t)\|\mathcal N(0,I)).
\]

总 loss：

\[
\mathcal L=\mathcal L_{L1}+\beta\mathcal L_{KL}.
\]

这就是一次 training forward/backward pass 的主要逻辑。

---

## Inference Data Flow

部署时没有 $A_t$，因此 latent branch 被整个删除。

### 1. Current observation

```text
latest images I_t
latest qpos q_t
```

### 2. Fixed latent

\[
z_t=0.
\]

### 3. Policy prediction

\[
(I_t,q_t,0)
\rightarrow
\hat A_t
=
[\hat a_t^{(t)},\ldots,\hat a_{t+k-1}^{(t)}].
\]

### 4. Save overlapping chunk

在 temporal aggregation 模式下，每个 timestep 都有新 chunk：

```text
t=0  [a0^0 a1^0 a2^0 a3^0 ...]
t=1       [a1^1 a2^1 a3^1 ...]
t=2            [a2^2 a3^2 ...]
```

### 5. Aggregate current-time predictions

执行时刻 $t$，收集这一列：

\[
a_t^{(0)},a_t^{(1)},\ldots,a_t^{(t)}
\]

中仍覆盖 $t$ 的预测，并做指数加权平均。

### 6. Execute and observe again

最终 action 经反标准化后作为 joint target 交给 robot/environment。执行一步后重新读取 image 与 qpos，重复整个 inference loop。

---

## 一张合并图

```text
                         TRAIN ONLY
                 true future action chunk
                    │             │
                    │             └────────────── target
                    ↓
q_t ─────────→ latent encoder
                    ↓
                    z
                    │
                    │
images → ResNet ────┤
q_t ─────────────────┤
                    ↓
             policy Transformer
                    ↑
             action queries
                    │
                    ↓
            predicted chunk
                    │
       ┌────────────┴────────────┐
       ↓                         ↓
  reconstruction loss      inference storage
       + KL                     │
                                ↓
                      overlapping predictions
                                ↓
                       Temporal Ensemble
                                ↓
                         executed action
```

Training 走左侧 loss；deployment 走右侧 temporal execution。两者共享 policy predictor，但不共享所有输入分支。

## Sources

- Zhao et al., **Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware**, 2023. https://arxiv.org/abs/2304.13705
- Official ACT repository. https://github.com/tonyzhaozh/act
