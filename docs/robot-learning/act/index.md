---
title: "ACT"
kind: "canonical"
domain: "Robot Learning / ACT"
parent: "ACT"
canonical: "/robot-learning/act/"
prerequisites:
  - "/robot-learning/behavior-cloning/"
related:
  - "/robot-learning/act/action-chunking/"
  - "/robot-learning/act/temporal-ensemble/"
  - "/robot-learning/act/architecture/"
  - "/robot-learning/act/cvae-in-act/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# ACT

ACT（Action Chunking with Transformers）是一种从人类示范学习精细机器人操作的 policy architecture。它面对的核心矛盾是：控制必须保持闭环，单步 [Behavior Cloning](/robot-learning/behavior-cloning/) 的误差却会沿长时序累积；人类示范还包含速度、姿态与操作风格差异，使相似观测并不总对应唯一动作。ACT 用 action chunk 改变预测单位，用 temporal ensemble 保持高频反馈，再用训练期 latent variable 表示示范差异。

## 从单步预测到动作块

单步策略在时刻 $t$ 只预测

\[
\hat a_t=\pi_\theta(o_t).
\]

若一次微小偏差把机器人带离 demonstration distribution，下一步预测面对的就是训练中较少出现的状态，误差可能继续扩大。ACT 改为输出长度为 $k$ 的未来动作：

\[
\hat A_t
=
(\hat a_{t|t},\hat a_{t+1|t},\ldots,\hat a_{t+k-1|t})
=\pi_\theta(o_t,z).
\]

[Action Chunking](/robot-learning/act/action-chunking/) 让一个 forward 直接表示局部轨迹的时间相关性，并把需要连续作出独立决策的次数从原始 horizon $T$ 降到约 $T/k$。但预测越远，对尚未观测到的环境变化越不确定，所以 $k$ 不是越大越好。

## Chunk Prediction 仍然可以闭环

一次预测 $k$ 步不等于必须 open-loop 执行完 $k$ 步。ACT 可以在每个 control timestep 重新读取最新图像与关节状态，再输出一个新 chunk。于是实际时刻 $t$ 同时收到多个历史 query 对它的估计：

\[
\hat a_{t|t},\ \hat a_{t|t-1},\ldots,\hat a_{t|t-k+1}.
\]

[Temporal Ensemble](/robot-learning/act/temporal-ensemble/) 对这些重叠预测加权：

\[
\tilde a_t
=
\frac{\sum_{i=0}^{k-1}w_i\hat a_{t|t-i}}
{\sum_{i=0}^{k-1}w_i},
\qquad
w_i=\exp(-mi).
\]

新的预测更贴近当前观测而权重更高，旧预测仍提供时间平滑。这个设计同时保留 chunk-level structure 和 timestep-level feedback；其代价是每步都要进行一次 policy forward。

## Observation 如何进入 Transformer

ACT 的当前 observation 可写为

\[
o_t=(I_t^1,\ldots,I_t^C,q_t),
\]

其中 $I_t^c$ 是第 $c$ 路相机图像，$q_t$ 是 proprioceptive state。在 ALOHA 双臂设置中，四路 RGB 图像分别经过共享 [ResNet](/deep-learning/cnn/resnet/) backbone；spatial feature maps 被展平为 visual tokens，并保留二维位置表示。joint state 与 latent $z$ 经线性投影成为额外 observation tokens。

[ACT Architecture](/robot-learning/act/architecture/) 随后用 Transformer encoder 融合这些条件，再由 $k$ 个 learned action queries 从 encoder memory 中读取与各未来位置相关的信息：

```text
multi-camera images → ResNet → spatial visual tokens ─┐
current joint state → projection ─────────────────────┤
training/inference latent z → projection ─────────────┤
                                                       ↓
                                             Transformer encoder
                                                       ↓
                                              observation memory
                                                       ↑
                                          k learned action queries
                                                       ↓
                                             Transformer decoder
                                                       ↓
                                           k × action-dim outputs
```

在论文的主要 ALOHA 配置中 $k=100$、action dimension 为 14，因此一次输出 $100\times14$ 个 absolute joint-position targets。learned queries 的结构来自 DETR，但其语义从 object slots 变成 future temporal slots。

## CVAE 只在训练时读取未来动作

相似 observation 下，人类 demonstration 可能沿不同但都合理的局部轨迹运动。若只用单一 L1/L2 regression，模型容易在多种行为之间取平均。ACT 在训练阶段加入 [CVAE in ACT](/robot-learning/act/cvae-in-act/)：recognition encoder 读取当前关节状态和真实 future chunk，估计

\[
q_\phi(z\mid q_t,A_t)
=\mathcal N\!\left(\mu_\phi,\operatorname{diag}(\sigma_\phi^2)\right).
\]

通过 reparameterization 采样

\[
z=\mu_\phi+\sigma_\phi\odot\epsilon,
\qquad \epsilon\sim\mathcal N(0,I),
\]

再令主 policy 预测 $\hat A_t=\pi_\theta(o_t,z)$。$z$ 因而能够携带仅从当前 observation 难以辨认、却能从完整示范片段推断的行为差异。

## 目标函数连接动作重建与潜空间

released ACT 的主要训练目标为

\[
\mathcal L
=
\mathcal L_{L1}(\hat A_t,A_t)
+\beta D_{KL}\!\left(q_\phi(z\mid q_t,A_t)\,\|\,\mathcal N(0,I)\right).
\]

reconstruction term 要求预测 chunk 接近示范；KL term 让训练时 posterior 保持在标准正态 prior 附近，使部署时在没有真实未来动作的条件下仍能提供合法 latent。$\beta$ 过小会使潜空间难以由 prior 使用，过大则可能使 posterior 不再携带动作信息。

## 推理时固定 z=0

部署阶段没有 $A_t$，recognition encoder 无法运行。released implementation 取 prior mean：

\[
z=0.
\]

因此同一 observation 下的 policy output 是确定性的。这个选择并不表示训练期 latent 没有作用；训练时它帮助网络分离 demonstration variation，而 [为什么 ACT 推理时令 $z=0$](/robot-learning/act/why-z-zero-at-inference/) 解释了 posterior、prior 与部署策略之间的接口差异。

完整 [Inference](/robot-learning/act/inference/) 过程是：读取最新 observation，使用 $z=0$ 生成新 chunk，将它写入 overlapping prediction buffer，再由 temporal ensemble 选出当前执行动作。执行一步后重新观测，闭环重复。

## 三项机制共同决定能力边界

Action chunking 缩短 effective horizon，却把更远的未观测变化纳入同一次预测；temporal ensemble 用最新观测修正旧计划，却增加每步推理成本并可能平滑快速变化；CVAE 表示训练示范的多样性，但固定 $z=0$ 的 released policy 不在部署时主动采样多种风格。

ACT 因而不是单纯“用 Transformer 做控制”。它是一套围绕长时序模仿学习组织的接口：视觉与状态形成当前条件，action queries 定义未来位置，latent branch 只服务训练期行为建模，overlapping chunks 则把局部轨迹预测重新接回闭环控制。

## Sources

- Zhao et al. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*. RSS 2023. https://arxiv.org/abs/2304.13705
- Official ACT repository. https://github.com/tonyzhaozh/act
