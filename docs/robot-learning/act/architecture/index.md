---
title: "Architecture"
kind: "act-topic"
domain: "Robot Learning / ACT"
parent: "ACT"
canonical: "/robot-learning/act/architecture/"
prerequisites:
  - "/robot-learning/act/"
related:
  - "/robot-learning/act/cvae-in-act/"
  - "/robot-learning/act/vision-pipeline/"
  - "/robot-learning/act/training/"
---

# Architecture

ACT 的 architecture 最好分成两条数据流理解：

1. **policy 主干**：当前 observation → future action chunk；
2. **training-only latent branch**：current qpos + ground-truth action chunk → latent $z$。

第二条只在训练时存在。先看部署时真正留下的主干，再把训练分支接回来。

## Policy 主干

```text
多路 RGB images
      │
      ↓
  ResNet backbone
      │
      ↓
spatial visual features ──┐
                          │
current qpos ── projection ├──→ Transformer
                          │
latent z ───── projection ┘
                          │
                          ↓
                    encoder memory
                          ↑
                          │
                  action queries
                          │
                          ↓
               Transformer decoder
                          │
                          ↓
                  action head
                          │
                          ↓
        [a_hat_t ... a_hat_t+k-1]
```

这张图里最重要的是“谁提供 memory，谁提供 query”。Observation information 主要被编码成 memory；$k$ 个 learnable action queries 则对应 $k$ 个未来输出槽位。

## 视觉输入

对于每个 camera image，官方实现使用 [ResNet](/deep-learning/cnn/resnet/) backbone 提取 feature map。随后用 $1\times1$ convolution 投影到 Transformer hidden dimension。

多相机 feature maps 在官方代码中沿 spatial width 方向拼接，再和对应 positional features 一起送入 Transformer。

因此 Transformer 看到的不是原始 RGB pixels，而是一组已经具有视觉语义的 spatial features。

更完整的过程见 [Vision Pipeline](/robot-learning/act/vision-pipeline/)。

## Proprioception 与 latent

当前 joint position $q_t$ 通过 linear projection 映射到 hidden dimension。

Latent $z$ 也先通过 linear layer 映射到相同 hidden dimension。官方实现还为 proprioception 与 latent 使用额外 learned position embeddings，使模型能区分这些非图像输入的角色。

因此进入 policy Transformer 的信息可以抽象成：

\[
\{\text{visual features},\text{proprio feature},\text{latent feature}\}.
\]

## Action queries

ACT 从 [DETR](/deep-learning/detr/) 借用了 [Object Query](/deep-learning/detr/object-query/) 的 learnable output-slot 思路。模型维护 $k$ 个 learnable embeddings：

\[
Q_{action}\in\mathbb R^{k\times d}.
\]

它们不是过去的真实动作，也不是未来动作数值本身，而是 $k$ 个输出槽位的 learned query states。

第 1 个 query 对应 chunk 中第 1 个动作位置，第 2 个对应第 2 个，以此类推。经过 Transformer decoder 后，每个 query 得到一个 contextual representation，再经过 linear action head 变成 joint target。

如果 state dimension 为 14：

\[
H_{dec}\in\mathbb R^{B\times k\times d}
\]

经过 action head 后：

\[
\hat A\in\mathbb R^{B\times k\times14}.
\]

## Training-only latent branch

训练时，模型已知真实 action chunk。ACT 使用另一套 Transformer encoder 来从

- 一个 learnable [CLS Token](/deep-learning/bert/cls-token/)；
- current qpos；
- ground-truth action sequence

中提取一个 summary representation：

```text
[CLS] , q_t , a_t , a_t+1 , ... , a_t+k-1
                     │
                     ↓
          latent Transformer encoder
                     │
                     ↓
               CLS output
                     │
                     ↓
              linear projection
                ↙          ↘
              μ              log σ²
                     │
                     ↓
                 sample z
```

官方实现的 latent dimension 是 32。这个 32 是 released code 的设计选择，不是 CVAE 的通用要求。

## 两个 Transformer 模块的不同职责

容易混淆的一点是：ACT 里不只有一个“Transformer”。

- latent encoder Transformer：训练时从 qpos + action sequence 得到 $q_\phi(z\mid\cdot)$；
- policy Transformer encoder-decoder：根据 observation + latent 预测 action chunk。

前者部署时消失；后者是真正执行 policy 的主干。

## Architecture 与算法思想的边界

[ResNet](/deep-learning/cnn/resnet/)、[Transformer](/deep-learning/transformer/)、[CLS Token](/deep-learning/bert/cls-token/) 与 [Object Query](/deep-learning/detr/object-query/) 都来自 ACT 之前的模型或机制。ACT 的模型设计价值在于它如何把这些组件放到 action-chunk imitation learning 中：

```text
human demonstration variation
          ↓
     CVAE latent
          ↓
observation + z
          ↓
Transformer memory/query architecture
          ↓
future action chunk
          ↓
temporal aggregation during execution
```

## Sources

- Zhao et al., **Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware**, 2023. https://arxiv.org/abs/2304.13705
- Official model implementation: https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py
- Carion et al., **DETR**, 2020. https://arxiv.org/abs/2005.12872
