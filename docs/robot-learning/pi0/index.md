---
title: "π0"
kind: "canonical"
domain: "Robot Learning / π0"
parent: "π0"
canonical: "/robot-learning/pi0/"
prerequisites:
  - "/robot-learning/vision-language-action-model/"
  - "/generative-models/flow-matching/"
  - "/robot-learning/act/action-chunking/"
related:
  - "/robot-learning/pi0/architecture/"
  - "/robot-learning/pi0/action-expert/"
  - "/robot-learning/pi0/training/"
  - "/robot-learning/pi0/inference/"
---

# π0

π0（pi-zero）是 Physical Intelligence 提出的 Vision-Language-Action（VLA）robot policy。它以预训练 Vision-Language Model 为语义 backbone，加入机器人专用 Action Expert，并通过 conditional Flow Matching 生成连续 action chunks。

给定当前 observation

\[
o_t=[I_t^1,\ldots,I_t^n,\ell_t,q_t],
\]

其中 $I_t^i$ 是 camera images、$\ell_t$ 是 language instruction、$q_t$ 是 proprioceptive robot state，π0 建模未来动作分布

\[
p(A_t\mid o_t),
\qquad
A_t=[a_t,\ldots,a_{t+H-1}].
\]

论文使用 action horizon

\[
H=50.
\]

整体数据流为

```text
camera images ───────┐
language instruction ├──→ π0 ───→ continuous action chunk
robot state ─────────┘
```

与直接回归 action chunk 的 policy 不同，π0 在 inference 中从 Gaussian action noise 出发，通过多次 vector-field prediction 与 numerical integration 得到最终动作。

## Vision-Language Backbone

π0 建立在预训练 [Vision-Language Model](/deep-learning/multimodal/vision-language-model/) 之上。VLM 已经学习了图像与语言之间的大规模语义对应关系，因此可以提供：

- visual object / scene representations；
- language instruction representations；
- cross-modal semantic grounding。

但 VLM 的原始输出空间主要面向 language / multimodal prediction，并不直接定义机器人连续 control action。π0 因此需要额外的 robotics-specific parameters 与 action-generation objective。

## Action Expert

π0 增加 [Action Expert](/robot-learning/pi0/action-expert/) 处理 robot state 与 noisy action tokens。

Architecture 中存在两组主要 Transformer weights：

```text
image + language tokens → VLM expert ──────┐
                                           │
state + action tokens   → Action Expert ───┤
                                           ↓
                               shared attention interaction
```

两组 tokens 不是串联的两个独立模型。它们在同一 Transformer computation 中通过 attention 交换信息，但不同 token groups 使用不同 expert parameters 处理其 modality-specific transformations。

Action Expert 的规模小于 VLM backbone，因为 Flow Matching inference 中 action-side computation 需要重复执行多次。

## Continuous Action Generation with Flow Matching

π0 不直接使用单次 regression

\[
\hat A_t=f_\theta(o_t)
\]

得到最终动作，而是学习 conditional vector field

\[
v_\theta(A_t^\tau,o_t).
\]

训练时把 ground-truth action chunk $A_t$ 与 Gaussian noise $\epsilon$ 沿一条 probability path 混合；模型学习该路径上的 velocity。

论文 convention 下：

\[
A_t^\tau
=
\tau A_t+(1-\tau)\epsilon,
\]

\[
\frac{dA_t^\tau}{d\tau}=A_t-\epsilon.
\]

因此 Flow Matching objective 训练模型预测

\[
v_\theta(A_t^\tau,o_t)
\approx
A_t-\epsilon.
\]

Inference 从随机 action noise 出发，通过 Euler integration 重复更新 action chunk。论文使用 10 个 integration steps。

完整数学见 [Flow Matching](/generative-models/flow-matching/) 与 [Flow Matching in π0](/robot-learning/pi0/flow-matching-in-pi0/)。

## Input Blocks and Attention Structure

π0 的 Transformer input 可按功能分为三个 blocks：

```text
[images + language] | [robot state] | [noisy action chunk]
```

模型使用 [Blockwise Causal Attention Mask](/robot-learning/pi0/blockwise-causal-attention-mask/) 控制不同 blocks 的依赖方向：

- image / language prefix 不读取 robot state 与 action suffix；
- state 可以读取 image / language，但不读取 action；
- action tokens 可以读取 observation prefix，并在 action block 内彼此双向 attention。

这种结构同时支持 observation prefix 的 KV caching 和 chunk-level joint action modeling。

## Inference Procedure

给定固定 observation，π0 的 action generation 过程为：

```text
encode image + language prefix
        ↓
cache observation-side keys / values
        ↓
initialize Gaussian action noise
        ↓
Action Expert predicts vector field
        ↓
Euler update
        ↓
repeat for 10 flow steps
        ↓
final action chunk
```

Observation prefix 在同一次 flow integration 中不变，因此可以缓存；每个 flow step 主要重新计算 state/action suffix 与 updated noisy action chunk。

生成一个 action chunk 后，robot controller 执行其中一部分动作，再获取新 observation 并重新运行 policy。

π0 论文报告早期尝试了 ACT-style Temporal Ensemble，但最终没有采用，因为在其实验中降低了 performance。

## Pre-Training and Post-Training

π0 的 generalist behavior 不只来自 architecture。论文使用大规模、多任务、多 embodiment robot data 进行 pre-training，再对困难任务使用更集中、高质量的数据 post-train。

Robot pre-training data 覆盖多种 single-arm、dual-arm 与 mobile manipulation settings，并结合 Open X-Embodiment 数据。VLM backbone 又提供 Internet-scale visual-language pretraining knowledge。

[Pre-training and Post-training in π0](/robot-learning/pi0/pretraining-and-posttraining/) 分别解释 broad data coverage 与 downstream behavior specialization 的作用。

## Relation to ACT

π0 与 ACT 都使用 future action chunks，但二者的 policy architecture 与生成机制不同。

| | ACT | π0 |
|---|---|---|
| Main conditions | images + robot state | images + language + robot state |
| Semantic backbone | task-specific visual policy | pretrained VLM |
| Action generation | decoder directly predicts chunk | conditional Flow Matching |
| Stochastic mechanism | training-time CVAE latent | noisy action flow during inference |
| Action chunking | yes | yes |
| Temporal Ensemble | ACT deployment design | tested but not used in final π0 |

π0 因此不是 ACT 的语言扩展版。二者共享 action-chunk control idea，但 probabilistic modeling、backbone 与 inference computation 都不同。

## Scope and Limitations

π0 的 generalist training 扩大了任务与 robot coverage，但并不消除 embodiment-specific action interfaces、camera setup、state representation 与 data quality differences。Cross-embodiment transfer 仍依赖 data transforms、shared representation 与 sufficiently compatible tasks。

Flow Matching 需要多次 network evaluation，因而 inference latency 高于单次 direct-regression policy；Action Expert 缩小与 prefix KV caching 都是为降低这一成本服务。

VLM semantic knowledge 也不保证直接得到精确 motor control。Fine-grained manipulation 仍依赖大量 robot trajectories 与 action-level supervision。

## Internal Topics

- [Architecture](/robot-learning/pi0/architecture/)
- [Action Expert](/robot-learning/pi0/action-expert/)
- [Blockwise Causal Attention Mask in π0](/robot-learning/pi0/blockwise-causal-attention-mask/)
- [Flow Matching in π0](/robot-learning/pi0/flow-matching-in-pi0/)
- [Pre-training and Post-training in π0](/robot-learning/pi0/pretraining-and-posttraining/)
- [Training](/robot-learning/pi0/training/)
- [Inference](/robot-learning/pi0/inference/)
- [Complete Data Flow](/robot-learning/pi0/complete-data-flow/)

## Sources

- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. 2024. https://arxiv.org/abs/2410.24164
- Physical Intelligence. *π0: Our First Generalist Policy*. https://www.pi.website/blog/pi0
- Official openpi repository. https://github.com/Physical-Intelligence/openpi
