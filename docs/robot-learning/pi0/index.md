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
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# π0

π0（pi-zero）是 Physical Intelligence 提出的 Vision-Language-Action robot policy。它把预训练视觉语言模型的语义能力、大规模跨机器人数据与连续动作生成结合起来：图像和语言由 VLM expert 处理，机器人状态与动作由较小的 Action Expert 处理，conditional Flow Matching 则把 Gaussian noise 逐步运输成一段连续动作。

## 策略的条件与输出

时刻 $t$ 的 observation 为

\[
o_t=(I_t^1,\ldots,I_t^n,\ell_t,q_t),
\]

其中 $I_t^i$ 是相机图像，$\ell_t$ 是语言指令，$q_t$ 是机器人状态。策略建模未来动作块的条件分布：

\[
p(A_t\mid o_t),
\qquad
A_t=(a_t,\ldots,a_{t+H-1}).
\]

论文取 $H=50$。与单次输出确定性 chunk 的回归策略不同，π0 从随机动作开始，经多次 vector-field prediction 得到 $A_t$。因此它既是 [Vision-Language-Action Model](/robot-learning/vision-language-action-model/)，也是一个以连续生成过程实现的 action-chunk policy。

## VLM 提供语义，但不直接提供控制

预训练 [Vision-Language Model](/deep-learning/multimodal/vision-language-model/) 已学习图像区域、物体概念和语言描述之间的对应关系。这使“拿起红色杯子”中的对象与属性能够落到视觉 token 上。但 VLM 的原始接口没有连续关节状态、动作维度或控制频率，也没有学习接触动力学。

π0 因而保留 image-language token 的 VLM 参数，同时为机器人 state 和 action token 引入新参数。语义复用降低了从机器人数据重新学习概念的负担；精确运动仍由轨迹监督建立。

## 双 Expert 在同一 Attention 中协作

[π0 Architecture](/robot-learning/pi0/architecture/) 把序列分为三块：

```text
[ image + language ] | [ robot state ] | [ noisy action chunk + flow time ]
      VLM expert            Action Expert              Action Expert
```

这不是“VLM 完成推理后再调用控制器”的串联结构。token 在 Transformer layers 中共同参与 attention，只是按模态路由到不同的参数集合。[Action Expert](/robot-learning/pi0/action-expert/) 因而能在每层读取视觉语言条件，并把信息逐步写入动作表示。

## Blockwise Mask 同时服务联合生成与缓存

三块 token 的读取方向由 [Blockwise Causal Attention Mask](/robot-learning/pi0/blockwise-causal-attention-mask/) 约束：image-language prefix 不读 state/action；state 读取 semantic prefix 但不读 action；action 读取全部 observation，并在 action block 内双向读取。

由此产生两个性质。第一，整段 action chunk 在每个 flow step 并行、联合更新，而不是 autoregressive 地逐动作生成。第二，固定 observation prefix 不依赖正在变化的 action，可以在一次生成过程中缓存 keys 与 values，避免每个积分步都重算大型 VLM。

## Flow Matching 把噪声运输成动作

对真实动作块 $A$ 与同 shape 噪声 $\epsilon\sim\mathcal N(0,I)$，论文定义线性 path：

\[
A^\tau=\tau A+(1-\tau)\epsilon,
\qquad \tau\in[0,1].
\]

其条件速度为

\[
\frac{dA^\tau}{d\tau}=A-\epsilon.
\]

[Flow Matching in π0](/robot-learning/pi0/flow-matching-in-pi0/) 训练网络预测

\[
v_\theta(A^\tau,\tau,o_t)\approx A-\epsilon.
\]

网络输出不是动作概率或最终动作，而是当前动作样本在路径上应移动的方向。由于 $v_\theta$ 以 observation 为条件，同一个 noise sample 会在不同图像、语言和机器人状态下沿不同轨迹演化。

## Training 直接构造中间点

[π0 Training](/robot-learning/pi0/training/) 从 demonstration trajectory 截取 $(o_t,A_t)$，采样 $\epsilon$ 与 $\tau$，再最小化

\[
\mathcal L(\theta)=
\mathbb E\left[
\left\|v_\theta(A_t^\tau,\tau,o_t)-(A_t-\epsilon)\right\|_2^2
\right].
\]

训练已知真实 $A_t$，所以不必从 noise 端逐步积分到 data 端；每个样本可直接跳到随机 $\tau$ 构造监督。机器人数据还要经过 embodiment-specific normalization、padding 和有效维度 mask，才能进入统一接口。

## Inference 用数值积分恢复 Action Chunk

部署时没有真实动作，先采样

\[
A^{(0)}\sim\mathcal N(0,I),
\]

再用 Euler method 重复更新：

\[
A^{(i+1)}
=
A^{(i)}+\Delta\tau\,
v_\theta(A^{(i)},\tau_i,o_t).
\]

论文使用 10 个 integration steps。[π0 Inference](/robot-learning/pi0/inference/) 在循环前缓存 observation prefix，在循环内主要重算 action suffix。生成完成后撤销 normalization，将统一动作维度映射回当前机器人，并执行 chunk 的前一部分；随后重新观测并规划，形成 receding-horizon closed loop。

## Pre-training 与 Post-training 决定支持域

π0 的 generalist behavior 来自两层知识：VLM 的 Internet-scale image-language pretraining 提供语义；大规模、多任务、多 embodiment robot data 建立语义到动作的映射。[Pre-training and Post-training in π0](/robot-learning/pi0/pretraining-and-posttraining/) 先在宽分布上学习共享能力，再用更集中、高质量的数据提高困难任务表现。

这套方法扩大了任务与机器人覆盖，却不消除 embodiment 差异。相机视角、state definition、action convention、控制频率和接触动力学若超出训练支持域，语义理解也不能保证 motor control 正确。

## 与 ACT 的共同点与分界

π0 与 ACT 都预测 future action chunks，并在执行若干动作后重新观测。共同点到此为止：ACT 使用 task-specific visual backbone、Transformer decoder direct regression 和 training-only CVAE latent；π0 使用 pretrained VLM、双 expert Transformer，以及 inference-time iterative Flow Matching。

π0 因此不是“给 ACT 加语言”。[Action Chunking](/robot-learning/act/action-chunking/) 只是二者共享的控制时间结构；语义 backbone、概率建模方式、训练目标和推理计算图都已改变。Flow Matching 提供更丰富的条件动作分布，也以多次网络评估换取表达能力。

## Sources

- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. 2024. https://arxiv.org/abs/2410.24164
- Physical Intelligence. *π0: Our First Generalist Policy*. https://www.pi.website/blog/pi0
- Physical Intelligence. *openpi*. https://github.com/Physical-Intelligence/openpi
