---
title: "Architecture"
kind: "pi0-topic"
domain: "Robot Learning / π0"
parent: "π0"
canonical: "/robot-learning/pi0/architecture/"
prerequisites:
  - "/robot-learning/pi0/"
  - "/deep-learning/multimodal/vision-language-model/"
  - "/deep-learning/attention/self-attention/"
related:
  - "/robot-learning/pi0/action-expert/"
  - "/robot-learning/pi0/blockwise-causal-attention-mask/"
  - "/robot-learning/pi0/complete-data-flow/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# π0 Architecture

[π0](/robot-learning/pi0/) 需要在一个网络中联合处理相机图像、语言指令、机器人状态和带噪动作。前三者描述条件，最后一项是被逐步生成的对象。架构的关键不是简单拼接四种输入，而是让预训练视觉语言参数与机器人专用参数在受控的信息流中协作。

## 四种输入形成三个功能块

时刻 $t$ 的条件可写为

\[
o_t=(I_t^1,\ldots,I_t^n,\ell_t,q_t),
\]

生成中的动作块为

\[
A_t^\tau=(a_t^\tau,\ldots,a_{t+H-1}^\tau).
\]

进入 Transformer 后，它们被组织成：

```text
[ image + language ] | [ robot state ] | [ noisy actions + flow time ]
   semantic prefix        state prefix          generated suffix
```

图像与语言共同描述任务语境；state 将语义落到当前机器人构型；action suffix 表示正在从噪声演化为动作的整段轨迹。这个划分随后决定参数路由、注意力可见性与缓存边界。

## 图像与语言保留 VLM 接口

每路 RGB 图像先经 PaliGemma 所使用的视觉编码器转换为 visual tokens，语言指令 $ell_t$ 经 tokenizer 与 embedding 进入相同的语义空间。[Vision-Language Model](/deep-learning/multimodal/vision-language-model/) 的预训练参数继续处理这些 token，因此物体、场景与指令之间的对应关系不必仅靠机器人轨迹重新学习。

这一复用只提供语义表征，并不直接给出连续控制。VLM 原有输出空间面向语言或多模态预测，动作生成仍需要新的输入投影、参数与目标函数。

## 状态与动作进入 Action Expert

连续状态 $q_t$ 经 learned projection 变成 state token。每个带噪动作 $a_i^\tau$ 则与 flow-time embedding 组合成 action token：

\[
e_i^a=g_\phi(W_a a_i^\tau,\varphi(\tau)).
\]

这些 token 由 [Action Expert](/robot-learning/pi0/action-expert/) 的参数处理。VLM expert 和 Action Expert 不是串联的两个网络；在每个 Transformer layer 中，token 按类型选用不同参数，同时通过注意力交换信息。

## Blockwise Mask 规定依赖方向

[Blockwise Causal Attention Mask](/robot-learning/pi0/blockwise-causal-attention-mask/) 把允许的信息流写入 attention matrix：

- image-language prefix 只依赖自身；
- state 可以读取 image-language prefix 与自身；
- action 可以读取全部 observation token，并在 action block 内双向交互。

因此，未来动作能够条件化于任务与现场观测，而固定条件不会被正在变化的动作噪声反向污染。动作位置之间的双向注意力又使整段 action chunk 被联合生成，不需要按时间位置 autoregressive decoding。

## Hidden State 被投影为条件速度场

最后一层得到动作位置表示

\[
H_a^{(L)}\in\mathbb R^{H\times d_h}.
\]

线性 head 输出同 shape 的动作速度场：

\[
v_\theta(A_t^\tau,o_t)=H_a^{(L)}W_o+b_o
\in\mathbb R^{H\times d_a}.
\]

这里的输出还不是最终 action chunk。[Flow Matching](/generative-models/flow-matching/) 的数值积分用它更新 $A_t^\tau$，重复若干次后才到达数据端。

## 结构同时创造了缓存边界

由于 observation prefix 不依赖 action suffix，同一次动作生成期间它的 keys 与 values 保持不变，可以由 [KV Cache](/deep-learning/transformer/kv-cache/) 复用。变化的主要是 action token 及其经较小 Action Expert 得到的表示。

这项效率来自因果结构本身：如果 prefix 能读取 action，动作每更新一次就必须重算全部视觉语言 token，较大的 VLM 会成为迭代生成的主要成本。相反，当前设计把大规模语义编码放在一次性计算中，把重复计算集中在较小的机器人 expert 上。

## 模型规模不是架构定义

论文使用 PaliGemma-based VLM 与约 300M 参数的 Action Expert，action horizon 为 $H=50$。released openpi 中的 `gemma_2b`、`gemma_300m` 和统一 32 维 action interface 是具体配置；替换宽度、层数或 padding 上限不会改变三块输入、双 expert 路由和 conditional vector-field output 这三项核心关系。

## Sources

- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. 2024, especially Appendix B. https://arxiv.org/abs/2410.24164
- Physical Intelligence. *openpi*. https://github.com/Physical-Intelligence/openpi
