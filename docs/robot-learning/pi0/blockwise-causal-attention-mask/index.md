---
title: "Blockwise Causal Attention Mask"
kind: "pi0-topic"
domain: "Robot Learning / π0"
parent: "π0"
canonical: "/robot-learning/pi0/blockwise-causal-attention-mask/"
prerequisites:
  - "/deep-learning/transformer/causal-mask/"
  - "/deep-learning/transformer/attention/self-attention/"
  - "/robot-learning/pi0/architecture/"
related:
  - "/robot-learning/pi0/inference/"
  - "/robot-learning/pi0/action-expert/"
---

# Blockwise Causal Attention Mask

π0 不让所有 token 无限制地互相 attention。它把输入分成三个 block，并规定每个 block 可以读取哪些信息：

```text
Block 1              Block 2       Block 3
images + language | robot state | noisy actions
```

核心规则可以画成：

```text
                    Can attend to
               B1       B2       B3
B1 image/lang   ✓        ✗        ✗
B2 state        ✓        ✓        ✗
B3 actions      ✓        ✓        ✓
```

同时，每个 block 内部是 full bidirectional attention。

## 它不是普通逐 token causal mask

语言模型常见 causal mask 是：第 $i$ 个 token 只能看自己和左边 token。

π0 的规则不同。它按**功能 block**控制信息：

- image + language block 内部可以双向交互；
- state block 可以看前面的 image/language；
- action block 可以看所有 observation，并且所有 action positions 彼此双向读取。

因此 action chunk 的第 30 个位置可以直接读取第 5 个 action position，而不是必须按时间顺序 autoregressive 生成。

## Block 1 为什么不能读取 state 和 action

Block 1 对应 PaliGemma VLM pre-training 中已经存在的 modalities：images 与 language。

论文刻意阻止这部分 token 读取后面新增的 robot state / action tokens，目的之一是尽量减少相对 VLM pre-training 分布的结构变化。

可以理解成：

```text
原来的 VLM 世界
images ↔ language

π0 新增机器人信息后
仍保持 image/language 的原有读取模式
```

而后面的 robotics-specific tokens 可以读取 VLM representations。

## State 为什么单独成为 Block 2

Robot state $q_t$ 在一次 flow matching inference 中是不变的。

真正反复变化的是 noisy action chunk：

\[
A_t^{\tau_0},
A_t^{\tau_1},
\ldots
\]

如果 state token 读取 action tokens，那么每次 action 改变以后，state 的 attention output 也需要重新计算。

π0 阻止 state 读取 action，因此 image/language/state 相关的 keys / values 可以缓存，重复 integration 时只更新 action suffix。

所以这个 mask 不只是“语义上的因果关系”，也是 inference efficiency 设计。

## Action block 为什么可以双向读取

π0 一次生成整个 action chunk：

\[
A_t=[a_t,\ldots,a_{t+H-1}].
\]

它不是逐 action autoregressive decoding。

因此 action block 内部可以使用 full bidirectional attention：

```text
a_t     ↔ a_t+1 ↔ a_t+2 ↔ ... ↔ a_t+H-1
```

这样整个未来动作片段可以共同决定一致的轨迹结构。

这与 [Action Chunking](/robot-learning/act/action-chunking/) 的思想非常契合：chunk 本身就是一个整体预测对象。

## 对 inference 的直接影响

一次 π0 inference 可以分为：

1. 先对 observation prefix 做一次 forward；
2. 保存它的 KV cache；
3. 每个 flow step 只重新计算新的 action suffix；
4. action suffix 使用缓存的 prefix keys / values。

因此 mask 与 KV caching 是同一套结构设计的两个方面。

## Sources

- Black et al., **π0: A Vision-Language-Action Flow Model for General Robot Control**, Appendix B. https://arxiv.org/abs/2410.24164
- Official openpi `pi0.py`. https://github.com/Physical-Intelligence/openpi
