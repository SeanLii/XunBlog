---
title: "Blockwise Causal Attention Mask in π0"
kind: "pi0-topic"
domain: "Robot Learning / π0"
parent: "π0"
canonical: "/robot-learning/pi0/blockwise-causal-attention-mask/"
prerequisites:
  - "/deep-learning/sequence-modeling/causal-mask/"
  - "/deep-learning/attention/self-attention/"
  - "/robot-learning/pi0/architecture/"
related:
  - "/deep-learning/transformer/kv-cache/"
  - "/robot-learning/pi0/inference/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Blockwise Causal Attention Mask in π0

π0 的序列包含 image-language、robot state 与 action 三类 token。普通 causal mask 会禁止动作位置读取其后的动作，完全双向的 mask 又会让固定 observation 依赖每一步都在变化的 noisy action。[π0 Architecture](/robot-learning/pi0/architecture/) 因而采用 blockwise mask：依赖关系按功能块定义，而不是简单按 token 下标定义。

## 三个 Block 的依赖目标

将输入写成

\[
X=[X_{vl}\mid X_s\mid X_a],
\]

其中 $X_{vl}$ 是 image-language prefix，$X_s$ 是 state block，$X_a$ 是长度为 $H$ 的 action block。模型需要同时满足：语义 prefix 可独立编码；state 读取任务语境；action 读取全部条件；动作位置彼此联合建模。

对应的可见性矩阵为

\[
M=
\begin{bmatrix}
0 & -\infty & -\infty\\
0 & 0 & -\infty\\
0 & 0 & 0
\end{bmatrix},
\]

每个元素代表 query block 是否允许读取 key block。attention logits 加上 $M$ 后再进入 Softmax：

\[
\operatorname{Attn}(Q,K,V)
=
\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.
\]

## Prefix 不被动作反向影响

$X_{vl}$ 只读取自身，$X_s$ 可以读取 $X_{vl}$ 与自身；两者都不能读取 $X_a$。这样固定 observation 的 hidden states 不会随着 noisy action 的迭代更新而改变。

这一方向性与普通 [Causal Mask](/deep-learning/sequence-modeling/causal-mask/) 的目的不同。语言模型用它阻止未来词泄漏；π0 用它隔离固定条件与生成变量，从结构上建立可缓存的 prefix。

## Action Block 内部保持双向

每个 action query 可以读取全部 observation，也可以读取 chunk 中任意动作位置。于是第 $i$ 个动作的更新能够参考第 $i-1$ 与第 $i+1$ 个动作：

\[
h_i^a=f(a_1^\tau,\ldots,a_H^\tau,o_t).
\]

这与 autoregressive action decoding 有根本差别。π0 在一个 flow step 中并行更新整段轨迹，不需要等待前一个动作生成完才计算下一个动作。由此得到更强的 chunk-level temporal coupling，同时也意味着输出长度 $H$ 在模型接口中预先确定。

## Mask 使 KV Cache 成立

flow integration 会多次改变 $X_a$，而 $X_{vl}$ 与 $X_s$ 在一次 policy query 内保持不变。因为 prefix 不读取 suffix，其 keys 与 values 可以先计算一次并放入 [KV Cache](/deep-learning/transformer/kv-cache/)；之后只追加并重算 action-side 表示。

如果允许 prefix 读取 action，缓存会失效，因为每次 action update 都会改变 prefix hidden states。因而缓存不是独立的工程技巧，而是由 attention dependency 直接产生的性质。

## 信息约束带来的边界

state block 不能读取 action，使它保持纯条件表示，但也限制了在同一层中由候选动作反向修正 state representation 的可能性。π0 选择这一约束，是为了得到稳定条件与高效迭代之间的平衡。后续模型若改变 mask，就同时改变了语义依赖、并行生成方式与缓存成本，不能只把它视作实现细节。

## Sources

- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. 2024, Appendix B. https://arxiv.org/abs/2410.24164
- Physical Intelligence. *openpi*, attention mask construction. https://github.com/Physical-Intelligence/openpi
