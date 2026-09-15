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
  - "/robot-learning/pi0/architecture/"
  - "/robot-learning/pi0/training/"
  - "/robot-learning/pi0/complete-data-flow/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# π0 Inference

训练得到的 π0 网络输出条件速度场，而不是可立即执行的动作。推理必须从随机动作块出发，沿该速度场完成多次数值更新，再把生成结果转换回具体机器人的控制接口。这个过程连接了 [Flow Matching in π0](/robot-learning/pi0/flow-matching-in-pi0/)、Transformer 缓存与 receding-horizon control。

## 固定一次 Policy Query 的条件

控制时刻 $t$ 先采集多路图像、语言指令和 proprioceptive state：

\[
o_t=(I_t^1,\ldots,I_t^n,\ell_t,q_t).
\]

图像与语言经过 VLM expert，state 经过 Action Expert，形成 observation prefix。一次 flow integration 内 $o_t$ 不变，因此 prefix 的 keys 与 values 只计算一次并写入 [KV Cache](/deep-learning/transformer/kv-cache/)。

## 从 Gaussian Action Noise 开始

初始化与输出 action chunk 同 shape 的噪声：

\[
A^{(0)}\sim\mathcal N(0,I),\qquad
A^{(0)}\in\mathbb R^{H\times d_a}.
\]

这不是向确定性回归结果附加扰动，而是生成过程的起点。不同噪声可以在相同 observation 下沿速度场到达不同的合理动作样本，从而保留 demonstration distribution 中的多样性。

## Euler Integration 逐步去噪

设总步数为 $N$，$\Delta\tau=1/N$。论文方向下，第 $i$ 步执行

\[
V^{(i)}=v_\theta(A^{(i)},\tau_i,o_t),
\]

\[
A^{(i+1)}=A^{(i)}+\Delta\tau V^{(i)}.
\]

π0 论文使用 $N=10$。这正是 [Euler Method](/mathematics/numerical-methods/euler-method/) 对条件 ODE 的离散化：步数越少，网络调用越省，但离散误差越大；步数越多，轨迹更细，却增加控制延迟。

released openpi 从 $t=1$ 向 $t=0$ 积分，时间方向与论文相反。实现中的负时间步必须与 `noise → data` 的方向一起理解，不能单独按论文符号替换。

## 每个 Flow Step 重新计算什么

动作样本 $A^{(i)}$ 和时间 embedding 在每一步都变化，所以 action suffix 必须重新经过 Action Expert。observation prefix 不读取 action suffix，缓存仍然有效。计算可分成：

```text
一次：image + language + state → prefix KV
循环：current noisy actions + flow time → vector field → Euler update
```

这解释了为何 architecture 同时采用小型 Action Expert 与 blockwise attention：它们把重复计算限制在动作侧，而较大的视觉语言语义编码不随十个 flow steps 重复。

## 反归一化与执行

积分终点得到模型空间中的 action chunk $\hat A_t$。数据变换随后撤销训练时的 normalization 与 padding，将统一 action interface 映射回当前 embodiment 的物理控制维度：

\[
\hat A_t^{robot}=T_e^{-1}(\hat A_t).
\]

控制器通常只执行 chunk 的前若干步，随后获取新 observation 并再次规划。这样，chunk 内动作具有联合时间结构，chunk 之间仍通过新视觉与状态形成闭环。

## 随机生成与实时控制的张力

噪声初始化和多步积分让策略能够表示多峰动作分布，却带来输出随机性与计算延迟。部署系统需要同时选择 random seed、integration steps、执行前缀长度和 policy query rate。π0 论文曾测试 ACT-style Temporal Ensemble，但最终配置未采用；因此不能把 ACT 的重叠加权直接当作 π0 的默认推理步骤。

## Sources

- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. 2024. https://arxiv.org/abs/2410.24164
- Physical Intelligence. *openpi*, policy and sampling implementation. https://github.com/Physical-Intelligence/openpi
