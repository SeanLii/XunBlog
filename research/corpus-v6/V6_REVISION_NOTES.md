# XunBlog Knowledge v6 — Revision Notes

v6 是在 v5 全量 82 页重审基础上的 **precision release**。本轮不改变知识树、canonical routes 或页面总数，重点把仍可能影响严谨性的边界写清楚。

## Content changes

### 1. Softmax

修正低 temperature 极限的条件：只有唯一最大 logit 时 \(T\to0^+\) 才收敛到唯一 one-hot；存在并列最大值时，概率质量保留在最大值集合并平均分配。

### 2. Cross-Entropy

加入 continuous density case：

\[
H(p,q)=-\int p(x)\log q(x)\,dx,
\]

并区分 discrete entropy \(H(p)\) 与 differential entropy \(h(p)\)，补充 \(H(p,q)=h(p)+D_{KL}(p\|q)\) 及 support / absolute-continuity 条件。

### 3. DETR Object Query

把 architecture-level shorthand 与原始官方实现分开：learned object queries 在官方 DETR 中作为 `query_pos`，decoder content state 从 zero tensor 开始；进一步区分 object-query embedding 与 attention projection 中的 \(Q\)。

### 4. ACT Temporal Ensemble

明确 ACT 论文的索引约定：\(w_0\) 对应 oldest prediction。官方 released code 的 `all_time_actions[:, t]` 顺序与该约定一致。

### 5. Flow Matching

明确三个对象：sample-level trajectory velocity、conditional vector field、marginal vector field。\(x_1-x_0\) 只属于特定 linear path 的 sample-level target，不是所有 Flow Matching 的一般 vector-field 定义。

### 6. Vector Norm

补全 finite-dimensional norm equivalence 的实际数学含义：不同 norms 诱导相同 convergence / continuity topology，但仍可产生不同 unit-ball geometry、loss geometry 与 optimization behavior。

## Standard changes

`CONTENT_DESIGN_STANDARD.md` 新增 v6 三轴要求：

1. **逻辑严谨**：条件、定义、结论、边界和来源不能混淆；
2. **数学性**：数学性由关系是否正式建立决定，不由公式数量决定；
3. **解释性**：核心定义 / 公式 / 机制必须解释对象、作用、变量、意义和边界。

统一验收闭环：

> **定义准确 → 数学建立关系 → 解释说明意义 → 条件与边界收口。**

## Compatibility

- canonical routes：不变；
- knowledge tree：不变；
- frontmatter schema：不变；
- Codex 导入方式：不变；
- 页面数量：不变。
