# XunBlog Knowledge Corpus — v3

v3 保留 v2 已经重写并验收通过的 ACT 知识体系，在同一套 canonical knowledge system 中新增 **π0** 板块，以及理解 π0 必须的通用前置知识。

## 本版新增内容

新增 16 个页面，整个 corpus 从 53 页扩展到 **69 页**。

### 新增通用 canonical pages

- `Ordinary Differential Equation`
- `Euler Method`
- `Vision-Language Model`
- `Flow Matching`
- `Vision-Language-Action Model`
- `Cross-Embodiment Learning`

这些知识不属于 π0。π0 只通过 canonical links 使用它们。

### 新增 π0 pages

- `π0`
- `Architecture`
- `Action Expert`
- `Blockwise Causal Attention Mask`
- `Flow Matching in π0`
- `Pre-training and Post-training in π0`
- `Training`
- `Inference`
- `Complete Data Flow`
- `Paper and Released Implementation`

## π0 的理解路径

π0 总览不会从论文术语直接开始，而是先建立：

```text
camera images + language + robot state
                ↓
               π0
                ↓
      future continuous action chunk
```

再逐层加入：

```text
pre-trained VLM
→ Action Expert
→ noisy action chunk
→ Flow Matching vector field
→ 10-step integration
→ final action chunk
→ partial open-loop execution
→ re-observe and infer again
```

论文只负责事实校验，不决定文章讲解顺序。

## 一个重要的 ACT / π0 区别

两者都使用 action chunking，但执行方式不同：ACT 以 Temporal Ensemble 融合 overlapping chunks；π0 论文报告早期尝试 temporal ensembling 后 performance 下降，因此最终执行 action chunk 的一部分后重新 inference，不进行 ACT-style temporal aggregation。

## 文件

- `KNOWLEDGE_TREE.md` — canonical ownership / sidebar hierarchy
- `LINK_GRAPH.md` — navigation graph
- `CONTENT_INDEX.md` — 69 个页面索引
- `SOURCE_MAP.md` — 主要论文、官方代码与页面依据
- `CONTENT_DESIGN_STANDARD.md` — 理解层级写作标准
- `QUALITY_AUDIT.md` — 自动结构检查与人工内容验收
- `content/` — 所有 Markdown 页面

## 当前范围

当前覆盖：

```text
ACT + ACT prerequisites
π0 + π0 prerequisites
```

π0-FAST 与 π0.5 当前没有并入 π0 页面。它们是后续独立模型，如果继续扩展应建立自己的 canonical pages，而不是把后续 architecture 反写进原始 π0。
