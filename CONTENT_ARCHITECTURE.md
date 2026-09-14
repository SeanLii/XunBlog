# XunBlog Knowledge Content Architecture

这份文件记录当前站点的内容边界与工程规则，不作为公开知识正文。

## Scope

当前内容覆盖 ACT、π0 以及理解它们所需的前置知识，共 82 个 canonical knowledge pages：

完整 v4 归属树见 `research/corpus-v4/KNOWLEDGE_TREE.md`。Attention、Sequence Modeling、BERT、DETR 与 Representation Learning 为独立分支；变分推断归属 Probability；Residual Connection 归属 ResNet。

不在当前 corpus 中的 LLM、Embodied AI、Projects、π0-FAST、π0.5 等栏目不创建占位页面。

## Knowledge ownership

- Knowledge Tree 决定知识归属；Navigation Graph 只建立跨树导航。
- 每个独立概念只有一个 canonical page。
- canonical page 的标题使用概念名称；问题型标题只用于模型-specific 的具体问题。
- ACT 页面只拥有 ACT-specific 内容，不复制 Transformer、CVAE、Gaussian、KL、Behavior Cloning 或 ResNet 的通用理论。

## URL and frontmatter contract

知识页面使用与目录结构一致、带末尾 `/` 的 canonical URL：

```yaml
---
title: "Query / Key / Value"
kind: "canonical"
domain: "Deep Learning / Transformer / Attention"
parent: "Attention"
canonical: "/deep-learning/transformer/attention/qkv/"
prerequisites:
  - "/mathematics/linear-algebra/dot-product/"
related:
  - "/deep-learning/transformer/attention/self-attention/"
---
```

VitePress 页面放在 canonical 目录的 `index.md` 中。`prerequisites` 与 `related` 会自动显示在页面底部的 Knowledge Relations；补充的跨树关系由 `docs/.vitepress/knowledge-graph.mjs` 提供。

## Source boundary

- 模型与算法结论以原论文为主要依据。
- released implementation 的具体行为以官方代码为主要依据。
- README、issue 或二手材料只补充实现语境，不能覆盖原论文或官方实现。
- 当 paper 与 code 不一致时，两边均保留，不静默合并为单一版本。

## Engineering rules

- 站内 Markdown 链接使用 canonical absolute path，不手写部署 base path。
- `BASE_PATH` 在构建时注入，GitHub Pages 当前使用 `/XunBlog/`。
- `\[ ... \]` 与 `$ ... $` 由本地 KaTeX 插件进行服务端渲染。
- `npm run docs:audit` 必须验证 82 个 canonical 页面、内部链接、frontmatter 关系和 Navigation Graph。
- `npm run docs:audit:math`、`npm run docs:build` 与 `npm run docs:audit:render` 必须全部通过后才能部署。
