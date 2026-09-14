# XunBlog UI Revision v1 — implementation and acceptance

本轮只修改 UI、导航、分类入口与路由。53 篇 canonical Markdown 文件已逐字节与上一版提交核对，全部不变。未改动 canonical ownership，未增加知识正文。

## Homepage

XunBlog / AI Knowledge Base → Search → Knowledge Map（可折叠的完整知识树）→ Currently Studying（ACT，仅一个专题区域）→ Recently Updated（Git 历史时间，5 个真实知识页面）→ XunBlog · 2026。

删除旧产品式 Hero、编号领域卡片、线性 Learning Path、Featured ACT 重复区域和品牌宣言。

## Routing / base path

`routes.mjs` 定义无部署前缀、以 `/` 结尾的唯一页面路径。自定义 UI 通过 `useKnowledge().href()` 调用 `pageHref()` 添加一次 base；Markdown 链接通过相同 canonical 验证后由 VitePress 原生 renderer 添加 base。默认开发 base 为 `/`；GitHub Actions 继续注入 `/XunBlog/`。页面输出为目录下 `index.html`，所有生成链接直接使用末尾斜杠，不依赖无斜杠跳转。SEO canonical 使用线上绝对 URL。

`npm run docs:build` 自动先执行 Markdown、导航和公式审计，后执行构建渲染与 HTML 链接审计；任一失败都会阻止原有 GitHub Pages 工作流继续发布。审计覆盖缺失目标、重复 canonical、非法路径、base、锚点、页面与资源，以及线性翻页残留。

## Sidebar / Breadcrumb / Knowledge Relations / Search

- `knowledge-model.mjs` 从原有 frontmatter 与 canonical 文件路径读取 53 篇知识数据，补充 9 个 category 节点；中间分类新增 5 个，顶级分类改造 4 个。
- Sidebar 与 Knowledge Map 共享树组件：独立链接与展开按钮、当前节点高亮、分支缩进、localStorage 保存展开状态。沿用 VitePress 移动 Sidebar 外壳、遮罩与 Esc 行为。
- Breadcrumb 由真实目录祖先生成，每一级可点击；不受上一跳或 graph 关系影响。QKV 始终归属于 Deep Learning / Transformer / Attention。
- 底部关系合并原有 prerequisites、related 与 navigationGraph。Parent Concept 取 canonical 父节点；Used In 来自前置依赖反向边与明确的组件使用边。每个目标只出现一次，空组不显示，禁用 Previous / Next。
- Category Page 包含领域介绍、可选择中心概念的分叉关系图和带说明的 Core Topics，保留跨领域导航；不复制 Sidebar。
- Search 使用 canonical title、aliases 与完整归属路径，可搜 QKV、Gaussian、CVAE 等，显示 breadcrumb；原生 modal dialog 提供焦点约束与 Esc，支持 Cmd/Ctrl+K、上下选择和 Enter。
- 标题下仅保留 Git 更新时间与估算阅读时间，通过 Markdown 渲染插入组件，未修改知识正文。TOC 继续只显示 H2/H3。

## 第 35 节逐项验收

“静态通过”表示源码与生产 HTML 已核对；不冒充真实浏览器交互实测。用户拒绝本地预览服务器权限后要求直接部署，本轮未执行本地浏览器视觉与交互验收。

| 范围 | 验收项 | 结果 |
| --- | --- | --- |
| Homepage | 无产品型 Hero marketing copy | 静态通过 |
| Homepage | 首页直接搜索 / 进入知识 | 控件、索引、目标存在；交互待浏览器实测 |
| Homepage | Knowledge Map 可用 | 全树与链接审计通过；折叠交互待实测 |
| Homepage | 不存在重复 ACT section | 静态通过，仅一个 Currently Studying；树中保留正常 ACT 节点 |
| Homepage | 不存在错误线性路径 | 静态通过 |
| Homepage | Recently Updated 真实页面 | 通过，Git 日期与 canonical 目标校验 |
| Homepage | Current Focus 只出现一次 | 通过，自动断言 |
| Navigation | Sidebar 无章节编号 | 静态通过 |
| Navigation | Breadcrumb 反映 ownership | 通过，所有节点祖先断言 |
| Navigation | Breadcrumb 每一级可点击 | 通过，所有祖先目标存在 |
| Navigation | 底部无上一篇 / 下一篇主导航 | 通过，生产 HTML 审计 |
| Navigation | Knowledge Relations 可点击 | 通过，图目标与生产链接审计 |
| Routing | 首页所有 link 无 404 | 通过，构建目标审计 |
| Routing | Normal Distribution 可访问 | 对应 index.html 与链接存在 |
| Routing | ACT 可访问 | 对应 index.html 与链接存在 |
| Routing | π0 可访问 | 未实现：现有语料无 π0，本轮禁止新增知识；未创建占位或错误重定向 |
| Routing | Projects / About 不会 404 | 采用规范方案 A，导航无此类入口 |
| Routing | trailing slash 统一 | 通过，全部生成链接以 `/` 结尾 |
| Routing | GitHub Pages base 正确 | `/` 与 `/XunBlog/` 构建及目标审计 |
| Routing | internal link audit 通过 | 见下方自动检查结果 |
| Category | 不只是 Sidebar 复制 | 静态通过，概念关系图 + 主题说明 |
| Category | 真实 Knowledge Structure | 通过，沿用现有知识关系并区分关系类型 |
| Category | 无空 section | 数据断言与条件渲染通过 |
| Category | 无虚假固定学习顺序 | 静态通过，分叉关系无课程编号 |
| Visual | 技术感保留 | 原中性色、深浅色、排版基础保留；视觉待实测 |
| Visual | 营销感降低 | 旧宣传内容已删除；视觉待实测 |
| Visual | 卡片数量减少 | 旧领域/路径/Featured 卡片删除；视觉待实测 |
| Visual | 装饰英文标签减少 | 删除 LEARNING LOG、FEATURED、Explore 等；视觉待实测 |
| Visual | 正文仍为中心 | 760px 阅读列、克制标题已实现；视觉待实测 |
| Visual | mobile 无 overflow | 已实现单列、折叠导航、滚动代码/公式；未实测，不能标为通过 |
| Visual | code / math 正常 | 383 个块公式、370 个行内公式及 KaTeX CSS 渲染审计通过；浏览器视觉待实测 |

## 自动检查结果

- 53 个 canonical 页面与 247 个 Markdown 内部链接通过。
- 62 个导航节点、459 个树/关系目标通过，含去重、归属、日期与非法路由拒绝。
- 64 个生产 HTML 页面、6,679 个内部页面/资源引用通过（含首页、Sidebar、Breadcrumb、Category、底部关系）。
- 383 个块公式、370 个行内公式及 KaTeX CSS 通过。
- canonical 文件逐字节保持不变；没有删除任何知识页面。

## 未完成项 / 冲突

1. π0 验收条目与“现有 corpus 不含 π0”及“本轮不扩展知识内容”冲突。遵循内容边界，不自行新增。
2. 真实浏览器中的桌面/移动端视觉、横向溢出、暗色、搜索快捷键与折叠持久化尚未实测。本地服务权限被拒绝后按用户要求直接部署。
3. 项目原有 VitePress/Mermaid 大 chunk 构建提示仍存在，不影响本次路由或构建通过。

## 文件清单

以下按本轮变更自动生成。没有删除文件。

### 修改文件

- `docs/.vitepress/config.ts`
- `docs/.vitepress/theme/components/HomePage.vue`
- `docs/.vitepress/theme/components/NoteMeta.vue`
- `docs/.vitepress/theme/index.ts`
- `docs/.vitepress/theme/styles.css`
- `docs/deep-learning/index.md`
- `docs/generative-models/index.md`
- `docs/index.md`
- `docs/mathematics/index.md`
- `docs/robot-learning/index.md`
- `package.json`

### 新增文件

- `UI_REVISION_ACCEPTANCE.md`
- `docs/.vitepress/knowledge-model.mjs`
- `docs/.vitepress/routes.mjs`
- `docs/.vitepress/theme/components/Breadcrumb.vue`
- `docs/.vitepress/theme/components/CategoryPage.vue`
- `docs/.vitepress/theme/components/KnowledgeRelations.vue`
- `docs/.vitepress/theme/components/KnowledgeSearch.vue`
- `docs/.vitepress/theme/components/KnowledgeSidebar.vue`
- `docs/.vitepress/theme/components/KnowledgeTree.vue`
- `docs/.vitepress/theme/components/SearchButton.vue`
- `docs/.vitepress/theme/composables/knowledge.ts`
- `docs/deep-learning/cnn/index.md`
- `docs/deep-learning/core/index.md`
- `docs/mathematics/information-theory/index.md`
- `docs/mathematics/linear-algebra/index.md`
- `docs/mathematics/probability/index.md`
- `scripts/audit-built-links.mjs`
- `scripts/audit-navigation.mjs`

### 删除文件

无。
