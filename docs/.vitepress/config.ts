import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import { useKatexMath } from './markdown/katex.mjs'

import { buildKnowledgeModel } from './knowledge-model.mjs'
import { normalizeBase, canonicalRoute, pageHref } from './routes.mjs'

const base = normalizeBase(process.env.BASE_PATH || '/')
const knowledge = buildKnowledgeModel()

export default withMermaid(defineConfig({
  lang: 'zh-CN',
  title: 'XunBlog',
  description: 'AI Knowledge Base',
  base,
  cleanUrls: true,
  lastUpdated: false,
  head: [
    ['meta', { name: 'theme-color', content: '#5b62e8' }],
    ['link', { rel: 'icon', href: `${base}logo.svg`, type: 'image/svg+xml' }]
  ],
  markdown: {
    config(markdown) {
      useKatexMath(markdown)
      markdown.core.ruler.push('knowledge-ui', (state) => {
        for (const token of state.tokens) {
          for (const child of token.children ?? []) {
            if (child.type === 'link_open') {
              const href = child.attrGet('href')
              if (href?.startsWith('/')) canonicalRoute(href.split(/[?#]/)[0])
            }
          }
        }
        // Inject UI metadata after the title without changing Markdown knowledge content.
        const index = state.tokens.findIndex(t => t.type === 'heading_close' && t.tag === 'h1')
        if (index >= 0) {
          const token = new state.Token('html_block', '', 0)
          token.content = '<NoteMeta />\n'
          state.tokens.splice(index + 1, 0, token)
        }
      })
    },
    lineNumbers: true,
    image: { lazyLoading: true },
    theme: {
      light: 'github-light',
      dark: 'github-dark'
    }
  },
  mermaid: {
    theme: 'base',
    themeVariables: {
      primaryColor: '#ececff',
      primaryTextColor: '#252633',
      primaryBorderColor: '#777ee9',
      lineColor: '#777b8d',
      secondaryColor: '#f4f4f1',
      tertiaryColor: '#ffffff',
      fontFamily: 'Inter, ui-sans-serif, sans-serif'
    }
  },
  transformPageData(pageData) {
    pageData.frontmatter.prev = false
    pageData.frontmatter.next = false
    const route = '/' + pageData.relativePath.replace(/index\.md$/, '')
    if (knowledge.nodes[route]) {
      pageData.frontmatter.head ??= []
      pageData.frontmatter.head.push(['link', { rel: 'canonical', href: 'https://seanlii.github.io' + pageHref(route, '/XunBlog/') }])
    }
  },
  themeConfig: {
    knowledge,
    logo: '/logo.svg',
    siteTitle: 'XunBlog',
    nav: [{ text: 'GitHub', link: 'https://github.com/SeanLii/XunBlog' }],
    // VitePress retains its responsive sidebar shell; our persistent tree fills its slot.
    sidebar: Object.fromEntries(knowledge.roots.map(route => [route, [{ text: 'Knowledge Tree', items: [] }]])),
    outline: { level: [2, 3], label: '本页目录' },
    docFooter: { prev: false, next: false },
    returnToTopLabel: '返回顶部',
    sidebarMenuLabel: 'Knowledge Tree',
    darkModeSwitchLabel: '外观',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式'
  }
}))
