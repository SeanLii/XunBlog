import { h } from 'vue'
import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import HomePage from './components/HomePage.vue'
import NoteMeta from './components/NoteMeta.vue'
import Breadcrumb from './components/Breadcrumb.vue'
import KnowledgeRelations from './components/KnowledgeRelations.vue'
import KnowledgeSidebar from './components/KnowledgeSidebar.vue'
import KnowledgeSearch from './components/KnowledgeSearch.vue'
import SearchButton from './components/SearchButton.vue'
import CategoryPage from './components/CategoryPage.vue'
import 'katex/dist/katex.min.css'
import './styles.css'

export default {
  extends: DefaultTheme,
  Layout: () => h(DefaultTheme.Layout, null, {
    'layout-top': () => h(KnowledgeSearch),
    'nav-bar-content-before': () => h(SearchButton),
    'sidebar-nav-before': () => h(KnowledgeSidebar),
    'doc-before': () => h(Breadcrumb),
    'doc-after': () => h(KnowledgeRelations),
    'layout-bottom': () => h('footer', { class: 'site-footer' }, 'XunBlog · 2026')
  }),
  enhanceApp({ app }) {
    app.component('HomePage', HomePage)
    app.component('NoteMeta', NoteMeta)
    app.component('CategoryPage', CategoryPage)
  }
} satisfies Theme
