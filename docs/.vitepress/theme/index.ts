import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import HomePage from './components/HomePage.vue'
import NoteMeta from './components/NoteMeta.vue'
import 'katex/dist/katex.min.css'
import './styles.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('HomePage', HomePage)
    app.component('NoteMeta', NoteMeta)
  }
} satisfies Theme
