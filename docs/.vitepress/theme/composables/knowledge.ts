import { computed } from 'vue'
import { useData } from 'vitepress'
import { pageHref, routeFromPath } from '../../routes.mjs'
export function useKnowledge() {
  const { theme, site, page } = useData()
  const model = computed(() => theme.value.knowledge)
  const route = computed(() => routeFromPath('/' + page.value.relativePath.replace(/index\.md$/, '').replace(/\.md$/, ''), '/'))
  const current = computed(() => model.value.nodes[route.value])
  const href = (route: string) => pageHref(route, site.value.base)
  return { model, current, route, href }
}
export function formatUpdated(value: string) {
  return new Intl.DateTimeFormat('en', { month: 'short', year: 'numeric', timeZone: 'UTC' }).format(new Date(value))
}
