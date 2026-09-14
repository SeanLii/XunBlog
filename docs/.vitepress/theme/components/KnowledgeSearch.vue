<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRouter } from 'vitepress'
import { useKnowledge } from '../composables/knowledge'
const { model, href } = useKnowledge()
const router = useRouter()
const dialog = ref<HTMLDialogElement>()
const input = ref<HTMLInputElement>()
const query = ref('')
const active = ref(0)
let opener: HTMLElement | null = null
const results = computed(() => {
  const terms = query.value.trim().toLowerCase().split(/\s+/).filter(Boolean)
  if (!terms.length) return []
  return Object.values(model.value.nodes).filter((n: any) => n.kind === 'canonical' && terms.every(t => n.searchText.includes(t)))
    .sort((a: any, b: any) => {
      const score = (n: any) => n.title.toLowerCase() === query.value.toLowerCase() || n.aliases.some((s: string) => s.toLowerCase() === query.value.toLowerCase()) ? 0 : 1
      return score(a) - score(b) || a.title.localeCompare(b.title)
    }) as any[]
})
watch(query, () => { active.value = 0 })
async function open() {
  if (dialog.value?.open) return
  opener = document.activeElement as HTMLElement
  query.value = ''; active.value = 0
  dialog.value?.showModal()
  document.documentElement.classList.add('search-open')
  await nextTick(); input.value?.focus()
}
function close() { dialog.value?.close() }
function closed() { document.documentElement.classList.remove('search-open'); opener?.focus() }
function choose(route: string) { close(); router.go(href(route)) }
function globalKey(e: KeyboardEvent) {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); dialog.value?.open ? close() : open() }
}
async function navigate(e: KeyboardEvent) {
  if (e.isComposing) return
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    e.preventDefault()
    if (!results.value.length) return
    active.value = (active.value + (e.key === 'ArrowDown' ? 1 : -1) + results.value.length) % results.value.length
    await nextTick(); document.getElementById(`knowledge-result-${active.value}`)?.scrollIntoView({ block: 'nearest' })
  } else if (e.key === 'Enter' && results.value[active.value]) { e.preventDefault(); choose(results.value[active.value].route) }
}
onMounted(() => { window.addEventListener('xunblog:search', open); window.addEventListener('keydown', globalKey) })
onUnmounted(() => { window.removeEventListener('xunblog:search', open); window.removeEventListener('keydown', globalKey); document.documentElement.classList.remove('search-open') })
</script>
<template>
  <dialog ref="dialog" class="knowledge-search" aria-labelledby="search-title" @close="closed" @click="e => { if (e.target === dialog) close() }">
    <div class="search-dialog-inner">
      <div class="search-heading"><h2 id="search-title">Search knowledge</h2><button aria-label="关闭搜索" @click="close">Esc</button></div>
      <input ref="input" v-model="query" type="search" placeholder="Search knowledge..." aria-label="Search knowledge" role="combobox" aria-autocomplete="list" aria-controls="knowledge-results" :aria-expanded="results.length > 0" :aria-activedescendant="results.length ? `knowledge-result-${active}` : undefined" @keydown="navigate" />
      <p class="search-status" role="status">{{ query.trim() ? `${results.length} 个结果` : '搜索概念名称或别名，例如 QKV、Gaussian、CVAE。' }}</p>
      <ul id="knowledge-results" role="listbox" aria-label="搜索结果">
        <li v-for="(node, index) in results" :id="`knowledge-result-${index}`" :key="node.route" role="option" :aria-selected="index === active" @mousemove="active = index">
          <a :href="href(node.route)" tabindex="-1" @click.prevent="choose(node.route)"><strong>{{ node.title }}</strong><small>{{ node.breadcrumbs.slice(0, -1).map(r => model.nodes[r].title).join(' / ') }}</small></a>
        </li>
      </ul>
      <p v-if="query.trim() && !results.length" class="search-empty">没有找到相关知识。试试其他概念名称或别名。</p>
      <div class="search-help">↑ ↓ 选择 · Enter 打开 · Esc 关闭</div>
    </div>
  </dialog>
</template>
