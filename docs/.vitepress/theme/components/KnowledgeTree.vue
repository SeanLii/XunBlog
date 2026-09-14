<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useKnowledge } from '../composables/knowledge'
const props = withDefaults(defineProps<{ routes: string[], depth?: number, scope?: string }>(), { depth: 0, scope: 'map' })
const { model, route, href } = useKnowledge()
const nodes = computed(() => props.routes.map(r => model.value.nodes[r]))
const expanded = ref<Record<string, boolean>>({})
const storageKey = computed(() => `xunblog-tree-${props.scope}`)
const isOpen = (r: string) => expanded.value[r] ?? (props.scope === 'sidebar' ? route.value.startsWith(r) : props.depth < 1)
onMounted(() => {
  try { expanded.value = JSON.parse(localStorage.getItem(storageKey.value) || '{}') } catch {}
})
watch(route, value => {
  // Reveal the active branch on navigation, while preserving other choices.
  for (const r of props.routes) if (value.startsWith(r)) expanded.value[r] = true
})
function toggle(r: string) {
  expanded.value[r] = !isOpen(r)
  try {
    const all = JSON.parse(localStorage.getItem(storageKey.value) || '{}')
    localStorage.setItem(storageKey.value, JSON.stringify({ ...all, [r]: expanded.value[r] }))
  } catch {}
}
</script>
<template>
  <ul class="knowledge-tree" :class="{ 'tree-root': depth === 0 }">
    <li v-for="node in nodes" :key="node.route">
      <div class="tree-row" :class="{ active: route === node.route }">
        <button v-if="node.children.length" class="tree-toggle" :aria-label="`${isOpen(node.route) ? '折叠' : '展开'} ${node.title}`" :aria-expanded="isOpen(node.route)" @click="toggle(node.route)">{{ isOpen(node.route) ? '−' : '+' }}</button>
        <span v-else class="tree-leaf" aria-hidden="true">·</span>
        <a :href="href(node.route)" :aria-current="route === node.route ? 'page' : undefined">{{ node.title }}</a>
      </div>
      <KnowledgeTree v-if="node.children.length" v-show="isOpen(node.route)" :routes="node.children" :depth="depth + 1" :scope="scope" />
    </li>
  </ul>
</template>
