<script setup lang="ts">
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'
import { navigationGraph } from '../../knowledge-graph.mjs'

const props = defineProps<{
  status?: string
  difficulty?: string
  updated?: string
}>()

const { frontmatter } = useData()
const status = computed(() => props.status ?? frontmatter.value.status)
const difficulty = computed(() => props.difficulty ?? frontmatter.value.difficulty)
const updated = computed(() => props.updated ?? frontmatter.value.updated)
const domain = computed<string>(() => frontmatter.value.domain ?? '')
const kind = computed<string>(() => frontmatter.value.kind ?? '')
const canonical = computed<string>(() => frontmatter.value.canonical ?? '')
const prerequisites = computed<string[]>(() => frontmatter.value.prerequisites ?? [])
const related = computed<string[]>(() => [...new Set([
  ...(frontmatter.value.related ?? []),
  ...(navigationGraph[canonical.value] ?? [])
])].filter((link) => !prerequisites.value.includes(link)))
const hasMetadata = computed(() => (
  status.value || difficulty.value || updated.value || domain.value || kind.value ||
  prerequisites.value.length || related.value.length
))

const titleOverrides: Record<string, string> = {
  act: 'ACT',
  qkv: 'Query / Key / Value',
  resnet: 'ResNet',
  softmax: 'Softmax',
  'cvae-in-act': 'CVAE in ACT',
  'cls-token': 'CLS Token',
  'kl-divergence': 'KL Divergence',
  'why-z-zero-at-inference': '为什么 ACT 推理时令 z = 0？'
}

function linkTitle(link: string) {
  const slug = link.split('/').filter(Boolean).at(-1) ?? link
  return titleOverrides[slug] ?? slug
    .split('-')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}
</script>

<template>
  <aside v-if="hasMetadata" class="knowledge-context" aria-label="知识页面信息">
    <div class="note-meta">
      <span v-if="domain"><b>Path</b>{{ domain }}</span>
      <span v-if="kind"><b>Type</b>{{ kind }}</span>
      <span v-if="status"><b>Status</b>{{ status }}</span>
      <span v-if="difficulty"><b>Level</b>{{ difficulty }}</span>
      <span v-if="updated"><b>Updated</b>{{ updated }}</span>
    </div>
    <div v-if="prerequisites.length || related.length" class="knowledge-links">
      <div v-if="prerequisites.length">
        <b>Prerequisites</b>
        <a v-for="link in prerequisites" :key="link" :href="withBase(link)">{{ linkTitle(link) }}</a>
      </div>
      <div v-if="related.length">
        <b>Related</b>
        <a v-for="link in related" :key="link" :href="withBase(link)">{{ linkTitle(link) }}</a>
      </div>
    </div>
  </aside>
</template>
