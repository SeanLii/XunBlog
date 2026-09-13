<script setup lang="ts">
import { computed } from 'vue'
import { useData } from 'vitepress'

const props = defineProps<{
  status?: string
  difficulty?: string
  updated?: string
}>()

const { frontmatter } = useData()
const status = computed(() => props.status ?? frontmatter.value.status)
const difficulty = computed(() => props.difficulty ?? frontmatter.value.difficulty)
const updated = computed(() => props.updated ?? frontmatter.value.updated)
</script>

<template>
  <div class="note-meta" aria-label="笔记信息">
    <span v-if="status"><b>Status</b>{{ status }}</span>
    <span v-if="difficulty"><b>Level</b>{{ difficulty }}</span>
    <span v-if="updated"><b>Updated</b>{{ updated }}</span>
  </div>
</template>
