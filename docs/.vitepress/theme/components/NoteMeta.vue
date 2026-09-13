<script setup lang="ts">
defineProps<{
  status?: string
  difficulty?: string
  updated?: string
}>()
</script>

<template>
  <div class="note-meta" aria-label="笔记信息">
    <span v-if="status"><b>Status</b>{{ status }}</span>
    <span v-if="difficulty"><b>Level</b>{{ difficulty }}</span>
    <span v-if="updated"><b>Updated</b>{{ updated }}</span>
  </div>
</template>

