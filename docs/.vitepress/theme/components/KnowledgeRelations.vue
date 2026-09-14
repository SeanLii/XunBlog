<script setup lang="ts">
import { useKnowledge } from '../composables/knowledge'
const { model, current, href } = useKnowledge()
</script>
<template>
  <section v-if="current?.relations?.length" class="knowledge-relations" aria-labelledby="relations-title">
    <h2 id="relations-title">Knowledge Relations</h2>
    <div class="relation-groups">
      <section v-for="group in current.relations" :key="group.label">
        <h3>{{ group.label }}</h3>
        <ul><li v-for="r in group.routes" :key="r"><a :href="href(r)">{{ model.nodes[r].title }}</a></li></ul>
      </section>
    </div>
  </section>
</template>
