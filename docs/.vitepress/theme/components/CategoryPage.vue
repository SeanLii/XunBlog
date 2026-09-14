<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useKnowledge } from '../composables/knowledge'
const { model, current, href } = useKnowledge()
const selected = ref('')
watch(current, node => { selected.value = node?.focus || '' }, { immediate: true })
const options = computed(() => Object.values(model.value.nodes).filter((n: any) => n.kind === 'canonical' && n.route.startsWith(current.value.route)) as any[])
const focus = computed(() => model.value.nodes[selected.value])
const branches = computed(() => focus.value?.relations || [])
</script>
<template>
  <div class="category-page" v-if="current">
    <p class="category-description">{{ current.description }}</p>
    <section aria-labelledby="structure-title">
      <h2 id="structure-title">Knowledge Structure</h2>
      <p class="structure-caption">选择概念查看其知识关系；跨领域链接不改变目录归属。</p>
      <label class="concept-select">查看概念 <select v-model="selected"><option v-for="node in options" :key="node.route" :value="node.route">{{ node.title }}</option></select></label>
      <div v-if="focus" class="relation-map" aria-label="知识关系图">
        <div class="map-center"><a :href="href(focus.route)">{{ focus.title }}</a><small>{{ focus.breadcrumbs.slice(0, -1).map(r => model.nodes[r].title).join(' / ') }}</small></div>
        <div class="map-branches"><section v-for="branch in branches" :key="branch.label"><h3>{{ branch.label }}</h3><ul><li v-for="r in branch.routes" :key="r"><a :href="href(r)">{{ model.nodes[r].title }}</a></li></ul></section></div>
      </div>
    </section>
    <section aria-labelledby="topics-title"><h2 id="topics-title">Core Topics</h2>
      <dl class="core-topics"><div v-for="r in current.children" :key="r"><dt><a :href="href(r)">{{ model.nodes[r].title }}</a></dt><dd>{{ model.nodes[r].description || model.nodes[r].summary }}</dd></div></dl>
    </section>
  </div>
</template>
