import assert from 'node:assert/strict'
import fs from 'node:fs'
import { buildKnowledgeModel } from '../docs/.vitepress/knowledge-model.mjs'
import { canonicalRoute, pageHref, routeFromPath } from '../docs/.vitepress/routes.mjs'
const model = buildKnowledgeModel()
assert.deepEqual(model.nodes['/deep-learning/attention/qkv/'].breadcrumbs, ['/deep-learning/', '/deep-learning/attention/', '/deep-learning/attention/qkv/'])
assert.equal(model.nodes['/deep-learning/bert/cls-token/'].parent, '/deep-learning/bert/')
let links = 0
const targets = ['/', ...Object.keys(model.nodes)]
for (const [route, node] of Object.entries(model.nodes)) {
  canonicalRoute(route)
  const expected = route.split('/').filter(Boolean).map((_, i, parts) => '/' + parts.slice(0, i + 1).join('/') + '/')
  assert.deepEqual(node.breadcrumbs, expected, `Ownership breadcrumb: ${route}`)
  const relations = (node.relations || []).flatMap(g => g.routes)
  assert.equal(new Set(relations).size, relations.length, `Duplicate relations: ${route}`)
  assert(!relations.includes(route), `Self relation: ${route}`)
  for (const link of [...node.breadcrumbs, ...node.children, ...relations, ...(node.focus ? [node.focus] : [])]) {
    assert(model.nodes[link], `Missing navigation target: ${link}`); links++
  }
  if (node.kind === 'category') {
    assert(node.description && node.children.length, `Empty category ${route}`)
    assert(model.nodes[node.focus].relations.length, `Empty graph ${route}`)
  }
}
for (const route of targets) {
  for (const base of ['/', '/XunBlog/']) {
    const href = pageHref(route, base)
    assert.equal(routeFromPath(href, base), route)
    assert.equal(routeFromPath(href + 'index.html', base), route)
    if (route !== '/') assert.equal(routeFromPath(href.slice(0, -1), base), route)
  }
}
for (const malformed of ['/normal-distribution', '/XunBlog/robot-learning/act/', '//robot-learning/', '/robot-learning/../act/', '/about.html']) {
  assert.throws(() => canonicalRoute(malformed), `Accepted malformed path: ${malformed}`)
}
assert(model.nodes['/robot-learning/act/'])
assert(model.nodes['/mathematics/probability/normal-distribution/'])
assert.equal(model.nodes['/robot-learning/pi0/'].kind, 'canonical')
assert.equal(model.nodes['/robot-learning/pi0/'].children.length, 9)
for (const alias of ['pi0', 'pi 0', 'pi-zero', 'π₀']) assert(model.nodes['/robot-learning/pi0/'].searchText.includes(alias))
assert.deepEqual(model.startingPoints.map(entry => entry.route), ['/robot-learning/act/', '/robot-learning/pi0/'])
for (const entry of model.startingPoints) {
  assert.equal(model.nodes[entry.route]?.kind, 'canonical')
  assert(entry.description)
  for (const slug of entry.links) assert(model.nodes[entry.route + slug + '/'])
}
for (const route of model.recent) assert.equal(model.nodes[route]?.kind, 'canonical')
for (const route of model.recent) assert(!Number.isNaN(Date.parse(model.nodes[route].updated)))
const homepage = fs.readFileSync('docs/.vitepress/theme/components/HomePage.vue', 'utf8')
assert.equal((homepage.match(/class="starting-card"/g) || []).length, 1)
assert(/model\.startingPoints/.test(homepage))
console.log(`Navigation audit: ${Object.keys(model.nodes).length} nodes, ${links} tree/graph links; ownership, deduplication, real update dates, root/base routes and malformed-route rejection passed.`)
