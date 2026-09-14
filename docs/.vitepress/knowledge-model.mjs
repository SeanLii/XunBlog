import fs from 'node:fs'
import path from 'node:path'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { navigationGraph } from './knowledge-graph.mjs'
import { canonicalRoute } from './routes.mjs'

const root = fileURLToPath(new URL('../../', import.meta.url))
export const categories = {
  '/mathematics/': { title: 'Mathematics', description: '数学为向量表示、概率分布与模型目标提供描述语言。这里按线性代数、概率与信息论组织已有概念。', focus: '/mathematics/probability/normal-distribution/' },
  '/mathematics/linear-algebra/': { title: 'Linear Algebra', description: '线性代数描述向量及其变换。这里连接向量、矩阵与点积，以及它们在神经网络中的使用。', focus: '/mathematics/linear-algebra/dot-product/' },
  '/mathematics/probability/': { title: 'Probability', description: '概率描述随机对象与不确定性。这里包含随机变量、分布及其统计量，并连接潜变量模型。', focus: '/mathematics/probability/normal-distribution/' },
  '/mathematics/information-theory/': { title: 'Information Theory', description: '信息论为分布之间的差异提供度量语言。当前内容聚焦 KL Divergence 及其在变分模型中的使用。', focus: '/mathematics/information-theory/kl-divergence/' },
  '/deep-learning/': { title: 'Deep Learning', description: '深度学习通过神经网络学习数据的表示与映射。这里组织基础网络运算、Transformer 与卷积网络，并连接它们在 ACT 中的使用。', focus: '/deep-learning/transformer/' },
  '/deep-learning/core/': { title: 'Core', description: '神经网络基础运算构成更大模型的模块。这里包含线性层、Softmax、Embedding、残差连接与层归一化。', focus: '/deep-learning/core/linear-layer/' },
  '/deep-learning/cnn/': { title: 'Convolutional Neural Networks', description: '卷积网络处理具有空间结构的输入。这里从卷积运算连接 CNN、ResNet 与 ACT 的视觉管线。', focus: '/deep-learning/cnn/resnet/' },
  '/generative-models/': { title: 'Generative Models', description: '生成模型描述数据如何由分布与潜变量产生。当前内容围绕变分推断、VAE 与 CVAE，并连接 ACT 的训练支路。', focus: '/generative-models/variational-autoencoder/' },
  '/robot-learning/': { title: 'Robot Learning', description: '机器人学习研究如何从数据中学习行为。当前内容覆盖模仿学习、行为克隆与 ACT，连接视觉观测、动作预测和训练推理过程。', focus: '/robot-learning/act/' }
}
const aliases = {
  act: ['Action Chunking with Transformers', '动作分块'], qkv: ['QKV', 'Query Key Value', '查询 键 值'],
  'normal-distribution': ['Gaussian', '高斯分布', '正态分布'],
  'multivariate-normal-distribution': ['Multivariate Gaussian', '多元正态分布'],
  'variational-autoencoder': ['VAE', '变分自编码器'],
  'conditional-variational-autoencoder': ['CVAE', 'Conditional VAE', '条件变分自编码器'],
  'evidence-lower-bound': ['ELBO', '证据下界'], 'kl-divergence': ['KLD', 'KL', '相对熵'],
  'convolutional-neural-network': ['CNN', '卷积神经网络'], 'resnet': ['Residual Network', '残差网络'],
  'behavior-cloning': ['BC', '行为克隆'], 'imitation-learning': ['模仿学习'],
  'attention': ['注意力'], 'self-attention': ['自注意力'], 'cross-attention': ['交叉注意力'],
  'multi-head-attention': ['多头注意力'], 'dot-product': ['点积'], 'variance': ['方差'],
  'expectation': ['期望'], 'random-variable': ['随机变量'], 'matrix': ['矩阵'], 'vector': ['向量']
}
// Explicit component-use edges, already represented in the existing navigation graph.
const uses = {
  '/robot-learning/act/': ['/deep-learning/transformer/', '/generative-models/conditional-variational-autoencoder/', '/deep-learning/cnn/resnet/'],
  '/robot-learning/act/architecture/': navigationGraph['/robot-learning/act/architecture/'],
  '/robot-learning/act/cvae-in-act/': navigationGraph['/robot-learning/act/cvae-in-act/'],
  '/robot-learning/act/vision-pipeline/': ['/deep-learning/cnn/resnet/'],
  '/deep-learning/transformer/': ['/deep-learning/transformer/attention/'],
  '/deep-learning/cnn/resnet/': ['/deep-learning/cnn/convolutional-neural-network/'],
  '/deep-learning/cnn/convolutional-neural-network/': ['/deep-learning/cnn/convolution/']
}
function walk(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap(e => {
    if (e.name.startsWith('.')) return []
    const f = path.join(dir, e.name)
    return e.isDirectory() ? walk(f) : e.name.endsWith('.md') ? [f] : []
  })
}
function field(source, key) {
  return source.match(new RegExp(`^${key}:\\s*"([^"]*)"\\s*$`, 'm'))?.[1]
}
function links(source, key) {
  const block = source.match(new RegExp(`^${key}:\\s*\\n((?:[ \\t]+-.*\\n)*)`, 'm'))?.[1] || ''
  return [...block.matchAll(/"(\/[^\"]*)"/g)].map(m => m[1])
}
export function buildKnowledgeModel() {
  const pages = []
  for (const file of walk(path.join(root, 'docs'))) {
    const source = fs.readFileSync(file, 'utf8')
    const fm = source.split('\n---')[0]
    const route = field(fm, 'canonical')
    if (!route) continue
    canonicalRoute(route)
    if (path.join(root, 'docs', route.slice(1), 'index.md') !== file) throw new Error(`Ownership/path mismatch: ${route}`)
    if (pages.some(p => p.route === route)) throw new Error(`Duplicate canonical: ${route}`)
    const body = source.slice(source.indexOf('\n---', 4) + 4)
    let updated = null
    try {
      const stamp = execFileSync('git', ['log', '-1', '--format=%cI', '--', path.relative(root, file)], { cwd: root, encoding: 'utf8' }).trim()
      if (stamp) updated = stamp
    } catch { /* No invented date when git history is unavailable. */ }
    const units = (body.match(/[\u3400-\u9fff]/g) || []).length + (body.match(/[A-Za-z0-9]+/g) || []).length
    const slug = route.split('/').filter(Boolean).at(-1)
    const summary = body.split(/\n\s*\n/).find(p => /^[A-Za-z\u3400-\u9fff]/.test(p.trim()))?.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1').replace(/[*`]/g, '').slice(0, 180) || ''
    pages.push({ summary, route, title: field(fm, 'title'), kind: 'canonical', updated, minutes: Math.max(1, Math.ceil(units / 250)), aliases: aliases[slug] || [], prerequisites: links(fm, 'prerequisites'), related: links(fm, 'related') })
  }
  if (pages.length !== 53) throw new Error(`Expected 53 canonical pages, found ${pages.length}`)
  const nodes = Object.fromEntries(pages.map(p => [p.route, p]))
  for (const [route, category] of Object.entries(categories)) {
    if (nodes[route]) throw new Error(`Category conflicts with canonical: ${route}`)
    if (!fs.existsSync(path.join(root, 'docs', route.slice(1), 'index.md'))) throw new Error(`Missing category: ${route}`)
    nodes[route] = { ...category, route, kind: 'category' }
  }
  const roots = []
  for (const node of Object.values(nodes)) {
    const segments = node.route.split('/').filter(Boolean)
    node.parent = segments.length > 1 ? '/' + segments.slice(0, -1).join('/') + '/' : null
    node.children = []
    node.breadcrumbs = segments.map((_, i) => '/' + segments.slice(0, i + 1).join('/') + '/')
    for (const r of node.breadcrumbs) if (!nodes[r]) throw new Error(`Missing tree ancestor ${r}`)
  }
  for (const node of Object.values(nodes)) {
    if (node.parent) nodes[node.parent].children.push(node.route)
    else roots.push(node.route)
  }
  const order = ['linear-algebra', 'probability', 'information-theory', 'core', 'transformer', 'cnn', 'imitation-learning', 'behavior-cloning', 'act']
  for (const node of Object.values(nodes)) node.children.sort((a,b) => {
    const rank = r => { const i = order.indexOf(r.split('/').filter(Boolean).at(-1)); return i < 0 ? 99 : i }
    return rank(a)-rank(b) || nodes[a].title.localeCompare(nodes[b].title)
  })
  roots.sort((a,b) => Object.keys(categories).indexOf(a)-Object.keys(categories).indexOf(b))
  for (const page of pages) {
    const usedIn = pages.filter(p => p.prerequisites.includes(page.route) || (uses[p.route] || []).includes(page.route)).map(p => p.route)
    const groups = [
      { label: 'Prerequisites', routes: page.prerequisites },
      { label: 'Parent Concept', routes: nodes[page.parent]?.kind === 'canonical' ? [page.parent] : [] },
      { label: 'Used In', routes: usedIn },
      { label: 'Related', routes: [...page.related, ...(navigationGraph[page.route] || [])] }
    ]
    const seen = new Set([page.route])
    page.relations = groups.map(g => ({ ...g, routes: g.routes.filter(r => {
      if (!nodes[r] || nodes[r].kind !== 'canonical') throw new Error(`Missing graph target ${r}`)
      if (seen.has(r)) return false
      seen.add(r); return true
    }) })).filter(g => g.routes.length)
    page.searchText = [page.title, ...page.aliases, page.route.replaceAll('/', ' ').replaceAll('-', ' '), ...page.breadcrumbs.map(r => nodes[r].title)].join(' ').toLowerCase()
  }
  const recent = pages.filter(p => p.updated).sort((a,b) => b.updated.localeCompare(a.updated) || a.route.localeCompare(b.route)).slice(0, 5).map(p => p.route)
  return { nodes, roots, recent, focus: '/robot-learning/act/' }
}
