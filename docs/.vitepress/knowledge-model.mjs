import fs from 'node:fs'
import path from 'node:path'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { navigationGraph } from './knowledge-graph.mjs'
import { routeMigrations } from './route-migrations.mjs'
import { canonicalRoute } from './routes.mjs'

const root = fileURLToPath(new URL('../../', import.meta.url))
export const categories = {
  '/mathematics/analysis/': { title: 'Analysis', description: '分析学中的运算与表示。当前内容聚焦卷积及其在神经网络中的使用。', focus: '/mathematics/analysis/convolution/' },
  '/deep-learning/sequence-modeling/': { title: 'Sequence Modeling', description: '序列建模中的位置表示与因果约束，以及它们在具体模型中的使用。', focus: '/deep-learning/sequence-modeling/positional-encoding/' },
  '/deep-learning/representation-learning/': { title: 'Representation Learning', description: '表示学习研究数据的编码方式。当前内容聚焦 Autoencoder 及其与生成模型的联系。', focus: '/deep-learning/representation-learning/autoencoder/' },
  '/mathematics/calculus/': { title: 'Calculus', description: '微积分描述连续变化。当前内容通过常微分方程连接 Flow Matching 的连续时间过程。', focus: '/mathematics/calculus/ordinary-differential-equation/' },
  '/mathematics/numerical-methods/': { title: 'Numerical Methods', description: '数值方法用有限计算近似连续问题。当前内容聚焦 Euler Method 与动作生成中的数值积分。', focus: '/mathematics/numerical-methods/euler-method/' },
  '/deep-learning/multimodal/': { title: 'Multimodal Models', description: '多模态模型联合处理不同类型的输入。当前内容聚焦视觉语言模型及其在机器人策略中的使用。', focus: '/deep-learning/multimodal/vision-language-model/' },
  '/mathematics/': { title: 'Mathematics', description: '数学为向量表示、概率分布与模型目标提供描述语言。这里按线性代数、概率、信息论、微积分与数值方法组织已有概念。', focus: '/mathematics/probability/normal-distribution/' },
  '/mathematics/linear-algebra/': { title: 'Linear Algebra', description: '线性代数描述向量及其变换。这里连接向量、矩阵与点积，以及它们在神经网络中的使用。', focus: '/mathematics/linear-algebra/dot-product/' },
  '/mathematics/probability/': { title: 'Probability', description: '概率描述随机对象与不确定性。这里包含随机变量、分布及其统计量，并连接潜变量模型。', focus: '/mathematics/probability/normal-distribution/' },
  '/mathematics/information-theory/': { title: 'Information Theory', description: '信息论为分布之间的差异提供度量语言。当前内容聚焦 KL Divergence 及其在变分模型中的使用。', focus: '/mathematics/information-theory/kl-divergence/' },
  '/deep-learning/': { title: 'Deep Learning', description: '深度学习通过神经网络学习数据的表示与映射。这里组织基础网络运算、Transformer、卷积网络与多模态模型，并连接它们在机器人策略中的使用。', focus: '/deep-learning/transformer/' },
  '/deep-learning/core/': { title: 'Core', description: '神经网络基础运算构成更大模型的模块。这里包含线性层、Softmax、Embedding、残差连接与层归一化。', focus: '/deep-learning/core/linear-layer/' },
  '/deep-learning/cnn/': { title: 'Convolutional Neural Networks', description: '卷积网络处理具有空间结构的输入。这里从卷积运算连接 CNN、ResNet 与 ACT 的视觉管线。', focus: '/deep-learning/cnn/resnet/' },
  '/generative-models/': { title: 'Generative Models', description: '生成模型描述数据如何由分布与潜变量产生。当前内容覆盖变分推断、VAE、CVAE 与 Flow Matching，连接 ACT 和 π0 的动作生成。', focus: '/generative-models/variational-autoencoder/' },
  '/robot-learning/': { title: 'Robot Learning', description: '机器人学习研究如何从数据中学习行为。当前内容覆盖模仿学习、行为克隆、VLA、跨形态学习、ACT 与 π0，连接视觉观测、动作预测和训练推理过程。', focus: '/robot-learning/pi0/', focusDescription: 'Vision-Language-Action robot policy with Flow Matching.', focusLinks: ['architecture', 'action-expert', 'training', 'inference'] }
}
const aliases = {
  'object-query': ['Object Query', 'Learnable Query Embedding', '对象查询'],
  'position-wise-feed-forward-network': ['FFN', 'Feed-Forward Network'],
  'multilayer-perceptron': ['MLP', '多层感知机'],
  'kv-cache': ['KV Cache', '键值缓存'],
  pi0: ['pi0', 'pi 0', 'pi-zero', 'pi zero', 'π 0', 'π₀'],
  'flow-matching': ['流匹配'],
  'vision-language-model': ['VLM', '视觉语言模型'],
  'vision-language-action-model': ['VLA', '视觉语言动作模型'],
  'ordinary-differential-equation': ['ODE', '常微分方程'],
  'euler-method': ['欧拉方法', '欧拉法'],
  'cross-embodiment-learning': ['跨形态学习'],
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
const originalUses = {
  '/robot-learning/pi0/': ['/deep-learning/multimodal/vision-language-model/', '/generative-models/flow-matching/', '/robot-learning/act/action-chunking/'],
  '/robot-learning/pi0/inference/': ['/mathematics/numerical-methods/euler-method/'],
  '/robot-learning/act/': ['/deep-learning/transformer/', '/generative-models/conditional-variational-autoencoder/', '/deep-learning/cnn/resnet/'],
  '/robot-learning/act/architecture/': navigationGraph['/robot-learning/act/architecture/'],
  '/robot-learning/act/cvae-in-act/': navigationGraph['/robot-learning/act/cvae-in-act/'],
  '/robot-learning/act/vision-pipeline/': ['/deep-learning/cnn/resnet/'],
  '/deep-learning/transformer/': ['/deep-learning/attention/'],
  '/deep-learning/cnn/resnet/': ['/deep-learning/cnn/convolutional-neural-network/'],
  '/deep-learning/cnn/convolutional-neural-network/': ['/mathematics/analysis/convolution/']
}
const uses = Object.fromEntries(Object.entries(originalUses).map(([source, targets]) => [routeMigrations[source] || source, targets.map(target => routeMigrations[target] || target)]))
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
  if (pages.length !== 82) throw new Error(`Expected 82 canonical pages, found ${pages.length}`)
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
  const order = ['linear-algebra', 'probability', 'information-theory', 'calculus', 'numerical-methods', 'core', 'attention', 'sequence-modeling', 'transformer', 'bert', 'detr', 'representation-learning', 'cnn', 'multimodal', 'imitation-learning', 'behavior-cloning', 'vision-language-action-model', 'cross-embodiment-learning', 'act', 'pi0']
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
  return { nodes, roots, recent, focus: '/robot-learning/pi0/', focusDescription: 'Vision-Language-Action robot policy with Flow Matching.', focusLinks: ['architecture', 'action-expert', 'training', 'inference'] }
}
