import fs from 'node:fs'
import path from 'node:path'

const docsRoot = path.resolve('docs')

function walk(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(directory, entry.name)
    if (entry.isDirectory() && entry.name === '.vitepress') return []
    if (entry.isDirectory()) return walk(full)
    return entry.isFile() && entry.name === 'index.md' ? [full] : []
  })
}

function frontmatterValue(frontmatter, key) {
  return frontmatter.match(new RegExp(`^${key}:\\s*["']([^"']+)["']\\s*$`, 'm'))?.[1]
}

function frontmatterLinks(frontmatter, key) {
  const block = frontmatter.match(new RegExp(`^${key}:\\s*\\n((?:[ \\t]+-.*\\n)*)`, 'm'))?.[1] || ''
  return [...block.matchAll(/["'](\/[^"']+)["']/g)].map((match) => match[1])
}

const titleByRoute = new Map()
const pages = []

for (const file of walk(docsRoot)) {
  const source = fs.readFileSync(file, 'utf8')
  if (!source.startsWith('---\n')) continue
  const end = source.indexOf('\n---', 4)
  const frontmatter = source.slice(4, end)
  const canonical = frontmatterValue(frontmatter, 'canonical')
  const title = frontmatterValue(frontmatter, 'title')
  if (!canonical) continue
  titleByRoute.set(canonical, title)
  pages.push({ file, source, end, frontmatter, canonical, title })
}

for (const page of pages) {
  let frontmatter = page.frontmatter
  if (!/^standard:/m.test(frontmatter)) frontmatter += '\nstandard: "XunBlog Content & Knowledge Architecture v1.0"'
  if (!/^rebuilt:/m.test(frontmatter)) frontmatter += '\nrebuilt: "2026-09-15"'

  let body = page.source.slice(page.end + 4).replace(/^\n+/, '')
  const heading = body.match(/^# .+$/m)?.[0]
  if (!heading) throw new Error(`Missing H1: ${page.file}`)

  // Generic evaluation buckets hide causality. Make their role explicit while
  // preserving the mechanism-specific discussion that follows.
  body = body
    .replace(/^## Scope and Limitations$/gm, '## 机制边界与适用范围')
    .replace(/^## Applications and Limitations$/gm, '## 表示能力导出的用途与边界')
    .replace(/^## Limitations and Scope$/gm, '## 机制导出的能力边界')
    .replace(/^## Limitations$/gm, '## 由机制产生的边界')

  if (!body.includes('> **知识边界**')) {
    const prerequisites = frontmatterLinks(frontmatter, 'prerequisites')
    const dependencyText = prerequisites.length
      ? `依赖机制由 ${prerequisites.map((route) => `[${titleByRoute.get(route) || route}](${route})`).join('、')} 的 canonical page 定义；本文只在当前语境中调用其接口。`
      : '本文从自身定义出发，不在这里扩展与当前对象无直接作用关系的背景知识。'
    const boundary = `\n\n> **知识边界**：本文的 canonical 对象是 **${page.title}**。${dependencyText}\n`
    body = body.replace(heading, heading + boundary)
  }

  fs.writeFileSync(page.file, `---\n${frontmatter}\n---\n${body.replace(/\s+$/, '')}\n`)
}

console.log(`Rebuilt ${pages.length} canonical pages against the v1.0 content standard.`)
