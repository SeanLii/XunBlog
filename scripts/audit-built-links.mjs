import fs from 'node:fs'
import path from 'node:path'
import assert from 'node:assert/strict'
import { normalizeBase } from '../docs/.vitepress/routes.mjs'
const root = path.resolve('docs/.vitepress/dist')
const base = normalizeBase(process.env.BASE_PATH || '/')
function walk(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap(e => e.isDirectory() ? walk(path.join(dir, e.name)) : e.name.endsWith('.html') ? [path.join(dir, e.name)] : [])
}
const files = walk(root)
const ids = new Map(files.map(f => [f, new Set([...fs.readFileSync(f, 'utf8').matchAll(/\sid="([^"]+)"/g)].map(m => m[1]))]))
const errors = []
let count = 0
for (const file of files) {
  const html = fs.readFileSync(file, 'utf8')
  if (/class="(?:[^"]* )?prev-next(?: |")/.test(html)) errors.push(`Linear pager rendered: ${file}`)
  if (/AI Learning Notes|Learning first\. Content first\./.test(html)) errors.push(`Old brand/footer: ${file}`)
  for (const match of html.matchAll(/<(a|script|link|img)\b[^>]*?\s(?:href|src)="([^"]+)"/g)) {
    const value = match[2].replaceAll('&amp;', '&')
    if (/^(https?:|mailto:|data:|tel:)/.test(value)) continue
    const [pathname, hash] = value.split('#')
    if (!pathname) {
      if (hash && !ids.get(file).has(decodeURIComponent(hash))) errors.push(`Missing anchor ${value} in ${file}`)
      continue
    }
    const clean = pathname.split('?')[0]
    if (!clean.startsWith(base)) { errors.push(`Missing base ${base}: ${value} in ${file}`); continue }
    const relative = clean.slice(base.length)
    const isPage = match[1] === 'a' && !path.extname(relative)
    if (isPage && !clean.endsWith('/')) errors.push(`Missing trailing slash: ${value}`)
    const target = path.resolve(root, relative, clean.endsWith('/') ? 'index.html' : '')
    if (!target.startsWith(root + path.sep) || !fs.existsSync(target)) errors.push(`Missing built target ${value} in ${file}`)
    else if (hash && ids.has(target) && !ids.get(target).has(decodeURIComponent(hash))) errors.push(`Missing target anchor ${value}`)
    count++
  }
}
if (errors.length) { console.error(errors.join('\n')); process.exitCode = 1 }
else console.log(`Built link audit: ${count} internal page/asset targets across ${files.length} HTML pages; base ${base}, trailing slashes, anchors and no linear pager verified.`)
