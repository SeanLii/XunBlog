import fs from 'node:fs'
import path from 'node:path'
import assert from 'node:assert/strict'
import { routeMigrations } from '../docs/.vitepress/route-migrations.mjs'
import { pageHref } from '../docs/.vitepress/routes.mjs'
const root = path.resolve('docs/.vitepress/dist')
const base = process.env.BASE_PATH || '/'
for (const [oldRoute, target] of Object.entries(routeMigrations)) {
  const href = pageHref(target, base)
  const destination = path.join(root, target, 'index.html')
  assert(fs.existsSync(destination), `Missing redirect destination: ${target}`)
  const file = path.join(root, oldRoute, 'index.html')
  assert(!fs.existsSync(file), `Redirect would overwrite content: ${oldRoute}`)
  fs.mkdirSync(path.dirname(file), { recursive: true })
  fs.writeFileSync(file, `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>页面已迁移 | XunBlog</title><meta name="robots" content="noindex"><link rel="canonical" href="https://seanlii.github.io${pageHref(target, '/XunBlog/')}"><meta http-equiv="refresh" content="0;url=${href}"><script>location.replace(${JSON.stringify(href)} + location.search + location.hash)</script></head><body><p>页面已迁移：<a href="${href}">打开新版知识页面</a></p></body></html>`)
}
console.log(`Generated ${Object.keys(routeMigrations).length} legacy route redirects with base ${base}.`)
