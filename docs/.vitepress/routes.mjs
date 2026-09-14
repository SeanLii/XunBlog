// Stored routes never contain a deployment prefix. All page URLs end in '/'.
export function normalizeBase(base = '/') {
  if (!/^\/(?:[A-Za-z0-9_-]+\/)*$/.test(base)) throw new Error(`Invalid base path: ${base}`)
  return base
}
export function canonicalRoute(input) {
  if (!/^\/(?:[a-z0-9-]+\/)*$/.test(input)) throw new Error(`Malformed canonical route: ${input}`)
  return input
}
export function pageHref(route, base = '/') {
  return normalizeBase(base) + canonicalRoute(route).slice(1)
}
export function routeFromPath(input, base = '/') {
  let route = input.split(/[?#]/)[0]
  if (base !== '/' && route.startsWith(base)) route = '/' + route.slice(base.length)
  route = route.replace(/index\.html$/, '').replace(/\.html$/, '')
  return canonicalRoute(route.endsWith('/') ? route : route + '/')
}
