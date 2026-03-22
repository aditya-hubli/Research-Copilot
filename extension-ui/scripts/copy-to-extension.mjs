/**
 * copy-to-extension.mjs — Vite edition
 * Copies out/ into ../extension/ after `vite build`.
 * Vite outputs no inline scripts — zero CSP issues.
 */
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const ROOT    = path.resolve(__dirname, '..')
const OUT_DIR = path.join(ROOT, 'out')
const EXT_DIR = path.join(ROOT, '..', 'extension')

function ensureDir(d) { fs.mkdirSync(d,{recursive:true}) }
function removeDir(d) { if(fs.existsSync(d)) fs.rmSync(d,{recursive:true,force:true}) }

function copyDir(src,dest) {
  ensureDir(dest)
  for (const e of fs.readdirSync(src,{withFileTypes:true})) {
    const s=path.join(src,e.name), d=path.join(dest,e.name)
    e.isDirectory() ? copyDir(s,d) : fs.copyFileSync(s,d)
  }
}

// Validate build exists
if (!fs.existsSync(path.join(OUT_DIR,'index.html'))) {
  console.error('\n❌  out/index.html not found. Run `npm run build` first.\n')
  process.exit(1)
}

console.log('\n📦  Deploying Vite build → extension/\n')

// 1. Clean only what we own
const OWNED = ['sidebar.html','ext_assets','index.html','404.html']
for (const name of OWNED) {
  const t = path.join(EXT_DIR,name)
  if (!fs.existsSync(t)) continue
  fs.statSync(t).isDirectory() ? fs.rmSync(t,{recursive:true,force:true}) : fs.unlinkSync(t)
}
console.log('✓  Cleaned old build artifacts')

// 2. Copy index.html → sidebar.html (no patching needed — Vite uses ./ paths natively)
fs.copyFileSync(path.join(OUT_DIR,'index.html'), path.join(EXT_DIR,'sidebar.html'))
console.log('✓  Wrote sidebar.html')

// 3. Copy ext_assets/
const assetSrc  = path.join(OUT_DIR,'ext_assets')
const assetDest = path.join(EXT_DIR,'ext_assets')
if (fs.existsSync(assetSrc)) {
  copyDir(assetSrc, assetDest)
  console.log('✓  Copied ext_assets/')
}

console.log('\n✅  Done!\n')
console.log('  Load unpacked:')
console.log(`  ${EXT_DIR}\n`)
