import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': path.resolve(__dirname, 'src') }
  },
  build: {
    outDir: 'out',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        // All assets go into ext_assets/ — never _next/, never underscore folders
        assetFileNames: 'ext_assets/[name]-[hash][extname]',
        chunkFileNames: 'ext_assets/[name]-[hash].js',
        entryFileNames: 'ext_assets/[name]-[hash].js',
      }
    }
  },
  // ./ relative paths so chrome-extension:// protocol resolves correctly
  base: './',
})
