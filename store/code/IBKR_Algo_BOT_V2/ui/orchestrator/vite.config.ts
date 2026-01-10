import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/orchestrator/',
  build: {
    outDir: 'build',
    sourcemap: true
  },
  server: {
    port: 3003,
    proxy: {
      '/api': 'http://localhost:9100'
    }
  }
})
