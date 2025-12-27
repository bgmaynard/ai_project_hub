import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/trading-new/', // Serve from /trading-new/ path
  server: {
    port: 3001,
    proxy: {
      '/api': {
        target: 'http://localhost:9100',
        changeOrigin: true
      },
      '/ws': {
        target: 'ws://localhost:9100',
        ws: true
      }
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true
  }
})
