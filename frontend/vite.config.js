import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Dev only. In production the API base comes from VITE_API_BASE.
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})
