import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Load .env from the monorepo root (parent of web/) so VITE_* and backend
  // vars can live in a single .env file at the project root.
  envDir: '..',
  server: {
    proxy: {
      '/api': {
        target: 'https://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
