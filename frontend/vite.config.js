import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Use 127.0.0.1, not localhost: Node 17+ resolves localhost to IPv6 (::1)
      // first, but Flask's dev server binds IPv4 only → ECONNREFUSED ::1:5000.
      '/api': 'http://127.0.0.1:5000',
    },
  },
})
