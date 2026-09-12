import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The browser never talks to AWS: every request goes to the FastAPI control plane,
// which the dev server proxies so local development matches the deployed setup.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.ML_FACTORY_API_URL ?? 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      // Ant Design dominates the bundle; splitting it keeps app changes cheap to re-download.
      output: {
        manualChunks: {
          react: ['react', 'react-dom', 'react-router-dom'],
          antd: ['antd', '@ant-design/icons'],
          query: ['@tanstack/react-query'],
        },
      },
    },
  },
});
