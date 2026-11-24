import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import fs from 'node:fs'
import path from 'node:path'

/**
 * Development-only middleware plugin serving the canonical benchmark sample
 * from the data/ directory without bundling or duplicating it in public/.
 */
function devSampleFixturePlugin(): Plugin {
  return {
    name: 'dev-sample-fixture-serve',
    apply: 'serve', // Strictly development server only; ignored in production build
    configureServer(server) {
      server.middlewares.use('/dev-samples/MVI_20011_sample.mp4', (_req, res) => {
        const fixturePath = path.resolve(import.meta.dirname, '../data/samples/MVI_20011_sample.mp4')
        if (fs.existsSync(fixturePath)) {
          res.setHeader('Content-Type', 'video/mp4')
          fs.createReadStream(fixturePath).pipe(res)
        } else {
          res.statusCode = 404
          res.end('Dev benchmark fixture not found on server')
        }
      })
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss(), devSampleFixturePlugin()],
  server: {
    port: 3000,
    host: '127.0.0.1',
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})