import { defineConfig } from 'tsup';

export default defineConfig([
  {
    entry: ['src/index.ts'],
    format: ['cjs', 'esm'],
    dts: true,
    splitting: false,
    sourcemap: true,
    clean: true,
    minify: false,
    target: 'node18',
    outDir: 'dist',
    external: ['vm'],
  },
  {
    entry: ['src/cli.ts'],
    format: ['cjs'],
    dts: false,
    splitting: false,
    sourcemap: true,
    clean: false,
    minify: false,
    target: 'node18',
    outDir: 'dist',
    external: ['vm'],
    banner: {
      js: '#!/usr/bin/env node',
    },
  },
]);
