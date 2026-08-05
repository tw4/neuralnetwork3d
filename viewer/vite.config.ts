import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";

export default defineConfig({
  plugins: [react(), viteSingleFile()],
  build: {
    outDir: "../src/nn3d/static",
    emptyOutDir: true,
    target: "es2020",
    assetsInlineLimit: 100_000_000,
    chunkSizeWarningLimit: 4000,
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8092",
    },
  },
});
