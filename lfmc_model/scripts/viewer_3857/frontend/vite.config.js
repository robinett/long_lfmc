import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "VITE_");
  if (mode === "production" && !env.VITE_CARTO_BASEMAP_KEY?.trim()) {
    throw new Error("VITE_CARTO_BASEMAP_KEY is required for production builds");
  }

  return {
    plugins: [react()],
    server: {
      host: "127.0.0.1",
      port: 4174,
      proxy: {
        "/api": {
          target: "http://127.0.0.1:8001",
          changeOrigin: true,
        },
        "/viewer-assets": {
          target: "http://127.0.0.1:8001",
          changeOrigin: true,
        },
      },
    },
  };
});
