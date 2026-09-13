import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite 开发服务器把 /api 请求转发给现有 FastAPI 服务。
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true
      }
    }
  }
});
