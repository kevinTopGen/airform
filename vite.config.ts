import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig(({ mode }) => ({
  base: mode === "pages" ? "/airform/" : "/",
  plugins: [react()],
  test: {
    include: ["src/**/*.test.ts"],
  },
}));
