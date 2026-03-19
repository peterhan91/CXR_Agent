import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: { DEFAULT: "#000000", surface: "#1C1C1E", elevated: "#2C2C2E" },
        glass: { DEFAULT: "rgba(44,44,46,0.72)" },
        text: {
          primary: "#F5F5F7",
          secondary: "#86868B",
          tertiary: "#48484A",
        },
        accent: { DEFAULT: "#0071E3", hover: "#0077ED" },
        semantic: { green: "#34C759", orange: "#FF9F0A", red: "#FF3B30" },
        separator: "rgba(255,255,255,0.06)",
      },
      fontFamily: {
        sans: [
          "-apple-system",
          "BlinkMacSystemFont",
          "SF Pro Text",
          "Inter",
          "sans-serif",
        ],
      },
      backdropBlur: { glass: "20px" },
      borderRadius: { panel: "12px" },
    },
  },
  plugins: [],
};
export default config;
