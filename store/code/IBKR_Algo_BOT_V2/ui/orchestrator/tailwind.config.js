/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Semantic colors for orchestrator
        'status-green': '#10b981',
        'status-yellow': '#f59e0b',
        'status-red': '#ef4444',
        'status-gray': '#6b7280',
        'bg-primary': '#0f172a',
        'bg-secondary': '#1e293b',
        'bg-tertiary': '#334155',
        'text-primary': '#f1f5f9',
        'text-secondary': '#94a3b8',
        'border-color': '#475569',
      }
    },
  },
  plugins: [],
}
