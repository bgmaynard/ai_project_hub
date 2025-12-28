/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Sterling Trader Pro Theme
        sterling: {
          bg: '#000000',
          panel: '#0a0a0a',
          border: '#1a1a1a',
          header: '#0d0d0d',
          text: '#c0c0c0',
          muted: '#666666',
        },
        // Trading colors
        buy: '#0066cc',
        sell: '#cc0000',
        up: '#00cc00',
        down: '#ff3333',
        neutral: '#999999',
        // Accent colors
        accent: {
          primary: '#3b82f6',
          success: '#10b981',
          warning: '#f59e0b',
          danger: '#ef4444',
        }
      },
      fontFamily: {
        mono: ['Consolas', 'Monaco', 'monospace'],
      },
      fontSize: {
        'xxs': '0.75rem',   // was 0.65rem
        'xs': '0.875rem',   // was 0.75rem
        'sm': '1rem',       // was 0.875rem
        'base': '1.125rem', // was 1rem
      }
    },
  },
  plugins: [],
}
