/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'ibkr-bg': '#1e1e1e',
        'ibkr-surface': '#252526',
        'ibkr-border': '#3e3e42',
        'ibkr-text': '#d4d4d4',
        'ibkr-text-secondary': '#888888',
        'ibkr-accent': '#007acc',
        'ibkr-success': '#4ec9b0',
        'ibkr-warning': '#dcdcaa',
        'ibkr-error': '#f48771',
      },
      fontSize: {
        'xs': '10px',
        'sm': '11px',
        'base': '11px',
        'lg': '12px',
        'xl': '14px',
      },
      spacing: {
        'xs': '4px',
        'sm': '8px',
        'md': '16px',
        'lg': '24px',
        'xl': '32px',
      }
    },
  },
  plugins: [],
}
