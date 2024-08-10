/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{html,ts}",
  ],
  plugins: [],
  theme:{
    extend: {
      colors: {
        'not-selected-green':'#59f761',
        'selected-green':'#4ade80',
        'text-green':'#4ade80',
        'text-red':'#dc2626',
      },
    }
  }
}

