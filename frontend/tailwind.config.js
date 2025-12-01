/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        medical: {
          bg: '#F7FBFC',
          primary: '#4BA3C3',
          secondary: '#9AD4D6',
          accent: '#2D6A7E',
          text: '#083344',
          'bg-dark': '#0F172A',
          'primary-dark': '#5BB3D3',
          'secondary-dark': '#7AB8BA',
          'accent-dark': '#3D8A9E',
          'text-dark': '#E2E8F0',
        },
      },
    },
  },
  plugins: [],
}

