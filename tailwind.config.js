module.exports = {
  content: [
    "./templates/**/*.html",
    "./static/**/*.js",
    "./node_modules/flowbite/**/*.js",
  ],
  theme: {
    extend: {
      height: {
        '90%': '90%',
        '144': '36rem',
        '160': '40rem',
      }
    },
  },
  plugins: [
    require("flowbite/plugin")
  ]
}