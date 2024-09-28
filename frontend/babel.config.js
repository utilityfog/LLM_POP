module.exports = {
  presets: [
      '@babel/preset-env',
      '@babel/preset-react'
  ],
  plugins: [
      'transform-inline-environment-variables'
      ["transform-require-ignore", {
          "extensions": [".scss", ".css"]
      }]
  ]
};
  