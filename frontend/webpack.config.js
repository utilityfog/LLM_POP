const webpack = require('webpack');
const NodePolyfillPlugin = require("node-polyfill-webpack-plugin");

module.exports = {
    module: {
        rules: [
            {
                test: /\.js$/,
                use: ['source-map-loader'],
                enforce: 'pre',
                exclude: /node_modules\/@microsoft\/fetch-event-source/, // Correctly exclude source maps for this package
            },
            // other rules...
        ],
    },
    plugins: [
        new NodePolyfillPlugin(),
        new webpack.ProvidePlugin({
            process: 'process/browser',
            Buffer: ['buffer', 'Buffer'],
        }),
    ],
    resolve: {
        fallback: {
            "fs": false,
            "path": require.resolve("path-browserify"),
            "http": require.resolve("stream-http"),
            "https": require.resolve("https-browserify"),
            "os": require.resolve("os-browserify/browser"),
            "crypto": require.resolve("crypto-browserify"),
            "stream": require.resolve("stream-browserify"),
            "constants": require.resolve("constants-browserify"),
            "zlib": require.resolve("browserify-zlib"),
            "child_process": false,
            "net": false,
            "tls": false,
            "dns": false,
            "module": false,
            "readline": false,
            "perf_hooks": false,
        }
    }
};
