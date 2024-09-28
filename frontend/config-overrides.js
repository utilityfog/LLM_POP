import { override } from 'customize-cra';
const addIgnoreWarnings = (_loaderOptions) => config => {
    // https://github.com/facebook/create-react-app/blob/0a827f69ab0d2ee3871ba9b71350031d8a81b7ae/packages/react-scripts/config/webpack.config.js#L353
    const loader = config.module.rules.find(({
      enforce, loader
    }) => enforce === 'pre' && typeof loader === 'string' && loader.includes('source-map-loader'));
    if (loader) {
      if (!config.ignoreWarnings) {
        config.ignoreWarnings = [];
      }
      config.ignoreWarnings.push(/Failed to parse source map/,)
    }
    return config
  }
  
  
export default {
    webpack: override(
        // ...
        addIgnoreWarnings(),
    )
};
