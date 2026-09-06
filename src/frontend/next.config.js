const { PHASE_DEVELOPMENT_SERVER } = require('next/constants');

/** @type {import('next').NextConfig} */
const createNextConfig = (phase) => ({
  // Keep `npm run build` from replacing a running development server's chunks.
  distDir: phase === PHASE_DEVELOPMENT_SERVER ? '.next-dev' : '.next',
  typescript: {
    // Next otherwise rewrites the production TypeScript project while dev is running.
    tsconfigPath: phase === PHASE_DEVELOPMENT_SERVER ? 'tsconfig.dev.json' : 'tsconfig.json',
  },
  reactStrictMode: true,
  transpilePackages: ['three'],
  webpack: (config) => {
    config.module.rules.push({
      test: /\.(glsl|vs|fs|vert|frag)$/,
      exclude: /node_modules/,
      use: ['raw-loader']
    });
    return config;
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: process.env.NEXT_PUBLIC_API_URL
          ? `${process.env.NEXT_PUBLIC_API_URL}/:path*`
          : 'http://localhost:8000/:path*'
      }
    ];
  }
});

module.exports = createNextConfig;
