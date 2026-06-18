import type { NextConfig } from "next";

const isProd = process.env.NODE_ENV === "production";

const nextConfig: NextConfig = {
  output: "export",
  // If you are deploying to https://<username>.github.io/<repository-name>/,
  // uncomment the lines below and replace 'repository-name' with your repository name:
  // basePath: isProd ? "/repository-name" : undefined,
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
