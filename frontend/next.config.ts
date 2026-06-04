import type { NextConfig } from "next";

// Reverse-proxy /api/* to the FastAPI backend. Client code always uses
// relative URLs, so there's no CORS surface to manage and the same code
// works in prod once the proxy or ingress is configured one level up.
const backend = process.env.BACKEND_INTERNAL_URL ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${backend}/:path*` },
    ];
  },
};

export default nextConfig;
