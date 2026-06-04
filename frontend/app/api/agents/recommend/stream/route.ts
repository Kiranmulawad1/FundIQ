/**
 * Streaming proxy for the long-pole agent endpoint.
 *
 * Same rationale as ../route.ts (the batch variant): Next 16's
 * `rewrites()` enforces a 30s undici timeout, which kills SSE streams
 * mid-flight. Here we proxy the upstream response body directly so the
 * client sees Server-Sent Events as they arrive.
 *
 * `runtime: "nodejs"` keeps us on the Node runtime where ReadableStream
 * passthrough works without Edge-runtime SSE quirks.
 */

import type { NextRequest } from "next/server";

const BACKEND = process.env.BACKEND_INTERNAL_URL ?? "http://localhost:8000";

// 3-minute ceiling — matches the batch handler.
const AGENT_TIMEOUT_MS = 180_000;

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: NextRequest): Promise<Response> {
  const body = await req.text();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), AGENT_TIMEOUT_MS);

  // Forward the Authorization header proxy.ts already injected for
  // signed-in users (see ../route.ts for the same plumbing on batch).
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  const auth = req.headers.get("authorization");
  if (auth) headers["Authorization"] = auth;

  const upstream = await fetch(`${BACKEND}/agents/recommend/stream`, {
    method: "POST",
    headers,
    body,
    signal: controller.signal,
    cache: "no-store",
  }).catch((err) => {
    clearTimeout(timer);
    throw err;
  });

  // We pass through the upstream body as-is. The browser sees a real
  // text/event-stream that it can read with a ReadableStream reader.
  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "Content-Type":
        upstream.headers.get("content-type") ?? "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      // Telling Next/Vercel not to buffer.
      "X-Accel-Buffering": "no",
    },
  });
}
