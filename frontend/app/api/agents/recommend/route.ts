/**
 * Explicit POST proxy for the long-pole agent endpoint.
 *
 * Why this exists: Next 16's `rewrites()` go through undici with a 30s
 * headers/body timeout. The agent graph takes 5-90s end-to-end
 * (Gemini Planner ~3s + RAG retrieval ~1-50s + Gemini Writer ~10-20s),
 * which the default rewrite kills with a 500.
 *
 * This handler routes the call through Node's fetch with an AbortSignal
 * we control. All other `/api/*` traffic stays on the lighter rewrites
 * path; only this one route opts out.
 */

import { NextResponse } from "next/server";

const BACKEND = process.env.BACKEND_INTERNAL_URL ?? "http://localhost:8000";

// 3-minute ceiling — generous enough for cold BGE reranker loads.
const AGENT_TIMEOUT_MS = 180_000;

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: Request) {
  const body = await req.text();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), AGENT_TIMEOUT_MS);
  try {
    // Forward the Authorization header that proxy.ts already injected
    // for signed-in users. The backend reads it via current_user /
    // optional_user. Signed-out requests carry no header — backend
    // falls through to anonymous handling.
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    const auth = req.headers.get("authorization");
    if (auth) headers["Authorization"] = auth;

    const upstream = await fetch(`${BACKEND}/agents/recommend`, {
      method: "POST",
      headers,
      body,
      signal: controller.signal,
      cache: "no-store",
    });
    const text = await upstream.text();
    return new NextResponse(text, {
      status: upstream.status,
      headers: { "Content-Type": upstream.headers.get("content-type") ?? "application/json" },
    });
  } catch (err) {
    const aborted = err instanceof DOMException && err.name === "AbortError";
    const message = aborted
      ? `Agent request exceeded ${AGENT_TIMEOUT_MS / 1000}s timeout.`
      : err instanceof Error
        ? err.message
        : "Upstream agent request failed.";
    return NextResponse.json({ detail: message }, { status: 504 });
  } finally {
    clearTimeout(timer);
  }
}
