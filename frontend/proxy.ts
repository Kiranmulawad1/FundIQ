/**
 * Clerk auth wiring — runs before every request, attaches an auth
 * context so server components / route handlers can call `auth()` and
 * get the current session.
 *
 * Next.js 16 renamed Middleware → Proxy (same runtime, clearer name).
 * Clerk's clerkMiddleware() returns a Next-compatible middleware fn
 * which works as the default Proxy export unchanged.
 *
 * We don't gate any routes here yet. Backend auth enforcement (slice 3
 * of the deployment+CI+auth bundle) will add a `createRouteMatcher`
 * for /agents/* + /admin/* and call `auth.protect()` on matches.
 */

import { clerkMiddleware } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";

/**
 * Clerk middleware + Authorization-header injection.
 *
 * Browser requests carry a Clerk session cookie. The Python backend
 * doesn't read that cookie — it reads a Bearer JWT from the
 * `Authorization` header (see `app/core/auth.py`). So we transparently
 * mint a session token here and forward it to the backend on every
 * `/api/*` request. Pages and route handlers don't need to know about
 * this; the Clerk SDK already attaches the cookie, and we attach the
 * Bearer token before the rewrite (or before our explicit route
 * handlers) sees the request.
 *
 * Signed-out visitors: `getToken()` returns null, no header added,
 * backend's `optional_user` resolves to None — same anonymous flow as
 * before.
 */
export default clerkMiddleware(async (authFn, req) => {
  // Only touch /api/* — pages don't need a header (Clerk reads the
  // cookie directly via ClerkProvider).
  if (!req.nextUrl.pathname.startsWith("/api/")) return;

  const { getToken } = await authFn();
  const token = await getToken();
  if (!token) return; // signed-out: leave the request untouched

  const headers = new Headers(req.headers);
  headers.set("Authorization", `Bearer ${token}`);
  return NextResponse.next({ request: { headers } });
});

export const config = {
  // Run on every page + API route EXCEPT Next's internals + static
  // assets. The recommended matcher from Clerk's docs, adjusted for our
  // tree: /api/* routes still go through so future protected endpoints
  // (saved-grants sync, agent sessions) get the auth context for free.
  matcher: [
    // Skip Next.js internals + static files unless found in search params.
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)",
    // Always run for API routes.
    "/(api|trpc)(.*)",
    // Clerk's auto-proxy path — handshake redirects + session refresh.
    // Required by the Clerk Next.js setup guide; omitting this means
    // some OAuth callbacks 404 silently.
    "/__clerk/:path*",
  ],
};
