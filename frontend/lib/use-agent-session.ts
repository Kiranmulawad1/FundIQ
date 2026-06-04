"use client";

import { useCallback, useEffect, useState } from "react";

const STORAGE_KEY = "fundiq:agent-session-id:v1";

/**
 * Per-browser anonymous chat-session id. The backend uses this to scope
 * `AgentSession` rows; when Clerk auth lands the same UUID continues to
 * work (we'll backfill `owner_user_id` at first sign-in).
 *
 * SSR-safe: starts as `null` on first render, rehydrates from
 * localStorage on mount.
 */
export function useAgentSession() {
  const [sessionId, setSessionIdState] = useState<string | null>(null);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (raw) setSessionIdState(raw);
    } catch {
      // localStorage may be unavailable (privacy mode). Continue without
      // persistence; each request will allocate a fresh session.
    }
    setHydrated(true);
  }, []);

  const setSessionId = useCallback((id: string | null) => {
    setSessionIdState(id);
    try {
      if (id) window.localStorage.setItem(STORAGE_KEY, id);
      else window.localStorage.removeItem(STORAGE_KEY);
    } catch {
      // ignored
    }
  }, []);

  const clear = useCallback(() => setSessionId(null), [setSessionId]);

  return { sessionId, setSessionId, clear, hydrated };
}
