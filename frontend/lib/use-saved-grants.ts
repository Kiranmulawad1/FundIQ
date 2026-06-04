"use client";

import { useCallback, useEffect, useState } from "react";

const STORAGE_KEY = "fundiq:saved-grants:v1";
const EVENT_NAME = "fundiq:saved-changed";

export interface SavedGrant {
  id: string;
  portal: string;
  title: string;
  source_url: string;
  source_doc_id: string | null;
  savedAt: number;
}

function readStorage(): SavedGrant[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (e): e is SavedGrant =>
        typeof e === "object"
        && e !== null
        && typeof (e as SavedGrant).id === "string"
        && typeof (e as SavedGrant).title === "string"
        && typeof (e as SavedGrant).savedAt === "number",
    );
  } catch {
    return [];
  }
}

function writeStorage(value: SavedGrant[]): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(value));
    // Notify other live components on this tab (the `storage` event only
    // fires cross-tab). Without this the count badge wouldn't update when
    // you click the star from the search page.
    window.dispatchEvent(new CustomEvent(EVENT_NAME));
  } catch {
    // Quota / privacy mode — ignore. Saving grants is a convenience.
  }
}

/**
 * SSR-safe saved-grants store. Same shape as useRecentSearches: empty on
 * first render, rehydrate on mount, broadcast cross-component updates via
 * a CustomEvent (since localStorage's `storage` event is cross-tab only).
 */
export function useSavedGrants() {
  const [items, setItems] = useState<SavedGrant[]>([]);

  useEffect(() => {
    setItems(readStorage());
    const sync = () => setItems(readStorage());
    window.addEventListener(EVENT_NAME, sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener(EVENT_NAME, sync);
      window.removeEventListener("storage", sync);
    };
  }, []);

  const isSaved = useCallback(
    (id: string) => items.some((e) => e.id === id),
    [items],
  );

  const toggle = useCallback((entry: Omit<SavedGrant, "savedAt">) => {
    setItems((prev) => {
      const existing = prev.find((e) => e.id === entry.id);
      const next = existing
        ? prev.filter((e) => e.id !== entry.id)
        : [{ ...entry, savedAt: Date.now() }, ...prev];
      writeStorage(next);
      return next;
    });
  }, []);

  const remove = useCallback((id: string) => {
    setItems((prev) => {
      const next = prev.filter((e) => e.id !== id);
      writeStorage(next);
      return next;
    });
  }, []);

  const clear = useCallback(() => {
    setItems([]);
    writeStorage([]);
  }, []);

  return { items, isSaved, toggle, remove, clear };
}
