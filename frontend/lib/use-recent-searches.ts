"use client";

import { useCallback, useEffect, useState } from "react";

import type { RetrievalMode } from "@/lib/api";

const STORAGE_KEY = "fundiq:recent-searches:v1";
const MAX_ENTRIES = 8;

export interface RecentSearch {
  query: string;
  mode: RetrievalMode;
  useHyde: boolean;
  ts: number;
}

function readStorage(): RecentSearch[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    // Defensive — drop entries that aren't shaped right (older versions, etc.).
    return parsed.filter(
      (e): e is RecentSearch =>
        typeof e === "object"
        && e !== null
        && typeof (e as RecentSearch).query === "string"
        && typeof (e as RecentSearch).ts === "number",
    );
  } catch {
    return [];
  }
}

function writeStorage(value: RecentSearch[]): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(value));
  } catch {
    // Quota / privacy mode — silently ignore. History is a convenience.
  }
}

/**
 * SSR-safe recent-searches store.
 *
 * Starts empty on first render so server + client HTML match; rehydrates
 * from localStorage on mount. `add` deduplicates by query (case-insensitive
 * trim) so spamming the same query doesn't crowd the list.
 */
export function useRecentSearches() {
  const [items, setItems] = useState<RecentSearch[]>([]);

  useEffect(() => {
    setItems(readStorage());
  }, []);

  const add = useCallback((entry: Omit<RecentSearch, "ts">) => {
    const trimmed = entry.query.trim();
    if (!trimmed) return;
    setItems((prev) => {
      const norm = trimmed.toLowerCase();
      const filtered = prev.filter((e) => e.query.trim().toLowerCase() !== norm);
      const next: RecentSearch[] = [
        { ...entry, query: trimmed, ts: Date.now() },
        ...filtered,
      ].slice(0, MAX_ENTRIES);
      writeStorage(next);
      return next;
    });
  }, []);

  const clear = useCallback(() => {
    setItems([]);
    writeStorage([]);
  }, []);

  return { items, add, clear };
}
