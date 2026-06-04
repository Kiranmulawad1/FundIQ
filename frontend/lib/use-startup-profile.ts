"use client";

import { useCallback, useEffect, useState } from "react";

const STORAGE_KEY = "fundiq:startup-profile:v1";
const EVENT_NAME = "fundiq:profile-changed";

export interface StartupProfile {
  name?: string;
  sector?: string;
  stage?: string;
  country?: string;
  federal_state?: string;
  funding_target_eur?: number;
  description?: string;
}

export const PROFILE_SECTORS = [
  "deeptech",
  "cleantech",
  "health",
  "biotech",
  "saas",
  "hardware",
  "fintech",
  "other",
] as const;

export const PROFILE_STAGES = [
  { value: "idea", label: "Idea / pre-revenue" },
  { value: "seed", label: "Seed / early traction" },
  { value: "growth", label: "Growth / scaling" },
] as const;

function readStorage(): StartupProfile | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    return parsed as StartupProfile;
  } catch {
    return null;
  }
}

function writeStorage(value: StartupProfile | null): void {
  if (typeof window === "undefined") return;
  try {
    if (value === null) {
      window.localStorage.removeItem(STORAGE_KEY);
    } else {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(value));
    }
    window.dispatchEvent(new CustomEvent(EVENT_NAME));
  } catch {
    // ignore quota / privacy mode
  }
}

/**
 * SSR-safe per-browser startup profile. Used to pre-condition the
 * Planner in the agent graph — saved once on /profile, sent with every
 * /agents/recommend POST so the Planner trusts these facts without
 * re-extracting them from the query.
 *
 * Cross-component updates (e.g. nav badge or recommend page) listen on
 * a CustomEvent because localStorage's `storage` event only fires
 * cross-tab.
 */
export function useStartupProfile() {
  const [profile, setProfileState] = useState<StartupProfile | null>(null);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    setProfileState(readStorage());
    setHydrated(true);
    const sync = () => setProfileState(readStorage());
    window.addEventListener(EVENT_NAME, sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener(EVENT_NAME, sync);
      window.removeEventListener("storage", sync);
    };
  }, []);

  const save = useCallback((value: StartupProfile) => {
    // Strip empty strings so the backend payload only carries real facts.
    const cleaned: StartupProfile = {};
    for (const [k, v] of Object.entries(value)) {
      if (v === "" || v === undefined || v === null) continue;
      cleaned[k as keyof StartupProfile] = v as never;
    }
    setProfileState(cleaned);
    writeStorage(cleaned);
  }, []);

  const clear = useCallback(() => {
    setProfileState(null);
    writeStorage(null);
  }, []);

  return { profile, hydrated, save, clear };
}
