"use client";

import { useEffect, useState } from "react";

/**
 * Resolves the user's primary modifier key for cross-platform shortcuts.
 * Stays `null` on the server so we don't mismatch SSR/CSR; consumers should
 * render a stable placeholder until mounted.
 */
export function useIsMac(): boolean | null {
  const [isMac, setIsMac] = useState<boolean | null>(null);
  useEffect(() => {
    if (typeof navigator === "undefined") {
      setIsMac(false);
      return;
    }
    // userAgentData is the modern API; fall back to platform string.
    const ua = navigator.userAgent || "";
    setIsMac(/mac|iphone|ipad|ipod/i.test(ua));
  }, []);
  return isMac;
}
