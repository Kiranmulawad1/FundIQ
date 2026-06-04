"use client";

import { Monitor, Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";

// Cycle: system → light → dark → system. Three states fit one button without
// a popover; users can land on any of them in two clicks. Matches the
// next-themes mental model (system is a real third state, not "default").
const ORDER = ["system", "light", "dark"] as const;
type Theme = (typeof ORDER)[number];

const NEXT: Record<Theme, Theme> = {
  system: "light",
  light: "dark",
  dark: "system",
};

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  // next-themes resolves the active theme on mount; rendering icons before
  // that would flicker between server and client. Render a stable placeholder.
  useEffect(() => {
    setMounted(true);
  }, []);

  const current = (theme as Theme | undefined) ?? "system";
  const next = NEXT[current];

  return (
    <Button
      type="button"
      variant="ghost"
      size="icon"
      aria-label={`Switch to ${next} theme`}
      onClick={() => setTheme(next)}
    >
      {!mounted ? (
        <Sun className="h-4 w-4 opacity-0" />
      ) : current === "system" ? (
        <Monitor className="h-4 w-4" />
      ) : current === "light" ? (
        <Sun className="h-4 w-4" />
      ) : (
        <Moon className="h-4 w-4" />
      )}
    </Button>
  );
}
