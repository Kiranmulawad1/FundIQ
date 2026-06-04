"use client";

import Link from "next/link";

import { useSavedGrants } from "@/lib/use-saved-grants";

export function SavedNavLink() {
  const { items } = useSavedGrants();
  const count = items.length;

  return (
    <Link
      href="/saved"
      className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground"
    >
      Saved
      {count > 0 && (
        <span
          className="inline-flex h-4 min-w-4 items-center justify-center rounded-full bg-foreground px-1 text-[10px] font-medium tabular-nums text-background"
          aria-label={`${count} saved`}
        >
          {count}
        </span>
      )}
    </Link>
  );
}
