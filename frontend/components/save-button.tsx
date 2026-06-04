"use client";

import { Bookmark, BookmarkCheck } from "lucide-react";

import { Button } from "@/components/ui/button";
import { type SavedGrant, useSavedGrants } from "@/lib/use-saved-grants";
import { cn } from "@/lib/utils";

interface Props {
  grant: Omit<SavedGrant, "savedAt">;
  size?: "icon" | "icon-sm";
  /** When true, render the button with a visible label ("Save" / "Saved"). */
  withLabel?: boolean;
}

export function SaveButton({ grant, size = "icon-sm", withLabel = false }: Props) {
  const { isSaved, toggle } = useSavedGrants();
  const saved = isSaved(grant.id);

  return (
    <Button
      type="button"
      variant={saved ? "secondary" : "ghost"}
      size={withLabel ? "sm" : size}
      onClick={(e) => {
        // Defend against propagation eating the click on cards that wrap a
        // link — the parent `<Link>` would navigate before our handler ran.
        e.preventDefault();
        e.stopPropagation();
        toggle(grant);
      }}
      aria-pressed={saved}
      aria-label={saved ? `Unsave ${grant.title}` : `Save ${grant.title}`}
      title={saved ? "Saved — click to remove" : "Save to shortlist"}
      className={cn(
        saved && "text-foreground",
        withLabel && "gap-1.5",
      )}
    >
      {saved ? (
        <BookmarkCheck className="h-4 w-4" />
      ) : (
        <Bookmark className="h-4 w-4" />
      )}
      {withLabel && (
        <span className="text-xs">{saved ? "Saved" : "Save"}</span>
      )}
    </Button>
  );
}
