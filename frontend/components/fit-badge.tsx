import type { GrantFit } from "@/lib/api";
import { cn } from "@/lib/utils";

const STYLES: Record<GrantFit, { label: string; className: string }> = {
  high: {
    label: "Strong fit",
    className:
      "bg-emerald-500/15 text-emerald-700 ring-1 ring-emerald-500/30 dark:text-emerald-300",
  },
  medium: {
    label: "Plausible fit",
    className:
      "bg-amber-500/15 text-amber-800 ring-1 ring-amber-500/30 dark:text-amber-200",
  },
  low: {
    label: "Stretch fit",
    className:
      "bg-zinc-500/15 text-zinc-700 ring-1 ring-zinc-500/30 dark:text-zinc-300",
  },
};

export function FitBadge({ fit }: { fit: GrantFit }) {
  const s = STYLES[fit];
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium",
        s.className,
      )}
    >
      {s.label}
    </span>
  );
}
