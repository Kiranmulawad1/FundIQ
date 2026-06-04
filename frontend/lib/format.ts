/**
 * Display helpers — keep formatting decisions in one place so the UI
 * stays consistent across cards, list rows, and the (future) detail page.
 */

export function formatEur(amount: number | null | undefined): string | null {
  if (amount == null) return null;
  if (amount >= 1_000_000) {
    const m = amount / 1_000_000;
    return `€${m % 1 === 0 ? m.toFixed(0) : m.toFixed(1)}M`;
  }
  if (amount >= 1_000) return `€${Math.round(amount / 1_000)}k`;
  return `€${amount}`;
}

export function formatFundingRange(
  min: number | null | undefined,
  max: number | null | undefined,
): string | null {
  const lo = formatEur(min);
  const hi = formatEur(max);
  if (lo && hi) return `${lo} – ${hi}`;
  return hi ?? lo ?? null;
}

export function formatDeadline(iso: string | null | undefined): string {
  if (!iso) return "Rolling / open-ended";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const now = Date.now();
  const days = Math.round((d.getTime() - now) / 86_400_000);
  const fmt = d.toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
  if (days < 0) return `Closed ${fmt}`;
  if (days <= 30) return `${fmt} (${days}d left)`;
  return `Deadline ${fmt}`;
}
