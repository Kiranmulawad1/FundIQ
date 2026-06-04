import Link from "next/link";

import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface Props {
  total: number;
  limit: number;
  offset: number;
  searchParams: Record<string, string | undefined>;
}

export function GrantPagination({ total, limit, offset, searchParams }: Props) {
  const page = Math.floor(offset / limit) + 1;
  const totalPages = Math.max(1, Math.ceil(total / limit));
  const prevOffset = Math.max(0, offset - limit);
  const nextOffset = offset + limit;
  const hasPrev = offset > 0;
  const hasNext = nextOffset < total;

  const linkFor = (newOffset: number) => {
    const params = new URLSearchParams();
    for (const [k, v] of Object.entries(searchParams)) {
      if (v !== undefined && k !== "offset") params.set(k, v);
    }
    if (newOffset > 0) params.set("offset", String(newOffset));
    return `?${params.toString()}`;
  };

  const disabled = "pointer-events-none opacity-50";

  return (
    <div className="flex items-center justify-between gap-4 text-sm text-muted-foreground">
      <span>
        Page {page} of {totalPages} · {total} total
      </span>
      <div className="flex gap-2">
        <Link
          href={linkFor(prevOffset)}
          className={cn(
            buttonVariants({ variant: "outline", size: "sm" }),
            !hasPrev && disabled,
          )}
          aria-disabled={!hasPrev}
        >
          ← Prev
        </Link>
        <Link
          href={linkFor(nextOffset)}
          className={cn(
            buttonVariants({ variant: "outline", size: "sm" }),
            !hasNext && disabled,
          )}
          aria-disabled={!hasNext}
        >
          Next →
        </Link>
      </div>
    </div>
  );
}
