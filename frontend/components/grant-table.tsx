import { ArrowDown, ArrowUp } from "lucide-react";
import Link from "next/link";

import { SaveButton } from "@/components/save-button";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { GrantListItem, GrantSortKey } from "@/lib/api";
import { formatDeadline, formatFundingRange } from "@/lib/format";
import { cn } from "@/lib/utils";

interface Props {
  items: GrantListItem[];
  sort: GrantSortKey;
  searchParams: Record<string, string | undefined>;
}

// Per backend (app/api/routes/grants.py): created_at desc, deadline asc,
// funding_max desc. Direction isn't toggleable on the wire, so we just
// show the implicit direction next to the active column.
const SORT_DIRECTION: Record<GrantSortKey, "asc" | "desc"> = {
  created_at: "desc",
  deadline: "asc",
  funding_max: "desc",
};

export function GrantTable({ items, sort, searchParams }: Props) {
  return (
    <div className="overflow-hidden rounded-xl border border-border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-24">Portal</TableHead>
            <TableHead>Title</TableHead>
            <TableHead className="w-24">Status</TableHead>
            <SortableHead
              column="funding_max"
              currentSort={sort}
              searchParams={searchParams}
              className="w-40 text-right"
            >
              Funding
            </SortableHead>
            <SortableHead
              column="deadline"
              currentSort={sort}
              searchParams={searchParams}
              className="w-44"
            >
              Deadline
            </SortableHead>
            <SortableHead
              column="created_at"
              currentSort={sort}
              searchParams={searchParams}
              className="w-32"
            >
              Indexed
            </SortableHead>
            <TableHead className="w-12" />
          </TableRow>
        </TableHeader>
        <TableBody>
          {items.map((g) => (
            <TableRow key={g.id}>
              <TableCell>
                <Badge variant="secondary" className="uppercase">
                  {g.portal}
                </Badge>
              </TableCell>
              <TableCell className="max-w-xl">
                <Link
                  href={`/grants/${g.id}`}
                  className="font-medium hover:underline"
                >
                  {g.title}
                </Link>
                {g.summary && (
                  <p className="text-xs text-muted-foreground line-clamp-1">
                    {g.summary}
                  </p>
                )}
              </TableCell>
              <TableCell>
                <Badge variant="outline" className="uppercase font-normal">
                  {g.status}
                </Badge>
              </TableCell>
              <TableCell className="text-right tabular-nums">
                {formatFundingRange(g.funding_min_eur, g.funding_max_eur) ?? (
                  <span className="text-muted-foreground">—</span>
                )}
              </TableCell>
              <TableCell className="text-sm text-muted-foreground">
                {formatDeadline(g.deadline)}
              </TableCell>
              <TableCell className="text-sm text-muted-foreground tabular-nums">
                {new Date(g.created_at).toLocaleDateString("en-GB", {
                  day: "2-digit",
                  month: "short",
                  year: "2-digit",
                })}
              </TableCell>
              <TableCell className="text-right">
                <SaveButton
                  grant={{
                    id: g.id,
                    portal: g.portal,
                    title: g.title,
                    source_url: g.source_url,
                    source_doc_id: g.source_doc_id,
                  }}
                />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

interface SortableHeadProps {
  column: GrantSortKey;
  currentSort: GrantSortKey;
  searchParams: Record<string, string | undefined>;
  className?: string;
  children: React.ReactNode;
}

function SortableHead({
  column,
  currentSort,
  searchParams,
  className,
  children,
}: SortableHeadProps) {
  const isActive = currentSort === column;
  const direction = SORT_DIRECTION[column];

  // Build the URL preserving every other filter; reset offset because the
  // first matching row changes on re-sort.
  const params = new URLSearchParams();
  for (const [k, v] of Object.entries(searchParams)) {
    if (v !== undefined && k !== "sort" && k !== "offset") params.set(k, v);
  }
  // Don't write the default in the URL — keeps shared links tidy.
  if (column !== "created_at") params.set("sort", column);
  const href = params.size > 0 ? `?${params.toString()}` : "?";

  return (
    <TableHead className={className}>
      <Link
        href={href}
        replace
        scroll={false}
        className={cn(
          "inline-flex items-center gap-1 hover:text-foreground",
          isActive ? "text-foreground" : "text-muted-foreground",
        )}
        aria-sort={isActive ? (direction === "asc" ? "ascending" : "descending") : "none"}
      >
        {children}
        {isActive && (
          direction === "asc" ? (
            <ArrowUp className="h-3 w-3" />
          ) : (
            <ArrowDown className="h-3 w-3" />
          )
        )}
      </Link>
    </TableHead>
  );
}
