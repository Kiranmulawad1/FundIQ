import { GrantFilters } from "@/components/grant-filters";
import { GrantPagination } from "@/components/grant-pagination";
import { GrantRow } from "@/components/grant-row";
import { GrantTable } from "@/components/grant-table";
import { Card, CardContent } from "@/components/ui/card";
import {
  type GrantPortal,
  type GrantSortKey,
  type GrantStatus,
  listGrants,
} from "@/lib/api";

const PAGE_LIMIT = 20;

const VALID_PORTALS: GrantPortal[] = [
  "bmbf", "exist", "kfw", "eic", "horizon", "bayern", "nrw", "bw",
];
const VALID_STATUSES: GrantStatus[] = ["open", "closed", "upcoming", "rolling"];
const VALID_SORTS: GrantSortKey[] = ["created_at", "deadline", "funding_max"];

interface PageProps {
  searchParams: Promise<{
    portal?: string;
    status?: string;
    country?: string;
    sort?: string;
    offset?: string;
  }>;
}

function parseEnum<T extends string>(
  raw: string | undefined,
  valid: readonly T[],
): T | undefined {
  if (!raw) return undefined;
  return (valid as readonly string[]).includes(raw) ? (raw as T) : undefined;
}

export default async function BrowsePage({ searchParams }: PageProps) {
  const sp = await searchParams;
  const portal = parseEnum(sp.portal, VALID_PORTALS);
  const status = parseEnum(sp.status, VALID_STATUSES);
  const country = sp.country?.length === 2 ? sp.country.toUpperCase() : undefined;
  const sort = parseEnum(sp.sort, VALID_SORTS) ?? "created_at";
  const offsetRaw = Number.parseInt(sp.offset ?? "0", 10);
  const offset = Number.isFinite(offsetRaw) && offsetRaw >= 0 ? offsetRaw : 0;

  const data = await listGrants({
    portal,
    status,
    country,
    sort,
    limit: PAGE_LIMIT,
    offset,
  });

  // Snapshot for the pagination component — only string-valued params.
  const spForPagination: Record<string, string | undefined> = {
    portal: portal ?? undefined,
    status: status ?? undefined,
    country: country ?? undefined,
    sort: sort === "created_at" ? undefined : sort,
  };

  return (
    <main className="mx-auto w-full max-w-5xl space-y-6 px-4 py-8 sm:py-12">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">
          Browse grants
        </h1>
        <p className="text-sm text-muted-foreground">
          Filter the full corpus by portal, status, country, or funding size.
          {" "}
          For semantic queries use{" "}
          <a href="/" className="underline hover:text-foreground">
            search
          </a>
          .
        </p>
      </header>

      <GrantFilters
        portal={portal}
        status={status}
        country={country}
        sort={sort}
      />

      {data.items.length === 0 ? (
        <Card className="border-dashed">
          <CardContent className="py-12 text-center text-sm text-muted-foreground">
            No grants match these filters. Try widening them or resetting.
          </CardContent>
        </Card>
      ) : (
        <>
          {/* Table layout on md+ — easier to scan, sortable headers. */}
          <div className="hidden md:block">
            <GrantTable
              items={data.items}
              sort={sort}
              searchParams={spForPagination}
            />
          </div>
          {/* Card layout on mobile — table doesn't fit narrow viewports. */}
          <div className="space-y-3 md:hidden">
            {data.items.map((g) => (
              <GrantRow key={g.id} grant={g} />
            ))}
          </div>
        </>
      )}

      <GrantPagination
        total={data.page.total}
        limit={data.page.limit}
        offset={data.page.offset}
        searchParams={spForPagination}
      />
    </main>
  );
}
