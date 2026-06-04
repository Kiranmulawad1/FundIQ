import { FundingByPortalChart } from "@/components/analytics/funding-by-portal-chart";
import { PortalCountChart } from "@/components/analytics/portal-count-chart";
import { StatCard } from "@/components/analytics/stat-card";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { getFundingAnalytics } from "@/lib/api";
import { formatEur } from "@/lib/format";

export const metadata = { title: "Analytics · FundIQ" };

export default async function AnalyticsPage() {
  const data = await getFundingAnalytics();

  const fundingRange =
    data.funding_global_min != null && data.funding_global_max != null
      ? `${formatEur(data.funding_global_min)} – ${formatEur(data.funding_global_max)}`
      : "—";
  const fundingAvg = data.funding_global_avg != null
    ? formatEur(data.funding_global_avg) ?? "—"
    : "—";
  const embeddedPct = data.total_grants > 0
    ? Math.round((data.embedded_grants / data.total_grants) * 100)
    : 0;

  return (
    <main className="mx-auto w-full max-w-5xl space-y-8 px-4 py-8 sm:py-12">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">
          Funding-corpus analytics
        </h1>
        <p className="text-sm text-muted-foreground">
          Snapshot of the indexed grants corpus, computed via DuckDB attached
          to Postgres ({data.elapsed_ms} ms).
        </p>
      </header>

      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Total grants"
          value={data.total_grants.toLocaleString()}
        />
        <StatCard
          label="RAG-ready"
          value={data.embedded_grants.toLocaleString()}
          hint={`${embeddedPct}% have embeddings`}
        />
        <StatCard
          label="Funding range"
          value={fundingRange}
          hint="Across portals reporting a ceiling"
        />
        <StatCard
          label="Average ceiling"
          value={fundingAvg}
          hint="Mean of funding_max_eur"
        />
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader className="pb-2">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
              Grants by portal
            </h2>
          </CardHeader>
          <CardContent>
            <PortalCountChart data={data.by_portal} />
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
              Avg funding ceiling by portal
            </h2>
          </CardHeader>
          <CardContent>
            <FundingByPortalChart data={data.by_portal} />
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader className="pb-2">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
              By status
            </h2>
          </CardHeader>
          <CardContent>
            <Distribution
              rows={data.by_status.map((s) => ({ label: s.status, n: s.n }))}
              total={data.total_grants}
            />
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
              By federal state
            </h2>
          </CardHeader>
          <CardContent>
            <Distribution
              rows={data.by_federal_state.map((fs) => ({
                label: fs.federal_state || "—",
                n: fs.n,
              }))}
              total={data.total_grants}
            />
          </CardContent>
        </Card>
      </section>

      <Separator />

      <footer className="text-xs text-muted-foreground">
        Computed via {data.computed_via}. Re-run for live numbers — no
        client-side caching on this page.
      </footer>
    </main>
  );
}

function Distribution({
  rows,
  total,
}: {
  rows: { label: string; n: number }[];
  total: number;
}) {
  if (rows.length === 0) {
    return (
      <p className="py-2 text-sm text-muted-foreground">
        No data yet.
      </p>
    );
  }
  const sorted = [...rows].sort((a, b) => b.n - a.n);
  const max = Math.max(...sorted.map((r) => r.n));
  return (
    <ul className="space-y-2">
      {sorted.map((r) => {
        const pct = total > 0 ? Math.round((r.n / total) * 100) : 0;
        const barWidth = max > 0 ? Math.round((r.n / max) * 100) : 0;
        return (
          <li key={r.label} className="space-y-1">
            <div className="flex items-baseline justify-between text-sm">
              <span className="font-medium capitalize">{r.label}</span>
              <span className="text-xs text-muted-foreground tabular-nums">
                {r.n} · {pct}%
              </span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-foreground/70"
                style={{ width: `${barWidth}%` }}
              />
            </div>
          </li>
        );
      })}
    </ul>
  );
}
