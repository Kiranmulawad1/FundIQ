import { ArrowLeft, ExternalLink } from "lucide-react";
import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";

import { SaveButton } from "@/components/save-button";
import { Badge } from "@/components/ui/badge";
import { buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { type GrantDetail, getGrant } from "@/lib/api";
import { formatDeadline, formatFundingRange } from "@/lib/format";

interface PageProps {
  params: Promise<{ id: string }>;
}

export async function generateMetadata(
  { params }: PageProps,
): Promise<Metadata> {
  const { id } = await params;
  const grant = await getGrant(id).catch(() => null);
  if (!grant) {
    return { title: "Grant not found · FundIQ" };
  }
  return {
    title: `${grant.title} · FundIQ`,
    description: grant.summary.slice(0, 200),
  };
}

export default async function GrantDetailPage({ params }: PageProps) {
  const { id } = await params;
  const grant = await getGrant(id);
  if (!grant) notFound();

  const funding = formatFundingRange(grant.funding_min_eur, grant.funding_max_eur);
  const eligibilityEntries = Object.entries(grant.eligibility ?? {});

  return (
    <main className="mx-auto w-full max-w-3xl px-4 py-8 sm:py-12">
      <div className="mb-6">
        <Link
          href="/"
          className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Back to search
        </Link>
      </div>

      <article className="space-y-8">
        <header className="space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="secondary" className="uppercase">
              {grant.portal}
            </Badge>
            <Badge variant="outline" className="uppercase">
              {grant.status}
            </Badge>
            <Badge variant="outline">{grant.country}</Badge>
            {grant.federal_state && (
              <Badge variant="outline">{grant.federal_state}</Badge>
            )}
            <div className="ml-auto">
              <SaveButton
                grant={{
                  id: grant.id,
                  portal: grant.portal,
                  title: grant.title,
                  source_url: grant.source_url,
                  source_doc_id: grant.source_doc_id,
                }}
                withLabel
              />
            </div>
          </div>
          <h1 className="text-3xl font-semibold leading-tight tracking-tight">
            {grant.title}
          </h1>
          {grant.title_en && grant.title_en !== grant.title && (
            <p className="text-lg text-muted-foreground">{grant.title_en}</p>
          )}
        </header>

        <FactsRow grant={grant} funding={funding} />

        <section className="space-y-3">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
            Summary
          </h2>
          <p className="leading-relaxed">{grant.summary}</p>
          {grant.summary_en && grant.summary_en !== grant.summary && (
            <p className="text-sm leading-relaxed text-muted-foreground">
              {grant.summary_en}
            </p>
          )}
        </section>

        {grant.body && grant.body.trim().length > 0 && (
          <section className="space-y-3">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
              Full text
            </h2>
            <div className="whitespace-pre-wrap rounded-lg border border-border/60 bg-muted/30 p-4 text-sm leading-relaxed">
              {grant.body}
            </div>
          </section>
        )}

        {eligibilityEntries.length > 0 && (
          <section className="space-y-3">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
              Eligibility
            </h2>
            <Card>
              <CardContent className="py-4">
                <dl className="grid grid-cols-1 gap-x-6 gap-y-2 text-sm sm:grid-cols-[max-content_1fr]">
                  {eligibilityEntries.map(([k, v]) => (
                    <div key={k} className="contents">
                      <dt className="font-medium text-muted-foreground">{k}</dt>
                      <dd className="text-foreground">{renderValue(v)}</dd>
                    </div>
                  ))}
                </dl>
              </CardContent>
            </Card>
          </section>
        )}

        <Separator />

        <footer className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
          <span>Source: {grant.source_doc_id ?? "—"}</span>
          {grant.created_at && (
            <>
              <span>·</span>
              <span>Indexed {new Date(grant.created_at).toLocaleDateString()}</span>
            </>
          )}
        </footer>
      </article>
    </main>
  );
}

function FactsRow({
  grant,
  funding,
}: {
  grant: GrantDetail;
  funding: string | null;
}) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          At a glance
        </h2>
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <Fact label="Funding">{funding ?? "—"}</Fact>
          <Fact label="Deadline">{formatDeadline(grant.deadline)}</Fact>
          <Fact label="Source">
            <a
              href={grant.source_url}
              target="_blank"
              rel="noreferrer"
              className={buttonVariants({ variant: "outline", size: "sm" })}
            >
              Open <ExternalLink className="ml-1 h-3 w-3" />
            </a>
          </Fact>
        </dl>
      </CardContent>
    </Card>
  );
}

function Fact({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1">
      <dt className="text-xs uppercase tracking-wide text-muted-foreground">
        {label}
      </dt>
      <dd className="text-sm">{children}</dd>
    </div>
  );
}

function renderValue(v: unknown): React.ReactNode {
  if (v == null) return <span className="text-muted-foreground">—</span>;
  if (Array.isArray(v)) {
    return (
      <ul className="list-inside list-disc space-y-0.5">
        {v.map((item, i) => (
          <li key={i}>{String(item)}</li>
        ))}
      </ul>
    );
  }
  if (typeof v === "object") {
    return (
      <pre className="overflow-x-auto rounded bg-muted/50 p-2 text-xs">
        {JSON.stringify(v, null, 2)}
      </pre>
    );
  }
  if (typeof v === "boolean") return v ? "Yes" : "No";
  return String(v);
}
