import { ExternalLink } from "lucide-react";
import Link from "next/link";

import { SaveButton } from "@/components/save-button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import type { GrantSearchHit } from "@/lib/api";
import { formatDeadline, formatFundingRange } from "@/lib/format";

const PORTAL_COLOR: Record<string, string> = {
  exist: "bg-emerald-100 text-emerald-900 dark:bg-emerald-900/40 dark:text-emerald-100",
  kfw: "bg-blue-100 text-blue-900 dark:bg-blue-900/40 dark:text-blue-100",
  eic: "bg-indigo-100 text-indigo-900 dark:bg-indigo-900/40 dark:text-indigo-100",
  horizon: "bg-violet-100 text-violet-900 dark:bg-violet-900/40 dark:text-violet-100",
  bayern: "bg-amber-100 text-amber-900 dark:bg-amber-900/40 dark:text-amber-100",
  nrw: "bg-rose-100 text-rose-900 dark:bg-rose-900/40 dark:text-rose-100",
  bw: "bg-orange-100 text-orange-900 dark:bg-orange-900/40 dark:text-orange-100",
  bmbf: "bg-cyan-100 text-cyan-900 dark:bg-cyan-900/40 dark:text-cyan-100",
};

interface Props {
  hit: GrantSearchHit;
  rank: number;
}

export function GrantCard({ hit, rank }: Props) {
  const funding = formatFundingRange(hit.funding_min_eur, hit.funding_max_eur);
  const deadline = formatDeadline(hit.deadline);
  const portalClass =
    PORTAL_COLOR[hit.portal] ?? "bg-zinc-100 text-zinc-900";

  return (
    <Card className="transition-colors hover:border-foreground/20">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-medium text-muted-foreground">
              #{rank}
            </span>
            <Badge className={portalClass} variant="secondary">
              {hit.portal.toUpperCase()}
            </Badge>
            <Badge variant="outline" className="font-normal">
              {hit.status.toUpperCase()}
            </Badge>
            {hit.country && (
              <Badge variant="outline" className="font-normal">
                {hit.country}
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-2">
            <div className="text-right text-xs text-muted-foreground tabular-nums">
              <div className="font-medium text-foreground">
                {hit.final_score.toFixed(3)}
              </div>
              <div>score</div>
            </div>
            <SaveButton
              grant={{
                id: hit.id,
                portal: hit.portal,
                title: hit.title,
                source_url: hit.source_url,
                source_doc_id: hit.source_doc_id,
              }}
            />
          </div>
        </div>
        <h3 className="mt-2 text-base font-semibold leading-snug">
          <Link href={`/grants/${hit.id}`} className="hover:underline">
            {hit.title}
          </Link>
        </h3>
        {hit.title_en && hit.title_en !== hit.title && (
          <p className="text-sm text-muted-foreground">{hit.title_en}</p>
        )}
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-sm leading-relaxed text-muted-foreground line-clamp-3">
          {hit.summary}
        </p>
        <Separator />
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-muted-foreground">
          {funding && (
            <span>
              <span className="text-foreground/80">Funding:</span> {funding}
            </span>
          )}
          <span>
            <span className="text-foreground/80">{deadline}</span>
          </span>
          <a
            href={hit.source_url}
            target="_blank"
            rel="noreferrer"
            className="ml-auto inline-flex items-center gap-1 hover:text-foreground hover:underline"
          >
            Source <ExternalLink className="h-3 w-3" />
          </a>
        </div>
        <Provenance hit={hit} />
      </CardContent>
    </Card>
  );
}

function Provenance({ hit }: { hit: GrantSearchHit }) {
  const bits: string[] = [];
  if (hit.dense_rank != null) bits.push(`dense #${hit.dense_rank}`);
  if (hit.sparse_rank != null) bits.push(`sparse #${hit.sparse_rank}`);
  if (hit.rrf_score != null) bits.push(`RRF ${hit.rrf_score.toFixed(3)}`);
  if (hit.rerank_score != null) bits.push(`rerank ${hit.rerank_score.toFixed(2)}`);
  if (bits.length === 0) return null;
  return (
    <div className="text-[11px] text-muted-foreground/80">
      {bits.join(" · ")}
    </div>
  );
}
