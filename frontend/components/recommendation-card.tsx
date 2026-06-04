import { ExternalLink } from "lucide-react";
import Link from "next/link";

import { FitBadge } from "@/components/fit-badge";
import { SaveButton } from "@/components/save-button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import type { AgentGrantRecommendation, CandidateScore } from "@/lib/api";

export function RecommendationCard({
  rec,
  rank,
  score,
}: {
  rec: AgentGrantRecommendation;
  rank: number;
  score?: CandidateScore;
}) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-medium text-muted-foreground">
            #{rank}
          </span>
          <Badge variant="secondary" className="uppercase">
            {rec.portal}
          </Badge>
          <FitBadge fit={rec.fit} />
          {score && (
            <span
              className="rounded-full border border-border bg-muted/40 px-2 py-0.5 text-xs font-medium tabular-nums text-muted-foreground"
              title="Eligibility score from the Scorer agent"
            >
              {score.eligibility_score}/100
            </span>
          )}
          <div className="ml-auto">
            <SaveButton
              grant={{
                id: rec.grant_id,
                portal: rec.portal,
                title: rec.grant_title,
                source_url: rec.source_url,
                source_doc_id: null,
              }}
            />
          </div>
        </div>
        <h3 className="mt-2 text-base font-semibold leading-snug">
          <Link
            href={`/grants/${rec.grant_id}`}
            className="hover:underline"
          >
            {rec.grant_title}
          </Link>
        </h3>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-sm leading-relaxed">{rec.rationale}</p>
        {score && (score.strengths.length > 0 || score.missing_info.length > 0) && (
          <>
            <Separator />
            <div className="grid gap-3 sm:grid-cols-2">
              {score.strengths.length > 0 && (
                <div className="space-y-1.5">
                  <div className="text-[11px] uppercase tracking-wide text-emerald-700 dark:text-emerald-300">
                    Why this fits
                  </div>
                  <ul className="ml-4 list-disc space-y-1 text-sm text-muted-foreground">
                    {score.strengths.map((s, i) => (
                      <li key={i}>{s}</li>
                    ))}
                  </ul>
                </div>
              )}
              {score.missing_info.length > 0 && (
                <div className="space-y-1.5">
                  <div className="text-[11px] uppercase tracking-wide text-amber-700 dark:text-amber-300">
                    Need to clarify
                  </div>
                  <ul className="ml-4 list-disc space-y-1 text-sm text-muted-foreground">
                    {score.missing_info.map((m, i) => (
                      <li key={i}>{m}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </>
        )}
        {rec.caveats.length > 0 && (
          <>
            <Separator />
            <div className="space-y-1.5">
              <div className="text-[11px] uppercase tracking-wide text-muted-foreground">
                Things to verify
              </div>
              <ul className="ml-4 list-disc space-y-1 text-sm text-muted-foreground">
                {rec.caveats.map((c, i) => (
                  <li key={i}>{c}</li>
                ))}
              </ul>
            </div>
          </>
        )}
        <div className="flex items-center gap-3 pt-1 text-xs text-muted-foreground">
          <Link
            href={`/grants/${rec.grant_id}`}
            className="hover:text-foreground hover:underline"
          >
            View full grant →
          </Link>
          <a
            href={rec.source_url}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 hover:text-foreground hover:underline"
          >
            Official source <ExternalLink className="h-3 w-3" />
          </a>
        </div>
      </CardContent>
    </Card>
  );
}
