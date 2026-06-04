import { AlertTriangle, CheckCircle2, ChevronDown } from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import type {
  AgentGrantRecommendation,
  CriticFinding,
  CriticFindingType,
  CriticSeverity,
} from "@/lib/api";
import { cn } from "@/lib/utils";

const TYPE_LABEL: Record<CriticFindingType, string> = {
  citation_faithfulness: "Citation faithfulness",
  fit_alignment: "Fit / score mismatch",
  caveat_omission: "Caveat omitted",
  language_mismatch: "Language mismatch",
  profile_misuse: "Profile misuse",
  other: "Other",
};

const SEVERITY_STYLE: Record<CriticSeverity, string> = {
  high: "border-rose-500/40 bg-rose-500/10 text-rose-700 dark:text-rose-300",
  medium: "border-amber-500/40 bg-amber-500/10 text-amber-800 dark:text-amber-200",
  low: "border-zinc-500/40 bg-zinc-500/10 text-zinc-700 dark:text-zinc-300",
};

interface Props {
  pass: boolean;
  summary: string;
  findings: CriticFinding[];
  /** Used to attribute findings back to a recommendation by title. */
  recommendations: AgentGrantRecommendation[];
}

export function CriticFindingsCard({
  pass,
  summary,
  findings,
  recommendations,
}: Props) {
  if (pass && findings.length === 0) {
    return (
      <Card className="border-emerald-500/20 bg-emerald-500/5">
        <CardContent className="flex items-center gap-2 py-3 text-sm text-emerald-800 dark:text-emerald-200">
          <CheckCircle2 className="h-4 w-4" />
          <span className="font-medium">Quality review passed</span>
          {summary && (
            <span className="text-emerald-700/80 dark:text-emerald-300/80">
              · {summary}
            </span>
          )}
        </CardContent>
      </Card>
    );
  }

  if (findings.length === 0) {
    return null;
  }

  const titleFor = (id: string | null): string | null => {
    if (!id) return null;
    return recommendations.find((r) => r.grant_id === id)?.grant_title ?? null;
  };

  return (
    <details className="rounded-xl border border-amber-500/25 bg-amber-500/[0.04]">
      <summary className="flex cursor-pointer select-none items-center gap-2 px-4 py-3 text-sm">
        <AlertTriangle className="h-4 w-4 text-amber-700 dark:text-amber-300" />
        <span className="font-medium text-foreground">
          Quality review: {findings.length} finding{findings.length === 1 ? "" : "s"}
        </span>
        {summary && (
          <span className="text-muted-foreground">· {summary}</span>
        )}
        <ChevronDown className="ml-auto h-4 w-4 text-muted-foreground transition-transform group-open:rotate-180" />
      </summary>
      <ul className="space-y-2 border-t border-amber-500/15 px-4 py-3">
        {findings.map((f, i) => (
          <li
            key={i}
            className={cn(
              "rounded-lg border px-3 py-2 text-sm",
              SEVERITY_STYLE[f.severity],
            )}
          >
            <div className="mb-1 flex flex-wrap items-center gap-2 text-xs">
              <span className="rounded-full bg-background/40 px-2 py-0.5 font-medium uppercase tracking-wide">
                {f.severity}
              </span>
              <span className="font-medium text-foreground/85">
                {TYPE_LABEL[f.type] ?? f.type}
              </span>
              {titleFor(f.grant_id) && (
                <span className="text-muted-foreground">
                  · {titleFor(f.grant_id)}
                </span>
              )}
            </div>
            <p className="text-foreground/90">{f.message}</p>
          </li>
        ))}
      </ul>
    </details>
  );
}
