import { Badge } from "@/components/ui/badge";
import type { AgentTrace } from "@/lib/api";

export function AgentTraceStrip({ trace }: { trace: AgentTrace }) {
  const facts = trace.extracted_facts;
  const factChips: { label: string; value: string }[] = [];
  if (facts.sector) factChips.push({ label: "sector", value: facts.sector });
  if (facts.stage) factChips.push({ label: "stage", value: facts.stage });
  if (facts.country) factChips.push({ label: "country", value: facts.country });
  if (facts.federal_state)
    factChips.push({ label: "state", value: facts.federal_state });
  if (facts.funding_target_eur != null)
    factChips.push({
      label: "target",
      value: `€${facts.funding_target_eur.toLocaleString()}`,
    });

  return (
    <details className="rounded-lg border border-border/60 bg-muted/30 text-xs">
      <summary className="cursor-pointer select-none px-3 py-2 text-muted-foreground hover:text-foreground">
        Agent trace · {trace.total_ms} ms · {trace.candidate_count} candidates considered
      </summary>
      <div className="space-y-3 px-3 pb-3 pt-1 text-muted-foreground">
        <div>
          <span className="font-medium text-foreground/80">Rewritten query:</span>{" "}
          <span className="font-mono">{trace.rewritten_query}</span>
        </div>
        {factChips.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-foreground/80">Extracted facts:</span>
            {factChips.map((c) => (
              <Badge key={c.label} variant="outline" className="font-normal">
                {c.label}={c.value}
              </Badge>
            ))}
          </div>
        )}
        {trace.planner_rationale && (
          <div>
            <span className="font-medium text-foreground/80">Planner rationale:</span>{" "}
            {trace.planner_rationale}
          </div>
        )}
        <div className="flex flex-wrap gap-x-4 gap-y-1 tabular-nums">
          <span>Planner: {trace.planner_ms} ms</span>
          <span>Retrieval: {trace.retrieval_ms} ms</span>
          <span>Scorer: {trace.scorer_ms} ms</span>
          <span>
            Writer: {trace.writer_ms} ms
            {trace.writer_attempts > 1 && (
              <span
                className="ml-1 rounded-full bg-amber-500/15 px-1.5 py-0.5 text-[10px] font-medium text-amber-800 dark:text-amber-200"
                title="The Critic rejected the first attempt; the Writer was re-run with the findings as feedback."
              >
                {trace.writer_attempts}× attempts
              </span>
            )}
          </span>
          <span>Critic: {trace.critic_ms} ms</span>
          <span>Total: {trace.total_ms} ms</span>
        </div>
        {trace.scores.length > 0 && (
          <div className="text-foreground/80">
            Scorer judgement on {trace.scores.length} candidate{trace.scores.length === 1 ? "" : "s"}
            {" · "}
            {trace.scores.filter((s) => s.fit_label === "high").length} high
            {" / "}
            {trace.scores.filter((s) => s.fit_label === "medium").length} medium
            {" / "}
            {trace.scores.filter((s) => s.fit_label === "low").length} low
          </div>
        )}
      </div>
    </details>
  );
}
