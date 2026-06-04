"use client";

import { Check, Loader2 } from "lucide-react";

import type { AgentStage } from "@/lib/api";
import { cn } from "@/lib/utils";

const STAGES: { id: AgentStage; label: string; hint: string }[] = [
  { id: "planner", label: "Planning", hint: "extracting structured facts" },
  { id: "retriever", label: "Searching corpus", hint: "dense + sparse + RRF + rerank" },
  { id: "scorer", label: "Scoring eligibility", hint: "per-candidate fit judgement" },
  { id: "writer", label: "Drafting answer", hint: "grounded summary + caveats" },
  { id: "critic", label: "Reviewing quality", hint: "groundedness + caveat audit" },
];

export type StageStatus = "idle" | "running" | "done";

export interface StageProgress {
  status: StageStatus;
  /** Per-stage wall-clock ms once `done`. */
  elapsed_ms?: number;
}

interface Props {
  progress: Record<AgentStage, StageProgress>;
  /** Total elapsed across all stages so far (driven by the page's clock). */
  totalElapsedMs: number;
}

/**
 * Real stage progress driven by SSE events. Replaces the fake cycling
 * loader for the streaming code path. Each row goes:
 *   ◯ idle (gray)
 *   ◐ running (spinner)
 *   ✓ done (checkmark + elapsed time)
 */
export function AgentStageProgress({ progress, totalElapsedMs }: Props) {
  return (
    <div className="space-y-2 rounded-xl border border-dashed border-border bg-muted/30 p-4 text-sm">
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span className="uppercase tracking-wide">Agent progress</span>
        <span className="tabular-nums">{Math.round(totalElapsedMs / 1000)}s elapsed</span>
      </div>
      <ul className="space-y-1.5">
        {STAGES.map((s) => {
          const p = progress[s.id];
          return (
            <li key={s.id} className="flex items-center gap-3">
              <StageIcon status={p.status} />
              <div className="flex-1">
                <div
                  className={cn(
                    "font-medium",
                    p.status === "done" && "text-foreground",
                    p.status === "idle" && "text-muted-foreground",
                  )}
                >
                  {s.label}
                </div>
                <div className="text-xs text-muted-foreground">{s.hint}</div>
              </div>
              {p.status === "done" && p.elapsed_ms != null && (
                <span className="text-xs tabular-nums text-muted-foreground">
                  {formatMs(p.elapsed_ms)}
                </span>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function StageIcon({ status }: { status: StageStatus }) {
  if (status === "running") {
    return <Loader2 className="h-4 w-4 animate-spin text-foreground" />;
  }
  if (status === "done") {
    return (
      <span className="flex h-4 w-4 items-center justify-center rounded-full bg-emerald-500/20 text-emerald-700 dark:text-emerald-300">
        <Check className="h-3 w-3" />
      </span>
    );
  }
  return <span className="h-2.5 w-2.5 rounded-full bg-muted-foreground/40" />;
}

function formatMs(ms: number): string {
  if (ms < 1000) return `${ms} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}
