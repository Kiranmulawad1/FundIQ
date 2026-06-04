"use client";

import { Loader2 } from "lucide-react";
import { useEffect, useState } from "react";

const STAGES = [
  "Understanding your question…",
  "Searching the grants corpus…",
  "Reading the top matches…",
  "Drafting recommendations…",
];

/**
 * Visual-only stage loader. The /agents/recommend endpoint doesn't stream
 * stage events yet (it returns the full payload at the end), so we cycle
 * through plausible status messages every 4s to make the 5-30s wait feel
 * less like a hung browser. Becomes obsolete once the endpoint streams.
 */
export function AgentLoader() {
  const [stage, setStage] = useState(0);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const stageTimer = setInterval(() => {
      setStage((s) => (s + 1) % STAGES.length);
    }, 4000);
    const tickTimer = setInterval(() => setTick((t) => t + 1), 1000);
    return () => {
      clearInterval(stageTimer);
      clearInterval(tickTimer);
    };
  }, []);

  return (
    <div className="flex items-center justify-center gap-3 rounded-xl border border-dashed border-border bg-muted/30 py-12 text-sm text-muted-foreground">
      <Loader2 className="h-4 w-4 animate-spin" />
      <div className="space-y-0.5">
        <div>{STAGES[stage]}</div>
        <div className="text-[11px] tabular-nums">{tick}s elapsed</div>
      </div>
    </div>
  );
}
