"use client";

import { Sparkles, Trash2, Wand2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import {
  AgentStageProgress,
  type StageProgress,
} from "@/components/agent-stage-progress";
import { AgentTraceStrip } from "@/components/agent-trace-strip";
import { CriticFindingsCard } from "@/components/critic-findings-card";
import { RecommendationCard } from "@/components/recommendation-card";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import {
  type AgentConversationEntry,
  type AgentStage,
  ApiError,
  deleteAgentSession,
  extractPartialSummary,
  getAgentSession,
  streamRecommendGrants,
} from "@/lib/api";
import { useAgentSession } from "@/lib/use-agent-session";
import { useStartupProfile } from "@/lib/use-startup-profile";

const INITIAL_PROGRESS: Record<AgentStage, StageProgress> = {
  planner: { status: "idle" },
  retriever: { status: "idle" },
  scorer: { status: "idle" },
  writer: { status: "idle" },
  critic: { status: "idle" },
};

const EXAMPLES = [
  "I'm a researcher at TU Munich building a deep tech robotics startup, pre-revenue, looking for ~150k EUR to bridge to seed.",
  "Wir gründen ein Healthtech-Startup in Berlin und brauchen Eigenkapital oder Wandeldarlehen in der Frühphase (~500k EUR).",
  "What EU-level grants are open for early-stage climate tech startups outside Germany?",
];

export default function RecommendPage() {
  const { sessionId, setSessionId, clear, hydrated } = useAgentSession();
  const { profile: startupProfile } = useStartupProfile();

  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState<AgentConversationEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<Record<AgentStage, StageProgress>>(INITIAL_PROGRESS);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [partialSummary, setPartialSummary] = useState("");
  const inflight = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const elapsedTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const writerBuffer = useRef<string>("");

  // Rehydrate session history once we know the sessionId from localStorage.
  useEffect(() => {
    if (!hydrated || !sessionId) return;
    let cancelled = false;
    void getAgentSession(sessionId)
      .then((s) => {
        if (cancelled) return;
        setHistory(s.history);
      })
      .catch(() => {
        // Stale/unknown session id — treat as empty. We don't surface
        // the error: it's an artefact of clearing the DB or migrating
        // schemas, not a user-actionable failure.
      });
    return () => {
      cancelled = true;
    };
  }, [hydrated, sessionId]);

  const runRecommend = useCallback(
    async (q: string) => {
      const trimmed = q.trim();
      if (trimmed.length < 3) return;
      inflight.current?.abort();
      const controller = new AbortController();
      inflight.current = controller;
      setLoading(true);
      setError(null);
      setProgress({ ...INITIAL_PROGRESS });
      setElapsedMs(0);
      setPartialSummary("");
      writerBuffer.current = "";

      // Tick the elapsed counter so the loader feels alive between
      // stage events. Stopped in `finally`.
      const startedAt = performance.now();
      elapsedTimer.current = setInterval(() => {
        setElapsedMs(Math.round(performance.now() - startedAt));
      }, 250);

      try {
        await streamRecommendGrants(trimmed, {
          sessionId,
          startupProfile,
          signal: controller.signal,
          onStage: (e) => {
            // On a Writer-retry-start, wipe the partial summary buffer so
            // the user sees the second attempt typing fresh instead of
            // appended onto the first attempt's text.
            if (e.stage === "writer" && e.status === "start" && e.retry) {
              writerBuffer.current = "";
              setPartialSummary("");
            }
            setProgress((prev) => ({
              ...prev,
              [e.stage]: {
                status: e.status === "done" ? "done" : "running",
                elapsed_ms:
                  e.status === "done"
                    ? e[`${e.stage}_ms` as keyof typeof e] as number | undefined ?? e.elapsed_ms
                    : prev[e.stage].elapsed_ms,
              },
            }));
          },
          onWriterDelta: (chunk) => {
            writerBuffer.current += chunk;
            const partial = extractPartialSummary(writerBuffer.current);
            // Only update React state when the visible string actually
            // changed — saves a render per chunk during the early
            // pre-`"summary"` prefix.
            if (partial) setPartialSummary(partial);
          },
          onDone: (res) => {
            if (res.session_id !== sessionId) {
              setSessionId(res.session_id);
            }
            const entry: AgentConversationEntry = {
              ts: new Date().toISOString(),
              query: trimmed,
              summary: res.summary,
              recommendations: res.recommendations,
              questions_for_user: res.questions_for_user,
              trace: res.trace,
            };
            setHistory((prev) => [...prev, entry]);
            setQuery("");
            setTimeout(() => {
              bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
            }, 50);
          },
          onError: (msg) => setError(msg),
        });
      } catch (err) {
        if (controller.signal.aborted) return;
        if (err instanceof ApiError) {
          setError(`Backend ${err.status}: ${err.message.slice(0, 200)}`);
        } else if (err instanceof Error) {
          setError(err.message);
        } else {
          setError("Unknown error");
        }
      } finally {
        if (elapsedTimer.current) {
          clearInterval(elapsedTimer.current);
          elapsedTimer.current = null;
        }
        if (!controller.signal.aborted) setLoading(false);
      }
    },
    [sessionId, setSessionId, startupProfile],
  );

  const onSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    void runRecommend(query);
  };

  const onKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      void runRecommend(query);
    }
  };

  const onClearChat = async () => {
    if (!confirm("Clear this chat history? This can't be undone.")) return;
    if (sessionId) {
      try {
        await deleteAgentSession(sessionId);
      } catch {
        // Soft-delete is best-effort; we still wipe local state.
      }
    }
    clear();
    setHistory([]);
    setError(null);
  };

  return (
    <main className="mx-auto w-full max-w-3xl space-y-6 px-4 py-8 sm:py-12">
      <header className="space-y-2">
        <div className="inline-flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground">
          <Wand2 className="h-3.5 w-3.5" />
          Multi-agent recommendation
        </div>
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">
              Get grant recommendations
            </h1>
            <p className="max-w-2xl text-sm text-muted-foreground">
              Describe your startup. A Planner agent extracts structured facts,
              the hybrid retriever pulls candidates, and a Writer agent ranks
              them with grounded rationale and caveats. Chats are saved per
              browser.
            </p>
          </div>
          <div className="flex items-center gap-2">
            {startupProfile && (
              <a
                href="/profile"
                className="inline-flex items-center gap-1 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-[11px] font-medium text-emerald-700 hover:bg-emerald-500/20 dark:text-emerald-300"
                title="A saved startup profile is biasing the Planner. Click to edit."
              >
                Profile active
              </a>
            )}
            {history.length > 0 && (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => void onClearChat()}
                className="gap-1.5"
              >
                <Trash2 className="h-3.5 w-3.5" />
                Clear chat
              </Button>
            )}
          </div>
        </div>
      </header>

      <form onSubmit={onSubmit} className="space-y-3">
        <Textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={onKey}
          placeholder="e.g. I'm a researcher at TU Munich building a deep tech robotics startup, pre-revenue, looking for ~150k EUR…"
          rows={4}
          className="min-h-[120px] resize-y text-base"
          aria-label="Your question"
          autoFocus
        />
        <div className="flex flex-wrap items-center gap-2">
          <Button
            type="submit"
            disabled={loading || query.trim().length < 3}
            className="gap-1.5"
          >
            <Sparkles className="h-4 w-4" />
            {history.length > 0 ? "Ask again" : "Recommend grants"}
          </Button>
          <span className="text-xs text-muted-foreground">
            <kbd className="rounded border border-border bg-muted/60 px-1.5 py-0.5 text-[10px] font-medium text-foreground/80">
              ⌘↩
            </kbd>{" "}
            to submit
          </span>
          {history.length === 0 && (
            <div className="ml-auto flex flex-wrap items-center gap-2">
              <span className="text-[11px] uppercase tracking-wide text-muted-foreground">
                Try
              </span>
              {EXAMPLES.map((ex, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => {
                    setQuery(ex);
                    void runRecommend(ex);
                  }}
                  className="rounded-full border border-border bg-secondary/60 px-3 py-1 text-xs text-muted-foreground hover:bg-secondary"
                  title={ex}
                >
                  Example {i + 1}
                </button>
              ))}
            </div>
          )}
        </div>
      </form>

      {error && (
        <Card className="border-destructive/30 bg-destructive/5">
          <CardContent className="py-4 text-sm text-destructive">
            {error}
          </CardContent>
        </Card>
      )}

      {history.length > 0 && (
        <ol className="space-y-8">
          {history.map((entry, i) => (
            <li key={`${entry.ts}-${i}`} className="space-y-3">
              <UserQueryBubble query={entry.query} ts={entry.ts} />
              <AgentTurn entry={entry} />
            </li>
          ))}
        </ol>
      )}

      {loading && (
        <div className="space-y-3">
          <AgentStageProgress progress={progress} totalElapsedMs={elapsedMs} />
          {partialSummary && (
            <Card className="border-foreground/15 bg-foreground/[0.02]">
              <CardContent className="space-y-2 py-4">
                <div className="text-[11px] uppercase tracking-wide text-muted-foreground">
                  Writer · streaming
                </div>
                <p className="whitespace-pre-wrap text-base leading-relaxed">
                  {partialSummary}
                  <span className="inline-block h-4 w-1 translate-y-0.5 animate-pulse bg-foreground/60 align-middle" />
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      <div ref={bottomRef} aria-hidden />
    </main>
  );
}

function UserQueryBubble({ query, ts }: { query: string; ts: string }) {
  const date = new Date(ts);
  const time = Number.isNaN(date.getTime())
    ? ts
    : date.toLocaleString(undefined, {
        day: "numeric",
        month: "short",
        hour: "2-digit",
        minute: "2-digit",
      });
  return (
    <div className="flex flex-col items-end">
      <div className="rounded-2xl rounded-tr-sm bg-primary/10 px-4 py-2 text-sm text-foreground">
        {query}
      </div>
      <div className="mt-1 pr-2 text-[10px] text-muted-foreground">{time}</div>
    </div>
  );
}

function AgentTurn({ entry }: { entry: AgentConversationEntry }) {
  return (
    <div className="space-y-3">
      <Card className="border-foreground/15 bg-foreground/[0.02]">
        <CardContent className="space-y-3 py-4">
          <div className="text-[11px] uppercase tracking-wide text-muted-foreground">
            Summary
          </div>
          <p className="text-base leading-relaxed">{entry.summary}</p>
        </CardContent>
      </Card>

      {entry.questions_for_user.length > 0 && (
        <Card className="border-violet-500/25 bg-violet-500/5">
          <CardContent className="space-y-2 py-4">
            <div className="text-[11px] uppercase tracking-wide text-violet-700 dark:text-violet-300">
              We have some clarifying questions
            </div>
            <ul className="ml-4 list-disc space-y-1 text-sm">
              {entry.questions_for_user.map((q, i) => (
                <li key={i}>{q}</li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {entry.recommendations.length === 0 ? (
        <Card>
          <CardContent className="py-6 text-center text-sm text-muted-foreground">
            No concrete recommendations were returned for this turn.
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-3">
          {entry.recommendations.map((rec, i) => (
            <RecommendationCard
              key={rec.grant_id}
              rec={rec}
              rank={i + 1}
              score={entry.trace.scores.find((s) => s.grant_id === rec.grant_id)}
            />
          ))}
        </div>
      )}

      <CriticFindingsCard
        pass={entry.trace.critic_pass}
        summary={entry.trace.critic_summary}
        findings={entry.trace.critic_findings}
        recommendations={entry.recommendations}
      />

      <AgentTraceStrip trace={entry.trace} />
    </div>
  );
}
