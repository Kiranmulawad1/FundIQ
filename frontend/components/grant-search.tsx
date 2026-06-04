"use client";

import { Loader2, Sparkles, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { GrantCard } from "@/components/grant-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import {
  ApiError,
  type GrantSearchResponse,
  type RetrievalMode,
  searchGrants,
} from "@/lib/api";
import { useIsMac } from "@/lib/use-platform";
import { useRecentSearches } from "@/lib/use-recent-searches";

const MODE_OPTIONS: Array<{
  value: RetrievalMode;
  label: string;
  hint: string;
}> = [
  { value: "dense", label: "Dense", hint: "pgvector cosine, fastest" },
  { value: "hybrid", label: "Hybrid", hint: "dense + sparse + RRF" },
  { value: "hybrid_rerank", label: "Hybrid + Rerank", hint: "BGE cross-encoder" },
];

const EXAMPLES = [
  "Stipendium für akademische Ausgründung",
  "EU funding for breakthrough deep tech",
  "Pre-seed grant Baden-Württemberg",
  "ZIM Mittelstand Innovation",
];

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
  if (target.isContentEditable) return true;
  return false;
}

export function GrantSearch() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<RetrievalMode>("hybrid_rerank");
  const [useHyde, setUseHyde] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<GrantSearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inflight = useRef<AbortController | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const recents = useRecentSearches();
  const isMac = useIsMac();

  const runSearch = useCallback(
    async (q: string, opts?: { mode?: RetrievalMode; useHyde?: boolean }) => {
      const trimmed = q.trim();
      if (!trimmed) return;
      const effMode = opts?.mode ?? mode;
      const effHyde = opts?.useHyde ?? useHyde;
      inflight.current?.abort();
      const controller = new AbortController();
      inflight.current = controller;
      setLoading(true);
      setError(null);
      try {
        const res = await searchGrants(
          { query: trimmed, mode: effMode, use_hyde: effHyde, limit: 10 },
          { signal: controller.signal },
        );
        if (controller.signal.aborted) return;
        setResult(res);
        recents.add({ query: trimmed, mode: effMode, useHyde: effHyde });
      } catch (err) {
        if (controller.signal.aborted) return;
        if (err instanceof ApiError) {
          setError(`Backend ${err.status}: ${err.message}`);
        } else if (err instanceof Error) {
          setError(err.message);
        } else {
          setError("Unknown error");
        }
        setResult(null);
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    },
    [mode, useHyde, recents],
  );

  // Global keyboard shortcuts:
  //   `/`        focus input (skip if already typing somewhere)
  //   Cmd/Ctrl+K focus + select existing text
  //   Esc        if input has value → clear; if empty → blur
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const input = inputRef.current;
      const isOurInput = document.activeElement === input;

      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        input?.focus();
        input?.select();
        return;
      }
      if (e.key === "/" && !isTypingTarget(e.target)) {
        e.preventDefault();
        input?.focus();
        return;
      }
      if (e.key === "Escape" && isOurInput) {
        if (query.length > 0) {
          e.preventDefault();
          setQuery("");
        } else {
          input?.blur();
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [query]);

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    void runSearch(query);
  };

  const showRecents = recents.items.length > 0;
  const chips = showRecents
    ? recents.items.slice(0, 5)
    : EXAMPLES.map((q) => ({
        query: q,
        mode,
        useHyde: false,
        ts: 0,
      }));

  return (
    <div className="space-y-6">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Input
              ref={inputRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Describe what you're looking for — DE or EN, vague is fine"
              className="h-11 pr-16 text-base"
              autoFocus
              aria-label="Search query"
            />
            <KbdHint isMac={isMac} hasValue={query.length > 0} />
          </div>
          <Button
            type="submit"
            disabled={loading || query.trim().length === 0}
            className="h-11 px-6"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Search"}
          </Button>
        </div>

        <div className="flex flex-wrap items-end gap-4">
          <div className="space-y-1.5">
            <Label htmlFor="mode" className="text-xs">
              Retrieval mode
            </Label>
            <Select
              value={mode}
              onValueChange={(v) => setMode((v ?? "hybrid_rerank") as RetrievalMode)}
            >
              <SelectTrigger id="mode" className="w-56">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {MODE_OPTIONS.map((m) => (
                  <SelectItem key={m.value} value={m.value}>
                    <div className="flex flex-col">
                      <span>{m.label}</span>
                      <span className="text-xs text-muted-foreground">
                        {m.hint}
                      </span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center gap-2 pb-2">
            <Switch
              id="hyde"
              checked={useHyde}
              onCheckedChange={(v) => setUseHyde(v === true)}
            />
            <Label htmlFor="hyde" className="cursor-pointer">
              <span className="inline-flex items-center gap-1">
                <Sparkles className="h-3.5 w-3.5" />
                HyDE rewriting
              </span>
            </Label>
          </div>
        </div>

        <ChipRow
          chips={chips}
          showRecents={showRecents}
          onPick={(c) => {
            setQuery(c.query);
            setMode(c.mode);
            setUseHyde(c.useHyde);
            void runSearch(c.query, { mode: c.mode, useHyde: c.useHyde });
          }}
          onClear={recents.clear}
        />
      </form>

      {error && (
        <Card className="border-destructive/30 bg-destructive/5">
          <CardContent className="py-4 text-sm text-destructive">
            {error}
          </CardContent>
        </Card>
      )}

      {loading && !result && <ResultsSkeleton />}

      {result && !loading && (
        <Results
          result={result}
          modeLabel={
            MODE_OPTIONS.find((m) => m.value === result.mode)?.label ??
            result.mode
          }
        />
      )}

      {!result && !loading && !error && (
        <EmptyState onPick={(q) => { setQuery(q); void runSearch(q); }} />
      )}
    </div>
  );
}

function KbdHint({
  isMac,
  hasValue,
}: {
  isMac: boolean | null;
  hasValue: boolean;
}) {
  // Hide once they're typing — the hint is noise at that point.
  if (hasValue) return null;
  // Show a placeholder slot until we know the platform so layout doesn't jump.
  return (
    <span
      aria-hidden
      className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 select-none"
    >
      <kbd className="rounded border border-border bg-muted/60 px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground">
        {isMac === null ? "   " : isMac ? "⌘K" : "Ctrl K"}
      </kbd>
    </span>
  );
}

function ChipRow({
  chips,
  showRecents,
  onPick,
  onClear,
}: {
  chips: Array<{ query: string; mode: RetrievalMode; useHyde: boolean; ts: number }>;
  showRecents: boolean;
  onPick: (c: { query: string; mode: RetrievalMode; useHyde: boolean }) => void;
  onClear: () => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-[11px] uppercase tracking-wide text-muted-foreground">
        {showRecents ? "Recent" : "Try"}
      </span>
      {chips.map((c) => (
        <button
          key={`${c.query}-${c.ts}`}
          type="button"
          onClick={() => onPick(c)}
          className="group inline-flex items-center gap-1 rounded-full border border-border bg-secondary/60 px-3 py-1 text-xs text-muted-foreground hover:bg-secondary"
          title={
            showRecents ? `${c.mode}${c.useHyde ? " + HyDE" : ""}` : undefined
          }
        >
          <span className="max-w-[18rem] truncate">{c.query}</span>
          {showRecents && c.useHyde && (
            <Sparkles className="h-3 w-3 text-violet-500/80" />
          )}
        </button>
      ))}
      {showRecents && (
        <button
          type="button"
          onClick={onClear}
          className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
        >
          <X className="h-3 w-3" /> Clear
        </button>
      )}
    </div>
  );
}

function ResultsSkeleton() {
  return (
    <div className="space-y-3">
      {Array.from({ length: 3 }).map((_, i) => (
        <Card key={i}>
          <CardContent className="space-y-3 py-4">
            <Skeleton className="h-4 w-2/3" />
            <Skeleton className="h-3 w-full" />
            <Skeleton className="h-3 w-5/6" />
            <Skeleton className="h-3 w-3/4" />
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

function Results({
  result,
  modeLabel,
}: {
  result: GrantSearchResponse;
  modeLabel: string;
}) {
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
        <Badge variant="outline">{modeLabel}</Badge>
        {result.used_hyde && (
          <Badge variant="outline" className="border-violet-500/40 text-violet-700 dark:text-violet-300">
            <Sparkles className="mr-1 h-3 w-3" />
            HyDE
          </Badge>
        )}
        {result.cache_hit && (
          <Badge
            variant="outline"
            className="border-emerald-500/40 text-emerald-700 dark:text-emerald-300"
          >
            cache hit
          </Badge>
        )}
        <span>{result.hits.length} hits</span>
        <span>·</span>
        <span>{result.elapsed_ms} ms</span>
        <span>·</span>
        <span>
          dense {result.dense_count} / sparse {result.sparse_count} / RRF{" "}
          {result.rrf_input_count}
          {result.rerank_input_count > 0 && (
            <> / rerank {result.rerank_input_count}</>
          )}
        </span>
      </div>

      {result.hypotheticals && result.hypotheticals.length > 0 && (
        <Card className="border-violet-500/20 bg-violet-500/5">
          <CardContent className="space-y-2 py-3">
            <div className="text-xs font-medium text-violet-700 dark:text-violet-300">
              HyDE hypotheticals fed to the dense retriever
            </div>
            <ol className="ml-4 list-decimal space-y-1 text-sm text-muted-foreground">
              {result.hypotheticals.map((h, i) => (
                <li key={i}>{h}</li>
              ))}
            </ol>
          </CardContent>
        </Card>
      )}

      {result.hits.length === 0 ? (
        <Card>
          <CardContent className="py-8 text-center text-sm text-muted-foreground">
            No grants matched. Try a broader query or toggle HyDE on.
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-3">
          {result.hits.map((hit, i) => (
            <GrantCard key={hit.id} hit={hit} rank={i + 1} />
          ))}
        </div>
      )}
    </div>
  );
}

function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  return (
    <Card className="border-dashed">
      <CardContent className="space-y-3 py-10 text-center text-sm text-muted-foreground">
        <p>
          Try one of the example queries above, or describe what you&apos;re
          looking for in your own words. Press{" "}
          <kbd className="rounded border border-border bg-muted/60 px-1.5 py-0.5 text-[10px] font-medium text-foreground/80">
            /
          </kbd>{" "}
          or{" "}
          <kbd className="rounded border border-border bg-muted/60 px-1.5 py-0.5 text-[10px] font-medium text-foreground/80">
            ⌘K
          </kbd>{" "}
          to focus the search box.
        </p>
        <div className="flex flex-wrap justify-center gap-2">
          {EXAMPLES.slice(0, 3).map((ex) => (
            <button
              key={ex}
              type="button"
              onClick={() => onPick(ex)}
              className="rounded-full bg-secondary px-3 py-1.5 text-xs hover:bg-secondary/80"
            >
              {ex}
            </button>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
