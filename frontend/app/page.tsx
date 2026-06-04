import { GrantSearch } from "@/components/grant-search";

export default function HomePage() {
  return (
    <main className="mx-auto flex min-h-[calc(100vh-3.25rem)] w-full max-w-4xl flex-col gap-8 px-4 py-10 sm:py-12">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">
          Search funding programmes
        </h1>
        <p className="max-w-2xl text-sm text-muted-foreground">
          Hybrid semantic retrieval across EXIST, EIC, Horizon Europe, KfW, and
          regional programmes. Toggle the mode to compare dense, hybrid, and
          cross-encoder rerank pipelines side by side.
        </p>
      </header>
      <GrantSearch />
      <footer className="mt-auto pt-12 text-xs text-muted-foreground">
        Phase 5B retrieval stack · multilingual-e5-large · BGE-reranker-v2-m3 ·
        HyDE via Gemini 2.5 Flash · semantic cache
      </footer>
    </main>
  );
}
