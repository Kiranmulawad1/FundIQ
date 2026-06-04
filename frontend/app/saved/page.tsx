"use client";

import { ExternalLink, Printer, Trash2 } from "lucide-react";
import Link from "next/link";

import { SaveButton } from "@/components/save-button";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { useSavedGrants } from "@/lib/use-saved-grants";

export default function SavedPage() {
  const { items, clear } = useSavedGrants();
  const sorted = [...items].sort((a, b) => b.savedAt - a.savedAt);

  const printedOn = new Date().toLocaleDateString("en-GB", {
    day: "numeric",
    month: "long",
    year: "numeric",
  });

  return (
    <main className="mx-auto w-full max-w-3xl space-y-6 px-4 py-8 sm:py-12 print:py-0">
      {/* Print-only cover header — replaces the on-screen header in PDFs. */}
      <div className="hidden print:block">
        <h1 className="text-xl font-semibold">FundIQ — saved grants shortlist</h1>
        <p className="text-xs text-muted-foreground">
          {sorted.length} grant{sorted.length === 1 ? "" : "s"} · printed {printedOn}
        </p>
        <hr className="my-3 border-border" />
      </div>

      <header className="flex flex-wrap items-baseline justify-between gap-3 print:hidden">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">
            Saved grants
          </h1>
          <p className="text-sm text-muted-foreground">
            Your shortlist. Stored in this browser only — no account, no sync.
          </p>
        </div>
        {sorted.length > 0 && (
          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => window.print()}
              className="gap-1.5"
              aria-label="Print or save as PDF"
            >
              <Printer className="h-3.5 w-3.5" />
              Print / PDF
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => {
                if (confirm("Clear all saved grants?")) clear();
              }}
              className="gap-1.5"
            >
              <Trash2 className="h-3.5 w-3.5" />
              Clear all
            </Button>
          </div>
        )}
      </header>

      {sorted.length === 0 ? (
        <Card className="border-dashed">
          <CardContent className="space-y-3 py-12 text-center text-sm text-muted-foreground">
            <p>You haven&apos;t saved any grants yet.</p>
            <p>
              From{" "}
              <Link href="/" className="underline hover:text-foreground">
                search
              </Link>{" "}
              or{" "}
              <Link href="/grants" className="underline hover:text-foreground">
                browse
              </Link>
              , click the bookmark icon on any grant to add it here.
            </p>
          </CardContent>
        </Card>
      ) : (
        <ul className="space-y-3 print:space-y-2">
          {sorted.map((g) => (
            <li key={g.id} data-print-block>
              <Card className="print:rounded-none print:border-none print:shadow-none print:ring-0">
                <CardContent className="space-y-2 py-4 print:py-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant="secondary" className="uppercase">
                      {g.portal}
                    </Badge>
                    <span className="ml-auto text-[11px] text-muted-foreground">
                      Saved {new Date(g.savedAt).toLocaleDateString("en-GB", {
                        day: "numeric",
                        month: "short",
                        year: "numeric",
                      })}
                    </span>
                  </div>
                  <h3 className="text-base font-semibold leading-snug">
                    <Link
                      href={`/grants/${g.id}`}
                      className="hover:underline"
                    >
                      {g.title}
                    </Link>
                  </h3>
                  <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
                    {g.source_doc_id && (
                      <span className="font-mono">{g.source_doc_id}</span>
                    )}
                    <a
                      href={g.source_url}
                      target="_blank"
                      rel="noreferrer"
                      className="ml-auto inline-flex items-center gap-1 hover:text-foreground hover:underline print:ml-0"
                    >
                      <span className="print:hidden">
                        Source <ExternalLink className="inline h-3 w-3" />
                      </span>
                      <span className="hidden print:inline">Source:</span>
                    </a>
                    <div className="print:hidden">
                      <SaveButton
                        grant={{
                          id: g.id,
                          portal: g.portal,
                          title: g.title,
                          source_url: g.source_url,
                          source_doc_id: g.source_doc_id,
                        }}
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}
