"use client";

import { Save, Trash2, UserCircle2 } from "lucide-react";
import { useEffect, useState } from "react";

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
import { Textarea } from "@/components/ui/textarea";
import {
  PROFILE_SECTORS,
  PROFILE_STAGES,
  type StartupProfile,
  useStartupProfile,
} from "@/lib/use-startup-profile";

const NONE = "__none";

const COUNTRY_OPTIONS = [
  { value: NONE, label: "Not specified" },
  { value: "DE", label: "Germany" },
  { value: "EU", label: "European Union (any member)" },
];

const FEDERAL_STATES = [
  "Baden-Württemberg",
  "Bayern",
  "Berlin",
  "Brandenburg",
  "Bremen",
  "Hamburg",
  "Hessen",
  "Mecklenburg-Vorpommern",
  "Niedersachsen",
  "Nordrhein-Westfalen",
  "Rheinland-Pfalz",
  "Saarland",
  "Sachsen",
  "Sachsen-Anhalt",
  "Schleswig-Holstein",
  "Thüringen",
];

export default function ProfilePage() {
  const { profile, hydrated, save, clear } = useStartupProfile();
  const [form, setForm] = useState<StartupProfile>({});
  const [savedFlash, setSavedFlash] = useState(false);

  // Rehydrate the form once the hook is ready. Keeps SSR/CSR stable.
  useEffect(() => {
    if (hydrated) setForm(profile ?? {});
  }, [hydrated, profile]);

  const onSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    save(form);
    setSavedFlash(true);
    setTimeout(() => setSavedFlash(false), 1500);
  };

  const onClear = () => {
    if (!confirm("Clear your saved startup profile?")) return;
    clear();
    setForm({});
  };

  const setField = <K extends keyof StartupProfile>(
    key: K,
    value: StartupProfile[K] | undefined,
  ) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  return (
    <main className="mx-auto w-full max-w-3xl space-y-6 px-4 py-8 sm:py-12">
      <header className="space-y-2">
        <div className="inline-flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground">
          <UserCircle2 className="h-3.5 w-3.5" />
          Startup profile
        </div>
        <h1 className="text-2xl font-semibold tracking-tight">
          Tell us about your startup
        </h1>
        <p className="max-w-2xl text-sm text-muted-foreground">
          Everything you save here gets sent with each agent recommendation
          so the Planner doesn&apos;t need to re-extract facts you&apos;ve
          already shared. Stored in this browser only — no account, no sync.
        </p>
      </header>

      <form onSubmit={onSubmit} className="space-y-5">
        <Card>
          <CardContent className="grid gap-4 py-5 sm:grid-cols-2">
            <Field label="Startup name" htmlFor="name">
              <Input
                id="name"
                value={form.name ?? ""}
                onChange={(e) => setField("name", e.target.value || undefined)}
                placeholder="Acme Robotics"
              />
            </Field>

            <Field label="Sector" htmlFor="sector">
              <Select
                value={form.sector ?? NONE}
                onValueChange={(v) =>
                  setField("sector", v && v !== NONE ? v : undefined)
                }
              >
                <SelectTrigger id="sector">
                  <SelectValue placeholder="Not specified" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={NONE}>Not specified</SelectItem>
                  {PROFILE_SECTORS.map((s) => (
                    <SelectItem key={s} value={s}>
                      {s}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>

            <Field label="Stage" htmlFor="stage">
              <Select
                value={form.stage ?? NONE}
                onValueChange={(v) =>
                  setField("stage", v && v !== NONE ? v : undefined)
                }
              >
                <SelectTrigger id="stage">
                  <SelectValue placeholder="Not specified" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={NONE}>Not specified</SelectItem>
                  {PROFILE_STAGES.map((s) => (
                    <SelectItem key={s.value} value={s.value}>
                      {s.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>

            <Field label="Country" htmlFor="country">
              <Select
                value={form.country ?? NONE}
                onValueChange={(v) =>
                  setField("country", v && v !== NONE ? v : undefined)
                }
              >
                <SelectTrigger id="country">
                  <SelectValue placeholder="Not specified" />
                </SelectTrigger>
                <SelectContent>
                  {COUNTRY_OPTIONS.map((c) => (
                    <SelectItem key={c.value} value={c.value}>
                      {c.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>

            <Field label="Federal state (DE)" htmlFor="federal_state">
              <Select
                value={form.federal_state ?? NONE}
                onValueChange={(v) =>
                  setField("federal_state", v && v !== NONE ? v : undefined)
                }
              >
                <SelectTrigger id="federal_state">
                  <SelectValue placeholder="Not specified" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={NONE}>Not specified</SelectItem>
                  {FEDERAL_STATES.map((s) => (
                    <SelectItem key={s} value={s}>
                      {s}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>

            <Field label="Funding target (EUR)" htmlFor="funding_target_eur">
              <Input
                id="funding_target_eur"
                type="number"
                min={0}
                step={10000}
                value={form.funding_target_eur ?? ""}
                onChange={(e) => {
                  const n = e.target.value === "" ? undefined : Number(e.target.value);
                  setField(
                    "funding_target_eur",
                    Number.isFinite(n) ? (n as number) : undefined,
                  );
                }}
                placeholder="150000"
              />
            </Field>

            <div className="sm:col-span-2">
              <Field label="Description" htmlFor="description">
                <Textarea
                  id="description"
                  rows={3}
                  value={form.description ?? ""}
                  onChange={(e) =>
                    setField("description", e.target.value || undefined)
                  }
                  placeholder="Team, traction, IP, what makes this startup specific…"
                />
              </Field>
            </div>
          </CardContent>
        </Card>

        <div className="flex flex-wrap items-center gap-3">
          <Button type="submit" className="gap-1.5">
            <Save className="h-4 w-4" />
            Save profile
          </Button>
          {profile && (
            <Button
              type="button"
              variant="ghost"
              onClick={onClear}
              className="gap-1.5"
            >
              <Trash2 className="h-3.5 w-3.5" />
              Clear
            </Button>
          )}
          {savedFlash && (
            <span className="text-xs text-emerald-700 dark:text-emerald-300">
              Saved — the next recommendation will use this.
            </span>
          )}
        </div>
      </form>
    </main>
  );
}

function Field({
  label,
  htmlFor,
  children,
}: {
  label: string;
  htmlFor: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <Label htmlFor={htmlFor} className="text-xs">
        {label}
      </Label>
      {children}
    </div>
  );
}
