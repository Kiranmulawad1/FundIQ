"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useTransition } from "react";

import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type {
  GrantPortal,
  GrantSortKey,
  GrantStatus,
} from "@/lib/api";

// Sentinel for "no filter" because shadcn Select treats "" as missing value.
const ALL = "__all";

const PORTALS: { value: GrantPortal | typeof ALL; label: string }[] = [
  { value: ALL, label: "All portals" },
  { value: "exist", label: "EXIST" },
  { value: "kfw", label: "KfW" },
  { value: "eic", label: "EIC" },
  { value: "horizon", label: "Horizon Europe" },
  { value: "bayern", label: "Bayern Kapital" },
  { value: "nrw", label: "NRW.BANK / förderdaten" },
  { value: "bw", label: "Baden-Württemberg" },
];

const STATUSES: { value: GrantStatus | typeof ALL; label: string }[] = [
  { value: ALL, label: "Any status" },
  { value: "open", label: "Open" },
  { value: "rolling", label: "Rolling" },
  { value: "upcoming", label: "Upcoming" },
  { value: "closed", label: "Closed" },
];

const COUNTRIES: { value: string; label: string }[] = [
  { value: ALL, label: "Any country" },
  { value: "DE", label: "Germany" },
  { value: "EU", label: "European Union" },
];

const SORTS: { value: GrantSortKey; label: string }[] = [
  { value: "created_at", label: "Newest first" },
  { value: "deadline", label: "Deadline soonest" },
  { value: "funding_max", label: "Funding (largest)" },
];

interface Props {
  portal: string | undefined;
  status: string | undefined;
  country: string | undefined;
  sort: GrantSortKey;
}

export function GrantFilters({ portal, status, country, sort }: Props) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [pending, startTransition] = useTransition();

  const setParam = useCallback(
    (key: string, value: string | null) => {
      const params = new URLSearchParams(searchParams);
      if (value === null || value === ALL) {
        params.delete(key);
      } else {
        params.set(key, value);
      }
      // Whenever a filter changes the result set we reset offset; otherwise
      // page 4 of "all grants" could land on a 1-page filtered result.
      if (key !== "offset" && key !== "sort") {
        params.delete("offset");
      }
      startTransition(() => {
        router.replace(`?${params.toString()}`, { scroll: false });
      });
    },
    [router, searchParams],
  );

  const anyActive =
    portal !== undefined ||
    status !== undefined ||
    country !== undefined ||
    sort !== "created_at";

  return (
    <div className="grid gap-3 sm:grid-cols-4">
      <FilterSelect
        id="portal"
        label="Portal"
        value={portal ?? ALL}
        options={PORTALS}
        onChange={(v) => setParam("portal", v)}
      />
      <FilterSelect
        id="status"
        label="Status"
        value={status ?? ALL}
        options={STATUSES}
        onChange={(v) => setParam("status", v)}
      />
      <FilterSelect
        id="country"
        label="Country"
        value={country ?? ALL}
        options={COUNTRIES}
        onChange={(v) => setParam("country", v)}
      />
      <FilterSelect
        id="sort"
        label="Sort"
        value={sort}
        options={SORTS}
        onChange={(v) => setParam("sort", v === "created_at" ? null : v)}
      />
      {anyActive && (
        <div className="sm:col-span-4">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            disabled={pending}
            onClick={() => {
              startTransition(() => {
                router.replace("?", { scroll: false });
              });
            }}
          >
            Reset filters
          </Button>
        </div>
      )}
    </div>
  );
}

function FilterSelect({
  id,
  label,
  value,
  options,
  onChange,
}: {
  id: string;
  label: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (v: string) => void;
}) {
  return (
    <div className="space-y-1.5">
      <Label htmlFor={id} className="text-xs">
        {label}
      </Label>
      <Select value={value} onValueChange={(v) => onChange(v ?? ALL)}>
        <SelectTrigger id={id}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((o) => (
            <SelectItem key={o.value} value={o.value}>
              {o.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
