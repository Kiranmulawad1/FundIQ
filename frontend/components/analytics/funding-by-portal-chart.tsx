"use client";

import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts";

import {
  type ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import type { PortalCount } from "@/lib/api";
import { formatEur } from "@/lib/format";

const chartConfig = {
  funding_avg: {
    label: "Avg funding ceiling",
    color: "var(--color-primary)",
  },
} satisfies ChartConfig;

interface Props {
  data: PortalCount[];
}

export function FundingByPortalChart({ data }: Props) {
  // Drop rows without funding data so the axis isn't dominated by zeros.
  const rows = data
    .filter((p) => p.funding_avg != null && p.funding_avg > 0)
    .sort((a, b) => (b.funding_avg ?? 0) - (a.funding_avg ?? 0));

  if (rows.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
        No portals report a funding ceiling yet.
      </div>
    );
  }

  return (
    <ChartContainer config={chartConfig} className="h-64 w-full">
      <BarChart
        data={rows}
        margin={{ top: 10, right: 12, bottom: 0, left: 0 }}
      >
        <CartesianGrid vertical={false} stroke="var(--color-border)" />
        <XAxis
          dataKey="portal"
          tickLine={false}
          axisLine={false}
          tickMargin={8}
          fontSize={11}
        />
        <YAxis
          tickLine={false}
          axisLine={false}
          width={56}
          fontSize={11}
          tickFormatter={(v: number) => formatEur(v) ?? String(v)}
        />
        <ChartTooltip
          cursor={false}
          content={
            <ChartTooltipContent
              hideLabel
              formatter={(value) => formatEur(Number(value)) ?? String(value)}
            />
          }
        />
        <Bar
          dataKey="funding_avg"
          fill="var(--color-funding_avg)"
          radius={[4, 4, 0, 0]}
        />
      </BarChart>
    </ChartContainer>
  );
}
