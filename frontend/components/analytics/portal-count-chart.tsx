"use client";

import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts";

import {
  type ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import type { PortalCount } from "@/lib/api";

const chartConfig = {
  n: {
    label: "Grants",
    color: "var(--color-primary)",
  },
} satisfies ChartConfig;

interface Props {
  data: PortalCount[];
}

export function PortalCountChart({ data }: Props) {
  // Sort largest → smallest so the chart reads as a ranking.
  const sorted = [...data].sort((a, b) => b.n - a.n);

  return (
    <ChartContainer config={chartConfig} className="h-64 w-full">
      <BarChart data={sorted} margin={{ top: 10, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid vertical={false} stroke="var(--color-border)" />
        <XAxis
          dataKey="portal"
          tickLine={false}
          axisLine={false}
          tickMargin={8}
          fontSize={11}
        />
        <YAxis
          allowDecimals={false}
          tickLine={false}
          axisLine={false}
          width={28}
          fontSize={11}
        />
        <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
        <Bar dataKey="n" fill="var(--color-n)" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ChartContainer>
  );
}
