import Link from "next/link";

import { SaveButton } from "@/components/save-button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import type { GrantListItem } from "@/lib/api";
import { formatDeadline, formatFundingRange } from "@/lib/format";

interface Props {
  grant: GrantListItem;
}

export function GrantRow({ grant }: Props) {
  const funding = formatFundingRange(grant.funding_min_eur, grant.funding_max_eur);
  return (
    <Card className="transition-colors hover:border-foreground/20">
      <CardContent className="space-y-2 py-4">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant="secondary" className="uppercase">
            {grant.portal}
          </Badge>
          <Badge variant="outline" className="uppercase">
            {grant.status}
          </Badge>
          {grant.country && <Badge variant="outline">{grant.country}</Badge>}
          {grant.federal_state && (
            <Badge variant="outline">{grant.federal_state}</Badge>
          )}
          <div className="ml-auto">
            <SaveButton
              grant={{
                id: grant.id,
                portal: grant.portal,
                title: grant.title,
                source_url: grant.source_url,
                source_doc_id: grant.source_doc_id,
              }}
            />
          </div>
        </div>
        <h3 className="text-base font-semibold leading-snug">
          <Link href={`/grants/${grant.id}`} className="hover:underline">
            {grant.title}
          </Link>
        </h3>
        {grant.summary && (
          <p className="text-sm leading-relaxed text-muted-foreground line-clamp-2">
            {grant.summary}
          </p>
        )}
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
          {funding && (
            <span>
              <span className="text-foreground/80">Funding:</span> {funding}
            </span>
          )}
          <span>{formatDeadline(grant.deadline)}</span>
        </div>
      </CardContent>
    </Card>
  );
}
