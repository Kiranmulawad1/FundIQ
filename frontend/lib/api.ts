/**
 * Typed client for the FundIQ FastAPI backend.
 *
 * Browser calls hit `/api/*` (relative) — Next's rewrites in
 * `next.config.ts` proxy those to the backend. Server components fetch
 * the backend directly via `BACKEND_INTERNAL_URL` because the rewrite
 * only intercepts the browser-visible request path.
 */

function apiBase(): string {
  if (typeof window === "undefined") {
    return process.env.BACKEND_INTERNAL_URL ?? "http://localhost:8000";
  }
  return "/api";
}

export type RetrievalMode = "dense" | "hybrid" | "hybrid_rerank";

// String enum values are lowercase on the wire — matches StrEnum on the
// backend (see app/models/base.py: GrantPortal / GrantStatus).
export type GrantPortal =
  | "bmbf"
  | "exist"
  | "kfw"
  | "eic"
  | "horizon"
  | "bayern"
  | "nrw"
  | "bw";

export type GrantStatus = "open" | "closed" | "upcoming" | "rolling";

export interface Citation {
  grant_id: string;
  source_doc_id: string | null;
  source_url: string;
  portal: GrantPortal;
  title: string;
}

export interface GrantSearchHit {
  id: string;
  portal: GrantPortal;
  status: GrantStatus;
  title: string;
  title_en: string | null;
  summary: string;
  sector: string | null;
  country: string;
  federal_state: string | null;
  funding_min_eur: number | null;
  funding_max_eur: number | null;
  deadline: string | null;
  opens_at: string | null;
  source_url: string;
  source_doc_id: string | null;
  created_at: string;
  updated_at: string;
  final_score: number;
  dense_rank: number | null;
  sparse_rank: number | null;
  rrf_score: number | null;
  rerank_score: number | null;
  citation: Citation;
}

export interface GrantSearchRequest {
  query: string;
  limit?: number;
  mode?: RetrievalMode;
  use_hyde?: boolean;
  portal?: GrantPortal;
  country?: string;
}

export type GrantSortKey = "created_at" | "deadline" | "funding_max";

export interface GrantListItem {
  id: string;
  portal: GrantPortal;
  status: GrantStatus;
  title: string;
  title_en: string | null;
  summary: string;
  sector: string | null;
  country: string;
  federal_state: string | null;
  funding_min_eur: number | null;
  funding_max_eur: number | null;
  deadline: string | null;
  opens_at: string | null;
  source_url: string;
  source_doc_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface GrantDetail extends GrantListItem {
  body: string;
  summary_en: string | null;
  eligibility: Record<string, unknown>;
  metadata: Record<string, unknown>;
}

export interface GrantSearchResponse {
  query: string;
  mode: RetrievalMode;
  hits: GrantSearchHit[];
  elapsed_ms: number;
  dense_count: number;
  sparse_count: number;
  rrf_input_count: number;
  rerank_input_count: number;
  used_hyde: boolean;
  hypotheticals: string[] | null;
  cache_hit: boolean;
  cached_for_query: string | null;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

export async function searchGrants(
  body: GrantSearchRequest,
  init?: { signal?: AbortSignal },
): Promise<GrantSearchResponse> {
  const res = await fetch(`${apiBase()}/grants/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      limit: 10,
      mode: "hybrid_rerank" satisfies RetrievalMode,
      use_hyde: false,
      ...body,
    }),
    signal: init?.signal,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new ApiError(res.status, text || `Request failed (${res.status})`);
  }
  return res.json() as Promise<GrantSearchResponse>;
}

export interface PageMeta {
  total: number;
  limit: number;
  offset: number;
  returned: number;
}

export interface GrantListResponse {
  items: GrantListItem[];
  page: PageMeta;
}

export interface GrantListParams {
  portal?: GrantPortal;
  status?: GrantStatus;
  country?: string;
  sort?: GrantSortKey;
  limit?: number;
  offset?: number;
}

export interface PortalCount {
  portal: string;
  n: number;
  n_with_funding_max: number;
  funding_min: number | null;
  funding_max: number | null;
  funding_avg: number | null;
}

export interface StatusCount {
  status: string;
  n: number;
}

export interface FederalStateCount {
  federal_state: string;
  n: number;
}

export interface FundingAnalyticsResponse {
  total_grants: number;
  embedded_grants: number;
  by_portal: PortalCount[];
  by_status: StatusCount[];
  by_federal_state: FederalStateCount[];
  funding_global_min: number | null;
  funding_global_max: number | null;
  funding_global_avg: number | null;
  computed_via: string;
  elapsed_ms: number;
}

export async function getFundingAnalytics(
  init?: { signal?: AbortSignal },
): Promise<FundingAnalyticsResponse> {
  const res = await fetch(`${apiBase()}/analytics/funding`, {
    signal: init?.signal,
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new ApiError(res.status, text || `Request failed (${res.status})`);
  }
  return res.json() as Promise<FundingAnalyticsResponse>;
}

export async function listGrants(
  params: GrantListParams = {},
  init?: { signal?: AbortSignal },
): Promise<GrantListResponse> {
  const qs = new URLSearchParams();
  if (params.portal) qs.set("portal", params.portal);
  if (params.status) qs.set("status", params.status);
  if (params.country) qs.set("country", params.country);
  if (params.sort) qs.set("sort", params.sort);
  if (params.limit != null) qs.set("limit", String(params.limit));
  if (params.offset != null) qs.set("offset", String(params.offset));
  const url = `${apiBase()}/grants${qs.size > 0 ? `?${qs.toString()}` : ""}`;
  const res = await fetch(url, { signal: init?.signal, cache: "no-store" });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new ApiError(res.status, text || `Request failed (${res.status})`);
  }
  return res.json() as Promise<GrantListResponse>;
}

// ---------------------------------------------------------------------------
// Agent recommend (Phase 6)
// ---------------------------------------------------------------------------
export type GrantFit = "high" | "medium" | "low";

export interface AgentGrantRecommendation {
  grant_id: string;
  grant_title: string;
  portal: GrantPortal;
  source_url: string;
  fit: GrantFit;
  rationale: string;
  caveats: string[];
}

export interface AgentExtractedFacts {
  sector: string | null;
  stage: string | null;
  country: string | null;
  federal_state: string | null;
  funding_target_eur: number | null;
}

export interface CandidateScore {
  grant_id: string;
  eligibility_score: number;
  fit_label: GrantFit;
  strengths: string[];
  concerns: string[];
  missing_info: string[];
}

export interface AgentTrace {
  rewritten_query: string;
  extracted_facts: AgentExtractedFacts;
  planner_ms: number;
  retrieval_ms: number;
  scorer_ms: number;
  writer_ms: number;
  critic_ms: number;
  total_ms: number;
  candidate_count: number;
  planner_rationale: string;
  scores: CandidateScore[];
  critic_pass: boolean;
  critic_summary: string;
  critic_findings: CriticFinding[];
  writer_attempts: number;
}

export interface AgentRecommendResponse {
  session_id: string;
  summary: string;
  recommendations: AgentGrantRecommendation[];
  questions_for_user: string[];
  trace: AgentTrace;
}

export interface AgentConversationEntry {
  ts: string;
  query: string;
  summary: string;
  recommendations: AgentGrantRecommendation[];
  questions_for_user: string[];
  trace: AgentTrace;
}

export interface AgentSessionResponse {
  session_id: string;
  history: AgentConversationEntry[];
  is_active: boolean;
}

export interface StartupProfilePayload {
  name?: string;
  sector?: string;
  stage?: string;
  country?: string;
  federal_state?: string;
  funding_target_eur?: number;
  description?: string;
}

export async function recommendGrants(
  query: string,
  opts: {
    sessionId?: string | null;
    startupProfile?: StartupProfilePayload | null;
    signal?: AbortSignal;
  } = {},
): Promise<AgentRecommendResponse> {
  const res = await fetch(`${apiBase()}/agents/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      session_id: opts.sessionId ?? null,
      startup_profile: opts.startupProfile ?? null,
    }),
    signal: opts.signal,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new ApiError(res.status, text || `Request failed (${res.status})`);
  }
  return res.json() as Promise<AgentRecommendResponse>;
}

export async function getAgentSession(
  sessionId: string,
  init?: { signal?: AbortSignal },
): Promise<AgentSessionResponse> {
  const res = await fetch(`${apiBase()}/agents/sessions/${encodeURIComponent(sessionId)}`, {
    signal: init?.signal,
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new ApiError(res.status, text || `Request failed (${res.status})`);
  }
  return res.json() as Promise<AgentSessionResponse>;
}

// ---------------------------------------------------------------------------
// SSE streaming — POST /agents/recommend/stream
// ---------------------------------------------------------------------------
export type AgentStage =
  | "planner"
  | "retriever"
  | "scorer"
  | "writer"
  | "critic";

export type CriticFindingType =
  | "citation_faithfulness"
  | "fit_alignment"
  | "caveat_omission"
  | "language_mismatch"
  | "profile_misuse"
  | "other";

export type CriticSeverity = "high" | "medium" | "low";

export interface CriticFinding {
  type: CriticFindingType;
  severity: CriticSeverity;
  grant_id: string | null;
  message: string;
}
export type AgentStageStatus = "start" | "done";

export interface AgentStageEvent {
  stage: AgentStage;
  status: AgentStageStatus;
  elapsed_ms: number;
  // Stage-specific payload (present on `status: done`):
  planner_ms?: number;
  retrieval_ms?: number;
  scorer_ms?: number;
  writer_ms?: number;
  critic_ms?: number;
  rewritten_query?: string;
  extracted_facts?: AgentExtractedFacts;
  rationale?: string;
  candidate_count?: number;
  score_count?: number;
  fit_labels?: GrantFit[];
  recommendation_count?: number;
  overall_pass?: boolean;
  finding_count?: number;
  // Retry-loop bookkeeping. `attempt` is 1 on the first pass, 2 on the
  // retry. `retry: true` on a Writer-start event signals the frontend
  // to clear its partial-summary buffer.
  attempt?: number;
  retry?: boolean;
}

export interface AgentStreamCallbacks {
  onStage?: (e: AgentStageEvent) => void;
  /** Fires for each Gemini-streamed text chunk during the Writer phase. */
  onWriterDelta?: (chunk: string) => void;
  onDone?: (response: AgentRecommendResponse) => void;
  onError?: (message: string) => void;
  signal?: AbortSignal;
}

/**
 * Stream the agent graph's progress. Resolves when the stream closes;
 * `onDone` fires with the full payload (same shape as `recommendGrants`)
 * once the `done` event arrives. `onError` fires for upstream errors
 * embedded in the SSE stream or for transport failures.
 */
export async function streamRecommendGrants(
  query: string,
  opts: {
    sessionId?: string | null;
    startupProfile?: StartupProfilePayload | null;
  } & AgentStreamCallbacks = {},
): Promise<void> {
  const res = await fetch(`${apiBase()}/agents/recommend/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({
      query,
      session_id: opts.sessionId ?? null,
      startup_profile: opts.startupProfile ?? null,
    }),
    signal: opts.signal,
  });

  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => res.statusText);
    throw new ApiError(res.status, text || `Stream request failed (${res.status})`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      // Normalise CRLF: sse-starlette uses \r\n per the SSE spec, but a
      // future producer might use \n only. One delimiter to handle both.
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

      // SSE messages are delimited by blank lines.
      let sepIndex = buffer.indexOf("\n\n");
      while (sepIndex !== -1) {
        const raw = buffer.slice(0, sepIndex);
        buffer = buffer.slice(sepIndex + 2);
        const parsed = parseSseMessage(raw);
        if (parsed) dispatch(parsed, opts);
        sepIndex = buffer.indexOf("\n\n");
      }
    }
  } finally {
    reader.releaseLock();
  }
}

interface SseMessage {
  event: string;
  data: string;
}

function parseSseMessage(raw: string): SseMessage | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
    // ignore id:, retry:, comments (`:` prefix).
  }
  if (dataLines.length === 0) return null;
  return { event, data: dataLines.join("\n") };
}

function dispatch(msg: SseMessage, opts: AgentStreamCallbacks): void {
  try {
    const payload = JSON.parse(msg.data);
    if (msg.event === "stage" && opts.onStage) {
      opts.onStage(payload as AgentStageEvent);
    } else if (msg.event === "writer_delta" && opts.onWriterDelta) {
      const text = (payload as { text?: string }).text;
      if (typeof text === "string") opts.onWriterDelta(text);
    } else if (msg.event === "done" && opts.onDone) {
      opts.onDone(payload as AgentRecommendResponse);
    } else if (msg.event === "error" && opts.onError) {
      opts.onError(typeof payload?.message === "string" ? payload.message : msg.data);
    }
  } catch {
    // Drop malformed events silently — the stream should self-recover.
  }
}

/**
 * Pull the `summary` string out of an in-progress JSON buffer. The Writer
 * uses Gemini's JSON mode, so the response shape is:
 *
 *     {"summary":"…","recommendations":[…],"questions_for_user":[…]}
 *
 * Because `summary` is the first field, it gets emitted first, and we
 * can rebuild it character-by-character as chunks arrive. Returns the
 * empty string until the `"summary":"` prefix has been seen.
 *
 * Handles standard JSON escape sequences (\n, \t, \\, \", \uXXXX).
 */
export function extractPartialSummary(buffer: string): string {
  const keyIdx = buffer.indexOf('"summary"');
  if (keyIdx < 0) return "";
  const colonIdx = buffer.indexOf(":", keyIdx);
  if (colonIdx < 0) return "";

  // Skip whitespace + opening quote.
  let i = colonIdx + 1;
  while (i < buffer.length && /\s/.test(buffer[i])) i++;
  if (buffer[i] !== '"') return "";
  i++;

  let out = "";
  while (i < buffer.length) {
    const ch = buffer[i];
    if (ch === "\\") {
      if (i + 1 >= buffer.length) break;
      const next = buffer[i + 1];
      if (next === "n") out += "\n";
      else if (next === "t") out += "\t";
      else if (next === "r") out += "\r";
      else if (next === '"') out += '"';
      else if (next === "\\") out += "\\";
      else if (next === "/") out += "/";
      else if (next === "u" && i + 5 < buffer.length) {
        const hex = buffer.slice(i + 2, i + 6);
        out += String.fromCharCode(parseInt(hex, 16));
        i += 6;
        continue;
      } else {
        out += next;
      }
      i += 2;
    } else if (ch === '"') {
      break; // closing quote — summary is complete
    } else {
      out += ch;
      i++;
    }
  }
  return out;
}

export async function deleteAgentSession(
  sessionId: string,
  init?: { signal?: AbortSignal },
): Promise<void> {
  const res = await fetch(`${apiBase()}/agents/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
    signal: init?.signal,
  });
  if (!res.ok && res.status !== 204) {
    const text = await res.text().catch(() => res.statusText);
    throw new ApiError(res.status, text || `Request failed (${res.status})`);
  }
}

export async function getGrant(
  id: string,
  init?: { signal?: AbortSignal },
): Promise<GrantDetail | null> {
  const res = await fetch(`${apiBase()}/grants/${encodeURIComponent(id)}`, {
    signal: init?.signal,
    // Detail pages render fresh — corpus updates from the scheduler should
    // be visible without a stale ISR cache holding them back.
    cache: "no-store",
  });
  if (res.status === 404) return null;
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new ApiError(res.status, text || `Request failed (${res.status})`);
  }
  return res.json() as Promise<GrantDetail>;
}
