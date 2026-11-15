// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The entry reads the trial's brief (provider directory, sealed quality
// table, workload manifest), probes every provider's catalog route, builds
// one catalog, registers its port and catalog with the env, serves the
// client's 20 requests, and exits 0 when the env says the workload is done.
//
// Routing (what the house baseline does not do):
//   - quality: every free model meeting a request's requirements is ranked
//     by the quality table's rating_milli, best first. A paid model is never
//     a legal pick and is never chosen.
//   - failover: a provider call refused with 402, 429 or 5xx is not the
//     answer. The request is retried on the next best free model that meets
//     it, and the refusal is counted against the provider.
//   - budgets are never published, so they are learned: a provider that
//     refuses once is suspect and skipped for the next request; one that
//     refuses twice running is dead for the rest of the trial (its budget is
//     spent or its outage has begun). A scheduled single-call failure costs
//     one retry and the provider comes back.
//   - wires: all three upstream wires are spoken. A JSON-mode request
//     (response_format on the OpenAI wire) is routed only to a wire that can
//     carry it (openai or gemini), never to an anthropic provider.
//   - the catalog is probed once and used for --list and for register, byte
//     for byte the same document.
//
// Determinism: nothing here reads the wall clock or unseeded randomness, so
// a replay sees the same state export byte for byte. Requests arrive one at
// a time and are answered in arrival order.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicToOpenAIResponse,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIChunksFromAnthropic,
  openAIChunksFromGemini,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  openAIRequestToAnthropic,
  openAIToGemini,
  sseEvent,
  type ServedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order. */
let manifestIndex = 0;
/** Provider calls made this trial, for the tie-break and for spread. */
let upstreamCalls = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

interface ProviderState {
  calls: number;
  consecutiveFailures: number;
  dead: boolean;
  /** Skip this provider for the next request after one refusal. */
  suspect: boolean;
}

const providerState = new Map<string, ProviderState>();

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, consecutiveFailures: 0, dead: false, suspect: false };
    providerState.set(providerId, state);
  }
  return state;
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/**
 * Every free model meeting a request's requirements, best rating first. A
 * JSON-mode request (response_format) is routed only to a wire that can
 * carry it: openai and gemini, never anthropic. Dead providers are out;
 * suspect ones sort last so a scheduled single-call failure does not take
 * the next request down with it.
 */
function candidatesFor(requires: Requirement, jsonWire: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonWire && probe.provider.wire === 'anthropic') continue;
    const state = stateOf(probe.provider.id);
    if (state.dead) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    const suspectA = stateOf(a.probe.provider.id).suspect ? 1 : 0;
    const suspectB = stateOf(b.probe.provider.id).suspect ? 1 : 0;
    if (suspectA !== suspectB) return suspectA - suspectB;
    if (b.rating !== a.rating) return b.rating - a.rating;
    const callsA = stateOf(a.probe.provider.id).calls;
    const callsB = stateOf(b.probe.provider.id).calls;
    if (callsA !== callsB) return callsA - callsB;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : 1;
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

type ClientWire = 'openai' | 'anthropic';

interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: Json;
}

/** The upstream request for one candidate: the client body rewritten into the provider's wire. */
function upstreamFor(clientWire: ClientWire, candidate: Candidate, body: Json): UpstreamCall {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  const stream = body.stream === true;
  if (provider.wire === 'openai') {
    if (stream) headers.accept = 'text/event-stream';
    // A streamed answer must carry its final usage chunk: the client checks
    // the response's usage, and the provider only sends usage when the
    // request asks for it. The Anthropic rewrite (anthropicToOpenAI) already
    // asks; the OpenAI pass-through must ask too.
    const upstreamBody: Json =
      clientWire === 'openai'
        ? { ...body, model: candidate.model.id, ...(stream ? { stream_options: { include_usage: true } } : {}) }
        : anthropicToOpenAI(body, candidate.model.id);
    return { url: `${base}/v1/chat/completions`, headers, body: upstreamBody };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    if (stream) headers.accept = 'text/event-stream';
    return { url: `${base}/v1/messages`, headers, body: clientWire === 'anthropic' ? { ...body, model: candidate.model.id } : openAIRequestToAnthropic(body, candidate.model.id) };
  }
  // gemini
  const converted = clientWire === 'openai' ? openAIToGemini(body, candidate.model.id) : anthropicToGemini(body, candidate.model.id);
  const suffix = stream ? ':streamGenerateContent?alt=sse' : ':generateContent';
  return { url: `${base}/v1beta/models/${encodeURIComponent(converted.model)}${suffix}`, headers, body: converted.request };
}

/** Record a provider's refusal: suspect after one, dead after two running. A success clears both. */
function recordRefusal(providerId: string): void {
  const state = stateOf(providerId);
  state.consecutiveFailures += 1;
  state.suspect = true;
  if (state.consecutiveFailures >= 2) state.dead = true;
  log(`provider ${providerId} refused (consecutive ${state.consecutiveFailures}${state.dead ? ', dead' : ''})`);
}

function recordSuccess(providerId: string): void {
  const state = stateOf(providerId);
  state.consecutiveFailures = 0;
  state.suspect = false;
}

/** Clear every suspect flag: a new request is a fresh chance for a provider that failed once. */
function clearSuspects(): void {
  for (const state of providerState.values()) state.suspect = false;
}

async function callUpstream(call: UpstreamCall): Promise<Response> {
  upstreamCalls += 1;
  return fetch(call.url, { method: 'POST', headers: call.headers, body: JSON.stringify(call.body) });
}

/** Translate one provider's JSON answer into the client's wire and send it. */
function sendAnswer(res: ServerResponse, clientWire: ClientWire, candidate: Candidate, answer: Json): void {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  if (clientWire === 'openai') {
    if (wire === 'openai') return sendJson(res, 200, { ...answer, model });
    if (wire === 'anthropic') return sendJson(res, 200, anthropicToOpenAIResponse(answer, model));
    return sendJson(res, 200, geminiToOpenAI(answer, model));
  }
  if (wire === 'anthropic') return sendJson(res, 200, { ...answer, model });
  if (wire === 'openai') return sendJson(res, 200, openAIToAnthropic(answer, model));
  return sendJson(res, 200, geminiToAnthropic(answer, model));
}

/** Stream one provider's SSE answer to the client in the client's wire. */
async function streamAnswer(res: ServerResponse, clientWire: ClientWire, candidate: Candidate, upstream: Response): Promise<void> {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  if (upstream.body === null) {
    res.end();
    return;
  }
  if (clientWire === 'openai') {
    if (wire === 'openai') {
      const reader = upstream.body.getReader();
      try {
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          res.write(Buffer.from(value));
        }
      } finally {
        reader.releaseLock();
        res.end();
      }
      return;
    }
    if (wire === 'anthropic') {
      for await (const event of openAIChunksFromAnthropic(upstream.body, model)) res.write(event);
      res.end();
      return;
    }
    for await (const event of openAIChunksFromGemini(upstream.body, model)) res.write(event);
    res.end();
    return;
  }
  // anthropic client
  if (wire === 'anthropic') {
    const reader = upstream.body.getReader();
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        res.write(Buffer.from(value));
      }
    } finally {
      reader.releaseLock();
      res.end();
    }
    return;
  }
  if (wire === 'openai') {
    for await (const event of anthropicEventsFromOpenAI(upstream.body, model)) res.write(event);
    res.end();
    return;
  }
  for await (const event of anthropicEventsFromGemini(upstream.body, model)) res.write(event);
  res.end();
}

/**
 * Serve one client request: try the ranked candidates in order, failing over
 * on every refusal, until one answers. Returns true when the request was
 * served, false when no free model could serve it.
 */
async function serve(req: IncomingMessage, res: ServerResponse, clientWire: ClientWire): Promise<boolean> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, clientWire, 400, 'invalid_request_error', 'the request body is not one JSON object');
    return false;
  }
  const stream = body.stream === true;
  const jsonWire = clientWire === 'openai' && isRecord(body.response_format);
  const candidates = candidatesFor(requirementsFor(body), jsonWire);
  if (candidates.length === 0) {
    sendError(res, clientWire, 503, clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
    return false;
  }
  clearSuspects();
  for (const candidate of candidates) {
    const providerId = candidate.probe.provider.id;
    const state = stateOf(providerId);
    let upstream: Response;
    try {
      upstream = await callUpstream(upstreamFor(clientWire, candidate, body));
    } catch (err) {
      recordRefusal(providerId);
      log(`call to ${providerId} failed to reach it: ${(err as Error).message}`);
      continue;
    }
    state.calls += 1;
    if (!upstream.ok) {
      const text = await upstream.text().catch(() => '');
      log(`provider ${providerId} answered HTTP ${upstream.status}: ${text.slice(0, 200)}`);
      recordRefusal(providerId);
      continue;
    }
    recordSuccess(providerId);
    log(`request ${manifestIndex} served by ${providerId}/${candidate.model.id} (rating ${candidate.rating}, calls ${upstreamCalls})`);
    if (stream && upstream.body !== null) {
      await streamAnswer(res, clientWire, candidate, upstream);
      return true;
    }
    let answer: unknown;
    try {
      answer = await upstream.json();
    } catch {
      sendError(res, clientWire, 502, clientWire === 'anthropic' ? 'api_error' : 'upstream_unreachable', 'the provider answered with a body that is not JSON');
      return false;
    }
    if (!isRecord(answer)) {
      sendError(res, clientWire, 502, clientWire === 'anthropic' ? 'api_error' : 'upstream_unreachable', 'the provider answered with a body that is not a JSON object');
      return false;
    }
    sendAnswer(res, clientWire, candidate, answer);
    return true;
  }
  sendError(res, clientWire, 503, clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'every free model that meets this request refused it');
  return false;
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; retry briefly while the env comes up. */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError = '';
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = (err as Error).message;
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
  throw new Error(`register failed: ${lastError}`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          log(`work done after ${manifestIndex} requests and ${upstreamCalls} provider calls`);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
