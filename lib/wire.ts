// The client wires and the three upstream wires.
//
// The trial's client calls the entry as an OpenAI service (POST
// /v1/chat/completions) or an Anthropic one (POST /v1/messages), with a JSON
// body or with server-sent events. Each request is forwarded to one provider
// on that provider's own wire and the answer is returned in the wire the
// client used:
//   openai client    -> openai provider: pass through (model replaced)
//   openai client    -> anthropic provider: rewrite to Messages, answer back
//   openai client    -> gemini provider: rewrite to generateContent, back
//   anthropic client -> anthropic provider: pass through (model replaced)
//   anthropic client -> openai provider: rewrite to Chat Completions, back
//   anthropic client -> gemini provider: rewrite to generateContent, back
//
// The provider echoes what it parsed and the client compares that echo with
// what it sent, so every field the request carried must survive the rewrite:
// system prompt, user text (images included), tools, stop sequences,
// max_tokens, JSON mode. A JSON-mode request carries response_format on the
// OpenAI wire; the Anthropic wire cannot carry it, so those requests are
// routed to a wire that can (the router in main.ts enforces this).
//
// Stop reasons map: end_turn <-> stop, max_tokens <-> length, tool_use <->
// tool_calls, refusal <-> content_filter. Usage passes through as the wire
// reports it. Everything here is a pure function of its arguments: no clock
// and no randomness, so the same body and the same provider answer give the
// same bytes on a replay. The model listings use fixed timestamps for the
// same reason.

import { isRecord, type Json } from './brief.ts';

export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

const FIXED_OPENAI_CREATED = 1767225600;
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function finishReasonOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as OpenAI's finish_reason. */
export function finishReasonOfGemini(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY' || finish === 'RECITATION' || finish === 'BLOCKLIST') return 'content_filter';
  return 'stop';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** One OpenAI message's image_url part as an Anthropic image block; null when it is not one. */
function imageBlockOf(part: Json): Json | null {
  if (part.type !== 'image_url' || !isRecord(part.image_url)) return null;
  const url = typeof part.image_url.url === 'string' ? part.image_url.url : '';
  if (url.startsWith('data:')) {
    const match = /^data:([^;,]+)?;base64,(.*)$/.exec(url);
    if (match !== null) {
      return { type: 'image', source: { type: 'base64', media_type: match[1] ?? 'image/png', data: match[2] } };
    }
    return null;
  }
  if (url !== '') return { type: 'image', source: { type: 'url', url } };
  return null;
}

/** An OpenAI message list as Anthropic messages: text, images, tool calls and tool results carried over. */
function openAIMessagesToAnthropic(messages: unknown): Json[] {
  const out: Json[] = [];
  if (!Array.isArray(messages)) return out;
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' ? 'assistant' : raw.role === 'system' ? 'system' : 'user';
    if (role === 'system') continue; // the system prompt travels separately
    if (raw.role === 'tool') {
      out.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof raw.content === 'string' ? raw.content : JSON.stringify(raw.content ?? ''),
          },
        ],
      });
      continue;
    }
    const content = raw.content;
    if (typeof content === 'string') {
      const blocks: Json[] = content !== '' ? [{ type: 'text', text: content }] : [];
      if (role === 'assistant' && Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          blocks.push({
            type: 'tool_use',
            id: typeof call.id === 'string' ? call.id : '',
            name: typeof call.function.name === 'string' ? call.function.name : '',
            input: argumentsOf(call.function.arguments),
          });
        }
      }
      out.push({ role, content: blocks.length === 1 && blocks[0].type === 'text' ? content : blocks });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
      else {
        const image = imageBlockOf(part);
        if (image !== null) blocks.push(image);
      }
    }
    if (role === 'assistant' && Array.isArray(raw.tool_calls)) {
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
    }
    if (blocks.length > 0) out.push({ role, content: blocks });
  }
  return out;
}

/** OpenAI tools as Anthropic tools. */
function openAIToolsToAnthropic(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      },
    ];
  });
}

/** OpenAI tool_choice as Anthropic tool_choice. */
function openAIToolChoiceToAnthropic(choice: unknown): Json | null {
  if (!isRecord(choice)) {
    if (choice === 'auto') return { type: 'auto' };
    if (choice === 'required' || choice === 'any') return { type: 'any' };
    if (choice === 'none') return { type: 'none' };
    return null;
  }
  if (choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    return { type: 'tool', name: choice.function.name };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that
 * names `model`: system prompt, messages, tools, tool choice, output cap,
 * temperature, top_p, stop sequences and streaming pass through.
 */
export function openAIRequestToAnthropic(body: Json, model: string): Json {
  const messages = openAIMessagesToAnthropic(body.messages);
  const system = Array.isArray(body.messages)
    ? body.messages.flatMap((m) => (isRecord(m) && m.role === 'system' && typeof m.content === 'string' ? [m.content] : [])).join('\n')
    : '';
  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stop = body.stop;
  if (typeof stop === 'string') out.stop_sequences = [stop];
  else if (Array.isArray(stop)) out.stop_sequences = stop.filter((s): s is string => typeof s === 'string');
  const tools = openAIToolsToAnthropic(body.tools);
  if (tools.length > 0) out.tools = tools;
  const choice = openAIToolChoiceToAnthropic(body.tool_choice);
  if (choice !== null) out.tool_choice = choice;
  if (body.stream === true) out.stream = true;
  return out;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that
 * names `model`: the system prompt becomes a system message, image blocks
 * become image_url data URLs, tool_use blocks become tool_calls, tool_result
 * blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** An Anthropic Messages object as a Chat Completions answer. */
export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const content: Json[] = Array.isArray(body.content) ? body.content : [];
  const text = textOf(content);
  const calls = content.flatMap((block) => {
    if (!isRecord(block) || block.type !== 'tool_use') return [];
    return [
      {
        id: typeof block.id === 'string' ? block.id : 'call_unknown',
        type: 'function',
        function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
      },
    ];
  });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `chatcmpl_${typeof body.id === 'string' ? body.id.replace(/^msg_/, '') : 'aggregate'}`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text !== '' ? text : null, ...(calls.length > 0 ? { tool_calls: calls } : {}) },
        finish_reason: finishReasonOf(body.stop_reason),
      },
    ],
    usage: { prompt_tokens: numberOrZero(usage.input_tokens), completion_tokens: numberOrZero(usage.output_tokens), total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Parse an SSE byte stream into its JSON data payloads, in order. */
export async function* sseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<unknown> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const line of frame.split('\n')) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;
          const payload = trimmed.slice(5).trim();
          if (payload === '[DONE]') {
            yield '[DONE]';
            continue;
          }
          try {
            yield JSON.parse(payload);
          } catch {
            // Not JSON: skip the frame.
          }
        }
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      for (const line of buffer.split('\n')) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('data:')) continue;
        const payload = trimmed.slice(5).trim();
        if (payload === '[DONE]') yield '[DONE]';
        else {
          try {
            yield JSON.parse(payload);
          } catch {
            // Not JSON: skip the frame.
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence:
 * message_start on the first chunk, content_block_start/delta/stop per text
 * block or tool call, then message_delta with the stop reason and usage, then
 * message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  for await (const frame of sseFrames(upstream)) {
    if (frame === '[DONE]') {
      yield* startEvents();
      yield* finishEvents(pendingStop ?? 'end_turn');
      continue;
    }
    if (!isRecord(frame)) continue;
    if (isRecord(frame.usage)) {
      inputTokens = numberOrZero(frame.usage.prompt_tokens);
      outputTokens = numberOrZero(frame.usage.completion_tokens);
    }
    const choices = Array.isArray(frame.choices) ? frame.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    yield* startEvents();
    if (typeof delta.content === 'string' && delta.content !== '') {
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        yield sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } });
      }
      yield sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } });
    }
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
      let index = toolIndexes.get(upstreamIndex);
      if (index === undefined) {
        if (textIndex !== -1) {
          yield sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex });
          textIndex = -1;
        }
        index = nextIndex;
        nextIndex += 1;
        toolIndexes.set(upstreamIndex, index);
        const fn = isRecord(call.function) ? call.function : {};
        yield sseEvent('content_block_start', {
          type: 'content_block_start',
          index,
          content_block: {
            type: 'tool_use',
            id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
            name: typeof fn.name === 'string' ? fn.name : '',
            input: {},
          },
        });
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string' && fn.arguments !== '') {
        yield sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } });
      }
    }
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
  }
  if (!finished) {
    yield* startEvents();
    yield* finishEvents(pendingStop ?? 'end_turn');
  }
}

/**
 * An Anthropic SSE stream as Chat Completions chunks: text deltas become
 * content deltas, tool_use blocks become tool_calls deltas, and the final
 * message_delta becomes the chunk carrying the finish reason and the usage.
 */
export async function* openAIChunksFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let id = 'chatcmpl_stream';
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason: string | null = null;
  let finished = false;
  const toolIds = new Map<number, string>();
  const toolNames = new Map<number, string>();

  const chunk = (delta: Json, finish: string | null): string =>
    `data: ${JSON.stringify({ id, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;

  for await (const frame of sseFrames(upstream)) {
    if (!isRecord(frame)) continue;
    const type = frame.type;
    if (type === 'message_start' && isRecord(frame.message)) {
      if (typeof frame.message.id === 'string') id = `chatcmpl_${frame.message.id.replace(/^msg_/, '')}`;
      if (isRecord(frame.message.usage)) inputTokens = numberOrZero(frame.message.usage.input_tokens);
      continue;
    }
    if (type === 'content_block_start' && isRecord(frame.content_block)) {
      const block = frame.content_block;
      const index = typeof frame.index === 'number' ? frame.index : 0;
      if (block.type === 'tool_use') {
        toolIds.set(index, typeof block.id === 'string' ? block.id : `call_${index}`);
        toolNames.set(index, typeof block.name === 'string' ? block.name : '');
        yield chunk({ tool_calls: [{ index, id: toolIds.get(index), type: 'function', function: { name: toolNames.get(index), arguments: '' } }] }, null);
      }
      continue;
    }
    if (type === 'content_block_delta' && isRecord(frame.delta)) {
      const delta = frame.delta;
      const index = typeof frame.index === 'number' ? frame.index : 0;
      if (delta.type === 'text_delta' && typeof delta.text === 'string') yield chunk({ content: delta.text }, null);
      else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string') {
        yield chunk({ tool_calls: [{ index, id: toolIds.get(index), type: 'function', function: { name: toolNames.get(index), arguments: delta.partial_json } }] }, null);
      }
      continue;
    }
    if (type === 'message_delta' && isRecord(frame.delta)) {
      if (typeof frame.delta.stop_reason === 'string') stopReason = finishReasonOf(frame.delta.stop_reason);
      if (isRecord(frame.usage)) outputTokens = numberOrZero(frame.usage.output_tokens);
      continue;
    }
    if (type === 'message_stop') {
      if (!finished) {
        finished = true;
        yield chunk({}, stopReason ?? 'stop');
        yield `data: ${JSON.stringify({ id, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta: {}, finish_reason: null }], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } })}\n\n`;
        yield 'data: [DONE]\n\n';
      }
      continue;
    }
  }
  if (!finished) {
    yield chunk({}, stopReason ?? 'stop');
    yield `data: ${JSON.stringify({ id, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta: {}, finish_reason: null }], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } })}\n\n`;
    yield 'data: [DONE]\n\n';
  }
}

// ---- Gemini ----

/** One Gemini content part from an OpenAI message part, or null when it is not one. */
function geminiPartOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const match = /^data:([^;,]+)?;base64,(.*)$/.exec(url);
    if (match !== null) return { inlineData: { mimeType: match[1] ?? 'image/png', data: match[2] } };
  }
  return null;
}

/** An OpenAI message list as Gemini contents. */
function openAIMessagesToGemini(messages: unknown): { contents: Json[]; systemInstruction: Json | null } {
  const contents: Json[] = [];
  let systemInstruction: Json | null = null;
  if (!Array.isArray(messages)) return { contents, systemInstruction };
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = typeof raw.content === 'string' ? raw.content : textOf(raw.content);
      if (text !== '') systemInstruction = { parts: [{ text }] };
      continue;
    }
    if (raw.role === 'tool') {
      contents.push({
        role: 'user',
        parts: [
          {
            functionResponse: {
              name: typeof raw.name === 'string' ? raw.name : '',
              response: { result: typeof raw.content === 'string' ? raw.content : JSON.stringify(raw.content ?? '') },
            },
          },
        ],
      });
      continue;
    }
    const role = raw.role === 'assistant' ? 'model' : 'user';
    const parts: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        const p = geminiPartOf(part);
        if (p !== null) parts.push(p);
      }
    }
    if (role === 'model' && Array.isArray(raw.tool_calls)) {
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
      }
    }
    if (parts.length > 0) contents.push({ role, parts });
  }
  return { contents, systemInstruction };
}

/** Anthropic messages as Gemini contents, with the system prompt as systemInstruction. */
function anthropicMessagesToGemini(body: Json): { contents: Json[]; systemInstruction: Json | null } {
  const system = textOf(body.system);
  const systemInstruction = system !== '' ? { parts: [{ text: system }] } : null;
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role: assistant ? 'model' : 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
      else if (block.type === 'image' && isRecord(block.source)) {
        const source = block.source;
        if (source.type === 'base64' && typeof source.data === 'string') {
          parts.push({ inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } });
        } else if (source.type === 'url' && typeof source.url === 'string') {
          parts.push({ fileData: { mimeType: 'image/png', fileUri: source.url } });
        }
      } else if (block.type === 'tool_use') {
        parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
      } else if (block.type === 'tool_result') {
        parts.push({
          functionResponse: {
            name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
            response: { result: toolResultText(block.content) },
          },
        });
      }
    }
    if (parts.length > 0) contents.push({ role: assistant ? 'model' : 'user', parts });
  }
  return { contents, systemInstruction };
}

/** OpenAI tools as Gemini function declarations. */
function openAIToolsToGemini(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      },
    ];
  });
}

/** Anthropic tools as Gemini function declarations. */
function anthropicToolsToGemini(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      },
    ];
  });
}

function geminiGenerationConfig(body: Json, maxTokens: number | undefined, jsonMode: boolean): Json {
  const config: Json = {};
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stop = body.stop;
  if (typeof stop === 'string') config.stopSequences = [stop];
  else if (Array.isArray(stop)) config.stopSequences = stop.filter((s): s is string => typeof s === 'string');
  else if (Array.isArray(body.stop_sequences)) config.stopSequences = body.stop_sequences.filter((s): s is string => typeof s === 'string');
  if (jsonMode) config.responseMimeType = 'application/json';
  return config;
}

function geminiToolConfig(toolChoice: unknown): Json | null {
  if (!isRecord(toolChoice)) {
    if (toolChoice === 'auto') return { functionCallingConfig: { mode: 'AUTO' } };
    if (toolChoice === 'required' || toolChoice === 'any') return { functionCallingConfig: { mode: 'ANY' } };
    if (toolChoice === 'none') return { functionCallingConfig: { mode: 'NONE' } };
    return null;
  }
  if (toolChoice.type === 'auto') return { functionCallingConfig: { mode: 'AUTO' } };
  if (toolChoice.type === 'any') return { functionCallingConfig: { mode: 'ANY' } };
  if (toolChoice.type === 'none') return { functionCallingConfig: { mode: 'NONE' } };
  if (toolChoice.type === 'tool' && typeof toolChoice.name === 'string') {
    return { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [toolChoice.name] } };
  }
  return null;
}

/** An OpenAI Chat Completions request as a Gemini generateContent request. */
export function openAIToGemini(body: Json, model: string): { model: string; request: Json } {
  const { contents, systemInstruction } = openAIMessagesToGemini(body.messages);
  const jsonMode = isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const request: Json = { contents };
  if (systemInstruction !== null) request.systemInstruction = systemInstruction;
  const declarations = openAIToolsToGemini(body.tools);
  if (declarations.length > 0) request.tools = [{ functionDeclarations: declarations }];
  const toolConfig = geminiToolConfig(body.tool_choice);
  if (toolConfig !== null) request.toolConfig = toolConfig;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  request.generationConfig = geminiGenerationConfig(body, maxTokens, jsonMode);
  return { model, request };
}

/** An Anthropic Messages request as a Gemini generateContent request. */
export function anthropicToGemini(body: Json, model: string): { model: string; request: Json } {
  const { contents, systemInstruction } = anthropicMessagesToGemini(body);
  const request: Json = { contents };
  if (systemInstruction !== null) request.systemInstruction = systemInstruction;
  const declarations = anthropicToolsToGemini(body.tools);
  if (declarations.length > 0) request.tools = [{ functionDeclarations: declarations }];
  const toolConfig = geminiToolConfig(body.tool_choice);
  if (toolConfig !== null) request.toolConfig = toolConfig;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : undefined;
  request.generationConfig = geminiGenerationConfig(body, maxTokens, false);
  return { model, request };
}

/** A Gemini generateContent answer as a Chat Completions answer. */
export function geminiToOpenAI(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const calls = parts.flatMap((p, i) => {
    if (!isRecord(p) || !isRecord(p.functionCall)) return [];
    const call = p.functionCall;
    return [
      {
        id: `call_${i}`,
        type: 'function',
        function: { name: typeof call.name === 'string' ? call.name : '', arguments: JSON.stringify(isRecord(call.args) ? call.args : {}) },
      },
    ];
  });
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const finish = calls.length > 0 ? 'tool_calls' : finishReasonOfGemini(candidate.finishReason);
  return {
    id: `chatcmpl_gemini`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text !== '' ? text : null, ...(calls.length > 0 ? { tool_calls: calls } : {}) },
        finish_reason: finish,
      },
    ],
    usage: { prompt_tokens: numberOrZero(usage.promptTokenCount), completion_tokens: numberOrZero(usage.candidatesTokenCount), total_tokens: numberOrZero(usage.totalTokenCount) },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
export function geminiToAnthropic(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const blocks: Json[] = [];
  for (const p of parts) {
    if (!isRecord(p)) continue;
    if (typeof p.text === 'string' && p.text !== '') blocks.push({ type: 'text', text: p.text });
    else if (isRecord(p.functionCall)) {
      blocks.push({
        type: 'tool_use',
        id: `toolu_gemini_${blocks.length}`,
        name: typeof p.functionCall.name === 'string' ? p.functionCall.name : '',
        input: isRecord(p.functionCall.args) ? p.functionCall.args : {},
      });
    }
  }
  if (blocks.length === 0) blocks.push({ type: 'text', text: '' });
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const stop = blocks.some((b) => b.type === 'tool_use') ? 'tool_use' : stopReasonOf(finishReasonOfGemini(candidate.finishReason));
  return {
    id: 'msg_gemini',
    type: 'message',
    role: 'assistant',
    model,
    content: blocks,
    stop_reason: stop,
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.promptTokenCount), output_tokens: numberOrZero(usage.candidatesTokenCount) },
  };
}

/**
 * A Gemini streamGenerateContent SSE stream as Chat Completions chunks: text
 * parts become content deltas, functionCall parts become tool_calls deltas,
 * and the final usageMetadata chunk becomes the chunk carrying the finish
 * reason and the usage.
 */
export async function* openAIChunksFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason: string | null = null;
  let finished = false;
  let callIndex = 0;
  const chunk = (delta: Json, finish: string | null): string =>
    `data: ${JSON.stringify({ id: 'chatcmpl_gemini_stream', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;

  for await (const frame of sseFrames(upstream)) {
    if (!isRecord(frame)) continue;
    if (isRecord(frame.usageMetadata)) {
      inputTokens = numberOrZero(frame.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(frame.usageMetadata.candidatesTokenCount);
    }
    const candidates = Array.isArray(frame.candidates) ? frame.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) continue;
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const p of parts) {
      if (!isRecord(p)) continue;
      if (typeof p.text === 'string' && p.text !== '') yield chunk({ content: p.text }, null);
      else if (isRecord(p.functionCall)) {
        const index = callIndex;
        callIndex += 1;
        yield chunk(
          {
            tool_calls: [
              {
                index,
                id: `call_${index}`,
                type: 'function',
                function: { name: typeof p.functionCall.name === 'string' ? p.functionCall.name : '', arguments: JSON.stringify(isRecord(p.functionCall.args) ? p.functionCall.args : {}) },
              },
            ],
          },
          null,
        );
      }
    }
    if (candidate.finishReason !== undefined && candidate.finishReason !== null) {
      stopReason = finishReasonOfGemini(candidate.finishReason);
    }
  }
  if (!finished) {
    finished = true;
    yield chunk({}, stopReason ?? 'stop');
    yield `data: ${JSON.stringify({ id: 'chatcmpl_gemini_stream', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta: {}, finish_reason: null }], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } })}\n\n`;
    yield 'data: [DONE]\n\n';
  }
}

/**
 * A Gemini streamGenerateContent SSE stream as the Anthropic event sequence.
 */
export async function* anthropicEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_gemini_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    out.push(
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    );
    return out;
  };

  for await (const frame of sseFrames(upstream)) {
    if (!isRecord(frame)) continue;
    if (isRecord(frame.usageMetadata)) {
      inputTokens = numberOrZero(frame.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(frame.usageMetadata.candidatesTokenCount);
    }
    const candidates = Array.isArray(frame.candidates) ? frame.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) continue;
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const p of parts) {
      if (!isRecord(p)) continue;
      if (typeof p.text === 'string' && p.text !== '') {
        yield* startEvents();
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          yield sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } });
        }
        yield sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: p.text } });
      } else if (isRecord(p.functionCall)) {
        yield* startEvents();
        if (textIndex !== -1) {
          yield sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex });
          textIndex = -1;
        }
        const index = nextIndex;
        nextIndex += 1;
        yield sseEvent('content_block_start', {
          type: 'content_block_start',
          index,
          content_block: {
            type: 'tool_use',
            id: `toolu_gemini_${index}`,
            name: typeof p.functionCall.name === 'string' ? p.functionCall.name : '',
            input: {},
          },
        });
        yield sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index,
          delta: { type: 'input_json_delta', partial_json: JSON.stringify(isRecord(p.functionCall.args) ? p.functionCall.args : {}) },
        });
        yield sseEvent('content_block_stop', { type: 'content_block_stop', index });
      }
    }
    if (candidate.finishReason !== undefined && candidate.finishReason !== null) {
      pendingStop = stopReasonOf(finishReasonOfGemini(candidate.finishReason));
    }
  }
  if (!finished) {
    yield* startEvents();
    yield* finishEvents(pendingStop ?? 'end_turn');
  }
}
