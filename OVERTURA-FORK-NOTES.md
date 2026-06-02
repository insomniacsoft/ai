# Overtura fork notes (multi-module layout)

This is the `insomniacsoft/ai` fork of `joakimcarlsson/ai`, re-established on the
upstream multi-module layout. Sync upstream by **rebase, not merge**. Tag each
forked module independently as `<module>/vX.Y.Z-overtura.N`.

## Locked fork set — 10 modules (U2 → 9; U5 added `message`)

`llm`, `agent`, `message`, `llm/anthropic`, `llm/openai`, `llm/gemini`, `image`,
`image/openai`, `image/gemini`, `image/openrouter` (fork-only).

`model`, `tool`, `types`, `session` are consumed **unforked** from upstream.

`message` was added to the fork set at U5 (user decision A, 2026-06-02): upstream's
new `message` module dropped `ToolCall.ThoughtSignature`, which the Gemini-3
thought-signature replay (a plan-listed correctness fix overtura relies on for
Gemini-3 + tools) requires. The earlier U2 "don't fork message" decision was
specifically about the unused `CacheBreakpoint` path (still dropped); ThoughtSignature
is a used correctness fix, so `message` is forked for it.

## Per-theme re-home decisions

- **Anthropic (`llm/anthropic`):** re-home TTL pinning (`WithAnthropicCacheTTL`,
  `AnthropicCacheTTL1h`), `WithAnthropicMetadataUserID`, `WithAnthropicOptions`,
  `WithAnthropicReasoningEffort`, and the **auto-last-block** cache_control
  emission. **DROPPED per user decision (U2, Option B, 2026-06-02): the
  caller-controlled per-block `cache_control` path driven by
  `message.TextContent.CacheBreakpoint` / `NewSystemMessageWithCacheBreakpoint`.**
  Reason: that capability is not consumed by overtura today (constructor never
  called, field never set), and keeping it would force forking `message` (a 10th
  module). The auto-last-block fallback preserves current overtura behavior. If
  multi-breakpoint cache carving is wanted later, re-introduce `message.CacheBreakpoint`
  and fork `message` then.

- **OpenAI (`llm/openai` + `llm` + `agent`):** prompt_cache_key, PreviousResponseID,
  Store, chaining, PromptCacheRetention, all `WithOpenAI*` options; response ID via
  `llm.Response.ProviderResponseID` threaded through `agent.ChatResponse` (upstream
  drops `resp.ID`; its `ProviderMetadata` map doesn't reach the agent layer).

- **Gemini (`llm/gemini`):** tool-result role=`user`; `ThoughtSignature` capture/replay.

- **Image (`image`, `image/openai`, `image/gemini`, `image/openrouter`):** upstream
  `image` is minimal (`GenerateImage(prompt)` + streaming + `Model()`; NO per-call
  options, NO edit). Re-home the fork's full optioned + `EditImage`/
  `EditingImageGenerationClient` architecture as fork additions. OpenAI edit via
  `Images.Edit` (/v1/images/edits) with the `namedImageReader` multipart fix +
  `image_edit.*` streaming-event discrimination. `image/openrouter` is fork-only,
  generate + edit, `image_config.{aspect_ratio,image_size}` / `message.images[]`.

- **Models/pricing (consumer-side, NOT a fork):** `model` consumed unforked;
  Anthropic cache-read pricing handled in overtura's `ModelEntry.CostInputCached`
  (remap to the correct upstream field — see plan U10). Gemini-3.5-Flash model-add
  dropped (now upstream).
