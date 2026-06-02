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

## U6 image re-home — reconciliation design (in progress)

Upstream `image` shares most types with the fork already (GenerationResponse/Result/
Usage, StreamEvent, EventPartialImage/Completed, ErrStreamingNotSupported, the
helpers). The fork adds: per-call GenerationOptions + option funcs, ErrEditNotSupported,
and an optioned `ImageGeneration` interface with EditImage/EditImageStreaming.

DONE (committed, additive — does NOT touch the existing optionless `Generation`
interface, so upstream `image/xai` is unaffected):
- `image/options.go`: GenerationOptions + all per-call With* funcs + ErrEditNotSupported
  + ApplyGenerationOptions.

REMAINING (the vendor re-home — the bulk):
1. `image/editing.go` (new): the optioned `ImageGeneration` interface
   (GenerateImage/GenerateImageStreaming/EditImage/EditImageStreaming with
   ...GenerationOption + Model()), the optional client interfaces
   (EditingImageGenerationClient etc.), and the generic base wrapper
   `baseImageGeneration[C]` that dispatches edit via type assertion + tracing
   (re-home from fork image_generation.go, 404 LOC). Keep upstream's `Generation`
   + WithTracing for xai compatibility.
2. `image/openai/openai.go`: re-home fork openai.go (462 LOC) — generate + edit via
   `Images.Edit` (/v1/images/edits) with namedImageReader multipart + image_edit.*
   streaming event discrimination. Implements the optioned interface.
3. `image/gemini/gemini.go`: re-home fork gemini.go + gemini_native.go (309 LOC) —
   native GenerateContent IMAGE generate+edit.
4. xai: decide — either adapt `image/xai` to the optioned interface, or have the
   consumer (U9) wrap it (grok-2-image is generate-only → EditImage returns
   ErrEditNotSupported). Lowest-risk: a thin consumer-side adapter.
5. Per-vendor construction options (WithAPIKey/WithModel/WithTimeout/WithOpenAIOptions/
   WithGeminiOptions/WithHTTPClient) move onto each vendor module.
Then U7 (image/openrouter, new module, from fork openrouter.go 264 LOC) + U8 (tag all,
ensure each module's go.sum is complete — note xai currently needs `go mod download`).
