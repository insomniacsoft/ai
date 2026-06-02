# Overtura fork notes (multi-module layout)

This is the `insomniacsoft/ai` fork of `joakimcarlsson/ai`, re-established on the
upstream multi-module layout. Sync upstream by **rebase, not merge**. Tag each
forked module independently as `<module>/vX.Y.Z-overtura.N`.

## Locked fork set — 11 modules (U2 → 9; U5 added `message`; U7 added `model`)

`llm`, `agent`, `message`, `model`, `llm/anthropic`, `llm/openai`, `llm/gemini`,
`image`, `image/openai`, `image/gemini`, `image/openrouter` (fork-only).

`tool`, `types`, `session` are consumed **unforked** from upstream.

`message` was added to the fork set at U5 (user decision A, 2026-06-02): upstream's
new `message` module dropped `ToolCall.ThoughtSignature`, which the Gemini-3
thought-signature replay (a plan-listed correctness fix overtura relies on for
Gemini-3 + tools) requires. The earlier U2 "don't fork message" decision was
specifically about the unused `CacheBreakpoint` path (still dropped); ThoughtSignature
is a used correctness fix, so `message` is forked for it.

`model` was added to the fork set at U7 (user decision A, 2026-06-02). Upstream's
`model.ImageGenerationModel` has no `OutputModalities` field and upstream defines
no OpenRouter image-generation models at all. The fork-only `image/openrouter`
module needs per-model output modalities (image-only models must request
`["image"]`, not `["image","text"]`), and overtura's consumer reads
`e.ImageLibraryModel.OutputModalities` at compile time. Same precedent as `message`
(U5): fork the module when upstream drops a capability field overtura uses.
Re-homed: the `OutputModalities []string` field on `ImageGenerationModel` + the 6
`OpenRouterImageGenerationModels` entries (Flux2 Max/Pro/Klein/Flex, Riverflow V2
Pro/Fast) + their ID constants. NOTE the consumer (U9) also references
`model.Gemini31FlashImage` (renamed upstream → `Gemini31FlashImagePreview`); decide
at U9 whether to re-add the old alias to the forked model or update the consumer.

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

DONE — U6 complete (commit pending). image, image/openai, image/gemini build +
vet + test clean; all files < 400 LOC; image/xai (upstream Generation) untouched.

1. `image/editing.go` (new): the optioned `ImageGeneration` interface +
   `WithEditingTracing(inner, attrs)` wrapper. **DESIGN DECISION (multi-module
   adaptation):** the monolith expressed optional capabilities (edit, streaming)
   as interfaces with UNEXPORTED methods (generate/edit/...) dispatched by a
   central `baseImageGeneration[C]` via type assertion. That cannot survive the
   module split — unexported interface methods are implementable only inside the
   declaring package, so a vendor module could never satisfy them. The interface
   now declares all methods EXPORTED (mirroring upstream's own `Generation`);
   vendors return `ErrEditNotSupported`/`ErrStreamingNotSupported` for unsupported
   capabilities, and `SupportsEditing()` probes edit capability. Tracing moved
   into `WithEditingTracing` (analogous to upstream `WithTracing`) instead of the
   factory base. The central `NewImageGeneration(provider, ...)` factory is GONE
   (it would create an import cycle image → image/openai → image); provider
   selection moves to the consumer (U9), matching upstream's per-vendor
   `NewGeneration` philosophy.
2. `image/openai/{openai.go,edit.go}`: generate + edit via `Images.Edit`
   (/v1/images/edits) with namedImageReader multipart + image_edit.* streaming
   discrimination. Implements optioned `ImageGeneration`. **openai-go stays at
   v1.12.0** (the monolith proved the full edit surface — OfFileArray, edit
   streaming — on v1.12.0; no bump needed; llm/openai's v3 bump is independent).
   Construction Options: WithAPIKey/WithModel/WithTimeout/WithBaseURL/
   WithExtraHeaders/WithStreamingOptions. All image knobs are per-call options.
3. `image/gemini/{gemini.go,imagen.go,helpers.go}`: NewGeneration switches on
   `isNativeModel` → nativeClient (GenerateContent IMAGE, generate+edit) vs
   imagenClient (GenerateImages, generate-only, edit→ErrEditNotSupported).
   Construction Options: WithAPIKey/WithModel/WithTimeout/WithBackend.
   **DESIGN DECISION 1:** model ID `Gemini31FlashImage` → `Gemini31FlashImagePreview`
   (upstream `model`, consumed unforked, renamed it; forced).
   **DESIGN DECISION 2 (labeled divergence):** nativeClient.EditImageStreaming
   returns `ErrStreamingNotSupported` (editing IS supported, only its streaming
   variant isn't) — the monolith returned `ErrEditNotSupported` here as an
   artifact of its type-assertion dispatch. Behaviorally inert (no overtura
   Gemini streaming-edit caller); VERIFY the consumer doesn't switch on the
   specific sentinel at U9.
4. **xai RESOLVED (no new code):** the consumer (U9) builds xAI image via
   `image/openai.NewGeneration(WithBaseURL("https://api.x.ai/v1"))` — exactly how
   the monolith routed ProviderXAI through its OpenAIClient. Upstream's prompt-only
   `image/xai` module stays in the tree, unused by overtura. (SupportsEditing
   returns true for the openai-backed xAI client, matching the monolith; grok-2-image
   edit calls would error at the x.ai endpoint — same as before.)
5. Per-vendor construction options: DONE (see 2, 3).

Tests added (no network): image/options_overtura_test.go (defaults seeding +
override + empty-ref filtering), image/openai/edit_overtura_test.go (multipart
MIME detection/rejection, single-vs-multi union, ext map, gpt-image gating),
image/gemini/gemini_overtura_test.go (native routing, aspect-ratio map, config
omission, edit-capability sentinels).

Then U7 (image/openrouter, new module, from fork openrouter.go 264 LOC — note its
mapToAspectRatio/mapToImageSize/resolveModalities/resolveImageSize helpers; the
gemini copy of mapToAspectRatio now lives in image/gemini/helpers.go) + U8 (tag
all, ensure each module's go.sum is complete — image/openai + image/gemini already
tidied this unit; xai also tidied).
