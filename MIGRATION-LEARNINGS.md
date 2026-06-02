# Multi-module migration — running learnings

Co-located with the fork work; distilled into overtura `docs/solutions/` at shipping.

## U1 — re-fork setup
- The blast radius of re-forking is the **fork checkout**, not the overtura repo:
  overtura's `go.mod` local-path `replace` points at the working tree, so resetting
  it breaks every parallel session. Mitigation: do multi-module work in a **separate
  fork worktree** (`/export/work/insomniacsoft-ai-mm`), leave the monolith checkout
  on `main` until the go.mod cutover (U12).
- Preserve before mutate: push unpushed fork commits + branch `overtura-monolith-archive`
  (the v0.18.5 + patches) so the patch inventory is recoverable for re-homing.
- Verified upstream: 15 needed modules present, pure workspace (no root go.mod),
  all cited tags exist, `image/openrouter` absent (fork-only premise holds).

## U2 — audit gate
- **A patch "theme" can span more modules than the obvious one.** The Anthropic
  cache theme touched `message` (CacheBreakpoint hook), `llm` (TokenUsage), and
  `llm/anthropic` (provider). The fork-set count is wrong until you trace every
  consumer-read field/symbol back to its owning module.
- Upstream `image` is **minimal** (`GenerateImage(prompt)` + streaming + `Model()`):
  NO per-call options, NO edit. The fork's optioned + EditImage architecture is a
  wholesale re-home, not an "adapt to typed enums" port.

## U3 — Anthropic re-home (decision B: 9 modules)
- Per user decision, the caller-controlled per-block `cache_control` path
  (`message.CacheBreakpoint`) was **dropped** (unused by overtura; would force
  forking `message`). Kept: cache-TTL pinning, `Metadata.UserID`, reasoning-effort
  (already upstream), auto-last-block fallback.
- **Don't silently drop per-tier billing fields.** Code review surfaced that the
  archived patch also carried `TokenUsage.CacheCreation5m/1hTokens`, which overtura
  *does* consume (`relay/producer.go`, live-smoke assertion). Re-homed into the
  forked `llm.TokenUsage` (we fork `llm` anyway for U4) + anthropic `usage()`. The
  1h tier is billed higher than 5m, so collapsing the split corrupts cost accounting.
- SDK port confirmed zero-risk: anthropic-sdk-go v1.36→v1.46 left every patched
  symbol unchanged (`CacheControlEphemeralTTL{,TTL1h,TTL5m}`, `MetadataParam.UserID`,
  `CacheCreation.Ephemeral{5m,1h}InputTokens`).
- Test technique: white-box (`package anthropic`) tests construct `&Client{options:…}`
  and assert on `preparedMessages`/`convertTools`/`usage` wire output. A fake
  `tool.BaseTool` must pass a real (empty) params struct to `tool.NewInfo`, not nil.

## U4 — OpenAI Responses + response-ID (llm/openai + llm + agent)
- The new module splits chat (`NewLLM`) from Responses (`NewResponsesLLM`); the
  Responses-only options (prompt-cache, chaining, response-id) live on
  `ResponsesOption`. The old monolith's /v1/responses-vs-chat routing pre-existed
  at the v0.18.5 base and is now overtura's construction choice (U9), not a patch.
- Store/chaining invariant + turn-1 chain-trap re-homed verbatim
  (`chainingEnabled || previousResponseID != ""` → Store; never auto-on).
- ProviderResponseID threaded llm.Response → agent.ChatResponse (chat.go + stream.go,
  nil-guarded). task_manager.go's degenerate ChatResponse legitimately omits it.
- **usage()/accounting is a repeat silent-parity trap** (like U3): review caught
  that `responsesClient.usage()` dropped `ReasoningTokens` and didn't subtract
  cached input (double-billing). The chat client already did both. Always diff
  every provider's `usage()` against the archive. Fixed in U4.
- PromptCacheRetention was a no-op on the old SDK; now actually wired (v3.37 field).

## U5 — Gemini fixes (llm/gemini + message) — fork set → 10
- Tool-result role "function"→"user" (documented 400 in overtura's setup-agent).
- ThoughtSignature: upstream `message.ToolCall` DROPPED the field; re-added it to the
  forked `message` (user decision A → message is now forked, 10 modules). Gemini
  captures it from function-call response parts (3 sites; the archive's 4 sites
  consolidate 4→3 because upstream merged the two streaming paths into streamInternal)
  and replays it on the outgoing genai.Part (NOT inside FunctionCall — the field is
  on genai.Part at v1.58). Without replay Gemini 3 rejects the follow-up turn.
- **Capture-direction unit test deliberately omitted.** The parse is inline in
  SendMessages' retry closure (no seam), and the 3 sites use different ID schemes so
  a shared helper would change behavior / add rebase surface. Field-drop is
  build-guarded; per-site wiring is covered by U13's live Gemini+tools test. Chose
  not to refactor upstream structure for a unit test (fork-maintenance discipline).

## U6 — Image generate+edit (image + image/openai + image/gemini)
- **Unexported optional-interface methods do NOT survive a module split.** The
  monolith expressed optional capabilities (edit/streaming) as interfaces with
  UNEXPORTED methods (`generate`/`edit`/...) that a central `baseImageGeneration[C]`
  dispatched via type assertion. Go only lets the DECLARING package implement an
  unexported interface method, so a vendor module can never satisfy it. The
  multi-module equivalent: declare ALL interface methods EXPORTED (this is exactly
  why upstream's own prompt-only `Generation` uses exported methods), and return
  sentinels (`ErrEditNotSupported`/`ErrStreamingNotSupported`) + a `SupportsEditing()`
  probe for unsupported capabilities. Tracing moves from the factory base into a
  `WithEditingTracing(inner, attrs)` wrapper (mirrors upstream `WithTracing`).
- **The central provider factory cannot live in the shared package** — `image`
  importing `image/openai`+`image/gemini` (which import `image`) is a cycle. The
  monolith's `NewImageGeneration(provider, ...)` is dropped; provider selection
  moves to the consumer (U9), matching upstream's per-vendor `NewGeneration`.
- **Same method name + different signatures ⇒ one type can't satisfy both
  interfaces.** `Generation.GenerateImage(ctx,prompt)` vs
  `ImageGeneration.GenerateImage(ctx,prompt,...opt)` collide, so the fork's
  openai/gemini implement ONLY the optioned `ImageGeneration`; `image/xai` keeps
  prompt-only `Generation`. No merge possible.
- **Verify the SDK version against the monolith before assuming a bump.** openai-go
  stayed at v1.12.0 — the monolith proved the full edit surface (OfFileArray,
  edit streaming, image_edit.* events) on v1.12.0. llm/openai's v3 bump is a
  SEPARATE module's concern; image/openai didn't need it.
- **Unforked `model` constants can be renamed upstream.** `Gemini31FlashImage` →
  `Gemini31FlashImagePreview` (alias `NanoBanana2`). Grep the model module for the
  real constant name; don't trust the monolith's name for unforked deps.
- **go.sum gaps after re-home**: importing the wider edit API pulled new transitive
  deps (tidwall/sjson/gjson) not in upstream's narrower go.sum. `go mod tidy` per
  module after re-home (U8 will re-verify all).
- Review (general-purpose Go reviewer, archive-diff): ZERO undocumented divergences
  across all 8 paths — the cleanest unit. The exported-interface + sentinel design
  was the load-bearing decision; once made, the per-method re-home was mechanical.
- xai needs NO code: consumer builds xAI via `image/openai.NewGeneration(WithBaseURL(
  "https://api.x.ai/v1"))`, exactly as the monolith routed ProviderXAI through its
  OpenAIClient. Upstream `image/xai` stays unused.

## U7 — image/openrouter (fork-only) + model fork → 11 modules
- **Upstream can lack an ENTIRE model family, not just a field.** Upstream's `model`
  module defines zero OpenRouter image-gen models. Re-homing meant porting the whole
  `OpenRouterImageGenerationModels` map (6 entries) + 6 ID constants, not just adding
  the `OutputModalities` field. Grep upstream for the map name before assuming a
  field-only fork.
- **The CONSUMER's compile-time dependency is what forces the fork.** The library
  `image/openrouter` could have used caller-only modalities, but overtura's
  `ModelEntry.OutputModalities()` reads `e.ImageLibraryModel.OutputModalities` off
  the library type — so the field MUST exist on `model.ImageGenerationModel`. When
  deciding fork-vs-consumer-side for a missing field, check what the consumer reads,
  not just what the library needs. This was a STOP-and-ask fork-set decision (user
  chose A: fork model) — same class as U2/U5, not something to improvise.
- **Forking a data module is mechanically cheap** (additive field + verbatim entries;
  `replace => ../../model` already points local; no consumer churn since the consumer
  already reads the field). The cost is fork-set size + rebase surface, not code.
- **`image/openrouter` is fork-only and SDK-free** — raw net/http + encoding/json
  (OpenRouter's modalities/image_config extensions aren't in the OpenAI SDK types).
  go.mod requires only image + model (OTEL deps come transitively via image/tracing).
- **Re-home was character-for-character** (verbatim port of the monolith openrouter.go
  + the 6 model entries); the archive-diff reviewer found ZERO divergences. The only
  adaptations were the U6-established ones (image.-prefixed types, exported interface,
  ApplyGenerationOptionsWithModelDefaults rename, package-scope maxResponseSize const).
- **U9 landmine flagged**: the consumer references `model.Gemini31FlashImage` (renamed
  upstream → `Gemini31FlashImagePreview`, alias `NanoBanana2`). Now that `model` is
  forked, U9 can either re-add the old alias to the fork OR update the consumer — decide
  at U9. Either way the consumer won't compile against the bare upstream name.
