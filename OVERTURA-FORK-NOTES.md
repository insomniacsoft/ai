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

## U8 — tag + go.work + GOPRIVATE (publish step)

State: `multimodule` branch is LOCAL ONLY (never pushed; 16 commits ahead of
upstream/main base af8b201). Existing `v0.18.5-overtura.N` tags are the MONOLITH
scheme the parallel sessions still consume — a different namespace from the
per-module tags below, so they do NOT collide (parallel sessions stay safe).

Per-module overtura tag scheme (`<module>/<upstream-base>-overtura.1`), 11 modules:
- model/v0.3.0-overtura.1
- message/v0.1.0-overtura.1
- llm/v0.2.0-overtura.1
- llm/anthropic/v0.2.2-overtura.1
- llm/openai/v0.3.2-overtura.1
- llm/gemini/v0.2.2-overtura.1
- agent/v0.2.1-overtura.1
- image/v0.1.0-overtura.1
- image/openai/v0.1.0-overtura.1
- image/gemini/v0.1.2-overtura.1
- image/openrouter/v0.1.0-overtura.1  (fork-only, no upstream base)

Publish steps (OUTWARD-FACING — gated on user authorization):
1. Require-pin rewrite: bump every FORKED sibling require to its -overtura.1
   version, at CONSISTENT versions (today they're inconsistent: model required at
   v0.1.0/v0.2.0/v0.3.0 across modules → all must become model/v0.3.0-overtura.1).
   KEEP the `replace => ../` directives (local dev resolves via replace; external
   consumers ignore replaces and use the pinned requires). Unforked siblings
   (tool/types/session/tracing) stay at their upstream versions.
2. Push `multimodule` branch to origin (github.com/insomniacsoft/ai).
3. Create + push the 11 per-module tags at the require-rewrite commit.
4. Verification: (1) local go.work `go list -m` resolves forked→local dirs;
   (2) GOWORK=off clean-checkout fetches each -overtura.1 tag via
   GOPRIVATE=github.com/insomniacsoft/* and cross-module requires resolve
   consistently. **Gate Phase 2 on pass (2).**
NOTE: pass (1) full overtura build + pass (2) are COUPLED TO U9 — overtura's
consumer still imports the monolith `github.com/joakimcarlsson/ai v0.18.5` and
`image_generation`/`llm.NewLLM`; it cannot build against the per-module fork until
U9 rewrites import sites. The plan acknowledges this ("resolution proven by U12's
clean build"). So U8's GOWORK=off proof is best run as a minimal synthetic consumer
OR folded into U9's first build.

## U10 impact (Phase 2) — "no model fork" is SUPERSEDED
Plan U10 says "no `model` fork (KTD-1)" and to consume upstream `model/v0.3.0`
unforked. The U7 decision A (fork `model` for OutputModalities + OpenRouter image
models) VOIDS that premise. When Phase 2 reaches U10: the catalog points at the
FORKED model (model/v0.3.0-overtura.1); the Anthropic cache-pricing cost-layer work
in U10 still applies, but the "unforked" framing does not. Also resolve the
`model.Gemini31FlashImage`→`Gemini31FlashImagePreview` rename (re-add alias to the
fork, or update the consumer) — the consumer's catalog_descriptors.go uses the old
name.

## U8 — DONE (published + verified)

Pushed `multimodule` branch + 11 per-module tags to origin
(github.com/insomniacsoft/ai). Final tags:
- agent/v0.2.1-overtura.2  (.1 was broken — see below — deleted; .2 is canonical)
- model/v0.3.0-overtura.1, message/v0.1.0-overtura.1, llm/v0.2.0-overtura.1,
  llm/anthropic/v0.2.2-overtura.1, llm/openai/v0.3.2-overtura.1,
  llm/gemini/v0.2.2-overtura.1, image/v0.1.0-overtura.1,
  image/openai/v0.1.0-overtura.1, image/gemini/v0.1.2-overtura.1,
  image/openrouter/v0.1.0-overtura.1

**Pass (2) GOWORK=off verification: GREEN.** Synthetic consumer fetching all 11
from published tags via GOPRIVATE, clean module cache, no go.work — `go mod tidy`
+ `go build ./...` both clean. Resolution confirmed: every fork module →
insomniacsoft tag; unforked deps (memory v0.2.0, session/tool/tracing v0.1.0) →
upstream joakimcarlsson. **Phase 2 gate satisfied.**

**Pass (2) caught a real broken pin** (the whole point of the GOWORK=off gate):
agent used `memory.Tools` (in memory/v0.2.0) but pinned `memory v0.1.0`; the local
`replace => ../memory` masked it (upstream main has the same mis-pin + relies on
the same workspace replace). Fixed: agent → memory/embeddings/schema v0.2.0.

### Consumer replace template (for U9 — overtura's go.mod)
Each forked joakimcarlsson path maps to the insomniacsoft repo at its tag; unforked
paths are NOT replaced (resolve from upstream). GOPRIVATE=github.com/insomniacsoft/*
on all dev/CI runners. Template (the 11 lines from /tmp/fork-verify3/go.mod):
`github.com/joakimcarlsson/ai/<mod> => github.com/insomniacsoft/ai/<mod> <tag>`.

### go.work deferred to U9 (deliberate)
The plan's "uncommitted go.work in the overtura worktree" was NOT created now: the
overtura worktree is SHARED with parallel sessions, and overtura still imports the
monolith `github.com/joakimcarlsson/ai v0.18.5` (not the per-module fork) until U9.
A live go.work there would change `go` resolution for parallel sessions for zero
pre-U9 benefit. Create it as the first step of U9, alongside the consumer import
rewrite, using `use ./server` + the 11 `../insomniacsoft-ai-mm/<mod>` dirs.

## U14 — per-module fork-exit classification (non-blocking)

Every forked module's delta classified so the rebase burden has a defined sunset.
**Opening the upstream PRs is an outward-facing action against the third-party
joakimcarlsson/ai repo — NOT yet done; awaiting authorization.**

| Module | Delta | Class | Exit path |
|---|---|---|---|
| llm/gemini | tool-result role `function`→`user`; ThoughtSignature capture/replay | **upstreamable correctness fix** | PR to upstream — both are plain bugs (Gemini 400s without them). Highest-value PR. |
| message | `ToolCall.ThoughtSignature []byte` field | **upstreamable** (enables the Gemini fix) | PR alongside the Gemini fix; fork exits when merged. |
| llm/openai | Responses prompt-cache-key / chaining / retention / service-tier / max-tool-calls options; turn-1 chain-trap Store invariant | **upstreamable feature + fix** | Propose options upstream; chain-trap is a fix. Larger PR; fork persists until accepted. |
| llm + agent | `Response.ProviderResponseID` threaded to `ChatResponse` | **upstreamable feature** | Propose; lowest-friction variant writes the ID into upstream's existing `ProviderMetadata` map, but the agent-layer threading is still needed. |
| llm/anthropic | cache-TTL pinning (1h), metadata.user_id, auto-last-block cache_control | **upstreamable feature** | Propose as options; fork persists until accepted. |
| model | `OutputModalities` field + OpenRouter image models + gpt-image-1/1-mini re-add + af8b201 Anthropic cache-read pricing fix | **mixed** | Pricing fix + OutputModalities are upstreamable; gpt-image-1/mini re-add is an overtura catalog choice (likely **permanent fork** — upstream deliberately dropped them). |
| image + image/openai + image/gemini | optioned `ImageGeneration` interface (per-call options + EditImage/EditImageStreaming) vs upstream's prompt-only `Generation` | **upstreamable feature (large)** | Biggest delta. Propose the edit/optioned surface upstream; until accepted this is the **main long-term fork cost**. |
| image/openrouter | fork-only module (upstream has no OpenRouter image vendor) | **new contribution OR permanent** | Offer upstream as a new vendor module, else permanent fork-only. |

Net: the cheapest wins (Gemini fixes + message field) retire ~2 modules' delta on
merge. The image edit/optioned surface is the durable cost. Rebase-on-upstream
(not merge) keeps each module's delta isolated for these PRs.

## Sync to upstream release-2026-06-04 (2026-06-09)

Rebased the fork onto `release-2026-06-04` (126 commits since the 2026-05-28 base).
All modules build + test green against the new release.

**Fork set shrank 11 → 9 modules.** Dropped (now upstreamed):
- `message` — upstream `message/v0.2.0` ships `ToolCall.ThoughtSignature` (our PR #181
  merged 2026-06-02). Consumed unforked.
- `llm/gemini` — PR #181 (tool-result role "user" + thought-signature replay) merged
  upstream; upstream `llm/gemini/v0.2.4` also adds base64URI parse + multimodel
  embeddings. Consumed unforked.

**Still forked (9), rebased onto new bases + new -overtura tags:**
- llm/v0.2.1-overtura.1, llm/anthropic/v0.2.3-overtura.1, llm/openai/v0.3.3-overtura.1,
  agent/v0.2.2-overtura.1, model/v0.3.0-overtura.3, image/v0.1.0-overtura.2,
  image/openai/v0.1.0-overtura.2, image/gemini/v0.1.2-overtura.2,
  image/openrouter/v0.1.0-overtura.2, batch/v0.1.1-overtura.1.

**Absorbed from upstream during the rebase (keep-both 3-way merges):**
- llm/anthropic: tool-choice support (#183) now coexists with our cache-TTL/metadata.
- Plus #173 param-handling fix, anthropic temperature/model-id updates, gemini parse
  fixes — all came in free with the new base.
- SDKs already matched (openai-go/v3 3.37, anthropic-sdk 1.46, genai 1.58) — no bump.

The obsolete cross-module pin commits (U8) were skipped and re-done against the new
versions. `image` fork remains the durable cost (upstream still has no EditImage).

## Upstream PR status (2026-06-09)

Merged to upstream main (await a release tag to consume):
- #181 — Gemini tool-role + thought-signature (already consumed; un-forked message + llm/gemini at the 2026-06-04 sync).
- #186 — ProviderResponseID on llm.Response + agent.ChatResponse. **When upstream
  next tags an `agent` release carrying this, `agent` UN-FORKS** (it's forked only
  for ProviderResponseID). `llm`/`llm/openai` also gain it but stay forked for their
  other patches (Responses options, per-tier TokenUsage).
- #187 — batch/gemini schema converter recursion (array items + nested objects).
  Our single-package batch converter fix becomes redundant with upstream's once
  aligned.

Next-sync action: drop `agent` from the fork set once a release tag includes #186.
