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
