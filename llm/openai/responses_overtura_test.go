package openai

import (
	"testing"

	"github.com/joakimcarlsson/ai/model"
	"github.com/openai/openai-go/v3/responses"
)

func prepResponses(opts ...ResponsesOption) responses.ResponseNewParams {
	o := ResponsesOptions{model: model.Model{APIModel: "gpt-5.4-mini"}}
	for _, opt := range opts {
		opt(&o)
	}
	c := &responsesClient{options: o}
	return c.preparedParams(nil, nil)
}

// TestResponses_StoreInvariant verifies Store flips on only when the caller
// opted into chaining (either trigger), never by accident.
func TestResponses_StoreInvariant(t *testing.T) {
	if p := prepResponses(); p.Store.Valid() {
		t.Errorf("no chaining opt-in: Store should be absent, got Valid()=true (%v)", p.Store.Or(false))
	}
	if p := prepResponses(WithResponsesChainingEnabled(true)); !p.Store.Valid() || !p.Store.Or(false) {
		t.Errorf("chaining enabled: Store should be true, got Valid()=%v value=%v", p.Store.Valid(), p.Store.Or(false))
	}
	if p := prepResponses(WithResponsesPreviousResponseID("resp_123")); !p.Store.Or(false) {
		t.Errorf("previous-id set: Store should be true")
	}
}

// TestResponses_Turn1ChainTrap verifies the chain-entry case: chaining enabled
// with NO previous id must Store=true yet must NOT send a PreviousResponseID it
// doesn't have (the turn-1 chain trap).
func TestResponses_Turn1ChainTrap(t *testing.T) {
	p := prepResponses(WithResponsesChainingEnabled(true))
	if !p.Store.Or(false) {
		t.Errorf("turn 1: Store should be true to enter the chain")
	}
	if p.PreviousResponseID.Valid() {
		t.Errorf("turn 1: PreviousResponseID must be absent, got %q", p.PreviousResponseID.Or(""))
	}
}

// TestResponses_Turn2Chaining verifies turn 2+ sends the previous id and stores.
func TestResponses_Turn2Chaining(t *testing.T) {
	p := prepResponses(WithResponsesPreviousResponseID("resp_abc"))
	if got := p.PreviousResponseID.Or(""); got != "resp_abc" {
		t.Errorf("PreviousResponseID = %q, want resp_abc", got)
	}
	if !p.Store.Or(false) {
		t.Errorf("turn 2: Store should be true")
	}
}

// TestResponses_PromptCacheParams verifies prompt_cache_key + retention wire through.
func TestResponses_PromptCacheParams(t *testing.T) {
	p := prepResponses(
		WithResponsesPromptCacheKey("tenant-42-stable-prefix"),
		WithResponsesPromptCacheRetention("24h"),
	)
	if got := p.PromptCacheKey.Or(""); got != "tenant-42-stable-prefix" {
		t.Errorf("PromptCacheKey = %q, want tenant-42-stable-prefix", got)
	}
	if p.PromptCacheRetention != responses.ResponseNewParamsPromptCacheRetention("24h") {
		t.Errorf("PromptCacheRetention = %q, want 24h", p.PromptCacheRetention)
	}
}

// TestResponses_UnsetPromptCacheParams verifies nothing is sent when unset.
func TestResponses_UnsetPromptCacheParams(t *testing.T) {
	p := prepResponses()
	if p.PromptCacheKey.Valid() {
		t.Errorf("unset prompt_cache_key should be absent, got %q", p.PromptCacheKey.Or(""))
	}
	if p.PromptCacheRetention != "" {
		t.Errorf("unset retention should be empty, got %q", p.PromptCacheRetention)
	}
}

// TestResponses_UsageSubtractsCachedAndCarriesReasoning verifies usage()
// subtracts cached input tokens (so they aren't billed at both rates) and
// carries ReasoningTokens — both consumed by overtura's cost layer.
func TestResponses_UsageSubtractsCachedAndCarriesReasoning(t *testing.T) {
	c := &responsesClient{}
	resp := &responses.Response{}
	resp.Usage.InputTokens = 100
	resp.Usage.OutputTokens = 40
	resp.Usage.InputTokensDetails.CachedTokens = 30
	resp.Usage.OutputTokensDetails.ReasoningTokens = 12

	u := c.usage(resp)
	if u.InputTokens != 70 {
		t.Errorf("InputTokens = %d, want 70 (100 total - 30 cached)", u.InputTokens)
	}
	if u.CacheReadTokens != 30 {
		t.Errorf("CacheReadTokens = %d, want 30", u.CacheReadTokens)
	}
	if u.ReasoningTokens != 12 {
		t.Errorf("ReasoningTokens = %d, want 12", u.ReasoningTokens)
	}
	if u.OutputTokens != 40 {
		t.Errorf("OutputTokens = %d, want 40", u.OutputTokens)
	}
}

// TestResponses_ServiceTierTruncationMaxToolCalls verifies the remaining knobs.
func TestResponses_ServiceTierTruncationMaxToolCalls(t *testing.T) {
	p := prepResponses(
		WithResponsesServiceTier("flex"),
		WithResponsesTruncation("auto"),
		WithResponsesMaxToolCalls(7),
		WithResponsesSafetyIdentifier("salted-hash"),
	)
	if p.ServiceTier != responses.ResponseNewParamsServiceTier("flex") {
		t.Errorf("ServiceTier = %q, want flex", p.ServiceTier)
	}
	if p.Truncation != responses.ResponseNewParamsTruncation("auto") {
		t.Errorf("Truncation = %q, want auto", p.Truncation)
	}
	if got := p.MaxToolCalls.Or(0); got != 7 {
		t.Errorf("MaxToolCalls = %d, want 7", got)
	}
	if got := p.SafetyIdentifier.Or(""); got != "salted-hash" {
		t.Errorf("SafetyIdentifier = %q, want salted-hash", got)
	}
}
