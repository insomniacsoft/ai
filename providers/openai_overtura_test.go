package llm

import (
	"encoding/json"
	"testing"

	"github.com/joakimcarlsson/ai/message"
	"github.com/joakimcarlsson/ai/model"
)

// makeMinimalResponsesClient returns a Responses-API-eligible client with
// just enough state to invoke prepareResponseParams. Mirrors the test
// helper at openai_responses_test.go but kept here to avoid a cross-file
// dependency on changes to that file's signature.
func makeMinimalResponsesClient(opts openaiOptions) *openaiClient {
	return &openaiClient{
		providerOptions: llmClientOptions{
			model: model.Model{APIModel: "gpt-5.4-mini", CanReason: true},
			maxTokens: 1024,
		},
		options:      opts,
		useResponses: true,
	}
}

// TestOpenAIResponsesNewOptionsRoundTrip verifies that the v0.18.5-overtura.5
// additions land in the openaiOptions struct as expected.
func TestOpenAIResponsesNewOptionsRoundTrip(t *testing.T) {
	opts := openaiOptions{}
	WithOpenAIPreviousResponseID("resp_abc123")(&opts)
	WithOpenAIPromptCacheKey("cache_xyz")(&opts)
	WithOpenAISafetyIdentifier("safety_def")(&opts)
	WithOpenAIServiceTier("flex")(&opts)
	WithOpenAIMaxToolCalls(50)(&opts)
	WithOpenAITruncation("auto")(&opts)
	WithOpenAIPromptCacheRetention("24h")(&opts)

	if opts.previousResponseID != "resp_abc123" {
		t.Errorf("previousResponseID = %q, want resp_abc123", opts.previousResponseID)
	}
	if opts.promptCacheKey != "cache_xyz" {
		t.Errorf("promptCacheKey = %q, want cache_xyz", opts.promptCacheKey)
	}
	if opts.safetyIdentifier != "safety_def" {
		t.Errorf("safetyIdentifier = %q, want safety_def", opts.safetyIdentifier)
	}
	if opts.serviceTier != "flex" {
		t.Errorf("serviceTier = %q, want flex", opts.serviceTier)
	}
	if opts.maxToolCalls != 50 {
		t.Errorf("maxToolCalls = %d, want 50", opts.maxToolCalls)
	}
	if opts.truncation != "auto" {
		t.Errorf("truncation = %q, want auto", opts.truncation)
	}
	if opts.promptCacheRetention != "24h" {
		t.Errorf("promptCacheRetention = %q, want 24h", opts.promptCacheRetention)
	}

	WithOpenAIChainingEnabled(true)(&opts)
	if !opts.chainingEnabled {
		t.Errorf("chainingEnabled = %v, want true", opts.chainingEnabled)
	}
}

// TestAnthropicCacheTTLRoundTrip verifies TTL option lands in the struct.
func TestAnthropicCacheTTLRoundTrip(t *testing.T) {
	opts := anthropicOptions{}
	WithAnthropicCacheTTL(AnthropicCacheTTL1h)(&opts)
	WithAnthropicMetadataUserID("hashed_user_abc")(&opts)

	if opts.cacheTTL != AnthropicCacheTTL1h {
		t.Errorf("cacheTTL = %q, want 1h", opts.cacheTTL)
	}
	if opts.metadataUserID != "hashed_user_abc" {
		t.Errorf("metadataUserID = %q, want hashed_user_abc", opts.metadataUserID)
	}
}

// TestPrepareResponseParams_StoreFalseWhenNoChain locks the invariant
// that calls without ANY chaining trigger (neither chainingEnabled nor
// previousResponseID) always have Store=false on the wire. Privacy
// gate: tenant data only persists at OpenAI when the caller explicitly
// opts in via either trigger.
func TestPrepareResponseParams_StoreFalseWhenNoChain(t *testing.T) {
	client := makeMinimalResponsesClient(openaiOptions{
		// No chaining trigger set.
		promptCacheKey:   "cache_xyz",
		safetyIdentifier: "user_hash",
	})

	params := client.prepareResponseParams([]message.Message{
		message.NewUserMessage("Hi"),
	}, nil)

	raw := mustMarshalAndParse(t, params)
	if raw["store"] != false {
		t.Fatalf("store = %v, want false (no chaining trigger → no tenant data at OpenAI)", raw["store"])
	}
	if _, present := raw["previous_response_id"]; present {
		t.Errorf("previous_response_id should be omitted when not chaining, got %v", raw["previous_response_id"])
	}
}

// TestPrepareResponseParams_StoreTrueOnChainingEnabledTurn1 covers the
// chicken-and-egg case: turn 1 of a chain has no PreviousResponseID
// but MUST set Store=true so the response_id OpenAI returns can be
// resolved on turn 2. WithOpenAIChainingEnabled(true) is the trigger.
func TestPrepareResponseParams_StoreTrueOnChainingEnabledTurn1(t *testing.T) {
	client := makeMinimalResponsesClient(openaiOptions{
		chainingEnabled: true,
		// previousResponseID intentionally empty (turn 1).
	})

	params := client.prepareResponseParams([]message.Message{
		message.NewUserMessage("Hi"),
	}, nil)

	raw := mustMarshalAndParse(t, params)
	if raw["store"] != true {
		t.Errorf("store = %v, want true (chainingEnabled=true must Store=true even on turn 1)", raw["store"])
	}
	if _, present := raw["previous_response_id"]; present {
		t.Errorf("previous_response_id should be omitted on turn 1, got %v", raw["previous_response_id"])
	}
}

// TestPrepareResponseParams_StoreTrueWhenChainSet verifies the same
// invariant from the other direction: Store=true is only emitted in the
// branch that emits PreviousResponseID. This is the contract Overtura's
// docs/reference/prompt-caching.md cites; if a future refactor flips
// these independently, that invariant breaks silently.
func TestPrepareResponseParams_StoreTrueWhenChainSet(t *testing.T) {
	client := makeMinimalResponsesClient(openaiOptions{
		previousResponseID: "resp_abc123",
	})

	params := client.prepareResponseParams([]message.Message{
		message.NewUserMessage("Hi"),
	}, nil)

	raw := mustMarshalAndParse(t, params)
	if raw["store"] != true {
		t.Errorf("store = %v, want true (chaining call must allow OpenAI to retain the response for next-turn lookup)", raw["store"])
	}
	if raw["previous_response_id"] != "resp_abc123" {
		t.Errorf("previous_response_id = %v, want resp_abc123", raw["previous_response_id"])
	}
}

// TestPrepareResponseParams_StoreTracksChainAcrossOptions verifies that
// Store=true requires an explicit chaining trigger (chainingEnabled OR
// previousResponseID) — every other Responses API knob (cache key,
// safety identifier, service tier, truncation) must NOT independently
// flip Store. A future refactor that lifts Store=true out of the
// chaining gate would silently retain tenant data on every call that
// happens to set any other option.
func TestPrepareResponseParams_StoreTracksChainAcrossOptions(t *testing.T) {
	tests := []struct {
		name      string
		opts      openaiOptions
		wantStore bool
	}{
		{
			name: "all optional knobs without chain — store stays false",
			opts: openaiOptions{
				promptCacheKey:   "cache_xyz",
				safetyIdentifier: "user_hash",
				serviceTier:      "default",
				truncation:       "auto",
			},
			wantStore: false,
		},
		{
			name: "all optional knobs WITH chainingEnabled — store flips to true",
			opts: openaiOptions{
				chainingEnabled:  true,
				promptCacheKey:   "cache_xyz",
				safetyIdentifier: "user_hash",
				serviceTier:      "default",
				truncation:       "auto",
			},
			wantStore: true,
		},
		{
			name: "all optional knobs WITH previousResponseID — store flips to true",
			opts: openaiOptions{
				previousResponseID: "resp_abc",
				promptCacheKey:     "cache_xyz",
				safetyIdentifier:   "user_hash",
				serviceTier:        "default",
				truncation:         "auto",
			},
			wantStore: true,
		},
		{
			name: "chainingEnabled + previousResponseID — store true, both wire-emitted",
			opts: openaiOptions{
				chainingEnabled:    true,
				previousResponseID: "resp_abc",
			},
			wantStore: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := makeMinimalResponsesClient(tt.opts)
			params := client.prepareResponseParams([]message.Message{
				message.NewUserMessage("Hi"),
			}, nil)
			raw := mustMarshalAndParse(t, params)
			if raw["store"] != tt.wantStore {
				t.Errorf("store = %v, want %v", raw["store"], tt.wantStore)
			}
		})
	}
}

func mustMarshalAndParse(t *testing.T, v any) map[string]any {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return raw
}

// TestAnthropicCacheTTLValueMapping verifies the helper returns the correct SDK constant.
func TestAnthropicCacheTTLValueMapping(t *testing.T) {
	tests := []struct {
		name    string
		ttl     AnthropicCacheTTL
		wantStr string
	}{
		{"default", "", ""},
		{"5m", AnthropicCacheTTL5m, "5m"},
		{"1h", AnthropicCacheTTL1h, "1h"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := &anthropicClient{options: anthropicOptions{cacheTTL: tt.ttl}}
			got := a.cacheTTLValue()
			if string(got) != tt.wantStr {
				t.Errorf("cacheTTLValue() = %q, want %q", string(got), tt.wantStr)
			}
		})
	}
}
