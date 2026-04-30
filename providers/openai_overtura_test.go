package llm

import (
	"testing"
)

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
