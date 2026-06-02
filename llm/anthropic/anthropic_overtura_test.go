package anthropic

import (
	"testing"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/joakimcarlsson/ai/message"
	"github.com/joakimcarlsson/ai/model"
)

// buildParams applies the given options and returns the prepared request params
// for a single-system-block, single-user-message conversation with one tool.
func buildParams(t *testing.T, opts ...Option) anthropicsdk.MessageNewParams {
	t.Helper()
	o := Options{model: model.Model{APIModel: "claude-sonnet-4-5", DefaultMaxTokens: 1024}}
	for _, opt := range opts {
		opt(&o)
	}
	c := &Client{options: o}
	msgs := []message.Message{
		message.NewSystemMessage("a stable system preamble"),
		message.NewUserMessage("hello"),
	}
	am, sys := c.convertMessages(msgs)
	return c.preparedMessages(am, nil, sys)
}

// lastSystemBlock returns the final system block, where the auto cache breakpoint
// is emitted.
func lastSystemBlock(t *testing.T, p anthropicsdk.MessageNewParams) anthropicsdk.TextBlockParam {
	t.Helper()
	if len(p.System) == 0 {
		t.Fatalf("expected at least one system block, got none")
	}
	return p.System[len(p.System)-1]
}

// TestCacheTTL_PinsOneHourOnBreakpoints verifies WithCacheTTL("1h") stamps the
// 1h TTL on the auto-emitted system cache_control breakpoint.
func TestCacheTTL_PinsOneHourOnBreakpoints(t *testing.T) {
	p := buildParams(t, WithCacheTTL(CacheTTL1h))
	got := lastSystemBlock(t, p).CacheControl.TTL
	if got != anthropicsdk.CacheControlEphemeralTTLTTL1h {
		t.Errorf("system block TTL = %q, want %q", got, anthropicsdk.CacheControlEphemeralTTLTTL1h)
	}
}

// TestCacheTTL_UnsetUsesAPIDefault verifies that without WithCacheTTL the
// breakpoint carries no TTL (empty → SDK omits → API default 5m), matching
// upstream behavior. This is the no-regression guard.
func TestCacheTTL_UnsetUsesAPIDefault(t *testing.T) {
	p := buildParams(t)
	block := lastSystemBlock(t, p)
	if block.CacheControl.Type != "ephemeral" {
		t.Errorf("expected ephemeral cache_control on last system block, got Type=%q", block.CacheControl.Type)
	}
	if block.CacheControl.TTL != "" {
		t.Errorf("unset cacheTTL should leave TTL empty, got %q", block.CacheControl.TTL)
	}
}

// TestMetadataUserID_SetsRequestField verifies WithMetadataUserID populates
// MessageNewParams.Metadata.UserID.
func TestMetadataUserID_SetsRequestField(t *testing.T) {
	p := buildParams(t, WithMetadataUserID("salted-tenant-hash"))
	if p.Metadata.UserID != anthropicsdk.String("salted-tenant-hash") {
		t.Errorf("Metadata.UserID = %+v, want salted-tenant-hash", p.Metadata.UserID)
	}
}

// TestMetadataUserID_UnsetLeavesMetadataEmpty verifies the metadata field is not
// populated when the option is unset.
func TestMetadataUserID_UnsetLeavesMetadataEmpty(t *testing.T) {
	p := buildParams(t)
	// An unset option leaves Metadata.UserID as the zero (absent) Opt; .Or returns
	// the sentinel only when the field is absent, distinguishing it from a
	// present-but-empty value.
	if got := p.Metadata.UserID.Or("\x00absent"); got != "\x00absent" {
		t.Errorf("unset metadataUserID should leave Metadata.UserID absent, got present value %q", got)
	}
}

// TestDisableCache_SuppressesBreakpoints verifies WithDisableCache removes the
// cache_control breakpoint entirely (upstream behavior preserved).
func TestDisableCache_SuppressesBreakpoints(t *testing.T) {
	p := buildParams(t, WithDisableCache(), WithCacheTTL(CacheTTL1h))
	block := lastSystemBlock(t, p)
	if block.CacheControl.Type != "" {
		t.Errorf("WithDisableCache should suppress cache_control, got Type=%q", block.CacheControl.Type)
	}
}
