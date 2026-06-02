package anthropic

import (
	"context"
	"testing"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/joakimcarlsson/ai/message"
	"github.com/joakimcarlsson/ai/model"
	"github.com/joakimcarlsson/ai/tool"
)

// fakeTool is a minimal tool.BaseTool for exercising convertTools.
type fakeTool struct{}

func (fakeTool) Info() tool.Info { return tool.NewInfo("noop", "does nothing", struct{}{}) }
func (fakeTool) Run(context.Context, tool.Call) (tool.Response, error) {
	return tool.NewTextResponse("ok"), nil
}

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

// TestCacheTTL_PinsOnUserContentAndToolSites verifies the TTL is stamped on the
// other two cache_control emission sites (last user content block, last tool),
// not just the system block — guarding against a missed call site silently
// disabling caching on one path.
func TestCacheTTL_PinsOnUserContentAndToolSites(t *testing.T) {
	o := Options{model: model.Model{APIModel: "claude-sonnet-4-5", DefaultMaxTokens: 1024}}
	WithCacheTTL(CacheTTL1h)(&o)
	c := &Client{options: o}

	am, _ := c.convertMessages([]message.Message{
		message.NewSystemMessage("sys"),
		message.NewUserMessage("hello"),
	})
	if len(am) == 0 || len(am[len(am)-1].Content) == 0 {
		t.Fatalf("expected a user message with content")
	}
	userBlock := am[len(am)-1].Content[0].OfText
	if userBlock == nil {
		t.Fatalf("expected last user content to be a text block")
	}
	if userBlock.CacheControl.TTL != anthropicsdk.CacheControlEphemeralTTLTTL1h {
		t.Errorf("user content TTL = %q, want 1h", userBlock.CacheControl.TTL)
	}

	tools := c.convertTools([]tool.BaseTool{fakeTool{}})
	if len(tools) == 0 || tools[len(tools)-1].OfTool == nil {
		t.Fatalf("expected a tool param")
	}
	if got := tools[len(tools)-1].OfTool.CacheControl.TTL; got != anthropicsdk.CacheControlEphemeralTTLTTL1h {
		t.Errorf("tool TTL = %q, want 1h", got)
	}
}

// TestCacheTTL_FiveMinutes verifies the explicit 5m case maps to the SDK 5m
// constant (distinct from the empty default).
func TestCacheTTL_FiveMinutes(t *testing.T) {
	p := buildParams(t, WithCacheTTL(CacheTTL5m))
	if got := lastSystemBlock(t, p).CacheControl.TTL; got != anthropicsdk.CacheControlEphemeralTTLTTL5m {
		t.Errorf("system block TTL = %q, want 5m", got)
	}
}

// TestUsage_MapsPerTierCacheCreation verifies usage() carries the Anthropic
// 5m/1h ephemeral cache-write split (consumed by overtura's relay + cost
// accounting and asserted by the live-smoke cache-breakpoint test).
func TestUsage_MapsPerTierCacheCreation(t *testing.T) {
	c := &Client{}
	msg := anthropicsdk.Message{}
	msg.Usage.InputTokens = 100
	msg.Usage.OutputTokens = 50
	msg.Usage.CacheCreationInputTokens = 30
	msg.Usage.CacheReadInputTokens = 5
	msg.Usage.CacheCreation.Ephemeral5mInputTokens = 10
	msg.Usage.CacheCreation.Ephemeral1hInputTokens = 20

	u := c.usage(msg)
	if u.CacheCreation5mTokens != 10 {
		t.Errorf("CacheCreation5mTokens = %d, want 10", u.CacheCreation5mTokens)
	}
	if u.CacheCreation1hTokens != 20 {
		t.Errorf("CacheCreation1hTokens = %d, want 20", u.CacheCreation1hTokens)
	}
	if u.CacheCreationTokens != 30 {
		t.Errorf("CacheCreationTokens = %d, want 30 (total preserved)", u.CacheCreationTokens)
	}
	if u.CacheReadTokens != 5 {
		t.Errorf("CacheReadTokens = %d, want 5", u.CacheReadTokens)
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
