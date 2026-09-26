package live

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"
)

// TestLiveSession drives one real GPT-Live session and asserts the whole
// startup/shutdown lifecycle against the live endpoint, not a fake one.
//
// Everything cheap about this integration is covered by client_test.go's
// fake-server tests; what they cannot tell you is whether the payload this
// client builds is one the provider actually accepts. Skipped rather than
// failed when the gate is unset, but a gate that IS set and wrong still
// fails — a probe that passes because it never ran is the worst of both.
//
//	AI_LIVE_TEST=1 OPENAI_API_KEY=sk-... go test -run TestLiveSession -v ./live/
func TestLiveSession(t *testing.T) {
	if os.Getenv("AI_LIVE_TEST") == "" {
		t.Skip("AI_LIVE_TEST not set; skipping the live provider probe")
	}
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		t.Fatal("AI_LIVE_TEST is set but OPENAI_API_KEY is empty")
	}

	c, err := New(Config{
		APIKey: key,
		Session: SessionConfig{
			Model:        "gpt-live-1",
			Instructions: "Say only: ready.",
		},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	c.Start(ctx)

	if err := c.WaitReady(ctx); err != nil {
		t.Fatalf("WaitReady() error = %v", err)
	}
	if c.SessionID() == "" {
		t.Error("SessionID() is empty after WaitReady succeeded")
	}

	closeCtx, closeCancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer closeCancel()
	closed, err := c.Close(closeCtx)
	if err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if closed.Reason == "" {
		t.Error("Close() returned a Closed with no Reason; want session.closed's own reason")
	}
	t.Logf("session %s closed after %.1fs (%s)", c.SessionID(), closed.Seconds, closed.Reason)
}

// TestLiveSessionRejectsBadKey confirms a rejected key surfaces as
// ErrUnauthorized rather than a generic error a caller cannot match on.
//
//	AI_LIVE_TEST=1 go test -run TestLiveSessionRejectsBadKey -v ./live/
func TestLiveSessionRejectsBadKey(t *testing.T) {
	if os.Getenv("AI_LIVE_TEST") == "" {
		t.Skip("AI_LIVE_TEST not set; skipping the live provider probe")
	}

	c, err := New(Config{
		APIKey: "sk-invalid-deliberately-wrong-key",
		Session: SessionConfig{
			Model:        "gpt-live-1",
			Instructions: "Say only: ready.",
		},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	c.Start(ctx)

	err = c.WaitReady(ctx)
	if err == nil {
		t.Fatal("WaitReady() error = nil, want ErrUnauthorized for a deliberately invalid key")
	}
	if !errors.Is(err, ErrUnauthorized) {
		t.Errorf("WaitReady() error = %v, want it to match ErrUnauthorized", err)
	}
}
