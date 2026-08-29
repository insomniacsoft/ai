package realtime

import (
	"context"
	"os"
	"testing"
	"time"
)

// TestLiveDialWindow measures the window in which the caller is already
// streaming microphone audio and the provider cannot yet be written to:
// Start is asynchronous, so audio can arrive while the WebSocket is still
// being dialled, and AppendAudio on a client with no connection returns
// ErrNotConnected. A caller that swallows that error loses whatever was
// spoken during the window. This measures how long that window actually is
// against the real endpoint, and how long a write is dropped rather than
// gated on readiness: sendLive gates on the connection context, not on
// readiness. Skipped unless REALTIME_PROBE_DIAL is set.
func TestLiveDialWindow(t *testing.T) {
	if os.Getenv("REALTIME_PROBE_DIAL") == "" {
		t.Skip("REALTIME_PROBE_DIAL not set; skipping the dial-window probe")
	}
	key := os.Getenv("REALTIME_PROBE_API_KEY")
	if key == "" {
		t.Fatal(
			"REALTIME_PROBE_DIAL is set but REALTIME_PROBE_API_KEY is empty",
		)
	}
	model := os.Getenv("REALTIME_PROBE_MODEL")
	if model == "" {
		model = "gpt-realtime-2.1"
	}

	for i := range 3 {
		c, err := New(Config{
			APIKey: key,
			Session: SessionConfig{
				Model:        model,
				Instructions: "test",
				Eagerness:    EagernessLow,
			},
		}, WithLogger(discardLogger()))
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		start := time.Now()
		c.Start(ctx)

		var writable time.Duration
		for {
			if err := c.AppendAudio(make([]byte, 2)); err == nil {
				writable = time.Since(start)
				break
			}
			if time.Since(start) > 20*time.Second {
				t.Fatal("the socket never became writable")
			}
			time.Sleep(time.Millisecond)
		}
		if err := c.WaitReady(ctx); err != nil {
			t.Fatalf("WaitReady() error = %v", err)
		}
		ready := time.Since(start)

		t.Logf(
			"attempt %d: writable after %v (%.0f ms of speech dropped), configured after %v",
			i+1,
			writable.Round(time.Millisecond),
			float64(writable.Milliseconds()),
			ready.Round(time.Millisecond),
		)
		c.Close()
		cancel()
	}
}
