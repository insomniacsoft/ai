package realtime

// Measures the window in which the downstream consumer is already streaming
// and the provider cannot yet be written to.
//
// The engine starts the provider and tells the device to begin its run in
// consecutive statements, with no wait between them. Start is asynchronous, so
// microphone audio arrives while the WebSocket is still being dialled — and
// AppendAudio on a client with no connection returns ErrNotConnected, and a
// caller that swallows it loses whatever was spoken during that window — which
// is always the beginning of the sentence.
//
// This measures how long that window actually is against the real endpoint,
// because the size of the loss is the whole question: 20 ms is a non-issue and
// 400 ms is the first word.
//
// Skipped unless REALTIME_PROBE_DIAL is set.

import (
	"context"
	"os"
	"testing"
	"time"
)

func TestLiveDialWindow(t *testing.T) {
	if os.Getenv("REALTIME_PROBE_DIAL") == "" {
		t.Skip("REALTIME_PROBE_DIAL not set; skipping the dial-window probe")
	}
	key := os.Getenv("REALTIME_PROBE_API_KEY")
	if key == "" {
		t.Fatal("REALTIME_PROBE_DIAL is set but REALTIME_PROBE_API_KEY is empty")
	}
	model := os.Getenv("REALTIME_PROBE_MODEL")
	if model == "" {
		model = "gpt-realtime-2.1"
	}

	for i := 0; i < 3; i++ {
		c, err := New(Config{
			APIKey:  key,
			Session: SessionConfig{Model: model, Instructions: "test", Eagerness: EagernessLow},
		}, WithLogger(discardLogger()))
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		start := time.Now()
		c.Start(ctx)

		// The moment a write would stop being dropped: sendLive gates on the
		// connection context, not on readiness.
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

		// 16 kHz mono s16 at the downstream consumer: 32000 bytes a second.
		t.Logf("attempt %d: writable after %v (%.0f ms of speech dropped), configured after %v",
			i+1, writable.Round(time.Millisecond), float64(writable.Milliseconds()), ready.Round(time.Millisecond))
		c.Close()
		cancel()
	}
}
