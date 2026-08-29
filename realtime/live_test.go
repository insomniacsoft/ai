package realtime

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/joakimcarlsson/ai/tool"
)

// TestLiveToolCall drives a real session and asserts the structured call
// that comes back, not merely that a connection succeeded: a 200 on connect
// proves nothing about whether the provider accepted the session payload.
// The provider's own session duration cap is asserted loosely, since an
// announced cap can change without notice; only that the number is present
// and sane.
//
// Gated and skipped rather than failed without the gate, since it spends
// money and needs a key; a gate that is set but wrong still fails.
//
//	REALTIME_PROBE=1 REALTIME_PROBE_API_KEY=… \
//	go test -run TestLiveToolCall -v ./realtime/
func TestLiveToolCall(t *testing.T) {
	if os.Getenv("REALTIME_PROBE") == "" {
		t.Skip("REALTIME_PROBE not set; skipping the live provider probe")
	}
	key := os.Getenv("REALTIME_PROBE_API_KEY")
	if key == "" {
		t.Fatal("REALTIME_PROBE is set but REALTIME_PROBE_API_KEY is empty")
	}

	c, err := New(Config{
		APIKey: key,
		Session: SessionConfig{
			Model: cmp(os.Getenv("REALTIME_PROBE_MODEL"), "gpt-realtime"),
			Instructions: "You are a voice assistant. Answer very briefly. " +
				"When the user asks to turn off a light, call the light_off tool.",
			Eagerness: EagernessLow,
			Tools:     []tool.BaseTool{lightTool()},
		},
	}, WithLogger(discardLogger()))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	defer c.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	c.Start(ctx)
	if err := c.WaitReady(ctx); err != nil {
		t.Fatalf("WaitReady() error = %v", err)
	}

	if ttl := c.SessionInfo().
		TimeToExpiry(time.Now()); ttl <= 0 ||
		ttl > 24*time.Hour {
		t.Errorf("TimeToExpiry = %v, want a positive, plausible cap", ttl)
	} else {
		t.Logf("provider reports the session cap as %v", ttl.Round(time.Second))
	}

	if err := c.ReplayHistory(
		[]HistoryItem{{Role: "user", Text: "turn off the lamp"}},
	); err != nil {
		t.Fatalf("sending the utterance: %v", err)
	}
	if err := c.CreateResponse(); err != nil {
		t.Fatalf("CreateResponse() error = %v", err)
	}

	deadline := time.After(45 * time.Second)
	for {
		select {
		case ev, ok := <-c.Events():
			if !ok {
				t.Fatal("event channel closed before a response arrived")
			}
			switch ev.Kind {
			case EventError:
				t.Fatalf("provider error: %v", ev.Err)
			case EventResponseDone:
				if len(ev.Calls) == 0 {
					t.Fatalf(
						"the model answered without calling light_off; it said %q",
						ev.Transcript,
					)
				}
				call := ev.Calls[0]
				t.Logf(
					"call %s(%s) preamble=%q",
					call.Name,
					call.Arguments,
					call.Preamble,
				)
				if call.Name != "light_off" {
					t.Errorf("tool = %q, want light_off", call.Name)
				}
				if call.CallID == "" {
					t.Error(
						"the call carries no call id; a result could not be addressed to it",
					)
				}
				var args struct {
					Name string `json:"name"`
				}
				if err := json.Unmarshal(
					[]byte(call.Arguments),
					&args,
				); err != nil {
					t.Fatalf(
						"arguments %q are not JSON: %v",
						call.Arguments,
						err,
					)
				}
				if args.Name == "" {
					t.Errorf(
						"arguments = %q, want a name extracted from the utterance",
						call.Arguments,
					)
				}
				return
			}
		case <-deadline:
			t.Fatal("no response within 45s")
		}
	}
}

// cmp returns v, or fallback when v is empty.
func cmp(v, fallback string) string {
	if v == "" {
		return fallback
	}
	return v
}
