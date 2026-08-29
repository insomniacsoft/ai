package realtime

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/joakimcarlsson/ai/tool"
)

// TestLiveRomanianToolCall drives a real session and asserts the STRUCTURED
// CALL that comes back, not that a connection succeeded.
//
// This verification has to be a live one because everything cheap about this
// integration is already covered by the fake-transport tests above: what they
// cannot tell you is whether the payload this client builds is a payload the
// provider agrees with. The repo has a recorded lesson that a 200 proves
// nothing about what was sent, and the session surface is the highest-drift
// part of the whole integration — a renamed field is accepted in silence and
// simply does not take effect.
//
// Gated, and skipped rather than failed without the gate: it spends money and
// needs a key. A gate that is SET and wrong still fails, because a probe that
// passes because it never ran is the worst of both.
//
//	REALTIME_PROBE=1 REALTIME_PROBE_API_KEY=… \
//	go test -run TestLiveRomanianToolCall -v ./internal/realtime/
func TestLiveRomanianToolCall(t *testing.T) {
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
			Instructions: "Esti asistentul casei. Raspunde foarte scurt in romana. " +
				"Cand utilizatorul cere sa se stinga o lumina, apeleaza unealta light_off.",
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

	// The provider announces its own duration cap. Measured 2026-08-23 at
	// exactly 60 minutes; asserted loosely because an announced cap can
	// change without notice, so this only depends on the number being
	// present and sane.
	if ttl := c.SessionInfo().TimeToExpiry(time.Now()); ttl <= 0 || ttl > 24*time.Hour {
		t.Errorf("TimeToExpiry = %v, want a positive, plausible cap", ttl)
	} else {
		t.Logf("provider reports the session cap as %v", ttl.Round(time.Second))
	}

	if err := c.ReplayHistory([]HistoryItem{{Role: "user", Text: "stinge lampa din hol"}}); err != nil {
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
					t.Fatalf("the model answered without calling light_off; it said %q", ev.Transcript)
				}
				call := ev.Calls[0]
				t.Logf("call %s(%s) preamble=%q", call.Name, call.Arguments, call.Preamble)
				if call.Name != "light_off" {
					t.Errorf("tool = %q, want light_off", call.Name)
				}
				if call.CallID == "" {
					t.Error("the call carries no call id; a result could not be addressed to it")
				}
				var args struct {
					Name string `json:"name"`
				}
				if err := json.Unmarshal([]byte(call.Arguments), &args); err != nil {
					t.Fatalf("arguments %q are not JSON: %v", call.Arguments, err)
				}
				if args.Name == "" {
					t.Errorf("arguments = %q, want a name extracted from the Romanian utterance", call.Arguments)
				}
				return
			}
		case <-deadline:
			t.Fatal("no response within 45s")
		}
	}
}

func cmp(v, fallback string) string {
	if v == "" {
		return fallback
	}
	return v
}
