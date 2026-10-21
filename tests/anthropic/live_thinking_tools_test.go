//go:build live

package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/joakimcarlsson/ai/agent"
	"github.com/joakimcarlsson/ai/llm"
	llmanthropic "github.com/joakimcarlsson/ai/llm/anthropic"
	"github.com/joakimcarlsson/ai/message"
	"github.com/joakimcarlsson/ai/session"
	"github.com/joakimcarlsson/ai/tool"
	"github.com/joakimcarlsson/ai/types"
)

// Opt-in live probes for KTD21, run with:
//
//	ANTHROPIC_API_KEY=... (cd tests && go test -tags live -run TestLiveThinking -v ./anthropic)
//
// They skip without a key. Every probe drives agent.ChatStream, never
// hand-built messages. A request-recording transport sits under the SDK so the
// probes can assert thinking was really replayed and, for the two features the
// adapter has no option for, rewrite the thinking config:
//
//   - between_tools thinking (Sonnet 5.5): thinking = {"type":"between_tools"}
//   - conversation binding enforced (beta thinking-binding-controls-2026-08-01):
//     thinking.block_binding.prefix_mismatch_behavior = "error"
//
// Negative controls (run by hand, expected to fail):
//
//	U13_DROP_SIGNATURE=1  blanks the signature of replayed thinking blocks in
//	                      the transport: the tool-loop probes fail with a 400.
//	U13_CORRUPT_SIGNATURE=1  flips one character of each replayed signature, so
//	                      only signature verification can reject it: the
//	                      tool-loop probes fail with a 400.
//	U13_KEEP_THINKING=1   keeps thinking in the history loaded after the
//	                      compaction: the binding probe fails with a 400.
//
// Observed 2026-10-21 (claude-opus-5-5): with no control set, all four probes
// pass. U13_DROP_SIGNATURE: 400 "messages.1.content.0.thinking: each thinking
// block must contain thinking" (Opus omits the displayed text, so a blanked
// signature leaves an empty block). U13_CORRUPT_SIGNATURE: 400 "Invalid
// `signature` in `thinking` block". U13_KEEP_THINKING: 400 "Invalid
// `signature` in `thinking` block. The block is bound to a different
// conversation".

// countingWeather counts how many times the model called the tool.
type countingWeather struct {
	weatherTool
	calls atomic.Int32
}

func (c *countingWeather) Run(
	ctx context.Context,
	call tool.Call,
) (tool.Response, error) {
	c.calls.Add(1)
	return c.weatherTool.Run(ctx, call)
}

// liveTransport records request bodies and applies the probe's rewrites.
type liveTransport struct {
	thinking map[string]any // replaces the request's thinking config when set
	betas    string         // anthropic-beta header when set

	mu       sync.Mutex
	requests []map[string]any
}

func (l *liveTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.Body != nil && strings.HasSuffix(req.URL.Path, "/v1/messages") {
		raw, err := io.ReadAll(req.Body)
		if err != nil {
			return nil, err
		}
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			return nil, err
		}
		if l.thinking != nil {
			body["thinking"] = l.thinking
		}
		if os.Getenv("U13_DROP_SIGNATURE") != "" {
			mutateSignatures(body, func(string) string { return "" })
		}
		if os.Getenv("U13_CORRUPT_SIGNATURE") != "" {
			mutateSignatures(body, flipOneChar)
		}
		l.mu.Lock()
		l.requests = append(l.requests, body)
		l.mu.Unlock()
		raw, err = json.Marshal(body)
		if err != nil {
			return nil, err
		}
		req = req.Clone(req.Context())
		req.Body = io.NopCloser(bytes.NewReader(raw))
		req.ContentLength = int64(len(raw))
		req.Header.Del("Content-Length")
		if l.betas != "" {
			req.Header.Set("anthropic-beta", l.betas)
		}
	}
	return http.DefaultTransport.RoundTrip(req)
}

// mutateSignatures rewrites the signature of every thinking block the request
// replays.
func mutateSignatures(body map[string]any, f func(string) string) {
	msgs, _ := body["messages"].([]any)
	for _, m := range msgs {
		content, _ := m.(map[string]any)["content"].([]any)
		for _, b := range content {
			if bm, ok := b.(map[string]any); ok && bm["type"] == "thinking" {
				sig, _ := bm["signature"].(string)
				bm["signature"] = f(sig)
			}
		}
	}
}

// flipOneChar changes a single character in the middle of a signature, so the
// value stays well-formed and only signature verification can reject it.
func flipOneChar(sig string) string {
	if sig == "" {
		return sig
	}
	i := len(sig) / 2
	c := byte('A')
	if sig[i] == 'A' {
		c = 'B'
	}
	return sig[:i] + string(c) + sig[i+1:]
}

// replayedThinkingRequests counts recorded requests that carried at least one
// assistant thinking or redacted_thinking block.
func (l *liveTransport) replayedThinkingRequests() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	n := 0
	for _, body := range l.requests {
		msgs, _ := body["messages"].([]any)
		found := false
		for _, m := range msgs {
			mm, _ := m.(map[string]any)
			if mm["role"] != "assistant" {
				continue
			}
			content, _ := mm["content"].([]any)
			for _, b := range content {
				if bm, ok := b.(map[string]any); ok &&
					(bm["type"] == "thinking" ||
						bm["type"] == "redacted_thinking") {
					found = true
				}
			}
		}
		if found {
			n++
		}
	}
	return n
}

func liveThinkingClient(
	t *testing.T,
	model llm.Model,
	effort llmanthropic.ReasoningEffort,
	tr *liveTransport,
) llm.LLM {
	t.Helper()
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}
	return llmanthropic.NewLLM(
		llmanthropic.WithAPIKey(key),
		llmanthropic.WithModel(model),
		llmanthropic.WithMaxTokens(8192),
		llmanthropic.WithReasoningEffort(effort),
		llmanthropic.WithTimeout(3*time.Minute),
		llmanthropic.WithHTTPClient(&http.Client{Transport: tr}),
	)
}

// runLiveTurn runs one agent turn through ChatStream and returns the final
// text. Any error event fails the probe.
func runLiveTurn(
	t *testing.T,
	a *agent.Agent,
	prompt string,
) string {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	var final string
	for ev := range a.ChatStream(ctx, prompt) {
		switch ev.Type {
		case types.EventError:
			t.Fatalf("ChatStream error: %v", ev.Error)
		case types.EventComplete:
			if ev.Response != nil {
				final = ev.Response.Content
			}
		}
	}
	return final
}

// twoRoundTripPrompt asks for reasoning before each tool call. Adaptive
// thinking lets the model skip thinking on an easy request -- "use the tool
// for Paris" drew no thinking block from Opus, Sonnet or Haiku 5.5 on
// 2026-10-21 -- and a probe that never receives a thinking block proves
// nothing about replaying one.
const twoRoundTripPrompt = "Three friends live in Paris, London and Oslo. " +
	"The one who lives furthest north hates the cold; the one in the city " +
	"between the other two loves rain. Work out which city the rain-lover " +
	"lives in, then use get_weather for that city only. After the result, " +
	"call get_weather once more for the city of the friend who hates the " +
	"cold. Then tell me who should carry an umbrella."

func liveToolLoop(
	t *testing.T,
	model llm.Model,
	effort llmanthropic.ReasoningEffort,
	tr *liveTransport,
	wantThinking bool,
) {
	t.Helper()
	weather := &countingWeather{}
	a := agent.New(
		liveThinkingClient(t, model, effort, tr),
		agent.WithTools(weather),
		agent.WithSequentialToolExecution(),
	)
	final := runLiveTurn(t, a, twoRoundTripPrompt)
	if weather.calls.Load() < 2 {
		t.Fatalf("tool ran %d times, want 2 round trips (final=%q)",
			weather.calls.Load(), final)
	}
	if final == "" {
		t.Fatal("empty final answer")
	}
	if wantThinking && tr.replayedThinkingRequests() == 0 {
		t.Fatal("no request carried replayed thinking; the model did not think")
	}
	t.Logf("%s: tool calls=%d, requests=%d, requests with replayed thinking=%d",
		model.APIModel, weather.calls.Load(), len(tr.requests),
		tr.replayedThinkingRequests())
}

func TestLiveThinkingToolLoopOpus55(t *testing.T) {
	liveToolLoop(t, llmanthropic.Models[llmanthropic.ClaudeOpus55], llmanthropic.ReasoningEffortHigh, &liveTransport{}, true)
}

func TestLiveThinkingToolLoopHaiku55(t *testing.T) {
	liveToolLoop(t, llmanthropic.Models[llmanthropic.ClaudeHaiku55], llmanthropic.ReasoningEffortHigh, &liveTransport{}, true)
}

// TestLiveThinkingToolLoopSonnet55BetweenTools is the "thinking off" case:
// Sonnet 5.5 refuses thinking {type: disabled} and names between_tools as the
// way to turn it off, and in that mode it returned no thinking block at low,
// medium or high effort on 2026-10-21. What this proves is that the tool loop
// completes in that mode; there is no thinking to replay.
func TestLiveThinkingToolLoopSonnet55BetweenTools(t *testing.T) {
	liveToolLoop(t, llmanthropic.Models[llmanthropic.ClaudeSonnet55], llmanthropic.ReasoningEffortLow, &liveTransport{
		thinking: map[string]any{"type": "between_tools"},
	}, false)
}

// stripAnthropicThinking removes Anthropic thinking and redacted_thinking
// parts from a message, as Hearth does for history loaded for a later turn.
func stripAnthropicThinking(msg message.Message) message.Message {
	kept := msg.Parts[:0:0]
	for _, p := range msg.Parts {
		if rc, ok := p.(message.ReasoningContent); ok &&
			llmanthropic.IsAnthropicReasoning(rc) {
			continue
		}
		kept = append(kept, p)
	}
	msg.Parts = kept
	return msg
}

// TestLiveThinkingAfterCompactionWithBindingEnforced runs three tool turns,
// compacts the way Hearth does (a summary plus the last four messages kept
// verbatim, changing the prefix every retained thinking block was bound to),
// then runs a fourth turn with prefix_mismatch_behavior "error". With the
// thinking stripped from the retained messages the turn completes; with
// U13_KEEP_THINKING=1 it must fail with a binding 400.
func TestLiveThinkingAfterCompactionWithBindingEnforced(t *testing.T) {
	tr := &liveTransport{
		betas: "thinking-binding-controls-2026-08-01",
		thinking: map[string]any{
			"type": "adaptive",
			"block_binding": map[string]any{
				"prefix_mismatch_behavior": "error",
			},
		},
	}
	client := liveThinkingClient(t, llmanthropic.Models[llmanthropic.ClaudeOpus55], llmanthropic.ReasoningEffortHigh, tr)
	store := session.MemoryStore()
	newAgent := func() *agent.Agent {
		return agent.New(
			client,
			agent.WithTools(&countingWeather{}),
			agent.WithSession("u13-compaction", store),
			agent.WithSequentialToolExecution(),
		)
	}
	a := newAgent()
	for _, city := range []string{"Paris", "London", "Rome"} {
		runLiveTurn(t, a, "Use the get_weather tool for "+city+", then answer in one sentence.")
	}

	ctx := context.Background()
	sess, err := store.Load(ctx, "u13-compaction")
	if err != nil {
		t.Fatal(err)
	}
	history, err := sess.GetMessages(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(history) < 8 {
		t.Fatalf("history has %d messages, want at least 8", len(history))
	}
	keep := history[len(history)-4:]
	if keep[0].Role != message.User {
		t.Fatalf("last four messages start with role %q, want a turn boundary", keep[0].Role)
	}
	retainedThinking := 0
	compacted := []message.Message{
		message.NewSystemMessage("Summary of the earlier conversation: the user asked for the weather in Paris and London."),
	}
	for _, m := range keep {
		for _, p := range m.Parts {
			if rc, ok := p.(message.ReasoningContent); ok &&
				llmanthropic.IsAnthropicReasoning(rc) {
				retainedThinking++
			}
		}
		if os.Getenv("U13_KEEP_THINKING") == "" {
			m = stripAnthropicThinking(m)
		}
		compacted = append(compacted, m)
	}
	if retainedThinking == 0 {
		t.Fatal("the retained turn held no thinking; the probe would prove nothing")
	}
	if err := sess.Clear(ctx); err != nil {
		t.Fatal(err)
	}
	if err := sess.AddMessages(ctx, compacted); err != nil {
		t.Fatal(err)
	}

	final := runLiveTurn(t, newAgent(), "Now use the get_weather tool for Madrid and answer in one sentence.")
	if final == "" {
		t.Fatal("empty final answer after compaction")
	}
	t.Logf("completed a turn after compaction; retained thinking parts in the kept turn: %d", retainedThinking)
}
