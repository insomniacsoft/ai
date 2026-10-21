package anthropic

// The bodies below are hand-built fixtures in the Messages API wire shape
// (thinking with signature, redacted_thinking with data, text, tool_use; and
// the SSE form with thinking_delta, signature_delta and input_json_delta). They
// are not captured from the live API; signatures and data are opaque
// placeholders, which is all the replay path treats them as.

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/joakimcarlsson/ai/agent"
	"github.com/joakimcarlsson/ai/llm"
	llmanthropic "github.com/joakimcarlsson/ai/llm/anthropic"
	"github.com/joakimcarlsson/ai/tool"
	"github.com/joakimcarlsson/ai/types"
)

// weatherTool is a one-argument tool for the agent-driven tests.
type weatherTool struct{}

func (weatherTool) Info() tool.Info {
	return tool.Info{
		Name:        "get_weather",
		Description: "Get the current weather for a city.",
		Parameters: map[string]any{
			"city": map[string]any{"type": "string", "description": "City"},
		},
		Required: []string{"city"},
	}
}

func (weatherTool) Run(context.Context, tool.Call) (tool.Response, error) {
	return tool.Response{Content: "sunny, 20C"}, nil
}

// rewriteTransport redirects Anthropic SDK requests to the test server.
type rewriteTransport struct{ host string }

func (r rewriteTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req = req.Clone(req.Context())
	req.URL.Scheme = "http"
	req.URL.Host = r.host
	return http.DefaultTransport.RoundTrip(req)
}

// block is one content block of a scripted response.
type block struct {
	kind      string // thinking, redacted, text, tool
	text      string // thinking text, or text
	signature string
	data      string
	id        string // tool_use id
}

func (b block) json() string {
	switch b.kind {
	case "thinking":
		return fmt.Sprintf(`{"type":"thinking","thinking":%q,"signature":%q}`, b.text, b.signature)
	case "redacted":
		return fmt.Sprintf(`{"type":"redacted_thinking","data":%q}`, b.data)
	case "text":
		return fmt.Sprintf(`{"type":"text","text":%q}`, b.text)
	default:
		return fmt.Sprintf(`{"type":"tool_use","id":%q,"name":"get_weather","input":{"city":%q}}`, b.id, b.id)
	}
}

func messageJSON(stop string, blocks []block) string {
	var parts []string
	for _, b := range blocks {
		parts = append(parts, b.json())
	}
	return `{"id":"msg_01","type":"message","role":"assistant","model":"claude-opus-5-5","content":[` +
		strings.Join(parts, ",") + `],"stop_reason":"` + stop + `","usage":{"input_tokens":10,"output_tokens":30}}`
}

func sseEvent(b *strings.Builder, name, data string) {
	fmt.Fprintf(b, "event: %s\ndata: %s\n\n", name, data)
}

// messageSSE streams blocks: thinking text in two deltas with the signature as
// its own delta, tool input in two partial_json deltas.
func messageSSE(stop string, blocks []block) string {
	var b strings.Builder
	sseEvent(&b, "message_start", `{"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","model":"claude-opus-5-5","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}`)
	for i, bl := range blocks {
		delta := func(d string) {
			sseEvent(&b, "content_block_delta", fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":%s}`, i, d))
		}
		start := func(cb string) {
			sseEvent(&b, "content_block_start", fmt.Sprintf(`{"type":"content_block_start","index":%d,"content_block":%s}`, i, cb))
		}
		switch bl.kind {
		case "thinking":
			start(`{"type":"thinking","thinking":"","signature":""}`)
			half := len(bl.text) / 2
			for _, piece := range []string{bl.text[:half], bl.text[half:]} {
				if piece != "" {
					delta(fmt.Sprintf(`{"type":"thinking_delta","thinking":%q}`, piece))
				}
			}
			delta(fmt.Sprintf(`{"type":"signature_delta","signature":%q}`, bl.signature))
		case "redacted":
			start(fmt.Sprintf(`{"type":"redacted_thinking","data":%q}`, bl.data))
		case "text":
			start(`{"type":"text","text":""}`)
			delta(fmt.Sprintf(`{"type":"text_delta","text":%q}`, bl.text))
		default:
			start(fmt.Sprintf(`{"type":"tool_use","id":%q,"name":"get_weather","input":{}}`, bl.id))
			delta(`{"type":"input_json_delta","partial_json":"{\"city\":"}`)
			delta(fmt.Sprintf(`{"type":"input_json_delta","partial_json":"%s"}`, fmt.Sprintf(`\"%s\"}`, bl.id)))
		}
		sseEvent(&b, "content_block_stop", fmt.Sprintf(`{"type":"content_block_stop","index":%d}`, i))
	}
	sseEvent(&b, "message_delta", `{"type":"message_delta","delta":{"stop_reason":"`+stop+`"},"usage":{"output_tokens":30}}`)
	sseEvent(&b, "message_stop", `{"type":"message_stop"}`)
	return b.String()
}

// scriptedServer answers each request with the next scripted body and records
// the decoded request bodies.
type scriptedServer struct {
	mu       sync.Mutex
	replies  []string
	stream   bool
	requests []map[string]any
}

func (s *scriptedServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	raw, _ := io.ReadAll(r.Body)
	var body map[string]any
	_ = json.Unmarshal(raw, &body)
	s.mu.Lock()
	i := len(s.requests)
	s.requests = append(s.requests, body)
	s.mu.Unlock()
	if i >= len(s.replies) {
		http.Error(w, "unscripted request", http.StatusInternalServerError)
		return
	}
	if s.stream {
		w.Header().Set("Content-Type", "text/event-stream")
	} else {
		w.Header().Set("Content-Type", "application/json")
	}
	_, _ = io.WriteString(w, s.replies[i])
}

func newClient(srv *httptest.Server) llm.LLM {
	return llmanthropic.NewLLM(
		llmanthropic.WithAPIKey("test"),
		llmanthropic.WithModel(llm.Model{APIModel: "claude-opus-5-5", CanReason: true}),
		llmanthropic.WithMaxTokens(4096),
		llmanthropic.WithReasoningEffort(llmanthropic.ReasoningEffortHigh),
		llmanthropic.WithHTTPClient(&http.Client{
			Transport: rewriteTransport{host: srv.Listener.Addr().String()},
		}),
	)
}

// assistantShape lists the assistant message's blocks in a recorded request
// as "thinking", "redacted_thinking", "text" or "tool_use:<id>".
func assistantShape(t *testing.T, req map[string]any) ([]string, []map[string]any) {
	t.Helper()
	for _, m := range req["messages"].([]any) {
		mm := m.(map[string]any)
		if mm["role"] != "assistant" {
			continue
		}
		var shape []string
		var blocks []map[string]any
		for _, b := range mm["content"].([]any) {
			bm := b.(map[string]any)
			blocks = append(blocks, bm)
			if bm["type"] == "tool_use" {
				shape = append(shape, "tool_use:"+bm["id"].(string))
			} else {
				shape = append(shape, bm["type"].(string))
			}
		}
		return shape, blocks
	}
	t.Fatalf("no assistant message in request: %v", req)
	return nil, nil
}

const (
	thinkingText = "The user wants Paris weather, so I should call the tool."
	signature    = "EsoBCkYIBxgCKkDfixture-signature-not-a-real-one"
	redacted     = "EmwKAhgBEgyfixture-redacted-data-not-a-real-one"
)

var shapes = map[string]struct {
	blocks []block
	want   string // the assistant blocks the second request must carry, in order
	tools  int
}{
	"thinking-redacted-text-tool": {
		[]block{
			{kind: "thinking", text: thinkingText, signature: signature},
			{kind: "redacted", data: redacted},
			{kind: "text", text: "Checking."},
			{kind: "tool", id: "toolu_01"},
		},
		"thinking,redacted_thinking,text,tool_use:toolu_01", 1,
	},
	// Thinking between tool calls: the order is part of what is verified.
	"interleaved": {
		[]block{
			{kind: "thinking", text: thinkingText, signature: signature},
			{kind: "tool", id: "toolu_01"},
			{kind: "thinking", text: "and London", signature: signature + "-2"},
			{kind: "tool", id: "toolu_02"},
		},
		"thinking,tool_use:toolu_01,thinking,tool_use:toolu_02", 2,
	},
	"text-first": {
		[]block{
			{kind: "text", text: "Checking."},
			{kind: "thinking", text: thinkingText, signature: signature},
			{kind: "tool", id: "toolu_01"},
		},
		"text,thinking,tool_use:toolu_01", 1,
	},
	// Opus 5.5 returns an empty thinking text with a signature.
	"empty-thinking-text": {
		[]block{
			{kind: "thinking", text: "", signature: signature},
			{kind: "tool", id: "toolu_01"},
		},
		"thinking,tool_use:toolu_01", 1,
	},
}

func runLoop(t *testing.T, blocks []block, stream bool) []map[string]any {
	t.Helper()
	final := []block{{kind: "text", text: "It is sunny."}}
	script := &scriptedServer{stream: stream}
	if stream {
		script.replies = []string{messageSSE("tool_use", blocks), messageSSE("end_turn", final)}
	} else {
		script.replies = []string{messageJSON("tool_use", blocks), messageJSON("end_turn", final)}
	}
	srv := httptest.NewServer(script)
	defer srv.Close()
	a := agent.New(newClient(srv), agent.WithTools(weatherTool{}))
	if stream {
		for ev := range a.ChatStream(context.Background(), "Weather in Paris?") {
			if ev.Type == types.EventError {
				t.Fatal(ev.Error)
			}
		}
	} else if _, err := a.Chat(context.Background(), "Weather in Paris?"); err != nil {
		t.Fatal(err)
	}
	if len(script.requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(script.requests))
	}
	return script.requests
}

// TestThinkingReplaysThroughAgentLoop: the block survives the agent loop's
// carrier (llm.Response.ReasoningItems onto the assistant message) and is
// replayed in the order the API returned it, signature and redacted data
// intact, for the non-streamed and streamed response alike.
func TestThinkingReplaysThroughAgentLoop(t *testing.T) {
	for name, tc := range shapes {
		for _, stream := range []bool{false, true} {
			mode := map[bool]string{false: "chat", true: "stream"}[stream]
			t.Run(name+"/"+mode, func(t *testing.T) {
				reqs := runLoop(t, tc.blocks, stream)
				for _, m := range reqs[0]["messages"].([]any) {
					if m.(map[string]any)["role"] == "assistant" {
						t.Fatalf("first request has an assistant message: %v", m)
					}
				}
				shape, blocks := assistantShape(t, reqs[1])
				if got := strings.Join(shape, ","); got != tc.want {
					t.Fatalf("assistant blocks = %s, want %s", got, tc.want)
				}
				// Every thinking and redacted block is replayed unchanged.
				next := 0
				for _, b := range tc.blocks {
					if b.kind != "thinking" && b.kind != "redacted" {
						continue
					}
					for blocks[next]["type"] != "thinking" && blocks[next]["type"] != "redacted_thinking" {
						next++
					}
					got := blocks[next]
					next++
					if b.kind == "thinking" {
						if th, present := got["thinking"]; !present || th != b.text || got["signature"] != b.signature {
							t.Errorf("thinking block = %v, want text %q signature %q", got, b.text, b.signature)
						}
					} else if got["data"] != b.data {
						t.Errorf("redacted block = %v", got)
					}
				}
			})
		}
	}
}
