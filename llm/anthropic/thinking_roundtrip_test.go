package anthropic

// The request and response bodies below are hand-built fixtures that follow the
// Messages API wire shape (thinking block with signature, redacted_thinking
// with data, tool_use; SSE with thinking_delta and signature_delta). They are
// not captured from the live API; the signatures and data are opaque
// placeholders, which is all the replay path ever treats them as.

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

	"github.com/joakimcarlsson/ai/llm"
	"github.com/joakimcarlsson/ai/message"
	"github.com/joakimcarlsson/ai/tool"
	"github.com/joakimcarlsson/ai/types"
)

const (
	fixtureThinking  = "The user wants Paris weather, so I should call the tool."
	fixtureSignature = "EsoBCkYIBxgCKkDfixture-signature-not-a-real-one"
	fixtureRedacted  = "EmwKAhgBEgyfixture-redacted-data-not-a-real-one"
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

// thinkingToolReply is a non-streamed response: thinking, redacted thinking,
// text, then a tool call. The API returns thinking blocks first.
const thinkingToolReply = `{"id":"msg_01","type":"message","role":"assistant","model":"claude-opus-5-5",` +
	`"content":[` +
	`{"type":"thinking","thinking":"` + fixtureThinking + `","signature":"` + fixtureSignature + `"},` +
	`{"type":"redacted_thinking","data":"` + fixtureRedacted + `"},` +
	`{"type":"text","text":"Checking."},` +
	`{"type":"tool_use","id":"toolu_01","name":"get_weather","input":{"city":"Paris"}}],` +
	`"stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":30}}`

const finalReply = `{"id":"msg_02","type":"message","role":"assistant","model":"claude-opus-5-5",` +
	`"content":[{"type":"text","text":"It is sunny."}],"stop_reason":"end_turn",` +
	`"usage":{"input_tokens":20,"output_tokens":5}}`

func sse(events ...[2]string) string {
	var b strings.Builder
	for _, e := range events {
		fmt.Fprintf(&b, "event: %s\ndata: %s\n\n", e[0], e[1])
	}
	return b.String()
}

// thinkingToolStream is thinkingToolReply as an SSE stream: the thinking text
// and the tool input arrive in pieces, the signature as its own delta.
func thinkingToolStream() string {
	half := len(fixtureThinking) / 2
	return sse(
		[2]string{"message_start", `{"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","model":"claude-opus-5-5","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}`},
		[2]string{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"` + fixtureThinking[:half] + `"}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"` + fixtureThinking[half:] + `"}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"` + fixtureSignature + `"}}`},
		[2]string{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		[2]string{"content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"` + fixtureRedacted + `"}}`},
		[2]string{"content_block_stop", `{"type":"content_block_stop","index":1}`},
		[2]string{"content_block_start", `{"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"Checking."}}`},
		[2]string{"content_block_stop", `{"type":"content_block_stop","index":2}`},
		[2]string{"content_block_start", `{"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"toolu_01","name":"get_weather","input":{}}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"\"Paris\"}"}}`},
		[2]string{"content_block_stop", `{"type":"content_block_stop","index":3}`},
		[2]string{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":30}}`},
		[2]string{"message_stop", `{"type":"message_stop"}`},
	)
}

func finalStream() string {
	return sse(
		[2]string{"message_start", `{"type":"message_start","message":{"id":"msg_02","type":"message","role":"assistant","model":"claude-opus-5-5","content":[],"usage":{"input_tokens":20,"output_tokens":0}}}`},
		[2]string{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"It is sunny."}}`},
		[2]string{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		[2]string{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`},
		[2]string{"message_stop", `{"type":"message_stop"}`},
	)
}

// scriptedServer answers each /v1/messages request with the next scripted
// body and records the decoded request bodies.
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

func newThinkingClient(
	t *testing.T,
	srv *httptest.Server,
	extra ...Option,
) llm.LLM {
	t.Helper()
	opts := []Option{
		WithAPIKey("test"),
		WithModel(llm.Model{APIModel: "claude-opus-5-5", CanReason: true}),
		WithMaxTokens(4096),
		WithReasoningEffort(ReasoningEffortHigh),
		WithHTTPClient(&http.Client{
			Transport: reasoningTransport{host: srv.Listener.Addr().String()},
		}),
	}
	return NewLLM(append(opts, extra...)...)
}

// assistantBlocks returns the content blocks of the assistant message in a
// recorded request.
func requestAssistantBlocks(t *testing.T, req map[string]any) []map[string]any {
	t.Helper()
	msgs, _ := req["messages"].([]any)
	for _, m := range msgs {
		mm := m.(map[string]any)
		if mm["role"] != "assistant" {
			continue
		}
		var out []map[string]any
		for _, b := range mm["content"].([]any) {
			out = append(out, b.(map[string]any))
		}
		return out
	}
	t.Fatalf("no assistant message in request: %v", req)
	return nil
}

func blockTypes(blocks []map[string]any) []string {
	var types []string
	for _, b := range blocks {
		types = append(types, b["type"].(string))
	}
	return types
}

// TestStreamAssemblesThinkingBlock checks the streamed response yields the
// same reasoning items as the non-streamed one: text joined from the deltas,
// signature from the signature delta, redacted data from the start event.
func TestStreamAssemblesThinkingBlock(t *testing.T) {
	collect := func(stream bool) []message.ReasoningContent {
		script := &scriptedServer{stream: stream}
		if stream {
			script.replies = []string{thinkingToolStream()}
		} else {
			script.replies = []string{thinkingToolReply}
		}
		srv := httptest.NewServer(script)
		defer srv.Close()
		c := newThinkingClient(t, srv)
		msgs := []message.Message{message.NewUserMessage("Weather in Paris?")}
		if !stream {
			resp, err := c.SendMessages(context.Background(), msgs, []tool.BaseTool{weatherTool{}})
			if err != nil {
				t.Fatal(err)
			}
			return resp.ReasoningItems
		}
		var items []message.ReasoningContent
		for ev := range c.StreamResponse(context.Background(), msgs, []tool.BaseTool{weatherTool{}}) {
			if ev.Type == types.EventError {
				t.Fatal(ev.Error)
			}
			if ev.Type == types.EventComplete {
				items = ev.Response.ReasoningItems
			}
		}
		return items
	}
	sent := collect(false)
	streamed := collect(true)
	if len(sent) != 2 || len(streamed) != 2 {
		t.Fatalf("items: sent=%d streamed=%d, want 2 each", len(sent), len(streamed))
	}
	for i := range sent {
		if sent[i].Field != streamed[i].Field ||
			string(sent[i].Details) != string(streamed[i].Details) ||
			sent[i].Text != streamed[i].Text {
			t.Errorf("item %d differs:\nsent     %+v\nstreamed %+v", i, sent[i], streamed[i])
		}
	}
	if streamed[0].Text != fixtureThinking {
		t.Errorf("thinking text = %q", streamed[0].Text)
	}
	var d struct{ Type, Thinking, Signature, Data string }
	if err := json.Unmarshal(streamed[0].Details, &d); err != nil ||
		d.Type != "thinking" || d.Signature != fixtureSignature {
		t.Errorf("thinking details = %s (%v)", streamed[0].Details, err)
	}
	if err := json.Unmarshal(streamed[1].Details, &d); err != nil ||
		d.Type != "redacted_thinking" || d.Data != fixtureRedacted {
		t.Errorf("redacted details = %s (%v)", streamed[1].Details, err)
	}
}

// replayRequest sends msgs through the adapter against a one-reply server and
// returns the recorded request.
func replayRequest(t *testing.T, msgs []message.Message) map[string]any {
	t.Helper()
	script := &scriptedServer{replies: []string{finalReply}}
	srv := httptest.NewServer(script)
	defer srv.Close()
	c := newThinkingClient(t, srv)
	if _, err := c.SendMessages(context.Background(), msgs, []tool.BaseTool{weatherTool{}}); err != nil {
		t.Fatal(err)
	}
	return script.requests[0]
}

func thinkingPart() message.ReasoningContent {
	return message.ReasoningContent{
		Text:  fixtureThinking,
		Field: ReasoningFieldThinking,
		Details: json.RawMessage(
			`{"type":"thinking","thinking":"` + fixtureThinking + `","signature":"` + fixtureSignature + `"}`,
		),
	}
}

func toolLoopHistory(parts ...message.ContentPart) []message.Message {
	assistant := message.Message{Role: message.Assistant, Parts: append(parts,
		message.ToolCall{ID: "toolu_01", Name: "get_weather", Input: `{"city":"Paris"}`, Finished: true})}
	return []message.Message{
		message.NewUserMessage("Weather in Paris?"),
		assistant,
		message.NewMessage(message.Tool, []message.ContentPart{message.ToolResult{ToolCallID: "toolu_01", Name: "get_weather", Content: "sunny"}}),
	}
}

// TestReplayedThinkingComesFirst: parts stored with text ahead of the
// reasoning still go out thinking first.
func TestReplayedThinkingComesFirst(t *testing.T) {
	req := replayRequest(t, toolLoopHistory(message.TextContent{Text: "Checking."}, thinkingPart()))
	got := strings.Join(blockTypes(requestAssistantBlocks(t, req)), ",")
	if got != "thinking,text,tool_use" {
		t.Fatalf("assistant blocks = %s", got)
	}
}

// TestNoThinkingContentSendsNone: a message with no reasoning parts, or only
// another provider's reasoning item, sends no thinking block.
func TestNoThinkingContentSendsNone(t *testing.T) {
	openaiItem := message.ReasoningContent{
		ID: "rs_1", EncryptedContent: "opaque", Summary: []string{"plan"}, Text: "plan",
	}
	for name, parts := range map[string][]message.ContentPart{
		"none":             {message.TextContent{Text: "Checking."}},
		"other-provider":   {message.TextContent{Text: "Checking."}, openaiItem},
		"text-only-reason": {message.TextContent{Text: "Checking."}, message.ReasoningContent{Text: "musing"}},
	} {
		t.Run(name, func(t *testing.T) {
			req := replayRequest(t, toolLoopHistory(parts...))
			got := strings.Join(blockTypes(requestAssistantBlocks(t, req)), ",")
			if got != "text,tool_use" {
				t.Fatalf("assistant blocks = %s", got)
			}
		})
	}
}

// TestXhighEffort: KTD15, xhigh reaches output_config.effort.
func TestXhighEffort(t *testing.T) {
	script := &scriptedServer{replies: []string{finalReply}}
	srv := httptest.NewServer(script)
	defer srv.Close()
	c := newThinkingClient(t, srv, WithReasoningEffort(ReasoningEffort("xhigh")))
	if _, err := c.SendMessages(context.Background(),
		[]message.Message{message.NewUserMessage("hi")}, nil); err != nil {
		t.Fatal(err)
	}
	oc, _ := script.requests[0]["output_config"].(map[string]any)
	if oc["effort"] != "xhigh" {
		t.Fatalf("output_config = %v", oc)
	}
}
