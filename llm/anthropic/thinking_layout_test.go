package anthropic

// Hand-built fixtures in the Messages API wire shape; signatures and data are
// opaque placeholders, not captured from the live API.

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/joakimcarlsson/ai/message"
	"github.com/joakimcarlsson/ai/tool"
)

func reply(blocks ...string) string {
	return `{"id":"msg_01","type":"message","role":"assistant","model":"claude-opus-5-5",` +
		`"content":[` + strings.Join(blocks, ",") + `],"stop_reason":"tool_use",` +
		`"usage":{"input_tokens":10,"output_tokens":30}}`
}

func thinkingBlock(text, sig string) string {
	return `{"type":"thinking","thinking":"` + text + `","signature":"` + sig + `"}`
}

func toolUseBlock(id string) string {
	return `{"type":"tool_use","id":"` + id + `","name":"get_weather","input":{"city":"` + id + `"}}`
}

func textBlock(text string) string { return `{"type":"text","text":"` + text + `"}` }

// capture sends one scripted reply through the adapter and returns the
// assistant message the agent loop would build from the response: reasoning
// items, then text, then tool calls, passed through message JSON the way
// Hearth persists it.
func capture(t *testing.T, body string) message.Message {
	t.Helper()
	srv := httptest.NewServer(&scriptedServer{replies: []string{body}})
	defer srv.Close()
	c := newThinkingClient(t, srv)
	resp, err := c.SendMessages(context.Background(),
		[]message.Message{message.NewUserMessage("hi")}, []tool.BaseTool{weatherTool{}})
	if err != nil {
		t.Fatal(err)
	}
	msg := message.Message{Role: message.Assistant}
	for _, item := range resp.ReasoningItems {
		msg.Parts = append(msg.Parts, item)
	}
	if resp.Content != "" {
		msg.Parts = append(msg.Parts, message.TextContent{Text: resp.Content})
	}
	for _, tc := range resp.ToolCalls {
		msg.Parts = append(msg.Parts, tc)
	}
	raw, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}
	var stored message.Message
	if err := json.Unmarshal(raw, &stored); err != nil {
		t.Fatal(err)
	}
	return stored
}

func replayHistory(assistant message.Message, results ...message.ToolResult) []message.Message {
	parts := make([]message.ContentPart, len(results))
	for i, r := range results {
		parts[i] = r
	}
	return []message.Message{
		message.NewUserMessage("hi"), assistant, message.NewMessage(message.Tool, parts),
	}
}

func blockIDs(blocks []map[string]any) string {
	var out []string
	for _, b := range blocks {
		switch b["type"] {
		case "tool_use":
			out = append(out, "tool_use:"+b["id"].(string))
		default:
			out = append(out, b["type"].(string))
		}
	}
	return strings.Join(out, ",")
}

func results(ids ...string) []message.ToolResult {
	var out []message.ToolResult
	for _, id := range ids {
		out = append(out, message.ToolResult{ToolCallID: id, Name: "get_weather", Content: "sunny"})
	}
	return out
}

// TestReplayKeepsOriginalBlockOrder: the response's block order survives
// capture, message JSON persistence and replay.
func TestReplayKeepsOriginalBlockOrder(t *testing.T) {
	cases := map[string]struct {
		body, want string
		ids        []string
	}{
		"interleaved": {
			reply(thinkingBlock("one", "sigA"), toolUseBlock("toolu_a"), thinkingBlock("two", "sigB"), toolUseBlock("toolu_b")),
			"thinking,tool_use:toolu_a,thinking,tool_use:toolu_b",
			[]string{"toolu_a", "toolu_b"},
		},
		"text-first": {
			reply(textBlock("Checking."), thinkingBlock("one", "sigA"), toolUseBlock("toolu_a")),
			"text,thinking,tool_use:toolu_a",
			[]string{"toolu_a"},
		},
		"thinking-first": {
			reply(thinkingBlock("one", "sigA"), textBlock("Checking."), toolUseBlock("toolu_a")),
			"thinking,text,tool_use:toolu_a",
			[]string{"toolu_a"},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			req := replayRequest(t, replayHistory(capture(t, tc.body), results(tc.ids...)...))
			if got := blockIDs(requestAssistantBlocks(t, req)); got != tc.want {
				t.Fatalf("assistant blocks = %s, want %s", got, tc.want)
			}
		})
	}
}

// TestReplayLayoutMismatchFallsBack: a layout that disagrees with the
// message's tool calls is ignored and thinking goes first.
func TestReplayLayoutMismatchFallsBack(t *testing.T) {
	msg := capture(t, reply(
		thinkingBlock("one", "sigA"), toolUseBlock("toolu_a"),
		thinkingBlock("two", "sigB"), toolUseBlock("toolu_b")))
	for i, p := range msg.Parts {
		if tc, ok := p.(message.ToolCall); ok && tc.ID == "toolu_b" {
			tc.ID = "toolu_other"
			msg.Parts[i] = tc
		}
	}
	req := replayRequest(t, replayHistory(msg, results("toolu_a", "toolu_other")...))
	want := "thinking,thinking,tool_use:toolu_a,tool_use:toolu_other"
	if got := blockIDs(requestAssistantBlocks(t, req)); got != want {
		t.Fatalf("assistant blocks = %s, want %s", got, want)
	}
}

// TestEmptyThinkingTextIsKeptAndReplayed: Opus 5.5 returns a thinking block
// with empty text and a signature. It is captured with an empty Text and the
// signature in Details (the text is stored once, in Text, never in Details),
// survives message JSON, and replays as thinking "" with the same signature.
func TestEmptyThinkingTextIsKeptAndReplayed(t *testing.T) {
	msg := capture(t, reply(thinkingBlock("", "sigEmpty"), toolUseBlock("toolu_a")))
	items := msg.ReasoningContent()
	if len(items) != 1 || items[0].Text != "" ||
		!strings.Contains(string(items[0].Details), `"signature":"sigEmpty"`) ||
		strings.Contains(string(items[0].Details), `"thinking":`) {
		t.Fatalf("captured items = %d, details %q", len(items), detailsOf(items))
	}
	req := replayRequest(t, replayHistory(msg, results("toolu_a")...))
	blocks := requestAssistantBlocks(t, req)
	if blockIDs(blocks) != "thinking,tool_use:toolu_a" {
		t.Fatalf("assistant blocks = %s", blockIDs(blocks))
	}
	th, present := blocks[0]["thinking"]
	if !present || th != "" || blocks[0]["signature"] != "sigEmpty" {
		t.Fatalf("replayed block = %v", blocks[0])
	}
}

func detailsOf(items []message.ReasoningContent) []string {
	out := make([]string, len(items))
	for i, it := range items {
		out[i] = string(it.Details)
	}
	return out
}

// TestDuplicateToolIDsKeepTheirOwnInput: two calls sharing an id (which the
// API itself never returns) each replay with their own input rather than
// collapsing into the last one.
func TestDuplicateToolIDsKeepTheirOwnInput(t *testing.T) {
	m := message.NewAssistantMessage()
	m.SetToolCalls([]message.ToolCall{
		{ID: "dup", Name: "get_weather", Input: `{"city":"Paris"}`, Type: "tool_use", Finished: true},
		{ID: "dup", Name: "get_weather", Input: `{"city":"London"}`, Type: "tool_use", Finished: true},
	})
	blocks, _ := assistantBlocks(m)
	var cities []string
	for _, b := range blocks {
		if b.OfToolUse != nil {
			in, _ := b.OfToolUse.Input.(map[string]any)
			cities = append(cities, fmt.Sprint(in["city"]))
		}
	}
	if strings.Join(cities, ",") != "Paris,London" {
		t.Fatalf("replayed inputs = %v, want Paris,London", cities)
	}
}
