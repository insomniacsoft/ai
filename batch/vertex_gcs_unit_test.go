package batch

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/joakimcarlsson/ai/message"
	"github.com/joakimcarlsson/ai/schema"
)

func TestSubmitWithModelRejectsEmptyModel(t *testing.T) {
	p := &VertexGCSProcessor{}
	_, err := p.SubmitWithModel(context.Background(), "", []Request{{ID: "rec-1"}})
	if err == nil || !strings.Contains(err.Error(), "model is required") {
		t.Fatalf("SubmitWithModel error = %v, want model is required", err)
	}
}

// TestBuildJSONLLine_MarkerAndSchema verifies the Vertex input line carries the
// correlation marker on the user part and a well-formed responseSchema (array
// items included — the bug that broke Gemini batch).
func TestBuildJSONLLine_MarkerAndSchema(t *testing.T) {
	p := &VertexGCSProcessor{cfg: VertexGCSConfig{MaxTokens: 4096}}
	req := &Request{
		ID:   "rec-123",
		Type: RequestTypeChat,
		Messages: []message.Message{
			message.NewSystemMessage("system text"),
			message.NewUserMessage("<article>body</article>"),
		},
		OutputSchema: schema.NewStructuredOutputInfo("X", "d",
			map[string]any{
				"summary": map[string]any{"type": "string"},
				"brands":  map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
			}, []string{"summary", "brands"}),
	}
	raw, err := p.buildJSONLLine(req)
	if err != nil {
		t.Fatalf("buildJSONLLine: %v", err)
	}
	var line map[string]any
	if err := json.Unmarshal(raw, &line); err != nil {
		t.Fatalf("line not json: %v", err)
	}
	reqObj := line["request"].(map[string]any)
	contents := reqObj["contents"].([]any)
	c0 := contents[0].(map[string]any)
	parts := c0["parts"].([]any)
	text := parts[0].(map[string]any)["text"].(string)
	if !strings.HasPrefix(text, "[[ovreq:rec-123]] ") {
		t.Errorf("marker not prefixed on user part: %q", text)
	}
	gc := reqObj["generationConfig"].(map[string]any)
	if gc["responseMimeType"] != "application/json" {
		t.Errorf("responseMimeType = %v", gc["responseMimeType"])
	}
	sch := gc["responseSchema"].(map[string]any)
	props := sch["properties"].(map[string]any)
	brands := props["brands"].(map[string]any)
	if brands["items"] == nil {
		t.Error("array field brands missing items in responseSchema")
	}
}

// TestParseVertexResultLine_Correlates verifies a Vertex output JSONL line is
// parsed: id from the echoed marker, content from the candidate, token usage.
func TestParseVertexResultLine_Correlates(t *testing.T) {
	line := `{"request":{"contents":[{"role":"user","parts":[{"text":"[[ovreq:rec-xyz]] <article>x</article>"}]}]},"response":{"candidates":[{"content":{"parts":[{"text":"{\"summary\":\"s\",\"brands\":[\"A\"]}"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":120,"candidatesTokenCount":30}},"status":""}`
	r := parseVertexResultLine(line)
	if r.Err != nil {
		t.Fatalf("unexpected err: %v", r.Err)
	}
	if r.ID != "rec-xyz" {
		t.Errorf("id = %q, want rec-xyz", r.ID)
	}
	if r.ChatResponse == nil || !strings.Contains(r.ChatResponse.Content, `"summary"`) {
		t.Errorf("content = %+v", r.ChatResponse)
	}
	if r.ChatResponse.Usage.InputTokens != 120 || r.ChatResponse.Usage.OutputTokens != 30 {
		t.Errorf("usage = %+v", r.ChatResponse.Usage)
	}
}

// TestParseVertexResultLine_EmptyResponseIsError verifies a line with no
// candidates yields a per-item error (record stays pending for re-sweep).
func TestParseVertexResultLine_EmptyResponseIsError(t *testing.T) {
	line := `{"request":{"contents":[{"role":"user","parts":[{"text":"[[ovreq:rec-empty]] x"}]}]},"status":"{\"code\":3,\"message\":\"bad\"}"}`
	r := parseVertexResultLine(line)
	if r.ID != "rec-empty" {
		t.Errorf("id = %q", r.ID)
	}
	if r.Err == nil {
		t.Error("expected per-item error for empty response")
	}
}
