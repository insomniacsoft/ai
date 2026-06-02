package batch

import (
	"context"
	"testing"

	"github.com/joakimcarlsson/ai/message"
	"github.com/joakimcarlsson/ai/model"
	"github.com/joakimcarlsson/ai/schema"
	"google.golang.org/genai"
)

// TestNew_DispatchesByProvider verifies New returns a Processor for each
// provider (the baseProcessor dispatch + vendor-client construction) without
// error or panic — construction must not touch the network.
func TestNew_DispatchesByProvider(t *testing.T) {
	for _, p := range []model.Provider{
		model.ProviderOpenAI,
		model.ProviderAnthropic,
		model.ProviderGemini,
		model.ProviderOpenRouter, // → concurrent fallback
	} {
		proc, err := New(p,
			WithAPIKey("test-key"),
			WithModel(model.Model{APIModel: "test-model"}),
		)
		if err != nil {
			t.Errorf("New(%s) error: %v", p, err)
		}
		if proc == nil {
			t.Errorf("New(%s) returned nil Processor", p)
		}
	}
}

// TestProcess_EmptyBatch verifies an empty request slice short-circuits to an
// empty Response without calling any provider.
func TestProcess_EmptyBatch(t *testing.T) {
	proc, err := New(model.ProviderGemini, WithAPIKey("k"), WithModel(model.Model{APIModel: "gemini-3-flash"}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := proc.Process(context.Background(), nil)
	if err != nil {
		t.Fatalf("Process(nil): %v", err)
	}
	if resp.Total != 0 || len(resp.Results) != 0 {
		t.Errorf("empty batch: got Total=%d Results=%d, want 0/0", resp.Total, len(resp.Results))
	}
}

// TestProcessAsync_EmptyBatch verifies the async path emits a single complete
// event for an empty batch.
func TestProcessAsync_EmptyBatch(t *testing.T) {
	proc, _ := New(model.ProviderGemini, WithAPIKey("k"), WithModel(model.Model{APIModel: "gemini-3-flash"}))
	ch, err := proc.ProcessAsync(context.Background(), nil)
	if err != nil {
		t.Fatalf("ProcessAsync(nil): %v", err)
	}
	var got []Event
	for ev := range ch {
		got = append(got, ev)
	}
	if len(got) != 1 || got[0].Type != EventComplete {
		t.Fatalf("empty async batch: want one EventComplete, got %+v", got)
	}
}

// TestConvertToGenaiSchema verifies the JSON-schema → genai.Schema conversion
// the structured-output path relies on: object type, required list, and typed
// scalar properties.
func TestConvertToGenaiSchema(t *testing.T) {
	props := map[string]any{
		"title":  map[string]any{"type": "string", "description": "the title"},
		"price":  map[string]any{"type": "number"},
		"in_stk": map[string]any{"type": "boolean"},
	}
	s := convertToGenaiSchema(props, []string{"title", "price"})

	if s.Type != genai.TypeObject {
		t.Errorf("root type = %v, want object", s.Type)
	}
	if len(s.Required) != 2 {
		t.Errorf("required = %v, want [title price]", s.Required)
	}
	if s.Properties["title"].Type != genai.TypeString {
		t.Errorf("title type = %v, want string", s.Properties["title"].Type)
	}
	if s.Properties["title"].Description != "the title" {
		t.Errorf("title desc = %q", s.Properties["title"].Description)
	}
	if s.Properties["price"].Type != genai.TypeNumber {
		t.Errorf("price type = %v, want number", s.Properties["price"].Type)
	}
	if s.Properties["in_stk"].Type != genai.TypeBoolean {
		t.Errorf("in_stk type = %v, want boolean", s.Properties["in_stk"].Type)
	}
}

// TestConvertToGenaiSchema_ArrayItemsAndNestedObject is the regression guard
// for the Gemini-Batch 400 "response_schema.properties[x].items: missing
// field". Array properties MUST carry Items and object properties MUST recurse
// into nested Properties — the original converter dropped both.
func TestConvertToGenaiSchema_ArrayItemsAndNestedObject(t *testing.T) {
	props := map[string]any{
		"brands": map[string]any{
			"type":        "array",
			"description": "brand names",
			"items":       map[string]any{"type": "string"},
		},
		"meta": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"lang": map[string]any{"type": "string"},
				"tags": map[string]any{
					"type":  "array",
					"items": map[string]any{"type": "string"},
				},
			},
			"required": []any{"lang"},
		},
		"status": map[string]any{
			"type": "string",
			"enum": []any{"draft", "published"},
		},
	}
	s := convertToGenaiSchema(props, []string{"brands"})

	brands := s.Properties["brands"]
	if brands.Type != genai.TypeArray {
		t.Fatalf("brands type = %v, want array", brands.Type)
	}
	if brands.Items == nil {
		t.Fatal("brands.Items is nil — this is the 400-causing bug")
	}
	if brands.Items.Type != genai.TypeString {
		t.Errorf("brands.Items type = %v, want string", brands.Items.Type)
	}

	meta := s.Properties["meta"]
	if meta.Type != genai.TypeObject {
		t.Fatalf("meta type = %v, want object", meta.Type)
	}
	if meta.Properties["lang"].Type != genai.TypeString {
		t.Errorf("meta.lang type = %v, want string", meta.Properties["lang"].Type)
	}
	if tags := meta.Properties["tags"]; tags.Type != genai.TypeArray || tags.Items == nil {
		t.Errorf("meta.tags = %+v, want array with items", tags)
	}
	if len(meta.Required) != 1 || meta.Required[0] != "lang" {
		t.Errorf("meta.Required = %v, want [lang]", meta.Required)
	}

	if got := s.Properties["status"].Enum; len(got) != 2 || got[0] != "draft" {
		t.Errorf("status.Enum = %v, want [draft published]", got)
	}
}

// TestStructuredOutputRequestAccepted verifies a chat Request carrying an
// OutputSchema is accepted (the field exists + a Gemini processor builds with
// it). The live constrained-JSON round-trip is covered by the overtura
// acceptance smoke (needs a real Gemini key).
func TestStructuredOutputRequestAccepted(t *testing.T) {
	out := schema.NewStructuredOutputInfo(
		"extraction",
		"5-field extraction",
		map[string]any{
			"title":  map[string]any{"type": "string"},
			"author": map[string]any{"type": "string"},
		},
		[]string{"title"},
	)
	req := Request{
		ID:           "r1",
		Type:         RequestTypeChat,
		Messages:     []message.Message{message.NewUserMessage("extract")},
		OutputSchema: out,
	}
	if req.OutputSchema == nil || req.OutputSchema.Name != "extraction" {
		t.Fatal("OutputSchema not carried on Request")
	}
	// Building a Gemini processor with such a request must not panic at
	// construction (network only happens at Process time).
	if _, err := New(model.ProviderGemini, WithAPIKey("k"), WithModel(model.Model{APIModel: "gemini-3.1-flash-lite"})); err != nil {
		t.Fatalf("New(Gemini): %v", err)
	}
}
