package llm

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/joakimcarlsson/ai/message"
	"github.com/joakimcarlsson/ai/model"
	"github.com/joakimcarlsson/ai/schema"
	"github.com/joakimcarlsson/ai/tool"
	"github.com/openai/openai-go/responses"
)

// ─── Test helpers ──────────────────────────────────────────────────────────────

// stubTool implements tool.BaseTool for testing.
type stubTool struct {
	info tool.Info
}

func (s stubTool) Info() tool.Info                                               { return s.info }
func (s stubTool) Run(_ context.Context, _ tool.Call) (tool.Response, error)     { return tool.Response{}, nil }

func newStubTool(name, desc string, params map[string]any, required []string) stubTool {
	return stubTool{info: tool.Info{
		Name:        name,
		Description: desc,
		Parameters:  params,
		Required:    required,
	}}
}

// makeOpenAIClient creates an openaiClient for testing conversion logic.
func makeOpenAIClient(canReason bool, baseURL string) *openaiClient {
	effort := OpenAIReasoningEffortMedium
	return &openaiClient{
		providerOptions: llmClientOptions{
			model: model.Model{
				APIModel:  "gpt-5.4-mini",
				CanReason: canReason,
			},
			maxTokens: 1024,
		},
		options: openaiOptions{
			baseURL:         baseURL,
			reasoningEffort: &effort,
		},
		useResponses: canReason && baseURL == "",
	}
}

// ─── newResponseParams: Store safety ───────────────────────────────────────────

func TestNewResponseParams_StoreIsFalse(t *testing.T) {
	client := makeOpenAIClient(true, "")
	params := client.newResponseParams()

	data, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal params: %v", err)
	}

	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	store, ok := raw["store"]
	if !ok {
		t.Fatal("store field is missing from serialized JSON — OpenAI defaults to true when omitted")
	}
	if store != false {
		t.Errorf("store = %v, want false", store)
	}
}

func TestNewResponseParams_StoreNotOmitted(t *testing.T) {
	// Verify that store:false is NOT omitted by omitzero.
	// The param.Opt[bool] with value=false and status=included should serialize.
	client := makeOpenAIClient(true, "")
	params := client.newResponseParams()

	data, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	jsonStr := string(data)
	if !strings.Contains(jsonStr, `"store":false`) && !strings.Contains(jsonStr, `"store": false`) {
		t.Errorf("JSON does not contain store:false.\nGot: %s", jsonStr)
	}
}

// ─── useResponses routing ──────────────────────────────────────────────────────

func TestUseResponses_DirectOpenAI_CanReason(t *testing.T) {
	client := makeOpenAIClient(true, "")
	if !client.useResponses {
		t.Error("expected useResponses=true for CanReason + no baseURL")
	}
}

func TestUseResponses_DirectOpenAI_NoReason(t *testing.T) {
	client := makeOpenAIClient(false, "")
	if client.useResponses {
		t.Error("expected useResponses=false for !CanReason + no baseURL")
	}
}

func TestUseResponses_OpenRouter(t *testing.T) {
	client := makeOpenAIClient(true, "https://openrouter.ai/api/v1")
	if client.useResponses {
		t.Error("expected useResponses=false for OpenRouter (custom baseURL)")
	}
}

func TestUseResponses_GROQ(t *testing.T) {
	client := makeOpenAIClient(true, "https://api.groq.com/openai/v1")
	if client.useResponses {
		t.Error("expected useResponses=false for GROQ (custom baseURL)")
	}
}

func TestUseResponses_XAI(t *testing.T) {
	client := makeOpenAIClient(true, "https://api.x.ai/v1")
	if client.useResponses {
		t.Error("expected useResponses=false for xAI (custom baseURL)")
	}
}

func TestUseResponses_Mistral(t *testing.T) {
	client := makeOpenAIClient(true, "https://api.mistral.ai/v1")
	if client.useResponses {
		t.Error("expected useResponses=false for Mistral (custom baseURL)")
	}
}

// ─── convertResponseMessages ───────────────────────────────────────────────────

func TestConvertResponseMessages_SystemExtractedToInstructions(t *testing.T) {
	client := makeOpenAIClient(true, "")
	msgs := []message.Message{
		message.NewSystemMessage("You are a calculator."),
		message.NewUserMessage("What is 2+2?"),
	}

	input, instructions := client.convertResponseMessages(msgs)

	if instructions != "You are a calculator." {
		t.Errorf("instructions = %q, want %q", instructions, "You are a calculator.")
	}

	items := input.OfInputItemList
	if len(items) != 1 {
		t.Fatalf("expected 1 input item (user only), got %d", len(items))
	}
	// System message should NOT appear as an input item.
}

func TestConvertResponseMessages_MultipleSystemsConcatenated(t *testing.T) {
	client := makeOpenAIClient(true, "")
	msgs := []message.Message{
		message.NewSystemMessage("Rule 1."),
		message.NewSystemMessage("Rule 2."),
		message.NewUserMessage("Hello"),
	}

	_, instructions := client.convertResponseMessages(msgs)

	if instructions != "Rule 1.\n\nRule 2." {
		t.Errorf("instructions = %q, want %q", instructions, "Rule 1.\n\nRule 2.")
	}
}

func TestConvertResponseMessages_UserMessage(t *testing.T) {
	client := makeOpenAIClient(true, "")
	msgs := []message.Message{
		message.NewUserMessage("Hello world"),
	}

	input, instructions := client.convertResponseMessages(msgs)

	if instructions != "" {
		t.Errorf("expected no instructions, got %q", instructions)
	}

	items := input.OfInputItemList
	if len(items) != 1 {
		t.Fatalf("expected 1 item, got %d", len(items))
	}

	// Verify it serializes correctly as a user message.
	data, _ := json.Marshal(items[0])
	var raw map[string]any
	json.Unmarshal(data, &raw)
	if raw["role"] != "user" {
		t.Errorf("role = %v, want user", raw["role"])
	}
}

func TestConvertResponseMessages_AssistantTextOnly(t *testing.T) {
	client := makeOpenAIClient(true, "")
	msgs := []message.Message{
		message.NewMessage(message.Assistant, []message.ContentPart{
			message.TextContent{Text: "The answer is 42."},
		}),
	}

	input, _ := client.convertResponseMessages(msgs)
	items := input.OfInputItemList

	if len(items) != 1 {
		t.Fatalf("expected 1 item, got %d", len(items))
	}

	data, _ := json.Marshal(items[0])
	var raw map[string]any
	json.Unmarshal(data, &raw)
	if raw["role"] != "assistant" {
		t.Errorf("role = %v, want assistant", raw["role"])
	}
}

func TestConvertResponseMessages_UserWithBinaryContent(t *testing.T) {
	client := makeOpenAIClient(true, "")
	msgs := []message.Message{
		message.NewMessage(message.User, []message.ContentPart{
			message.TextContent{Text: "What is in this image?"},
			message.BinaryContent{
				MIMEType: "image/png",
				Data:     []byte{0x89, 0x50, 0x4E, 0x47}, // PNG header bytes
			},
		}),
	}

	input, _ := client.convertResponseMessages(msgs)
	items := input.OfInputItemList

	if len(items) != 1 {
		t.Fatalf("expected 1 item, got %d", len(items))
	}

	data, _ := json.Marshal(items[0])
	jsonStr := string(data)

	// Should have content array with text + image parts
	if !strings.Contains(jsonStr, "input_text") {
		t.Error("expected input_text content part for text")
	}
	if !strings.Contains(jsonStr, "input_image") {
		t.Error("expected input_image content part for binary image")
	}
	// Verify it's a data URI (not double-prefixed)
	if !strings.Contains(jsonStr, "data:image/png;base64,") {
		t.Error("expected data URI for binary content")
	}
	if strings.Contains(jsonStr, "data:image/png;base64,data:") {
		t.Error("double data-URI prefix detected")
	}
}

func TestConvertResponseMessages_UserWithImageURL(t *testing.T) {
	client := makeOpenAIClient(true, "")
	msgs := []message.Message{
		message.NewMessage(message.User, []message.ContentPart{
			message.TextContent{Text: "Describe this"},
			message.ImageURLContent{URL: "https://example.com/photo.jpg", Detail: "high"},
		}),
	}

	input, _ := client.convertResponseMessages(msgs)
	items := input.OfInputItemList

	if len(items) != 1 {
		t.Fatalf("expected 1 item, got %d", len(items))
	}

	data, _ := json.Marshal(items[0])
	jsonStr := string(data)

	if !strings.Contains(jsonStr, "https://example.com/photo.jpg") {
		t.Error("expected image URL in serialized output")
	}
	if !strings.Contains(jsonStr, `"detail":"high"`) {
		t.Errorf("expected detail=high in serialized output, got: %s", jsonStr)
	}
}

func TestConvertResponseMessages_UserWithImageURLNoDetail(t *testing.T) {
	client := makeOpenAIClient(true, "")
	msgs := []message.Message{
		message.NewMessage(message.User, []message.ContentPart{
			message.TextContent{Text: "What's this?"},
			message.ImageURLContent{URL: "https://example.com/photo.jpg"},
		}),
	}

	input, _ := client.convertResponseMessages(msgs)
	items := input.OfInputItemList

	data, _ := json.Marshal(items[0])
	jsonStr := string(data)

	// When Detail is empty, it should not be serialized
	if strings.Contains(jsonStr, `"detail"`) {
		t.Errorf("empty Detail should not appear in JSON, got: %s", jsonStr)
	}
}

func TestConvertResponseMessages_AssistantToolCalls(t *testing.T) {
	client := makeOpenAIClient(true, "")
	msgs := []message.Message{
		message.NewMessage(message.Assistant, []message.ContentPart{
			message.ToolCall{
				ID:    "call_abc",
				Name:  "get_weather",
				Input: `{"city":"NYC"}`,
				Type:  "function",
			},
		}),
	}

	input, _ := client.convertResponseMessages(msgs)
	items := input.OfInputItemList

	if len(items) != 1 {
		t.Fatalf("expected 1 item (function_call), got %d", len(items))
	}

	data, _ := json.Marshal(items[0])
	var raw map[string]any
	json.Unmarshal(data, &raw)
	if raw["type"] != "function_call" {
		t.Errorf("type = %v, want function_call", raw["type"])
	}
	if raw["call_id"] != "call_abc" {
		t.Errorf("call_id = %v, want call_abc", raw["call_id"])
	}
	if raw["name"] != "get_weather" {
		t.Errorf("name = %v, want get_weather", raw["name"])
	}
}

func TestConvertResponseMessages_AssistantTextAndToolCalls(t *testing.T) {
	client := makeOpenAIClient(true, "")
	msgs := []message.Message{
		message.NewMessage(message.Assistant, []message.ContentPart{
			message.TextContent{Text: "Let me check the weather."},
			message.ToolCall{
				ID:    "call_abc",
				Name:  "get_weather",
				Input: `{"city":"NYC"}`,
				Type:  "function",
			},
			message.ToolCall{
				ID:    "call_def",
				Name:  "get_weather",
				Input: `{"city":"LA"}`,
				Type:  "function",
			},
		}),
	}

	input, _ := client.convertResponseMessages(msgs)
	items := input.OfInputItemList

	// Should emit: 1 assistant message + 2 function_call items = 3 items
	if len(items) != 3 {
		t.Fatalf("expected 3 items (text + 2 tool calls), got %d", len(items))
	}

	// First item: assistant message
	data0, _ := json.Marshal(items[0])
	var raw0 map[string]any
	json.Unmarshal(data0, &raw0)
	if raw0["role"] != "assistant" {
		t.Errorf("item[0] role = %v, want assistant", raw0["role"])
	}

	// Second item: function_call
	data1, _ := json.Marshal(items[1])
	var raw1 map[string]any
	json.Unmarshal(data1, &raw1)
	if raw1["type"] != "function_call" {
		t.Errorf("item[1] type = %v, want function_call", raw1["type"])
	}
	if raw1["name"] != "get_weather" {
		t.Errorf("item[1] name = %v, want get_weather", raw1["name"])
	}

	// Third item: function_call
	data2, _ := json.Marshal(items[2])
	var raw2 map[string]any
	json.Unmarshal(data2, &raw2)
	if raw2["call_id"] != "call_def" {
		t.Errorf("item[2] call_id = %v, want call_def", raw2["call_id"])
	}
}

func TestConvertResponseMessages_ToolResults(t *testing.T) {
	client := makeOpenAIClient(true, "")
	msgs := []message.Message{
		message.NewMessage(message.Tool, []message.ContentPart{
			message.ToolResult{
				ToolCallID: "call_abc",
				Name:       "get_weather",
				Content:    `{"temp": 72}`,
			},
		}),
	}

	input, _ := client.convertResponseMessages(msgs)
	items := input.OfInputItemList

	if len(items) != 1 {
		t.Fatalf("expected 1 item, got %d", len(items))
	}

	data, _ := json.Marshal(items[0])
	var raw map[string]any
	json.Unmarshal(data, &raw)
	if raw["type"] != "function_call_output" {
		t.Errorf("type = %v, want function_call_output", raw["type"])
	}
	if raw["call_id"] != "call_abc" {
		t.Errorf("call_id = %v, want call_abc", raw["call_id"])
	}
	if raw["output"] != `{"temp": 72}` {
		t.Errorf("output = %v, want {\"temp\": 72}", raw["output"])
	}
}

func TestConvertResponseMessages_SummaryAsUser(t *testing.T) {
	client := makeOpenAIClient(true, "")
	msgs := []message.Message{
		message.NewSummaryMessage("Previously we discussed weather."),
	}

	input, _ := client.convertResponseMessages(msgs)
	items := input.OfInputItemList

	if len(items) != 1 {
		t.Fatalf("expected 1 item, got %d", len(items))
	}

	data, _ := json.Marshal(items[0])
	var raw map[string]any
	json.Unmarshal(data, &raw)
	if raw["role"] != "user" {
		t.Errorf("role = %v, want user (summary maps to user)", raw["role"])
	}
}

func TestConvertResponseMessages_MultiTurnWithTools(t *testing.T) {
	// Full multi-turn conversation: system → user → assistant(tool call) → tool result → user
	client := makeOpenAIClient(true, "")
	msgs := []message.Message{
		message.NewSystemMessage("You are helpful."),
		message.NewUserMessage("What's the weather in NYC?"),
		message.NewMessage(message.Assistant, []message.ContentPart{
			message.ToolCall{
				ID:    "call_123",
				Name:  "get_weather",
				Input: `{"city":"NYC"}`,
				Type:  "function",
			},
		}),
		message.NewMessage(message.Tool, []message.ContentPart{
			message.ToolResult{
				ToolCallID: "call_123",
				Name:       "get_weather",
				Content:    `{"temp": 72, "condition": "sunny"}`,
			},
		}),
		message.NewUserMessage("Thanks!"),
	}

	input, instructions := client.convertResponseMessages(msgs)

	if instructions != "You are helpful." {
		t.Errorf("instructions = %q, want %q", instructions, "You are helpful.")
	}

	items := input.OfInputItemList
	// Expected: user + function_call + function_call_output + user = 4 items (system → instructions)
	if len(items) != 4 {
		t.Fatalf("expected 4 input items, got %d", len(items))
	}

	// Verify types in order
	types := make([]string, len(items))
	for i, item := range items {
		data, _ := json.Marshal(item)
		var raw map[string]any
		json.Unmarshal(data, &raw)
		if typ, ok := raw["type"]; ok {
			types[i] = typ.(string)
		} else if _, ok := raw["role"]; ok {
			types[i] = "message:" + raw["role"].(string)
		}
	}

	expected := []string{"message:user", "function_call", "function_call_output", "message:user"}
	for i, exp := range expected {
		if types[i] != exp {
			t.Errorf("item[%d] type = %q, want %q", i, types[i], exp)
		}
	}
}

func TestConvertResponseMessages_EmptyMessages(t *testing.T) {
	client := makeOpenAIClient(true, "")
	input, instructions := client.convertResponseMessages(nil)

	if instructions != "" {
		t.Errorf("expected empty instructions, got %q", instructions)
	}
	if len(input.OfInputItemList) != 0 {
		t.Errorf("expected 0 items, got %d", len(input.OfInputItemList))
	}
}

// ─── convertResponseTools ──────────────────────────────────────────────────────

func TestConvertResponseTools_SingleTool(t *testing.T) {
	client := makeOpenAIClient(true, "")
	tools := []tool.BaseTool{
		newStubTool("calculate", "Evaluate math", map[string]any{
			"expression": map[string]any{"type": "string"},
		}, []string{"expression"}),
	}

	result := client.convertResponseTools(tools)

	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}

	ft := result[0].OfFunction
	if ft == nil {
		t.Fatal("expected OfFunction to be set")
	}
	if ft.Name != "calculate" {
		t.Errorf("name = %q, want calculate", ft.Name)
	}

	// Verify flat format (no function wrapper)
	data, _ := json.Marshal(result[0])
	var raw map[string]any
	json.Unmarshal(data, &raw)
	if raw["type"] != "function" {
		t.Errorf("type = %v, want function", raw["type"])
	}
	// Should NOT have a nested "function" key (that's chat/completions format)
	if _, hasFunctionKey := raw["function"]; hasFunctionKey {
		t.Error("should not have nested 'function' key — Responses API uses flat format")
	}
}

func TestConvertResponseTools_MultipleTools(t *testing.T) {
	client := makeOpenAIClient(true, "")
	tools := []tool.BaseTool{
		newStubTool("search", "Search the web", map[string]any{
			"query": map[string]any{"type": "string"},
		}, []string{"query"}),
		newStubTool("calculate", "Do math", map[string]any{
			"expr": map[string]any{"type": "string"},
		}, nil),
	}

	result := client.convertResponseTools(tools)

	if len(result) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(result))
	}
	if result[0].OfFunction.Name != "search" {
		t.Errorf("tool[0] name = %q, want search", result[0].OfFunction.Name)
	}
	if result[1].OfFunction.Name != "calculate" {
		t.Errorf("tool[1] name = %q, want calculate", result[1].OfFunction.Name)
	}
}

func TestConvertResponseTools_NoRequired(t *testing.T) {
	client := makeOpenAIClient(true, "")
	tools := []tool.BaseTool{
		newStubTool("ping", "Ping something", map[string]any{
			"host": map[string]any{"type": "string"},
		}, nil),
	}

	result := client.convertResponseTools(tools)
	data, _ := json.Marshal(result[0])
	var raw map[string]any
	json.Unmarshal(data, &raw)

	params := raw["parameters"].(map[string]any)
	if _, hasRequired := params["required"]; hasRequired {
		t.Error("should not include 'required' key when no required params")
	}
}

func TestConvertResponseTools_Empty(t *testing.T) {
	client := makeOpenAIClient(true, "")
	result := client.convertResponseTools(nil)
	if len(result) != 0 {
		t.Errorf("expected 0 tools, got %d", len(result))
	}
}

// ─── parseResponseOutput ───────────────────────────────────────────────────────

func TestParseResponseOutput_TextOnly(t *testing.T) {
	resp := &responses.Response{
		Output: []responses.ResponseOutputItemUnion{
			{
				Type: "message",
				Content: []responses.ResponseOutputMessageContentUnion{
					{Type: "output_text", Text: "Hello world"},
				},
			},
		},
	}

	content, toolCalls := parseResponseOutput(resp)

	if content != "Hello world" {
		t.Errorf("content = %q, want %q", content, "Hello world")
	}
	if len(toolCalls) != 0 {
		t.Errorf("expected 0 tool calls, got %d", len(toolCalls))
	}
}

func TestParseResponseOutput_MultipleTextParts(t *testing.T) {
	resp := &responses.Response{
		Output: []responses.ResponseOutputItemUnion{
			{
				Type: "message",
				Content: []responses.ResponseOutputMessageContentUnion{
					{Type: "output_text", Text: "Hello "},
					{Type: "output_text", Text: "world"},
				},
			},
		},
	}

	content, _ := parseResponseOutput(resp)

	if content != "Hello world" {
		t.Errorf("content = %q, want %q", content, "Hello world")
	}
}

func TestParseResponseOutput_FunctionCallsOnly(t *testing.T) {
	resp := &responses.Response{
		Output: []responses.ResponseOutputItemUnion{
			{
				Type:      "function_call",
				CallID:    "call_abc",
				Name:      "get_weather",
				Arguments: `{"city":"NYC"}`,
			},
		},
	}

	content, toolCalls := parseResponseOutput(resp)

	if content != "" {
		t.Errorf("content = %q, want empty", content)
	}
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(toolCalls))
	}
	tc := toolCalls[0]
	if tc.ID != "call_abc" {
		t.Errorf("ID = %q, want call_abc", tc.ID)
	}
	if tc.Name != "get_weather" {
		t.Errorf("Name = %q, want get_weather", tc.Name)
	}
	if tc.Input != `{"city":"NYC"}` {
		t.Errorf("Input = %q, want {\"city\":\"NYC\"}", tc.Input)
	}
	if tc.Type != "function" {
		t.Errorf("Type = %q, want function", tc.Type)
	}
	if !tc.Finished {
		t.Error("expected Finished=true")
	}
}

func TestParseResponseOutput_MultipleFunctionCalls(t *testing.T) {
	resp := &responses.Response{
		Output: []responses.ResponseOutputItemUnion{
			{
				Type:      "function_call",
				CallID:    "call_1",
				Name:      "search",
				Arguments: `{"q":"a"}`,
			},
			{
				Type:      "function_call",
				CallID:    "call_2",
				Name:      "search",
				Arguments: `{"q":"b"}`,
			},
		},
	}

	_, toolCalls := parseResponseOutput(resp)

	if len(toolCalls) != 2 {
		t.Fatalf("expected 2 tool calls, got %d", len(toolCalls))
	}
	if toolCalls[0].ID != "call_1" {
		t.Errorf("toolCalls[0].ID = %q, want call_1", toolCalls[0].ID)
	}
	if toolCalls[1].ID != "call_2" {
		t.Errorf("toolCalls[1].ID = %q, want call_2", toolCalls[1].ID)
	}
}

func TestParseResponseOutput_MixedTextAndFunctionCalls(t *testing.T) {
	resp := &responses.Response{
		Output: []responses.ResponseOutputItemUnion{
			{
				Type: "message",
				Content: []responses.ResponseOutputMessageContentUnion{
					{Type: "output_text", Text: "Let me check."},
				},
			},
			{
				Type:      "function_call",
				CallID:    "call_x",
				Name:      "lookup",
				Arguments: `{"term":"foo"}`,
			},
		},
	}

	content, toolCalls := parseResponseOutput(resp)

	if content != "Let me check." {
		t.Errorf("content = %q, want %q", content, "Let me check.")
	}
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(toolCalls))
	}
	if toolCalls[0].Name != "lookup" {
		t.Errorf("name = %q, want lookup", toolCalls[0].Name)
	}
}

func TestParseResponseOutput_EmptyOutput(t *testing.T) {
	resp := &responses.Response{
		Output: nil,
	}

	content, toolCalls := parseResponseOutput(resp)

	if content != "" {
		t.Errorf("content = %q, want empty", content)
	}
	if len(toolCalls) != 0 {
		t.Errorf("expected 0 tool calls, got %d", len(toolCalls))
	}
}

func TestParseResponseOutput_IgnoresUnknownTypes(t *testing.T) {
	resp := &responses.Response{
		Output: []responses.ResponseOutputItemUnion{
			{Type: "reasoning"},
			{Type: "web_search_call"},
			{
				Type: "message",
				Content: []responses.ResponseOutputMessageContentUnion{
					{Type: "output_text", Text: "actual content"},
				},
			},
		},
	}

	content, toolCalls := parseResponseOutput(resp)

	if content != "actual content" {
		t.Errorf("content = %q, want %q", content, "actual content")
	}
	if len(toolCalls) != 0 {
		t.Errorf("expected 0 tool calls, got %d", len(toolCalls))
	}
}

// ─── responseUsage ─────────────────────────────────────────────────────────────

func TestResponseUsage_NormalUsage(t *testing.T) {
	resp := &responses.Response{
		Usage: responses.ResponseUsage{
			InputTokens:  1000,
			OutputTokens: 500,
			InputTokensDetails: responses.ResponseUsageInputTokensDetails{
				CachedTokens: 200,
			},
		},
	}

	usage := responseUsage(resp)

	if usage.InputTokens != 800 {
		t.Errorf("InputTokens = %d, want 800 (1000-200 cached)", usage.InputTokens)
	}
	if usage.OutputTokens != 500 {
		t.Errorf("OutputTokens = %d, want 500", usage.OutputTokens)
	}
	if usage.CacheReadTokens != 200 {
		t.Errorf("CacheReadTokens = %d, want 200", usage.CacheReadTokens)
	}
	if usage.CacheCreationTokens != 0 {
		t.Errorf("CacheCreationTokens = %d, want 0", usage.CacheCreationTokens)
	}
}

func TestResponseUsage_NoCachedTokens(t *testing.T) {
	resp := &responses.Response{
		Usage: responses.ResponseUsage{
			InputTokens:  500,
			OutputTokens: 100,
		},
	}

	usage := responseUsage(resp)

	if usage.InputTokens != 500 {
		t.Errorf("InputTokens = %d, want 500", usage.InputTokens)
	}
	if usage.CacheReadTokens != 0 {
		t.Errorf("CacheReadTokens = %d, want 0", usage.CacheReadTokens)
	}
}

func TestResponseUsage_NilResponse(t *testing.T) {
	usage := responseUsage(nil)

	if usage.InputTokens != 0 || usage.OutputTokens != 0 {
		t.Errorf("expected zero usage for nil response, got %+v", usage)
	}
}

// ─── responsesFinishReason ─────────────────────────────────────────────────────

func TestResponsesFinishReason_Completed(t *testing.T) {
	reason := responsesFinishReason(responses.ResponseStatus("completed"), nil)
	if reason != message.FinishReasonEndTurn {
		t.Errorf("reason = %q, want %q", reason, message.FinishReasonEndTurn)
	}
}

func TestResponsesFinishReason_Incomplete(t *testing.T) {
	reason := responsesFinishReason(responses.ResponseStatus("incomplete"), nil)
	if reason != message.FinishReasonMaxTokens {
		t.Errorf("reason = %q, want %q", reason, message.FinishReasonMaxTokens)
	}
}

func TestResponsesFinishReason_Failed(t *testing.T) {
	reason := responsesFinishReason(responses.ResponseStatus("failed"), nil)
	if reason != message.FinishReasonError {
		t.Errorf("reason = %q, want %q", reason, message.FinishReasonError)
	}
}

func TestResponsesFinishReason_ToolCallsOverride(t *testing.T) {
	// Tool calls override status-based reason
	toolCalls := []message.ToolCall{{ID: "call_1", Name: "test"}}
	reason := responsesFinishReason(responses.ResponseStatus("completed"), toolCalls)
	if reason != message.FinishReasonToolUse {
		t.Errorf("reason = %q, want %q", reason, message.FinishReasonToolUse)
	}
}

func TestResponsesFinishReason_EmptyStatus(t *testing.T) {
	// Empty status (e.g., when finalResponse is nil) defaults to end_turn
	reason := responsesFinishReason(responses.ResponseStatus(""), nil)
	if reason != message.FinishReasonEndTurn {
		t.Errorf("reason = %q, want %q", reason, message.FinishReasonEndTurn)
	}
}

// ─── prepareResponseParams ─────────────────────────────────────────────────────

func TestPrepareResponseParams_BasicFields(t *testing.T) {
	client := makeOpenAIClient(true, "")
	msgs := []message.Message{
		message.NewSystemMessage("Be brief."),
		message.NewUserMessage("Hi"),
	}

	params := client.prepareResponseParams(msgs, nil)

	data, _ := json.Marshal(params)
	var raw map[string]any
	json.Unmarshal(data, &raw)

	// Model
	if raw["model"] != "gpt-5.4-mini" {
		t.Errorf("model = %v, want gpt-5.4-mini", raw["model"])
	}

	// Store: false (from newResponseParams)
	if raw["store"] != false {
		t.Errorf("store = %v, want false", raw["store"])
	}

	// Instructions (from system message)
	if raw["instructions"] != "Be brief." {
		t.Errorf("instructions = %v, want 'Be brief.'", raw["instructions"])
	}

	// Reasoning effort
	reasoning, ok := raw["reasoning"].(map[string]any)
	if !ok {
		t.Fatal("reasoning not set")
	}
	if reasoning["effort"] != "medium" {
		t.Errorf("reasoning.effort = %v, want medium", reasoning["effort"])
	}
}

func TestPrepareResponseParams_WithTools(t *testing.T) {
	client := makeOpenAIClient(true, "")
	tools := []tool.BaseTool{
		newStubTool("calc", "Calculate", map[string]any{
			"expr": map[string]any{"type": "string"},
		}, []string{"expr"}),
	}

	params := client.prepareResponseParams(nil, tools)

	if len(params.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(params.Tools))
	}
}

func TestPrepareResponseParams_NoToolsOmitted(t *testing.T) {
	client := makeOpenAIClient(true, "")
	params := client.prepareResponseParams(nil, nil)

	if params.Tools != nil {
		t.Errorf("expected nil tools when none provided, got %d", len(params.Tools))
	}
}

func TestPrepareResponseParams_ParallelToolCalls(t *testing.T) {
	enabled := true
	client := makeOpenAIClient(true, "")
	client.options.parallelToolCalls = &enabled

	params := client.prepareResponseParams(nil, nil)

	data, _ := json.Marshal(params)
	var raw map[string]any
	json.Unmarshal(data, &raw)
	if raw["parallel_tool_calls"] != true {
		t.Errorf("parallel_tool_calls = %v, want true", raw["parallel_tool_calls"])
	}
}

func TestPrepareResponseParams_NoReasoningWhenNil(t *testing.T) {
	client := makeOpenAIClient(true, "")
	client.options.reasoningEffort = nil

	params := client.prepareResponseParams(nil, nil)

	data, _ := json.Marshal(params)
	var raw map[string]any
	json.Unmarshal(data, &raw)
	// Reasoning should not be present when effort is nil
	if _, hasReasoning := raw["reasoning"]; hasReasoning {
		t.Error("reasoning should be omitted when effort is nil")
	}
}

// ─── Structured output params ──────────────────────────────────────────────────

func TestApplyResponsesSchema_WithRequired(t *testing.T) {
	params := responses.ResponseNewParams{}
	outputSchema := &schema.StructuredOutputInfo{
		Name: "person",
		Parameters: map[string]any{
			"name": map[string]any{"type": "string"},
			"age":  map[string]any{"type": "integer"},
		},
		Required: []string{"name"},
	}

	applyResponsesSchema(&params, outputSchema)

	data, _ := json.Marshal(params)
	var raw map[string]any
	json.Unmarshal(data, &raw)

	text, ok := raw["text"].(map[string]any)
	if !ok {
		t.Fatal("text config not present")
	}
	format, ok := text["format"].(map[string]any)
	if !ok {
		t.Fatal("format not present")
	}
	if format["type"] != "json_schema" {
		t.Errorf("format.type = %v, want json_schema", format["type"])
	}
	if format["name"] != "person" {
		t.Errorf("format.name = %v, want person", format["name"])
	}
	if format["strict"] != true {
		t.Errorf("format.strict = %v, want true", format["strict"])
	}

	schemaObj, ok := format["schema"].(map[string]any)
	if !ok {
		t.Fatal("schema not present in format")
	}
	if schemaObj["additionalProperties"] != false {
		t.Errorf("additionalProperties = %v, want false", schemaObj["additionalProperties"])
	}
	req, ok := schemaObj["required"].([]any)
	if !ok {
		t.Fatal("required not present in schema")
	}
	if len(req) != 1 || req[0] != "name" {
		t.Errorf("required = %v, want [name]", req)
	}
}

func TestApplyResponsesSchema_NoRequired(t *testing.T) {
	params := responses.ResponseNewParams{}
	outputSchema := &schema.StructuredOutputInfo{
		Name: "status",
		Parameters: map[string]any{
			"ok": map[string]any{"type": "boolean"},
		},
		Required: nil,
	}

	applyResponsesSchema(&params, outputSchema)

	data, _ := json.Marshal(params)
	var raw map[string]any
	json.Unmarshal(data, &raw)

	text := raw["text"].(map[string]any)
	format := text["format"].(map[string]any)
	schemaObj := format["schema"].(map[string]any)

	if _, hasRequired := schemaObj["required"]; hasRequired {
		t.Error("should not include 'required' key when no required params")
	}
}

// ─── ThoughtSignature JSON round-trip ──────────────────────────────────────────

func TestToolCall_ThoughtSignatureJSONRoundTrip(t *testing.T) {
	original := message.ToolCall{
		ID:               "call_xyz",
		Name:             "get_weather",
		Input:            `{"city":"NYC"}`,
		Type:             "function",
		Finished:         true,
		ThoughtSignature: []byte{0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03},
	}

	data, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var restored message.ToolCall
	if err := json.Unmarshal(data, &restored); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if string(restored.ThoughtSignature) != string(original.ThoughtSignature) {
		t.Errorf("ThoughtSignature mismatch.\n  got:  %x\n  want: %x", restored.ThoughtSignature, original.ThoughtSignature)
	}
	if restored.ID != original.ID {
		t.Errorf("ID = %q, want %q", restored.ID, original.ID)
	}
	if restored.Name != original.Name {
		t.Errorf("Name = %q, want %q", restored.Name, original.Name)
	}
}

func TestToolCall_ThoughtSignatureNilOmitted(t *testing.T) {
	tc := message.ToolCall{
		ID:    "call_123",
		Name:  "test",
		Input: "{}",
		Type:  "function",
	}

	data, _ := json.Marshal(tc)
	jsonStr := string(data)

	if strings.Contains(jsonStr, "thought_signature") {
		t.Errorf("nil ThoughtSignature should be omitted from JSON.\nGot: %s", jsonStr)
	}
}

func TestToolCall_ThoughtSignatureBackwardCompat(t *testing.T) {
	// JSON without thought_signature field should deserialize with nil.
	oldJSON := `{"id":"call_old","name":"test","input":"{}","type":"function","finished":true}`

	var tc message.ToolCall
	if err := json.Unmarshal([]byte(oldJSON), &tc); err != nil {
		t.Fatalf("unmarshal old JSON: %v", err)
	}

	if tc.ThoughtSignature != nil {
		t.Errorf("expected nil ThoughtSignature for old JSON, got %x", tc.ThoughtSignature)
	}
	if tc.ID != "call_old" {
		t.Errorf("ID = %q, want call_old", tc.ID)
	}
}

// ─── Message JSON round-trip with ThoughtSignature ─────────────────────────────

func TestMessage_ThoughtSignatureRoundTripThroughMessage(t *testing.T) {
	// Test the full path: Message → JSON → Message with ThoughtSignature on ToolCall.
	original := message.NewMessage(message.Assistant, []message.ContentPart{
		message.TextContent{Text: "I'll look that up."},
		message.ToolCall{
			ID:               "call_gem",
			Name:             "search",
			Input:            `{"q":"test"}`,
			Type:             "function",
			Finished:         true,
			ThoughtSignature: []byte{0xAA, 0xBB, 0xCC},
		},
	})

	data, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("marshal message: %v", err)
	}

	var restored message.Message
	if err := json.Unmarshal(data, &restored); err != nil {
		t.Fatalf("unmarshal message: %v", err)
	}

	toolCalls := restored.ToolCalls()
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(toolCalls))
	}

	if string(toolCalls[0].ThoughtSignature) != string([]byte{0xAA, 0xBB, 0xCC}) {
		t.Errorf("ThoughtSignature not preserved through Message round-trip.\n  got:  %x\n  want: aabbcc",
			toolCalls[0].ThoughtSignature)
	}
}

