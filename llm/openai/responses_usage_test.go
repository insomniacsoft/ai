package openai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/joakimcarlsson/ai/llm"
	"github.com/joakimcarlsson/ai/message"
	"github.com/joakimcarlsson/ai/types"
)

// A Responses body whose prompt was mostly served from the cache. The API
// reports input_tokens as the WHOLE prompt and cached_tokens as the part of
// it that was a cache hit -- a subset, not an addition.
const responsesCachedBody = `{"id":"resp_c","object":"response","status":"completed",` +
	`"output":[{"type":"message","role":"assistant",` +
	`"content":[{"type":"output_text","text":"hi"}]}],` +
	`"usage":{"input_tokens":1000,"input_tokens_details":{"cached_tokens":800},` +
	`"output_tokens":50,"output_tokens_details":{"reasoning_tokens":12},"total_tokens":1050}}`

// TestResponsesUsageExcludesCachedFromInput confirms the Responses mapping
// matches the chat-completions contract: InputTokens is the uncached input,
// and CacheReadTokens carries the rest. Covered on both the plain and
// streamed path.
func TestResponsesUsageExcludesCachedFromInput(t *testing.T) {
	check := func(t *testing.T, u llm.TokenUsage) {
		t.Helper()
		if u.InputTokens != 200 {
			t.Errorf(
				"InputTokens = %d, want 200 (1000 prompt minus 800 cached)",
				u.InputTokens,
			)
		}
		if u.CacheReadTokens != 800 {
			t.Errorf("CacheReadTokens = %d, want 800", u.CacheReadTokens)
		}
		if u.ReasoningTokens != 12 {
			t.Errorf("ReasoningTokens = %d, want 12", u.ReasoningTokens)
		}
		if u.OutputTokens != 50 {
			t.Errorf("OutputTokens = %d, want 50", u.OutputTokens)
		}
		if got := u.InputTokens + u.CacheReadTokens + u.CacheCreationTokens; got != 1000 {
			t.Errorf(
				"input + cache read + cache write = %d, want the whole prompt, 1000",
				got,
			)
		}
	}

	t.Run("send", func(t *testing.T) {
		srv := newResponsesServer(t, nil, responsesCachedBody)
		defer srv.Close()
		client := NewResponsesLLM(
			WithResponsesAPIKey("test-key"),
			WithResponsesBaseURL(srv.URL),
			WithResponsesModel(llm.Model{APIModel: "gpt-4o-mini"}),
		)
		resp, err := client.SendMessages(context.Background(),
			[]message.Message{message.NewUserMessage("hi")}, nil)
		if err != nil {
			t.Fatalf("SendMessages: %v", err)
		}
		check(t, resp.Usage)
	})

	t.Run("stream", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(
			func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, "event: response.completed\n"+
					`data: {"type":"response.completed","sequence_number":1,"response":`+
					responsesCachedBody+"}\n\n")
			}))
		defer srv.Close()
		client := NewResponsesLLM(
			WithResponsesAPIKey("test-key"),
			WithResponsesBaseURL(srv.URL),
			WithResponsesModel(llm.Model{APIModel: "gpt-4o-mini"}),
		)
		var final *llm.Response
		for ev := range client.StreamResponse(context.Background(),
			[]message.Message{message.NewUserMessage("hi")}, nil) {
			if ev.Type == types.EventError {
				t.Fatalf("stream error: %v", ev.Error)
			}
			if ev.Type == types.EventComplete {
				final = ev.Response
			}
		}
		if final == nil {
			t.Fatal("stream ended without a complete event")
		}
		check(t, final.Usage)
	})
}

// TestResponsesUsageClampsACachedFigureAboveTheTotal confirms a cached
// figure above the reported total clamps InputTokens to zero rather than
// going negative.
func TestResponsesUsageClampsACachedFigureAboveTheTotal(t *testing.T) {
	body := `{"id":"resp_x","object":"response","status":"completed",` +
		`"output":[{"type":"message","role":"assistant",` +
		`"content":[{"type":"output_text","text":"hi"}]}],` +
		`"usage":{"input_tokens":10,"input_tokens_details":{"cached_tokens":40},"output_tokens":1}}`
	srv := newResponsesServer(t, nil, body)
	defer srv.Close()
	client := NewResponsesLLM(
		WithResponsesAPIKey("test-key"),
		WithResponsesBaseURL(srv.URL),
		WithResponsesModel(llm.Model{APIModel: "gpt-4o-mini"}),
	)
	resp, err := client.SendMessages(context.Background(),
		[]message.Message{message.NewUserMessage("hi")}, nil)
	if err != nil {
		t.Fatalf("SendMessages: %v", err)
	}
	if resp.Usage.InputTokens != 0 {
		t.Errorf(
			"InputTokens = %d, want 0 when cached exceeds the total",
			resp.Usage.InputTokens,
		)
	}
}

// TestResponsesUsageSeparatesCacheWrites: current models bill a cache write
// above the input rate (1.25x on the GPT-6 family), and the API reports it as
// input_tokens_details.cache_write_tokens, another subset of input_tokens. The
// shape is the one the API returned on 2026-10-21 for an 8026-token prompt's
// first call. InputTokens must hold only what was neither read from nor
// written to the cache, so the three add back up to the prompt.
//
// Negative control, observed 2026-10-21 before the mapping read
// cache_write_tokens: "usage = input 8026, cache write 0, cache read 0; want
// 3, 8023, 0".
func TestResponsesUsageSeparatesCacheWrites(t *testing.T) {
	body := `{"id":"resp_w","object":"response","status":"completed",` +
		`"output":[{"type":"message","role":"assistant",` +
		`"content":[{"type":"output_text","text":"hi"}]}],` +
		`"usage":{"input_tokens":8026,"input_tokens_details":{"cache_write_tokens":8023,"cached_tokens":0},` +
		`"output_tokens":29,"output_tokens_details":{"reasoning_tokens":22},"total_tokens":8055}}`
	srv := newResponsesServer(t, nil, body)
	defer srv.Close()
	client := NewResponsesLLM(
		WithResponsesAPIKey("test-key"),
		WithResponsesBaseURL(srv.URL),
		WithResponsesModel(llm.Model{APIModel: "gpt-6-luna"}),
	)
	resp, err := client.SendMessages(context.Background(),
		[]message.Message{message.NewUserMessage("hi")}, nil)
	if err != nil {
		t.Fatalf("SendMessages: %v", err)
	}
	u := resp.Usage
	if u.InputTokens != 3 || u.CacheCreationTokens != 8023 || u.CacheReadTokens != 0 {
		t.Errorf("usage = input %d, cache write %d, cache read %d; want 3, 8023, 0",
			u.InputTokens, u.CacheCreationTokens, u.CacheReadTokens)
	}
}
