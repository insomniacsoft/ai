package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/joakimcarlsson/ai/llm"

	"github.com/joakimcarlsson/ai/message"
	"github.com/openai/openai-go/v3/responses"
)

const responsesOK = `{"id":"resp_1","object":"response","status":"completed",` +
	`"output":[{"type":"message","role":"assistant",` +
	`"content":[{"type":"output_text","text":"hi"}]}],` +
	`"usage":{"input_tokens":1,"output_tokens":1}}`

// TestResponsesWithHTTPClientTransportUsed confirms a client injected via
// WithResponsesHTTPClient handles outgoing requests: the wrapped transport's
// counter increments, proving the SDK default client was replaced.
func TestResponsesWithHTTPClientTransportUsed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, responsesOK)
		}))
	defer srv.Close()

	var n int
	client := NewResponsesLLM(
		WithResponsesAPIKey("test-key"),
		WithResponsesBaseURL(srv.URL),
		WithResponsesModel(llm.Model{APIModel: "gpt-4o-mini"}),
		WithResponsesHTTPClient(&http.Client{
			Transport: countingRT{RoundTripper: http.DefaultTransport, n: &n},
		}),
	)

	if _, err := client.SendMessages(context.Background(),
		[]message.Message{message.NewUserMessage("hi")}, nil); err != nil {
		t.Fatalf("SendMessages: %v", err)
	}

	if n == 0 {
		t.Error("injected transport was not used for the request")
	}
}

// TestUserInputContent_MapsImagesForResponses is the regression for the
// Responses API dropping image content: a user message that carries an image
// part must be emitted as an input_text + input_image content list (so the
// image reaches the model), while a text-only message keeps the plain string
// form. It also pins the detail mapping, since a detail:low request is how a
// vision caller keeps per-call cost bounded.
func TestUserInputContent_MapsImagesForResponses(t *testing.T) {
	// text + image -> a two-part content list carrying the image + detail.
	msg := message.NewUserMessage("what is this?")
	msg.AddImageURL("data:image/jpeg;base64,AAAA", "low")

	got := userInputContent(msg)
	if got.OfString.Valid() {
		t.Errorf("OfString is set, want the content-list form when an image is present")
	}
	if len(got.OfInputItemContentList) != 2 {
		t.Fatalf("content list has %d parts, want 2 (text + image)", len(got.OfInputItemContentList))
	}
	if txt := got.OfInputItemContentList[0].OfInputText; txt == nil || txt.Text != "what is this?" {
		t.Errorf("part[0] = %+v, want an input_text with the question", got.OfInputItemContentList[0])
	}
	img := got.OfInputItemContentList[1].OfInputImage
	if img == nil {
		t.Fatal("part[1] is not an input_image")
	}
	if img.ImageURL.Or("") != "data:image/jpeg;base64,AAAA" {
		t.Errorf("input_image URL = %q, want the data URL", img.ImageURL.Or(""))
	}
	if img.Detail != responses.ResponseInputImageDetailLow {
		t.Errorf("input_image detail = %q, want low", img.Detail)
	}

	// text-only -> the plain string form, no content list.
	textOnly := userInputContent(message.NewUserMessage("hi"))
	if len(textOnly.OfInputItemContentList) != 0 {
		t.Errorf("text-only message produced a %d-part content list, want the string form", len(textOnly.OfInputItemContentList))
	}
	if textOnly.OfString.Or("") != "hi" {
		t.Errorf("text-only OfString = %q, want %q", textOnly.OfString.Or(""), "hi")
	}

	// detail mapping: empty/unknown -> auto; explicit high preserved.
	if d := inputImageDetail(""); d != responses.ResponseInputImageDetailAuto {
		t.Errorf("inputImageDetail(\"\") = %q, want auto", d)
	}
	if d := inputImageDetail("high"); d != responses.ResponseInputImageDetailHigh {
		t.Errorf("inputImageDetail(\"high\") = %q, want high", d)
	}
}

// newResponsesServer returns a test server that captures the request body into
// capture (when non-nil) and replies with the given Responses-API JSON.
// Mirrors newCompletionServer (openai_test.go) for the Responses request shape.
func newResponsesServer(
	t *testing.T,
	capture *map[string]any,
	response string,
) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			if capture != nil {
				raw, _ := io.ReadAll(r.Body)
				_ = json.Unmarshal(raw, capture)
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, response)
		}))
}

// TestWireReasoningEffortLevels_Responses confirms each of the six
// ReasoningEffort levels maps to the expected reasoning.effort wire value on
// the Responses API path. The Responses and chat-completions paths hold two
// separate switches over the same six-case enum (responses.go, openai.go)
// that have to be kept in sync by hand, so this pins the Responses side
// independently of TestWireReasoningEffortLevels in openai_test.go.
func TestWireReasoningEffortLevels_Responses(t *testing.T) {
	tests := []struct {
		name   string
		effort ReasoningEffort
		want   string
	}{
		{"none", ReasoningEffortNone, "none"},
		{"minimal", ReasoningEffortMinimal, "minimal"},
		{"low", ReasoningEffortLow, "low"},
		{"medium", ReasoningEffortMedium, "medium"},
		{"high", ReasoningEffortHigh, "high"},
		{"xhigh", ReasoningEffortXhigh, "xhigh"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var body map[string]any
			srv := newResponsesServer(t, &body, responsesOK)
			defer srv.Close()

			client := NewResponsesLLM(
				WithResponsesAPIKey("test-key"),
				WithResponsesBaseURL(srv.URL),
				WithResponsesModel(llm.Model{APIModel: "gpt-5", CanReason: true}),
				WithResponsesReasoningEffort(tt.effort),
			)

			if _, err := client.SendMessages(context.Background(),
				[]message.Message{message.NewUserMessage("hi")}, nil); err != nil {
				t.Fatalf("SendMessages: %v", err)
			}

			reasoning, ok := body["reasoning"].(map[string]any)
			if !ok {
				t.Fatalf("reasoning = %v (%T), want object",
					body["reasoning"], body["reasoning"])
			}
			got, _ := reasoning["effort"].(string)
			if got != tt.want {
				t.Errorf("reasoning.effort = %v, want %q",
					reasoning["effort"], tt.want)
			}
		})
	}
}

// TestWireReasoningEffortOmittedWhenModelCannotReason_Responses mirrors
// TestWireReasoningEffortOmittedWhenModelCannotReason for the Responses API
// path: the CanReason gate in responsesClient.preparedParams must omit the
// whole reasoning object -- not send it with a zero-value effort -- when the
// model can't reason, for every level.
func TestWireReasoningEffortOmittedWhenModelCannotReason_Responses(t *testing.T) {
	levels := []ReasoningEffort{
		ReasoningEffortNone, ReasoningEffortMinimal, ReasoningEffortLow,
		ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXhigh,
	}

	for _, effort := range levels {
		t.Run(string(effort), func(t *testing.T) {
			var body map[string]any
			srv := newResponsesServer(t, &body, responsesOK)
			defer srv.Close()

			client := NewResponsesLLM(
				WithResponsesAPIKey("test-key"),
				WithResponsesBaseURL(srv.URL),
				WithResponsesModel(llm.Model{APIModel: "gpt-4o-mini", CanReason: false}),
				WithResponsesReasoningEffort(effort),
			)

			if _, err := client.SendMessages(context.Background(),
				[]message.Message{message.NewUserMessage("hi")}, nil); err != nil {
				t.Fatalf("SendMessages: %v", err)
			}

			if _, present := body["reasoning"]; present {
				t.Errorf(
					"reasoning should be omitted when CanReason=false, got %v",
					body["reasoning"],
				)
			}
		})
	}
}

// TestWireReasoningEffortOmittedWhenUnset_Responses confirms that a
// reasoning-capable model with no WithResponsesReasoningEffort call sends no
// reasoning object on the Responses path -- there is no default effort level
// applied on the caller's behalf.
func TestWireReasoningEffortOmittedWhenUnset_Responses(t *testing.T) {
	var body map[string]any
	srv := newResponsesServer(t, &body, responsesOK)
	defer srv.Close()

	client := NewResponsesLLM(
		WithResponsesAPIKey("test-key"),
		WithResponsesBaseURL(srv.URL),
		WithResponsesModel(llm.Model{APIModel: "gpt-5", CanReason: true}),
	)

	if _, err := client.SendMessages(context.Background(),
		[]message.Message{message.NewUserMessage("hi")}, nil); err != nil {
		t.Fatalf("SendMessages: %v", err)
	}

	if _, present := body["reasoning"]; present {
		t.Errorf("reasoning should be omitted when unset, got %v",
			body["reasoning"])
	}
}
