package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/joakimcarlsson/ai/llm"
	"github.com/joakimcarlsson/ai/message"
	"github.com/joakimcarlsson/ai/tool"
)

// stubTool is a no-op BaseTool used to populate the request's tools slice so
// tool-choice assertions have something to attach to.
type stubTool struct{ name string }

func (s stubTool) Info() tool.Info {
	return tool.Info{
		Name:        s.name,
		Description: "d",
		Parameters:  map[string]any{},
	}
}

func (s stubTool) Run(context.Context, tool.Call) (tool.Response, error) {
	return tool.Response{}, nil
}

func optsFrom(opts ...Option) Options {
	o := Options{}
	for _, f := range opts {
		f(&o)
	}
	return o
}

// toolChoiceBody builds a request for the given options and one tool, then
// returns the marshaled request body as a generic map for wire assertions.
func toolChoiceBody(
	t *testing.T,
	tools []tool.BaseTool,
	opts ...Option,
) map[string]any {
	t.Helper()
	c := &Client{options: optsFrom(opts...)}
	params := c.preparedMessages(nil, c.convertTools(tools), nil)
	raw, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal params: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	return body
}

// TestToolChoiceRequired confirms a Required choice maps to {"type":"any"}.
func TestToolChoiceRequired(t *testing.T) {
	body := toolChoiceBody(t,
		[]tool.BaseTool{stubTool{name: "get_weather"}},
		WithToolChoice(llm.ToolChoice{Mode: llm.ToolChoiceRequired}))

	tc, ok := body["tool_choice"].(map[string]any)
	if !ok || tc["type"] != "any" {
		t.Fatalf("tool_choice = %v, want {type:any}", body["tool_choice"])
	}
}

// TestToolChoiceNone confirms a None choice maps to {"type":"none"}.
func TestToolChoiceNone(t *testing.T) {
	body := toolChoiceBody(t,
		[]tool.BaseTool{stubTool{name: "get_weather"}},
		WithToolChoice(llm.ToolChoice{Mode: llm.ToolChoiceNone}))

	tc, ok := body["tool_choice"].(map[string]any)
	if !ok || tc["type"] != "none" {
		t.Fatalf("tool_choice = %v, want {type:none}", body["tool_choice"])
	}
}

// TestToolChoiceSpecific confirms a Specific choice names the tool via
// {"type":"tool","name":...}.
func TestToolChoiceSpecific(t *testing.T) {
	body := toolChoiceBody(t,
		[]tool.BaseTool{stubTool{name: "get_weather"}},
		WithToolChoice(llm.ToolChoice{
			Mode: llm.ToolChoiceSpecific,
			Name: "get_weather",
		}))

	tc, ok := body["tool_choice"].(map[string]any)
	if !ok || tc["type"] != "tool" || tc["name"] != "get_weather" {
		t.Fatalf("tool_choice = %v, want {type:tool,name:get_weather}",
			body["tool_choice"])
	}
}

// TestToolChoiceOmittedWithoutTools confirms no tool_choice is emitted when the
// tools slice is empty.
func TestToolChoiceOmittedWithoutTools(t *testing.T) {
	body := toolChoiceBody(t, nil,
		WithToolChoice(llm.ToolChoice{Mode: llm.ToolChoiceRequired}))

	if _, present := body["tool_choice"]; present {
		t.Errorf("tool_choice should be omitted with no tools, got %v",
			body["tool_choice"])
	}
}

// TestToolChoiceSpecificEmptyNameRejected confirms a Specific choice with no
// name is rejected before any request is sent.
func TestToolChoiceSpecificEmptyNameRejected(t *testing.T) {
	c := &Client{options: optsFrom(
		WithModel(llm.Model{APIModel: "claude"}),
		WithToolChoice(llm.ToolChoice{Mode: llm.ToolChoiceSpecific}),
	)}

	_, err := c.SendMessages(context.Background(),
		[]message.Message{message.NewUserMessage("hi")},
		[]tool.BaseTool{stubTool{name: "get_weather"}})
	if !errors.Is(err, llm.ErrToolChoiceNameRequired) {
		t.Fatalf("expected ErrToolChoiceNameRequired, got %v", err)
	}
}

const messageOK = `{"id":"msg_1","type":"message","role":"assistant",` +
	`"model":"claude","content":[{"type":"text","text":"hi"}],` +
	`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`

// redirectRT rewrites every request to point at the test server's host before
// delegating to the wrapped transport, and counts the requests it handled. The
// Anthropic SDK exposes no base-URL option here, so redirecting at the transport
// is how a test reaches a local httptest server through an injected client.
type redirectRT struct {
	base http.RoundTripper
	host string
	n    *int
}

func (c redirectRT) RoundTrip(r *http.Request) (*http.Response, error) {
	*c.n++
	r.URL.Scheme = "http"
	r.URL.Host = c.host
	return c.base.RoundTrip(r)
}

// TestWithHTTPClientTransportUsed confirms a client injected via WithHTTPClient
// handles outgoing requests: the wrapped transport's counter increments, proving
// the SDK default client was replaced.
func TestWithHTTPClientTransportUsed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, messageOK)
		}))
	defer srv.Close()

	var n int
	client := NewLLM(
		WithAPIKey("test-key"),
		WithModel(llm.Model{APIModel: "claude"}),
		WithHTTPClient(&http.Client{
			Transport: redirectRT{
				base: http.DefaultTransport,
				host: srv.Listener.Addr().String(),
				n:    &n,
			},
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

// thinkingBody marshals the request the options build, with no messages or
// tools, for wire assertions on thinking and effort.
func thinkingBody(t *testing.T, opts ...Option) map[string]any {
	t.Helper()
	return toolChoiceBody(t, nil, opts...)
}

// TestBetweenToolsReplacesAdaptiveThinkingAndKeepsEffort: the option sends
// thinking {"type":"between_tools"} with the effort still set, never
// "disabled"; without it the same options send adaptive thinking.
func TestBetweenToolsReplacesAdaptiveThinkingAndKeepsEffort(t *testing.T) {
	model := llm.Model{APIModel: "claude-sonnet-5-5", CanReason: true}

	body := thinkingBody(t, WithModel(model), WithMaxTokens(1000),
		WithReasoningEffort(ReasoningEffortLow), WithThinkingBetweenTools())
	thinking, _ := body["thinking"].(map[string]any)
	if thinking["type"] != "between_tools" {
		t.Errorf("thinking = %v, want between_tools", body["thinking"])
	}
	if oc, _ := body["output_config"].(map[string]any); oc["effort"] != "low" {
		t.Errorf("output_config = %v, want effort low kept", body["output_config"])
	}

	adaptive := thinkingBody(t, WithModel(model), WithMaxTokens(1000),
		WithReasoningEffort(ReasoningEffortLow))
	if th, _ := adaptive["thinking"].(map[string]any); th["type"] != "adaptive" {
		t.Errorf("without the option thinking = %v, want adaptive", adaptive["thinking"])
	}

	// No effort, or a model without CanReason: the option sends nothing.
	for name, opts := range map[string][]Option{
		"no effort":    {WithModel(model), WithMaxTokens(1000), WithThinkingBetweenTools()},
		"no CanReason": {WithModel(llm.Model{APIModel: "claude-sonnet-5-5"}), WithMaxTokens(1000), WithReasoningEffort(ReasoningEffortLow), WithThinkingBetweenTools()},
	} {
		if th, ok := thinkingBody(t, opts...)["thinking"]; ok {
			t.Errorf("%s: thinking = %v, want none sent", name, th)
		}
	}
}

// TestWithoutEnvironmentIgnoresTheSDKsEnvironmentVariables: with the option,
// ANTHROPIC_BASE_URL cannot redirect a request and ANTHROPIC_API_KEY and
// ANTHROPIC_AUTH_TOKEN cannot supply credentials; only WithAPIKey's key is
// sent. Without it the SDK honours them (the control half).
func TestWithoutEnvironmentIgnoresTheSDKsEnvironmentVariables(t *testing.T) {
	var envHits int
	envSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		envHits++
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, messageOK)
	}))
	defer envSrv.Close()
	t.Setenv("ANTHROPIC_BASE_URL", envSrv.URL)
	t.Setenv("ANTHROPIC_API_KEY", "env-key")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "env-token")

	var got http.Header
	var n int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, messageOK)
	}))
	defer srv.Close()

	// The transport records the host the SDK chose before redirecting it, so
	// the test sees whether ANTHROPIC_BASE_URL was honoured.
	var chosenHost string
	rt := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		n++
		chosenHost = r.URL.Host
		r.URL.Scheme, r.URL.Host = "http", srv.Listener.Addr().String()
		return http.DefaultTransport.RoundTrip(r)
	})
	send := func(opts ...Option) {
		t.Helper()
		c := NewLLM(append([]Option{
			WithAPIKey("explicit-key"),
			WithModel(llm.Model{APIModel: "claude"}),
			WithHTTPClient(&http.Client{Transport: rt}),
		}, opts...)...)
		if _, err := c.SendMessages(context.Background(),
			[]message.Message{message.NewUserMessage("hi")}, nil); err != nil {
			t.Fatalf("SendMessages: %v", err)
		}
	}

	send(WithoutEnvironment())
	if chosenHost != "api.anthropic.com" {
		t.Errorf("with WithoutEnvironment the request was addressed to %q, want api.anthropic.com", chosenHost)
	}
	if got.Get("X-Api-Key") != "explicit-key" || got.Get("Authorization") != "" {
		t.Errorf("credentials sent = x-api-key %q, authorization %q; want only the explicit key",
			got.Get("X-Api-Key"), got.Get("Authorization"))
	}

	send()
	if chosenHost == "api.anthropic.com" {
		t.Errorf("control: without the option ANTHROPIC_BASE_URL was not honoured; the test proves nothing")
	}
	if envHits != 0 || n != 2 {
		t.Errorf("env server hits = %d, requests = %d", envHits, n)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
