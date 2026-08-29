package realtime

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/joakimcarlsson/ai/tool"
)

// stubTool is a minimal BaseTool whose Info is fixed at construction.
type stubTool struct {
	info tool.Info
}

// Info returns the tool's fixed metadata.
func (s stubTool) Info() tool.Info { return s.info }

// Run is never called by these tests; only Info is exercised.
func (s stubTool) Run(context.Context, tool.Call) (tool.Response, error) {
	return tool.Response{}, nil
}

// namedTool builds a stubTool with the given schema.
func namedTool(
	name, desc string,
	props map[string]any,
	required ...string,
) stubTool {
	return stubTool{info: tool.Info{
		Name: name, Description: desc, Parameters: props, Required: required,
	}}
}

// sampleConfig is a SessionConfig with two tools and instructions carrying
// embedded data, varying only in cityDesc, for comparing serialised output.
func sampleConfig(cityDesc string) SessionConfig {
	return SessionConfig{
		Model:        "gpt-realtime",
		Instructions: "You are a voice assistant.\n\n<<<DATA>>>\nregion: north\n<<<END>>>",
		Voice:        "cedar",
		Tools: []tool.BaseTool{
			namedTool(
				"get_weather",
				"Reports the current weather.",
				map[string]any{
					"city": map[string]any{
						"type":        "string",
						"description": cityDesc,
					},
				},
				"city",
			),
			namedTool("send_email", "Sends an email.", map[string]any{
				"to": map[string]any{
					"type":        "string",
					"description": "The recipient.",
				},
			}, "to"),
		},
		MaxOutputTokens: 2400,
		RetentionRatio:  0.8,
	}
}

// marshalParams renders c's session params and marshals them to JSON.
func marshalParams(t *testing.T, c SessionConfig) string {
	t.Helper()
	p, err := c.sessionParams()
	if err != nil {
		t.Fatalf("sessionParams() error = %v", err)
	}
	b, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshalling session params: %v", err)
	}
	return string(b)
}

// Prompt caching needs a byte-identical prefix; an unstable payload (a map
// iterated straight into JSON, a set rendered unsorted) silently costs the
// whole prefix at the uncached rate every session. Run repeatedly because Go
// randomises map iteration per run, not per call.
func TestTwoSessionsWithTheSameInputSerializeIdentically(t *testing.T) {
	want := marshalParams(t, sampleConfig("The city."))
	for i := range 50 {
		if got := marshalParams(t, sampleConfig("The city.")); got != want {
			t.Fatalf(
				"two builds from identical input differ on run %d:\n%s\n%s",
				i,
				want,
				got,
			)
		}
	}
}

// Negative control for the test above: a comparison of two payloads that are
// equal because the tool schemas never reached the payload at all would pass
// it perfectly, and would be exactly the bug that matters.
func TestAChangedInputChangesThePayload(t *testing.T) {
	before := marshalParams(t, sampleConfig("The city."))
	after := marshalParams(t, sampleConfig("The city, e.g. Stockholm."))
	if before == after {
		t.Fatal(
			"changing a parameter description did not change the payload; " +
				"the stability assertion is vacuous because the schemas are not in it",
		)
	}

	changed := sampleConfig("The city.")
	changed.Instructions += "\nregion: south"
	if marshalParams(t, changed) == before {
		t.Fatal("changing the instructions did not change the payload")
	}
}

// The provider's default retention ratio is 1.0, which trims the
// conversation to exactly the limit so the next turn overruns it again,
// paying for the whole prefix uncached every time.
func TestTheSessionCarriesARetentionRatioBelowOne(t *testing.T) {
	payload := marshalParams(t, sampleConfig("The city."))

	var got map[string]any
	if err := json.Unmarshal([]byte(payload), &got); err != nil {
		t.Fatalf("unmarshalling the payload: %v", err)
	}
	trunc, ok := got["truncation"].(map[string]any)
	if !ok {
		t.Fatalf(
			"the session payload carries no truncation strategy: %s",
			payload,
		)
	}
	if trunc["type"] != "retention_ratio" {
		t.Fatalf("truncation type = %v, want retention_ratio", trunc["type"])
	}
	ratio, ok := trunc["retention_ratio"].(float64)
	if !ok {
		t.Fatalf(
			"retention_ratio is %T, not a number",
			trunc["retention_ratio"],
		)
	}
	if !(ratio > 0 && ratio < 1) {
		t.Fatalf(
			"retention_ratio = %v; 1.0 is the provider default and the expensive one",
			ratio,
		)
	}
}

// Negative control on the wiring: a field that serialises as 0 rather than
// being omitted would tell the provider to retain nothing, discarding the
// conversation after every turn.
func TestAnUnsetRetentionRatioIsOmittedEntirely(t *testing.T) {
	c := sampleConfig("The city.")
	c.RetentionRatio = 0
	payload := marshalParams(t, c)

	var got map[string]any
	if err := json.Unmarshal([]byte(payload), &got); err != nil {
		t.Fatalf("unmarshalling the payload: %v", err)
	}
	if _, present := got["truncation"]; present {
		t.Fatalf(
			"an unset ratio still sent a truncation strategy, which would discard "+
				"the conversation every turn: %s",
			payload,
		)
	}
}
