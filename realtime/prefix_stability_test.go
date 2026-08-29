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

func (s stubTool) Info() tool.Info { return s.info }
func (s stubTool) Run(context.Context, tool.Call) (tool.Response, error) {
	return tool.Response{}, nil
}

func namedTool(name, desc string, props map[string]any, required ...string) stubTool {
	return stubTool{info: tool.Info{
		Name: name, Description: desc, Parameters: props, Required: required,
	}}
}

func sampleConfig(entityDesc string) SessionConfig {
	return SessionConfig{
		Model:        "gpt-realtime",
		Instructions: "You are a voice assistant.\n\n<<<DATA>>>\narea: Kitchen\n<<<END>>>",
		Voice:        "cedar",
		Tools: []tool.BaseTool{
			namedTool("home_state", "Reads one entity.", map[string]any{
				"entity_id": map[string]any{"type": "string", "description": entityDesc},
			}, "entity_id"),
			namedTool("call_service", "Calls a service.", map[string]any{
				"domain": map[string]any{"type": "string", "description": "The domain."},
			}, "domain"),
		},
		MaxOutputTokens: 2400,
		RetentionRatio:  0.8,
	}
}

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

// TestTwoSessionsOverAnUnchangedHouseSerializeIdentically.
//
// Prompt caching needs a BYTE-identical prefix. Anything that reorders between
// two runs — a map iterated straight into the payload, a set rendered without
// sorting — silently costs the whole prefix at the uncached rate on every
// session, and nothing in the output looks wrong. Measured 2026-08-24, uncached
// text input was 81% of the day's spend.
//
// Run repeatedly, because Go randomises map iteration per run rather than per
// call: a single comparison can pass on an unstable payload by luck.
func TestTwoSessionsOverAnUnchangedHouseSerializeIdentically(t *testing.T) {
	want := marshalParams(t, sampleConfig("The entity id."))
	for i := 0; i < 50; i++ {
		if got := marshalParams(t, sampleConfig("The entity id.")); got != want {
			t.Fatalf("two builds from identical input differ on run %d:\n%s\n%s", i, want, got)
		}
	}
}

// TestAChangedHouseChangesThePayload.
//
// Negative control for the test above. A comparison of two payloads that are
// equal because the tool schemas never reached the payload at all would pass
// it perfectly, and would be exactly the bug that matters.
func TestAChangedHouseChangesThePayload(t *testing.T) {
	before := marshalParams(t, sampleConfig("The entity id."))
	after := marshalParams(t, sampleConfig("The entity id, e.g. light.office_lamp."))
	if before == after {
		t.Fatal("changing a parameter description did not change the payload; " +
			"the stability assertion is vacuous because the schemas are not in it")
	}

	changed := sampleConfig("The entity id.")
	changed.Instructions += "\narea: Hallway / hol"
	if marshalParams(t, changed) == before {
		t.Fatal("changing the domain terms did not change the payload")
	}
}

// TestTheSessionCarriesARetentionRatioBelowOne.
//
// The provider's default is 1.0, which trims the conversation to exactly the
// limit so the next turn overruns it again — truncating continuously, and
// paying for the whole prefix uncached each time. A ratio below 1.0 drops a
// block at once and leaves headroom.
func TestTheSessionCarriesARetentionRatioBelowOne(t *testing.T) {
	payload := marshalParams(t, sampleConfig("The entity id."))

	var got map[string]any
	if err := json.Unmarshal([]byte(payload), &got); err != nil {
		t.Fatalf("unmarshalling the payload: %v", err)
	}
	trunc, ok := got["truncation"].(map[string]any)
	if !ok {
		t.Fatalf("the session payload carries no truncation strategy: %s", payload)
	}
	if trunc["type"] != "retention_ratio" {
		t.Fatalf("truncation type = %v, want retention_ratio", trunc["type"])
	}
	ratio, ok := trunc["retention_ratio"].(float64)
	if !ok {
		t.Fatalf("retention_ratio is %T, not a number", trunc["retention_ratio"])
	}
	if !(ratio > 0 && ratio < 1) {
		t.Fatalf("retention_ratio = %v; 1.0 is the provider default and the expensive one", ratio)
	}
}

// TestAnUnsetRetentionRatioIsOmittedEntirely.
//
// Negative control on the wiring: a field that serialises as 0 rather than
// being omitted would tell the provider to retain NOTHING, discarding the
// conversation after every turn. Leaving it unset must mean "the provider's
// default", not "zero".
func TestAnUnsetRetentionRatioIsOmittedEntirely(t *testing.T) {
	c := sampleConfig("The entity id.")
	c.RetentionRatio = 0
	payload := marshalParams(t, c)

	var got map[string]any
	if err := json.Unmarshal([]byte(payload), &got); err != nil {
		t.Fatalf("unmarshalling the payload: %v", err)
	}
	if _, present := got["truncation"]; present {
		t.Fatalf("an unset ratio still sent a truncation strategy, which would discard "+
			"the conversation every turn: %s", payload)
	}
}
