package convert

import (
	"strings"
	"testing"

	"github.com/joakimcarlsson/ai/modelsync"
	"github.com/joakimcarlsson/ai/realtime"
)

// doc is a small api.json of invented models. The Claude-shaped entry carries
// what the live feed carries for Claude: no cached-input rate, no input
// modalities, no context window.
const doc = `{"providers":[` +
	`{"id":"openai","name":"OpenAI","models":[` +
	`{"id":"demo-realtime","provider":"openai","name":"Demo Realtime",` +
	`"kind":"realtime","prices":[` +
	`{"metric":"input_tokens","unit":"per_1m_tokens","amount":0.6,` +
	`"currency":"USD","dims":{"modality":"text"}},` +
	`{"metric":"input_tokens","unit":"per_1m_tokens","amount":10,` +
	`"currency":"USD","dims":{"modality":"audio"}},` +
	`{"metric":"output_tokens","unit":"per_1m_tokens","amount":20,` +
	`"currency":"USD","dims":{"modality":"audio"}}]},` +
	`{"id":"demo-chat","provider":"openai","name":"Demo Chat","kind":"chat",` +
	`"prices":[` +
	`{"metric":"input_tokens","unit":"per_1m_tokens","amount":2,` +
	`"currency":"USD"},` +
	`{"metric":"cached_input_tokens","unit":"per_1m_tokens","amount":0.5,` +
	`"currency":"USD"},` +
	`{"metric":"cache_write_tokens","unit":"per_1m_tokens","amount":2.5,` +
	`"currency":"USD"}],` +
	`"attrs":{"state":"active","api_id":"demo-chat-api"},` +
	`"limits":{"context_window":200000,"max_output_tokens":8000},` +
	`"lists":{"features":["reasoning"],"input_modalities":["text","image"]}},` +
	`{"id":"demo-voice","provider":"openai","name":"Demo Voice",` +
	`"kind":"speech","prices":[{"metric":"speech","unit":"per_1m_characters",` +
	`"amount":15,"currency":"USD"}],"attrs":{"api_id":"demo-voice-api"},` +
	`"limits":{"character_limit":4096},` +
	`"lists":{"features":["streaming"],"output_formats":["mp3"]}},` +
	`{"id":"web-search","provider":"openai","name":"Web search","kind":"tool",` +
	`"prices":[` +
	`{"metric":"tool_call","unit":"per_1k_calls","amount":25,` +
	`"currency":"USD","dims":{"detail":"web search preview"}},` +
	`{"metric":"tool_call","unit":"per_1k_calls","amount":10,` +
	`"currency":"USD","dims":{"detail":"Web search (all models)"}}]},` +
	`{"id":"demo-claude","provider":"openai","name":"Demo Claude",` +
	`"kind":"chat","prices":[` +
	`{"metric":"input_tokens","unit":"per_1m_tokens","amount":2,` +
	`"currency":"USD","dims":{"tier":"standard"}},` +
	`{"metric":"cache_write_tokens","unit":"per_1m_tokens","amount":2.5,` +
	`"currency":"USD","dims":{"cache_ttl":"5m","tier":"standard"}}],` +
	`"lists":{"features":["reasoning"]}}]}]}`

func entries(t *testing.T) map[string]modelsync.Entry {
	t.Helper()
	res, err := modelsync.Parse(strings.NewReader(doc))
	if err != nil {
		t.Fatal(err)
	}
	out := map[string]modelsync.Entry{}
	for _, e := range res.Entries {
		out[e.ID] = e
	}
	return out
}

func TestToRealtime(t *testing.T) {
	m := ToRealtime(entries(t)["demo-realtime"])

	if m.CostPer1MTextIn != 0.6 || m.CostPer1MAudioIn != 10 ||
		m.CostPer1MAudioOut != 20 {
		t.Errorf("realtime model = %+v", m)
	}
	if _, ok := m.Rate(realtime.ClassImageInput); ok {
		t.Error("an unpublished class must report no rate")
	}
}

func TestToLLMKeysByAPIIDAndKeepsCacheWriteOutOfCachedOutput(t *testing.T) {
	m := ToLLM(entries(t)["demo-chat"])

	if m.ID != "demo-chat-api" || m.APIModel != "demo-chat-api" {
		t.Errorf("keyed %q / %q, want the API id", m.ID, m.APIModel)
	}
	if m.CostPer1MInCached != 0.5 {
		t.Errorf("CostPer1MInCached = %v, want the cache read rate", m.CostPer1MInCached)
	}
	if m.CostPer1MOutCached != 0 {
		t.Errorf("CostPer1MOutCached = %v, want 0: cache write has its own rate",
			m.CostPer1MOutCached)
	}
	if !m.CanReason || !m.SupportsAttachments {
		t.Errorf("capabilities = %+v", m)
	}
	if m.ContextWindow != 200000 || m.DefaultMaxTokens != 8000 ||
		m.State != "active" || m.Provider != "openai" {
		t.Errorf("model = %+v", m)
	}
}

func TestToLLMLeavesAnUnpublishedCapabilityFalse(t *testing.T) {
	e := entries(t)["demo-claude"]
	m := ToLLM(e)

	if m.SupportsAttachments {
		t.Error("no input modalities are published")
	}
	if _, published := e.SupportsAttachments(); published {
		t.Error("the entry itself must still say the fact is unpublished")
	}
	if m.CostPer1MInCached != 0 {
		t.Errorf("an unpublished cache-read rate read as %v", m.CostPer1MInCached)
	}
}

func TestToLLMDefaultsMaxTokensFromTheContextWindow(t *testing.T) {
	m := ToLLM(modelsync.Entry{
		ID: "p", Limits: map[string]int64{"context_window": 100000},
	})
	if m.DefaultMaxTokens != 8192 || m.CanReason {
		t.Errorf("model = %+v", m)
	}
}

func TestToTTSAndToHostedToolKeyByAPIID(t *testing.T) {
	all := entries(t)

	a := ToTTS(all["demo-voice"])
	if a.ID != "demo-voice-api" || a.APIModel != "demo-voice-api" ||
		a.CostPer1MChars != 15 || a.MaxCharacters != 4096 ||
		!a.SupportsStreaming || len(a.SupportedFormats) != 1 {
		t.Errorf("tts model = %+v", a)
	}

	h := ToHostedTool(all["web-search"])
	if h.ID != "web-search" || h.CostPer1KCalls != 10 {
		t.Errorf("hosted tool = %+v, want the plain 10 rate", h)
	}
}
