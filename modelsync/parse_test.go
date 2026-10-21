package modelsync

import (
	"errors"
	"io"
	"slices"
	"strings"
	"testing"
)

// realClaudeEntry is a model entry copied from the live feed (claude-sonnet-5,
// 2026-10): a Claude model publishes no cached-input rate, no context window,
// no input modalities and no endpoints, and files its cache write once per
// cache lifetime.
const realClaudeEntry = `{"id":"claude-sonnet-5","provider":"anthropic","name":"Claude Sonnet 5","kind":"chat","prices":[{"metric":"cache_write_tokens","unit":"per_1m_tokens","amount":4,"currency":"USD","dims":{"cache_ttl":"1h","tier":"standard"}},{"metric":"cache_write_tokens","unit":"per_1m_tokens","amount":2.5,"currency":"USD","dims":{"cache_ttl":"5m","tier":"standard"}},{"metric":"input_tokens","unit":"per_1m_tokens","amount":1,"currency":"USD","dims":{"tier":"batch"}},{"metric":"input_tokens","unit":"per_1m_tokens","amount":2,"currency":"USD","dims":{"tier":"standard"}},{"metric":"output_tokens","unit":"per_1m_tokens","amount":5,"currency":"USD","dims":{"tier":"batch"}},{"metric":"output_tokens","unit":"per_1m_tokens","amount":10,"currency":"USD","dims":{"tier":"standard"}}],"attrs":{"api_id":"claude-sonnet-5","release_date":"2026-06-30","retirement_date":"2027-06-30","retirement_date_qualifier":"not sooner than","state":"active","summary":"the next generation of our Sonnet model family","thinking_default":"on","thinking_types":"adaptive only"},"limits":{"itpm_tier_build":5000000,"itpm_tier_scale":10000000,"itpm_tier_start":2000000,"max_output_tokens_batch":300000,"otpm_tier_build":1000000,"otpm_tier_scale":2000000,"otpm_tier_start":400000,"rpm_tier_build":5000,"rpm_tier_scale":10000,"rpm_tier_start":1000},"lists":{"effort_levels":["max","xhigh"],"features":["reasoning"]},"sources":["https://platform.claude.com/docs/en/about-claude/model-deprecations.md","https://platform.claude.com/docs/en/about-claude/pricing.md","https://platform.claude.com/docs/en/api/rate-limits.md","https://platform.claude.com/docs/en/build-with-claude/effort.md","https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting.md","https://platform.claude.com/docs/en/release-notes/overview.md"]}`

// fixture is a small api.json: an OpenAI provider with a realtime and a chat
// model, an Anthropic provider holding the real entry above, a synthetic
// provider whose model is keyed by a catalog id that is not its API id and
// publishes a cache-read rate, and an aggregator that files another
// provider's model under its own object.
const fixture = `{"generated":"2026-10-01","providers":[` +
	`{"id":"openai","name":"OpenAI","models":[` +
	`{"id":"gpt-realtime-2.1-mini","provider":"openai",` +
	`"name":"GPT Realtime 2.1 Mini","kind":"realtime","prices":[` +
	`{"metric":"input_tokens","unit":"per_1m_tokens","amount":0.6,` +
	`"currency":"USD","dims":{"modality":"text"}},` +
	`{"metric":"output_tokens","unit":"per_1m_tokens","amount":2.4,` +
	`"currency":"USD","dims":{"modality":"text"}},` +
	`{"metric":"input_tokens","unit":"per_1m_tokens","amount":10,` +
	`"currency":"USD","dims":{"modality":"audio"}},` +
	`{"metric":"output_tokens","unit":"per_1m_tokens","amount":20,` +
	`"currency":"USD","dims":{"modality":"audio"}}]},` +
	`{"id":"demo-chat","provider":"openai","name":"Demo Chat",` +
	`"kind":"chat","attrs":{"state":"active","future_attr":"kept"},` +
	`"limits":{"context_window":200000,"max_output_tokens":8000},` +
	`"lists":{"endpoints":["v1/responses"],"features":["reasoning"],` +
	`"input_modalities":["text","image"],` +
	`"future_list":["a","b"]},"sources":["https://example.test/a"]}]},` +
	`{"id":"anthropic","name":"Anthropic","models":[` + realClaudeEntry + `]},` +
	// Synthetic: not a feed entry.
	`{"id":"synthetic","name":"Synthetic","models":[` +
	`{"id":"synth-sonnet","provider":"synthetic","name":"Synth Sonnet",` +
	`"kind":"chat","prices":[` +
	`{"metric":"input_tokens","unit":"per_1m_tokens","amount":3,` +
	`"currency":"USD","dims":{"tier":"standard"}},` +
	`{"metric":"cached_input_tokens","unit":"per_1m_tokens","amount":0.3,` +
	`"currency":"USD","dims":{"tier":"standard"}}],` +
	`"attrs":{"api_id":"synth-sonnet-api"}}]},` +
	`{"id":"aggregator","name":"Aggregator","models":[` +
	`{"id":"openai/demo-chat","provider":"openai","name":"Demo Chat (via)",` +
	`"kind":"chat"}]}]}`

func parse(t *testing.T, doc string, providers ...string) Result {
	t.Helper()
	res, err := Parse(strings.NewReader(doc), providers...)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	return res
}

func entriesByID(t *testing.T, doc string, providers ...string) map[string]Entry {
	t.Helper()
	byID := map[string]Entry{}
	for _, e := range parse(t, doc, providers...).Entries {
		byID[e.ID] = e
	}
	return byID
}

func TestParseAllowListKeepsOnlyThatProvider(t *testing.T) {
	got := entriesByID(t, fixture, "openai")

	if len(got) != 2 {
		t.Fatalf("got %d entries, want 2: %v", len(got), got)
	}
	for id, e := range got {
		if e.EnclosingProvider != "openai" {
			t.Errorf("%s filed under %q, want openai", id, e.EnclosingProvider)
		}
	}
	if _, ok := got["openai/demo-chat"]; ok {
		t.Error("the aggregator's copy is filed under aggregator, not openai")
	}
}

func TestParseWithoutAllowListKeepsEverythingInOrder(t *testing.T) {
	var ids []string
	for _, e := range parse(t, fixture).Entries {
		ids = append(ids, e.ID)
	}
	want := []string{
		"gpt-realtime-2.1-mini", "demo-chat", "claude-sonnet-5",
		"synth-sonnet", "openai/demo-chat",
	}
	if !slices.Equal(ids, want) {
		t.Errorf("ids = %v, want %v", ids, want)
	}
}

func TestParseKeepsEveryField(t *testing.T) {
	e := entriesByID(t, fixture, "openai")["demo-chat"]

	if e.Name != "Demo Chat" || e.Kind != "chat" {
		t.Errorf("name/kind = %q/%q", e.Name, e.Kind)
	}
	if e.Attrs["future_attr"] != "kept" {
		t.Errorf("an attr the module has no field for is lost: %v", e.Attrs)
	}
	if !slices.Equal(e.Lists["future_list"], []string{"a", "b"}) {
		t.Errorf("a list the module has no field for is lost: %v", e.Lists)
	}
	if e.Limits["context_window"] != 200000 ||
		e.Limits["max_output_tokens"] != 8000 {
		t.Errorf("limits = %v", e.Limits)
	}
	if !slices.Equal(e.Lists["endpoints"], []string{"v1/responses"}) {
		t.Errorf("endpoints = %v", e.Lists["endpoints"])
	}
	if !slices.Equal(e.Sources, []string{"https://example.test/a"}) {
		t.Errorf("sources = %v", e.Sources)
	}

	claude := entriesByID(t, fixture, "anthropic")["claude-sonnet-5"]
	if claude.Attrs["retirement_date_qualifier"] != "not sooner than" ||
		claude.Attrs["thinking_types"] != "adaptive only" ||
		!slices.Equal(claude.Lists["effort_levels"], []string{"max", "xhigh"}) {
		t.Errorf("attrs/lists of the real entry = %v / %v",
			claude.Attrs, claude.Lists)
	}
}

func TestParseKeysByAPIIDWhenItDiffersFromID(t *testing.T) {
	e := entriesByID(t, fixture, "synthetic")["synth-sonnet"]

	if e.ID != "synth-sonnet" {
		t.Fatalf("ID = %q", e.ID)
	}
	if got := e.APIID(); got != "synth-sonnet-api" {
		t.Errorf("APIID = %q, want synth-sonnet-api", got)
	}
	if got := entriesByID(t, fixture, "openai")["demo-chat"].APIID(); got !=
		"demo-chat" {
		t.Errorf("without attrs.api_id the id is used, got %q", got)
	}
}

func TestParseReportsEnclosingAndOwnProvider(t *testing.T) {
	e := entriesByID(t, fixture, "aggregator")["openai/demo-chat"]

	if e.EnclosingProvider != "aggregator" || e.Provider != "openai" {
		t.Errorf(
			"enclosing/own = %q/%q, want aggregator/openai",
			e.EnclosingProvider, e.Provider,
		)
	}
	if e.EnclosingProviderName != "Aggregator" {
		t.Errorf("enclosing name = %q", e.EnclosingProviderName)
	}
}

func TestParseReadsModelsListedBeforeTheirProviderIdentity(t *testing.T) {
	doc := `{"providers":[{"models":[{"id":"m","kind":"chat"}],` +
		`"id":"late","name":"Late"},` +
		`{"models":[{"id":"x","kind":"chat"}],"id":"other","name":"O"}]}`

	res := parse(t, doc, "late")
	if len(res.Entries) != 1 || res.Entries[0].ID != "m" ||
		res.Entries[0].EnclosingProvider != "late" {
		t.Errorf("got %+v", res.Entries)
	}
}

func TestParseDecidesWantedOnTheProviderIDAlone(t *testing.T) {
	// The name comes after the models, so it is not known when they are
	// read; the models are still kept.
	doc := `{"providers":[{"id":"p","models":[{"id":"m"}],"name":"P"}]}`

	res := parse(t, doc, "p")
	if len(res.Entries) != 1 || res.Entries[0].EnclosingProviderName != "" {
		t.Errorf("got %+v", res.Entries)
	}
}

func TestParseSkipsUnknownDocumentKeys(t *testing.T) {
	doc := `{"meta":{"a":[1,2,{"b":null}]},"providers":[{"id":"p","name":"P",` +
		`"extra":[{"x":1}],"models":[{"id":"m","kind":"chat"}]}],"tail":3}`

	if res := parse(t, doc, "p"); len(res.Entries) != 1 {
		t.Fatalf("got %+v", res)
	}
}

func TestParseTruncatedDocumentReturnsAnErrorNotAPartialResult(t *testing.T) {
	// Every proper prefix of the document is cut short somewhere: inside a
	// string, a price, a model list, a provider, or just before the closing
	// brace.
	for n := range len(fixture) {
		res, err := Parse(strings.NewReader(fixture[:n]))
		if !errors.Is(err, io.ErrUnexpectedEOF) {
			t.Fatalf("prefix of %d bytes: err = %v (entries %v), want %v",
				n, err, res.Entries, io.ErrUnexpectedEOF)
		}
		if res.Entries != nil || res.EntryErrors != nil ||
			res.ProvidersSeen != nil {
			t.Fatalf("prefix of %d bytes returned a partial result", n)
		}
	}
	parse(t, fixture)
}

func TestParseRejectsADocumentThatIsNotJSON(t *testing.T) {
	if _, err := Parse(strings.NewReader("<html>nope")); err == nil {
		t.Error("expected an error for an HTML body")
	}
}

func TestParseRejectsADocumentWithNoProvidersKey(t *testing.T) {
	for _, doc := range []string{`{}`, `{"models":[]}`, `null`} {
		_, err := Parse(strings.NewReader(doc))
		if !errors.Is(err, ErrNoProviders) {
			t.Errorf("%s: err = %v, want ErrNoProviders", doc, err)
		}
	}
	// An empty provider list is a document that says there are none.
	if res := parse(t, `{"providers":[]}`); len(res.Entries) != 0 {
		t.Errorf("got %+v", res)
	}
}

func TestParseRejectsDataAfterTheDocument(t *testing.T) {
	for _, tail := range []string{`{}`, `x`, `,`, `]`, `1`} {
		if _, err := Parse(strings.NewReader(fixture + tail)); err == nil {
			t.Errorf("data after the document %q was accepted", tail)
		}
	}
	if _, err := Parse(strings.NewReader(fixture + " \n\t")); err != nil {
		t.Errorf("trailing whitespace: %v", err)
	}
}

func TestParseToleratesWhatEncodingJSONTolerates(t *testing.T) {
	// Keys match without regard to case, and null reads as empty.
	doc := `{"Providers":[null,{"ID":"p","NAME":"P","Models":[` +
		`{"ID":"m","Kind":"chat"}]},{"id":"q","models":null}]}`

	res := parse(t, doc)
	if len(res.Entries) != 1 || res.Entries[0].ID != "m" ||
		res.Entries[0].EnclosingProviderName != "P" {
		t.Errorf("got %+v", res.Entries)
	}
	if !slices.Equal(res.ProvidersSeen, []string{"p", "q"}) {
		t.Errorf("seen = %v", res.ProvidersSeen)
	}
	if res := parse(t, `{"providers":null}`); len(res.Entries) != 0 {
		t.Errorf("got %+v", res)
	}
}

// badEntry is badly typed: a limit that is not an integer and an attr that is
// not a string. The document around it is well formed.
const badEntry = `{"id":"broken","kind":"chat","attrs":{"api_id":"broken-api",` +
	`"x":true},"limits":{"context_window":1.5}}`

func TestParseSkipsAnUnwantedProviderWithoutDecodingIt(t *testing.T) {
	doc := `{"providers":[` +
		`{"id":"unwanted","name":"U","models":[` + badEntry + `]},` +
		`{"id":"wanted","name":"W","models":[` + badEntry + `]}]}`

	res := parse(t, doc, "wanted")
	if len(res.EntryErrors) != 1 || res.EntryErrors[0].Provider != "wanted" {
		t.Errorf("entry errors = %+v, want one, under wanted", res.EntryErrors)
	}

	res = parse(t, doc, "unwanted")
	if len(res.EntryErrors) != 1 || res.EntryErrors[0].Provider != "unwanted" {
		t.Errorf("entry errors = %+v, want one, under unwanted", res.EntryErrors)
	}

	// The same entry under a provider nobody asked for is not even looked at.
	res = parse(t, doc, "somebody-else")
	if len(res.EntryErrors) != 0 || len(res.Entries) != 0 {
		t.Errorf("got %+v", res)
	}
}

func TestParseReportsABadEntryAndKeepsItsSiblings(t *testing.T) {
	doc := `{"providers":[{"id":"p","name":"P","models":[` +
		`{"id":"first","kind":"chat"},` + badEntry + `,` +
		`{"id":"third","kind":"chat"}]}]}`

	res := parse(t, doc, "p")

	var ids []string
	for _, e := range res.Entries {
		ids = append(ids, e.ID)
	}
	if !slices.Equal(ids, []string{"first", "third"}) {
		t.Errorf("entries = %v, want first and third", ids)
	}
	if len(res.EntryErrors) != 1 {
		t.Fatalf("entry errors = %+v", res.EntryErrors)
	}
	bad := res.EntryErrors[0]
	if bad.Provider != "p" || bad.Index != 1 || bad.APIID != "broken-api" ||
		bad.Err == nil {
		t.Errorf("entry error = %+v", bad)
	}
}

func TestParseReportsWhichProvidersTheDocumentHas(t *testing.T) {
	doc := `{"providers":[` +
		`{"id":"openai","name":"O","models":[{"id":"a"}]},` +
		`{"id":"anthropic","name":"A","models":[]},` +
		`{"id":"other","name":"X","models":[{"id":"b"}]}]}`

	res := parse(t, doc, "openai", "anthropic", "google")
	want := []string{"openai", "anthropic", "other"}
	if !slices.Equal(res.ProvidersSeen, want) {
		t.Errorf("seen = %v, want %v", res.ProvidersSeen, want)
	}
	// anthropic is in the document with no models; google is not in it.
	if !slices.Contains(res.ProvidersSeen, "anthropic") ||
		slices.Contains(res.ProvidersSeen, "google") {
		t.Errorf("seen = %v", res.ProvidersSeen)
	}
	if len(res.Entries) != 1 {
		t.Errorf("entries = %+v", res.Entries)
	}
}

func TestWalkStopsOnTheCallbacksError(t *testing.T) {
	stop := errors.New("stop")
	calls := 0
	err := Walk(strings.NewReader(fixture), nil, Visitor{
		Entry: func(Entry) error {
			calls++
			return stop
		},
	})
	if !errors.Is(err, stop) || calls != 1 {
		t.Errorf("calls = %d, err = %v", calls, err)
	}
}

func TestParseSkipsANullEntryAsNoModel(t *testing.T) {
	doc := `{"providers":[{"id":"openai","models":[null,{"id":"a","provider":"openai","kind":"chat"}]}]}`
	res, err := Parse(strings.NewReader(doc), "openai")
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Entries) != 1 || len(res.EntryErrors) != 0 {
		t.Errorf("entries %d, errors %v; want 1 entry and no errors", len(res.Entries), res.EntryErrors)
	}
}

func TestParseSkipsAProviderWhoseIDIsNotAString(t *testing.T) {
	doc := `{"providers":[` +
		`{"id":7,"name":false,"models":[{"id":"x","provider":"x","kind":"chat"}]},` +
		`{"id":"openai","name":3,"models":[{"id":"a","provider":"openai","kind":"chat"}]}]}`
	for _, allow := range [][]string{{"openai"}, nil} {
		res, err := Parse(strings.NewReader(doc), allow...)
		if err != nil {
			t.Fatalf("allow %v: %v", allow, err)
		}
		if len(res.Entries) != 1 || res.Entries[0].APIID() != "a" {
			t.Errorf("allow %v: entries %+v, want only openai's a", allow, res.Entries)
		}
	}
}
