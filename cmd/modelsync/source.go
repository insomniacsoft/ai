package main

import (
	"context"
	"fmt"
	"maps"
	"net/http"
	"slices"
	"strings"

	"github.com/joakimcarlsson/ai/modelsync"
)

const (
	// sourceRepo is the catalog this tool is generated from.
	sourceRepo = "https://github.com/JoakimCarlsson/model-sync"
	// sourceURL is the raw api.json that repository publishes.
	sourceURL = "https://raw.githubusercontent.com/JoakimCarlsson/" +
		"model-sync/main/api.json"
)

// apiModel and apiPrice are the module's types: the generator reads the
// document through the same parser and the same rate selection an application
// refreshing its catalog at run time uses, so the two cannot disagree about
// what a model costs.
type (
	apiModel = modelsync.Entry
	apiPrice = modelsync.Price
)

// apiIndex is every model the model-sync api.json document lists, in document
// order.
type apiIndex struct {
	entries []apiModel
}

func fetchIndex(ctx context.Context, url string) (*apiIndex, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetching %s: %w", url, err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf(
			"fetching %s: unexpected status %s",
			url,
			resp.Status,
		)
	}

	res, err := modelsync.Parse(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("decoding %s: %w", url, err)
	}
	// A model the parser could not read would be absent from the catalog
	// and reported as removed, so the run stops rather than guess.
	if len(res.EntryErrors) > 0 {
		return nil, fmt.Errorf(
			"decoding %s: %d unreadable model(s), first: %w",
			url, len(res.EntryErrors), res.EntryErrors[0],
		)
	}
	return &apiIndex{entries: res.Entries}, nil
}

// models returns the models one provider publishes for one kind.
func (idx *apiIndex) models(provider, kind string) []apiModel {
	var out []apiModel
	for _, m := range idx.entries {
		if m.EnclosingProvider == provider && m.Kind == kind {
			out = append(out, m)
		}
	}
	return out
}

// The unit tables and rate selection live in the module; these names keep the
// generator's own code and tests reading as before.
var (
	tokenUnits  = modelsync.TokenUnits
	charUnits   = modelsync.CharUnits
	minuteUnits = modelsync.MinuteUnits
	imageUnits  = modelsync.ImageUnits
)

func rate(
	prices []apiPrice,
	currency string,
	units map[string]float64,
	metrics ...string,
) (float64, bool) {
	return modelsync.SelectRate(prices, currency, units, metrics...)
}

func withDim(prices []apiPrice, dim, want string) []apiPrice {
	return modelsync.WithDim(prices, dim, want)
}

// entry converts one source model into the Go literals a catalog entry of the
// target's kind is written with.
//
// Only fields the source actually describes are set. A field it says nothing
// about is left out, so the value the catalog already holds survives the
// regeneration, and target defaults fill it for a model seen for the first
// time.
func modelFor(t target, m apiModel) model {
	currency := m.Currency()
	fields := map[string]string{
		"Provider": quote(t.provider),
		"APIModel": quote(m.APIID()),
		"Currency": quote(currency),
	}
	seed := map[string]string{"Name": quote(t.displayName(m.Name))}

	switch t.kind {
	case kindChat:
		chatFieldsFor(m, currency, fields, seed)
	case kindImage:
		imageFieldsFor(m, currency, fields, seed)
	case kindSpeech:
		speechFieldsFor(m, currency, fields)
	case kindTranscription:
		transcriptionFieldsFor(m, currency, fields)
	case kindTool:
		toolFieldsFor(m, currency, fields)
	case kindEmbedding:
		embeddingFieldsFor(m, currency, fields)
	case kindRerank:
		rerankFieldsFor(m, currency, fields)
	}

	return model{
		apiModel: m.APIID(),
		fields:   fields,
		seed:     seed,
		owned:    sourceOwned[t.kind],
	}
}

func chatFieldsFor(
	m apiModel,
	currency string,
	fields, seed map[string]string,
) {
	// The compiled-in llm.Model has no cache-write field, so the catalogs
	// keep recording the cache-write rate in CostPer1MOutCached as they
	// always have; the module reports it as the separate rate it is.
	rates := m.ChatRates()
	setPublished(fields, "CostPer1MIn", rates.Input)
	setPublished(fields, "CostPer1MOut", rates.Output)
	setPublished(fields, "CostPer1MInCached", rates.CacheRead)
	setPublished(fields, "CostPer1MOutCached", rates.CacheWrite)

	window := m.ContextWindow()
	setInt(fields, "ContextWindow", window)

	if out := m.MaxOutputTokens(); out > 0 {
		fields["DefaultMaxTokens"] = integer(out)
	} else if window > 0 {
		seed["DefaultMaxTokens"] = integer(min(window/4, 8192))
	}

	// The compiled-in catalogs have no unknown state, so a capability the
	// source does not publish is recorded as false.
	reasons, _ := m.CanReason()
	attachments, _ := m.SupportsAttachments()
	fields["CanReason"] = boolean(reasons)
	fields["SupportsAttachments"] = boolean(attachments)
	fields["SupportsStructuredOut"] = boolean(m.Feature("structured_outputs"))
	fields["SupportsImageGeneration"] = boolean(m.EmitsModality("image"))

	for field, attr := range lifecycleAttrs {
		setAttr(fields, field, m.Attrs[attr])
	}
}

// lifecycleAttrs maps each chat lifecycle field to the api.json attribute it
// is read from.
var lifecycleAttrs = map[string]string{
	"State":          "state",
	"ReleaseDate":    "release_date",
	"LastUpdated":    "last_updated",
	"RetirementDate": "retirement_date",
	"ReplacedBy":     "recommended_replacement",
}

// sourceOwned lists, per kind, the fields only the source can know. An
// existing entry never keeps its own value for one of these: when the source
// stops publishing it, the field is cleared rather than carried over, so a
// withdrawn retirement date or replacement does not survive forever, and an
// alias paired with a dated snapshot does not inherit the snapshot's dates.
var sourceOwned = map[kind][]string{
	kindChat: slices.Sorted(maps.Keys(lifecycleAttrs)),
}

// setAttr writes a string field when the source publishes one, and leaves it
// out when it does not. An absent field is a different fact from one the
// source published as empty: the lifecycle attributes are read straight
// through rather than interpreted, and an empty State would read as "active"
// rather than "the provider says nothing".
func setAttr(fields map[string]string, field, value string) {
	if value == "" {
		return
	}
	fields[field] = quote(value)
}

func imageFieldsFor(
	m apiModel,
	currency string,
	fields, seed map[string]string,
) {
	if v, ok := rate(
		m.Prices, currency, imageUnits, "image_output", "usage",
	); ok && v > 0 {
		seed["Pricing"] = imagePricing(v)
	}

	setInt(fields, "MaxPromptTokens", m.Limit(
		"max_input_tokens",
		"context_window",
		"top_provider_context_window",
	))
	setStrings(fields, "SupportedSizes", m.List("sizes", "image_sizes"))
	setStrings(fields, "SupportedAspectRatios", m.Lists["aspect_ratios"])
	setStrings(fields, "SupportedQualities", m.Lists["qualities"])
	fields["SupportsStreaming"] = boolean(m.Feature("streaming"))
}

func speechFieldsFor(m apiModel, currency string, fields map[string]string) {
	setPublished(fields, "CostPer1MChars", m.SpeechRate())
	setInt(fields, "MaxCharacters", m.Limit("character_limit"))
	setStrings(fields, "SupportedFormats", m.Lists["output_formats"])
	fields["SupportsStreaming"] = boolean(m.Feature("streaming"))
}

// toolFieldsFor reads what one thousand invocations of a hosted tool cost;
// see modelsync.Entry.ToolRate for how a tool's several standard rates are
// told apart.
func toolFieldsFor(m apiModel, currency string, fields map[string]string) {
	setPublished(fields, "CostPer1KCalls", m.ToolRate())
}

func transcriptionFieldsFor(
	m apiModel,
	currency string,
	fields map[string]string,
) {
	in := withDim(m.Prices, "direction", "input")
	if !setRate(fields, "CostPer1MIn", in, currency, tokenUnits,
		"input_tokens", "usage") {
		setRate(fields, "CostPer1MIn", in, currency, minuteUnits,
			"audio", "audio_input", "usage")
	}

	out := withDim(m.Prices, "direction", "output")
	if !setRate(fields, "CostPer1MOut", out, currency, tokenUnits,
		"output_tokens") {
		setRate(fields, "CostPer1MOut", out, currency, minuteUnits,
			"output_tokens")
	}

	setInt(fields, "MaxFileSizeMB", megabytes(m.Attrs["max_file_size"]))
	setStrings(fields, "SupportedFormats", m.Lists["audio_formats"])
	setStrings(fields, "SupportedResponseFormats", m.Lists["response_formats"])
	fields["SupportsTimestamps"] = boolean(
		m.AnyFeature("timestamps", "word_timestamps", "utterances"),
	)
	fields["SupportsWordTimestamps"] = boolean(m.Feature("word_timestamps"))
	fields["SupportsDiarization"] = boolean(
		m.AnyFeature("speaker_diarization", "diarization"),
	)
	fields["SupportsTranslation"] = boolean(
		m.AnyFeature("translation", "live_translation"),
	)
	fields["SupportsStreaming"] = boolean(
		m.AnyFeature("streaming", "realtime"),
	)
}

func embeddingFieldsFor(
	m apiModel,
	currency string,
	fields map[string]string,
) {
	setRate(fields, "CostPer1MTokens", m.Prices, currency, tokenUnits,
		"input_tokens", "usage")
	setInt(fields, "MaxInputTokens", m.Limit(
		"max_tokens_per_request",
		"context_window",
		"top_provider_context_window",
	))
	setInt(fields, "EmbeddingDims", number(
		m.Attrs["default_embedding_dimension"],
	))
	setInts(fields, "SupportedDimensions", m.Lists["embedding_dimensions"])
	setInt(fields, "MaxBatchSize", m.Limit(
		"max_inputs_per_request",
		"max_texts_per_call",
	))
	fields["SupportsOutputDtype"] = boolean(
		len(m.Lists["output_dtypes"]) > 0,
	)
}

func rerankFieldsFor(m apiModel, currency string, fields map[string]string) {
	setRate(fields, "CostPer1MTokens", m.Prices, currency, tokenUnits,
		"input_tokens", "usage")
	setInt(fields, "MaxQueryTokens", m.Limit("max_query_tokens"))
	setInt(fields, "MaxTotalTokens", m.Limit(
		"context_window",
		"max_tokens_per_request",
		"top_provider_context_window",
	))
}

func setRate(
	fields map[string]string,
	field string,
	prices []apiPrice,
	currency string,
	units map[string]float64,
	metrics ...string,
) bool {
	v, ok := rate(prices, currency, units, metrics...)
	if !ok {
		return false
	}
	fields[field] = amount(v)
	return true
}

// setPublished writes a selected rate when the source publishes one.
func setPublished(fields map[string]string, field string, r modelsync.Rate) {
	if r.Published {
		fields[field] = amount(r.Per)
	}
}

func setInt(fields map[string]string, field string, v int64) {
	if v > 0 {
		fields[field] = integer(v)
	}
}

func setStrings(fields map[string]string, field string, vals []string) {
	if len(vals) == 0 {
		return
	}
	quoted := make([]string, 0, len(vals))
	for _, v := range vals {
		quoted = append(quoted, quote(v))
	}
	fields[field] = slice("[]string", quoted)
}

// setInts writes a list the source publishes as strings as a Go []int, in
// ascending order rather than the lexical order api.json sorts it in.
func setInts(fields map[string]string, field string, vals []string) {
	nums := make([]int64, 0, len(vals))
	for _, v := range vals {
		if n := number(v); n > 0 {
			nums = append(nums, n)
		}
	}
	if len(nums) == 0 {
		return
	}
	slices.Sort(nums)

	rendered := make([]string, 0, len(nums))
	for _, n := range nums {
		rendered = append(rendered, integer(n))
	}
	fields[field] = slice("[]int", rendered)
}

// sliceWidth is the longest single-line slice literal a catalog is written
// with. A longer one is broken over one element per line, so the 80 column
// limit the repository formats to never has to break it afterwards.
const sliceWidth = 40

func slice(typeExpr string, elems []string) string {
	body := strings.Join(elems, ", ")
	if len(body) <= sliceWidth {
		return typeExpr + "{" + body + "}"
	}
	return typeExpr + "{\n" + strings.Join(elems, ",\n") + ",\n}"
}

// imagePricing renders a flat per-image rate. The source publishes one rate
// per model rather than a size and quality table, so the table an entry holds
// is only ever seeded, never overwritten.
func imagePricing(perImage float64) string {
	return "map[string]map[string]float64{\n\"default\": {\"default\": " +
		amount(perImage) + "},\n}"
}
