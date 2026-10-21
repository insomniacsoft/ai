package modelsync

import (
	"sort"
	"strings"
)

// Unit tables convert a published rate to the unit a catalog field holds. A
// unit absent from the table means the rate cannot fill that field.
var (
	// TokenUnits converts per-token rates to a price per million tokens.
	TokenUnits = map[string]float64{
		"per_1m_tokens": 1,
		"per_1k_tokens": 1000,
	}
	// CharUnits converts per-character rates to a price per million
	// characters.
	CharUnits = map[string]float64{
		"per_1m_characters": 1,
		"per_1k_characters": 1000,
	}
	// MinuteUnits converts time-billed rates to a price per minute.
	MinuteUnits = map[string]float64{
		"per_minute": 1,
		"per_second": 60,
		"per_hour":   1.0 / 60,
	}
	// ImageUnits holds the per-image rate.
	ImageUnits = map[string]float64{"per_image": 1}
	// CallUnits keeps the vendor's own per-1k-calls unit; see
	// hostedtool.Tool.CostPer1KCalls.
	CallUnits = map[string]float64{"per_1k_calls": 1}
)

// variantDims mark a rate as belonging to something other than plain
// inference: a fine-tune, a training run, a promotion, a rate that has not
// taken effect yet. None of them describe what a request costs today.
var variantDims = map[string]bool{
	"batch":            true,
	"commitment":       true,
	"data_logging":     true,
	"discount":         true,
	"effective_from":   true,
	"effective_until":  true,
	"fine_tuned":       true,
	"legacy":           true,
	"method":           true,
	"model_grader":     true,
	"plan":             true,
	"promotional":      true,
	"service_tier":     true,
	"stage":            true,
	"surface":          true,
	"training":         true,
	"underlying_model": true,
}

// defaultDims are the dimension values a rate carries when it describes the
// ordinary way to call a model. A rate that only pins these is as unqualified
// as one that pins nothing.
var defaultDims = map[string]string{
	"deployment": "global",
	"modality":   "text",
	"tier":       "standard",
}

// SelectRate picks the rate a catalog field should hold from everything the
// source publishes for a model, and reports whether it found one.
//
// Metrics are tried in the given order and the first one with a usable rate
// wins. Within a metric the least qualified rate wins: a rate that names no
// region, context band or endpoint describes the model itself, while one that
// names several describes a corner of its pricing table. Ties go to the lower
// amount, then to the dimensions in name order, so a run is reproducible.
// Cache lifetimes are not a tie this function breaks: ChatRates picks the
// 5-minute cache write by its cache_ttl, never by its price.
func SelectRate(
	prices []Price,
	currency string,
	units map[string]float64,
	metrics ...string,
) (float64, bool) {
	for _, metric := range metrics {
		var best Price
		var bestAmount float64
		bestScore := -1

		for _, p := range prices {
			scale, ok := units[p.Unit]
			if !ok || p.Metric != metric || p.Currency != currency {
				continue
			}
			if !usableDims(p.Dims) {
				continue
			}

			amount := p.Amount * scale
			score := dimScore(p.Dims)
			if bestScore >= 0 && !better(
				score, amount, p.Dims,
				bestScore, bestAmount, best.Dims,
			) {
				continue
			}
			best, bestAmount, bestScore = p, amount, score
		}

		if bestScore >= 0 {
			return bestAmount, true
		}
	}
	return 0, false
}

// usableDims rejects a rate that is priced for something other than an
// ordinary request: a variant of the model, or a tier that has to be asked for.
func usableDims(dims map[string]string) bool {
	if tier, ok := dims["tier"]; ok && tier != defaultDims["tier"] {
		return false
	}
	for k := range dims {
		if variantDims[k] {
			return false
		}
	}
	return true
}

// dimScore counts how far a rate is qualified beyond the default way to call
// the model.
func dimScore(dims map[string]string) int {
	score := 0
	for k, v := range dims {
		if want, ok := defaultDims[k]; ok && v == want {
			continue
		}
		score++
	}
	return score
}

func better(
	score int, amount float64, dims map[string]string,
	bestScore int, bestAmount float64, bestDims map[string]string,
) bool {
	if score != bestScore {
		return score < bestScore
	}
	if amount != bestAmount {
		return amount < bestAmount
	}
	return dimKey(dims) < dimKey(bestDims)
}

func dimKey(dims map[string]string) string {
	pairs := make([]string, 0, len(dims))
	for k, v := range dims {
		pairs = append(pairs, k+"="+v)
	}
	sort.Strings(pairs)
	return strings.Join(pairs, ",")
}

// WithDim narrows a model's rates to those a dimension either names as want or
// leaves unstated, so an input rate is not filled from an output one.
func WithDim(prices []Price, dim, want string) []Price {
	out := make([]Price, 0, len(prices))
	for _, p := range prices {
		if v, ok := p.Dims[dim]; !ok || v == want {
			out = append(out, p)
		}
	}
	return out
}

// Rate is one selected rate. Published separates a rate the source states
// from one it does not: a missing rate read as a bare zero would be spend
// recorded as free.
type Rate struct {
	// Per is the amount in the unit the selector converts to: per million
	// tokens, per million characters or per thousand calls.
	Per float64
	// Published reports whether the source states the rate at all.
	Published bool
}

func selected(
	prices []Price,
	currency string,
	units map[string]float64,
	metrics ...string,
) Rate {
	v, ok := SelectRate(prices, currency, units, metrics...)
	return Rate{Per: v, Published: ok}
}

// ChatRates are the per-million-token rates of a chat model. CacheWrite is
// its own rate: a cache write is billed on input, at a premium, once, and
// belongs in no cached-output field.
type ChatRates struct {
	Input      Rate
	Output     Rate
	CacheRead  Rate
	CacheWrite Rate
}

// cacheWriteTTL is the cache lifetime whose write rate a chat model is priced
// at. A cache write is published once per lifetime, and the 5-minute one is
// the default a request that does not ask for a longer one pays.
const cacheWriteTTL = "5m"

// ChatRates selects a chat model's rates.
//
// The cache-write rate is the one published for the 5-minute lifetime
// (dims.cache_ttl "5m"), whatever the other lifetimes cost and however many
// dims they carry. When no rate names that lifetime, the ordinary selection
// applies to the rates that name no lifetime at all: a provider that publishes
// only a 1-hour or 30-minute write has no 5-minute rate, and reporting one of
// those as it would misprice every write.
func (e Entry) ChatRates() ChatRates {
	c, p := e.Currency(), e.Prices

	write := selected(withTTL(p, cacheWriteTTL), c, TokenUnits, "cache_write_tokens")
	if !write.Published {
		write = selected(withTTL(p, ""), c, TokenUnits, "cache_write_tokens")
	}

	return ChatRates{
		Input:      selected(p, c, TokenUnits, "input_tokens", "usage"),
		Output:     selected(p, c, TokenUnits, "output_tokens"),
		CacheRead:  selected(p, c, TokenUnits, "cached_input_tokens"),
		CacheWrite: write,
	}
}

// withTTL keeps the rates that name the given cache lifetime.
//
// An empty ttl keeps the rates that name no lifetime.
func withTTL(prices []Price, ttl string) []Price {
	var out []Price
	for _, p := range prices {
		if p.Dims["cache_ttl"] == ttl {
			out = append(out, p)
		}
	}
	return out
}

// RealtimeRates are the per-million-token rates of a realtime model, one per
// billable class. The source prices a realtime model per metric and per
// modality: the same "input_tokens" metric appears once each for text, audio
// and image, at rates an order of magnitude apart, so both dimensions are
// read together. Image has no output rate: nothing generates an image in a
// realtime session, and the source publishes none.
type RealtimeRates struct {
	TextIn, TextInCached, TextOut    Rate
	AudioIn, AudioInCached, AudioOut Rate
	ImageIn, ImageInCached           Rate
}

// RealtimeRates selects a realtime model's rates.
func (e Entry) RealtimeRates() RealtimeRates {
	c := e.Currency()
	in := func(modality string) (in, cached Rate) {
		p := WithDim(e.Prices, "modality", modality)
		return selected(p, c, TokenUnits, "input_tokens"),
			selected(p, c, TokenUnits, "cached_input_tokens")
	}
	out := func(modality string) Rate {
		return selected(
			WithDim(e.Prices, "modality", modality),
			c, TokenUnits, "output_tokens",
		)
	}

	var r RealtimeRates
	r.TextIn, r.TextInCached = in("text")
	r.TextOut = out("text")
	r.AudioIn, r.AudioInCached = in("audio")
	r.AudioOut = out("audio")
	r.ImageIn, r.ImageInCached = in("image")
	return r
}

// SpeechRate selects a speech model's price per million characters.
func (e Entry) SpeechRate() Rate {
	return selected(
		e.Prices, e.Currency(), CharUnits,
		"speech", "audio", "input_characters", "input_tokens", "usage",
	)
}

// ToolRate selects what one thousand invocations of a hosted tool cost. A
// tool may publish several standard rates for the same metric, told apart
// only by their detail dim (web search lists "web search", "image web
// search" and two "web search preview" rates), so the rate is read from the
// plain ones first; see [PlainToolRates].
func (e Entry) ToolRate() Rate {
	return selected(
		PlainToolRates(e), e.Currency(), CallUnits, "tool_call",
	)
}

// PlainToolRates narrows a hosted tool's rates to the ones that price the
// tool itself: a rate with no detail dim, or one whose detail, ignoring a
// trailing parenthetical such as "(all models)", is the tool's own name or
// the generic "tool call". A detail naming a variant, such as "image web
// search" or "web search preview", is not plain. A tool publishing no plain
// rate keeps all of its rates, for [SelectRate]'s ordinary tie-break to choose
// from.
func PlainToolRates(e Entry) []Price {
	plain := map[string]bool{
		"tool call":                             true,
		strings.ToLower(e.Name):                 true,
		strings.ReplaceAll(e.ID, "-", " "):      true,
		strings.ReplaceAll(e.APIID(), "-", " "): true,
	}

	out := make([]Price, 0, len(e.Prices))
	for _, p := range e.Prices {
		detail, ok := p.Dims["detail"]
		if !ok || plain[unqualified(detail)] {
			out = append(out, p)
		}
	}
	if len(out) == 0 {
		return e.Prices
	}
	return out
}

// unqualified lowercases a detail dim and drops a trailing parenthetical, so
// "Web search (all models)" reads as "web search".
func unqualified(detail string) string {
	base, _, _ := strings.Cut(detail, "(")
	return strings.ToLower(strings.TrimSpace(base))
}

// HasList reports whether the source publishes the named list at all. A model
// whose feed entry has no "input_modalities" list is a model the feed says
// nothing about, which is a different fact from one that lists modalities
// without a given one.
func (e Entry) HasList(name string) bool {
	_, ok := e.Lists[name]
	return ok
}

// CanReason reports whether the entry says the model reasons: the feature, a
// mandatory-reasoning attribute or a list of reasoning efforts. published
// reports whether the entry states anything a verdict could rest on. When it
// does not, reasons is false only because nothing said otherwise; the caller
// decides what an unknown capability means, and a fact the provider does not
// publish should not exclude a model.
func (e Entry) CanReason() (reasons, published bool) {
	_, mandatory := e.Attrs["reasoning_mandatory"]
	published = e.HasList("features") ||
		e.HasList("reasoning_efforts") || mandatory
	reasons = e.Feature("reasoning") ||
		e.Attrs["reasoning_mandatory"] == "true" ||
		len(e.Lists["reasoning_efforts"]) > 0
	return reasons, published
}

// SupportsAttachments reports whether the model takes anything but text in.
// published is false when the entry lists no input modalities at all, in
// which case supported is false only for want of information; see
// [Entry.CanReason] for who decides what that means.
func (e Entry) SupportsAttachments() (supported, published bool) {
	return e.TakesModality("image", "audio", "video", "file"),
		e.HasList("input_modalities")
}

// ContextWindow is the model's context window, or zero when not published.
func (e Entry) ContextWindow() int64 {
	return e.Limit("context_window", "top_provider_context_window")
}

// MaxOutputTokens is the most the model can emit, or zero when not published.
func (e Entry) MaxOutputTokens() int64 {
	return e.Limit("max_output_tokens")
}
