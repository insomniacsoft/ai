package realtime

// Cost accounting for a Realtime session.
//
// # Why this is not one number
//
// A voice session's tokens are billed at rates that differ by up to 160x —
// measured 2026-08-23, gpt-realtime ran from $0.40 per million for a cached
// token to $64.00 per million for an audio output token. A single "tokens
// used" figure therefore says almost nothing about what a day cost, which is
// one deployment learned when 306,653 tokens could only be bounded somewhere
// between twelve cents and twenty dollars.
//
// So usage is decomposed into the units that are actually priced, and the
// prices themselves are configuration rather than constants. Hardcoding a
// vendor's rate card guarantees it is wrong: it changes without notice, it
// differs per model, and a second provider would need a second code path
// instead of a second row.

// TokenClass is one billable line on an invoice.
type TokenClass string

const (
	ClassTextInput        TokenClass = "text_input"
	ClassTextInputCached  TokenClass = "text_input_cached"
	ClassAudioInput       TokenClass = "audio_input"
	ClassAudioInputCached TokenClass = "audio_input_cached"
	ClassImageInput       TokenClass = "image_input"
	ClassImageInputCached TokenClass = "image_input_cached"
	ClassTextOutput       TokenClass = "text_output"
	ClassAudioOutput      TokenClass = "audio_output"
)

// TokenClasses is every class, in a stable order for reporting and storage.
var TokenClasses = []TokenClass{
	ClassTextInput, ClassTextInputCached,
	ClassAudioInput, ClassAudioInputCached,
	ClassImageInput, ClassImageInputCached,
	ClassTextOutput, ClassAudioOutput,
}

// Billable decomposes usage into the classes an invoice charges for.
//
// The subtraction is the whole point, and it is the part that is easy to get
// backwards: cached tokens are a SUBSET of their modality's count, not an
// addition to it. See InputTokenDetails for the measured payload this is
// derived from. Every count is clamped at zero so a provider that ever reports
// a cached figure larger than its total produces an understated bill rather
// than a negative one.
func (u *Usage) Billable() map[TokenClass]int64 {
	if u == nil {
		return nil
	}
	in, cached, out := u.InputDetails, u.InputDetails.CachedDetails, u.OutputDetails
	m := map[TokenClass]int64{
		ClassTextInput:        nonNegative(in.TextTokens - cached.TextTokens),
		ClassTextInputCached:  nonNegative(cached.TextTokens),
		ClassAudioInput:       nonNegative(in.AudioTokens - cached.AudioTokens),
		ClassAudioInputCached: nonNegative(cached.AudioTokens),
		ClassImageInput:       nonNegative(in.ImageTokens - cached.ImageTokens),
		ClassImageInputCached: nonNegative(cached.ImageTokens),
		ClassTextOutput:       nonNegative(out.TextTokens),
		ClassAudioOutput:      nonNegative(out.AudioTokens),
	}

	// A provider that reports totals but omits the breakdown must not vanish
	// from the accounting. Attributing the remainder to the most expensive
	// plausible class means an unfamiliar payload shape produces an
	// OVERstated bill, which is the safe direction for a spending guard: it
	// refuses too early rather than too late.
	if sum(m) == 0 && u.TotalTokens > 0 {
		m[ClassAudioInput] = u.InputTokens
		m[ClassAudioOutput] = u.OutputTokens
		if u.InputTokens == 0 && u.OutputTokens == 0 {
			m[ClassAudioOutput] = u.TotalTokens
		}
	}
	return m
}

func nonNegative(v int64) int64 {
	if v < 0 {
		return 0
	}
	return v
}

func sum(m map[TokenClass]int64) int64 {
	var t int64
	for _, v := range m {
		t += v
	}
	return t
}

// TextCacheHitRate is the share of a usage record's TEXT input that the
// provider served from its prefix cache, and whether there was any text input
// to compute it over.
//
// Text only, deliberately. Audio input is never a cached prefix — it is what
// the room just said, fresh every turn — so folding it in would report a rate
// that falls whenever somebody speaks for longer, which is the opposite of
// what the number is for. Text is the prefix that caching actually shrinks,
// and this is the number that says whether that shrinking is being paid for.
//
// The second return distinguishes "nothing to measure" from "measured zero".
// A session that has not yet sent any text has no hit rate; reporting 0%
// would look like a total cache miss, which is a different and much worse
// thing to see in a log.
func TextCacheHitRate(billable map[TokenClass]int64) (float64, bool) {
	cached := billable[ClassTextInputCached]
	total := cached + billable[ClassTextInput]
	if total <= 0 {
		return 0, false
	}
	return float64(cached) / float64(total), true
}
