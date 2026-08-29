package realtime

// TokenClass is one billable line on an invoice. Usage is decomposed into
// these classes because a Realtime session's per-token rates differ sharply
// across them, and the rates themselves are configuration, not constants.
type TokenClass string

// Every class a Realtime session can be billed for.
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

// Billable decomposes usage into the classes an invoice charges for. Cached
// tokens are a subset of their modality's count, not an addition to it, and
// every count is clamped at zero. When the provider reports only a total with
// no per-modality breakdown, the total is attributed to the most expensive
// class (audio) so an unfamiliar payload overstates the bill rather than
// hiding it.
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
	if sum(m) == 0 && u.TotalTokens > 0 {
		m[ClassAudioInput] = u.InputTokens
		m[ClassAudioOutput] = u.OutputTokens
		if u.InputTokens == 0 && u.OutputTokens == 0 {
			m[ClassAudioOutput] = u.TotalTokens
		}
	}
	return m
}

// nonNegative clamps v to zero, since a usage delta computed from two
// provider counts can go negative when they are inconsistent.
func nonNegative(v int64) int64 {
	if v < 0 {
		return 0
	}
	return v
}

// sum adds every value in m.
func sum(m map[TokenClass]int64) int64 {
	var t int64
	for _, v := range m {
		t += v
	}
	return t
}

// TextCacheHitRate is the share of a usage record's text input that the
// provider served from its prefix cache. Text only: audio input is never a
// cached prefix, so folding it in would report a rate that falls whenever
// someone speaks longer. The second return is false when there was no text
// input to measure, distinguishing that from a measured 0%.
func TextCacheHitRate(billable map[TokenClass]int64) (float64, bool) {
	cached := billable[ClassTextInputCached]
	total := cached + billable[ClassTextInput]
	if total <= 0 {
		return 0, false
	}
	return float64(cached) / float64(total), true
}
