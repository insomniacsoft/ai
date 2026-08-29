package realtime

// Model is one realtime model, with what each of its billable classes costs.
//
// It exists because a Realtime session's rates differ by up to 160x across the
// classes in one conversation (see cost.go), so "what did that cost" cannot be
// answered without a rate per class -- and until now nothing published one.
// Every caller had to copy eight numbers per model off a price page by hand,
// for the one channel where getting it wrong is most expensive.
//
// The rates here are a DEFAULT, not an authority. cost.go's own note stands: a
// vendor's rate card changes without notice, and a catalog generated at build
// time is a snapshot of the day it was cut. A caller that lets an operator
// state a rate must let that rate win over this one -- what this removes is the
// obligation to state one at all, not the ability to.
type Model struct {
	// ID is the catalog key and what a session is opened with.
	ID string
	// Name is the human-readable name, for a chooser.
	Name string
	// Provider is the service this model is served by.
	Provider string
	// APIModel is the id the provider's own API expects, which is not always
	// the catalog key.
	APIModel string

	// Currency is the ISO 4217 code the cost fields are denominated in.
	Currency string

	// The per-class rates, per 1 million tokens, in Currency.
	//
	// One field per TokenClass, named after it, so the mapping in Rate below
	// is a transcription rather than a judgement. A zero means the source
	// published no rate for that class -- NOT that the class is free, which is
	// why Rate reports whether it found one rather than returning a bare
	// number.
	CostPer1MTextIn        float64
	CostPer1MTextInCached  float64
	CostPer1MTextOut       float64
	CostPer1MAudioIn       float64
	CostPer1MAudioInCached float64
	CostPer1MAudioOut      float64
	CostPer1MImageIn       float64
	CostPer1MImageInCached float64
}

// Rate returns what one million tokens of class cost, and whether the catalog
// publishes a rate for it at all.
//
// The two-value return is the point. A missing rate reported as zero would be
// spend recorded as free, and a total that omits the expensive half of a voice
// conversation reads exactly like a correct one -- which is the failure this
// whole file exists to prevent, not to introduce.
//
// Kept beside the class constants it maps, in the package that owns both, so
// adding a TokenClass and forgetting its rate is a change in one file rather
// than a silent zero in somebody else's.
func (m Model) Rate(class TokenClass) (float64, bool) {
	var rate float64
	switch class {
	case ClassTextInput:
		rate = m.CostPer1MTextIn
	case ClassTextInputCached:
		rate = m.CostPer1MTextInCached
	case ClassTextOutput:
		rate = m.CostPer1MTextOut
	case ClassAudioInput:
		rate = m.CostPer1MAudioIn
	case ClassAudioInputCached:
		rate = m.CostPer1MAudioInCached
	case ClassAudioOutput:
		rate = m.CostPer1MAudioOut
	case ClassImageInput:
		rate = m.CostPer1MImageIn
	case ClassImageInputCached:
		rate = m.CostPer1MImageInCached
	default:
		return 0, false
	}
	if rate <= 0 {
		return 0, false
	}
	return rate, true
}
