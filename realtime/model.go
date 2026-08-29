package realtime

// Model is one realtime model, with what each of its billable classes costs.
// The rates here are a default, not an authority: a vendor's rate card
// changes without notice, and a caller that lets an operator state a rate
// must let that rate win over this one.
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

	// The per-class rates, per 1 million tokens, in Currency. A zero means
	// no rate was published for that class, not that the class is free;
	// Rate reports whether it found one rather than returning a bare number.
	CostPer1MTextIn        float64
	CostPer1MTextInCached  float64
	CostPer1MTextOut       float64
	CostPer1MAudioIn       float64
	CostPer1MAudioInCached float64
	CostPer1MAudioOut      float64
	CostPer1MImageIn       float64
	CostPer1MImageInCached float64
}

// Rate returns what one million tokens of class cost, and whether the
// catalog publishes a rate for it at all: a missing rate reported as zero
// would be spend recorded as free.
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
